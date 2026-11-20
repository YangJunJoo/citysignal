#!/usr/bin/env python3
"""
train_expert_hurdle.py — Hurdle & ZIP Models for Zero-Inflated Metrics

Targets:
  - QueueLength: Hurdle model (binary clf + conditional regression)
  - Safety (RedRunning, DilemmaRunning): ZIP (binary clf + conditional Poisson)
  - Volume: Regime-specific (peak/offpeak/night separate models)

All use spatial interpolation (NB_Avg) as base, with ML residual correction.
"""

import os, json, time, math
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
N_ATSPM = 15
TEMPORAL_K = 4

def log(msg):
    print(msg, flush=True)

def build_adjacency(edge_index, N):
    adj = {i: [] for i in range(N)}
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, tgt):
        adj[t].append(s)
    return adj

def compute_temporal_cv_features(data, mask, node, t, cv_start, n_cv, K):
    t_lo, t_hi = max(0, t - K), min(data.shape[1], t + K + 1)
    cv_window = data[node, t_lo:t_hi, cv_start:cv_start + n_cv]
    cv_mask = mask[node, t_lo:t_hi, cv_start:cv_start + n_cv]
    cv_means = np.zeros(n_cv, dtype=np.float32)
    cv_stds = np.zeros(n_cv, dtype=np.float32)
    for j in range(n_cv):
        valid = cv_mask[:, j].astype(bool)
        if valid.sum() > 0:
            cv_means[j] = np.mean(cv_window[valid, j])
            cv_stds[j] = np.std(cv_window[valid, j]) if valid.sum() > 1 else 0.0
    journey_idx = n_cv - 1
    jm = cv_mask[:, journey_idx].astype(bool)
    total_j = np.sum(cv_window[jm, journey_idx]) if jm.sum() > 0 else 0.0
    cv_cov = cv_mask.any(axis=1).mean()
    return cv_means, cv_stds, total_j, cv_cov

def compute_temporal_nb_features(data, mask, neighbors, t, fidx, K):
    t_lo = max(0, t - K)
    nb_vals = []
    for tau in range(t_lo, t + 1):
        for nb, _ in neighbors:
            if mask[nb, tau, fidx]:
                nb_vals.append(data[nb, tau, fidx])
    if not nb_vals:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([np.mean(nb_vals), np.min(nb_vals), np.max(nb_vals)], dtype=np.float32)

def build_full_features(data, mask, n, t, nbs, phys_mat, time_feats_all, cv_start, n_cv, fidx):
    """Build full v2-style feature vector for a (node, time) pair."""
    p_feat = phys_mat[n]
    own_cv = data[n, t, cv_start:cv_start + n_cv]
    tf = time_feats_all[t]

    nb_vals, nb_dists = [], []
    for nb, dist in nbs:
        if mask[nb, t, fidx]:
            nb_vals.append(data[nb, t, fidx])
            nb_dists.append(dist)

    if not nb_vals:
        return None, None, 0.0

    nb_avg = np.mean(nb_vals)
    weights_idw = [1.0 / (d ** 2) for d in nb_dists]
    w_sum = sum(weights_idw)
    idw_pred = sum(v * w / w_sum for v, w in zip(nb_vals, weights_idw))

    has_cv = mask[n, t, cv_start:cv_start + n_cv].sum() >= 3
    cv_pred = nb_avg
    if has_cv:
        cv_weights = []
        for nb, dist in nbs:
            if mask[nb, t, fidx] and mask[nb, t, cv_start:cv_start + n_cv].sum() >= 3:
                nb_cv = data[nb, t, cv_start:cv_start + n_cv]
                dot = np.dot(own_cv, nb_cv)
                norm = (np.linalg.norm(own_cv) * np.linalg.norm(nb_cv)) + 1e-8
                sim = max(dot / norm, 0.01)
                cv_weights.append((data[nb, t, fidx], sim))
        if cv_weights:
            cv_w_sum = sum(w for _, w in cv_weights)
            cv_pred = sum(v * w / cv_w_sum for v, w in cv_weights)

    cv_roll_mean, cv_roll_std, total_j, cv_cov = \
        compute_temporal_cv_features(data, mask, n, t, cv_start, n_cv, TEMPORAL_K)
    nb_roll = compute_temporal_nb_features(data, mask, nbs, t, fidx, TEMPORAL_K)

    feat = np.concatenate([
        own_cv, p_feat, tf,
        [nb_avg], [idw_pred], [cv_pred],
        [len(nb_vals)],
        [np.std(nb_vals) if len(nb_vals) > 1 else 0],
        [np.min(nb_vals), np.max(nb_vals)],
        [np.median(nb_vals)],
        cv_roll_mean, cv_roll_std,
        [total_j], [cv_cov],
        nb_roll,
    ])
    return feat, nb_avg, nb_avg


def main():
    log("=" * 70)
    log("Hurdle & ZIP Models for Zero-Inflated Metrics")
    log("=" * 70)

    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True).numpy()
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True).numpy()
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)

    phys_df = pd.read_csv(os.path.join(DATA_DIR, 'physical_features.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    phys_df['Seg_Unique'] = phys_df['Seg_Unique'].astype(str)
    node_map['Seg_Unique'] = node_map['Seg_Unique'].astype(str)
    merged = pd.merge(node_map, phys_df, on='Seg_Unique', how='left')
    merged['thru_lanes'] = merged['thru_lanes'].fillna(2)
    merged['left_lanes'] = merged['left_lanes'].fillna(1)
    merged['speed_limit_mph'] = merged['speed_limit_mph'].fillna(45)
    phys_mat = merged[['thru_lanes', 'left_lanes', 'speed_limit_mph']].values.astype(np.float32)

    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'], node_map['node_id']))
    v = edges_df[edges_df['source'].astype(str).isin(seg2id) & edges_df['target'].astype(str).isin(seg2id)]
    src_e = np.array([seg2id[str(s)] for s in v['source']])
    tgt_e = np.array([seg2id[str(s)] for s in v['target']])
    adj = build_adjacency([torch.tensor(src_e), torch.tensor(tgt_e)], len(node_map))

    time_df = pd.read_csv(os.path.join(DATA_DIR, 'time_index_v3.csv'), parse_dates=['time'])
    def get_time_feats(t_idx):
        t = time_df.iloc[t_idx]['time']
        h = t.hour + t.minute / 60.0
        d = t.dayofweek
        return [math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
                1.0 if (7<=h<=9 or 16<=h<=18) else 0.0, h, d]
    time_feats_all = np.array([get_time_feats(i) for i in range(len(time_df))], dtype=np.float32)

    cv_names = features[N_ATSPM:]
    N, T, F = data.shape
    t_train, t_val, t_tot = split['time_train_end'], split['time_val_end'], split['time_total']
    train_nodes, heldout_nodes = split['train_nodes'], split['heldout_nodes']
    cv_start, n_cv = N_ATSPM, len(cv_names)

    # Pre-compute neighbors
    nb_with_dist = {}
    for n in set(train_nodes) | set(heldout_nodes):
        neighbors = {}
        for nb in adj.get(n, []):
            neighbors[nb] = 1
        for nb1 in adj.get(n, []):
            for nb2 in adj.get(nb1, []):
                if nb2 != n and nb2 not in neighbors:
                    neighbors[nb2] = 2
        for nb1 in adj.get(n, []):
            for nb2 in adj.get(nb1, []):
                for nb3 in adj.get(nb2, []):
                    if nb3 != n and nb3 not in neighbors:
                        neighbors[nb3] = 3
        nb_with_dist[n] = list(neighbors.items())

    # ════════════════════════════════════════
    # Define target groups
    # ════════════════════════════════════════
    hurdle_targets = [
        (14, 'QueueLength_Through', 'hurdle'),
        (6,  'RedRunning_Through', 'zip'),
        (8,  'DilemmaRunning_Through', 'zip'),
        (4,  'SplitFailure_Through', 'zip'),
    ]

    # Also run standard residual for comparison
    all_results = {}

    for fidx, fname, model_type in hurdle_targets:
        log(f"\n{'─'*60}")
        log(f"  {fname} ({model_type.upper()} model)")
        log(f"{'─'*60}")
        t0_metric = time.time()

        # ── Collect data ──
        datasets = {'train': (train_nodes, 0, t_train),
                     'val': (train_nodes, t_train, t_val),
                     'test': (heldout_nodes, t_val, t_tot)}
        collected = {}
        for split_name, (node_list, ts, te) in datasets.items():
            feats, gts, nb_avgs = [], [], []
            for n in node_list:
                nbs = nb_with_dist.get(n, [])
                for t in range(ts, te):
                    if not mask[n, t, fidx]:
                        continue
                    feat, nb_avg, _ = build_full_features(
                        data, mask, n, t, nbs, phys_mat, time_feats_all, cv_start, n_cv, fidx)
                    if feat is None:
                        continue
                    feats.append(feat)
                    gts.append(data[n, t, fidx])
                    nb_avgs.append(nb_avg)
            collected[split_name] = {
                'X': np.nan_to_num(np.array(feats, dtype=np.float32)),
                'Y': np.array(gts, dtype=np.float32),
                'nb_avg': np.array(nb_avgs, dtype=np.float32),
            }
            log(f"  {split_name}: {len(feats)} samples")

        X_tr, Y_tr = collected['train']['X'], collected['train']['Y']
        X_vl, Y_vl = collected['val']['X'], collected['val']['Y']
        X_te, Y_te = collected['test']['X'], collected['test']['Y']
        nb_avg_te = collected['test']['nb_avg']

        # ── Standard Residual (baseline) ──
        residual_tr = Y_tr - collected['train']['nb_avg']
        residual_vl = Y_vl - collected['val']['nb_avg']

        residual_model = lgb.LGBMRegressor(
            objective='regression_l1', metric='mae', n_estimators=1000,
            learning_rate=0.05, num_leaves=63, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
        residual_model.fit(X_tr, residual_tr,
                           eval_set=[(X_vl, residual_vl)],
                           callbacks=[lgb.early_stopping(50, verbose=False)])
        pred_standard = nb_avg_te + residual_model.predict(X_te)

        mae_std = np.mean(np.abs(pred_standard - Y_te))
        rel_std = mae_std / np.mean(Y_te) * 100 if np.mean(Y_te) > 0 else float('nan')
        log(f"  Standard Residual: MAE={mae_std:.2f}, RelError={rel_std:.1f}%")

        # ── Hurdle / ZIP Model ──
        threshold = 0.5 if model_type == 'zip' else 1.0  # queue needs higher threshold

        # Part 1: Binary classification — is the metric > 0?
        y_bin_tr = (Y_tr > threshold).astype(int)
        y_bin_vl = (Y_vl > threshold).astype(int)

        pos_rate_tr = y_bin_tr.mean()
        log(f"  Positive rate (train): {pos_rate_tr:.3f}")

        if pos_rate_tr < 0.005 or pos_rate_tr > 0.995:
            log(f"  SKIPPED hurdle (extreme class imbalance)")
            pred_hurdle = pred_standard
        else:
            clf = lgb.LGBMClassifier(
                objective='binary', metric='binary_logloss', n_estimators=500,
                learning_rate=0.05, num_leaves=31, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1,
                scale_pos_weight=(1 - pos_rate_tr) / pos_rate_tr)
            clf.fit(X_tr, y_bin_tr,
                    eval_set=[(X_vl, y_bin_vl)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

            # Part 2: Conditional regression on positive samples only
            pos_mask_tr = Y_tr > threshold
            pos_mask_vl = Y_vl > threshold

            if model_type == 'zip':
                # Zero-Inflated Poisson: use poisson for count
                reg = lgb.LGBMRegressor(
                    objective='poisson', metric='poisson', n_estimators=500,
                    learning_rate=0.05, num_leaves=31, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
                reg.fit(X_tr[pos_mask_tr], Y_tr[pos_mask_tr],
                        eval_set=[(X_vl[pos_mask_vl], Y_vl[pos_mask_vl])] if pos_mask_vl.sum() > 0 else None,
                        callbacks=[lgb.early_stopping(50, verbose=False)])
                cond_pred = reg.predict(X_te)
            else:
                # Hurdle: log1p regression for magnitude
                reg = lgb.LGBMRegressor(
                    objective='regression', metric='mse', n_estimators=500,
                    learning_rate=0.05, num_leaves=31, min_child_samples=30,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
                reg.fit(X_tr[pos_mask_tr], np.log1p(Y_tr[pos_mask_tr]),
                        eval_set=[(X_vl[pos_mask_vl], np.log1p(Y_vl[pos_mask_vl]))] if pos_mask_vl.sum() > 0 else None,
                        callbacks=[lgb.early_stopping(50, verbose=False)])
                cond_pred = np.expm1(reg.predict(X_te))

            # Combined prediction: P(event) × E[value | event]
            p_event = clf.predict_proba(X_te)[:, 1]
            pred_hurdle = p_event * cond_pred

        mae_hurdle = np.mean(np.abs(pred_hurdle - Y_te))
        rel_hurdle = mae_hurdle / np.mean(Y_te) * 100 if np.mean(Y_te) > 0 else float('nan')
        rmse_hurdle = np.sqrt(np.mean((pred_hurdle - Y_te) ** 2))

        improvement = rel_std - rel_hurdle
        log(f"  {model_type.upper()} Model: MAE={mae_hurdle:.2f}, RelError={rel_hurdle:.1f}%")
        log(f"  Improvement: {improvement:+.1f}pp")

        all_results[fname] = {
            'standard': {'mae': float(mae_std), 'rel': float(rel_std)},
            model_type: {'mae': float(mae_hurdle), 'rel': float(rel_hurdle), 'rmse': float(rmse_hurdle)},
            'gt_mean': float(np.mean(Y_te)),
            'n_samples': len(Y_te),
            'positive_rate': float(pos_rate_tr),
            'improvement_pp': float(improvement),
        }
        log(f"  Elapsed: {time.time() - t0_metric:.0f}s")

    # ══════════════════════════════════════════
    # Volume Regime-Specific Models
    # ══════════════════════════════════════════
    fidx_vol, fname_vol = 2, 'Volume_Through'
    log(f"\n{'─'*60}")
    log(f"  {fname_vol} (REGIME-SPECIFIC)")
    log(f"{'─'*60}")
    t0_vol = time.time()

    # Collect Volume data
    datasets_vol = {'train': (train_nodes, 0, t_train),
                    'val': (train_nodes, t_train, t_val),
                    'test': (heldout_nodes, t_val, t_tot)}
    col_vol = {}
    for sname, (nl, ts, te) in datasets_vol.items():
        feats, gts, nb_avgs, hours = [], [], [], []
        for n in nl:
            nbs = nb_with_dist.get(n, [])
            for t in range(ts, te):
                if not mask[n, t, fidx_vol]:
                    continue
                feat, nb_avg, _ = build_full_features(
                    data, mask, n, t, nbs, phys_mat, time_feats_all, cv_start, n_cv, fidx_vol)
                if feat is None:
                    continue
                feats.append(feat)
                gts.append(data[n, t, fidx_vol])
                nb_avgs.append(nb_avg)
                hours.append(time_feats_all[t][3])  # hour feature
        col_vol[sname] = {
            'X': np.nan_to_num(np.array(feats, dtype=np.float32)),
            'Y': np.array(gts, dtype=np.float32),
            'nb_avg': np.array(nb_avgs, dtype=np.float32),
            'hour': np.array(hours, dtype=np.float32),
        }
        log(f"  {sname}: {len(feats)} samples")

    # Standard single-model baseline
    res_tr = col_vol['train']['Y'] - col_vol['train']['nb_avg']
    res_vl = col_vol['val']['Y'] - col_vol['val']['nb_avg']

    single_model = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae', n_estimators=1000,
        learning_rate=0.05, num_leaves=63, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
    single_model.fit(col_vol['train']['X'], res_tr,
                     eval_set=[(col_vol['val']['X'], res_vl)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
    pred_single = col_vol['test']['nb_avg'] + single_model.predict(col_vol['test']['X'])

    # Regime-specific models
    regimes = {
        'night': lambda h: (h < 6) | (h >= 22),
        'peak': lambda h: ((h >= 7) & (h <= 9)) | ((h >= 16) & (h <= 18)),
        'offpeak': lambda h: ((h >= 6) & (h < 7)) | ((h > 9) & (h < 16)) | ((h > 18) & (h < 22)),
    }

    pred_regime = np.zeros_like(col_vol['test']['Y'])
    assigned = np.zeros(len(pred_regime), dtype=bool)

    for rname, rfunc in regimes.items():
        tr_mask = rfunc(col_vol['train']['hour'])
        vl_mask = rfunc(col_vol['val']['hour'])
        te_mask = rfunc(col_vol['test']['hour'])

        if tr_mask.sum() < 100 or te_mask.sum() == 0:
            pred_regime[te_mask] = pred_single[te_mask]
            assigned[te_mask] = True
            log(f"  {rname}: too few samples, using single model")
            continue

        regime_res_tr = col_vol['train']['Y'][tr_mask] - col_vol['train']['nb_avg'][tr_mask]
        regime_res_vl = col_vol['val']['Y'][vl_mask] - col_vol['val']['nb_avg'][vl_mask]

        rm = lgb.LGBMRegressor(
            objective='regression_l1', metric='mae', n_estimators=1000,
            learning_rate=0.05, num_leaves=63, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=8, verbose=-1)
        rm.fit(col_vol['train']['X'][tr_mask], regime_res_tr,
               eval_set=[(col_vol['val']['X'][vl_mask], regime_res_vl)] if vl_mask.sum() > 0 else None,
               callbacks=[lgb.early_stopping(50, verbose=False)])

        pred_regime[te_mask] = col_vol['test']['nb_avg'][te_mask] + rm.predict(col_vol['test']['X'][te_mask])
        assigned[te_mask] = True

        # Per-regime evaluation
        mae_r = np.mean(np.abs(pred_regime[te_mask] - col_vol['test']['Y'][te_mask]))
        gt_r = np.mean(col_vol['test']['Y'][te_mask])
        log(f"  {rname}: n={te_mask.sum()}, GT_mean={gt_r:.1f}, MAE={mae_r:.1f}, RelErr={mae_r/gt_r*100:.1f}%")

    # Fallback for any unassigned
    if not assigned.all():
        pred_regime[~assigned] = pred_single[~assigned]

    # Compare
    Y_vol_te = col_vol['test']['Y']
    mae_single = np.mean(np.abs(pred_single - Y_vol_te))
    mae_regime = np.mean(np.abs(pred_regime - Y_vol_te))
    gt_vol = np.mean(Y_vol_te)
    log(f"\n  Volume Results:")
    log(f"    Single model: MAE={mae_single:.1f}, RelError={mae_single/gt_vol*100:.1f}%")
    log(f"    Regime model: MAE={mae_regime:.1f}, RelError={mae_regime/gt_vol*100:.1f}%")
    log(f"    Improvement:  {(mae_single/gt_vol - mae_regime/gt_vol)*100:+.1f}pp")

    all_results[fname_vol] = {
        'standard': {'mae': float(mae_single), 'rel': float(mae_single / gt_vol * 100)},
        'regime': {'mae': float(mae_regime), 'rel': float(mae_regime / gt_vol * 100)},
        'gt_mean': float(gt_vol),
        'n_samples': len(Y_vol_te),
    }

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"  {'Metric':<28} {'Standard':>12} {'New Model':>12} {'Δpp':>8}")
    log("-" * 70)
    for fname, res in all_results.items():
        std_rel = res['standard']['rel']
        new_key = [k for k in res if k not in ('standard', 'gt_mean', 'n_samples', 'positive_rate', 'improvement_pp')][0]
        new_rel = res[new_key]['rel']
        diff = std_rel - new_rel
        log(f"  {fname:<28} {std_rel:>11.1f}% {new_rel:>11.1f}% {diff:>+7.1f}")

    # Save
    out = os.path.join(DATA_DIR, 'results', 'evaluation_hurdle_zip.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    log(f"\nResults saved: {out}")
    log(f"Total elapsed: {time.time() - t0_vol:.0f}s")


if __name__ == '__main__':
    main()
