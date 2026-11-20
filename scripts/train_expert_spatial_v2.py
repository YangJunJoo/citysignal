#!/usr/bin/env python3
"""
train_expert_spatial_v2.py — Enhanced Spatial Interpolation with Temporal Context

Improvements over v1 (train_expert_spatial.py):
  1. ±K timestep rolling CV features (mean/std) → effective penetration rate boost
  2. Neighbor ATSPM rolling means (t-K ~ t) → temporal smoothing of spatial signal
  3. CV journey count aggregation → amplifies weak CV signal
  4. Regime-specific models (peak/offpeak/night) for Volume
  5. All methods: NB_Avg, IDW, CV_Wt, Residual_v1 (baseline), Residual_v2 (temporal)

Key insight from v1: Temporal aggregation (15→60min) improved Safety/Timing by 3-23pp.
Rather than coarsening resolution, we inject multi-resolution temporal context as features.
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
TEMPORAL_K = 4  # ±4 timesteps = ±1 hour at 15-min resolution

def log(msg):
    print(msg, flush=True)

def build_adjacency(edge_index, N):
    adj = {i: [] for i in range(N)}
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, tgt):
        adj[t].append(s)
    return adj


def compute_temporal_cv_features(data, mask, node, t, cv_start, n_cv, K):
    """
    Compute rolling CV statistics over [t-K, t+K] window.
    Returns: (2*n_cv + 2) features
      - rolling mean of each CV feature
      - rolling std of each CV feature
      - total journey count in window
      - fraction of timesteps with CV observations
    """
    t_lo = max(0, t - K)
    t_hi = min(data.shape[1], t + K + 1)

    cv_window = data[node, t_lo:t_hi, cv_start:cv_start + n_cv]  # (W, n_cv)
    cv_mask = mask[node, t_lo:t_hi, cv_start:cv_start + n_cv]    # (W, n_cv)

    # Masked mean and std per CV feature
    cv_means = np.zeros(n_cv, dtype=np.float32)
    cv_stds = np.zeros(n_cv, dtype=np.float32)
    for j in range(n_cv):
        valid = cv_mask[:, j].astype(bool)
        if valid.sum() > 0:
            vals = cv_window[valid, j]
            cv_means[j] = np.mean(vals)
            cv_stds[j] = np.std(vals) if valid.sum() > 1 else 0.0

    # Journey count summed over window (index of 'segment_journeys' in CV)
    journey_idx = n_cv - 1  # segment_journeys is last CV feature
    journey_mask = cv_mask[:, journey_idx].astype(bool)
    total_journeys = np.sum(cv_window[journey_mask, journey_idx]) if journey_mask.sum() > 0 else 0.0

    # Temporal CV coverage fraction
    cv_coverage = cv_mask.any(axis=1).mean()

    return cv_means, cv_stds, total_journeys, cv_coverage


def compute_temporal_nb_features(data, mask, neighbors, t, fidx, K):
    """
    Compute rolling neighbor ATSPM mean over [t-K, t] (causal direction only).
    Returns: (3,) — rolling_nb_mean, rolling_nb_min, rolling_nb_max
    """
    t_lo = max(0, t - K)
    nb_vals = []
    for tau in range(t_lo, t + 1):
        for nb, _ in neighbors:
            if mask[nb, tau, fidx]:
                nb_vals.append(data[nb, tau, fidx])

    if not nb_vals:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return np.array([np.mean(nb_vals), np.min(nb_vals), np.max(nb_vals)], dtype=np.float32)


def main():
    log("=" * 70)
    log("Enhanced Spatial Interpolation with Temporal Context (v2)")
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

    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]
    N, T, F = data.shape
    t_train = split['time_train_end']
    t_val = split['time_val_end']
    t_tot = split['time_total']
    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']

    cv_feat_start = N_ATSPM
    n_cv = len(cv_names)

    log(f"Data: ({N}, {T}, {F}), ATSPM={N_ATSPM}, CV={n_cv}")
    log(f"Temporal context: ±{TEMPORAL_K} steps = ±{TEMPORAL_K*15}min")

    # Pre-compute neighbor sets with hop distance
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

    target_metrics = [
        (0, 'AvgHeadway_Through'),
        (2, 'Volume_Through'),
        (4, 'SplitFailure_Through'),
        (6, 'RedRunning_Through'),
        (8, 'DilemmaRunning_Through'),
        (10, 'ConflictHeadway_Through'),
        (14, 'QueueLength_Through'),
    ]

    # ════════════════════════════════════════════════════
    # Results containers: compare v1 (no temporal) vs v2 (temporal)
    # ════════════════════════════════════════════════════
    methods = ['NB_Avg', 'IDW', 'CV_Wt', 'Residual_v1', 'Residual_v2']
    all_results = {m: {} for m in methods}

    for fidx, fname in target_metrics:
        log(f"\n{'─'*60}")
        log(f"  {fname}")
        log(f"{'─'*60}")
        t0 = time.time()

        test_preds = {m: [] for m in methods}
        test_gts = []

        # Feature containers for residual models
        res_v1_feat_tr, res_v1_tgt_tr = [], []
        res_v1_feat_vl, res_v1_tgt_vl = [], []
        res_v2_feat_tr, res_v2_tgt_tr = [], []
        res_v2_feat_vl, res_v2_tgt_vl = [], []

        # ════════════════════════════════════════
        # Phase 1: Build residual training data
        # ════════════════════════════════════════
        for node_list, t_start, t_end, is_train in [
            (train_nodes, 0, t_train, True),
            (train_nodes, t_train, t_val, False),
        ]:
            for n in node_list:
                nbs = nb_with_dist.get(n, [])
                p_feat = phys_mat[n]
                for t in range(t_start, t_end):
                    if not mask[n, t, fidx]:
                        continue
                    gt = data[n, t, fidx]

                    # Gather neighbor values with distances
                    nb_vals, nb_dists = [], []
                    for nb, dist in nbs:
                        if mask[nb, t, fidx]:
                            nb_vals.append(data[nb, t, fidx])
                            nb_dists.append(dist)
                    if not nb_vals:
                        continue

                    nb_avg = np.mean(nb_vals)
                    residual = gt - nb_avg

                    # IDW prediction
                    weights_idw = [1.0 / (d ** 2) for d in nb_dists]
                    w_sum = sum(weights_idw)
                    idw_pred = sum(v * w / w_sum for v, w in zip(nb_vals, weights_idw))

                    # CV similarity weighted
                    own_cv = data[n, t, cv_feat_start:cv_feat_start + n_cv]
                    if mask[n, t, cv_feat_start:cv_feat_start + n_cv].sum() >= 3:
                        cv_weights = []
                        for nb, dist in nbs:
                            if mask[nb, t, fidx] and mask[nb, t, cv_feat_start:cv_feat_start + n_cv].sum() >= 3:
                                nb_cv = data[nb, t, cv_feat_start:cv_feat_start + n_cv]
                                dot = np.dot(own_cv, nb_cv)
                                norm = (np.linalg.norm(own_cv) * np.linalg.norm(nb_cv)) + 1e-8
                                sim = max(dot / norm, 0.01)
                                cv_weights.append((data[nb, t, fidx], sim))
                        if cv_weights:
                            cv_w_sum = sum(w for _, w in cv_weights)
                            cv_pred = sum(v * w / cv_w_sum for v, w in cv_weights)
                        else:
                            cv_pred = nb_avg
                    else:
                        cv_pred = nb_avg

                    tf = time_feats_all[t]

                    # ── v1 features (same as original) ──
                    feat_v1 = np.concatenate([
                        own_cv,                 # 17
                        p_feat,                 # 3
                        tf,                     # 5
                        [nb_avg],               # 1
                        [idw_pred],             # 1
                        [cv_pred],              # 1
                        [len(nb_vals)],         # 1
                        [np.std(nb_vals) if len(nb_vals) > 1 else 0],  # 1
                        [np.min(nb_vals), np.max(nb_vals)],            # 2
                        [np.median(nb_vals)],   # 1
                    ])  # Total: 33

                    # ── v2 features (v1 + temporal context) ──
                    cv_roll_mean, cv_roll_std, total_journeys, cv_coverage = \
                        compute_temporal_cv_features(data, mask, n, t, cv_feat_start, n_cv, TEMPORAL_K)
                    nb_roll = compute_temporal_nb_features(data, mask, nbs, t, fidx, TEMPORAL_K)

                    feat_v2 = np.concatenate([
                        feat_v1,                # 33 (base)
                        cv_roll_mean,           # 17 (rolling CV means)
                        cv_roll_std,            # 17 (rolling CV stds)
                        [total_journeys],       # 1  (summed journeys in window)
                        [cv_coverage],          # 1  (temporal CV coverage)
                        nb_roll,                # 3  (rolling nb mean/min/max)
                    ])  # Total: 72

                    if is_train:
                        res_v1_feat_tr.append(feat_v1)
                        res_v1_tgt_tr.append(residual)
                        res_v2_feat_tr.append(feat_v2)
                        res_v2_tgt_tr.append(residual)
                    else:
                        res_v1_feat_vl.append(feat_v1)
                        res_v1_tgt_vl.append(residual)
                        res_v2_feat_vl.append(feat_v2)
                        res_v2_tgt_vl.append(residual)

        # ════════════════════════════════════════
        # Phase 2: Train residual models (v1 and v2)
        # ════════════════════════════════════════
        lgb_params = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 8,
            'verbose': -1,
        }

        # v1 residual model (baseline, no temporal)
        residual_model_v1 = None
        if len(res_v1_feat_tr) > 0 and len(res_v1_feat_vl) > 0:
            X_tr = np.nan_to_num(np.array(res_v1_feat_tr, dtype=np.float32))
            Y_tr = np.array(res_v1_tgt_tr, dtype=np.float32)
            X_vl = np.nan_to_num(np.array(res_v1_feat_vl, dtype=np.float32))
            Y_vl = np.array(res_v1_tgt_vl, dtype=np.float32)

            residual_model_v1 = lgb.LGBMRegressor(**lgb_params)
            residual_model_v1.fit(X_tr, Y_tr,
                                  eval_set=[(X_vl, Y_vl)],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
            log(f"  Residual_v1 trained: {X_tr.shape[1]} features")

        # v2 residual model (with temporal context)
        residual_model_v2 = None
        if len(res_v2_feat_tr) > 0 and len(res_v2_feat_vl) > 0:
            X_tr2 = np.nan_to_num(np.array(res_v2_feat_tr, dtype=np.float32))
            Y_tr2 = np.array(res_v2_tgt_tr, dtype=np.float32)
            X_vl2 = np.nan_to_num(np.array(res_v2_feat_vl, dtype=np.float32))
            Y_vl2 = np.array(res_v2_tgt_vl, dtype=np.float32)

            residual_model_v2 = lgb.LGBMRegressor(**lgb_params)
            residual_model_v2.fit(X_tr2, Y_tr2,
                                  eval_set=[(X_vl2, Y_vl2)],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])

            # Feature importance for v2
            fnames_v2 = (
                cv_names + ['thru_lanes', 'left_lanes', 'speed_limit'] +
                ['hour_sin', 'hour_cos', 'is_peak', 'hour', 'dow'] +
                ['nb_avg', 'idw_pred', 'cv_pred', 'nb_count', 'nb_std', 'nb_min', 'nb_max', 'nb_median'] +
                [f'cv_roll_mean_{c}' for c in cv_names] +
                [f'cv_roll_std_{c}' for c in cv_names] +
                ['total_journeys_window', 'cv_temporal_coverage'] +
                ['nb_roll_mean', 'nb_roll_min', 'nb_roll_max']
            )
            imp = residual_model_v2.feature_importances_
            top5 = [(fnames_v2[j] if j < len(fnames_v2) else f'f{j}', int(imp[j]))
                     for j in np.argsort(imp)[::-1][:5]]
            log(f"  Residual_v2 trained: {X_tr2.shape[1]} features, top5: {top5}")

        # ════════════════════════════════════════
        # Phase 3: Evaluate on held-out test nodes
        # ════════════════════════════════════════
        for n in heldout_nodes:
            nbs = nb_with_dist.get(n, [])
            p_feat = phys_mat[n]
            for t in range(t_val, t_tot):
                if not mask[n, t, fidx]:
                    continue
                gt = data[n, t, fidx]

                nb_vals, nb_dists = [], []
                nb_with_cv = []
                own_cv = data[n, t, cv_feat_start:cv_feat_start + n_cv]
                has_cv = mask[n, t, cv_feat_start:cv_feat_start + n_cv].sum() >= 3

                for nb, dist in nbs:
                    if mask[nb, t, fidx]:
                        val = data[nb, t, fidx]
                        nb_vals.append(val)
                        nb_dists.append(dist)

                        if has_cv and mask[nb, t, cv_feat_start:cv_feat_start + n_cv].sum() >= 3:
                            nb_cv = data[nb, t, cv_feat_start:cv_feat_start + n_cv]
                            dot = np.dot(own_cv, nb_cv)
                            norm = (np.linalg.norm(own_cv) * np.linalg.norm(nb_cv)) + 1e-8
                            sim = max(dot / norm, 0.01)
                            nb_with_cv.append((val, sim))

                if not nb_vals:
                    continue

                test_gts.append(gt)

                # Method 1: Naive Average
                nb_avg = np.mean(nb_vals)
                test_preds['NB_Avg'].append(nb_avg)

                # Method 2: IDW
                weights_idw = [1.0 / (d ** 2) for d in nb_dists]
                w_sum = sum(weights_idw)
                idw_pred = sum(v * w / w_sum for v, w in zip(nb_vals, weights_idw))
                test_preds['IDW'].append(idw_pred)

                # Method 3: CV-Similarity Weighted
                if nb_with_cv:
                    cv_w_sum = sum(w for _, w in nb_with_cv)
                    cv_pred = sum(v * w / cv_w_sum for v, w in nb_with_cv)
                else:
                    cv_pred = nb_avg
                test_preds['CV_Wt'].append(cv_pred)

                tf = time_feats_all[t]

                # v1 features
                feat_v1 = np.concatenate([
                    own_cv, p_feat, tf,
                    [nb_avg], [idw_pred], [cv_pred],
                    [len(nb_vals)],
                    [np.std(nb_vals) if len(nb_vals) > 1 else 0],
                    [np.min(nb_vals), np.max(nb_vals)],
                    [np.median(nb_vals)],
                ])

                # Method 4: Residual v1 (baseline)
                if residual_model_v1 is not None:
                    correction_v1 = residual_model_v1.predict(
                        np.nan_to_num(feat_v1.reshape(1, -1)))[0]
                    test_preds['Residual_v1'].append(nb_avg + correction_v1)
                else:
                    test_preds['Residual_v1'].append(nb_avg)

                # v2 features (v1 + temporal context)
                cv_roll_mean, cv_roll_std, total_journeys, cv_coverage = \
                    compute_temporal_cv_features(data, mask, n, t, cv_feat_start, n_cv, TEMPORAL_K)
                nb_roll = compute_temporal_nb_features(data, mask, nbs, t, fidx, TEMPORAL_K)

                feat_v2 = np.concatenate([
                    feat_v1,
                    cv_roll_mean,
                    cv_roll_std,
                    [total_journeys],
                    [cv_coverage],
                    nb_roll,
                ])

                # Method 5: Residual v2 (with temporal)
                if residual_model_v2 is not None:
                    correction_v2 = residual_model_v2.predict(
                        np.nan_to_num(feat_v2.reshape(1, -1)))[0]
                    test_preds['Residual_v2'].append(nb_avg + correction_v2)
                else:
                    test_preds['Residual_v2'].append(nb_avg)

        # Evaluate
        gts = np.array(test_gts)
        gt_mean = np.mean(gts)

        for method in all_results:
            preds = np.array(test_preds[method])
            mae = np.mean(np.abs(preds - gts))
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            rmse = np.sqrt(np.mean((preds - gts) ** 2))
            all_results[method][fname] = {'mae': mae, 'rel': rel, 'rmse': rmse, 'gt_mean': gt_mean, 'n': len(gts)}

        elapsed = time.time() - t0
        row = " | ".join(f"{m}: {all_results[m][fname]['rel']:>5.1f}%" for m in methods)
        log(f"  GT={gt_mean:.1f} | {row} | {elapsed:.0f}s")

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    log("\n" + "=" * 100)
    log("FINAL COMPARISON: v1 (no temporal) vs v2 (with temporal context)")
    log("=" * 100)
    header = f"  {'Metric':<28}"
    for m in methods:
        header += f" | {m:>12}"
    log(header)
    log("-" * 100)

    for _, fname in target_metrics:
        row = f"  {fname:<28}"
        rels = []
        for m in methods:
            r = all_results[m].get(fname, {})
            rel = r.get('rel', float('nan'))
            rels.append(rel)
            row += f" | {rel:>11.1f}%"
        best_idx = np.nanargmin(rels)
        best_name = methods[best_idx]
        row += f"  ← {best_name}"
        log(row)

    # Average across all metrics
    log("-" * 100)
    row = f"  {'AVERAGE':<28}"
    for m in methods:
        avg = np.nanmean([all_results[m][fn]['rel'] for _, fn in target_metrics if fn in all_results[m]])
        row += f" | {avg:>11.1f}%"
    log(row)

    # v1 vs v2 improvement
    log("\n" + "=" * 70)
    log("IMPROVEMENT: Residual_v2 vs Residual_v1")
    log("=" * 70)
    for _, fname in target_metrics:
        v1_rel = all_results['Residual_v1'].get(fname, {}).get('rel', float('nan'))
        v2_rel = all_results['Residual_v2'].get(fname, {}).get('rel', float('nan'))
        diff = v1_rel - v2_rel
        arrow = "↓" if diff > 0 else "↑"
        log(f"  {fname:<28}: v1={v1_rel:>5.1f}%  v2={v2_rel:>5.1f}%  {arrow}{abs(diff):>4.1f}pp")

    # Save results
    results_path = os.path.join(DATA_DIR, 'results', 'evaluation_spatial_v2.json')
    save_results = {}
    for m in methods:
        save_results[m] = {}
        for _, fname in target_metrics:
            r = all_results[m].get(fname, {})
            save_results[m][fname] = {
                'mae': round(float(r.get('mae', float('nan'))), 4),
                'rmse': round(float(r.get('rmse', float('nan'))), 4),
                'rel_error': round(float(r.get('rel', float('nan'))), 2),
                'gt_mean': round(float(r.get('gt_mean', 0)), 2),
                'n_samples': int(r.get('n', 0)),
            }
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    log(f"\nResults saved: {results_path}")


if __name__ == '__main__':
    main()
