#!/usr/bin/env python3
"""
train_hybrid_v2.py — Improved Hybrid CV + Neighbor ATSPM + Temporal → ATSPM

Improvements over v1:
  1. Temporal features (hour, dow, sin/cos, is_peak)
  2. LightGBM per-target models (not MultiOutput) — 10x faster
  3. Full 1.2M training data (no subsampling needed)
  4. Target normalization + proper evaluation
"""

import os, json, time, math
import numpy as np
import torch
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
N_ATSPM = 15

def log(msg):
    print(msg, flush=True)


def build_adjacency(edge_index, N):
    adj = {i: [] for i in range(N)}
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, tgt):
        adj[t].append(s)
    return adj


def build_features(data, mask, adj, node_list, t_start, t_end, time_feats):
    """
    Features per (node, time):
    - Own CV (17)
    - 1-hop neighbor ATSPM mean (15) + count (1)
    - 2-hop neighbor ATSPM mean (15) + count (1)
    - Temporal: hour_sin, hour_cos, dow_sin, dow_cos, is_peak, hour, dow (7)
    Total: 17 + 16 + 16 + 7 = 56
    """
    X_list, Y_list, obs_mask_list = [], [], []

    for n in node_list:
        nb1 = adj.get(n, [])
        nb2 = set()
        for nb in nb1:
            nb2.update(adj.get(nb, []))
        nb2 -= {n}
        nb2 -= set(nb1)
        nb2 = list(nb2)

        for t in range(t_start, t_end):
            cv_mask = mask[n, t, N_ATSPM:]
            if cv_mask.sum() < 3:
                continue

            own_cv = data[n, t, N_ATSPM:].copy()

            # 1-hop
            nb_vals = []
            for nb in nb1:
                if mask[nb, t, :N_ATSPM].sum() > 0:
                    nb_vals.append(data[nb, t, :N_ATSPM].copy())
            if nb_vals:
                nb1_mean = np.nanmean(nb_vals, axis=0)
                nb1_count = np.float32(len(nb_vals))
            else:
                nb1_mean = np.zeros(N_ATSPM, dtype=np.float32)
                nb1_count = np.float32(0)

            # 2-hop
            nb2_vals = []
            for nb in nb2:
                if mask[nb, t, :N_ATSPM].sum() > 0:
                    nb2_vals.append(data[nb, t, :N_ATSPM].copy())
            if nb2_vals:
                nb2_mean = np.nanmean(nb2_vals, axis=0)
                nb2_count = np.float32(len(nb2_vals))
            else:
                nb2_mean = np.zeros(N_ATSPM, dtype=np.float32)
                nb2_count = np.float32(0)

            # Temporal
            tf = time_feats[t]

            features = np.concatenate([
                own_cv,                    # 17
                nb1_mean, [nb1_count],     # 16
                nb2_mean, [nb2_count],     # 16
                tf,                        # 7
            ])

            X_list.append(features)
            Y_list.append(data[n, t, :N_ATSPM].copy())
            obs_mask_list.append(mask[n, t, :N_ATSPM].copy())

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(obs_mask_list, dtype=bool))


def build_time_features(time_index_path):
    """Build temporal feature array: (T, 7) — hour_sin, hour_cos, dow_sin, dow_cos, is_peak, hour, dow."""
    df = pd.read_csv(time_index_path, parse_dates=['time'])
    T = len(df)
    feats = np.zeros((T, 7), dtype=np.float32)
    for i, row in df.iterrows():
        t = row['time']
        h = t.hour + t.minute / 60.0
        d = t.dayofweek
        feats[i, 0] = math.sin(2 * math.pi * h / 24)
        feats[i, 1] = math.cos(2 * math.pi * h / 24)
        feats[i, 2] = math.sin(2 * math.pi * d / 7)
        feats[i, 3] = math.cos(2 * math.pi * d / 7)
        feats[i, 4] = 1.0 if (7 <= h <= 9 or 16 <= h <= 18) else 0.0
        feats[i, 5] = h
        feats[i, 6] = d
    return feats


def main():
    log("Loading data...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True).numpy()
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True).numpy()
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)

    # Edge index
    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    v = edges_df[edges_df['source'].astype(str).isin(seg2id) &
                 edges_df['target'].astype(str).isin(seg2id)]
    src = np.array([seg2id[str(s)] for s in v['source']])
    tgt = np.array([seg2id[str(s)] for s in v['target']])
    edge_index = torch.tensor(np.stack([src, tgt]), dtype=torch.long)

    N, T, F = data.shape
    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]
    t_train = split['time_train_end']
    t_val = split['time_val_end']
    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']

    adj = build_adjacency(edge_index, N)

    # Build temporal features
    log("Building temporal features...")
    time_feats = build_time_features(os.path.join(DATA_DIR, 'time_index_v3.csv'))
    log(f"  time_feats: {time_feats.shape}")

    log("\nBuilding train features...")
    t0 = time.time()
    X_train, Y_train, _ = build_features(data, mask, adj, train_nodes, 0, t_train, time_feats)
    log(f"  Train: {X_train.shape} in {time.time()-t0:.1f}s")

    log("Building val features...")
    t0 = time.time()
    X_val, Y_val, val_mask = build_features(data, mask, adj, train_nodes, t_train, t_val, time_feats)
    log(f"  Val: {X_val.shape} in {time.time()-t0:.1f}s")

    log("Building test features...")
    t0 = time.time()
    X_test, Y_test, test_mask = build_features(data, mask, adj, heldout_nodes, t_val, T, time_feats)
    log(f"  Test: {X_test.shape} in {time.time()-t0:.1f}s")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0)
    Y_val = np.nan_to_num(Y_val, nan=0.0)

    feat_dim = X_train.shape[1]
    feat_names = (
        cv_names +
        [f'nb1_{a}' for a in atspm_names] + ['nb1_count'] +
        [f'nb2_{a}' for a in atspm_names] + ['nb2_count'] +
        ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_peak', 'hour', 'dow']
    )
    log(f"\nFeature dim: {feat_dim} ({len(feat_names)} named)")

    # ═══════════════════════════════════════════
    # LightGBM per-target models
    # ═══════════════════════════════════════════
    log(f"\n{'='*60}")
    log("Training LightGBM per-target models...")

    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_leaves': 63,
        'min_child_samples': 50,
        'n_jobs': 8,
        'verbose': -1,
        'random_state': 42,
    }

    lgb_preds = np.zeros_like(Y_test)
    lgb_importances = {}
    total_t0 = time.time()

    for i, fname in enumerate(atspm_names):
        t0 = time.time()
        model = lgb.LGBMRegressor(**lgb_params)

        # Use val as early stopping
        model.fit(
            X_train, Y_train[:, i],
            eval_set=[(X_val, Y_val[:, i])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        lgb_preds[:, i] = model.predict(X_test)
        elapsed = time.time() - t0

        # Feature importance
        imp = model.feature_importances_
        top5_idx = np.argsort(imp)[::-1][:5]
        top5 = [(feat_names[j] if j < len(feat_names) else f'f{j}', int(imp[j])) for j in top5_idx]
        lgb_importances[fname] = top5

        if (i + 1) % 5 == 0 or i == 0:
            log(f"  [{i+1}/{N_ATSPM}] {fname:<25} {elapsed:.1f}s  top5: {top5}")

    lgb_total = time.time() - total_t0
    log(f"  LightGBM total: {lgb_total:.1f}s")

    # ═══════════════════════════════════════════
    # MLP baseline (with temporal features)
    # ═══════════════════════════════════════════
    log(f"\n{'='*60}")
    log("Training Hybrid_MLP_v2 (with temporal features)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True,
        validation_fraction=0.1, random_state=42, learning_rate_init=0.001
    )
    t0 = time.time()
    mlp.fit(X_train_s, Y_train)
    mlp_elapsed = time.time() - t0
    log(f"  MLP trained in {mlp_elapsed:.1f}s")
    mlp_preds = mlp.predict(X_test_s)

    # ═══════════════════════════════════════════
    # Evaluation
    # ═══════════════════════════════════════════
    # Load previous results for comparison
    gnn_path = os.path.join(RESULTS_DIR, 'evaluation_v3.json')
    hybrid_v1_path = os.path.join(RESULTS_DIR, 'evaluation_hybrid.json')
    prev = {}
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            prev.update(json.load(f))
    if os.path.exists(hybrid_v1_path):
        with open(hybrid_v1_path) as f:
            prev.update(json.load(f))

    # Compute metrics for new models
    new_models = {
        'LightGBM': lgb_preds,
        'MLP_v2': mlp_preds,
    }

    all_results = {}
    for mname, preds in new_models.items():
        metrics = []
        for i, fname in enumerate(atspm_names):
            obs = test_mask[:, i]
            if obs.sum() == 0:
                metrics.append({'feature': fname, 'MAE': float('nan'), 'RMSE': float('nan'),
                                'RelError': float('nan'), 'N_obs': 0, 'GT_mean': 0.0})
                continue
            p, t = preds[obs, i], Y_test[obs, i]
            mae = float(np.mean(np.abs(p - t)))
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            gt_mean = float(np.mean(np.abs(t)))
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            metrics.append({'feature': fname, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
                            'RelError': round(rel, 1), 'GT_mean': round(gt_mean, 2),
                            'N_obs': int(obs.sum())})
        all_results[mname] = metrics

    # Comparison table
    compare = ['spin', 'Hybrid_XGB']  # previous best
    all_models = compare + list(all_results.keys())

    log(f"\n{'='*100}")
    log("FULL COMPARISON (Relative Error %) — Best GNN vs Hybrid v1 vs v2")
    log(f"{'='*100}")

    header = f"{'Feature':<25} {'GT':>8}"
    for m in all_models:
        header += f" {m:>14}"
    log(header)
    log("-" * len(header))

    for i, fname in enumerate(atspm_names):
        gt = None
        for m in all_results:
            if all_results[m][i]['N_obs'] > 0:
                gt = all_results[m][i]['GT_mean']
                break
        if gt is None or gt == 0:
            continue

        row = f"{fname:<25} {gt:>8.1f}"
        for m in all_models:
            if m in prev and i < len(prev[m]):
                mae = prev[m][i]['MAE']
            elif m in all_results:
                mae = all_results[m][i]['MAE']
            else:
                row += f" {'--':>14}"
                continue

            if np.isnan(mae):
                row += f" {'N/A':>14}"
            else:
                rel = mae / gt * 100
                row += f" {mae:>7.1f}({rel:>4.0f}%)"
        log(row)

    log(f"\n--- OVERALL AVG MAE ---")
    for m in all_models:
        if m in prev:
            maes = [r['MAE'] for r in prev[m] if not np.isnan(r['MAE'])]
        else:
            maes = [r['MAE'] for r in all_results[m] if not np.isnan(r['MAE'])]
        if maes:
            log(f"  {m:>15}: {np.mean(maes):.2f}")

    # Feature importance summary
    log(f"\n{'='*60}")
    log("TOP-5 FEATURE IMPORTANCE (LightGBM per target)")
    log(f"{'='*60}")
    for fname, top5 in lgb_importances.items():
        log(f"  {fname:<25}: {', '.join(f'{n}({v})' for n, v in top5)}")

    # Save
    out = os.path.join(RESULTS_DIR, 'evaluation_hybrid_v2.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out}")
    log("Done!")


if __name__ == '__main__':
    main()
