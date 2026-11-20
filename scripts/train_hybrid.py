#!/usr/bin/env python3
"""
train_hybrid.py — Hybrid CV + Neighbor-ATSPM → Target-ATSPM regression.

For each (node, time):
  X = [own_CV(17)] + [mean_neighbor_ATSPM(15)] + [count_neighbor_obs(15)]
  Y = own_ATSPM(15)

This combines the strength of direct regression (own CV)
with spatial information (neighbor ATSPM).
"""

import os, json, time
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
N_ATSPM = 15

def log(msg):
    print(msg, flush=True)


def build_adjacency(edge_index, N):
    """Build adjacency list from edge index."""
    adj = {i: [] for i in range(N)}
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, tgt):
        adj[t].append(s)  # t's neighbors include s
    return adj


def build_features(data, mask, adj, node_list, t_start, t_end, include_2hop=True):
    """
    Build feature matrix with:
    - Own CV features (17)
    - Neighbor ATSPM mean (15) — 1-hop average
    - Neighbor ATSPM obs count (15)
    - (Optional) 2-hop ATSPM mean (15) + count (15)
    Total: 17 + 15 + 15 [+ 15 + 15] = 47 or 77
    """
    N_all = data.shape[0]
    X_list, Y_list, obs_mask_list = [], [], []

    for n in node_list:
        neighbors_1hop = adj.get(n, [])
        neighbors_2hop = set()
        if include_2hop:
            for nb in neighbors_1hop:
                neighbors_2hop.update(adj.get(nb, []))
            neighbors_2hop -= {n}
            neighbors_2hop -= set(neighbors_1hop)
            neighbors_2hop = list(neighbors_2hop)

        for t in range(t_start, t_end):
            cv_mask = mask[n, t, N_ATSPM:]
            if cv_mask.sum() < 3:
                continue

            own_cv = data[n, t, N_ATSPM:].copy()  # (17,)

            # 1-hop neighbor ATSPM aggregation
            nb_atspm_vals = []
            for nb in neighbors_1hop:
                nb_mask = mask[nb, t, :N_ATSPM]
                if nb_mask.sum() > 0:
                    nb_atspm_vals.append(data[nb, t, :N_ATSPM].copy())

            if nb_atspm_vals:
                nb_arr = np.array(nb_atspm_vals)  # (K, 15)
                nb_mean = np.nanmean(nb_arr, axis=0)
                nb_count = np.array([len(nb_atspm_vals)] * N_ATSPM, dtype=np.float32)
            else:
                nb_mean = np.zeros(N_ATSPM, dtype=np.float32)
                nb_count = np.zeros(N_ATSPM, dtype=np.float32)

            features = np.concatenate([own_cv, nb_mean, nb_count])

            # 2-hop
            if include_2hop:
                nb2_vals = []
                for nb in neighbors_2hop:
                    nb_mask = mask[nb, t, :N_ATSPM]
                    if nb_mask.sum() > 0:
                        nb2_vals.append(data[nb, t, :N_ATSPM].copy())
                if nb2_vals:
                    nb2_arr = np.array(nb2_vals)
                    nb2_mean = np.nanmean(nb2_arr, axis=0)
                    nb2_count = np.array([len(nb2_vals)] * N_ATSPM, dtype=np.float32)
                else:
                    nb2_mean = np.zeros(N_ATSPM, dtype=np.float32)
                    nb2_count = np.zeros(N_ATSPM, dtype=np.float32)
                features = np.concatenate([features, nb2_mean, nb2_count])

            X_list.append(features)
            Y_list.append(data[n, t, :N_ATSPM].copy())
            obs_mask_list.append(mask[n, t, :N_ATSPM].copy())

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(obs_mask_list, dtype=bool))


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
    t_train = split['time_train_end']
    t_val = split['time_val_end']
    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']

    adj = build_adjacency(edge_index, N)

    # Count neighbor stats
    nb_counts = [len(adj[n]) for n in range(N)]
    log(f"Adjacency: mean={np.mean(nb_counts):.1f}, max={np.max(nb_counts)}")

    log("\nBuilding train features (CV + neighbor ATSPM)...")
    t0 = time.time()
    X_train, Y_train, _ = build_features(data, mask, adj, train_nodes, 0, t_train)
    log(f"  Train: {X_train.shape} in {time.time()-t0:.1f}s")

    log("Building test features (held-out nodes, test time)...")
    t0 = time.time()
    X_test, Y_test, test_mask = build_features(data, mask, adj, heldout_nodes, t_val, T)
    log(f"  Test: {X_test.shape} in {time.time()-t0:.1f}s")

    # Standardize
    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0)

    # Subsample train to 200K to avoid memory issues
    MAX_TRAIN = 200_000
    if len(X_train_s) > MAX_TRAIN:
        np.random.seed(42)
        idx = np.random.choice(len(X_train_s), MAX_TRAIN, replace=False)
        X_train_s = X_train_s[idx]
        Y_train = Y_train[idx]
        log(f"  Subsampled train: {MAX_TRAIN:,} / {len(X_train):,}")

    feat_dim = X_train_s.shape[1]
    log(f"\nFeature dim: {feat_dim} (17 CV + 15 nb_mean + 15 nb_count + 15 nb2_mean + 15 nb2_count)")

    # Train models
    models = {
        'Hybrid_XGB': MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       subsample=0.8, random_state=42), n_jobs=4),
        'Hybrid_MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=42, learning_rate_init=0.001),
    }

    # Load previous results for comparison
    gnn_path = os.path.join(RESULTS_DIR, 'evaluation_v3.json')
    reg_path = os.path.join(RESULTS_DIR, 'evaluation_regression.json')
    prev_results = {}
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            prev_results.update(json.load(f))
    # Add best GNN and best regression for comparison
    best_gnn = 'spin'
    best_reg = 'XGBoost'

    all_results = {}
    for mname, model in models.items():
        log(f"\n{'='*60}")
        log(f"Training {mname}...")
        t0 = time.time()
        model.fit(X_train_s, Y_train)
        log(f"  Trained in {time.time()-t0:.1f}s")

        Y_pred = model.predict(X_test_s)

        metrics = []
        for i, fname in enumerate(atspm_names):
            obs = test_mask[:, i]
            if obs.sum() == 0:
                metrics.append({'feature': fname, 'MAE': float('nan'), 'RMSE': float('nan'),
                                'RelError': float('nan'), 'N_obs': 0})
                continue
            p, t = Y_pred[obs, i], Y_test[obs, i]
            mae = float(np.mean(np.abs(p - t)))
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            gt_mean = float(np.mean(np.abs(t)))
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            metrics.append({'feature': fname, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
                            'RelError': round(rel, 1), 'GT_mean': round(gt_mean, 2), 'N_obs': int(obs.sum())})
        all_results[mname] = metrics

    # Comparison table
    log(f"\n{'='*100}")
    log("COMPARISON: Best GNN vs CV-only vs Hybrid (Relative Error %)")
    log(f"{'='*100}")

    # Load GNN results
    gnn_path = os.path.join(RESULTS_DIR, 'evaluation_v3.json')
    gnn_results = {}
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            gnn_results = json.load(f)

    best_gnn = 'spin'
    compare_models = [best_gnn] + list(all_results.keys())

    header = f"{'Feature':<25} {'GT':>8}"
    for m in compare_models:
        header += f" {m:>14}"
    log(header)
    log("-" * len(header))

    for i, fname in enumerate(atspm_names):
        # Get GT mean from hybrid results
        gt = None
        for m in all_results:
            if all_results[m][i]['N_obs'] > 0:
                gt = all_results[m][i]['GT_mean']
                break
        if gt is None or gt == 0:
            continue

        row = f"{fname:<25} {gt:>8.1f}"

        for m in compare_models:
            if m in gnn_results and i < len(gnn_results[m]):
                mae = gnn_results[m][i]['MAE']
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
    for m in compare_models:
        if m in gnn_results:
            maes = [r['MAE'] for r in gnn_results[m] if not np.isnan(r['MAE'])]
        else:
            maes = [r['MAE'] for r in all_results[m] if not np.isnan(r['MAE'])]
        if maes:
            log(f"  {m:>15}: {np.mean(maes):.2f}")

    # Save hybrid results
    out_path = os.path.join(RESULTS_DIR, 'evaluation_hybrid.json')
    # Convert all values to float for JSON
    save_results = {}
    for mname, metrics in all_results.items():
        save_results[mname] = []
        for m in metrics:
            save_results[mname].append({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()})
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    log(f"\nSaved: {out_path}")
    log("\nDone!")


if __name__ == '__main__':
    main()
