#!/usr/bin/env python3
"""
train_expert_ratio.py — Ratio-Based Expert Model
Instead of predicting raw ATSPM values, predict the RATIO to neighbor averages.
This removes the absolute scale problem and focuses on relative spatial differences.

Also computes a "naive neighbor average" baseline to check if the current
Expert model actually adds value over simple spatial interpolation.
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

def log(msg):
    print(msg, flush=True)

def build_adjacency(edge_index, N):
    adj = {i: [] for i in range(N)}
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for s, t in zip(src, tgt):
        adj[t].append(s)
    return adj

def main():
    log("="*70)
    log("Ratio-Based Expert + Neighbor Baseline Analysis")
    log("="*70)

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
        return [
            math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
            1.0 if (7 <= h <= 9 or 16 <= h <= 18) else 0.0,
            h, d
        ]
    time_feats_all = np.array([get_time_feats(i) for i in range(len(time_df))], dtype=np.float32)

    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]
    N, T, F = data.shape
    t_train = split['time_train_end']
    t_val = split['time_val_end']
    t_tot = split['time_total']
    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']

    # Pre-compute 1-hop and 2-hop neighbor sets
    nb1_cache, nb2_cache = {}, {}
    all_nodes = set(train_nodes) | set(heldout_nodes)
    for n in all_nodes:
        nb1 = set(adj.get(n, []))
        nb2 = set()
        for nb in nb1:
            nb2.update(adj.get(nb, []))
        nb2 = nb2 - {n} - nb1
        nb1_cache[n] = list(nb1)
        nb2_cache[n] = list(nb2)

    # All ATSPM targets (through only for simplicity, with observed data)
    target_metrics = [
        (0, 'AvgHeadway_Through'),
        (2, 'Volume_Through'),
        (4, 'SplitFailure_Through'),
        (6, 'RedRunning_Through'),
        (8, 'DilemmaRunning_Through'),
        (10, 'ConflictHeadway_Through'),
        (14, 'QueueLength_Through'),
    ]

    # ══════════════════════════════════════════════════════════
    # STEP 1: Neighbor-Average Baseline
    # ══════════════════════════════════════════════════════════
    log("\n" + "="*70)
    log("BASELINE: Nearest Neighbor Average (no ML)")
    log("="*70)
    
    baseline_results = {}
    for fidx, fname in target_metrics:
        preds, gts = [], []
        for n in heldout_nodes:
            nb_all = nb1_cache.get(n, []) + nb2_cache.get(n, [])
            for t in range(t_val, t_tot):
                if not mask[n, t, fidx]:
                    continue
                gt = data[n, t, fidx]
                # Find neighbor average
                nb_vals = []
                for nb in nb_all:
                    if mask[nb, t, fidx]:
                        nb_vals.append(data[nb, t, fidx])
                if nb_vals:
                    pred = np.mean(nb_vals)
                else:
                    continue  # no neighbor data, skip
                preds.append(pred)
                gts.append(gt)
        
        if len(preds) > 0:
            preds, gts = np.array(preds), np.array(gts)
            mae = np.mean(np.abs(preds - gts))
            gt_mean = np.mean(gts)
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            baseline_results[fname] = {'mae': mae, 'rel': rel, 'gt_mean': gt_mean}
            log(f"  {fname:<28} | GT: {gt_mean:>7.1f} | MAE: {mae:>7.1f} | Rel: {rel:>5.1f}%")

    # ══════════════════════════════════════════════════════════
    # STEP 2: Ratio-Based Expert Model
    # ══════════════════════════════════════════════════════════
    log("\n" + "="*70)
    log("RATIO MODEL: Predict target / neighbor_avg, then rescale")
    log("="*70)

    ratio_results = {}

    for fidx, fname in target_metrics:
        t0 = time.time()
        
        # Build ratio dataset
        X_tr_list, Y_ratio_tr, nb_avg_tr = [], [], []
        X_vl_list, Y_ratio_vl, nb_avg_vl = [], [], []
        X_te_list, Y_te_raw, nb_avg_te = [], [], []
        
        for node_list, t_start, t_end, x_out, yr_out, nb_out in [
            (train_nodes, 0, t_train, X_tr_list, Y_ratio_tr, nb_avg_tr),
            (train_nodes, t_train, t_val, X_vl_list, Y_ratio_vl, nb_avg_vl),
            (heldout_nodes, t_val, t_tot, X_te_list, Y_te_raw, nb_avg_te),
        ]:
            for n in node_list:
                nb_all = nb1_cache.get(n, []) + nb2_cache.get(n, [])
                p_feat = phys_mat[n]
                
                for t in range(t_start, t_end):
                    if not mask[n, t, fidx]:
                        continue
                    if mask[n, t, N_ATSPM:].sum() < 3:
                        continue
                    
                    gt = data[n, t, fidx]
                    
                    # Compute neighbor average for this metric
                    nb_vals = []
                    for nb in nb_all:
                        if mask[nb, t, fidx]:
                            nb_vals.append(data[nb, t, fidx])
                    
                    if not nb_vals:
                        continue
                    
                    nb_mean = np.mean(nb_vals)
                    
                    # Ratio = target / neighbor_avg (capped to avoid explosion)
                    if nb_mean > 0.01:
                        ratio = gt / nb_mean
                    else:
                        ratio = 1.0  # default
                    
                    # Clip extreme ratios
                    ratio = np.clip(ratio, 0.01, 20.0)
                    
                    tf = time_feats_all[t]
                    own_cv = data[n, t, N_ATSPM:]
                    
                    # Compute CV-based features that indicate relative intensity
                    # CV ratio to neighbor CV (if available)
                    cv_journey_idx = cv_names.index('segment_journeys')
                    my_cv = own_cv[cv_journey_idx]
                    nb_cv_vals = []
                    for nb in nb_all:
                        if mask[nb, t, cv_journey_idx + N_ATSPM]:
                            nb_cv_vals.append(data[nb, t, cv_journey_idx + N_ATSPM])
                    nb_cv_mean = np.mean(nb_cv_vals) if nb_cv_vals else 1.0
                    cv_ratio = my_cv / (nb_cv_mean + 0.01)  # Relative CV intensity
                    
                    feat = np.concatenate([
                        own_cv,              # 17
                        p_feat,              # 3
                        tf,                  # 5
                        [nb_mean],           # 1 (neighbor ATSPM avg)
                        [len(nb_vals)],      # 1 (number of neighbors with data)
                        [cv_ratio],          # 1 (relative CV intensity)
                        [np.std(nb_vals) if len(nb_vals) > 1 else 0],  # 1 (neighbor variance)
                        [np.min(nb_vals), np.max(nb_vals)],            # 2 (neighbor range)
                    ])
                    
                    x_out.append(feat)
                    yr_out.append(ratio)
                    nb_out.append(nb_mean)
        
        if len(X_tr_list) == 0 or len(X_te_list) == 0:
            log(f"  {fname:<28} | SKIP (insufficient data)")
            continue
            
        X_tr = np.nan_to_num(np.array(X_tr_list, dtype=np.float32))
        X_vl = np.nan_to_num(np.array(X_vl_list, dtype=np.float32))
        X_te = np.nan_to_num(np.array(X_te_list, dtype=np.float32))
        Y_r_tr = np.array(Y_ratio_tr, dtype=np.float32)
        Y_r_vl = np.array(Y_ratio_vl, dtype=np.float32)
        Y_raw_te = np.array(Y_te_raw, dtype=np.float32)
        nb_te = np.array(nb_avg_te, dtype=np.float32)
        
        # Train on log(ratio) for symmetry (ratio=0.5 and ratio=2 are equidistant)
        y_tr_log = np.log(Y_r_tr + 1e-6)
        y_vl_log = np.log(Y_r_vl + 1e-6)
        
        params = {
            'objective': 'regression',
            'metric': 'mse',
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
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr_log,
                  eval_set=[(X_vl, y_vl_log)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Predict ratio, then multiply by neighbor average
        pred_log_ratio = model.predict(X_te)
        pred_ratio = np.exp(pred_log_ratio)
        pred_final = pred_ratio * nb_te
        
        mae = np.mean(np.abs(pred_final - Y_raw_te))
        gt_mean = np.mean(Y_raw_te)
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        ratio_results[fname] = {'mae': mae, 'rel': rel, 'gt_mean': gt_mean}
        
        # Also compute: what if we just predict ratio=1.0 (i.e. naive baseline)?
        naive_mae = np.mean(np.abs(nb_te - Y_raw_te))
        naive_rel = naive_mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        
        # Importance
        imp = model.feature_importances_
        fnames = cv_names + ['thru_lanes', 'left_lanes', 'speed_limit'] + \
                 ['hour_sin', 'hour_cos', 'is_peak', 'hour', 'dow'] + \
                 ['nb_atspm_avg', 'nb_count', 'cv_ratio', 'nb_std', 'nb_min', 'nb_max']
        top3 = [(fnames[j], int(imp[j])) for j in np.argsort(imp)[::-1][:3]]
        
        log(f"  {fname:<28} | GT: {gt_mean:>7.1f} | Baseline: {naive_mae:>7.1f} ({naive_rel:>5.1f}%) → Ratio: {mae:>7.1f} ({rel:>5.1f}%) | {time.time()-t0:.0f}s")
        log(f"    Top features: {top3}")

    # ══════════════════════════════════════════════════════════
    # Summary comparison
    # ══════════════════════════════════════════════════════════
    log("\n" + "="*70)
    log("SUMMARY: Baseline vs Ratio Model vs Previous v1 Expert")
    log("="*70)
    log(f"  {'Metric':<28} | {'NB Avg':>10} | {'Ratio':>10} | {'v1 Expert':>10}")
    log(f"  {'':<28} | {'Rel%':>10} | {'Rel%':>10} | {'Rel%':>10}")
    log("-"*75)
    
    # Previous v1 results for reference
    v1_results = {
        'AvgHeadway_Through': 56.7,
        'Volume_Through': 60.5,
        'SplitFailure_Through': 87.5,
        'RedRunning_Through': 77.0,
        'DilemmaRunning_Through': 69.5,
        'ConflictHeadway_Through': 75.8,
        'QueueLength_Through': 96.7,
    }
    
    for _, fname in target_metrics:
        bl = baseline_results.get(fname, {})
        rt = ratio_results.get(fname, {})
        v1 = v1_results.get(fname, None)
        
        bl_str = f"{bl['rel']:>9.1f}%" if bl else f"{'N/A':>10}"
        rt_str = f"{rt['rel']:>9.1f}%" if rt else f"{'N/A':>10}"
        v1_str = f"{v1:>9.1f}%" if v1 else f"{'N/A':>10}"
        
        log(f"  {fname:<28} | {bl_str} | {rt_str} | {v1_str}")


if __name__ == '__main__':
    main()
