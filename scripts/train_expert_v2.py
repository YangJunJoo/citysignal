#!/usr/bin/env python3
"""
train_expert_v2.py — Enhanced Expert Model Pipeline
Enhancements over v1:
  1. Time-varying Penetration Rate (hourly PR table from train nodes)
  2. Stacking: Volume Expert output → Safety/Timing input features
  3. Target Encoding: node-level baseline from training data
"""

import os, json, time, math
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
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

def main():
    log("="*60)
    log("Expert v2: PR Table + Stacking + Target Encoding")
    log("="*60)

    # ── Load data ──
    log("\n[1/7] Loading data...")
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
    hours_all = np.array([t.hour for t in time_df['time']], dtype=np.int32)
    dows_all = np.array([t.dayofweek for t in time_df['time']], dtype=np.int32)
    
    def get_time_feats(t_idx):
        t = time_df.iloc[t_idx]['time']
        h = t.hour + t.minute / 60.0
        d = t.dayofweek
        return [
            math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
            math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7),
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

    vol_indices = [i for i, name in enumerate(atspm_names) if 'Volume' in name]
    safety_indices = [i for i, name in enumerate(atspm_names) if 'Running' in name or 'Conflict' in name]
    timing_indices = [i for i, name in enumerate(atspm_names) if 'AvgHeadway' in name or 'SplitFailure' in name or 'QueueLength' in name]
    cv_journey_idx = cv_names.index('segment_journeys')

    # ── Enhancement 1: Time-varying Penetration Rate Table ──
    log("\n[2/7] Building hourly Penetration Rate table from train nodes...")
    # PR(hour) = sum(CV_journeys) / sum(ATSPM_Volume_Through) for each hour bucket
    pr_table = np.ones(24, dtype=np.float32) * 0.01  # default 1%
    for h in range(24):
        cv_sum, vol_sum = 0.0, 0.0
        for n in train_nodes:
            for t in range(t_train):
                if hours_all[t] != h:
                    continue
                if mask[n, t, cv_journey_idx + N_ATSPM] and mask[n, t, vol_indices[0]]:
                    cv_val = data[n, t, cv_journey_idx + N_ATSPM]
                    vol_val = data[n, t, vol_indices[0]]
                    if cv_val > 0 and vol_val > 0:
                        cv_sum += cv_val
                        vol_sum += vol_val
        if vol_sum > 0:
            pr_table[h] = cv_sum / vol_sum
    
    log("  Hourly PR table:")
    for h in range(24):
        log(f"    {h:02d}:00 → {pr_table[h]*100:.2f}%")
    log(f"  Range: {pr_table.min()*100:.2f}% – {pr_table.max()*100:.2f}%")

    # ── Enhancement 3: Target Encoding ──
    log("\n[3/7] Building Target Encoding (node-level baselines from train period)...")
    # For each node, compute mean of each ATSPM metric during train period
    # For heldout nodes: use mean of physically similar nodes (same speed_limit bucket + similar lane count)
    node_target_enc = np.zeros((N, N_ATSPM), dtype=np.float32)
    
    for n in train_nodes:
        for fi in range(N_ATSPM):
            vals = data[n, :t_train, fi][mask[n, :t_train, fi]]
            node_target_enc[n, fi] = np.mean(vals) if len(vals) > 0 else 0.0

    # For heldout nodes: find similar train nodes by physical features
    for n in heldout_nodes:
        sl = phys_mat[n, 2]  # speed_limit
        tl = phys_mat[n, 0]  # thru_lanes
        # Find train nodes with similar physical characteristics
        dists = np.abs(phys_mat[train_nodes, 2] - sl) + np.abs(phys_mat[train_nodes, 0] - tl) * 5
        top_k = np.argsort(dists)[:10]
        similar_nodes = [train_nodes[k] for k in top_k]
        node_target_enc[n] = np.mean([node_target_enc[sn] for sn in similar_nodes], axis=0)
    
    log(f"  Target enc shape: {node_target_enc.shape}")
    log(f"  Example (node 0): Vol_Thru={node_target_enc[0, vol_indices[0]]:.1f}")

    # ── Pre-compute neighbor sets ──
    log("\n[4/7] Pre-computing neighbor sets...")
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

    # ── Build features (unified for all 3 experts) ──
    def build_features(node_list, t_start, t_end, target_indices, extra_preds=None):
        """
        extra_preds: dict of {(n,t): np.array} — stacked predictions from previous expert
        """
        X_list, Y_list, obs_mask_list, meta_list = [], [], [], []
        
        for n in node_list:
            nb1 = nb1_cache.get(n, [])
            nb2 = nb2_cache.get(n, [])
            p_feat = phys_mat[n]
            te_feat = node_target_enc[n, target_indices]  # Target Encoding for this expert's targets
            
            for t in range(t_start, t_end):
                if mask[n, t, N_ATSPM:].sum() < 3:
                    continue
                
                # 1-hop neighbor ATSPM
                nb1_tgt = []
                for nb in nb1:
                    if mask[nb, t, target_indices].sum() > 0:
                        nb1_tgt.append(data[nb, t, target_indices])
                nb1_mean = np.nanmean(nb1_tgt, axis=0) if nb1_tgt else np.zeros(len(target_indices), dtype=np.float32)
                
                # 2-hop neighbor ATSPM
                nb2_tgt = []
                for nb in nb2:
                    if mask[nb, t, target_indices].sum() > 0:
                        nb2_tgt.append(data[nb, t, target_indices])
                nb2_mean = np.nanmean(nb2_tgt, axis=0) if nb2_tgt else np.zeros(len(target_indices), dtype=np.float32)
                nb2_count = np.float32(len(nb2_tgt))
                
                tf = time_feats_all[t]
                own_cv = data[n, t, N_ATSPM:]
                
                # Enhancement 1: Hourly PR as feature
                h = hours_all[t]
                pr = pr_table[h]
                cv_journeys = own_cv[cv_journey_idx]
                estimated_volume = cv_journeys / (pr + 1e-8)  # Explicit volume estimate
                
                feat_parts = [
                    own_cv,            # 17
                    p_feat,            # 3
                    tf,                # 7
                    nb1_mean,          # len(target_indices)
                    nb2_mean,          # len(target_indices)
                    [nb2_count],       # 1
                    te_feat,           # len(target_indices) — Target Encoding
                    [pr, estimated_volume],  # 2 — Hourly PR + estimated volume
                ]
                
                # Enhancement 2: Stacked volume predictions
                if extra_preds is not None:
                    key = (n, t)
                    if key in extra_preds:
                        feat_parts.append(extra_preds[key])
                    else:
                        feat_parts.append(np.zeros(len(vol_indices), dtype=np.float32))
                
                features = np.concatenate(feat_parts)
                X_list.append(features)
                Y_list.append(data[n, t, target_indices].copy())
                obs_mask_list.append(mask[n, t, target_indices].copy())
                meta_list.append((n, t))
                
        return (np.array(X_list, dtype=np.float32), 
                np.array(Y_list, dtype=np.float32), 
                np.array(obs_mask_list, dtype=bool),
                meta_list)

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Volume Expert
    # ══════════════════════════════════════════════════════════
    log("\n[5/7] Training Volume Expert (Stage 1)...")
    X_tr, Y_tr, _, meta_tr = build_features(train_nodes, 0, t_train, vol_indices)
    X_vl, Y_vl, _, meta_vl = build_features(train_nodes, t_train, t_val, vol_indices)
    X_te, Y_te, te_mask_vol, meta_te = build_features(heldout_nodes, t_val, t_tot, vol_indices)
    
    X_tr = np.nan_to_num(X_tr)
    X_vl = np.nan_to_num(X_vl)
    X_te = np.nan_to_num(X_te)
    log(f"  Train: {X_tr.shape} | Val: {X_vl.shape} | Test: {X_te.shape}")

    lgb_vol_params = {
        'objective': 'regression',
        'metric': 'mse',
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 8,
        'verbose': -1,
    }

    vol_preds_te = np.zeros_like(Y_te)
    vol_models = {}
    
    # Also generate predictions for ALL train+val+test data to use as stacking features
    vol_stacked = {}  # (n, t) → predicted volume array
    
    for idx, vidx in enumerate(vol_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        
        tr_valid = Y_tr[:, idx] >= 0
        vl_valid = Y_vl[:, idx] >= 0
        
        y_train_log = np.log1p(Y_tr[tr_valid, idx])
        y_val_log = np.log1p(Y_vl[vl_valid, idx])
        
        model = lgb.LGBMRegressor(**lgb_vol_params)
        model.fit(
            X_tr[tr_valid], y_train_log,
            eval_set=[(X_vl[vl_valid], y_val_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        vol_models[vidx] = model
        
        # Predict on test
        vol_preds_te[:, idx] = np.expm1(model.predict(X_te))
        
        # Eval
        obs = te_mask_vol[:, idx]
        if obs.sum() > 0:
            p, gt = vol_preds_te[obs, idx], Y_te[obs, idx]
            mae = np.mean(np.abs(p - gt))
            gt_mean = np.mean(gt)
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            log(f"  {name:<25} | GT Mean: {gt_mean:>7.1f} | MAE: {mae:>7.1f} | RelError: {rel:>5.1f}% | {time.time()-t0:.1f}s")
        else:
            log(f"  {name:<25} | No observed test data | {time.time()-t0:.1f}s")

    # Generate stacking features using BATCH predictions (vectorized)
    log("  Generating stacking features (batch)...")
    
    # Batch predict for all data splits and build (n,t) → vol_pred mapping
    vol_stacked = {}  # (n, t) → np.array of predicted volumes
    
    for meta_list_src, X_src, label in [(meta_tr, X_tr, "train"), (meta_vl, X_vl, "val"), (meta_te, X_te, "test")]:
        # Batch prediction for each volume model
        batch_preds = np.zeros((len(meta_list_src), len(vol_indices)), dtype=np.float32)
        for idx, vidx in enumerate(vol_indices):
            batch_preds[:, idx] = np.expm1(vol_models[vidx].predict(X_src))
        # Map to (n, t) dict
        for i, (n, t) in enumerate(meta_list_src):
            vol_stacked[(n, t)] = batch_preds[i]
        log(f"    {label}: {len(meta_list_src)} samples cached")
    
    del X_tr, X_vl  # free memory (keep X_te for later)
    log(f"  Stacking cache size: {len(vol_stacked)}")

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Safety Expert (with stacked Volume)
    # ══════════════════════════════════════════════════════════
    log("\n[6/7] Training Safety Expert (Stage 2, with Volume stacking)...")
    X_tr2, Y_tr2, _, _ = build_features(train_nodes, 0, t_train, safety_indices, extra_preds=vol_stacked)
    X_vl2, Y_vl2, _, _ = build_features(train_nodes, t_train, t_val, safety_indices, extra_preds=vol_stacked)
    X_te2, Y_te2, te_mask_saf, _ = build_features(heldout_nodes, t_val, t_tot, safety_indices, extra_preds=vol_stacked)
    
    X_tr2 = np.nan_to_num(X_tr2)
    X_vl2 = np.nan_to_num(X_vl2)
    X_te2 = np.nan_to_num(X_te2)
    log(f"  Train: {X_tr2.shape} | Val: {X_vl2.shape} | Test: {X_te2.shape}")

    lgb_safety_params = {
        'objective': 'poisson',
        'metric': 'poisson',
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 8,
        'verbose': -1,
    }

    for idx, vidx in enumerate(safety_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        tr_valid = Y_tr2[:, idx] >= 0
        vl_valid = Y_vl2[:, idx] >= 0
        y_t = np.round(Y_tr2[tr_valid, idx])
        y_v = np.round(Y_vl2[vl_valid, idx])
        
        if np.sum(y_t) == 0:
            log(f"  {name:<25} | SKIPPED (all zeros)")
            continue
        
        model = lgb.LGBMRegressor(**lgb_safety_params)
        model.fit(
            X_tr2[tr_valid], y_t,
            eval_set=[(X_vl2[vl_valid], y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict(X_te2)
        
        obs = te_mask_saf[:, idx]
        if obs.sum() > 0:
            p, gt = preds[obs], Y_te2[obs, idx]
            mae = np.mean(np.abs(p - gt))
            gt_mean = np.mean(gt)
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            log(f"  {name:<25} | GT Mean: {gt_mean:>7.2f} | MAE: {mae:>7.2f} | RelError: {rel:>5.1f}% | {time.time()-t0:.1f}s")
        else:
            log(f"  {name:<25} | No observed test data | {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Cycle/Timing Expert (with stacked Volume)
    # ══════════════════════════════════════════════════════════
    log("\n[7/7] Training Cycle/Timing Expert (Stage 3, with Volume stacking)...")
    X_tr3, Y_tr3, _, _ = build_features(train_nodes, 0, t_train, timing_indices, extra_preds=vol_stacked)
    X_vl3, Y_vl3, _, _ = build_features(train_nodes, t_train, t_val, timing_indices, extra_preds=vol_stacked)
    X_te3, Y_te3, te_mask_tim, _ = build_features(heldout_nodes, t_val, t_tot, timing_indices, extra_preds=vol_stacked)
    
    X_tr3 = np.nan_to_num(X_tr3)
    X_vl3 = np.nan_to_num(X_vl3)
    X_te3 = np.nan_to_num(X_te3)
    log(f"  Train: {X_tr3.shape} | Val: {X_vl3.shape} | Test: {X_te3.shape}")

    lgb_timing_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 8,
        'verbose': -1,
    }

    for idx, vidx in enumerate(timing_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        tr_valid = Y_tr3[:, idx] >= 0
        vl_valid = Y_vl3[:, idx] >= 0
        
        model = lgb.LGBMRegressor(**lgb_timing_params)
        model.fit(
            X_tr3[tr_valid], Y_tr3[tr_valid, idx],
            eval_set=[(X_vl3[vl_valid], Y_vl3[vl_valid, idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds = model.predict(X_te3)
        
        obs = te_mask_tim[:, idx]
        if obs.sum() > 0:
            p, gt = preds[obs], Y_te3[obs, idx]
            mae = np.mean(np.abs(p - gt))
            gt_mean = np.mean(gt)
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            log(f"  {name:<25} | GT Mean: {gt_mean:>7.1f} | MAE: {mae:>7.1f} | RelError: {rel:>5.1f}% | {time.time()-t0:.1f}s")
        else:
            log(f"  {name:<25} | No observed test data | {time.time()-t0:.1f}s")

    log("\n" + "="*60)
    log("Done! All 3 Expert stages complete.")

if __name__ == '__main__':
    main()
