#!/usr/bin/env python3
"""
train_expert_multires.py — Multi-Resolution Expert Models
Compare 15-min, 30-min, 60-min temporal aggregation.
Uses v1 Expert architecture (proven best) at each resolution.
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

def aggregate_temporal(data, mask, time_df, factor):
    """Aggregate consecutive time slots by factor (2=30min, 4=60min).
    Volume/count metrics: SUM.  Rate/average metrics: MEAN.
    """
    N, T_old, F = data.shape
    T_new = T_old // factor
    
    data_agg = np.zeros((N, T_new, F), dtype=np.float32)
    mask_agg = np.zeros((N, T_new, F), dtype=bool)
    
    # For each new time slot, aggregate `factor` consecutive old slots
    for t_new in range(T_new):
        t_start = t_new * factor
        t_end = t_start + factor
        
        block_data = data[:, t_start:t_end, :]  # (N, factor, F)
        block_mask = mask[:, t_start:t_end, :]   # (N, factor, F)
        
        # Sum metrics: Volume (idx 2,3), RedRunning (6,7), DilemmaRunning (8,9),
        #              ConflictHeadway (10,11), ConflictPET (12,13), segment_journeys (31)
        # Mean metrics: AvgHeadway (0,1), SplitFailure (4,5), QueueLength (14),
        #               and all other CV features
        sum_indices = [2, 3, 6, 7, 8, 9, 10, 11, 12, 13]  # ATSPM counts
        # CV count features (add N_ATSPM offset)
        cv_count_indices = [N_ATSPM + 9, N_ATSPM + 10, N_ATSPM + 11, N_ATSPM + 12,
                           N_ATSPM + 13, N_ATSPM + 14, N_ATSPM + 15, N_ATSPM + 16]
        all_sum = sum_indices + cv_count_indices
        
        for f_idx in range(F):
            valid = block_mask[:, :, f_idx]  # (N, factor)
            vals = block_data[:, :, f_idx]   # (N, factor)
            
            any_valid = valid.any(axis=1)    # (N,)
            mask_agg[:, t_new, f_idx] = any_valid
            
            if f_idx in all_sum:
                # SUM: accumulate valid values
                data_agg[:, t_new, f_idx] = np.where(any_valid, np.nansum(np.where(valid, vals, 0), axis=1), 0)
            else:
                # MEAN: average valid values
                valid_count = valid.sum(axis=1).clip(min=1)
                data_agg[:, t_new, f_idx] = np.where(any_valid, np.nansum(np.where(valid, vals, 0), axis=1) / valid_count, 0)
    
    # Aggregate time index
    time_agg = time_df.iloc[::factor].reset_index(drop=True).head(T_new)
    
    return data_agg, mask_agg, time_agg


def run_expert(data, mask, time_df, split, features, phys_mat, adj, resolution_label):
    """Run the v1 Expert pipeline on the given data (any temporal resolution)."""
    log(f"\n{'='*60}")
    log(f"Resolution: {resolution_label}")
    log(f"{'='*60}")
    
    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]
    N, T, F = data.shape
    
    # Recompute split boundaries proportionally
    t_train = int(T * 0.70)
    t_val = int(T * 0.85)
    t_tot = T
    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    
    vol_indices = [i for i, name in enumerate(atspm_names) if 'Volume' in name]
    safety_indices = [i for i, name in enumerate(atspm_names) if 'Running' in name or 'Conflict' in name]
    timing_indices = [i for i, name in enumerate(atspm_names) if 'AvgHeadway' in name or 'SplitFailure' in name or 'QueueLength' in name]
    
    # Pre-compute time features
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
    time_feats_all = np.array([get_time_feats(i) for i in range(T)], dtype=np.float32)
    
    # Pre-compute neighbor sets
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

    def build_features(node_list, t_start, t_end, target_indices):
        X_list, Y_list, obs_mask_list = [], [], []
        for n in node_list:
            nb2 = nb2_cache.get(n, [])
            p_feat = phys_mat[n]
            for t in range(t_start, t_end):
                if mask[n, t, N_ATSPM:].sum() < 3:
                    continue
                nb2_tgt = []
                for nb in nb2:
                    if mask[nb, t, target_indices].sum() > 0:
                        nb2_tgt.append(data[nb, t, target_indices])
                nb2_mean = np.nanmean(nb2_tgt, axis=0) if nb2_tgt else np.zeros(len(target_indices), dtype=np.float32)
                nb2_count = np.float32(len(nb2_tgt))
                
                tf = time_feats_all[t]
                own_cv = data[n, t, N_ATSPM:]
                
                feats = np.concatenate([own_cv, p_feat, tf, nb2_mean, [nb2_count]])
                X_list.append(feats)
                Y_list.append(data[n, t, target_indices].copy())
                obs_mask_list.append(mask[n, t, target_indices].copy())
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), np.array(obs_mask_list, dtype=bool)

    # Penetration rate check
    cv_idx = cv_names.index('segment_journeys') + N_ATSPM
    cv_sum, vol_sum = 0.0, 0.0
    for n in train_nodes:
        for t in range(t_train):
            if mask[n, t, cv_idx] and mask[n, t, vol_indices[0]]:
                cv_v = data[n, t, cv_idx]
                vol_v = data[n, t, vol_indices[0]]
                if cv_v > 0 and vol_v > 0:
                    cv_sum += cv_v
                    vol_sum += vol_v
    pr = cv_sum / vol_sum * 100 if vol_sum > 0 else 0
    log(f"  Penetration Rate: {pr:.2f}%  (CV={cv_sum:.0f}, ATSPM={vol_sum:.0f})")
    log(f"  Shape: ({N}, {T}, {F})  Train/Val/Test time: {t_train}/{t_val-t_train}/{t_tot-t_val}")

    results = {}
    
    # --- Volume Expert ---
    log(f"\n  Volume Expert...")
    X_tr, Y_tr, _ = build_features(train_nodes, 0, t_train, vol_indices)
    X_vl, Y_vl, _ = build_features(train_nodes, t_train, t_val, vol_indices)
    X_te, Y_te, te_m = build_features(heldout_nodes, t_val, t_tot, vol_indices)
    X_tr, X_vl, X_te = np.nan_to_num(X_tr), np.nan_to_num(X_vl), np.nan_to_num(X_te)
    
    params_vol = {'objective': 'regression', 'metric': 'mse', 'n_estimators': 1000,
                  'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42, 'n_jobs': 8, 'verbose': -1}
    
    for idx, vidx in enumerate(vol_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        tr_v = Y_tr[:, idx] >= 0
        vl_v = Y_vl[:, idx] >= 0
        m = lgb.LGBMRegressor(**params_vol)
        m.fit(X_tr[tr_v], np.log1p(Y_tr[tr_v, idx]),
              eval_set=[(X_vl[vl_v], np.log1p(Y_vl[vl_v, idx]))],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = np.expm1(m.predict(X_te))
        obs = te_m[:, idx]
        if obs.sum() > 0:
            mae = np.mean(np.abs(preds[obs] - Y_te[obs, idx]))
            gt_m = np.mean(Y_te[obs, idx])
            rel = mae / gt_m * 100 if gt_m > 0 else float('nan')
            log(f"    {name:<25} | GT: {gt_m:>7.1f} | MAE: {mae:>7.1f} | Rel: {rel:>5.1f}% | {time.time()-t0:.0f}s")
            results[name] = {'mae': mae, 'gt_mean': gt_m, 'rel': rel}
    
    del X_tr, X_vl, X_te, Y_tr, Y_vl, Y_te

    # --- Safety Expert ---
    log(f"\n  Safety Expert...")
    X_tr, Y_tr, _ = build_features(train_nodes, 0, t_train, safety_indices)
    X_vl, Y_vl, _ = build_features(train_nodes, t_train, t_val, safety_indices)
    X_te, Y_te, te_m = build_features(heldout_nodes, t_val, t_tot, safety_indices)
    X_tr, X_vl, X_te = np.nan_to_num(X_tr), np.nan_to_num(X_vl), np.nan_to_num(X_te)
    
    params_saf = {'objective': 'poisson', 'metric': 'poisson', 'n_estimators': 1000,
                  'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42, 'n_jobs': 8, 'verbose': -1}
    
    for idx, vidx in enumerate(safety_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        tr_v = Y_tr[:, idx] >= 0
        vl_v = Y_vl[:, idx] >= 0
        y_t = np.round(Y_tr[tr_v, idx])
        y_v = np.round(Y_vl[vl_v, idx])
        if np.sum(y_t) == 0:
            log(f"    {name:<25} | SKIP (all zeros)")
            continue
        try:
            m = lgb.LGBMRegressor(**params_saf)
            m.fit(X_tr[tr_v], y_t, eval_set=[(X_vl[vl_v], y_v)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = m.predict(X_te)
            obs = te_m[:, idx]
            if obs.sum() > 0:
                mae = np.mean(np.abs(preds[obs] - Y_te[obs, idx]))
                gt_m = np.mean(Y_te[obs, idx])
                rel = mae / gt_m * 100 if gt_m > 0 else float('nan')
                log(f"    {name:<25} | GT: {gt_m:>7.2f} | MAE: {mae:>7.2f} | Rel: {rel:>5.1f}% | {time.time()-t0:.0f}s")
                results[name] = {'mae': mae, 'gt_mean': gt_m, 'rel': rel}
        except Exception as e:
            log(f"    {name:<25} | ERROR: {e}")
    
    del X_tr, X_vl, X_te, Y_tr, Y_vl, Y_te

    # --- Timing Expert ---
    log(f"\n  Timing Expert...")
    X_tr, Y_tr, _ = build_features(train_nodes, 0, t_train, timing_indices)
    X_vl, Y_vl, _ = build_features(train_nodes, t_train, t_val, timing_indices)
    X_te, Y_te, te_m = build_features(heldout_nodes, t_val, t_tot, timing_indices)
    X_tr, X_vl, X_te = np.nan_to_num(X_tr), np.nan_to_num(X_vl), np.nan_to_num(X_te)
    
    params_tim = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
                  'learning_rate': 0.05, 'num_leaves': 63, 'random_state': 42, 'n_jobs': 8, 'verbose': -1}
    
    for idx, vidx in enumerate(timing_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        tr_v = Y_tr[:, idx] >= 0
        vl_v = Y_vl[:, idx] >= 0
        m = lgb.LGBMRegressor(**params_tim)
        m.fit(X_tr[tr_v], Y_tr[tr_v, idx], eval_set=[(X_vl[vl_v], Y_vl[vl_v, idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = m.predict(X_te)
        obs = te_m[:, idx]
        if obs.sum() > 0:
            mae = np.mean(np.abs(preds[obs] - Y_te[obs, idx]))
            gt_m = np.mean(Y_te[obs, idx])
            rel = mae / gt_m * 100 if gt_m > 0 else float('nan')
            log(f"    {name:<25} | GT: {gt_m:>7.1f} | MAE: {mae:>7.1f} | Rel: {rel:>5.1f}% | {time.time()-t0:.0f}s")
            results[name] = {'mae': mae, 'gt_mean': gt_m, 'rel': rel}
    
    return results


def main():
    log("Loading base data (15-min resolution)...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True).numpy()
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True).numpy()
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)
    time_df = pd.read_csv(os.path.join(DATA_DIR, 'time_index_v3.csv'), parse_dates=['time'])
    
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

    all_results = {}
    
    # 15-min (baseline)
    r15 = run_expert(data, mask, time_df, split, features, phys_mat, adj, "15-min (baseline)")
    all_results['15min'] = r15
    
    # 30-min
    log("\n\nAggregating to 30-min...")
    d30, m30, t30 = aggregate_temporal(data, mask, time_df, 2)
    r30 = run_expert(d30, m30, t30, split, features, phys_mat, adj, "30-min")
    all_results['30min'] = r30
    del d30, m30
    
    # 60-min
    log("\n\nAggregating to 60-min...")
    d60, m60, t60 = aggregate_temporal(data, mask, time_df, 4)
    r60 = run_expert(d60, m60, t60, split, features, phys_mat, adj, "60-min")
    all_results['60min'] = r60
    del d60, m60

    # Summary comparison
    log("\n\n" + "="*80)
    log("COMPARISON: 15min vs 30min vs 60min")
    log("="*80)
    log(f"{'Metric':<30} | {'15min MAE':>10} {'Rel%':>6} | {'30min MAE':>10} {'Rel%':>6} | {'60min MAE':>10} {'Rel%':>6}")
    log("-"*100)
    
    all_metrics = set()
    for res in all_results.values():
        all_metrics.update(res.keys())
    
    for metric in sorted(all_metrics):
        parts = []
        for res_key in ['15min', '30min', '60min']:
            r = all_results[res_key].get(metric, {})
            if r:
                parts.append(f"{r['mae']:>10.1f} {r['rel']:>5.1f}%")
            else:
                parts.append(f"{'N/A':>10} {'N/A':>6}")
        log(f"  {metric:<28} | {' | '.join(parts)}")

if __name__ == '__main__':
    main()
