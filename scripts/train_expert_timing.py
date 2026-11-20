#!/usr/bin/env python3
"""
train_expert_timing.py — Cycle/Timing Expert Model
Goal: Estimate delay metrics, average headways, and split failures
      using CV kinematic control variables and spatial correlations.
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
    log("Loading data...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True).numpy()
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True).numpy()
    
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)

    # Phys Features
    phys_df = pd.read_csv(os.path.join(DATA_DIR, 'physical_features.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    
    phys_df['Seg_Unique'] = phys_df['Seg_Unique'].astype(str)
    node_map['Seg_Unique'] = node_map['Seg_Unique'].astype(str)
    merged = pd.merge(node_map, phys_df, on='Seg_Unique', how='left')
    merged['speed_limit_mph'] = merged['speed_limit_mph'].fillna(45)
    phys_mat = merged[['speed_limit_mph']].values.astype(np.float32)
    
    # Edge index
    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'], node_map['node_id']))
    v = edges_df[edges_df['source'].astype(str).isin(seg2id) & edges_df['target'].astype(str).isin(seg2id)]
    src = np.array([seg2id[str(s)] for s in v['source']])
    tgt = np.array([seg2id[str(s)] for s in v['target']])
    adj = build_adjacency([torch.tensor(src), torch.tensor(tgt)], len(node_map))

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
    
    timing_indices = [i for i, name in enumerate(atspm_names) if 'AvgHeadway' in name or 'SplitFailure' in name or 'QueueLength' in name]
    print(f"Timing targets indices: {[atspm_names[i] for i in timing_indices]}")

    def build_features(node_list, t_start, t_end):
        X_list, Y_list, obs_mask_list = [], [], []
        
        for n in node_list:
            nb1 = set(adj.get(n, []))
            nb2 = set()
            for nb in nb1:
                nb2.update(adj.get(nb, []))
            nb2 = list(nb2 - {n} - set(nb1))
            nb1 = list(nb1)
            
            p_feat = phys_mat[n]
            
            for t in range(t_start, t_end):
                if mask[n, t, N_ATSPM:].sum() < 3: continue
                # 2-hop timing ATSPM (Queue spills back, Split Failures cascade)
                nb2_tim = []
                for nb in nb2:
                    if mask[nb, t, timing_indices].sum() > 0:
                        nb2_tim.append(data[nb, t, timing_indices])
                nb2_mean = np.nanmean(nb2_tim, axis=0) if nb2_tim else np.zeros(len(timing_indices), dtype=np.float32)
                
                tf = time_feats_all[t]
                own_cv = data[n, t, N_ATSPM:]
                
                features = np.concatenate([
                    own_cv,            # 17
                    p_feat,            # 1 (Speed Limit)
                    tf,                # 5
                    nb2_mean           # len(timing_indices)
                ])
                
                X_list.append(features)
                Y_list.append(data[n, t, timing_indices].copy())
                obs_mask_list.append(mask[n, t, timing_indices].copy())
                
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), np.array(obs_mask_list, dtype=bool)

    t_train = split['time_train_end']
    t_val = split['time_val_end']
    t_tot = split['time_total']

    log("\nBuilding train features...")
    X_tr, Y_tr, _ = build_features(split['train_nodes'], 0, t_train)
    log(f"  Train: {X_tr.shape}")
    
    log("Building val features...")
    X_vl, Y_vl, _ = build_features(split['train_nodes'], t_train, t_val)
    
    log("Building test features...")
    X_te, Y_te, te_mask = build_features(split['heldout_nodes'], t_val, t_tot)

    X_tr = np.nan_to_num(X_tr)
    X_vl = np.nan_to_num(X_vl)
    X_te = np.nan_to_num(X_te)
    
    # Feature names
    fnames = cv_names + ['speed_limit'] + \
             ['hour_sin', 'hour_cos', 'is_peak', 'hour', 'dow'] + \
             [f'nb2_{atspm_names[v]}' for v in timing_indices]

    # Standard Regression for AvgHeadway, QueueLength, SplitFailure
    lgb_params = {
        'objective': 'regression_l1', # MAE
        'metric': 'mae',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'random_state': 42,
        'n_jobs': 8,
        'verbose': -1
    }

    log("\nTraining Cycle/Timing Expert Models...")
    preds = np.zeros_like(Y_te)
    
    for idx, vidx in enumerate(timing_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        
        tr_valid = Y_tr[:, idx] >= 0
        vl_valid = Y_vl[:, idx] >= 0
        
        y_t, y_v = Y_tr[tr_valid, idx], Y_vl[vl_valid, idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr[tr_valid], y_t,
            eval_set=[(X_vl[vl_valid], y_v)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds[:, idx] = model.predict(X_te)
        
        # Eval
        obs = te_mask[:, idx]
        p, t = preds[obs, idx], Y_te[obs, idx]
        mae = np.mean(np.abs(p - t))
        gt_mean = np.mean(t)
        rel_error = mae / gt_mean * 100 if gt_mean > 0 else np.nan
        
        log(f"  {name:<25} | GT Mean: {gt_mean:>6.1f} | MAE: {mae:>6.1f} | RelError: {rel_error:>5.1f}% | {time.time()-t0:.1f}s")
        
        # Importance
        imp = model.feature_importances_
        top3 = [(fnames[j], int(imp[j])) for j in np.argsort(imp)[::-1][:3]]
        log(f"    Top: {top3}")

if __name__ == '__main__':
    main()
