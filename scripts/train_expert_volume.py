#!/usr/bin/env python3
"""
train_expert_volume.py — Volume Expert Model
Goal: Explicitly account for CV Penetration Rate by scaling CV counts 
      using physical intersection characteristics (Lane Count, Speed Limit)
      and spatial constraints (2-hop Neighbor Volume).
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
    
    # Map physical features to node index
    phys_df['Seg_Unique'] = phys_df['Seg_Unique'].astype(str)
    node_map['Seg_Unique'] = node_map['Seg_Unique'].astype(str)
    merged = pd.merge(node_map, phys_df, on='Seg_Unique', how='left')
    merged['thru_lanes'] = merged['thru_lanes'].fillna(2)
    merged['left_lanes'] = merged['left_lanes'].fillna(1)
    merged['speed_limit_mph'] = merged['speed_limit_mph'].fillna(45)
    phys_mat = merged[['thru_lanes', 'left_lanes', 'speed_limit_mph']].values.astype(np.float32)
    
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
            math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7),
            1.0 if (7 <= h <= 9 or 16 <= h <= 18) else 0.0,
            h, d
        ]
    
    time_feats_all = np.array([get_time_feats(i) for i in range(len(time_df))], dtype=np.float32)

    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]
    
    vol_indices = [i for i, name in enumerate(atspm_names) if 'Volume' in name]
    print(f"Volume targets indices: {vol_indices}")

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
                # 2-hop volume ATSPM
                nb2_vol = []
                for nb in nb2:
                    if mask[nb, t, vol_indices].sum() > 0:
                        nb2_vol.append(data[nb, t, vol_indices])
                nb2_mean = np.nanmean(nb2_vol, axis=0) if nb2_vol else np.zeros(len(vol_indices), dtype=np.float32)
                nb2_count = np.float32(len(nb2_vol))
                
                tf = time_feats_all[t]
                own_cv = data[n, t, N_ATSPM:]
                
                # Penetration Rate Interaction: CV_Journeys / (Lanes * SpeedLimit + 1)
                cv_journeys = own_cv[cv_names.index('segment_journeys')]
                pen_rate_proxy1 = cv_journeys / (p_feat[0] + 1)
                pen_rate_proxy2 = cv_journeys / (p_feat[1] + 1)
                
                features = np.concatenate([
                    own_cv,            # 17
                    p_feat,            # 3
                    tf,                # 7
                    nb2_mean,          # 2
                    [nb2_count],       # 1
                    [pen_rate_proxy1, pen_rate_proxy2] # 2 (Explicit penetration features)
                ])
                
                X_list.append(features)
                Y_list.append(data[n, t, vol_indices].copy())
                obs_mask_list.append(mask[n, t, vol_indices].copy())
                
        return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32), np.array(obs_mask_list, dtype=bool)

    t_train = split['time_train_end']
    t_val = split['time_val_end']
    t_tot = split['time_total']

    log("\nBuilding train features...")
    X_tr, Y_tr, _ = build_features(split['train_nodes'], 0, t_train)
    log(f"  Train: {X_tr.shape}")
    
    log("Building val features...")
    X_vl, Y_vl, _ = build_features(split['train_nodes'], t_train, t_val)
    log(f"  Val: {X_vl.shape}")
    
    log("Building test features...")
    X_te, Y_te, te_mask = build_features(split['heldout_nodes'], t_val, t_tot)
    log(f"  Test: {X_te.shape}")

    X_tr = np.nan_to_num(X_tr)
    X_vl = np.nan_to_num(X_vl)
    X_te = np.nan_to_num(X_te)
    
    # Feature names
    fnames = cv_names + ['thru_lanes', 'left_lanes', 'speed_limit'] + \
             ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_peak', 'hour', 'dow'] + \
             [f'nb2_{atspm_names[v]}' for v in vol_indices] + ['nb2_count', 'pen_proxy_thru', 'pen_proxy_left']

    # Switch to MSE on log-transformed targets instead of unstable MAPE
    lgb_params = {
        'objective': 'regression',
        'metric': 'mse',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'random_state': 42,
        'n_jobs': 8,
        'verbose': -1
    }

    log("\nTraining Expert Models for Volume...")
    preds = np.zeros_like(Y_te)
    
    for idx, vidx in enumerate(vol_indices):
        name = atspm_names[vidx]
        t0 = time.time()
        
        # We model Volume using log1p to directly optimize for relative percentage error smoothly
        tr_valid = Y_tr[:, idx] >= 0
        vl_valid = Y_vl[:, idx] >= 0
        
        y_train_log = np.log1p(Y_tr[tr_valid, idx])
        y_val_log = np.log1p(Y_vl[vl_valid, idx])
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr[tr_valid], y_train_log,
            eval_set=[(X_vl[vl_valid], y_val_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds_log = model.predict(X_te)
        preds[:, idx] = np.expm1(preds_log)
        
        # Eval
        obs = te_mask[:, idx]
        p, t = preds[obs, idx], Y_te[obs, idx]
        mae = np.mean(np.abs(p - t))
        mape = np.mean(np.abs(p - t) / (t + 1e-6)) * 100
        gt_mean = np.mean(t)
        rel_error = mae / gt_mean * 100
        
        log(f"  {name:<25} | GT Mean: {gt_mean:>6.1f} | MAE: {mae:>6.1f} | RelError: {rel_error:>5.1f}% | MAPE: {mape:>5.1f}% | {time.time()-t0:.1f}s")
        
        # Importance
        imp = model.feature_importances_
        top5 = [(fnames[j], int(imp[j])) for j in np.argsort(imp)[::-1][:5]]
        log(f"    Top: {top5}")

if __name__ == '__main__':
    main()
