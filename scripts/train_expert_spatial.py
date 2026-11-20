#!/usr/bin/env python3
"""
train_expert_spatial.py — Spatial Interpolation Methods Comparison
Builds on the finding that simple neighbor averaging (43.8% RelError on Volume)
already beats the v1 Expert model (60.5%).

Methods:
  1. Naive Neighbor Average (baseline)
  2. IDW - Inverse Distance Weighting (graph hops as distance)
  3. CV-Similarity Weighted Average (weight neighbors by CV pattern similarity)
  4. Residual ML - Neighbor Average + LightGBM correction on residuals
  5. Combined: IDW + CV-Similarity + Residual ML
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
    log("Spatial Interpolation Methods Comparison")
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

    # Pre-compute neighbor sets WITH hop distance
    nb_with_dist = {}  # node -> [(neighbor, hop_distance), ...]
    for n in set(train_nodes) | set(heldout_nodes):
        neighbors = {}
        # 1-hop
        for nb in adj.get(n, []):
            neighbors[nb] = 1
        # 2-hop
        for nb1 in adj.get(n, []):
            for nb2 in adj.get(nb1, []):
                if nb2 != n and nb2 not in neighbors:
                    neighbors[nb2] = 2
        # 3-hop
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

    # CV feature indices for similarity computation
    cv_feat_start = N_ATSPM
    n_cv = len(cv_names)

    all_results = {m: {} for m in ['NB_Avg', 'IDW', 'CV_Wt', 'Residual', 'Combined']}

    for fidx, fname in target_metrics:
        log(f"\n{'─'*60}")
        log(f"  {fname}")
        log(f"{'─'*60}")
        t0 = time.time()

        # Collect test samples
        test_preds = {m: [] for m in all_results}
        test_gts = []
        residual_features_tr, residual_targets_tr = [], []
        residual_features_vl, residual_targets_vl = [], []
        residual_features_te, residual_nb_avgs_te = [], []

        # ════════════════════════════════════════
        # Phase 1: Build residual training data from train nodes
        # ════════════════════════════════════════
        for node_list, t_start, t_end, feat_out, tgt_out in [
            (train_nodes, 0, t_train, residual_features_tr, residual_targets_tr),
            (train_nodes, t_train, t_val, residual_features_vl, residual_targets_vl),
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
                    residual = gt - nb_avg  # what the ML needs to correct
                    
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
                                # Cosine similarity on CV features
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
                    
                    feat = np.concatenate([
                        own_cv,                 # 17
                        p_feat,                 # 3
                        tf,                     # 5
                        [nb_avg],               # 1 (neighbor avg)
                        [idw_pred],             # 1 (IDW prediction)
                        [cv_pred],              # 1 (CV-weighted prediction)
                        [len(nb_vals)],         # 1 (num neighbors)
                        [np.std(nb_vals) if len(nb_vals) > 1 else 0],  # 1
                        [np.min(nb_vals), np.max(nb_vals)],            # 2
                        [np.median(nb_vals)],   # 1
                    ])
                    feat_out.append(feat)
                    tgt_out.append(residual)

        # ════════════════════════════════════════
        # Phase 2: Train residual ML model
        # ════════════════════════════════════════
        if len(residual_features_tr) > 0 and len(residual_features_vl) > 0:
            X_tr = np.nan_to_num(np.array(residual_features_tr, dtype=np.float32))
            Y_tr = np.array(residual_targets_tr, dtype=np.float32)
            X_vl = np.nan_to_num(np.array(residual_features_vl, dtype=np.float32))
            Y_vl = np.array(residual_targets_vl, dtype=np.float32)
            
            params = {
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
            
            residual_model = lgb.LGBMRegressor(**params)
            residual_model.fit(X_tr, Y_tr,
                              eval_set=[(X_vl, Y_vl)],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            
            # Feature importance
            fnames = cv_names + ['thru_lanes', 'left_lanes', 'speed_limit'] + \
                     ['hour_sin', 'hour_cos', 'is_peak', 'hour', 'dow'] + \
                     ['nb_avg', 'idw_pred', 'cv_pred', 'nb_count', 'nb_std', 'nb_min', 'nb_max', 'nb_median']
            imp = residual_model.feature_importances_
            top3 = [(fnames[j], int(imp[j])) for j in np.argsort(imp)[::-1][:3]]
            log(f"  Residual model top features: {top3}")
        else:
            residual_model = None

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
                nb_with_cv = []  # (value, cv_similarity)
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
                
                # Method 4: Residual ML
                if residual_model is not None:
                    tf = time_feats_all[t]
                    feat = np.concatenate([
                        own_cv, p_feat, tf,
                        [nb_avg], [idw_pred], [cv_pred],
                        [len(nb_vals)],
                        [np.std(nb_vals) if len(nb_vals) > 1 else 0],
                        [np.min(nb_vals), np.max(nb_vals)],
                        [np.median(nb_vals)],
                    ])
                    correction = residual_model.predict(np.nan_to_num(feat.reshape(1, -1)))[0]
                    test_preds['Residual'].append(nb_avg + correction)
                else:
                    test_preds['Residual'].append(nb_avg)
                
                # Method 5: Combined (IDW-CV blend + Residual correction)
                # Blend IDW and CV-weighted, then add residual
                blend = 0.5 * idw_pred + 0.5 * cv_pred
                if residual_model is not None:
                    feat_comb = np.concatenate([
                        own_cv, p_feat, tf,
                        [blend], [idw_pred], [cv_pred],
                        [len(nb_vals)],
                        [np.std(nb_vals) if len(nb_vals) > 1 else 0],
                        [np.min(nb_vals), np.max(nb_vals)],
                        [np.median(nb_vals)],
                    ])
                    correction_comb = residual_model.predict(np.nan_to_num(feat_comb.reshape(1, -1)))[0]
                    test_preds['Combined'].append(blend + correction_comb)
                else:
                    test_preds['Combined'].append(blend)

        # Evaluate
        gts = np.array(test_gts)
        gt_mean = np.mean(gts)
        
        for method in all_results:
            preds = np.array(test_preds[method])
            mae = np.mean(np.abs(preds - gts))
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            all_results[method][fname] = {'mae': mae, 'rel': rel, 'gt_mean': gt_mean}
        
        elapsed = time.time() - t0
        row = " | ".join(f"{m}: {all_results[m][fname]['rel']:>5.1f}%" for m in all_results)
        log(f"  GT={gt_mean:.1f} | {row} | {elapsed:.0f}s")

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    log("\n" + "="*90)
    log("FINAL COMPARISON")
    log("="*90)
    header = f"  {'Metric':<28}"
    for m in all_results:
        header += f" | {m:>10}"
    log(header)
    log("-"*90)
    
    for _, fname in target_metrics:
        row = f"  {fname:<28}"
        rels = []
        for m in all_results:
            r = all_results[m].get(fname, {})
            rel = r.get('rel', float('nan'))
            rels.append(rel)
            row += f" | {rel:>9.1f}%"
        # Mark best
        best_idx = np.nanargmin(rels)
        best_name = list(all_results.keys())[best_idx]
        row += f"  ← {best_name}"
        log(row)
    
    # Average across all metrics
    log("-"*90)
    row = f"  {'AVERAGE':<28}"
    for m in all_results:
        avg = np.nanmean([all_results[m][fn]['rel'] for _, fn in target_metrics if fn in all_results[m]])
        row += f" | {avg:>9.1f}%"
    log(row)

if __name__ == '__main__':
    main()
