#!/usr/bin/env python3
"""Shared utilities for imputation experiments."""
import os, json, numpy as np, pandas as pd, torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
WINDOW = 24; NODE_SAMPLE = 200; BATCH = 4; EPOCHS = 80; LR = 0.001
HIDDEN = 64; MASK_EXTRA = 0.10; N_ATSPM = 15; N_CV = 17

def log(msg): print(msg, flush=True)

def load_all():
    log("Loading tensors...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True)
    time_df = pd.read_csv(os.path.join(DATA_DIR, 'time_index_v3.csv'), parse_dates=['time'])
    hours = torch.tensor(time_df['time'].dt.hour.values, dtype=torch.float32)
    dows = torch.tensor(time_df['time'].dt.dayofweek.values, dtype=torch.long)
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f: split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f: features = json.load(f)
    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    src_list, tgt_list = [], []
    for s, t in zip(edges_df['source'], edges_df['target']):
        s_str, t_str = str(int(s)), str(int(t))
        if s_str in seg2id and t_str in seg2id:
            src_list.append(seg2id[s_str]); tgt_list.append(seg2id[t_str])
    edge_index = torch.tensor(np.stack([src_list, tgt_list]), dtype=torch.long)
    log(f"  data: {data.shape}, mask: {mask.shape}, edge_index: {edge_index.shape}")
    return data, mask, hours, dows, split, features, edge_index

def compute_historical_mean(data, mask, hours, dows, train_nodes):
    """Compute per-(node, hour) historical mean for ATSPM features."""
    N, T, F = data.shape
    hist_mean = torch.zeros(N, 24, N_ATSPM)
    h_int = hours.long()
    for hour in range(24):
        t_mask = (h_int == hour)
        if t_mask.sum() == 0: continue
        for f in range(N_ATSPM):
            obs = mask[train_nodes][:, t_mask, f]
            vals = data[train_nodes][:, t_mask, f]
            sums = (vals * obs.float()).sum(dim=1)
            counts = obs.float().sum(dim=1).clamp(min=1)
            hist_mean[train_nodes, hour, f] = sums / counts
    global_mean = hist_mean[train_nodes].mean(dim=0)
    heldout = set(range(N)) - set(train_nodes)
    for node in heldout:
        hist_mean[node] = global_mean
    return hist_mean

def get_hist_for_window(hist_mean, node_ids, hours_window):
    """Get historical mean for specific nodes and timesteps. Returns (S, W, N_ATSPM)."""
    h_idx = hours_window.long().clamp(0, 23)
    return hist_mean[node_ids][:, h_idx, :]

def build_sub_adj_sparse(edge_index, node_ids, device):
    id_map = {gid: i for i, gid in enumerate(node_ids.tolist())}
    S = len(node_ids)
    rows, cols = [], []
    src, tgt = edge_index
    for e in range(src.shape[0]):
        s, t = src[e].item(), tgt[e].item()
        if s in id_map and t in id_map:
            rows.append(id_map[t]); cols.append(id_map[s])
    if not rows:
        return torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long, device=device),
                                       torch.zeros(0, device=device), (S, S))
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), device=device)
    adj = torch.sparse_coo_tensor(indices, values, (S, S))
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    norm_vals = (1.0 / deg)[indices[0]]
    return torch.sparse_coo_tensor(indices, norm_vals, (S, S))

def build_dense_adj(edge_index, node_ids, device):
    """Build dense adjacency for transformer attention masking."""
    id_map = {gid: i for i, gid in enumerate(node_ids.tolist())}
    S = len(node_ids)
    adj = torch.eye(S, device=device)  # self-connections
    src, tgt = edge_index
    for e in range(src.shape[0]):
        s, t = src[e].item(), tgt[e].item()
        if s in id_map and t in id_map:
            adj[id_map[t], id_map[s]] = 1.0
            adj[id_map[s], id_map[t]] = 1.0
    return adj

def compute_laplacian(edge_index, node_ids, device):
    """Compute normalized Laplacian for ChebConv: L_scaled = 2*L/lambda_max - I."""
    adj = build_dense_adj(edge_index, node_ids, device)
    deg = adj.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / deg.clamp(min=1).sqrt())
    L = torch.eye(len(node_ids), device=device) - D_inv_sqrt @ adj @ D_inv_sqrt
    # Approximate lambda_max ≈ 2 for normalized Laplacian
    L_scaled = L  # already normalized
    return L_scaled.to_sparse()

def evaluate_model(model_fn, data, mask, hours, hist_mean, heldout_nodes,
                   edge_index, t_val, features, device, model_name):
    """Evaluate imputation on held-out nodes (6AM-9PM)."""
    N, T, F_dim = data.shape
    atspm_names = features[:N_ATSPM]
    all_preds, all_true, all_mask_out, all_h = [], [], [], []

    with torch.no_grad():
        for t0 in range(t_val, T, WINDOW // 2):
            t1 = min(t0 + WINDOW, T)
            if t1 - t0 < 4: continue
            actual_len = t1 - t0
            x = data[heldout_nodes, t0:t1, :].to(device)
            m = mask[heldout_nodes, t0:t1, :].to(device)
            h = hours[t0:t1]
            nids = torch.tensor(heldout_nodes, dtype=torch.long)
            hist = get_hist_for_window(hist_mean, heldout_nodes, h).to(device)

            if actual_len < WINDOW:
                pad = WINDOW - actual_len
                x = torch.cat([x, torch.zeros(len(heldout_nodes), pad, F_dim, device=device)], 1)
                m = torch.cat([m, torch.zeros(len(heldout_nodes), pad, F_dim, dtype=torch.bool, device=device)], 1)
                h_pad = torch.cat([h, torch.zeros(pad)])
                hist = get_hist_for_window(hist_mean, heldout_nodes, h_pad).to(device)

            input_m = m.clone(); input_m[:, :, :N_ATSPM] = False
            input_x = x.clone(); input_x[:, :, :N_ATSPM] = 0.0

            pred_residual = model_fn(input_x, input_m, nids, device)
            pred_atspm = hist[:, :actual_len, :] + pred_residual[:, :actual_len, :]

            all_preds.append(pred_atspm.cpu())
            all_true.append(data[heldout_nodes, t0:t1, :N_ATSPM])
            all_mask_out.append(mask[heldout_nodes, t0:t1, :N_ATSPM])
            h_vals = h[:actual_len].unsqueeze(0).expand(len(heldout_nodes), -1)
            all_h.append(h_vals)

    preds = torch.cat(all_preds, dim=1)
    trues = torch.cat(all_true, dim=1)
    masks_out = torch.cat(all_mask_out, dim=1)
    h_out = torch.cat(all_h, dim=1)
    time_mask = (h_out >= 6) & (h_out < 21)

    log(f"\n{'Feature':<28} {'MAE':>8} {'RelErr':>8} {'N':>8}")
    log("-" * 60)
    results = []
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i] & time_mask
        if m_feat.sum() == 0:
            log(f"  {fname:<28} {'N/A':>8}"); continue
        p = preds[:, :, i][m_feat]; t_vals = trues[:, :, i][m_feat]
        mae = (p - t_vals).abs().mean().item()
        gt_mean = t_vals.abs().mean().item()
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        results.append({'feature': fname, 'MAE': mae, 'RelError': rel})
        log(f"  {fname:<28} {mae:>8.2f} {rel:>7.1f}% {int(m_feat.sum()):>8}")
    if results:
        avg_rel = np.mean([r['RelError'] for r in results])
        log(f"\n  {'AVERAGE (6AM-9PM)':<28} {'':>8} {avg_rel:>7.1f}%")
    return results
