#!/usr/bin/env python3
"""
train_robust_v1.py — Dual-Stream GNN with Target Log-Transformation (log1p/expm1)

This script implements Phase 1 of our optimization pipeline:
- Log-transforms ATSPM target and input features during training to stabilize gradients.
- Reverts predictions to raw scale during evaluation to compare MAE/RelErr accurately.
"""

import os, json, argparse, time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

WINDOW = 24
NODE_SAMPLE = 200
BATCH = 4
EPOCHS = 80
LR = 0.001
HIDDEN = 64
MASK_EXTRA = 0.10
N_ATSPM = 15
N_CV = 17

def log(msg):
    print(msg, flush=True)

# ═══════════════════════════════════════════════
# DATA & ANOMALY PREPROCESSING
# ═══════════════════════════════════════════════
def prepare_anomaly_data(data, mask):
    """
    Compute CV anomaly scores:
    Anomaly = Current - Moving Average (last 4 timesteps / 1 hr)
    """
    S, T, F_dim = data.shape
    anomaly = torch.zeros_like(data)
    
    cv_data = data[:, :, N_ATSPM:N_ATSPM+N_CV]
    cv_mask = mask[:, :, N_ATSPM:N_ATSPM+N_CV].float()
    
    cv_flat = (cv_data * cv_mask).permute(0, 2, 1).reshape(S * N_CV, 1, T)
    m_flat = cv_mask.permute(0, 2, 1).reshape(S * N_CV, 1, T)
    
    kernel = torch.ones(1, 1, 4)
    cv_pad = F.pad(cv_flat, (3, 0))
    m_pad = F.pad(m_flat, (3, 0))
    
    roll_sum = F.conv1d(cv_pad, kernel)
    roll_cnt = F.conv1d(m_pad, kernel)
    
    roll_mean = roll_sum / roll_cnt.clamp(min=1)
    roll_mean = roll_mean.reshape(S, N_CV, T).permute(0, 2, 1)
    
    anomaly_cv = cv_data - roll_mean
    valid = (cv_mask > 0) & (roll_cnt.reshape(S, N_CV, T).permute(0, 2, 1) > 0)
    anomaly_cv = anomaly_cv * valid.float()
    
    anomaly[:, :, N_ATSPM:N_ATSPM+N_CV] = anomaly_cv
    return anomaly


class ImputationDataset(Dataset):
    def __init__(self, data, mask, anomaly, hours, node_ids, t_start, t_end, window):
        self.data = data
        self.mask = mask
        self.anomaly = anomaly
        self.hours = hours
        self.nodes = node_ids
        self.t_start = t_start
        self.t_end = t_end
        self.window = window
        self.n_windows = max(1, (t_end - t_start - window) // (window // 2) + 1)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        t0 = self.t_start + idx * (self.window // 2)
        t1 = min(t0 + self.window, self.t_end)
        perm = torch.randperm(len(self.nodes))[:NODE_SAMPLE]
        nids = [self.nodes[i] for i in perm]
        
        x = self.data[nids, t0:t1, :]
        m = self.mask[nids, t0:t1, :]
        a = self.anomaly[nids, t0:t1, :]
        h = self.hours[t0:t1]
        
        return x, m, a, h, torch.tensor(nids, dtype=torch.long)


def load_all():
    log("Loading tensors...")
    data_raw = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True)
    
    time_df = pd.read_csv(os.path.join(DATA_DIR, 'time_index_v3.csv'), parse_dates=['time'])
    hours = torch.tensor(time_df['time'].dt.hour.values, dtype=torch.float32)
    
    # Pre-compute anomaly on raw CV data (before log1p is applied to ATSPM)
    anomaly = prepare_anomaly_data(data_raw, mask)
    
    # Log-transform ATSPM variables in data (clamp to min=0.0 to prevent NaNs)
    data_log = data_raw.clone()
    data_log[:, :, :N_ATSPM] = torch.log1p(data_log[:, :, :N_ATSPM].clamp(min=0.0))
    
    log(f"  data raw: {data_raw.shape}, data log: {data_log.shape}, mask: {mask.shape}")
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)
        
    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    
    src_list, tgt_list = [], []
    for s, t in zip(edges_df['source'], edges_df['target']):
        s_str, t_str = str(int(s)), str(int(t))
        if s_str in seg2id and t_str in seg2id:
            src_list.append(seg2id[s_str])
            tgt_list.append(seg2id[t_str])
            
    edge_index = torch.tensor(np.stack([src_list, tgt_list]), dtype=torch.long)
    
    return data_raw, data_log, mask, anomaly, hours, split, features, edge_index

def build_sub_adj_sparse(edge_index, node_ids, device):
    id_map = {gid: i for i, gid in enumerate(node_ids.tolist())}
    S = len(node_ids)
    rows, cols = [], []
    src, tgt = edge_index
    for e in range(src.shape[0]):
        s, t = src[e].item(), tgt[e].item()
        if s in id_map and t in id_map:
            rows.append(id_map[t])
            cols.append(id_map[s])
    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, device=device), (S, S))
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), device=device)
    adj = torch.sparse_coo_tensor(indices, values, (S, S))
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    norm_vals = (1.0 / deg)[indices[0]]
    return torch.sparse_coo_tensor(indices, norm_vals, (S, S))

# ═══════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════
class SpatialAgg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, h, adj):
        neigh = torch.sparse.mm(adj, h)
        return h + torch.relu(self.lin(neigh))

class DualStreamGRIN(nn.Module):
    def __init__(self, F_atspm, F_cv, H):
        super().__init__()
        self.H = H
        self.F_atspm = F_atspm
        self.F_cv = F_cv
        
        self.streamA_fwd = nn.GRUCell(F_atspm * 2 + 1, H)
        self.streamA_bwd = nn.GRUCell(F_atspm * 2 + 1, H)
        self.spatialA_fwd = SpatialAgg(H)
        self.spatialA_bwd = SpatialAgg(H)
        
        self.streamB_fwd = nn.GRUCell(F_cv * 2, H // 2)
        self.streamB_bwd = nn.GRUCell(F_cv * 2, H // 2)
        
        self.out_base = nn.Linear(H * 2, F_atspm)
        self.out_anomaly = nn.Linear((H // 2) * 2, F_atspm)
        
        self.gate = nn.Sequential(
            nn.Linear(H * 2 + (H // 2) * 2, H),
            nn.ReLU(),
            nn.Linear(H, F_atspm),
            nn.Sigmoid()
        )

    def forward(self, x, m, anomaly, hours, adj):
        S, W, F_dim = x.shape
        device = x.device
        
        x_atspm, m_atspm = x[:, :, :self.F_atspm], m[:, :, :self.F_atspm].float()
        a_cv, m_cv = anomaly[:, :, self.F_atspm:self.F_atspm+self.F_cv], m[:, :, self.F_atspm:self.F_atspm+self.F_cv].float()
        h_hour = hours.unsqueeze(0).unsqueeze(-1).expand(S, -1, 1).to(device)
        
        hA_f = torch.zeros(S, self.H, device=device)
        fwd_A = []
        for t in range(W):
            inp_A = torch.cat([x_atspm[:, t] * m_atspm[:, t], m_atspm[:, t], h_hour[:, t]], -1)
            hA_f = self.streamA_fwd(inp_A, hA_f)
            hA_f = self.spatialA_fwd(hA_f, adj)
            fwd_A.append(hA_f)
            
        hB_f = torch.zeros(S, self.H // 2, device=device)
        fwd_B = []
        for t in range(W):
            inp_B = torch.cat([a_cv[:, t] * m_cv[:, t], m_cv[:, t]], -1)
            hB_f = self.streamB_fwd(inp_B, hB_f)
            fwd_B.append(hB_f)
            
        hA_b = torch.zeros(S, self.H, device=device)
        bwd_A = [None] * W
        hB_b = torch.zeros(S, self.H // 2, device=device)
        bwd_B = [None] * W
        
        for t in range(W - 1, -1, -1):
            inp_A = torch.cat([x_atspm[:, t] * m_atspm[:, t], m_atspm[:, t], h_hour[:, t]], -1)
            hA_b = self.streamA_bwd(inp_A, hA_b)
            hA_b = self.spatialA_bwd(hA_b, adj)
            bwd_A[t] = hA_b
            
            inp_B = torch.cat([a_cv[:, t] * m_cv[:, t], m_cv[:, t]], -1)
            hB_b = self.streamB_bwd(inp_B, hB_b)
            bwd_B[t] = hB_b
            
        outputs = []
        for t in range(W):
            cat_A = torch.cat([fwd_A[t], bwd_A[t]], dim=-1)
            cat_B = torch.cat([fwd_B[t], bwd_B[t]], dim=-1)
            
            base_pred = self.out_base(cat_A)
            anomaly_pred = self.out_anomaly(cat_B)
            
            gate_val = self.gate(torch.cat([cat_A, cat_B], dim=-1))
            final_pred = base_pred + gate_val * anomaly_pred
            outputs.append(final_pred)
            
        return torch.stack(outputs, 1)

# ═══════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════
def train_and_evaluate(args):
    device = torch.device(f'cuda:{args.gpu}')
    data_raw, data_log, mask, anomaly, hours, split, features, edge_index = load_all()
    N, T, F_dim = data_log.shape

    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']
    t_val = split['time_val_end']

    train_ds = ImputationDataset(data_log, mask, anomaly, hours, train_nodes, 0, t_train, WINDOW)
    val_ds = ImputationDataset(data_log, mask, anomaly, hours, train_nodes, t_train, t_val, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = DualStreamGRIN(N_ATSPM, N_CV, HIDDEN).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"Model: DualStreamGRIN (Log-scale), params: {nparam:,}")

    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.L1Loss(reduction='none')

    res_dir = os.path.join(RESULTS_DIR, 'robust_v1_log')
    os.makedirs(res_dir, exist_ok=True)
    best_val = float('inf')

    log(f"\nTraining DUALSTREAM_GRIN (Log-scale) for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        ep_loss, n_batch = 0, 0

        for batch_x, batch_m, batch_a, batch_h, batch_nids in train_loader:
            B = batch_x.shape[0]
            total_loss = 0
            for b in range(B):
                x = batch_x[b].to(device)
                m = batch_m[b].to(device)
                a = batch_a[b].to(device)
                h = batch_h[b]
                nids = batch_nids[b]

                atspm_m = m[:, :, :N_ATSPM].clone()
                drop = torch.rand_like(atspm_m.float()) < MASK_EXTRA
                train_m = m.clone()
                train_m[:, :, :N_ATSPM] = atspm_m & ~drop
                eval_m = m.clone()
                eval_m[:, :, :N_ATSPM] = atspm_m & drop
                
                time_mask = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                eval_m[:, :, :N_ATSPM] = eval_m[:, :, :N_ATSPM] & time_mask

                adj = build_sub_adj_sparse(edge_index, nids, device)
                pred = model(x, train_m, a, h, adj)

                if eval_m[:, :, :N_ATSPM].sum() > 0:
                    loss = criterion(pred[eval_m[:, :, :N_ATSPM]], x[:, :, :N_ATSPM][eval_m[:, :, :N_ATSPM]]).mean()
                else:
                    train_op_m = train_m[:, :, :N_ATSPM] & time_mask
                    if train_op_m.sum() > 0:
                        loss = criterion(pred[train_op_m], x[:, :, :N_ATSPM][train_op_m]).mean()
                    else:
                        continue
                
                total_loss += loss

            if total_loss > 0:
                opt.zero_grad()
                (total_loss / B).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                ep_loss += total_loss.item() / B
                n_batch += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss, v_cnt = 0, 0
        with torch.no_grad():
            for bx, bm, ba, bh, bn in val_loader:
                for b in range(bx.shape[0]):
                    x = bx[b].to(device)
                    m = bm[b].to(device)
                    a = ba[b].to(device)
                    h = bh[b]
                    nids = bn[b]
                    
                    drop = torch.rand_like(m[:, :, :N_ATSPM].float()) < MASK_EXTRA
                    tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~drop
                    em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & drop
                    
                    time_mask = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                    em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & time_mask

                    adj = build_sub_adj_sparse(edge_index, nids, device)
                    pred = model(x, tm, a, h, adj)

                    if em[:, :, :N_ATSPM].sum() > 0:
                        val_loss += criterion(pred[em[:, :, :N_ATSPM]], x[:, :, :N_ATSPM][em[:, :, :N_ATSPM]]).mean().item()
                        v_cnt += 1

        at = ep_loss / max(n_batch, 1)
        av = val_loss / max(v_cnt, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            log(f"  [E{epoch:03d}] T:{at:.4f} V:{av:.4f} | {elapsed:.1f}s")

        if av < best_val:
            best_val = av
            torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))

    log(f"\nTraining DONE! Best Val (Log-scale): {best_val:.4f}")

    # ═══════════════════════════════════════════════
    # EVALUATION (Raw scale MAE/RelErr)
    # ═══════════════════════════════════════════════
    log(f"\n{'='*60}")
    log(f"Evaluating on held-out nodes (6AM-9PM, RAW scale)...")
    log(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'),
                                     map_location=device, weights_only=True))
    model.eval()

    atspm_names = features[:N_ATSPM]
    all_preds, all_true, all_mask_out, all_h = [], [], [], []

    with torch.no_grad():
        for t0_eval in range(t_val, T, WINDOW // 2):
            t1_eval = min(t0_eval + WINDOW, T)
            if t1_eval - t0_eval < 4: continue
            actual_len = t1_eval - t0_eval

            # Inputs must be in log-scale
            x_log = data_log[heldout_nodes, t0_eval:t1_eval, :].to(device)
            m = mask[heldout_nodes, t0_eval:t1_eval, :].to(device)
            a = anomaly[heldout_nodes, t0_eval:t1_eval, :].to(device)
            h = hours[t0_eval:t1_eval]
            nids = torch.tensor(heldout_nodes, dtype=torch.long)

            if actual_len < WINDOW:
                pad = WINDOW - actual_len
                x_log = torch.cat([x_log, torch.zeros(len(heldout_nodes), pad, F_dim, device=device)], dim=1)
                m = torch.cat([m, torch.zeros(len(heldout_nodes), pad, F_dim, dtype=torch.bool, device=device)], dim=1)
                a = torch.cat([a, torch.zeros(len(heldout_nodes), pad, F_dim, device=device)], dim=1)
                h = torch.cat([h, torch.zeros(pad)])

            input_m = m.clone()
            input_m[:, :, :N_ATSPM] = False
            input_x = x_log.clone()
            input_x[:, :, :N_ATSPM] = 0.0

            adj = build_sub_adj_sparse(edge_index, nids, device)
            pred_log = model(input_x, input_m, a, h, adj)
            
            # Convert prediction to raw scale using expm1
            pred_raw = torch.expm1(pred_log)

            all_preds.append(pred_raw[:, :actual_len, :].cpu())
            # Ground truth must be raw data
            all_true.append(data_raw[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            all_mask_out.append(mask[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            h_vals = h[:actual_len].unsqueeze(0).expand(len(heldout_nodes), -1)
            all_h.append(h_vals)

    preds = torch.cat(all_preds, dim=1)
    trues = torch.cat(all_true, dim=1)
    masks_out = torch.cat(all_mask_out, dim=1)
    h_out = torch.cat(all_h, dim=1)

    results = []
    log(f"\n{'Feature':<28} {'MAE':>8} {'RMSE':>8} {'RelErr':>8} {'N':>8}")
    log("-" * 70)
    
    time_mask = (h_out >= 6) & (h_out < 21)
    
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i] & time_mask
        if m_feat.sum() == 0:
            results.append({'feature': fname, 'MAE': None, 'RMSE': None, 'RelError': None, 'N': 0})
            log(f"  {fname:<28} {'N/A':>8}")
            continue
            
        p = preds[:, :, i][m_feat]
        t = trues[:, :, i][m_feat]
        mae = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        gt_mean = t.abs().mean().item()
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        results.append({'feature': fname, 'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
                        'RelError': round(rel, 1), 'GT_mean': round(gt_mean, 2),
                        'N': int(m_feat.sum().item())})
        log(f"  {fname:<28} {mae:>8.2f} {rmse:>8.2f} {rel:>7.1f}% {int(m_feat.sum()):>8}")

    avg_mae = np.mean([r['MAE'] for r in results if r['MAE'] is not None])
    avg_rel = np.mean([r['RelError'] for r in results if r['RelError'] is not None])
    log(f"\n  {'AVERAGE (6AM-9PM)':<28} {avg_mae:>8.2f} {'':>8} {avg_rel:>7.1f}%")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    train_and_evaluate(args)
