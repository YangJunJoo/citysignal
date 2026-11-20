#!/usr/bin/env python3
"""
train_anomaly_gnn_v2.py — Fixed Hybrid Dual-Stream GNN

Key fix from v1: Stream A now receives ALL features (ATSPM + CV raw),
making it equivalent to baseline GRIN. Stream B ADDITIONALLY processes
CV anomaly scores and gates the output.

This tests whether an explicit anomaly pathway ADDS value on top of
the baseline, rather than replacing it.

Architecture:
  Stream A (Base GRIN): ATSPM + CV raw + time → Spatial-Temporal GNN
  Stream B (Anomaly):   CV anomaly scores → Local RNN
  Fusion: final = base_pred + gate * anomaly_correction
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
# ANOMALY PREPROCESSING
# ═══════════════════════════════════════════════
def prepare_anomaly_features(data, mask):
    """
    Compute CV anomaly = Current - RollingMean(last 4 steps).
    Returns tensor of same shape as data, with anomaly scores
    in CV feature positions and zeros elsewhere.
    """
    S, T, F_dim = data.shape
    anomaly = torch.zeros(S, T, N_CV, dtype=data.dtype)

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
    anomaly = anomaly_cv * valid.float()

    return anomaly  # (S, T, N_CV)


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
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True)

    time_df = pd.read_csv(os.path.join(DATA_DIR, 'time_index_v3.csv'), parse_dates=['time'])
    hours = torch.tensor(time_df['time'].dt.hour.values, dtype=torch.float32)

    anomaly = prepare_anomaly_features(data, mask)

    log(f"  data: {data.shape}, mask: {mask.shape}, anomaly: {anomaly.shape}")
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
    log(f"  edge_index: {edge_index.shape}")

    return data, mask, anomaly, hours, split, features, edge_index


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


class HybridDualStreamGRIN(nn.Module):
    """
    Fixed Dual-Stream: Stream A = full GRIN (ATSPM + CV raw),
    Stream B = CV anomaly detector, fused via gating.
    """
    def __init__(self, F_all, F_anom, H):
        """
        F_all: total feature dim (ATSPM + CV = 32)
        F_anom: anomaly feature dim (N_CV = 17)
        H: hidden dim
        """
        super().__init__()
        self.H = H
        self.F_all = F_all
        self.F_anom = F_anom

        # Stream A: Full GRIN (same as baseline)
        # Input: x*m + m = F_all*2
        self.streamA_fwd = nn.GRUCell(F_all * 2, H)
        self.streamA_bwd = nn.GRUCell(F_all * 2, H)
        self.spatialA_fwd = SpatialAgg(H)
        self.spatialA_bwd = SpatialAgg(H)

        # Stream B: CV Anomaly RNN (local, no spatial)
        # Input: anomaly*anom_mask + anom_mask = F_anom*2
        self.streamB_fwd = nn.GRUCell(F_anom * 2, H // 2)
        self.streamB_bwd = nn.GRUCell(F_anom * 2, H // 2)

        # Base output (p_logits, mu)
        self.out_base = nn.Linear(H * 2, N_ATSPM * 2)

        # Anomaly correction (p_logits, mu)
        self.out_correction = nn.Linear((H // 2) * 2, N_ATSPM * 2)

        # Gate: decides how much to trust anomaly correction
        self.gate = nn.Sequential(
            nn.Linear(H * 2 + (H // 2) * 2, H),
            nn.ReLU(),
            nn.Linear(H, N_ATSPM * 2),
            nn.Sigmoid()
        )

    def forward(self, x, m, anomaly, anom_mask, adj):
        """
        x: (S, W, F_all) — raw features
        m: (S, W, F_all) — mask for raw features
        anomaly: (S, W, F_anom) — CV anomaly scores
        anom_mask: (S, W, F_anom) — mask for anomaly (valid where CV observed)
        adj: sparse adjacency
        """
        S, W, F_dim = x.shape
        device = x.device

        # Stream A: full feature GRIN (identical to baseline)
        hA_f = torch.zeros(S, self.H, device=device)
        fwd_A = []
        for t in range(W):
            inp_A = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hA_f = self.streamA_fwd(inp_A, hA_f)
            hA_f = self.spatialA_fwd(hA_f, adj)
            fwd_A.append(hA_f)

        hA_b = torch.zeros(S, self.H, device=device)
        bwd_A = [None] * W
        for t in range(W - 1, -1, -1):
            inp_A = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hA_b = self.streamA_bwd(inp_A, hA_b)
            hA_b = self.spatialA_bwd(hA_b, adj)
            bwd_A[t] = hA_b

        # Stream B: anomaly RNN (local only)
        hB_f = torch.zeros(S, self.H // 2, device=device)
        fwd_B = []
        for t in range(W):
            inp_B = torch.cat([anomaly[:, t] * anom_mask[:, t].float(),
                              anom_mask[:, t].float()], -1)
            hB_f = self.streamB_fwd(inp_B, hB_f)
            fwd_B.append(hB_f)

        hB_b = torch.zeros(S, self.H // 2, device=device)
        bwd_B = [None] * W
        for t in range(W - 1, -1, -1):
            inp_B = torch.cat([anomaly[:, t] * anom_mask[:, t].float(),
                              anom_mask[:, t].float()], -1)
            hB_b = self.streamB_bwd(inp_B, hB_b)
            bwd_B[t] = hB_b

        # Fusion
        outputs = []
        for t in range(W):
            cat_A = torch.cat([fwd_A[t], bwd_A[t]], dim=-1)  # (S, 2H)
            cat_B = torch.cat([fwd_B[t], bwd_B[t]], dim=-1)  # (S, H)

            base_pred = self.out_base(cat_A)
            correction = self.out_correction(cat_B)

            gate_val = self.gate(torch.cat([cat_A, cat_B], dim=-1))

            # Hard gate: 1 if any CV anomaly data is observed at this timestep for this node, else 0
            # anom_mask[:, t] is (S, F_anom)
            hard_gate = (anom_mask[:, t].sum(dim=-1, keepdim=True) > 0).float()

            # Final = Base + (Gate * HardGate) * Correction
            final = base_pred + (gate_val * hard_gate) * correction
            outputs.append(final)

        return torch.stack(outputs, 1)  # (S, W, N_ATSPM * 2)


# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════
def hurdle_loss(pred, target, mask):
    p_logits = pred[:, :, :N_ATSPM]
    mu = pred[:, :, N_ATSPM:]
    
    y_bin = (target > 1e-4).float()
    
    bce = F.binary_cross_entropy_with_logits(p_logits[mask], y_bin[mask], reduction='none')
    
    pos_mask = mask & (y_bin > 0.5)
    if pos_mask.sum() > 0:
        l1 = F.l1_loss(mu[pos_mask], target[pos_mask], reduction='none')
        return bce.mean() + l1.mean()
    else:
        return bce.mean()

def train_and_evaluate(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, anomaly, hours, split, features, edge_index = load_all()
    N, T, F_dim = data.shape

    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']
    t_val = split['time_val_end']

    log(f"Data: ({N}, {T}, {F_dim}), train={len(train_nodes)}, held-out={len(heldout_nodes)}")

    train_ds = ImputationDataset(data, mask, anomaly, hours, train_nodes, 0, t_train, WINDOW)
    val_ds = ImputationDataset(data, mask, anomaly, hours, train_nodes, t_train, t_val, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = HybridDualStreamGRIN(F_dim, N_CV, HIDDEN).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"Model: HybridDualStreamGRIN, params: {nparam:,}")

    # Also train baseline GRIN for comparison
    from train_obs_aware_gnn import BaselineGRIN
    baseline = BaselineGRIN(F_dim, HIDDEN).to(device)
    nparam_b = sum(p.numel() for p in baseline.parameters())
    log(f"Baseline: BaselineGRIN, params: {nparam_b:,}")

    for model_name, model_obj in [('hybrid_dualstream_zignn', model)]:
        log(f"\n{'='*60}")
        log(f"Training {model_name.upper()} for {EPOCHS} epochs (6AM-9PM loss)...")
        log(f"{'='*60}")

        opt = optim.Adam(model_obj.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_val = float('inf')

        res_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(res_dir, exist_ok=True)

        is_hybrid = 'hybrid' in model_name

        for epoch in range(EPOCHS):
            t0 = time.time()
            model_obj.train()
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

                    # Extra masking for self-supervision (Node-Level Masking)
                    atspm_m = m[:, :, :N_ATSPM].clone()
                    S, W, _ = atspm_m.shape
                    node_drop = torch.rand(S, device=device) < MASK_EXTRA
                    drop = node_drop.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
                    
                    train_m = m.clone()
                    train_m[:, :, :N_ATSPM] = atspm_m & ~drop
                    eval_m = m.clone()
                    eval_m[:, :, :N_ATSPM] = atspm_m & drop

                    # 6AM-9PM time filter
                    time_mask = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                    eval_m_filtered = eval_m.clone()
                    eval_m_filtered[:, :, :N_ATSPM] = eval_m[:, :, :N_ATSPM] & time_mask

                    adj = build_sub_adj_sparse(edge_index, nids, device)

                    if is_hybrid:
                        anom_mask = m[:, :, N_ATSPM:N_ATSPM+N_CV]
                        pred = model_obj(x, train_m, a, anom_mask, adj)
                    else:
                        pred = model_obj(x, train_m, adj)
                        pred = pred[:, :, :N_ATSPM]  # baseline outputs F_dim, slice to ATSPM

                    em_atspm = eval_m_filtered[:, :, :N_ATSPM]
                    if em_atspm.sum() > 0:
                        loss = hurdle_loss(pred, x[:, :, :N_ATSPM], em_atspm)
                    elif eval_m[:, :, :N_ATSPM].sum() > 0:
                        em2 = eval_m[:, :, :N_ATSPM]
                        loss = hurdle_loss(pred, x[:, :, :N_ATSPM], em2)
                    else:
                        continue

                    total_loss += loss

                if total_loss > 0:
                    opt.zero_grad()
                    (total_loss / B).backward()
                    torch.nn.utils.clip_grad_norm_(model_obj.parameters(), 5.0)
                    opt.step()
                    ep_loss += total_loss.item() / B
                    n_batch += 1

            scheduler.step()

            # Validation
            model_obj.eval()
            val_loss, v_cnt = 0, 0
            with torch.no_grad():
                for bx, bm, ba, bh, bn in val_loader:
                    for b in range(bx.shape[0]):
                        x = bx[b].to(device)
                        m = bm[b].to(device)
                        a = ba[b].to(device)
                        h = bh[b]
                        nids = bn[b]

                        S, W, _ = m[:, :, :N_ATSPM].shape
                        node_drop = torch.rand(S, device=device) < MASK_EXTRA
                        drop = node_drop.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
                        
                        tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~drop
                        em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & drop

                        time_mask = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                        em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & time_mask

                        adj = build_sub_adj_sparse(edge_index, nids, device)

                        if is_hybrid:
                            anom_mask = m[:, :, N_ATSPM:N_ATSPM+N_CV]
                            pred = model_obj(x, tm, a, anom_mask, adj)
                        else:
                            pred = model_obj(x, tm, adj)
                            pred = pred[:, :, :N_ATSPM]

                        em_atspm = em[:, :, :N_ATSPM]
                        if em_atspm.sum() > 0:
                            val_loss += hurdle_loss(pred, x[:, :, :N_ATSPM], em_atspm).item()
                            v_cnt += 1

            at = ep_loss / max(n_batch, 1)
            av = val_loss / max(v_cnt, 1)
            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                log(f"  [{model_name}] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {elapsed:.1f}s")

            if av < best_val:
                best_val = av
                torch.save(model_obj.state_dict(), os.path.join(res_dir, 'best_model.pt'))

        log(f"\n{model_name.upper()} DONE! Best Val: {best_val:.4f}")

        # ═══════════════════════════════════════════════
        # EVALUATION on held-out test nodes
        # ═══════════════════════════════════════════════
        log(f"\nEvaluating {model_name} on held-out nodes (6AM-9PM)...")

        model_obj.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'),
                                             map_location=device, weights_only=True))
        model_obj.eval()

        atspm_names = features[:N_ATSPM]
        all_preds, all_true, all_mask_out, all_h = [], [], [], []

        with torch.no_grad():
            for t0_eval in range(t_val, T, WINDOW // 2):
                t1_eval = min(t0_eval + WINDOW, T)
                if t1_eval - t0_eval < 4: continue
                actual_len = t1_eval - t0_eval

                x = data[heldout_nodes, t0_eval:t1_eval, :].to(device)
                m = mask[heldout_nodes, t0_eval:t1_eval, :].to(device)
                a = anomaly[heldout_nodes, t0_eval:t1_eval, :].to(device)
                h = hours[t0_eval:t1_eval]
                nids = torch.tensor(heldout_nodes, dtype=torch.long)

                if actual_len < WINDOW:
                    pad = WINDOW - actual_len
                    x = torch.cat([x, torch.zeros(len(heldout_nodes), pad, F_dim, device=device)], 1)
                    m = torch.cat([m, torch.zeros(len(heldout_nodes), pad, F_dim, dtype=torch.bool, device=device)], 1)
                    a = torch.cat([a, torch.zeros(len(heldout_nodes), pad, N_CV, device=device)], 1)
                    h = torch.cat([h, torch.zeros(pad)])

                # Zero out ATSPM for held-out
                input_m = m.clone()
                input_m[:, :, :N_ATSPM] = False
                input_x = x.clone()
                input_x[:, :, :N_ATSPM] = 0.0

                adj = build_sub_adj_sparse(edge_index, nids, device)

                if is_hybrid:
                    anom_mask = m[:, :, N_ATSPM:N_ATSPM+N_CV]
                    pred = model_obj(input_x, input_m, a, anom_mask, adj)
                    p_logits = pred[:, :actual_len, :N_ATSPM]
                    mu = pred[:, :actual_len, N_ATSPM:]
                    pred_final = torch.sigmoid(p_logits) * mu
                else:
                    pred = model_obj(input_x, input_m, adj)
                    pred_final = pred[:, :actual_len, :N_ATSPM]

                all_preds.append(pred_final.cpu())
                all_true.append(data[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
                all_mask_out.append(mask[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
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
                log(f"  {fname:<28} {'N/A':>8}")
                continue

            p = preds[:, :, i][m_feat]
            t_vals = trues[:, :, i][m_feat]
            mae = (p - t_vals).abs().mean().item()
            gt_mean = t_vals.abs().mean().item()
            rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            results.append({'feature': fname, 'MAE': mae, 'RelError': rel})
            log(f"  {fname:<28} {mae:>8.2f} {rel:>7.1f}% {int(m_feat.sum()):>8}")

        if results:
            avg_rel = np.mean([r['RelError'] for r in results])
            log(f"\n  {'AVERAGE (6AM-9PM)':<28} {'':>8} {avg_rel:>7.1f}%")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    train_and_evaluate(args)
