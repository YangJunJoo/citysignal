#!/usr/bin/env python3
"""
train_obs_aware_gnn.py — Observation-Aware Spatial Gating GNN

Key innovation: Replace uniform row-normalized message passing with
observation-quality-aware attention. Neighbors with observed ATSPM
get higher attention weight than those with only CV data.

Models:
  1. Baseline GRIN (from train_all_v2.py, retrained for comparison)
  2. ObsAware-GRIN: GRIN with ObsAwareSpatialAgg replacing SpatialAgg

Obs Quality features per (node, timestep):
  - atspm_obs_rate: fraction of 15 ATSPM features observed
  - cv_obs_rate: fraction of 17 CV features observed
  - cv_journey_count: raw journey count (proxy for data quality)
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
OBS_DIM = 3  # observation quality feature dimension

def log(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════
class ImputationDataset(Dataset):
    def __init__(self, data, mask, node_ids, t_start, t_end, window):
        self.data = data
        self.mask = mask
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
        return x, m, torch.tensor(nids, dtype=torch.long)


def load_all():
    log("Loading tensors...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True)
    log(f"  data: {data.shape}, mask: {mask.shape}")
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)
    edges_df = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping_v2.csv'))
    seg2id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    v = edges_df[edges_df['source'].astype(str).isin(seg2id) &
                 edges_df['target'].astype(str).isin(seg2id)]
    src = np.array([seg2id[str(s)] for s in v['source']])
    tgt = np.array([seg2id[str(s)] for s in v['target']])
    edge_index = torch.tensor(np.stack([src, tgt]), dtype=torch.long)
    log(f"  edge_index: {edge_index.shape}")
    return data, mask, split, features, edge_index


# ═══════════════════════════════════════════════
# OBSERVATION QUALITY COMPUTATION
# ═══════════════════════════════════════════════
def compute_obs_quality(mask_t, n_atspm=N_ATSPM, n_cv=N_CV):
    """
    Compute observation quality for each node at a single timestep.
    mask_t: (S, F) boolean mask
    Returns: (S, OBS_DIM) — [atspm_rate, cv_rate, journey_obs]
    """
    atspm_rate = mask_t[:, :n_atspm].float().mean(dim=1, keepdim=True)
    cv_rate = mask_t[:, n_atspm:n_atspm + n_cv].float().mean(dim=1, keepdim=True)
    # Use last CV feature (segment_journeys) observation as proxy
    journey_obs = mask_t[:, n_atspm + n_cv - 1:n_atspm + n_cv].float()
    return torch.cat([atspm_rate, cv_rate, journey_obs], dim=1)  # (S, 3)


# ═══════════════════════════════════════════════
# ADJACENCY BUILDERS
# ═══════════════════════════════════════════════
def build_sub_adj_sparse(edge_index, node_ids, device):
    """Build sparse normalized adjacency (for baseline GRIN)."""
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


def build_sub_adj_dense(edge_index, node_ids, device):
    """Build dense boolean adjacency mask (for obs-aware attention)."""
    id_map = {gid: i for i, gid in enumerate(node_ids.tolist())}
    S = len(node_ids)
    adj = torch.zeros(S, S, dtype=torch.bool, device=device)
    src, tgt = edge_index
    for e in range(src.shape[0]):
        s, t = src[e].item(), tgt[e].item()
        if s in id_map and t in id_map:
            adj[id_map[t], id_map[s]] = True
    # Add self-loops
    for i in range(S):
        adj[i, i] = True
    return adj


# ═══════════════════════════════════════════════
# SPATIAL AGGREGATION MODULES
# ═══════════════════════════════════════════════
class SpatialAgg(nn.Module):
    """Baseline: uniform row-normalized message passing."""
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h, adj, obs_quality=None):
        neigh = torch.sparse.mm(adj, h)
        return h + torch.relu(self.lin(neigh))


class ObsAwareSpatialAgg(nn.Module):
    """
    Observation-aware spatial attention.
    Uses node hidden states + observation quality to compute attention weights.
    Only attends to graph neighbors (masked by adj_mask).
    """
    def __init__(self, dim, obs_dim=OBS_DIM, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        # Observation quality → per-head attention bias
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, n_heads),
        )

        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** 0.5

    def forward(self, h, adj_mask, obs_quality):
        """
        h: (S, D) node hidden states
        adj_mask: (S, S) dense boolean adjacency (True = neighbor or self)
        obs_quality: (S, OBS_DIM) per-node observation quality
        """
        S, D = h.shape

        Q = self.W_q(h).view(S, self.n_heads, self.head_dim)  # (S, H, d)
        K = self.W_k(h).view(S, self.n_heads, self.head_dim)
        V = self.W_v(h).view(S, self.n_heads, self.head_dim)

        # (S, H, d) @ (S, d, H) via einsum → (H, S, S)
        attn = torch.einsum('ihd,jhd->hij', Q, K) / self.scale  # (H, S, S)

        # Observation quality bias: source node obs quality → bias
        obs_bias = self.obs_proj(obs_quality)  # (S, n_heads)
        # Broadcast: for each target i, add source j's obs bias
        attn = attn + obs_bias.T.unsqueeze(2)  # (H, S, 1) + ... → (H, S, S)

        # Mask non-neighbors
        neighbor_mask = adj_mask.unsqueeze(0).expand(self.n_heads, -1, -1)
        attn = attn.masked_fill(~neighbor_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        # Aggregate
        out = torch.einsum('hij,jhd->ihd', attn, V)  # (S, H, d)
        out = out.reshape(S, D)
        out = self.out_proj(out)

        return h + F.gelu(out)


# ═══════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════
class BaselineGRIN(nn.Module):
    """Baseline GRIN with uniform SpatialAgg (for fair comparison)."""
    def __init__(self, F, H):
        super().__init__()
        self.H = H
        self.gru_fwd = nn.GRUCell(F * 2, H)
        self.gru_bwd = nn.GRUCell(F * 2, H)
        self.spatial = SpatialAgg(H)
        self.out = nn.Linear(H * 2, F)

    def forward(self, x, m, adj, obs_quality=None):
        S, W, F_dim = x.shape
        device = x.device
        h_f = torch.zeros(S, self.H, device=device)
        fwd = []
        for t in range(W):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            h_f = self.gru_fwd(inp, h_f)
            h_f = self.spatial(h_f, adj)
            fwd.append(h_f)
        h_b = torch.zeros(S, self.H, device=device)
        bwd = [None] * W
        for t in range(W - 1, -1, -1):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            h_b = self.gru_bwd(inp, h_b)
            h_b = self.spatial(h_b, adj)
            bwd[t] = h_b
        return torch.stack([self.out(torch.cat([fwd[t], bwd[t]], -1)) for t in range(W)], 1)


class ObsAwareGRIN(nn.Module):
    """GRIN with Observation-Aware Spatial Attention."""
    def __init__(self, F, H, obs_dim=OBS_DIM):
        super().__init__()
        self.H = H
        self.gru_fwd = nn.GRUCell(F * 2, H)
        self.gru_bwd = nn.GRUCell(F * 2, H)
        self.spatial_fwd = ObsAwareSpatialAgg(H, obs_dim)
        self.spatial_bwd = ObsAwareSpatialAgg(H, obs_dim)
        self.out = nn.Linear(H * 2, F)

    def forward(self, x, m, adj_mask, obs_quality):
        """
        obs_quality: (S, W, OBS_DIM) — per node per timestep
        adj_mask: (S, S) dense boolean
        """
        S, W, F_dim = x.shape
        device = x.device
        h_f = torch.zeros(S, self.H, device=device)
        fwd = []
        for t in range(W):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            h_f = self.gru_fwd(inp, h_f)
            h_f = self.spatial_fwd(h_f, adj_mask, obs_quality[:, t])
            fwd.append(h_f)
        h_b = torch.zeros(S, self.H, device=device)
        bwd = [None] * W
        for t in range(W - 1, -1, -1):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            h_b = self.gru_bwd(inp, h_b)
            h_b = self.spatial_bwd(h_b, adj_mask, obs_quality[:, t])
            bwd[t] = h_b
        return torch.stack([self.out(torch.cat([fwd[t], bwd[t]], -1)) for t in range(W)], 1)


# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════
def train_and_evaluate(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, split, features, edge_index = load_all()
    N, T, F_dim = data.shape

    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']
    t_val = split['time_val_end']

    log(f"Data: ({N}, {T}, {F_dim}), train={len(train_nodes)}, held-out={len(heldout_nodes)}")
    log(f"Time: train=0-{t_train}, val={t_train}-{t_val}, test={t_val}-{T}")

    train_ds = ImputationDataset(data, mask, train_nodes, 0, t_train, WINDOW)
    val_ds = ImputationDataset(data, mask, train_nodes, t_train, t_val, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    is_obsaware = args.model == 'obsaware_grin'

    if is_obsaware:
        model = ObsAwareGRIN(F_dim, HIDDEN).to(device)
    else:
        model = BaselineGRIN(F_dim, HIDDEN).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.model}, params: {nparam:,}")

    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.L1Loss()

    res_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(res_dir, exist_ok=True)
    best_val = float('inf')

    log(f"\nTraining {args.model.upper()} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        ep_loss, n_batch = 0, 0

        for batch_x, batch_m, batch_nids in train_loader:
            B = batch_x.shape[0]
            total_loss = 0
            for b in range(B):
                x = batch_x[b].to(device)
                m = batch_m[b].to(device)
                nids = batch_nids[b]

                # Extra masking
                atspm_m = m[:, :, :N_ATSPM].clone()
                drop = torch.rand_like(atspm_m.float()) < MASK_EXTRA
                train_m = m.clone()
                train_m[:, :, :N_ATSPM] = atspm_m & ~drop
                eval_m = m.clone()
                eval_m[:, :, :N_ATSPM] = atspm_m & drop

                if is_obsaware:
                    adj = build_sub_adj_dense(edge_index, nids, device)
                    # Compute obs quality from TRAINING mask (after dropping)
                    obs_q = torch.stack([compute_obs_quality(train_m[:, t])
                                        for t in range(x.shape[1])], dim=1)  # (S, W, 3)
                    pred = model(x, train_m, adj, obs_q)
                else:
                    adj = build_sub_adj_sparse(edge_index, nids, device)
                    pred = model(x, train_m, adj)

                loss = criterion(pred[eval_m], x[eval_m]) if eval_m.sum() > 0 else criterion(pred[m], x[m])
                total_loss += loss

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
            for bx, bm, bn in val_loader:
                for b in range(bx.shape[0]):
                    x = bx[b].to(device)
                    m = bm[b].to(device)
                    nids = bn[b]
                    drop = torch.rand_like(m[:, :, :N_ATSPM].float()) < MASK_EXTRA
                    tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~drop
                    em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & drop

                    if is_obsaware:
                        adj = build_sub_adj_dense(edge_index, nids, device)
                        obs_q = torch.stack([compute_obs_quality(tm[:, t])
                                             for t in range(x.shape[1])], dim=1)
                        pred = model(x, tm, adj, obs_q)
                    else:
                        adj = build_sub_adj_sparse(edge_index, nids, device)
                        pred = model(x, tm, adj)

                    if em.sum() > 0:
                        val_loss += criterion(pred[em], x[em]).item()
                    v_cnt += 1

        at = ep_loss / max(n_batch, 1)
        av = val_loss / max(v_cnt, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            log(f"  [{args.model}] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {elapsed:.1f}s")

        if av < best_val:
            best_val = av
            torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))

    log(f"\n{args.model.upper()} Training DONE! Best Val: {best_val:.4f}")

    # ═══════════════════════════════════════════════
    # EVALUATION on held-out test nodes
    # ═══════════════════════════════════════════════
    log(f"\n{'='*60}")
    log(f"Evaluating on held-out nodes (test time {t_val}-{T})...")
    log(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'),
                                     map_location=device, weights_only=True))
    model.eval()

    atspm_names = features[:N_ATSPM]
    all_preds, all_true, all_mask_out = [], [], []

    with torch.no_grad():
        for t0_eval in range(t_val, T, WINDOW // 2):
            t1_eval = min(t0_eval + WINDOW, T)
            if t1_eval - t0_eval < 4:
                continue
            actual_len = t1_eval - t0_eval

            x = data[heldout_nodes, t0_eval:t1_eval, :].to(device)
            m = mask[heldout_nodes, t0_eval:t1_eval, :].to(device)
            nids = torch.tensor(heldout_nodes, dtype=torch.long)

            if actual_len < WINDOW:
                S_h, _, F_h = x.shape
                pad = WINDOW - actual_len
                x = torch.cat([x, torch.zeros(S_h, pad, F_h, device=device)], dim=1)
                m = torch.cat([m, torch.zeros(S_h, pad, F_h, dtype=torch.bool, device=device)], dim=1)

            # Zero out ATSPM for held-out (simulate full missing)
            input_m = m.clone()
            input_m[:, :, :N_ATSPM] = False
            input_x = x.clone()
            input_x[:, :, :N_ATSPM] = 0.0

            if is_obsaware:
                adj = build_sub_adj_dense(edge_index, nids, device)
                obs_q = torch.stack([compute_obs_quality(input_m[:, t])
                                     for t in range(x.shape[1])], dim=1)
                pred = model(input_x, input_m, adj, obs_q)
            else:
                adj = build_sub_adj_sparse(edge_index, nids, device)
                pred = model(input_x, input_m, adj)

            all_preds.append(pred[:, :actual_len, :N_ATSPM].cpu())
            all_true.append(data[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            all_mask_out.append(mask[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])

    preds = torch.cat(all_preds, dim=1)
    trues = torch.cat(all_true, dim=1)
    masks_out = torch.cat(all_mask_out, dim=1)

    # Per-feature metrics
    results = []
    log(f"\n{'Feature':<28} {'MAE':>8} {'RMSE':>8} {'RelErr':>8} {'N':>8}")
    log("-" * 70)
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i]
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
    log(f"\n  {'AVERAGE':<28} {avg_mae:>8.2f} {'':>8} {avg_rel:>7.1f}%")

    # Save
    out = os.path.join(res_dir, 'evaluation.json')
    with open(out, 'w') as f:
        json.dump({'model': args.model, 'results': results,
                   'avg_mae': round(avg_mae, 2), 'avg_rel_error': round(avg_rel, 1),
                   'best_val_loss': best_val, 'epochs': EPOCHS}, f, indent=2)
    log(f"Saved: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['baseline_grin', 'obsaware_grin'])
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    train_and_evaluate(args)
