#!/usr/bin/env python3
"""
train_all_v2.py — Train 4 imputation models on V3 Phase-agnostic tensor.
Each model runs on a separate GPU via CUDA_VISIBLE_DEVICES.

Key fixes:
  - SpatialAgg uses sparse matmul (not python for-loop over edges)
  - All prints flush immediately for nohup/log visibility
  - Node-sampling (200 nodes/batch) + window (24 steps) to fit 24GB VRAM
"""

import os, json, argparse, time, sys
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
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
N_ATSPM = 15  # first 15 features are ATSPM

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

    log("Loading edge_index...")
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
# SPATIAL AGGREGATION (sparse matmul, no python loop)
# ═══════════════════════════════════════════════
class SpatialAgg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, h, adj):
        """h: (S, D), adj: (S, S) sparse normalized adjacency."""
        neigh = torch.sparse.mm(adj, h)
        return h + torch.relu(self.lin(neigh))


def build_sub_adj(edge_index, node_ids, device):
    """Build sparse normalized adjacency for sampled nodes."""
    id_set = set(node_ids.tolist())
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
            torch.zeros(0, device=device), (S, S)
        )
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), device=device)
    adj = torch.sparse_coo_tensor(indices, values, (S, S))
    # Row-normalize
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    deg_inv = 1.0 / deg
    # Scale values
    norm_vals = deg_inv[indices[0]]
    adj = torch.sparse_coo_tensor(indices, norm_vals, (S, S))
    return adj


# ═══════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════
class GRINModel(nn.Module):
    def __init__(self, F, H):
        super().__init__()
        self.H = H
        self.gru_fwd = nn.GRUCell(F * 2, H)
        self.gru_bwd = nn.GRUCell(F * 2, H)
        self.spatial = SpatialAgg(H)
        self.out = nn.Linear(H * 2, F)

    def forward(self, x, m, adj):
        S, W, F = x.shape
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
        out = torch.stack([self.out(torch.cat([fwd[t], bwd[t]], -1)) for t in range(W)], 1)
        return out


class SPINModel(nn.Module):
    def __init__(self, F, H, n_heads=4, n_layers=3):
        super().__init__()
        self.proj_in = nn.Linear(F + 1, H)
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(H, n_heads, H * 2, batch_first=True)
            for _ in range(n_layers)
        ])
        self.spatial = SpatialAgg(H)
        self.proj_out = nn.Linear(H, F)

    def forward(self, x, m, adj):
        S, W, F = x.shape
        mi = m.float().mean(-1, keepdim=True)
        h = torch.relu(self.proj_in(torch.cat([x * m.float(), mi], -1)))
        for layer in self.temporal_layers:
            h = layer(h)
        h_out = []
        for t in range(W):
            h_out.append(self.spatial(h[:, t], adj))
        h = torch.stack(h_out, 1)
        return self.proj_out(h)


class IGNNKModel(nn.Module):
    def __init__(self, F, H, W, n_layers=3):
        super().__init__()
        self.proj_in = nn.Linear(W * F * 2, H)
        self.gnns = nn.ModuleList([nn.Sequential(
            nn.Linear(H, H), nn.LayerNorm(H), nn.ReLU()
        ) for _ in range(n_layers)])
        self.spatial_layers = nn.ModuleList([SpatialAgg(H) for _ in range(n_layers)])
        self.proj_out = nn.Linear(H, W * F)
        self.W, self.F = W, F

    def forward(self, x, m, adj):
        S, W, F = x.shape
        flat = torch.cat([x.reshape(S, -1) * m.reshape(S, -1).float(),
                          m.reshape(S, -1).float()], -1)
        h = torch.relu(self.proj_in(flat))
        for gnn, spa in zip(self.gnns, self.spatial_layers):
            h = gnn(h) + spa(h, adj)
        return self.proj_out(h).reshape(S, W, F)


class PriSTIModel(nn.Module):
    def __init__(self, F, H, diff_steps=50):
        super().__init__()
        self.diff_steps = diff_steps
        betas = torch.linspace(0.0001, 0.02, diff_steps)
        alphas = 1.0 - betas
        self.register_buffer('alpha_bar', torch.cumprod(alphas, 0))
        self.prior_conv = nn.Conv1d(F + 1, H, 3, padding=1)
        self.spatial = SpatialAgg(H)
        self.time_emb = nn.Embedding(diff_steps, H)
        self.net = nn.Sequential(
            nn.Linear(H * 2 + F, H * 2), nn.GELU(),
            nn.Linear(H * 2, H), nn.GELU(),
            nn.Linear(H, F)
        )

    def get_prior(self, x, m, adj):
        S, W, F = x.shape
        mi = m.float().mean(-1, keepdim=True)
        inp = torch.cat([x * m.float(), mi], -1).permute(0, 2, 1)
        h = torch.relu(self.prior_conv(inp)).permute(0, 2, 1)
        p = []
        for t in range(W):
            p.append(self.spatial(h[:, t], adj))
        return torch.stack(p, 1)

    def forward(self, x, m, adj):
        S, W, F = x.shape
        prior = self.get_prior(x, m, adj)
        t = torch.randint(0, self.diff_steps, (S,), device=x.device)
        noise = torch.randn_like(x)
        ab = self.alpha_bar[t].sqrt().reshape(S, 1, 1)
        ab1 = (1 - self.alpha_bar[t]).sqrt().reshape(S, 1, 1)
        x_noisy = ab * x + ab1 * noise
        x_noisy = x * m.float() + x_noisy * (1 - m.float())
        te = self.time_emb(t).unsqueeze(1).expand(-1, W, -1)
        inp = torch.cat([x_noisy, prior, te], -1)
        pred_noise = self.net(inp)
        return pred_noise, noise


# ═══════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════
def train_model(args):
    device = torch.device('cuda:0')
    data, mask, split, features, edge_index = load_all()
    N, T, F = data.shape

    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']
    t_val = split['time_val_end']

    log(f"Data: ({N}, {T}, {F}), ATSPM={N_ATSPM}, train={len(train_nodes)}, held-out={len(heldout_nodes)}")
    log(f"Time: train={t_train}, val={t_val-t_train}, test={T-t_val}")

    train_ds = ImputationDataset(data, mask, train_nodes, 0, t_train, WINDOW)
    val_ds = ImputationDataset(data, mask, train_nodes, t_train, t_val, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    if args.model == 'grin':
        model = GRINModel(F, HIDDEN).to(device)
    elif args.model == 'spin':
        model = SPINModel(F, HIDDEN).to(device)
    elif args.model == 'ignnk':
        model = IGNNKModel(F, HIDDEN, WINDOW).to(device)
    elif args.model == 'pristi':
        model = PriSTIModel(F, HIDDEN).to(device)

    nparam = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.model}, params: {nparam:,}")

    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    is_diff = args.model == 'pristi'
    criterion = nn.MSELoss() if is_diff else nn.L1Loss()

    res_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(res_dir, exist_ok=True)
    best_val = float('inf')

    log(f"\nTraining {args.model.upper()} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        ep_loss = 0
        n_batch = 0

        for batch_x, batch_m, batch_nids in train_loader:
            B = batch_x.shape[0]
            total_loss = 0
            for b in range(B):
                x = batch_x[b].to(device)
                m = batch_m[b].to(device)
                nids = batch_nids[b]

                # Build sub-adjacency once per sample
                adj = build_sub_adj(edge_index, nids, device)

                # Extra masking for self-supervised loss
                atspm_m = m[:, :, :N_ATSPM].clone()
                drop = torch.rand_like(atspm_m.float()) < MASK_EXTRA
                train_m = m.clone()
                train_m[:, :, :N_ATSPM] = atspm_m & ~drop
                eval_m = m.clone()
                eval_m[:, :, :N_ATSPM] = atspm_m & drop

                if is_diff:
                    pn, tn = model(x, train_m, adj)
                    loss = criterion(pn[~train_m], tn[~train_m]) if (~train_m).sum() > 0 else criterion(pn, tn)
                else:
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

        # Val
        model.eval()
        val_loss, v_cnt = 0, 0
        with torch.no_grad():
            for bx, bm, bn in val_loader:
                for b in range(bx.shape[0]):
                    x = bx[b].to(device); m = bm[b].to(device); nids = bn[b]
                    adj = build_sub_adj(edge_index, nids, device)
                    drop = torch.rand_like(m[:,:,:N_ATSPM].float()) < MASK_EXTRA
                    tm = m.clone(); tm[:,:,:N_ATSPM] = m[:,:,:N_ATSPM] & ~drop
                    em = m.clone(); em[:,:,:N_ATSPM] = m[:,:,:N_ATSPM] & drop
                    if is_diff:
                        pn, tn = model(x, tm, adj)
                        if em.sum()>0: val_loss += criterion(pn[em], tn[em]).item()
                    else:
                        pred = model(x, tm, adj)
                        if em.sum()>0: val_loss += criterion(pred[em], x[em]).item()
                    v_cnt += 1

        at = ep_loss / max(n_batch, 1)
        av = val_loss / max(v_cnt, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            log(f"  [{args.model}] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {elapsed:.1f}s | LR:{opt.param_groups[0]['lr']:.6f}")

        if av < best_val:
            best_val = av
            torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))

    log(f"\n{args.model.upper()} DONE! Best Val: {best_val:.4f}")
    with open(os.path.join(res_dir, 'metrics.json'), 'w') as f:
        json.dump({'best_val': best_val, 'epochs': EPOCHS, 'model': args.model}, f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['grin', 'spin', 'ignnk', 'pristi'])
    p.add_argument('--gpu', type=int, default=0)
    train_model(p.parse_args())
