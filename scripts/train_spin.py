#!/usr/bin/env python3
"""
train_spin.py - SPIN (Sparse Spatiotemporal Imputation Network)
Attention-based imputation: inter-node cross-attention + intra-node self-attention.
Ref: Marisca et al., "Learning to Reconstruct Missing Data from Spatiotemporal Graphs
     with Sparse Observations", NeurIPS 2022
"""

import os, json, torch, numpy as np, pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import math

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results', 'spin')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 100; BATCH_SIZE = 16; LR = 0.001; D_MODEL = 64; N_HEADS = 4
MASK_RATIO = 0.15; WINDOW_SIZE = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor.pt'), weights_only=True)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor.pt'), weights_only=True)
    edges = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping.csv'))
    seg_to_id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    valid = edges[edges['source'].astype(str).isin(seg_to_id) & edges['target'].astype(str).isin(seg_to_id)]
    src = valid['source'].astype(str).map(seg_to_id).values
    tgt = valid['target'].astype(str).map(seg_to_id).values
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    return data, mask, edge_index

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SPINBlock(nn.Module):
    """One SPIN block: intra-node self-attention + inter-node cross-attention."""
    def __init__(self, d_model, n_heads, n_nodes, edge_index):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.inter_linear_q = nn.Linear(d_model, d_model)
        self.inter_linear_k = nn.Linear(d_model, d_model)
        self.inter_linear_v = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.edge_index = edge_index

    def forward(self, x, mask_bool):
        # x: (B, N, T, D), mask_bool: (B, N, T)
        B, N, T, D = x.shape
        # 1. Intra-node self-attention (temporal, per node)
        x_flat = x.reshape(B * N, T, D)
        m_flat = mask_bool.reshape(B * N, T)
        # Create attention mask: only attend to observed timesteps
        attn_mask = ~m_flat.unsqueeze(1).expand(-1, T, -1)  # (BN, T, T)
        residual = x_flat
        x_attn, _ = self.intra_attn(x_flat, x_flat, x_flat, key_padding_mask=~m_flat)
        x_flat = self.norm1(residual + x_attn)

        # 2. Inter-node cross-attention (spatial, per timestep)
        x_spatial = x_flat.reshape(B, N, T, D)
        src, tgt = self.edge_index
        if src.device != x.device:
            self.edge_index = self.edge_index.to(x.device)
            src, tgt = self.edge_index
        # Simple neighbor aggregation via attention
        x_agg = torch.zeros_like(x_spatial)
        cnt = torch.zeros(N, 1, 1, device=x.device)
        cnt.index_add_(0, tgt, torch.ones(src.shape[0], 1, 1, device=x.device))
        cnt = cnt.clamp(min=1)
        for b in range(B):
            neighbor_vals = x_spatial[b, src]  # (E, T, D)
            x_agg[b].index_add_(0, tgt, neighbor_vals)
        x_agg = x_agg / cnt.unsqueeze(0)
        x_spatial = self.norm2(x_spatial + x_agg)

        # 3. FFN
        x_out = self.norm3(x_spatial + self.ffn(x_spatial))
        return x_out

class SPIN(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_nodes, edge_index, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + 1, d_model)  # +1 for mask indicator
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([SPINBlock(d_model, n_heads, n_nodes, edge_index) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x, mask):
        # x: (B, N, T, F), mask: (B, N, T, F)
        B, N, T, F = x.shape
        mask_indicator = mask.float().mean(dim=-1, keepdim=True)  # (B, N, T, 1)
        x_in = torch.cat([x * mask.float(), mask_indicator], dim=-1)
        h = self.input_proj(x_in)  # (B, N, T, D)
        # Add positional encoding per node
        h = h.reshape(B * N, T, -1)
        h = self.pos_enc(h)
        h = h.reshape(B, N, T, -1)
        mask_bool = mask.any(dim=-1)  # (B, N, T)
        for block in self.blocks:
            h = block(h, mask_bool)
        return self.output_proj(h)

def create_windows(data, mask, ws):
    N, T, F = data.shape
    wx, wm = [], []
    for t in range(0, T - ws + 1, ws // 2):
        wx.append(data[:, t:t+ws, :]); wm.append(mask[:, t:t+ws, :])
    return torch.stack(wx), torch.stack(wm)

def train():
    data, mask, edge_index = load_data()
    edge_index = edge_index.to(DEVICE)
    N, T, F = data.shape
    X, M = create_windows(data, mask, WINDOW_SIZE)
    n_train = int(X.shape[0] * 0.7)
    train_loader = DataLoader(TensorDataset(X[:n_train], M[:n_train]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X[n_train:], M[n_train:]), batch_size=BATCH_SIZE)

    model = SPIN(F, D_MODEL, N_HEADS, N, edge_index).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    best_val = float('inf')

    print(f"Training SPIN on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for bx, bm in train_loader:
            bx, bm = bx.to(DEVICE), bm.to(DEVICE)
            drop = torch.rand_like(bm.float()) < MASK_RATIO
            tm = bm & ~drop; em = bm & drop
            optimizer.zero_grad()
            pred = model(bx * tm.float(), tm)
            loss = criterion(pred[em], bx[em]) if em.sum() > 0 else criterion(pred[bm], bx[bm])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); tl += loss.item()

        model.eval(); vl = 0
        with torch.no_grad():
            for bx, bm in val_loader:
                bx, bm = bx.to(DEVICE), bm.to(DEVICE)
                drop = torch.rand_like(bm.float()) < MASK_RATIO
                tm = bm & ~drop; em = bm & drop
                pred = model(bx * tm.float(), tm)
                if em.sum() > 0: vl += criterion(pred[em], bx[em]).item()

        at, av = tl/len(train_loader), vl/max(len(val_loader),1)
        if epoch % 10 == 0: print(f"  Epoch {epoch:03d} | Train: {at:.4f} | Val: {av:.4f}")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))

    print(f"\nSPIN Complete! Best Val MAE: {best_val:.4f}")
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_val_mae': best_val, 'epochs': EPOCHS}, f)

if __name__ == '__main__':
    train()
