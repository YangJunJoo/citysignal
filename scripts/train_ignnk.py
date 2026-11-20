#!/usr/bin/env python3
"""
train_ignnk.py - IGNNK (Inductive Graph Neural Network Kriging)
Random subgraph sampling + GNN kriging for spatiotemporal imputation.
Ref: Wu et al., "Inductive Graph Neural Networks for Spatiotemporal Kriging", AAAI 2021
"""

import os, json, torch, numpy as np, pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results', 'ignnk')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 100; BATCH_SIZE = 32; LR = 0.001; HIDDEN_DIM = 64
MASK_RATIO = 0.15; WINDOW_SIZE = 24; SUBGRAPH_SIZE = 200
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

class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        # x: (N, D)
        src, tgt = edge_index
        agg = torch.zeros_like(x[:, :self.lin_neigh.out_features] if x.shape[1] != self.lin_neigh.out_features else x)
        agg = torch.zeros(x.shape[0], self.lin_neigh.out_features, device=x.device)
        cnt = torch.zeros(x.shape[0], 1, device=x.device)
        neighbor_features = self.lin_neigh(x[src])
        agg.index_add_(0, tgt, neighbor_features)
        cnt.index_add_(0, tgt, torch.ones(src.shape[0], 1, device=x.device))
        cnt = cnt.clamp(min=1)
        h = self.lin_self(x) + agg / cnt
        return torch.relu(self.norm(h))

class IGNNK(nn.Module):
    """IGNNK: multi-layer GNN with random subgraph training."""
    def __init__(self, input_dim, hidden_dim, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim * 2, hidden_dim)  # [x, mask]
        self.gnns = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, mask, edge_index):
        # x: (N, T*F), mask: (N, T*F)
        h = torch.relu(self.input_proj(torch.cat([x * mask, mask], dim=-1)))
        for gnn in self.gnns:
            h = gnn(h, edge_index)
        return self.output_proj(h)

def sample_subgraph(N, edge_index, size):
    """Sample a random connected subgraph of given size."""
    nodes = torch.randperm(N)[:min(size, N)]
    node_set = set(nodes.tolist())
    src, tgt = edge_index
    mask = torch.tensor([s.item() in node_set and t.item() in node_set for s, t in zip(src, tgt)])
    sub_edges = edge_index[:, mask]
    # Reindex
    old2new = {old: new for new, old in enumerate(sorted(node_set))}
    if sub_edges.shape[1] > 0:
        sub_edges = torch.tensor([[old2new[s.item()] for s in sub_edges[0]],
                                   [old2new[t.item()] for t in sub_edges[1]]], dtype=torch.long)
    return sorted(node_set), sub_edges

def create_windows(data, mask, ws):
    """Create temporal windows: (num_windows, N, ws, F) -> flatten to (num_windows, N, ws*F)"""
    N, T, F = data.shape
    wx, wm = [], []
    for t in range(0, T - ws + 1, ws // 2):
        wx.append(data[:, t:t+ws, :])
        wm.append(mask[:, t:t+ws, :])
    return torch.stack(wx), torch.stack(wm)  # (W, N, ws, F)

def train():
    data, mask, edge_index = load_data()
    N, T, F = data.shape

    # Use windowed approach to avoid OOM
    print(f"Creating windows (size={WINDOW_SIZE})...")
    X_wins, M_wins = create_windows(data, mask, WINDOW_SIZE)
    W = X_wins.shape[0]
    input_dim = WINDOW_SIZE * F  # 24 * 47 = 1128 (manageable)
    
    # Flatten each window: (W, N, ws*F)
    X_flat = X_wins.reshape(W, N, input_dim)
    M_flat = M_wins.float().reshape(W, N, input_dim)
    
    # Train/val split by time windows
    n_train = int(W * 0.7)
    
    model = IGNNK(input_dim, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    best_val = float('inf')

    print(f"Training IGNNK on {DEVICE} for {EPOCHS} epochs...")
    print(f"  Windows: {W} (train: {n_train}, val: {W - n_train}), Input dim: {input_dim}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Iterate over training windows
        for w_idx in range(n_train):
            # Sample subgraph for each window
            sub_nodes, sub_edges = sample_subgraph(N, edge_index, SUBGRAPH_SIZE)
            sub_x = X_flat[w_idx, sub_nodes].to(DEVICE)
            sub_m = M_flat[w_idx, sub_nodes].to(DEVICE)
            sub_edges = sub_edges.to(DEVICE)

            # Mask some observed nodes
            n_sub = len(sub_nodes)
            n_mask = max(1, int(n_sub * MASK_RATIO))
            mask_idx = torch.randperm(n_sub)[:n_mask]

            train_m = sub_m.clone()
            train_m[mask_idx] = 0

            optimizer.zero_grad()
            pred = model(sub_x, train_m, sub_edges)

            target_m = torch.zeros_like(sub_m, dtype=torch.bool)
            target_m[mask_idx] = sub_m[mask_idx].bool()

            if target_m.sum() > 0:
                loss = criterion(pred[target_m], sub_x[target_m])
            else:
                loss = criterion(pred, sub_x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation on held-out windows with subgraph sampling
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for w_idx in range(n_train, W):
                sub_nodes, sub_edges = sample_subgraph(N, edge_index, SUBGRAPH_SIZE)
                sub_x = X_flat[w_idx, sub_nodes].to(DEVICE)
                sub_m = M_flat[w_idx, sub_nodes].to(DEVICE)
                sub_edges = sub_edges.to(DEVICE)

                n_sub = len(sub_nodes)
                n_mask = max(1, int(n_sub * MASK_RATIO))
                mask_idx = torch.randperm(n_sub)[:n_mask]

                val_m = sub_m.clone()
                val_m[mask_idx] = 0
                target_m = torch.zeros_like(sub_m, dtype=torch.bool)
                target_m[mask_idx] = sub_m[mask_idx].bool()

                pred = model(sub_x, val_m, sub_edges)
                if target_m.sum() > 0:
                    val_loss += criterion(pred[target_m], sub_x[target_m]).item()
                    val_count += 1

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_count, 1)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))

    print(f"\nIGNNK Complete! Best Val MAE: {best_val:.4f}")
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_val_mae': best_val, 'epochs': EPOCHS}, f)

if __name__ == '__main__':
    train()
