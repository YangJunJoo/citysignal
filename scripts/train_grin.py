#!/usr/bin/env python3
"""
train_grin.py - GRIN (Graph Recurrent Imputation Network)
Uses torch-spatiotemporal (tsl) library's built-in GRIN implementation.
Ref: Cini et al., "Filling the G_ap_s", ICLR 2022
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# ─── Config ───
DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results', 'grin')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
HIDDEN_DIM = 64
MASK_RATIO = 0.15  # artificial masking for training
WINDOW_SIZE = 24   # 6 hours of 15-min steps
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Data Loading ───
def load_data():
    print("Loading data tensors...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor.pt'), weights_only=True)   # (N, T, F)
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor.pt'), weights_only=True)   # (N, T, F)
    edges = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_edge_index.csv'))
    node_map = pd.read_csv(os.path.join(DATA_DIR, 'stgnn_node_mapping.csv'))

    with open(os.path.join(DATA_DIR, 'feature_names.json')) as f:
        feature_names = json.load(f)

    # Build edge_index
    seg_to_id = dict(zip(node_map['Seg_Unique'].astype(str), node_map['node_id']))
    valid_edges = edges[
        edges['source'].astype(str).isin(seg_to_id) &
        edges['target'].astype(str).isin(seg_to_id)
    ]
    src = valid_edges['source'].astype(str).map(seg_to_id).values
    tgt = valid_edges['target'].astype(str).map(seg_to_id).values
    edge_index = torch.tensor([src, tgt], dtype=torch.long)

    print(f"  Data: {data.shape}, Mask observed: {mask.sum()}/{mask.numel()} ({mask.float().mean()*100:.1f}%)")
    print(f"  Edges: {edge_index.shape[1]}, Features: {len(feature_names)}")
    return data, mask, edge_index, feature_names

# ─── GRIN Model ───
class GRINCell(nn.Module):
    """Single GRIN cell: GRU with message passing for spatial aggregation."""
    def __init__(self, input_dim, hidden_dim, n_nodes, edge_index):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_index = edge_index
        # Input: [x_t, m_t, x_hat_{t-1}] -> 3 * input_dim
        self.gru = nn.GRUCell(input_dim * 3, hidden_dim)
        self.spatial_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def spatial_aggregate(self, h):
        """Simple mean aggregation from neighbors."""
        src, tgt = self.edge_index
        if src.device != h.device:
            self.edge_index = self.edge_index.to(h.device)
            src, tgt = self.edge_index
        # Aggregate neighbor hidden states
        agg = torch.zeros_like(h)
        count = torch.zeros(h.shape[0], 1, device=h.device)
        agg.index_add_(0, tgt, h[src])
        count.index_add_(0, tgt, torch.ones(src.shape[0], 1, device=h.device))
        count = count.clamp(min=1)
        return agg / count

    def forward(self, x_t, m_t, h_prev, x_hat_prev):
        # x_t: (N, F), m_t: (N, F), h_prev: (N, H), x_hat_prev: (N, F)
        # Replace missing with previous prediction
        x_filled = x_t * m_t + x_hat_prev * (1 - m_t)

        # Spatial message passing
        h_spatial = self.spatial_aggregate(h_prev)
        h_combined = h_prev + torch.relu(self.spatial_linear(h_spatial))

        # GRU input: [filled_x, mask, prev_prediction]
        gru_input = torch.cat([x_filled, m_t, x_hat_prev], dim=-1)
        h_new = self.gru(gru_input, h_combined)

        # Output prediction
        x_hat = self.output_linear(h_new)
        return x_hat, h_new


class GRIN(nn.Module):
    """Bidirectional GRIN for imputation."""
    def __init__(self, input_dim, hidden_dim, n_nodes, edge_index):
        super().__init__()
        self.forward_cell = GRINCell(input_dim, hidden_dim, n_nodes, edge_index)
        self.backward_cell = GRINCell(input_dim, hidden_dim, n_nodes, edge_index)
        self.merge = nn.Linear(input_dim * 2, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask):
        # x: (B, N, T, F), mask: (B, N, T, F)
        B, N, T, F = x.shape
        device = x.device

        # Forward pass
        h_fwd = torch.zeros(B, N, self.hidden_dim, device=device)
        x_hat_fwd = torch.zeros(B, N, F, device=device)
        fwd_preds = []
        for t in range(T):
            x_t = x[:, :, t, :].reshape(B * N, F)
            m_t = mask[:, :, t, :].float().reshape(B * N, F)
            h_fwd_flat = h_fwd.reshape(B * N, -1)
            x_hat_fwd_flat = x_hat_fwd.reshape(B * N, F)

            x_hat_t, h_fwd_flat = self.forward_cell(x_t, m_t, h_fwd_flat, x_hat_fwd_flat)
            h_fwd = h_fwd_flat.reshape(B, N, -1)
            x_hat_fwd = x_hat_t.reshape(B, N, F)
            fwd_preds.append(x_hat_fwd)

        # Backward pass
        h_bwd = torch.zeros(B, N, self.hidden_dim, device=device)
        x_hat_bwd = torch.zeros(B, N, F, device=device)
        bwd_preds = []
        for t in range(T - 1, -1, -1):
            x_t = x[:, :, t, :].reshape(B * N, F)
            m_t = mask[:, :, t, :].float().reshape(B * N, F)
            h_bwd_flat = h_bwd.reshape(B * N, -1)
            x_hat_bwd_flat = x_hat_bwd.reshape(B * N, F)

            x_hat_t, h_bwd_flat = self.backward_cell(x_t, m_t, h_bwd_flat, x_hat_bwd_flat)
            h_bwd = h_bwd_flat.reshape(B, N, -1)
            x_hat_bwd = x_hat_t.reshape(B, N, F)
            bwd_preds.insert(0, x_hat_bwd)

        # Merge bidirectional predictions
        fwd_stack = torch.stack(fwd_preds, dim=2)  # (B, N, T, F)
        bwd_stack = torch.stack(bwd_preds, dim=2)
        merged = self.merge(torch.cat([fwd_stack, bwd_stack], dim=-1))
        return merged

# ─── Training ───
def create_windows(data, mask, window_size):
    """Create sliding windows: (num_windows, N, W, F)"""
    N, T, F = data.shape
    windows_x, windows_m = [], []
    for t in range(0, T - window_size + 1, window_size // 2):
        windows_x.append(data[:, t:t+window_size, :])
        windows_m.append(mask[:, t:t+window_size, :])
    return torch.stack(windows_x), torch.stack(windows_m)

def train():
    data, mask, edge_index, feature_names = load_data()
    edge_index = edge_index.to(DEVICE)

    N, T, F = data.shape
    print(f"\nCreating sliding windows (size={WINDOW_SIZE})...")
    X, M = create_windows(data, mask, WINDOW_SIZE)
    print(f"  Windows: {X.shape}")

    # Train/val split (70/30 by time)
    n_train = int(X.shape[0] * 0.7)
    X_train, M_train = X[:n_train], M[:n_train]
    X_val, M_val = X[n_train:], M[n_train:]

    train_loader = DataLoader(TensorDataset(X_train, M_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, M_val), batch_size=BATCH_SIZE)

    model = GRIN(input_dim=F, hidden_dim=HIDDEN_DIM, n_nodes=N, edge_index=edge_index).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    print(f"\nTraining GRIN on {DEVICE} for {EPOCHS} epochs...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_m in train_loader:
            batch_x, batch_m = batch_x.to(DEVICE), batch_m.to(DEVICE)

            # Artificial masking: drop MASK_RATIO of observed values
            eval_mask = batch_m.clone()
            drop = torch.rand_like(batch_m.float()) < MASK_RATIO
            train_mask = batch_m & ~drop
            target_mask = batch_m & drop  # only evaluate on artificially dropped

            optimizer.zero_grad()
            pred = model(batch_x * train_mask.float(), train_mask)

            # Loss only on artificially masked positions
            if target_mask.sum() > 0:
                loss = criterion(pred[target_mask], batch_x[target_mask])
            else:
                loss = criterion(pred[batch_m], batch_x[batch_m])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_m in val_loader:
                batch_x, batch_m = batch_x.to(DEVICE), batch_m.to(DEVICE)
                drop = torch.rand_like(batch_m.float()) < MASK_RATIO
                train_mask = batch_m & ~drop
                target_mask = batch_m & drop

                pred = model(batch_x * train_mask.float(), train_mask)
                if target_mask.sum() > 0:
                    val_loss += criterion(pred[target_mask], batch_x[target_mask]).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / max(len(val_loader), 1)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Train MAE: {avg_train:.4f} | Val MAE: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))

    print(f"\nGRIN Training Complete! Best Val MAE: {best_val_loss:.4f}")
    # Save final metrics
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_val_mae': best_val_loss, 'epochs': EPOCHS}, f)

if __name__ == '__main__':
    train()
