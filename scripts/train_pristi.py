#!/usr/bin/env python3
"""
train_pristi.py - PriSTI (Prior-guided Spatiotemporal Transformer for Imputation)
Conditional diffusion model for spatiotemporal imputation.
Ref: Liu et al., "PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation", ICDE 2023
"""

import os, json, torch, numpy as np, pandas as pd, math
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results', 'pristi')
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 100; BATCH_SIZE = 16; LR = 0.0005; D_MODEL = 64
MASK_RATIO = 0.15; WINDOW_SIZE = 24
DIFFUSION_STEPS = 50; BETA_START = 0.0001; BETA_END = 0.02
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

# ─── Diffusion Schedule ───
def get_beta_schedule(steps, start, end):
    return torch.linspace(start, end, steps)

class SpatioTemporalPrior(nn.Module):
    """Extract coarse spatiotemporal features as conditioning prior."""
    def __init__(self, input_dim, d_model, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.temporal_conv = nn.Conv1d(input_dim + 1, d_model, kernel_size=3, padding=1)
        self.spatial_lin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x: (B, N, T, F), mask: (B, N, T, F)
        B, N, T, F = x.shape
        mask_ind = mask.float().mean(dim=-1, keepdim=True)
        x_in = torch.cat([x * mask.float(), mask_ind], dim=-1)  # (B, N, T, F+1)

        # Temporal conv per node
        x_flat = x_in.reshape(B * N, T, F + 1).permute(0, 2, 1)  # (BN, F+1, T)
        h = torch.relu(self.temporal_conv(x_flat))  # (BN, D, T)
        h = h.permute(0, 2, 1).reshape(B, N, T, -1)  # (B, N, T, D)

        # Spatial aggregation
        src, tgt = self.edge_index.to(x.device)
        D = h.shape[-1]
        agg = torch.zeros_like(h)
        cnt = torch.zeros(N, 1, 1, device=x.device).clamp(min=1)
        cnt_calc = torch.zeros(N, 1, 1, device=x.device)
        cnt_calc.index_add_(0, tgt, torch.ones(src.shape[0], 1, 1, device=x.device))
        cnt = cnt_calc.clamp(min=1)
        for b in range(B):
            agg[b].index_add_(0, tgt, h[b, src])
        h = self.norm(h + agg / cnt.unsqueeze(0))
        return h

class NoiseEstimator(nn.Module):
    """Estimate noise at diffusion step t, conditioned on prior."""
    def __init__(self, input_dim, d_model, diffusion_steps):
        super().__init__()
        self.time_embed = nn.Embedding(diffusion_steps, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.layers = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, input_dim)
        )

    def forward(self, x_noisy, t, prior):
        # x_noisy: (B, N, T, F), t: (B,), prior: (B, N, T, D)
        B, N, T, F = x_noisy.shape
        t_emb = self.time_embed(t).unsqueeze(1).unsqueeze(1).expand(B, N, T, -1)
        x_proj = self.input_proj(x_noisy)
        cond = self.cond_proj(prior)
        h = torch.cat([x_proj + t_emb, cond], dim=-1)
        return self.layers(h)

class PriSTI(nn.Module):
    def __init__(self, input_dim, d_model, edge_index, diffusion_steps):
        super().__init__()
        self.prior = SpatioTemporalPrior(input_dim, d_model, edge_index)
        self.noise_est = NoiseEstimator(input_dim, d_model, diffusion_steps)
        self.diffusion_steps = diffusion_steps
        betas = get_beta_schedule(diffusion_steps, BETA_START, BETA_END)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def forward(self, x, mask, t):
        prior = self.prior(x, mask)
        # Add noise to target (missing) positions
        noise = torch.randn_like(x)
        sqrt_acp = self.alphas_cumprod[t].sqrt().reshape(-1, 1, 1, 1)
        sqrt_1_acp = (1 - self.alphas_cumprod[t]).sqrt().reshape(-1, 1, 1, 1)
        x_noisy = sqrt_acp * x + sqrt_1_acp * noise
        # Keep observed positions clean
        x_noisy = x * mask.float() + x_noisy * (1 - mask.float())
        pred_noise = self.noise_est(x_noisy, t, prior)
        return pred_noise, noise

    @torch.no_grad()
    def impute(self, x, mask):
        """Reverse diffusion to impute missing values."""
        B, N, T, F = x.shape
        prior = self.prior(x, mask)
        x_t = torch.randn_like(x)
        # Keep observed values
        x_t = x * mask.float() + x_t * (1 - mask.float())

        for t_idx in range(self.diffusion_steps - 1, -1, -1):
            t = torch.full((B,), t_idx, device=x.device, dtype=torch.long)
            pred_noise = self.noise_est(x_t, t, prior)
            acp = self.alphas_cumprod[t_idx]
            beta = 1 - acp / (self.alphas_cumprod[t_idx - 1] if t_idx > 0 else torch.tensor(1.0))
            x_t = (x_t - beta / (1 - acp).sqrt() * pred_noise) / (1 - beta).sqrt()
            if t_idx > 0:
                x_t = x_t + beta.sqrt() * torch.randn_like(x_t) * 0.5
            x_t = x * mask.float() + x_t * (1 - mask.float())
        return x_t

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

    model = PriSTI(F, D_MODEL, edge_index, DIFFUSION_STEPS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val = float('inf')

    print(f"Training PriSTI on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train(); tl = 0
        for bx, bm in train_loader:
            bx, bm = bx.to(DEVICE), bm.to(DEVICE)
            # Artificial masking
            drop = torch.rand_like(bm.float()) < MASK_RATIO
            tm = bm & ~drop
            t = torch.randint(0, DIFFUSION_STEPS, (bx.shape[0],), device=DEVICE)
            optimizer.zero_grad()
            pred_noise, true_noise = model(bx, tm, t)
            # Loss on noise prediction at missing positions
            target_mask = ~tm
            if target_mask.sum() > 0:
                loss = criterion(pred_noise[target_mask], true_noise[target_mask])
            else:
                loss = criterion(pred_noise, true_noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); tl += loss.item()

        # Validation
        model.eval(); vl = 0
        with torch.no_grad():
            for bx, bm in val_loader:
                bx, bm = bx.to(DEVICE), bm.to(DEVICE)
                drop = torch.rand_like(bm.float()) < MASK_RATIO
                tm = bm & ~drop; em = bm & drop
                t = torch.randint(0, DIFFUSION_STEPS, (bx.shape[0],), device=DEVICE)
                pred_noise, true_noise = model(bx, tm, t)
                if em.sum() > 0: vl += criterion(pred_noise[em], true_noise[em]).item()

        at, av = tl/len(train_loader), vl/max(len(val_loader),1)
        if epoch % 10 == 0: print(f"  Epoch {epoch:03d} | Train: {at:.4f} | Val: {av:.4f}")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_model.pt'))

    print(f"\nPriSTI Complete! Best Val Loss: {best_val:.4f}")
    with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({'best_val_mse': best_val, 'epochs': EPOCHS}, f)

if __name__ == '__main__':
    train()
