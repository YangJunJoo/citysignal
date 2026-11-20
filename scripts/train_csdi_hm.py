#!/usr/bin/env python3
"""CSDI (Conditional Score-based Diffusion Imputation) + Historical Mean."""
import os, sys, time, argparse, math, torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from shared_imputation import *

class SpatialAggD(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, h, adj):
        return h + torch.relu(self.lin(torch.sparse.mm(adj, h)))

class DiffusionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        return self.proj(t.float().unsqueeze(-1))

class DenoiserBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.spatial = SpatialAggD(dim)
    def forward(self, x, adj):
        h = self.ln(x)
        h = self.spatial(h, adj)
        return x + self.ff(h)

class GraphDenoiser(nn.Module):
    def __init__(self, F_cond, H, n_steps=50):
        super().__init__()
        self.cond_proj = nn.Linear(F_cond * 2, H)
        self.noisy_proj = nn.Linear(N_ATSPM, H)
        self.time_emb = DiffusionEmbedding(H)
        self.blocks = nn.ModuleList([DenoiserBlock(H) for _ in range(4)])
        self.out = nn.Linear(H, N_ATSPM)

    def forward(self, x_noisy, t_diff, cond, cond_mask, adj):
        # x_noisy: (S, N_ATSPM) noisy target
        # t_diff: scalar diffusion timestep
        # cond: (S, F_cond) conditioning features
        # cond_mask: (S, F_cond) mask
        h_cond = self.cond_proj(torch.cat([cond * cond_mask.float(), cond_mask.float()], -1))
        h_noisy = self.noisy_proj(x_noisy)
        h_time = self.time_emb(t_diff.expand(x_noisy.shape[0]))
        h = h_cond + h_noisy + h_time
        for block in self.blocks:
            h = block(h, adj)
        return self.out(h)

class CSDI_Graph(nn.Module):
    def __init__(self, F_cond, H, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        betas = np.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer('alpha_bars', torch.tensor(alpha_bars, dtype=torch.float32))
        self.denoiser = GraphDenoiser(F_cond, H, n_steps)

    def forward(self, x_0, cond, cond_mask, adj):
        """Training: noise prediction loss."""
        S, _F = x_0.shape
        t = torch.randint(0, self.n_steps, (1,), device=x_0.device)
        noise = torch.randn_like(x_0)
        ab = self.alpha_bars[t]
        x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * noise
        pred_noise = self.denoiser(x_t, t.float() / self.n_steps, cond, cond_mask, adj)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, cond, cond_mask, adj, shape):
        """Inference: iterative denoising."""
        device = cond.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
            pred_noise = self.denoiser(x, t_tensor / self.n_steps, cond, cond_mask, adj)
            beta = self.betas[t]; alpha = self.alphas[t]; ab = self.alpha_bars[t]
            x = (x - beta / (1 - ab).sqrt() * pred_noise) / alpha.sqrt()
            if t > 0:
                x += beta.sqrt() * torch.randn_like(x)
        return x

def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, hours, dows, split, features, edge_index = load_all()
    N, T, F_dim = data.shape
    train_nodes = split['train_nodes']; heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']; t_val = split['time_val_end']
    hist_mean = compute_historical_mean(data, mask, hours, dows, train_nodes)

    model = CSDI_Graph(F_dim, HIDDEN, n_steps=50).to(device)
    log(f"Model: CSDI_Graph, params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=LR * 0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf')
    res_dir = os.path.join(RESULTS_DIR, 'csdi_hm'); os.makedirs(res_dir, exist_ok=True)

    log(f"\n{'='*60}\nTraining CSDI_HM for {EPOCHS} epochs...\n{'='*60}")
    for epoch in range(EPOCHS):
        t0 = time.time(); model.train(); ep_loss = 0; n_batch = 0
        perm_t = torch.randperm(max(1, (t_train - WINDOW) // (WINDOW // 2)))
        for idx in perm_t:
            tw0 = idx.item() * (WINDOW // 2); tw1 = min(tw0 + WINDOW, t_train)
            if tw1 - tw0 < 4: continue
            nids = [train_nodes[i] for i in torch.randperm(len(train_nodes))[:NODE_SAMPLE]]
            x = data[nids, tw0:tw1, :].to(device); m = mask[nids, tw0:tw1, :].to(device)
            h = hours[tw0:tw1]; hist = get_hist_for_window(hist_mean, nids, h).to(device)
            S, W, _ = m[:, :, :N_ATSPM].shape
            nd = torch.rand(S, device=device) < MASK_EXTRA
            dr = nd.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
            em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & dr
            tf = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
            em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & tf
            adj = build_sub_adj_sparse(edge_index, torch.tensor(nids), device)
            # Train per-timestep (CSDI processes one timestep at a time for graph)
            total_loss = 0; cnt = 0
            for t in range(W):
                em_t = em[:, t, :N_ATSPM]
                if em_t.sum() == 0: continue
                target_res = x[:, t, :N_ATSPM] - hist[:, t, :]
                cond = x[:, t, :]  # all features as conditioning
                cond_m = m[:, t, :].clone()
                cond_m[:, :N_ATSPM] = cond_m[:, :N_ATSPM] & ~dr[:, t, :]  # hide dropped ATSPM
                loss = model(target_res, cond, cond_m, adj)
                total_loss += loss; cnt += 1
            if cnt > 0:
                (total_loss / cnt).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); opt.zero_grad()
                ep_loss += (total_loss / cnt).item(); n_batch += 1
        scheduler.step()
        # Validation
        model.eval(); val_loss = 0; v_cnt = 0
        with torch.no_grad():
            for idx in range(min(10, max(1, (t_val - t_train - WINDOW) // (WINDOW // 2)))):
                tw0 = t_train + idx * (WINDOW // 2); tw1 = min(tw0 + WINDOW, t_val)
                if tw1 - tw0 < 4: continue
                nids = [train_nodes[i] for i in torch.randperm(len(train_nodes))[:NODE_SAMPLE]]
                x = data[nids, tw0:tw1, :].to(device); m = mask[nids, tw0:tw1, :].to(device)
                h = hours[tw0:tw1]; hist = get_hist_for_window(hist_mean, nids, h).to(device)
                S, W, _ = m[:, :, :N_ATSPM].shape
                nd = torch.rand(S, device=device) < MASK_EXTRA
                dr = nd.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
                em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & dr
                tf = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & tf
                adj = build_sub_adj_sparse(edge_index, torch.tensor(nids), device)
                for t in range(W):
                    em_t = em[:, t, :N_ATSPM]
                    if em_t.sum() == 0: continue
                    target_res = x[:, t, :N_ATSPM] - hist[:, t, :]
                    cond = x[:, t, :]; cond_m = m[:, t, :].clone()
                    cond_m[:, :N_ATSPM] = cond_m[:, :N_ATSPM] & ~dr[:, t, :]
                    # Sample and compute L1
                    sampled = model.sample(cond, cond_m, adj, target_res.shape)
                    val_loss += F.l1_loss(sampled[em_t], target_res[em_t]).item()
                    v_cnt += 1
        at = ep_loss / max(n_batch, 1); av = val_loss / max(v_cnt, 1)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log(f"  [csdi_hm] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {time.time()-t0:.1f}s")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))
    log(f"\nCSDI_HM DONE! Best Val: {best_val:.4f}")

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()
    atspm_names = features[:N_ATSPM]
    all_preds, all_true, all_mask_out, all_h_vals = [], [], [], []
    with torch.no_grad():
        for t0_eval in range(t_val, T, WINDOW // 2):
            t1_eval = min(t0_eval + WINDOW, T)
            if t1_eval - t0_eval < 4: continue
            actual_len = t1_eval - t0_eval
            x = data[heldout_nodes, t0_eval:t1_eval, :].to(device)
            m = mask[heldout_nodes, t0_eval:t1_eval, :].to(device)
            h = hours[t0_eval:t1_eval]
            nids = torch.tensor(heldout_nodes, dtype=torch.long)
            hist = get_hist_for_window(hist_mean, heldout_nodes, h).to(device)
            adj = build_sub_adj_sparse(edge_index, nids, device)
            input_m = m.clone(); input_m[:, :, :N_ATSPM] = False
            preds_t = []
            for t in range(actual_len):
                cond = x[:, t, :].clone(); cond[:, :N_ATSPM] = 0
                cond_m = input_m[:, t, :]
                # Average 3 samples for stability
                samples = [model.sample(cond, cond_m, adj, (len(heldout_nodes), N_ATSPM)) for _ in range(3)]
                pred_res = torch.stack(samples).mean(0)
                pred_atspm = hist[:, t, :] + pred_res
                preds_t.append(pred_atspm)
            all_preds.append(torch.stack(preds_t, 1).cpu())
            all_true.append(data[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            all_mask_out.append(mask[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            h_v = h[:actual_len].unsqueeze(0).expand(len(heldout_nodes), -1)
            all_h_vals.append(h_v)
    preds = torch.cat(all_preds, dim=1); trues = torch.cat(all_true, dim=1)
    masks_out = torch.cat(all_mask_out, dim=1); h_out = torch.cat(all_h_vals, dim=1)
    time_mask = (h_out >= 6) & (h_out < 21)
    log(f"\n{'Feature':<28} {'MAE':>8} {'RelErr':>8} {'N':>8}")
    log("-" * 60)
    results = []
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i] & time_mask
        if m_feat.sum() == 0: log(f"  {fname:<28} {'N/A':>8}"); continue
        p = preds[:, :, i][m_feat]; t_vals = trues[:, :, i][m_feat]
        mae = (p - t_vals).abs().mean().item()
        gt_mean = t_vals.abs().mean().item()
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        results.append({'feature': fname, 'MAE': mae, 'RelError': rel})
        log(f"  {fname:<28} {mae:>8.2f} {rel:>7.1f}% {int(m_feat.sum()):>8}")
    if results:
        avg_rel = np.mean([r['RelError'] for r in results])
        log(f"\n  {'AVERAGE (6AM-9PM)':<28} {'':>8} {avg_rel:>7.1f}%")

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=3)
    main(p.parse_args())
