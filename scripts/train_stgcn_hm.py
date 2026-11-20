#!/usr/bin/env python3
"""STGCN + Historical Mean Anchoring."""
import os, sys, time, argparse, torch
from torch import nn, optim
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from shared_imputation import *

class ChebConv(nn.Module):
    def __init__(self, in_ch, out_ch, K=3):
        super().__init__()
        self.K = K
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(in_ch, out_ch) * 0.1) for _ in range(K)])
        self.bias = nn.Parameter(torch.zeros(out_ch))
    def forward(self, x, L):
        # x: (S, F_in), L: sparse normalized Laplacian
        T_list = [x]
        if self.K > 1: T_list.append(torch.sparse.mm(L, x))
        for k in range(2, self.K):
            T_list.append(2 * torch.sparse.mm(L, T_list[-1]) - T_list[-2])
        out = sum(T_list[k] @ self.weights[k] for k in range(min(self.K, len(T_list))))
        return out + self.bias

class GatedTemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch * 2, kernel, padding=kernel // 2)
    def forward(self, x):
        # x: (S, C, W)
        y = self.conv(x)
        a, b = y.chunk(2, dim=1)
        return a * torch.sigmoid(b)

class STConvBlock(nn.Module):
    def __init__(self, channels, K=3, kernel=3):
        super().__init__()
        self.temp1 = GatedTemporalConv(channels, channels, kernel)
        self.cheb = ChebConv(channels, channels, K)
        self.temp2 = GatedTemporalConv(channels, channels, kernel)
        self.bn = nn.BatchNorm1d(channels)
    def forward(self, x, L):
        # x: (S, C, W)
        S, C, W = x.shape
        h = self.temp1(x)  # (S, C, W)
        # Apply ChebConv per timestep
        h_spatial = []
        for t in range(W):
            h_spatial.append(self.cheb(h[:, :, t], L))
        h = torch.stack(h_spatial, dim=2)  # (S, C, W)
        h = self.temp2(h)
        h = self.bn(h)
        return h + x  # residual

class STGCN_HM(nn.Module):
    def __init__(self, F_all, H):
        super().__init__()
        self.input_proj = nn.Linear(F_all * 2, H)
        self.block1 = STConvBlock(H, K=3, kernel=3)
        self.block2 = STConvBlock(H, K=3, kernel=3)
        self.out = nn.Linear(H, N_ATSPM)
    def forward(self, x, m, L):
        S, W, F = x.shape
        inp = torch.cat([x * m.float(), m.float()], -1)
        h = self.input_proj(inp)  # (S, W, H)
        h = h.permute(0, 2, 1)   # (S, H, W)
        h = self.block1(h, L)
        h = self.block2(h, L)
        h = h.permute(0, 2, 1)   # (S, W, H)
        return self.out(h)        # (S, W, N_ATSPM)

def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, hours, dows, split, features, edge_index = load_all()
    N, T, F_dim = data.shape
    train_nodes = split['train_nodes']; heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']; t_val = split['time_val_end']
    hist_mean = compute_historical_mean(data, mask, hours, dows, train_nodes)

    model = STGCN_HM(F_dim, HIDDEN).to(device)
    log(f"Model: STGCN_HM, params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf')
    res_dir = os.path.join(RESULTS_DIR, 'stgcn_hm'); os.makedirs(res_dir, exist_ok=True)

    log(f"\n{'='*60}\nTraining STGCN_HM for {EPOCHS} epochs...\n{'='*60}")
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
            tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~dr
            em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & dr
            tf = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
            em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & tf
            L = compute_laplacian(edge_index, torch.tensor(nids), device)
            pred_res = model(x, tm, L)
            target_res = x[:, :, :N_ATSPM] - hist
            em_a = em[:, :, :N_ATSPM]
            if em_a.sum() == 0: continue
            loss = F.l1_loss(pred_res[em_a], target_res[em_a])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); ep_loss += loss.item(); n_batch += 1
        scheduler.step()
        model.eval(); val_loss = 0; v_cnt = 0
        with torch.no_grad():
            for idx in range(max(1, (t_val - t_train - WINDOW) // (WINDOW // 2))):
                tw0 = t_train + idx * (WINDOW // 2); tw1 = min(tw0 + WINDOW, t_val)
                if tw1 - tw0 < 4: continue
                nids = [train_nodes[i] for i in torch.randperm(len(train_nodes))[:NODE_SAMPLE]]
                x = data[nids, tw0:tw1, :].to(device); m = mask[nids, tw0:tw1, :].to(device)
                h = hours[tw0:tw1]; hist = get_hist_for_window(hist_mean, nids, h).to(device)
                S, W, _ = m[:, :, :N_ATSPM].shape
                nd = torch.rand(S, device=device) < MASK_EXTRA
                dr = nd.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
                tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~dr
                em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & dr
                tf = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & tf
                L = compute_laplacian(edge_index, torch.tensor(nids), device)
                pred_res = model(x, tm, L)
                target_res = x[:, :, :N_ATSPM] - hist
                em_a = em[:, :, :N_ATSPM]
                if em_a.sum() > 0:
                    val_loss += F.l1_loss(pred_res[em_a], target_res[em_a]).item(); v_cnt += 1
        at = ep_loss / max(n_batch, 1); av = val_loss / max(v_cnt, 1)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log(f"  [stgcn_hm] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {time.time()-t0:.1f}s")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))
    log(f"\nSTGCN_HM DONE! Best Val: {best_val:.4f}")

    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()
    def model_fn(input_x, input_m, nids, dev):
        L = compute_laplacian(edge_index, nids, dev)
        return model(input_x.to(dev), input_m.to(dev), L)
    evaluate_model(model_fn, data, mask, hours, hist_mean, heldout_nodes, edge_index, t_val, features, device, 'stgcn_hm')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=1)
    main(p.parse_args())
