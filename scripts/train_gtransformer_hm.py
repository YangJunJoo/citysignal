#!/usr/bin/env python3
"""Graph Transformer + Historical Mean Anchoring."""
import os, sys, time, argparse, math, torch
from torch import nn, optim
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from shared_imputation import *

class GraphMultiHeadAttn(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
    def forward(self, x, adj_dense):
        # x: (S, D), adj_dense: (S, S) binary mask
        S, D = x.shape; H = self.n_heads; d = self.head_dim
        qkv = self.qkv(x).reshape(S, 3, H, d).permute(1, 2, 0, 3)  # (3, H, S, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (H, S, S)
        mask = adj_dense.unsqueeze(0).expand(H, -1, -1)
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).permute(1, 0, 2).reshape(S, D)
        return self.out_proj(out)

class GraphTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = GraphMultiHeadAttn(dim, n_heads)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
    def forward(self, x, adj):
        x = x + self.attn(self.ln1(x), adj)
        x = x + self.ff(self.ln2(x))
        return x

class GraphTransformer_HM(nn.Module):
    def __init__(self, F_all, H):
        super().__init__()
        self.H = H
        self.input_proj = nn.Linear(F_all * 2, H)
        self.time_fwd = nn.GRUCell(H, H)
        self.time_bwd = nn.GRUCell(H, H)
        self.gt_block1 = GraphTransformerBlock(H * 2, n_heads=4)
        self.gt_block2 = GraphTransformerBlock(H * 2, n_heads=4)
        self.out = nn.Linear(H * 2, N_ATSPM)

    def forward(self, x, m, adj_dense):
        S, W, F = x.shape; device = x.device
        inp = self.input_proj(torch.cat([x * m.float(), m.float()], -1))
        # Temporal: bidirectional GRU
        hf = torch.zeros(S, self.H, device=device); fwd = []
        for t in range(W): hf = self.time_fwd(inp[:, t], hf); fwd.append(hf)
        hb = torch.zeros(S, self.H, device=device); bwd = [None] * W
        for t in range(W - 1, -1, -1): hb = self.time_bwd(inp[:, t], hb); bwd[t] = hb
        # Spatial: Graph Transformer per timestep
        outs = []
        for t in range(W):
            h_cat = torch.cat([fwd[t], bwd[t]], -1)
            h_cat = self.gt_block1(h_cat, adj_dense)
            h_cat = self.gt_block2(h_cat, adj_dense)
            outs.append(self.out(h_cat))
        return torch.stack(outs, 1)

def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, hours, dows, split, features, edge_index = load_all()
    N, T, F_dim = data.shape
    train_nodes = split['train_nodes']; heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']; t_val = split['time_val_end']
    hist_mean = compute_historical_mean(data, mask, hours, dows, train_nodes)

    model = GraphTransformer_HM(F_dim, HIDDEN).to(device)
    log(f"Model: GraphTransformer_HM, params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf')
    res_dir = os.path.join(RESULTS_DIR, 'gtransformer_hm'); os.makedirs(res_dir, exist_ok=True)

    log(f"\n{'='*60}\nTraining GTRANSFORMER_HM for {EPOCHS} epochs...\n{'='*60}")
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
            adj_d = build_dense_adj(edge_index, torch.tensor(nids), device)
            pred_res = model(x, tm, adj_d)
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
                adj_d = build_dense_adj(edge_index, torch.tensor(nids), device)
                pred_res = model(x, tm, adj_d)
                target_res = x[:, :, :N_ATSPM] - hist
                em_a = em[:, :, :N_ATSPM]
                if em_a.sum() > 0:
                    val_loss += F.l1_loss(pred_res[em_a], target_res[em_a]).item(); v_cnt += 1
        at = ep_loss / max(n_batch, 1); av = val_loss / max(v_cnt, 1)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log(f"  [gtransformer_hm] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {time.time()-t0:.1f}s")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))
    log(f"\nGTRANSFORMER_HM DONE! Best Val: {best_val:.4f}")

    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()
    def model_fn(input_x, input_m, nids, dev):
        adj_d = build_dense_adj(edge_index, nids, dev)
        return model(input_x.to(dev), input_m.to(dev), adj_d)
    evaluate_model(model_fn, data, mask, hours, hist_mean, heldout_nodes, edge_index, t_val, features, device, 'gtransformer_hm')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=2)
    main(p.parse_args())
