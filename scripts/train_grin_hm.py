#!/usr/bin/env python3
"""GRIN + Historical Mean Anchoring baseline."""
import os, sys, time, argparse, torch
from torch import nn, optim
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from shared_imputation import *

class SpatialAgg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, h, adj):
        return h + torch.relu(self.lin(torch.sparse.mm(adj, h)))

class GRIN_HM(nn.Module):
    def __init__(self, F_all, H):
        super().__init__()
        self.H = H
        self.fwd_gru = nn.GRUCell(F_all * 2, H)
        self.bwd_gru = nn.GRUCell(F_all * 2, H)
        self.sp_fwd = SpatialAgg(H)
        self.sp_bwd = SpatialAgg(H)
        self.out = nn.Linear(H * 2, N_ATSPM)

    def forward(self, x, m, adj):
        S, W, _ = x.shape; device = x.device
        hf = torch.zeros(S, self.H, device=device)
        fwd = []
        for t in range(W):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hf = self.fwd_gru(inp, hf); hf = self.sp_fwd(hf, adj); fwd.append(hf)
        hb = torch.zeros(S, self.H, device=device)
        bwd = [None] * W
        for t in range(W - 1, -1, -1):
            inp = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hb = self.bwd_gru(inp, hb); hb = self.sp_bwd(hb, adj); bwd[t] = hb
        outs = []
        for t in range(W):
            outs.append(self.out(torch.cat([fwd[t], bwd[t]], -1)))
        return torch.stack(outs, 1)

def main(args):
    device = torch.device(f'cuda:{args.gpu}')
    data, mask, hours, dows, split, features, edge_index = load_all()
    N, T, F_dim = data.shape
    train_nodes = split['train_nodes']; heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']; t_val = split['time_val_end']

    log("Computing historical mean...")
    hist_mean = compute_historical_mean(data, mask, hours, dows, train_nodes)
    log(f"  hist_mean: {hist_mean.shape}")

    model = GRIN_HM(F_dim, HIDDEN).to(device)
    log(f"Model: GRIN_HM, params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float('inf')
    res_dir = os.path.join(RESULTS_DIR, 'grin_hm'); os.makedirs(res_dir, exist_ok=True)

    log(f"\n{'='*60}\nTraining GRIN_HM for {EPOCHS} epochs...\n{'='*60}")
    for epoch in range(EPOCHS):
        t0 = time.time(); model.train(); ep_loss = 0; n_batch = 0
        perm_t = torch.randperm(max(1, (t_train - WINDOW) // (WINDOW // 2)))
        for idx in perm_t[:len(perm_t)]:
            tw0 = idx.item() * (WINDOW // 2); tw1 = min(tw0 + WINDOW, t_train)
            if tw1 - tw0 < 4: continue
            nids_perm = torch.randperm(len(train_nodes))[:NODE_SAMPLE]
            nids = [train_nodes[i] for i in nids_perm]
            x = data[nids, tw0:tw1, :].to(device)
            m = mask[nids, tw0:tw1, :].to(device)
            h = hours[tw0:tw1]
            hist = get_hist_for_window(hist_mean, nids, h).to(device)
            S, W, _ = m[:, :, :N_ATSPM].shape
            node_drop = torch.rand(S, device=device) < MASK_EXTRA
            drop = node_drop.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
            train_m = m.clone(); train_m[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~drop
            eval_m = m.clone(); eval_m[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & drop
            time_filt = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
            eval_m_f = eval_m.clone(); eval_m_f[:, :, :N_ATSPM] = eval_m[:, :, :N_ATSPM] & time_filt
            adj = build_sub_adj_sparse(edge_index, torch.tensor(nids), device)
            pred_res = model(x, train_m, adj)
            target_res = x[:, :, :N_ATSPM] - hist
            em = eval_m_f[:, :, :N_ATSPM]
            if em.sum() == 0: em = eval_m[:, :, :N_ATSPM]
            if em.sum() == 0: continue
            loss = F.l1_loss(pred_res[em], target_res[em])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); ep_loss += loss.item(); n_batch += 1
        scheduler.step()
        # Validation
        model.eval(); val_loss = 0; v_cnt = 0
        with torch.no_grad():
            for idx in range(max(1, (t_val - t_train - WINDOW) // (WINDOW // 2))):
                tw0 = t_train + idx * (WINDOW // 2); tw1 = min(tw0 + WINDOW, t_val)
                if tw1 - tw0 < 4: continue
                nids_perm = torch.randperm(len(train_nodes))[:NODE_SAMPLE]
                nids = [train_nodes[i] for i in nids_perm]
                x = data[nids, tw0:tw1, :].to(device)
                m = mask[nids, tw0:tw1, :].to(device)
                h = hours[tw0:tw1]
                hist = get_hist_for_window(hist_mean, nids, h).to(device)
                S, W, _ = m[:, :, :N_ATSPM].shape
                nd = torch.rand(S, device=device) < MASK_EXTRA
                dr = nd.unsqueeze(-1).unsqueeze(-1).expand(-1, W, N_ATSPM)
                tm = m.clone(); tm[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & ~dr
                em = m.clone(); em[:, :, :N_ATSPM] = m[:, :, :N_ATSPM] & dr
                tf = ((h >= 6) & (h < 21)).unsqueeze(0).unsqueeze(-1).to(device)
                em[:, :, :N_ATSPM] = em[:, :, :N_ATSPM] & tf
                adj = build_sub_adj_sparse(edge_index, torch.tensor(nids), device)
                pred_res = model(x, tm, adj)
                target_res = x[:, :, :N_ATSPM] - hist
                em_a = em[:, :, :N_ATSPM]
                if em_a.sum() > 0:
                    val_loss += F.l1_loss(pred_res[em_a], target_res[em_a]).item(); v_cnt += 1
        at = ep_loss / max(n_batch, 1); av = val_loss / max(v_cnt, 1)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log(f"  [grin_hm] E{epoch:03d} | T:{at:.4f} V:{av:.4f} | {time.time()-t0:.1f}s")
        if av < best_val:
            best_val = av; torch.save(model.state_dict(), os.path.join(res_dir, 'best_model.pt'))
    log(f"\nGRIN_HM DONE! Best Val: {best_val:.4f}")

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(res_dir, 'best_model.pt'), map_location=device, weights_only=True))
    model.eval()
    def model_fn(input_x, input_m, nids, dev):
        adj = build_sub_adj_sparse(edge_index, nids, dev)
        return model(input_x.to(dev), input_m.to(dev), adj)
    evaluate_model(model_fn, data, mask, hours, hist_mean, heldout_nodes, edge_index, t_val, features, device, 'grin_hm')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=0)
    main(p.parse_args())
