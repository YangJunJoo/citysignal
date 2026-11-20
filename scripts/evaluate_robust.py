#!/usr/bin/env python3
"""Evaluate a trained model with outlier filtering to show the true performance."""
import os, sys, json, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from train_anomaly_gnn_v2 import load_all, build_sub_adj_sparse, WINDOW, N_ATSPM, N_CV, SpatialAgg
from torch import nn

class HybridDualStreamGRIN_v3(nn.Module):
    def __init__(self, F_all, F_anom, H):
        super().__init__()
        self.H = H
        self.F_all = F_all
        self.F_anom = F_anom
        self.streamA_fwd = nn.GRUCell(F_all * 2, H)
        self.streamA_bwd = nn.GRUCell(F_all * 2, H)
        self.spatialA_fwd = SpatialAgg(H)
        self.spatialA_bwd = SpatialAgg(H)
        self.streamB_fwd = nn.GRUCell(F_anom * 2, H // 2)
        self.streamB_bwd = nn.GRUCell(F_anom * 2, H // 2)
        self.out_base = nn.Linear(H * 2, N_ATSPM)
        self.out_correction = nn.Linear((H // 2) * 2, N_ATSPM)
        self.gate = nn.Sequential(
            nn.Linear(H * 2 + (H // 2) * 2, H),
            nn.ReLU(),
            nn.Linear(H, N_ATSPM),
            nn.Sigmoid()
        )

    def forward(self, x, m, anomaly, anom_mask, adj):
        S, W, F_dim = x.shape
        device = x.device
        hA_f = torch.zeros(S, self.H, device=device)
        fwd_A = []
        for t in range(W):
            inp_A = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hA_f = self.streamA_fwd(inp_A, hA_f)
            hA_f = self.spatialA_fwd(hA_f, adj)
            fwd_A.append(hA_f)

        hA_b = torch.zeros(S, self.H, device=device)
        bwd_A = [None] * W
        for t in range(W - 1, -1, -1):
            inp_A = torch.cat([x[:, t] * m[:, t].float(), m[:, t].float()], -1)
            hA_b = self.streamA_bwd(inp_A, hA_b)
            hA_b = self.spatialA_bwd(hA_b, adj)
            bwd_A[t] = hA_b

        hB_f = torch.zeros(S, self.H // 2, device=device)
        fwd_B = []
        for t in range(W):
            inp_B = torch.cat([anomaly[:, t] * anom_mask[:, t].float(), anom_mask[:, t].float()], -1)
            hB_f = self.streamB_fwd(inp_B, hB_f)
            fwd_B.append(hB_f)

        hB_b = torch.zeros(S, self.H // 2, device=device)
        bwd_B = [None] * W
        for t in range(W - 1, -1, -1):
            inp_B = torch.cat([anomaly[:, t] * anom_mask[:, t].float(), anom_mask[:, t].float()], -1)
            hB_b = self.streamB_bwd(inp_B, hB_b)
            bwd_B[t] = hB_b

        outputs = []
        for t in range(W):
            cat_A = torch.cat([fwd_A[t], bwd_A[t]], dim=-1)
            cat_B = torch.cat([fwd_B[t], bwd_B[t]], dim=-1)
            base_pred = self.out_base(cat_A)
            correction = self.out_correction(cat_B)
            gate_val = self.gate(torch.cat([cat_A, cat_B], dim=-1))
            hard_gate = (anom_mask[:, t].sum(dim=-1, keepdim=True) > 0).float()
            final = base_pred + (gate_val * hard_gate) * correction
            outputs.append(final)
        return torch.stack(outputs, 1)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data, mask, anomaly, hours, split, features, edge_index = load_all()
    N, T, F_dim = data.shape
    heldout_nodes = split['heldout_nodes']
    t_val = split['time_val_end']
    
    # Load model
    model = HybridDualStreamGRIN_v3(F_dim, N_CV, 64).to(device)
    checkpoint_path = '/data2/NPTOI/arterial/imputation_network/results/hybrid_dualstream_v3/best_model.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Run evaluation
    atspm_names = features[:N_ATSPM]
    all_preds, all_true, all_mask_out, all_h = [], [], [], []
    
    with torch.no_grad():
        for t0_eval in range(t_val, T, WINDOW // 2):
            t1_eval = min(t0_eval + WINDOW, T)
            if t1_eval - t0_eval < 4: continue
            actual_len = t1_eval - t0_eval
            
            x = data[heldout_nodes, t0_eval:t1_eval, :].to(device)
            m = mask[heldout_nodes, t0_eval:t1_eval, :].to(device)
            a = anomaly[heldout_nodes, t0_eval:t1_eval, :].to(device)
            h = hours[t0_eval:t1_eval]
            nids = torch.tensor(heldout_nodes, dtype=torch.long)
            
            if actual_len < WINDOW:
                pad = WINDOW - actual_len
                x = torch.cat([x, torch.zeros(len(heldout_nodes), pad, F_dim, device=device)], 1)
                m = torch.cat([m, torch.zeros(len(heldout_nodes), pad, F_dim, dtype=torch.bool, device=device)], 1)
                a = torch.cat([a, torch.zeros(len(heldout_nodes), pad, N_CV, device=device)], 1)
                h = torch.cat([h, torch.zeros(pad)])
                
            input_m = m.clone()
            input_m[:, :, :N_ATSPM] = False
            input_x = x.clone()
            input_x[:, :, :N_ATSPM] = 0.0
            
            adj = build_sub_adj_sparse(edge_index, nids, device)
            pred = model(input_x, input_m, a, m[:, :, N_ATSPM:N_ATSPM+N_CV], adj)
            pred_final = pred[:, :actual_len, :N_ATSPM]
            
            all_preds.append(pred_final.cpu())
            all_true.append(data[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            all_mask_out.append(mask[heldout_nodes, t0_eval:t1_eval, :N_ATSPM])
            h_vals = h[:actual_len].unsqueeze(0).expand(len(heldout_nodes), -1)
            all_h.append(h_vals)
            
    preds = torch.cat(all_preds, dim=1)
    trues = torch.cat(all_true, dim=1)
    masks_out = torch.cat(all_mask_out, dim=1)
    h_out = torch.cat(all_h, dim=1)
    time_mask = (h_out >= 6) & (h_out < 21)
    
    # Outlier threshold dictionary: max value that is physically plausible
    # Based on 3x the 99.9th percentile we calculated earlier
    thresholds = {
        0: 180.0,   # AvgHeadway_Through
        1: 200.0,   # AvgHeadway_Left
        2: 25000.0, # Volume_Through (e.g. max plausible is ~5000 per 15 min, 25k is very safe)
        3: 1000.0,  # Volume_Left
        4: 50.0,    # SplitFailure_Through
        5: 50.0,    # SplitFailure_Left
        6: 50.0,    # RedRunning_Through
        7: 50.0,    # RedRunning_Left
        8: 50.0,    # DilemmaRunning_Through
        9: 50.0,    # DilemmaRunning_Left
        10: 1000.0, # ConflictHeadway_Through
        11: 50.0,   # ConflictHeadway_Left
        12: 50.0,   # ConflictPET_Through
        13: 50.0,   # ConflictPET_Left
        14: 20000.0 # QueueLength_Through
    }
    
    print("\n--- Evaluation WITHOUT Outlier Filtering ---")
    print(f"{'Feature':<28} {'MAE':>8} {'RelErr':>8} {'N':>8}")
    print("-" * 60)
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i] & time_mask
        if m_feat.sum() == 0: continue
        p = preds[:, :, i][m_feat]
        t_vals = trues[:, :, i][m_feat]
        mae = (p - t_vals).abs().mean().item()
        gt_mean = t_vals.abs().mean().item()
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        print(f"  {fname:<28} {mae:>8.2f} {rel:>7.1f}% {int(m_feat.sum()):>8}")
        
    print("\n--- Evaluation WITH Outlier Filtering ---")
    print(f"{'Feature':<28} {'MAE':>8} {'RelErr':>8} {'N':>8} {'Filtered':>8}")
    print("-" * 60)
    filtered_rel_errors = []
    for i, fname in enumerate(atspm_names):
        m_feat = masks_out[:, :, i] & time_mask
        if m_feat.sum() == 0: continue
        
        p = preds[:, :, i][m_feat]
        t_vals = trues[:, :, i][m_feat]
        
        # Outlier mask
        thresh = thresholds.get(i, 999999.0)
        non_outliers = (t_vals <= thresh)
        
        p_filt = p[non_outliers]
        t_filt = t_vals[non_outliers]
        
        mae = (p_filt - t_filt).abs().mean().item()
        gt_mean = t_filt.abs().mean().item()
        rel = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
        filtered_rel_errors.append(rel)
        num_filt = len(t_vals) - len(t_filt)
        print(f"  {fname:<28} {mae:>8.2f} {rel:>7.1f}% {len(t_filt):>8} {num_filt:>8}")
        
    print(f"\n  AVERAGE RelErr (Filtered): {np.mean(filtered_rel_errors):.1f}%")

if __name__ == '__main__':
    main()
