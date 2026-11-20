#!/usr/bin/env python3
"""
evaluate_v3.py — Per-ATSPM-metric evaluation on held-out corridor-block nodes.
Loads best model checkpoints, runs inference on held-out nodes (test time),
computes MAE / RMSE per feature.
"""

import os, json, sys
import numpy as np
import torch
from torch import nn
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_all_v2 import (
    GRINModel, SPINModel, IGNNKModel, PriSTIModel,
    load_all, build_sub_adj, HIDDEN, WINDOW, N_ATSPM
)

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

def log(msg):
    print(msg, flush=True)


def evaluate_model(model_name, model, data, mask, split, edge_index, features, device):
    """Evaluate model on held-out nodes during test time."""
    heldout = split['heldout_nodes']
    t_val = split['time_val_end']
    T = split['time_total']

    # Test window: t_val → T
    all_preds = []
    all_true = []
    all_mask = []

    model.eval()
    with torch.no_grad():
        for t0 in range(t_val, T, WINDOW // 2):
            t1 = min(t0 + WINDOW, T)
            if t1 - t0 < 4:
                continue

            actual_len = t1 - t0
            x = data[heldout, t0:t1, :].to(device)
            m = mask[heldout, t0:t1, :].to(device)
            nids = torch.tensor(heldout, dtype=torch.long)
            adj = build_sub_adj(edge_index, nids, device)

            # Pad short windows (needed for IGNNK fixed input size)
            if actual_len < WINDOW:
                S, _, F_dim = x.shape
                pad = WINDOW - actual_len
                x = torch.cat([x, torch.zeros(S, pad, F_dim, device=device)], dim=1)
                m = torch.cat([m, torch.zeros(S, pad, F_dim, dtype=torch.bool, device=device)], dim=1)

            # Zero out ATSPM for held-out (simulate full missing)
            input_m = m.clone()
            input_m[:, :, :N_ATSPM] = False
            input_x = x.clone()
            input_x[:, :, :N_ATSPM] = 0.0

            if model_name == 'pristi':
                # Simple denoising: use forward at t=0 (least noise)
                pred_noise, _ = model(input_x, input_m, adj)
                pred = input_x - pred_noise  # approximate denoised
            else:
                pred = model(input_x, input_m, adj)

            # Trim back to actual length, ATSPM only
            all_preds.append(pred[:, :actual_len, :N_ATSPM].cpu())
            all_true.append(data[heldout, t0:t1, :N_ATSPM])
            all_mask.append(mask[heldout, t0:t1, :N_ATSPM])

    preds = torch.cat(all_preds, dim=1)  # (H, T_test, 15)
    trues = torch.cat(all_true, dim=1)
    masks = torch.cat(all_mask, dim=1)

    return preds, trues, masks


def compute_metrics(preds, trues, masks, feature_names):
    """Compute per-feature MAE, RMSE, MAPE."""
    results = []
    for i, fname in enumerate(feature_names):
        m = masks[:, :, i]
        if m.sum() == 0:
            results.append({'feature': fname, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'N_obs': 0})
            continue
        p = preds[:, :, i][m]
        t = trues[:, :, i][m]
        mae = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        # MAPE: avoid div by zero
        nonzero = t.abs() > 1e-6
        if nonzero.sum() > 0:
            mape = ((p[nonzero] - t[nonzero]).abs() / t[nonzero].abs()).mean().item() * 100
        else:
            mape = np.nan
        results.append({'feature': fname, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
                        'MAPE': round(mape, 2) if not np.isnan(mape) else np.nan,
                        'N_obs': int(m.sum().item())})
    return results


def main():
    log("Loading data...")
    data, mask, split, features, edge_index = load_all()
    N, T, F = data.shape
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    atspm_names = features[:N_ATSPM]
    log(f"ATSPM features: {atspm_names}")
    log(f"Held-out nodes: {len(split['heldout_nodes'])}")
    log(f"Test time: {split['time_val_end']} → {T}")

    models_config = {
        'grin': GRINModel(F, HIDDEN),
        'spin': SPINModel(F, HIDDEN),
        'ignnk': IGNNKModel(F, HIDDEN, WINDOW),
        'pristi': PriSTIModel(F, HIDDEN),
    }

    all_results = {}
    for mname, model in models_config.items():
        ckpt_path = os.path.join(RESULTS_DIR, mname, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            log(f"  {mname}: checkpoint not found, skipping")
            continue

        log(f"\nEvaluating {mname.upper()}...")
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        model = model.to(device)

        preds, trues, masks = evaluate_model(mname, model, data, mask, split, edge_index, features, device)
        metrics = compute_metrics(preds, trues, masks, atspm_names)
        all_results[mname] = metrics

        model.cpu()
        torch.cuda.empty_cache()

    # Build comparison table
    log("\n" + "=" * 80)
    log("PER-METRIC COMPARISON (MAE on held-out corridor-block nodes, test time)")
    log("=" * 80)

    header = f"{'Feature':<25}" + "".join(f"{'  ' + m.upper():>12}" for m in all_results.keys())
    log(header)
    log("-" * len(header))

    for i, fname in enumerate(atspm_names):
        row = f"{fname:<25}"
        for mname in all_results:
            val = all_results[mname][i]['MAE']
            row += f"{val:>12.4f}" if not np.isnan(val) else f"{'N/A':>12}"
        log(row)

    log("\n" + "=" * 80)
    log("PER-METRIC COMPARISON (RMSE)")
    log("=" * 80)
    header = f"{'Feature':<25}" + "".join(f"{'  ' + m.upper():>12}" for m in all_results.keys())
    log(header)
    log("-" * len(header))

    for i, fname in enumerate(atspm_names):
        row = f"{fname:<25}"
        for mname in all_results:
            val = all_results[mname][i]['RMSE']
            row += f"{val:>12.4f}" if not np.isnan(val) else f"{'N/A':>12}"
        log(row)

    # Average across features
    log("\n--- OVERALL AVERAGE ---")
    for mname in all_results:
        maes = [r['MAE'] for r in all_results[mname] if not np.isnan(r['MAE'])]
        rmses = [r['RMSE'] for r in all_results[mname] if not np.isnan(r['RMSE'])]
        log(f"  {mname.upper():>8}: MAE={np.mean(maes):.4f}, RMSE={np.mean(rmses):.4f}")

    # Save detailed results
    out_path = os.path.join(RESULTS_DIR, 'evaluation_v3.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
