#!/usr/bin/env python3
"""
train_regression.py — CV → ATSPM direct supervised regression.

Instead of spatial imputation (using neighboring ATSPM), this approach
trains a model to predict ATSPM features directly from CV features
at the SAME segment. Uses paired (CV, ATSPM) observations for training.

Models: XGBoost, RandomForest, MLP (neural net)
Evaluation: Same held-out corridor-block nodes as GNN experiment.
"""

import os, json, time, warnings
import numpy as np
import torch
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATA_DIR = '/data2/NPTOI/arterial/imputation_network'
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_ATSPM = 15

def log(msg):
    print(msg, flush=True)


def main():
    log("Loading V3 tensor...")
    data = torch.load(os.path.join(DATA_DIR, 'data_tensor_v3.pt'), weights_only=True).numpy()
    mask = torch.load(os.path.join(DATA_DIR, 'mask_tensor_v3.pt'), weights_only=True).numpy()
    with open(os.path.join(DATA_DIR, 'split_info_v3.json')) as f:
        split = json.load(f)
    with open(os.path.join(DATA_DIR, 'feature_names_v2.json')) as f:
        features = json.load(f)

    N, T, F = data.shape
    atspm_names = features[:N_ATSPM]
    cv_names = features[N_ATSPM:]

    train_nodes = split['train_nodes']
    heldout_nodes = split['heldout_nodes']
    t_train = split['time_train_end']
    t_val = split['time_val_end']

    log(f"Data: ({N}, {T}, {F})")
    log(f"ATSPM features: {N_ATSPM}, CV features: {len(cv_names)}")
    log(f"Train nodes: {len(train_nodes)}, Held-out: {len(heldout_nodes)}")

    # ─── Build flat paired dataset ───
    # Only use samples where BOTH CV and ATSPM are observed
    log("\nBuilding paired (CV, ATSPM) dataset...")

    # Train data: train nodes × train time
    X_train_list, Y_train_list = [], []
    for n in train_nodes:
        for t in range(t_train):
            cv_mask = mask[n, t, N_ATSPM:]
            atspm_mask = mask[n, t, :N_ATSPM]
            # Need at least some CV features observed AND at least 1 ATSPM observed
            if cv_mask.sum() >= 5 and atspm_mask.sum() >= 1:
                X_train_list.append(data[n, t, N_ATSPM:])
                Y_train_list.append(data[n, t, :N_ATSPM])

    X_train = np.array(X_train_list, dtype=np.float32)
    Y_train = np.array(Y_train_list, dtype=np.float32)
    log(f"  Train samples: {len(X_train):,} (paired CV+ATSPM)")

    # Validation: train nodes × val time
    X_val_list, Y_val_list = [], []
    for n in train_nodes:
        for t in range(t_train, t_val):
            cv_mask = mask[n, t, N_ATSPM:]
            atspm_mask = mask[n, t, :N_ATSPM]
            if cv_mask.sum() >= 5 and atspm_mask.sum() >= 1:
                X_val_list.append(data[n, t, N_ATSPM:])
                Y_val_list.append(data[n, t, :N_ATSPM])

    X_val = np.array(X_val_list, dtype=np.float32) if X_val_list else np.zeros((0, len(cv_names)))
    Y_val = np.array(Y_val_list, dtype=np.float32) if Y_val_list else np.zeros((0, N_ATSPM))
    log(f"  Val samples: {len(X_val):,}")

    # Test: held-out nodes × test time
    X_test_list, Y_test_list, test_atspm_mask_list = [], [], []
    for n in heldout_nodes:
        for t in range(t_val, T):
            cv_mask = mask[n, t, N_ATSPM:]
            atspm_mask = mask[n, t, :N_ATSPM]
            if cv_mask.sum() >= 5:
                X_test_list.append(data[n, t, N_ATSPM:])
                Y_test_list.append(data[n, t, :N_ATSPM])
                test_atspm_mask_list.append(atspm_mask)

    X_test = np.array(X_test_list, dtype=np.float32)
    Y_test = np.array(Y_test_list, dtype=np.float32)
    test_atspm_mask = np.array(test_atspm_mask_list, dtype=bool)
    log(f"  Test samples: {len(X_test):,} (held-out nodes, test time)")

    # ─── Standardize ───
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_val_s = scaler_x.transform(X_val) if len(X_val) > 0 else X_val
    X_test_s = scaler_x.transform(X_test)

    # Handle NaN in X (missing CV features → fill with 0 after scaling)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0)
    Y_train = np.nan_to_num(Y_train, nan=0.0)

    # ─── Train Models ───
    models = {
        'XGBoost': MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       subsample=0.8, random_state=42),
            n_jobs=4
        ),
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=4)
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True,
            validation_fraction=0.1, random_state=42, learning_rate_init=0.001
        ),
    }

    all_results = {}

    for mname, model in models.items():
        log(f"\n{'='*60}")
        log(f"Training {mname}...")
        t0 = time.time()
        model.fit(X_train_s, Y_train)
        elapsed = time.time() - t0
        log(f"  Trained in {elapsed:.1f}s")

        # Predict
        Y_pred = model.predict(X_test_s)

        # Per-feature metrics (only where ATSPM is observed in test set)
        metrics = []
        for i, fname in enumerate(atspm_names):
            obs = test_atspm_mask[:, i]
            if obs.sum() == 0:
                metrics.append({'feature': fname, 'MAE': float('nan'), 'RMSE': float('nan'),
                                'MAPE': float('nan'), 'N_obs': 0})
                continue
            p = Y_pred[obs, i]
            t = Y_test[obs, i]
            mae = np.mean(np.abs(p - t))
            rmse = np.sqrt(np.mean((p - t) ** 2))
            nonzero = np.abs(t) > 1e-6
            mape = np.mean(np.abs(p[nonzero] - t[nonzero]) / np.abs(t[nonzero])) * 100 if nonzero.sum() > 0 else float('nan')
            gt_mean = np.mean(np.abs(t))
            rel_err = mae / gt_mean * 100 if gt_mean > 0 else float('nan')
            metrics.append({'feature': fname, 'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
                            'MAPE': round(mape, 2), 'RelError': round(rel_err, 1),
                            'N_obs': int(obs.sum()), 'GT_mean': round(gt_mean, 2)})
        all_results[mname] = metrics

    # ─── Load GNN results for comparison ───
    gnn_path = os.path.join(RESULTS_DIR, 'evaluation_v3.json')
    gnn_results = {}
    if os.path.exists(gnn_path):
        with open(gnn_path) as f:
            gnn_results = json.load(f)

    # ─── Print Comparison Table ───
    log(f"\n{'='*100}")
    log("FULL COMPARISON: GNN vs Direct Regression (MAE / Relative Error %)")
    log(f"{'='*100}")

    all_models = list(gnn_results.keys()) + list(all_results.keys())
    header = f"{'Feature':<25} {'GT_Mean':>8}"
    for m in all_models:
        header += f" {m:>14}"
    log(header)
    log("-" * len(header))

    for i, fname in enumerate(atspm_names):
        # Get GT mean from regression results
        gt_mean = '--'
        for mname in all_results:
            if all_results[mname][i]['N_obs'] > 0:
                gt_mean = f"{all_results[mname][i]['GT_mean']:>8.1f}"
                break

        row = f"{fname:<25} {gt_mean}"
        for m in all_models:
            if m in gnn_results:
                mae = gnn_results[m][i]['MAE']
                if np.isnan(mae):
                    row += f" {'N/A':>14}"
                else:
                    # Compute relative error
                    gt = all_results[list(all_results.keys())[0]][i].get('GT_mean', 1)
                    rel = mae / gt * 100 if gt > 0 else float('nan')
                    row += f" {mae:>7.1f}({rel:>4.0f}%)"
            else:
                mae = all_results[m][i]['MAE']
                if np.isnan(mae):
                    row += f" {'N/A':>14}"
                else:
                    rel = all_results[m][i].get('RelError', float('nan'))
                    row += f" {mae:>7.1f}({rel:>4.0f}%)"

        log(row)

    log(f"\n--- OVERALL AVERAGE MAE ---")
    for m in all_models:
        if m in gnn_results:
            maes = [r['MAE'] for r in gnn_results[m] if not np.isnan(r['MAE'])]
        else:
            maes = [r['MAE'] for r in all_results[m] if not np.isnan(r['MAE'])]
        log(f"  {m:>15}: {np.mean(maes):.2f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'evaluation_regression.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
