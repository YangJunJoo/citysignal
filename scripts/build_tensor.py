#!/usr/bin/env python3
"""
build_tensor.py - Build unified 3D tensor (N, T, F) + missingness mask
from matched CV-ATSPM parquet files for ST-GNN imputation experiments.

Output:
  /data2/NPTOI/arterial/imputation_network/
    ├── data_tensor.pt        # (N, T, F) float32
    ├── mask_tensor.pt        # (N, T, F) bool - True=observed, False=missing
    ├── node_mapping.csv      # Seg_Unique -> node_id
    ├── feature_names.json    # ordered feature column names
    └── time_index.csv        # ordered datetime index
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# ─── Config ───
MATCHED_DIR = '/data2/NPTOI/arterial/matched_metrics'
OUTPUT_DIR = '/data2/NPTOI/arterial/imputation_network'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select key numeric features (avoid sparse per-phase columns for now)
# We'll pick the top CV metrics + aggregate ATSPM metrics
CV_FEATURES = [
    'travel_speed_mph', 'free_flow_speed_mph', 'travel_time_s',
    'travel_control_delay', 'stop_delay_s', 'average_stop_count',
    'average_queue_m', 'AOG', 'split_failure',
    'through_RLR', 'left_RLR', 'through_dilemma', 'left_dilemma',
    'total_hard_acc', 'total_hard_bra', 'speeding_ratio',
    'segment_journeys'
]

# We'll dynamically detect ATSPM columns
ATSPM_PREFIX = 'ATSPM_'

def build_tensor():
    print("Step 1: Discovering matched files...")
    all_files = sorted(glob.glob(os.path.join(MATCHED_DIR, '*', '*.parquet')))
    print(f"  Found {len(all_files)} parquet files.")

    # Step 2: Collect all unique (Seg_Unique, datetime) combinations
    print("Step 2: Scanning for unique segments and time indices...")
    seg_date_data = defaultdict(dict)  # {seg_unique: {datetime: row_dict}}
    all_segs = set()
    all_times = set()
    atspm_cols_found = set()

    for i, f in enumerate(all_files):
        if i % 200 == 0:
            print(f"  Scanning [{i}/{len(all_files)}]...")
        try:
            df = pd.read_parquet(f)
            if 'Seg_Unique' not in df.columns or 'Time' not in df.columns:
                continue

            # Detect ATSPM columns
            file_atspm = [c for c in df.columns if c.startswith(ATSPM_PREFIX)]
            atspm_cols_found.update(file_atspm)

            for _, row in df.iterrows():
                seg = str(int(row['Seg_Unique'])) if pd.notna(row['Seg_Unique']) else None
                time_val = row['Time']
                if seg is None or pd.isna(time_val):
                    continue
                all_segs.add(seg)
                all_times.add(time_val)

                # Store CV features
                row_data = {}
                for col in CV_FEATURES:
                    if col in df.columns:
                        row_data[col] = row[col] if pd.notna(row[col]) else np.nan
                # Store ATSPM features
                for col in file_atspm:
                    row_data[col] = row[col] if pd.notna(row[col]) else np.nan

                seg_date_data[seg][time_val] = row_data
        except Exception as e:
            continue

    # Step 3: Build ordered indices
    print("Step 3: Building ordered indices...")
    seg_list = sorted(all_segs, key=lambda x: int(x))
    time_list = sorted(all_times)
    seg_to_idx = {s: i for i, s in enumerate(seg_list)}
    time_to_idx = {t: i for i, t in enumerate(time_list)}

    N = len(seg_list)
    T = len(time_list)

    # Select top ATSPM features (by non-NaN count) to keep tensor manageable
    print(f"  Total ATSPM columns discovered: {len(atspm_cols_found)}")
    # Count non-NaN occurrences for each ATSPM column
    atspm_counts = defaultdict(int)
    for seg, times in seg_date_data.items():
        for t, row_data in times.items():
            for col in atspm_cols_found:
                if col in row_data and not np.isnan(row_data[col]):
                    atspm_counts[col] += 1

    # Pick top 30 most populated ATSPM columns
    top_atspm = sorted(atspm_counts.keys(), key=lambda c: atspm_counts[c], reverse=True)[:30]
    print(f"  Selected top {len(top_atspm)} ATSPM features by coverage.")

    feature_names = CV_FEATURES + top_atspm
    F = len(feature_names)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    print(f"\n  Tensor dimensions: N={N} nodes, T={T} timesteps, F={F} features")

    # Step 4: Populate tensor
    print("Step 4: Populating tensor...")
    data = np.full((N, T, F), np.nan, dtype=np.float32)

    for seg, times in seg_date_data.items():
        n_idx = seg_to_idx[seg]
        for t, row_data in times.items():
            t_idx = time_to_idx[t]
            for col, val in row_data.items():
                if col in feat_to_idx:
                    data[n_idx, t_idx, feat_to_idx[col]] = val

    # Step 5: Create mask (True = observed)
    mask = ~np.isnan(data)

    # Replace NaN with 0 for the data tensor
    data = np.nan_to_num(data, nan=0.0)

    # Step 6: Save
    print("Step 5: Saving tensors...")
    torch.save(torch.tensor(data, dtype=torch.float32),
               os.path.join(OUTPUT_DIR, 'data_tensor.pt'))
    torch.save(torch.tensor(mask, dtype=torch.bool),
               os.path.join(OUTPUT_DIR, 'mask_tensor.pt'))

    # Save metadata
    pd.DataFrame({'Seg_Unique': seg_list, 'node_id': range(N)}).to_csv(
        os.path.join(OUTPUT_DIR, 'stgnn_node_mapping.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    pd.DataFrame({'time': time_list, 'time_idx': range(T)}).to_csv(
        os.path.join(OUTPUT_DIR, 'time_index.csv'), index=False)

    # Stats
    total_cells = N * T * F
    observed = mask.sum()
    print(f"\n{'='*50}")
    print(f"Tensor saved successfully!")
    print(f"  Shape: ({N}, {T}, {F})")
    print(f"  Observed: {observed:,} / {total_cells:,} ({observed/total_cells*100:.1f}%)")
    print(f"  Missing:  {total_cells - observed:,} ({(total_cells - observed)/total_cells*100:.1f}%)")
    print(f"  Size: {data.nbytes / 1e6:.1f} MB")
    print(f"  Output: {OUTPUT_DIR}")

if __name__ == '__main__':
    build_tensor()
