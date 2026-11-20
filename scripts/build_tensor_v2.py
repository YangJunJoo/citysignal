#!/usr/bin/env python3
"""
build_tensor_v2.py — Build Phase-agnostic ATSPM tensor (N, T, F)
OPTIMIZED: Pre-computes column→feature mapping per file, then uses
vectorized pandas operations instead of per-row regex.
"""

import os, re, glob, json, ast
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

MATCHED_DIR = '/data2/NPTOI/arterial/matched_metrics'
OUTPUT_DIR = '/data2/NPTOI/arterial/imputation_network'
MAPPING_FILE = '/home/ya195253/YJ/python_scripts/ATSPM/data/ATSPM_CV_matched.csv'
BASEMAP_FILE = '/data2/NPTOI/arterial/basemap/OS_Primary_Basemap_segments.shp'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CV_FEATURES = [
    'travel_speed_mph', 'free_flow_speed_mph', 'travel_time_s',
    'travel_control_delay', 'stop_delay_s', 'average_stop_count',
    'average_queue_m', 'AOG', 'split_failure',
    'through_RLR', 'left_RLR', 'through_dilemma', 'left_dilemma',
    'total_hard_acc', 'total_hard_bra', 'speeding_ratio',
    'segment_journeys'
]

THROUGH_SUFFIXES = {'T', 'TR', 'R', ''}
LEFT_SUFFIXES = {'L', 'LT'}

# Output feature names
ATSPM_FEATURES = [
    'AvgHeadway_Through', 'AvgHeadway_Left',
    'Volume_Through', 'Volume_Left',
    'SplitFailure_Through', 'SplitFailure_Left',
    'RedRunning_Through', 'RedRunning_Left',
    'DilemmaRunning_Through', 'DilemmaRunning_Left',
    'ConflictHeadway_Through', 'ConflictHeadway_Left',
    'ConflictPET_Through', 'ConflictPET_Left',
    'QueueLength_Through',
]

ALL_FEATURES = ATSPM_FEATURES + CV_FEATURES
F = len(ALL_FEATURES)
FEAT_IDX = {f: i for i, f in enumerate(ALL_FEATURES)}

# Metric patterns: maps ATSPM column base patterns → (output_metric, agg_method)
METRIC_RULES = [
    (re.compile(r'(green|yellow|red|redClearance)AvgHeadway'), 'AvgHeadway', 'mean'),
    (re.compile(r'[vV]olume'), 'Volume', 'sum'),
    (re.compile(r'(green|red5)SplitFailure'), 'SplitFailure', 'max'),
    (re.compile(r'(red|redClearance)RunningFlag'), 'RedRunning', 'max'),
    (re.compile(r'yellowRunningFlag'), 'DilemmaRunning', 'max'),
    (re.compile(r'(green|yellow|red|redClearance)Conflict1\.0'), 'ConflictHeadway', 'sum'),
    # ConflictPET: columns named like greenConflictPET1.0Phase...
    # But current data may not have these yet. Check for any ConflictPET pattern
    (re.compile(r'(green|yellow|red|redClearance)ConflictPET1\.0'), 'ConflictPET', 'sum'),
    (re.compile(r'(L_max|queueLength)'), 'QueueLength', 'mean'),
]

PHASE_REGEX = re.compile(r'Phase(\d+)(.*?)(?:_[xy])?$')


def load_mapping():
    mapping = pd.read_csv(MAPPING_FILE)
    seg_phase = {}
    for _, row in mapping.iterrows():
        seg = str(int(row['Seg_Unique']))
        through, left = set(), set()
        for col in ['Through', 'Through Right']:
            try: through.update(ast.literal_eval(str(row[col])))
            except: pass
        for col in ['Left', 'Left Through']:
            try: left.update(ast.literal_eval(str(row[col])))
            except: pass
        seg_phase[seg] = {'through': through, 'left': left}
    return seg_phase


def classify_columns(atspm_cols, seg_phase_info):
    """
    Pre-classify each ATSPM column → (output_feature, agg_method) for a given segment.
    Returns dict: {output_feature: [(col_name, agg_method), ...]}
    """
    col_map = defaultdict(list)
    through_phases = seg_phase_info.get('through', set())
    left_phases = seg_phase_info.get('left', set())

    for col in atspm_cols:
        base = col.replace('ATSPM_', '')
        # Find phase number and suffix
        pm = PHASE_REGEX.search(base)
        if not pm:
            continue
        phase_no = int(pm.group(1))
        lane_suffix = pm.group(2)

        # Determine which metric this column belongs to
        metric_name, agg_method = None, None
        for pattern, mname, magg in METRIC_RULES:
            if pattern.search(base):
                metric_name, agg_method = mname, magg
                break
        if metric_name is None:
            continue

        # Map to Through or Left based on phase
        if phase_no in through_phases and lane_suffix in THROUGH_SUFFIXES:
            out_feat = f'{metric_name}_Through'
            col_map[out_feat].append((col, agg_method))
        if phase_no in left_phases and lane_suffix in LEFT_SUFFIXES:
            if metric_name != 'QueueLength':
                out_feat = f'{metric_name}_Left'
                col_map[out_feat].append((col, agg_method))

    return col_map


def aggregate_row_vectorized(df, col_map):
    """
    Aggregate ATSPM columns for entire DataFrame at once.
    Returns DataFrame with aggregated columns.
    """
    result = pd.DataFrame(index=df.index)
    for out_feat, col_agg_list in col_map.items():
        cols = [c for c, _ in col_agg_list]
        agg_method = col_agg_list[0][1]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            result[out_feat] = np.nan
            continue
        sub = df[existing].apply(pd.to_numeric, errors='coerce')
        if agg_method == 'mean':
            result[out_feat] = sub.mean(axis=1)
        elif agg_method == 'sum':
            result[out_feat] = sub.sum(axis=1, min_count=1)
        elif agg_method == 'max':
            result[out_feat] = sub.max(axis=1)
    return result


def build_corridor_heldout(seg_list, matched_segs, target_ratio=0.20):
    import geopandas as gpd
    segs_gdf = gpd.read_file(BASEMAP_FILE)
    segs_gdf['Seg_Unique'] = segs_gdf['Seg_Unique'].astype(str)
    observed = set(seg_list) & matched_segs
    road_groups = segs_gdf[segs_gdf['Seg_Unique'].isin(observed)].groupby('Road_name')

    held_out = set()
    target = int(len(observed) * target_ratio)
    np.random.seed(42)
    roads = list(road_groups.groups.keys())
    np.random.shuffle(roads)

    for road in roads:
        if len(held_out) >= target:
            break
        rsegs = road_groups.get_group(road)['Seg_Unique'].tolist()
        robs = [s for s in rsegs if s in observed]
        if len(robs) < 5:
            continue
        bsz = min(np.random.randint(3, 6), len(robs) - 1)
        si = np.random.randint(0, len(robs) - bsz + 1)
        held_out.update(robs[si:si+bsz])

    print(f"  Held-out: {len(held_out)}/{len(observed)} ({len(held_out)/len(observed)*100:.1f}%)")
    return held_out


def build_tensor():
    print("Step 1: Loading mapping...")
    seg_phase = load_mapping()
    matched_segs = set(seg_phase.keys())
    print(f"  Matched: {len(matched_segs)}")

    # Pre-classify columns per segment
    # First, collect all unique ATSPM column sets
    print("\nStep 2: Scanning files (vectorized)...")
    all_files = sorted(glob.glob(os.path.join(MATCHED_DIR, '*', '*.parquet')))
    print(f"  Files: {len(all_files)}")

    seg_time_data = {}  # (seg, time) → feature_values array
    all_segs = set()
    all_times = set()

    for fi, fpath in enumerate(all_files):
        if fi % 100 == 0:
            print(f"  [{fi}/{len(all_files)}]...")
        try:
            df = pd.read_parquet(fpath)
            if 'Seg_Unique' not in df.columns or 'Time' not in df.columns:
                continue
            if len(df) == 0:
                continue

            # Get unique segments in this file
            df['_seg'] = df['Seg_Unique'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
            df = df.dropna(subset=['_seg', 'Time'])

            atspm_cols = [c for c in df.columns if c.startswith('ATSPM_')]

            for seg_id, seg_df in df.groupby('_seg'):
                all_segs.add(seg_id)

                # ATSPM aggregation (vectorized per segment-group)
                atspm_result = pd.DataFrame(index=seg_df.index)
                if seg_id in seg_phase and len(atspm_cols) > 0:
                    col_map = classify_columns(atspm_cols, seg_phase[seg_id])
                    atspm_result = aggregate_row_vectorized(seg_df, col_map)

                # CV features
                cv_result = pd.DataFrame(index=seg_df.index)
                for cv in CV_FEATURES:
                    if cv in seg_df.columns:
                        cv_result[cv] = pd.to_numeric(seg_df[cv], errors='coerce')
                    else:
                        cv_result[cv] = np.nan

                # Store per (seg, time)
                for idx, row in seg_df.iterrows():
                    t = row['Time']
                    all_times.add(t)
                    key = (seg_id, t)
                    vals = np.full(F, np.nan, dtype=np.float32)
                    for feat in ATSPM_FEATURES:
                        if feat in atspm_result.columns:
                            v = atspm_result.at[idx, feat]
                            if pd.notna(v):
                                vals[FEAT_IDX[feat]] = v
                    for feat in CV_FEATURES:
                        if feat in cv_result.columns:
                            v = cv_result.at[idx, feat]
                            if pd.notna(v):
                                vals[FEAT_IDX[feat]] = v
                    seg_time_data[key] = vals

        except Exception as e:
            continue

    print(f"\nStep 3: Building indices...")
    seg_list = sorted(all_segs, key=lambda x: int(x))
    time_list = sorted(all_times)
    seg_idx = {s: i for i, s in enumerate(seg_list)}
    time_idx = {t: i for i, t in enumerate(time_list)}
    N, T = len(seg_list), len(time_list)
    print(f"  ({N}, {T}, {F})")

    print("Step 4: Populating tensor...")
    data = np.full((N, T, F), np.nan, dtype=np.float32)
    for (seg, t), vals in seg_time_data.items():
        data[seg_idx[seg], time_idx[t]] = vals
    mask = ~np.isnan(data)
    data = np.nan_to_num(data, nan=0.0)

    print("Step 5: Held-out selection...")
    held_out = build_corridor_heldout(seg_list, matched_segs)
    ho_idx = sorted([seg_idx[s] for s in held_out if s in seg_idx])
    tr_idx = sorted([seg_idx[s] for s in seg_list if s not in held_out and s in matched_segs])
    ms_idx = sorted([seg_idx[s] for s in seg_list if s not in matched_segs])

    nt_train = int(T * 0.70)
    nt_val = int(T * 0.85)

    print("Step 6: Saving...")
    torch.save(torch.tensor(data, dtype=torch.float32), os.path.join(OUTPUT_DIR, 'data_tensor_v3.pt'))
    torch.save(torch.tensor(mask, dtype=torch.bool), os.path.join(OUTPUT_DIR, 'mask_tensor_v3.pt'))

    with open(os.path.join(OUTPUT_DIR, 'feature_names_v3.json'), 'w') as f:
        json.dump(ALL_FEATURES, f)
    with open(os.path.join(OUTPUT_DIR, 'split_info_v3.json'), 'w') as f:
        json.dump({'train_nodes': tr_idx, 'heldout_nodes': ho_idx, 'missing_nodes': ms_idx,
                   'time_train_end': nt_train, 'time_val_end': nt_val,
                   'time_total': T, 'N': N, 'T': T, 'F': F}, f)

    pd.DataFrame({'Seg_Unique': seg_list, 'node_id': range(N)}).to_csv(
        os.path.join(OUTPUT_DIR, 'stgnn_node_mapping_v3.csv'), index=False)
    pd.DataFrame({'time': time_list, 'time_idx': range(T)}).to_csv(
        os.path.join(OUTPUT_DIR, 'time_index_v3.csv'), index=False)

    am = mask[:, :, :len(ATSPM_FEATURES)]
    cm = mask[:, :, len(ATSPM_FEATURES):]
    print(f"\n{'='*60}")
    print(f"Tensor: ({N}, {T}, {F})")
    print(f"  ATSPM ({len(ATSPM_FEATURES)}): {am.mean()*100:.1f}% observed")
    print(f"  CV ({len(CV_FEATURES)}): {cm.mean()*100:.1f}% observed")
    print(f"  Train/Heldout/Missing nodes: {len(tr_idx)}/{len(ho_idx)}/{len(ms_idx)}")
    print(f"  Time train/val/test: {nt_train}/{nt_val-nt_train}/{T-nt_val}")


if __name__ == '__main__':
    build_tensor()
