
from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame

#!/usr/bin/env python3
# match_cv_atspm_proportional.py

import os
import glob
import pandas as pd
import numpy as np
from datetime import timedelta
import ast
from zoneinfo import ZoneInfo
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def load_atspm_mapping(mapping_path):
    mapping = pd.read_csv(mapping_path)
    mapping['Phases'] = mapping['Phases'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
    )
    return mapping

def load_atspm_features_direct(atspm_base_dir, signal_id, date_str):
    """
    Load ATSPM features using DIRECT path. 
    It checks all feature subdirectories and merges them on cycle definitions.
    
    NOTE: Parquet files retain the UTC source-partition date but contain Eastern wall-clock time
    timestamps, so each EDT day is split across two Parquet files (date_str and
    the next day). We load both and filter to the target EDT date.
    """
    features = ['conflict', 'conflict_propensity', 'dilemma_running', 'gap', 
                'headway', 'occupancy', 'pet', 'red_running', 'shockwave', 
                'split_failure', 'volume']
    
    # Determine which pkl dates to load for the target EDT date:
    # EDT day X is covered by MTN Parquet files for date X (hours 0-1 EDT)
    # and date X+1 (hours 2-23 EDT, which is MTN day X+1)
    target_date = pd.Timestamp(date_str)
    next_date_str = (target_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    source_dates = [date_str, next_date_str]
    
    merged_df = None
    merge_keys = ['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd']
    
    for feat in features:
        feat_dfs = []
        for d in source_dates:
            feature_path = os.path.join(atspm_base_dir, feat, signal_id, f"{d}.parquet")
            if os.path.exists(feature_path):
                try:
                    df = read_frame(feature_path)
                    if not df.empty:
                        feat_dfs.append(df)
                except Exception:
                    continue
        
        if not feat_dfs:
            continue
        
        feat_df = pd.concat(feat_dfs, ignore_index=True) if len(feat_dfs) > 1 else feat_dfs[0]
        cols_to_use = [c for c in feat_df.columns if c not in ['cycleLength']] 
        feat_df = feat_df[cols_to_use]
        
        # Deduplicate safely across days
        if 'cycleNo' in feat_df.columns and 'date' in feat_df.columns and 'signalID' in feat_df.columns:
            feat_df = feat_df.drop_duplicates(subset=['signalID', 'date', 'cycleNo'], keep='first')
        
        if merged_df is None:
            merged_df = feat_df
        else:
            try:
                feat_df = feat_df.drop(columns=['cycleBegin', 'cycleEnd'], errors='ignore')
                merged_df = pd.merge(merged_df, feat_df, on=['signalID', 'date', 'cycleNo'], how='outer')
            except Exception:
                pass
    
    if merged_df is not None and 'cycleBegin' in merged_df.columns:
        # Filter to only cycles within the target EDT date
        merged_df['cycleBegin'] = pd.to_datetime(merged_df['cycleBegin'])
        edt_start = pd.Timestamp(date_str)
        edt_end = edt_start + pd.Timedelta(days=1)
        merged_df = merged_df[(merged_df['cycleBegin'] >= edt_start) & 
                              (merged_df['cycleBegin'] < edt_end)]
        if merged_df.empty:
            return pd.DataFrame()
    
    return merged_df if merged_df is not None else pd.DataFrame()

def split_cycle_to_15min_bins(atspm_df):
    """
    Distributes cycle metrics proportionally across 15-minute boundaries based on overlap.
    """
    results = []
    
    for _, row in atspm_df.iterrows():
        cycle_begin = row['cycleBegin']
        cycle_end = row['cycleEnd']
        
        # Round down to nearest 15 mins for the begin time
        current_bin_start = cycle_begin.floor('15min')
        total_duration = (cycle_end - cycle_begin).total_seconds()
        
        if total_duration <= 0:
            continue
            
        while current_bin_start < cycle_end:
            current_bin_end = current_bin_start + pd.Timedelta(minutes=15)
            
            overlap_start = max(cycle_begin, current_bin_start)
            overlap_end = min(cycle_end, current_bin_end)
            overlap_duration = (overlap_end - overlap_start).total_seconds()
            
            if overlap_duration > 0:
                weight = overlap_duration / total_duration
                
                new_row = row.copy()
                new_row['Time_15min'] = current_bin_start
                new_row['overlap_weight'] = weight
                results.append(new_row)
            
            current_bin_start = current_bin_end
            
    return pd.DataFrame(results)

def process_cv_file(cv_file, mapping_df, atspm_base_dir, output_dir):
    """Process a single CV file to be run by ProcessPoolExecutor."""
    try:
        print(f"Processing: {cv_file}", flush=True)
        seg_dir_name = os.path.basename(os.path.dirname(cv_file)) # e.g. '101_25_ST'
        date_str = os.path.basename(cv_file).replace('.parquet', '') # e.g. '2024-05-02'
        
        cv_df = pd.read_parquet(cv_file)
        if cv_df.empty:
            return f"[{seg_dir_name}/{date_str}] Skipped: Empty file."
            
        if cv_df['Time'].dt.tz is not None:
            cv_df['Time'] = cv_df['Time'].dt.tz_localize(None)
            
        unique_segs = cv_df['Seg_Unique'].unique()
        matched_results = []
        
        for seg_unique in unique_segs:
            cv_sub_df = cv_df[cv_df['Seg_Unique'] == seg_unique].copy()
            
            # 1. Match specific CV Seg_Unique to ATSPM SignalID
            try:
                seg_unique_int = int(seg_unique)
            except ValueError:
                continue
                
            signal_info = mapping_df[mapping_df['Seg_Unique'] == seg_unique_int]
            if signal_info.empty:
                continue
                
            signal_id = str(signal_info.iloc[0]['signalID'])
            
            # 2. Optimized ATSPM File search (DIRECT LOAD)
            atspm_df = load_atspm_features_direct(atspm_base_dir, signal_id, date_str)
            
            if atspm_df.empty or 'cycleBegin' not in atspm_df.columns:
                matched_results.append(cv_sub_df)
                continue
                
            atspm_df['cycleBegin'] = pd.to_datetime(atspm_df['cycleBegin'])
            atspm_df['cycleEnd'] = pd.to_datetime(atspm_df['cycleEnd'])
            
            weighted_atspm = split_cycle_to_15min_bins(atspm_df)
            
            if weighted_atspm.empty:
                matched_results.append(cv_sub_df)
                continue
                
            # Prefix all numerical columns from ATSPM with 'ATSPM_' to avoid collisions with CV dataset
            numeric_cols = weighted_atspm.select_dtypes(include=[np.number]).columns.drop(['overlap_weight', 'cycleLength', 'cycleNo'], errors='ignore')
            
            # Apply proportional weighting
            for col in numeric_cols:
                weighted_atspm[col] = weighted_atspm[col] * weighted_atspm['overlap_weight']
                
            # Aggregate by 15min bin
            agg_funcs = {col: 'sum' for col in numeric_cols}
            atspm_15m = weighted_atspm.groupby('Time_15min').agg(agg_funcs).reset_index()
            
            # Rename the agg columns to have ATSPM_ prefix
            rename_dict = {col: f"ATSPM_{col}" for col in numeric_cols}
            atspm_15m = atspm_15m.rename(columns=rename_dict)
            
            # Left join on Time
            matched_sub = pd.merge(cv_sub_df, atspm_15m, left_on='Time', right_on='Time_15min', how='left')
            matched_sub.drop(columns=['Time_15min'], inplace=True, errors='ignore')
            
            matched_results.append(matched_sub)
            
        if matched_results:
            final_matched_df = pd.concat(matched_results, ignore_index=True)
            out_path = os.path.join(output_dir, seg_dir_name)
            os.makedirs(out_path, exist_ok=True)
            final_matched_df.to_parquet(os.path.join(out_path, f"{date_str}.parquet"))
            return f"[{seg_dir_name}/{date_str}] Successfully matched and saved."
            
        return f"[{seg_dir_name}/{date_str}] Processed but no valid mappings found."
        
    except Exception as e:
        return f"[{os.path.basename(cv_file)}] Error: {str(e)}"

def process_matching():
    cv_base_dir = '/data2/NPTOI/arterial/metrics'
    mapping_path = '/home/ya195253/YJ/python_scripts/ATSPM/data/ATSPM_CV_matched.csv'
    atspm_base_dir = '/data2/ATSPM/processed_data/fdot_d5/feature_extraction/feature/cycle/vehicle_traffic'
    output_dir = '/data2/NPTOI/arterial/matched_metrics'
    max_workers = 4
    
    os.makedirs(output_dir, exist_ok=True)
    mapping_df = load_atspm_mapping(mapping_path)
    
    print("Mapping loaded. Discovering CV segment files...")
    cv_files = glob.glob(os.path.join(cv_base_dir, '*', '*.parquet'))
    print(f"Total CV Files found: {len(cv_files)}", flush=True)
    
    worker_func = partial(process_cv_file, mapping_df=mapping_df, atspm_base_dir=atspm_base_dir, output_dir=output_dir)
    
    success_count = 0
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as executor:
        futures = {executor.submit(worker_func, cv_f): cv_f for cv_f in cv_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if i % 10 == 0 or "Error" in result:
                    print(f"[{i}/{len(cv_files)}] {result}", flush=True)
                if "Successfully" in result:
                    success_count += 1
            except Exception as e:
                print(f"Worker crashed with error: {e}", flush=True)
                
    print("==========================================")
    print(f"Done! Successfully generated matched outputs: {success_count} / {len(cv_files)}")
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    process_matching()
