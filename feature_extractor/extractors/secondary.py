"""
Secondary feature extractors — vectorized.

These features are derived from already-extracted primary features:
- split_failure: Combines SPaT + occupancy data  
- conflict: Derives from headway data
- conflict_propensity: Combines volume + pedestrian activity

They operate on DataFrames rather than raw events, so the main speedup
is from eliminating redundant disk I/O (they no longer re-load from pkl files
since the DataFrames are passed directly).
"""

import pandas as pd
import numpy as np
import logging
from collections import Counter


def extract_split_failure(cache, df_spat: pd.DataFrame, df_occupancy: pd.DataFrame,
                          purdue_standard: bool = True) -> pd.DataFrame:
    """
    Extract split failure data by combining SPaT and occupancy.
    
    Split failure = when occupancy/green_duration ratio exceeds threshold.
    
    Parameters
    ----------
    cache : SignalDataCache
        Pre-loaded data cache
    df_spat : pd.DataFrame
        Already-extracted SPaT data
    df_occupancy : pd.DataFrame
        Already-extracted occupancy data
    purdue_standard : bool
        If True, threshold = 0.8; If False, threshold = 1.0
    
    Returns
    -------
    pd.DataFrame
        Split failure flags per cycle-phase-laneType
    """
    if len(df_occupancy) == 0 or len(df_spat) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'vehicle_traffic', 'split_failure')
        return result
    
    # Extract relevant columns
    spat_cols = [col for col in df_spat.columns if "greenDurationPhase" in col]
    green_occ_cols = [col for col in df_occupancy.columns if "greenAvgOccupancyPhase" in col]
    red5_occ_cols = [col for col in df_occupancy.columns if "redAvg5OccupancyPhase" in col]
    
    merge_keys = ["signalID", "cycleNo", "date", "cycleBegin", "cycleEnd", "cycleLength"]
    spat_subset = df_spat[merge_keys + spat_cols]
    occ_subset = df_occupancy[merge_keys + green_occ_cols + red5_occ_cols]
    
    result = pd.merge(spat_subset, occ_subset, on=merge_keys)
    
    # Find matching phases
    all_cols = spat_cols + green_occ_cols
    phase_nos = [col.split('Phase')[-1][0] for col in all_cols if 'Phase' in col]
    phase_counts = Counter(phase_nos)
    phase_nos = [p for p, count in phase_counts.items() if count > 1]
    
    threshold = 0.8 if purdue_standard else 1.0
    
    for phase_no in phase_nos:
        spat_col = next((col for col in spat_cols if phase_no in col), None)
        if not spat_col:
            continue
        
        for green_occ_col in green_occ_cols:
            if phase_no not in green_occ_col:
                continue
            
            lane_type = green_occ_col.split(f"Phase{phase_no}")[-1]
            out_col = f"greenSplitFailurePhase{phase_no}{lane_type}"
            
            # Vectorized: ratio = greenOccupancy / greenDuration
            ratio = result[green_occ_col] / result[spat_col].replace(0, 1)
            result[out_col] = (ratio >= threshold).astype(int)
        
        if purdue_standard:
            for red5_col in red5_occ_cols:
                if f"Phase{phase_no}" not in red5_col:
                    continue
                lane_type = red5_col.split(f"Phase{phase_no}")[-1]
                out_col = f"red5SplitFailurePhase{phase_no}{lane_type}"
                
                ratio = result[red5_col] / 5
                result[out_col] = (ratio >= threshold).astype(int)
    
    # Drop intermediate columns
    occ_cols_to_drop = spat_cols + green_occ_cols + (red5_occ_cols if purdue_standard else [])
    result = result.drop(columns=[c for c in occ_cols_to_drop if c in result.columns])
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'split_failure')
    
    logging.info(f"Split failure extracted: {len(result)} cycles")
    return result


def _extract_conflict_from(cache, df_source: pd.DataFrame,
                           source_keyword: str, conflict_prefix: str,
                           feature_name: str,
                           thresholds: np.ndarray = np.arange(0.5, 5.5, 0.5)) -> pd.DataFrame:
    """
    Generic conflict extraction from any time-interval data (headway or PET).
    
    Conflict = count of time-interval values below threshold.
    
    Parameters
    ----------
    cache : SignalDataCache
        Pre-loaded data cache
    df_source : pd.DataFrame
        Source data (headway or PET DataFrame)
    source_keyword : str
        Column keyword to match (e.g. 'Headway' or 'PET')
    conflict_prefix : str
        Prefix for output columns (e.g. 'ConflictHeadway' or 'ConflictPET')
    feature_name : str
        Feature name for cache saving (e.g. 'conflict_headway' or 'conflict_pet')
    thresholds : array-like
        Time thresholds for conflict counting (seconds)
    
    Returns
    -------
    pd.DataFrame
        Conflict counts per cycle-phase-laneType-threshold
    """
    if len(df_source) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'vehicle_traffic', feature_name)
        return result
    
    # Find source columns with object dtype (lists of values)
    source_cols = [col for col in df_source.columns if source_keyword in col and df_source[col].dtype == 'O']
    
    if not source_cols:
        result = df_source[["signalID", "cycleNo", "cycleBegin", "cycleEnd", "cycleLength", "date"]].copy()
        cache.save_feature(result, 'vehicle_traffic', feature_name)
        return result
    
    # Count conflicts vectorized
    conflict_dfs = []
    
    for col in source_cols:
        for threshold in thresholds:
            conflict_col = col.replace(source_keyword, f"{conflict_prefix}{threshold}")
            
            def count_below_threshold(vals, t=threshold):
                if not isinstance(vals, list):
                    return 0
                # Flatten nested lists
                flat = []
                for v in vals:
                    if isinstance(v, list):
                        flat.extend(v)
                    else:
                        flat.append(v)
                return sum(1 for v in flat if isinstance(v, (int, float)) and not np.isnan(v) and 0 < v < t)
            
            conflict_dfs.append((conflict_col, df_source[col].apply(count_below_threshold)))
    
    # Build result
    meta_cols = ["signalID", "cycleNo", "cycleBegin", "cycleEnd", "cycleLength", "date"]
    result = df_source[meta_cols].copy()
    
    for col_name, series in conflict_dfs:
        result[col_name] = series
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', feature_name)
    
    logging.info(f"{feature_name} extracted: {len(result)} cycles")
    return result


def extract_conflict_headway(cache, df_headway: pd.DataFrame,
                              thresholds: np.ndarray = np.arange(0.5, 5.5, 0.5)) -> pd.DataFrame:
    """
    Extract Conflict_Headway: count of headway values below threshold.
    Uses front (stopbar) detector headway data.
    """
    return _extract_conflict_from(
        cache, df_headway,
        source_keyword='Headway', conflict_prefix='ConflictHeadway',
        feature_name='conflict_headway', thresholds=thresholds
    )


def extract_conflict_pet(cache, df_pet: pd.DataFrame,
                          thresholds: np.ndarray = np.arange(0.5, 5.5, 0.5)) -> pd.DataFrame:
    """
    Extract Conflict_PET: count of PET (Post-Encroachment Time) values below threshold.
    Uses back (setback) detector PET data.
    """
    return _extract_conflict_from(
        cache, df_pet,
        source_keyword='PET', conflict_prefix='ConflictPET',
        feature_name='conflict_pet', thresholds=thresholds
    )


# Backward compatibility alias
def extract_conflict(cache, df_headway: pd.DataFrame,
                     thresholds: np.ndarray = np.arange(0.5, 5.5, 0.5)) -> pd.DataFrame:
    """Backward-compatible alias for extract_conflict_headway."""
    return extract_conflict_headway(cache, df_headway, thresholds)


def extract_conflict_propensity(cache, df_volume: pd.DataFrame = None,
                                 df_pedestrian_activity: pd.DataFrame = None) -> pd.DataFrame:
    """
    Extract vehicle-pedestrian conflict propensity.
    
    Combines right-turn vehicle volume with pedestrian activity to identify
    potential conflict situations.
    
    Parameters
    ----------
    cache : SignalDataCache
        Pre-loaded data cache
    df_volume : pd.DataFrame, optional
        Already-extracted volume data
    df_pedestrian_activity : pd.DataFrame, optional
        Already-extracted pedestrian activity data
    
    Returns
    -------
    pd.DataFrame
        Conflict propensity data per cycle
    """
    # Missing measures are not a valid zero-count result. In particular, the
    # current activity extractor's event-code-specific column names must not be
    # guessed into this legacy calculation's different measure contract.
    def missing_inputs(message):
        result = pd.DataFrame()
        result.attrs['reason_code'] = 'missing_metric_inputs'
        result.attrs['message'] = message
        cache.save_feature(result, 'vehicle_traffic', 'conflict_propensity')
        return result

    if df_volume is None or df_pedestrian_activity is None:
        return missing_inputs('Volume and pedestrian activity inputs are required.')
    if df_volume.empty or df_pedestrian_activity.empty:
        return missing_inputs('Volume or pedestrian activity input is empty.')

    right_turn_cols = [col for col in df_volume.columns
                       if 'VolumePhase' in col and 'R' in col.split('Phase')[-1]]
    ped_activity_cols = [col for col in df_pedestrian_activity.columns
                         if 'pedestrianActivityPhase' in col]
    if not right_turn_cols or not ped_activity_cols:
        missing = []
        if not right_turn_cols:
            missing.append('right-turn volume measures')
        if not ped_activity_cols:
            missing.append('pedestrianActivityPhase measures')
        return missing_inputs('Missing metric inputs: ' + ', '.join(missing))

    merge_keys = ['signalID', 'cycleNo']
    for label, frame in [('volume', df_volume), ('pedestrian activity', df_pedestrian_activity)]:
        missing = set(merge_keys + ['cycleBegin', 'cycleEnd']) - set(frame.columns)
        if missing:
            raise ValueError(f'{label} input is missing cycle metadata: {sorted(missing)}')

    result = pd.merge(df_volume, df_pedestrian_activity, on=merge_keys,
                      how='outer', suffixes=('_vol', '_ped'), validate='one_to_one')
    # These fields describe the same signal/cycle. Coalescing is valid only if
    # both sources agree wherever they provide a value; never select one side
    # silently when its interval conflicts with the other source.
    for column in ['cycleBegin', 'cycleEnd', 'cycleLength', 'date']:
        volume_column, pedestrian_column = column + '_vol', column + '_ped'
        if volume_column not in result or pedestrian_column not in result:
            continue
        volume_values, pedestrian_values = result[volume_column], result[pedestrian_column]
        both_present = volume_values.notna() & pedestrian_values.notna()
        if not volume_values[both_present].eq(pedestrian_values[both_present]).all():
            raise ValueError(f'Conflicting {column} metadata for the same signal/cycle')
        result[column] = volume_values.combine_first(pedestrian_values)
        result = result.drop(columns=[volume_column, pedestrian_column])

    # Create conflict propensity flags
    for rt_col in right_turn_cols:
        phase_suffix = rt_col.split('Volume')[-1]
        for pa_col in ped_activity_cols:
            cp_col = f"conflictPropensity{phase_suffix}_{pa_col.split('Phase')[-1]}"
            if rt_col in result.columns and pa_col in result.columns:
                result[cp_col] = ((result[rt_col] > 0) & (result[pa_col] > 0)).astype(int)
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'conflict_propensity')
    
    logging.info(f"Conflict propensity extracted: {len(result)} cycles")
    return result
