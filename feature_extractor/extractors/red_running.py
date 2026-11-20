"""
Red running and dilemma running feature extractors — vectorized.

red_running: Detects vehicles passing through during red clearance or red signal.
dilemma_running: Detects vehicles passing through during yellow signal.

Both are binary flags per (cycle, phase, laneType).
Uses searchsorted cycle assignment + interval join instead of per-cycle loops.
"""

import pandas as pd
import numpy as np
import logging
from ..profiles import before_interval_end


def extract_red_running(cache, with_countbar: bool = False) -> pd.DataFrame:
    """
    Extract red light running data matching v1 output exactly.
    Added RLR_crossing and detectorOffAfterRedClearEnd metrics.
    Fully vectorized — no per-cycle Python loops.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "count" if with_countbar else "front"
    event_seq = (82, 81)
    
    df_ev = cache.assign_cycle_index(
        event_sequence=event_seq, detector_type=detector_type, tolerance='exact'
    )
    
    if len(df_ev) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'red_running')
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_ev)
    df_ev = df_ev[df_ev['cycle_idx'] >= 0].copy()
    
    if len(df_ev) == 0 or len(phase_nos) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'red_running')
        return result
    
    # Get signal type intervals for redClearance and red only
    df_intervals = cache.build_signal_type_intervals()
    df_intervals_rr = df_intervals[
        (df_intervals['signal_type'].isin(['redClearance', 'red'])) &
        (df_intervals['phase_no'].isin(phase_nos))
    ].copy()

    # Pre-calculate Red Clearance End for each cycle/phase interval
    df_intervals_rr['red_clearance_end'] = np.where(
        df_intervals_rr['signal_type'] == 'redClearance',
        df_intervals_rr['interval_end'],
        df_intervals_rr['interval_start']
    )

    # Unique identifier for each interval row
    df_intervals_rr['interval_id'] = range(len(df_intervals_rr))

    # Intersecting phases presence check
    df_intervals_rr['intersecting_group'] = np.where(df_intervals_rr['phase_no'].isin([1, 2, 5, 6]), 1, 
                                            np.where(df_intervals_rr['phase_no'].isin([3, 4, 7, 8]), 2, 0))

    df_ev_81 = df_ev[df_ev['eventCode'] == 81].copy()
    df_ev_81['event_phase_group'] = np.where(df_ev_81['phaseNo'].isin([1, 2, 5, 6]), 1,
                                    np.where(df_ev_81['phaseNo'].isin([3, 4, 7, 8]), 2, 0))

    df_merged_int = pd.merge(
        df_intervals_rr[['interval_id', 'cycle_idx', 'phase_no', 'signal_type', 'interval_start', 'interval_end', 'intersecting_group']],
        df_ev_81[['cycle_idx', 'phaseNo', 'timeStamp', 'event_phase_group']],
        on='cycle_idx'
    )

    mask_int = (df_merged_int['intersecting_group'] + df_merged_int['event_phase_group'] == 3) & \
               (df_merged_int['intersecting_group'] > 0) & \
               (df_merged_int['timeStamp'] >= df_merged_int['interval_start']) & \
               before_interval_end(cache, df_merged_int['timeStamp'], df_merged_int['interval_end'])

    intersecting_ids = set(df_merged_int.loc[mask_int, 'interval_id'].unique())

    df_intervals_rr['has_intersecting'] = df_intervals_rr['interval_id'].isin(intersecting_ids)

    # Pair det_on and det_off vectorially
    # To match the sequence [82, 81] logic of V1/V2:
    # A valid sequence has 82 first, then 81.
    # Therefore, the det_off (82) is the event immediately PRECEDING the det_on (81).
    df_ev = df_ev.sort_values(['cycle_idx', 'phaseNo', 'channelNo', 'timeStamp']).reset_index(drop=True)
    
    grp_cols = ['cycle_idx', 'phaseNo', 'channelNo']
    df_ev['prev_eventCode'] = df_ev.groupby(grp_cols)['eventCode'].shift(1)
    df_ev['prev_timeStamp'] = df_ev.groupby(grp_cols)['timeStamp'].shift(1)
    
    df_arrivals = df_ev[df_ev['eventCode'] == 81].copy()
    df_arrivals['det_off'] = np.where(df_arrivals['prev_eventCode'] == 82, df_arrivals['prev_timeStamp'], pd.NaT)
    df_arrivals['det_off'] = pd.to_datetime(df_arrivals['det_off'])
    
    df_merged = pd.merge(
        df_arrivals[['cycle_idx', 'phaseNo', 'channelNo', 'laneType', 'timeStamp', 'det_off']],
        df_intervals_rr,
        left_on=['cycle_idx', 'phaseNo'],
        right_on=['cycle_idx', 'phase_no'],
        how='inner'
    )
    
    if with_countbar:
        mask_rlr = (df_merged['timeStamp'] >= df_merged['interval_start']) & \
                   before_interval_end(cache, df_merged['det_off'], df_merged['interval_end']) & \
                   (df_merged['det_off'].notna())
    else:
        mask_rlr = (df_merged['timeStamp'] >= df_merged['interval_start']) & \
                   before_interval_end(cache, df_merged['timeStamp'], df_merged['interval_end'])
                   
    df_rlr = df_merged[mask_rlr].copy()
    
    if len(df_rlr) > 0:
        df_rlr['diff_sec'] = (df_rlr['det_off'] - df_rlr['red_clearance_end']).dt.total_seconds().round(4)
        gp_flags = df_rlr.groupby(['cycle_idx', 'phaseNo', 'laneType', 'signal_type']).agg(
            is_crossing=('has_intersecting', 'any')
        ).reset_index()
        gp_diff = df_rlr.groupby(['cycle_idx', 'phaseNo', 'laneType'])['diff_sec'].max().reset_index()
    else:
        gp_flags = pd.DataFrame(columns=['cycle_idx', 'phaseNo', 'laneType', 'signal_type', 'is_crossing'])
        gp_diff = pd.DataFrame(columns=['cycle_idx', 'phaseNo', 'laneType', 'diff_sec'])

    result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
    for phase_no in phase_nos:
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        for lane_type in lane_types:
            result[f"redClearanceRunningFlagPhase{phase_no}{lane_type}"] = 0
            result[f"redRunningFlagPhase{phase_no}{lane_type}"] = 0
            result[f"redClearanceRunningCrossingFlagPhase{phase_no}{lane_type}"] = 0
            result[f"redRunningCrossingFlagPhase{phase_no}{lane_type}"] = 0
            result[f"detectorOffAfterRedClearEndPhase{phase_no}{lane_type}"] = np.nan

    for phase_no in phase_nos:
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        for lane_type in lane_types:
            sub_flags = gp_flags[(gp_flags['phaseNo'] == phase_no) & (gp_flags['laneType'] == lane_type)]
            if len(sub_flags) > 0:
                rc_flags = sub_flags[sub_flags['signal_type'] == 'redClearance']
                if len(rc_flags) > 0:
                    idx = rc_flags['cycle_idx'].values
                    result.loc[idx, f"redClearanceRunningFlagPhase{phase_no}{lane_type}"] = 1
                    crossing_idx = rc_flags[rc_flags['is_crossing']]['cycle_idx'].values
                    if len(crossing_idx) > 0:
                        result.loc[crossing_idx, f"redClearanceRunningCrossingFlagPhase{phase_no}{lane_type}"] = 1
                        
                r_flags = sub_flags[sub_flags['signal_type'] == 'red']
                if len(r_flags) > 0:
                    idx = r_flags['cycle_idx'].values
                    result.loc[idx, f"redRunningFlagPhase{phase_no}{lane_type}"] = 1
                    crossing_idx = r_flags[r_flags['is_crossing']]['cycle_idx'].values
                    if len(crossing_idx) > 0:
                        result.loc[crossing_idx, f"redRunningCrossingFlagPhase{phase_no}{lane_type}"] = 1
            
            sub_diff = gp_diff[(gp_diff['phaseNo'] == phase_no) & (gp_diff['laneType'] == lane_type)]
            if len(sub_diff) > 0:
                sub_diff = sub_diff.dropna(subset=['diff_sec'])
                if len(sub_diff) > 0:
                    idx = sub_diff['cycle_idx'].values
                    vals = sub_diff['diff_sec'].values
                    result.loc[idx, f"detectorOffAfterRedClearEndPhase{phase_no}{lane_type}"] = vals

    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    cache.save_feature(result, 'vehicle_traffic', 'red_running')
    logging.info(f"Red running extracted: {len(result)} cycles")
    return result


def extract_dilemma_running(cache, with_countbar: bool = False) -> pd.DataFrame:
    """
    Extract dilemma zone running data matching v1 output exactly.
    Fully vectorized — no per-cycle Python loops.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "count" if with_countbar else "front"
    
    # Step 1: Assign events to cycles
    df_ev = cache.assign_cycle_index(
        event_sequence=(81,), detector_type=detector_type, tolerance='exact'
    )
    
    if len(df_ev) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'dilemma_running')
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_ev)
    df_ev = df_ev[df_ev['cycle_idx'] >= 0].copy()
    
    if len(df_ev) == 0 or len(phase_nos) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'dilemma_running')
        return result
    
    # Step 2: Get yellow signal type intervals
    df_intervals = cache.build_signal_type_intervals()
    df_intervals_y = df_intervals[
        (df_intervals['signal_type'] == 'yellow') &
        (df_intervals['phase_no'].isin(phase_nos))
    ].copy()
    
    # Merge events with yellow intervals
    df_ev_y = pd.merge(
        df_ev[['timeStamp', 'cycle_idx', 'phaseNo', 'channelNo', 'laneType']],
        df_intervals_y,
        left_on=['cycle_idx', 'phaseNo'],
        right_on=['cycle_idx', 'phase_no'],
        how='inner'
    )
    mask = (df_ev_y['timeStamp'] >= df_ev_y['interval_start']) & \
           before_interval_end(cache, df_ev_y['timeStamp'], df_ev_y['interval_end'])
    df_ev_y = df_ev_y[mask]
    
    # Group: any match = flag=1
    flags = (
        df_ev_y.groupby(['cycle_idx', 'phaseNo', 'laneType'])
        .size()
        .reset_index(name='count')
    )
    
    # Build result
    result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
    for phase_no in phase_nos:
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        for lane_type in lane_types:
            result[f"yellowRunningFlagPhase{phase_no}{lane_type}"] = 0
    
    for _, row in flags.iterrows():
        ci = int(row['cycle_idx'])
        pn = int(row['phaseNo'])
        lt = row['laneType']
        col = f"yellowRunningFlagPhase{pn}{lt}"
        if col in result.columns:
            result.loc[ci, col] = 1
    
    # Add date
    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'dilemma_running')
    
    logging.info(f"Dilemma running extracted: {len(result)} cycles")
    return result
