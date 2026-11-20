"""
Volume feature extractor — vectorized.

Matches v1's TrafficFeatureExtract.extract_volume output exactly.
Uses searchsorted cycle assignment + interval join for signal type classification
+ groupby counting instead of per-cycle Python loops.
"""

import pandas as pd
import logging
from ..profiles import before_interval_end


def extract_volume(cache, with_countbar: bool = False) -> pd.DataFrame:
    """
    Extract cycle-level volume data matching v1 output exactly.
    Fully vectorized — no per-cycle Python loops.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "count" if with_countbar else "back"
    
    # Step 1: Assign every event to a cycle index using searchsorted
    df_ev = cache.assign_cycle_index(
        event_sequence=(81,), detector_type=detector_type, tolerance='exact'
    )
    
    if len(df_ev) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'volume')
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_ev)
    
    # Keep only events assigned to a cycle
    df_ev = df_ev[df_ev['cycle_idx'] >= 0].copy()
    
    if len(df_ev) == 0 or len(phase_nos) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'volume')
        return result
    
    # Step 2: Assign signal type to each event via interval join
    df_intervals = cache.build_signal_type_intervals()
    
    # For volume: only need phases that appear in events
    df_intervals_vol = df_intervals[
        df_intervals['phase_no'].isin(phase_nos)
    ].copy()
    
    # Assign signal type: for each event, find which interval it falls into
    # Use a merge + filter approach: merge events with intervals on (cycle_idx, phaseNo==phase_no)
    # then filter where interval_start <= timeStamp <= interval_end
    df_ev_st = pd.merge(
        df_ev[['timeStamp', 'cycle_idx', 'phaseNo', 'channelNo', 'laneType']],
        df_intervals_vol,
        left_on=['cycle_idx', 'phaseNo'],
        right_on=['cycle_idx', 'phase_no'],
        how='inner'
    )
    # Filter events within signal type window
    mask = (df_ev_st['timeStamp'] >= df_ev_st['interval_start']) & \
           before_interval_end(cache, df_ev_st['timeStamp'], df_ev_st['interval_end'])
    df_ev_st = df_ev_st[mask].copy()
    
    # Step 3: Count volumes by groupby
    # Per signal type, per phase, per lane type
    counts_by_signal = (
        df_ev_st.groupby(['cycle_idx', 'phaseNo', 'laneType', 'signal_type'])
        .size()
        .reset_index(name='count')
    )
    
    # Total volume per phase per lane (sum across signal types)
    counts_total = (
        counts_by_signal.groupby(['cycle_idx', 'phaseNo', 'laneType'])['count']
        .sum()
        .reset_index(name='total_count')
    )
    
    # Phase-level volume (all lanes for a phase)
    # Use events assigned to cycles, count by (cycle_idx, phaseNo)
    counts_phase = (
        df_ev[df_ev['phaseNo'].isin(phase_nos)]
        .groupby(['cycle_idx', 'phaseNo'])
        .size()
        .reset_index(name='phase_count')
    )
    
    # Step 4: Build result DataFrame
    result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
    
    # Initialize all columns to 0
    for phase_no in phase_nos:
        result[f"volumePhase{phase_no}"] = 0
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        for lane_type in lane_types:
            result[f"volumePhase{phase_no}{lane_type}"] = 0
            for st in ['green', 'yellow', 'redClearance', 'red']:
                result[f"{st}VolumePhase{phase_no}{lane_type}"] = 0
    
    # Fill phase-level volumes
    for _, row in counts_phase.iterrows():
        ci = int(row['cycle_idx'])
        pn = int(row['phaseNo'])
        col = f"volumePhase{pn}"
        if col in result.columns:
            result.loc[ci, col] = int(row['phase_count'])
    
    # Fill signal-type-level volumes
    for _, row in counts_by_signal.iterrows():
        ci = int(row['cycle_idx'])
        pn = int(row['phaseNo'])
        lt = row['laneType']
        st = row['signal_type']
        col = f"{st}VolumePhase{pn}{lt}"
        if col in result.columns:
            result.loc[ci, col] = int(row['count'])
    
    # Fill total (per-lane) volumes
    for _, row in counts_total.iterrows():
        ci = int(row['cycle_idx'])
        pn = int(row['phaseNo'])
        lt = row['laneType']
        col = f"volumePhase{pn}{lt}"
        if col in result.columns:
            result.loc[ci, col] = int(row['total_count'])
    
    # Add date
    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'volume')
    
    logging.info(f"Volume extracted: {len(result)} cycles")
    return result
