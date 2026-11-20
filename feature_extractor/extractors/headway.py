"""
Headway feature extractor — vectorized.

Matches v1's TrafficFeatureExtract.extract_headway exactly.
Uses searchsorted cycle assignment + interval join for signal type classification
+ groupby-based consecutive time diffs instead of per-cycle Python loops.
"""

import pandas as pd
import numpy as np
import logging
from ..profiles import before_interval_end

from ..utils import calculate_signal_type_stats


def extract_headway(cache) -> pd.DataFrame:
    """
    Extract cycle-level headway data matching v1 output exactly.
    Fully vectorized — no per-cycle Python loops.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "back"
    
    # Step 1: Assign events to cycles
    df_ev = cache.assign_cycle_index(
        event_sequence=(82,), detector_type=detector_type, tolerance='exact'
    )
    
    if len(df_ev) == 0:
        df_result = pd.DataFrame()
        cache.save_feature(df_result, 'vehicle_traffic', 'headway')
        return df_result
    
    phase_nos = cache.get_valid_phase_nos(df_ev)
    df_ev = df_ev[df_ev['cycle_idx'] >= 0].copy()
    
    if len(df_ev) == 0 or len(phase_nos) == 0:
        df_result = pd.DataFrame()
        cache.save_feature(df_result, 'vehicle_traffic', 'headway')
        return df_result
    
    n_cycles = len(df_cycle)
    
    # Step 2: Assign signal type via interval join
    df_intervals = cache.build_signal_type_intervals()
    df_intervals_hw = df_intervals[
        df_intervals['phase_no'].isin(phase_nos)
    ].copy()
    
    # Merge events with signal type intervals
    df_ev_st = pd.merge(
        df_ev[['timeStamp', 'cycle_idx', 'phaseNo', 'channelNo', 'laneType']],
        df_intervals_hw,
        left_on=['cycle_idx', 'phaseNo'],
        right_on=['cycle_idx', 'phase_no'],
        how='inner'
    )
    mask = (df_ev_st['timeStamp'] >= df_ev_st['interval_start']) & \
           before_interval_end(cache, df_ev_st['timeStamp'], df_ev_st['interval_end'])
    df_ev_st = df_ev_st[mask].copy()
    
    # Sort within each group to compute headway as consecutive time diffs
    df_ev_st = df_ev_st.sort_values(
        ['cycle_idx', 'phaseNo', 'channelNo', 'signal_type', 'interval_start', 'timeStamp']
    ).reset_index(drop=True)
    
    # Step 3: Compute headway per (cycle, phase, channel, signal_type) group
    # V1 computes diffs WITHIN each signal-type interval separately, then
    # concatenates them with a single [nan] prefix per channel per signal type.
    # Use interval_start to separate windows, compute diffs per window, then merge.
    window_keys = ['cycle_idx', 'phaseNo', 'channelNo', 'signal_type', 'laneType', 'interval_start']
    
    # First: compute headways per window
    window_records = []
    for keys, grp in df_ev_st.groupby(window_keys):
        ci, pn, ch, st, lt, _int_start = keys
        ts = grp['timeStamp'].values
        if len(ts) <= 1:
            hw = []  # V1 skips windows with <= 1 event
        else:
            diffs = np.diff(ts.astype('datetime64[ns]').astype(np.int64)) / 1e9
            hw = [round(d, 4) for d in diffs]
        window_records.append({
            'cycle_idx': ci, 'phaseNo': pn, 'channelNo': ch,
            'signal_type': st, 'laneType': lt, 'headways': hw
        })
    
    # Merge windows: concatenate headways per (cycle, phase, channel, signal_type, laneType)
    # with a single [nan] prefix, matching V1's initialization
    from collections import defaultdict
    merged = defaultdict(list)
    for rec in window_records:
        key = (rec['cycle_idx'], rec['phaseNo'], rec['channelNo'], rec['signal_type'], rec['laneType'])
        merged[key].extend(rec['headways'])
    
    headway_records = []
    for (ci, pn, ch, st, lt), hw_list in merged.items():
        headway_records.append({
            'cycle_idx': ci, 'phaseNo': pn, 'channelNo': ch,
            'signal_type': st, 'laneType': lt,
            'headways': [np.nan] + hw_list if hw_list else [np.nan]
        })
    
    df_headways = pd.DataFrame(headway_records) if headway_records else pd.DataFrame()
    
    # Step 4: Build result DataFrame (wide format matching v1)
    result_rows = []
    
    for i in range(n_cycles):
        row_dict = {
            'signalID': df_cycle.loc[i, 'signalID'],
            'cycleNo': df_cycle.loc[i, 'cycleNo'],
            'cycleBegin': df_cycle.loc[i, 'cycleBegin'],
            'cycleEnd': df_cycle.loc[i, 'cycleEnd'],
            'cycleLength': df_cycle.loc[i, 'cycleLength'],
        }
        
        # Initialize all headway columns
        for phase_no in phase_nos:
            lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
            for lane_type in lane_types:
                channel_nos = cache.get_channels_for_phase_lane(
                    detector_type, phase_no, lane_type
                )
                row_dict[f"channelNos{phase_no}{lane_type}"] = channel_nos
                for st in ['green', 'yellow', 'redClearance', 'red']:
                    row_dict[f"{st}HeadwayPhase{phase_no}{lane_type}"] = [
                        [np.nan]
                    ] * len(channel_nos)
        
        # Fill from computed headways
        if len(df_headways) > 0:
            cycle_data = df_headways[df_headways['cycle_idx'] == i]
            for _, hrow in cycle_data.iterrows():
                pn = int(hrow['phaseNo'])
                ch = int(hrow['channelNo']) if pd.notna(hrow['channelNo']) else hrow['channelNo']
                st = hrow['signal_type']
                lt = hrow['laneType']
                headways = hrow['headways']
                
                col_key = f"channelNos{pn}{lt}"
                hw_key = f"{st}HeadwayPhase{pn}{lt}"
                
                if col_key in row_dict and hw_key in row_dict:
                    ch_list = row_dict[col_key]
                    if ch in ch_list:
                        ch_idx = ch_list.index(ch)
                        current = list(row_dict[hw_key])
                        current[ch_idx] = headways
                        row_dict[hw_key] = current
        
        result_rows.append(row_dict)
    
    result = pd.DataFrame(result_rows)
    
    if len(result) > 0:
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        
        # Calculate statistics (v1's _calculate_stats with include_sum=False)
        columns = [col for col in result.columns if 'Headway' in col]
        result = calculate_signal_type_stats(result, columns, include_sum=False)
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'headway')
    
    logging.info(f"Headway extracted: {len(result)} cycles")
    return result


