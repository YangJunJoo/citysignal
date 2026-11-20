"""
Occupancy feature extractor — vectorized.

Matches v1's TrafficFeatureExtract.extract_occupancy output exactly.
Uses searchsorted cycle assignment + vectorized sequence pairing
instead of per-cycle Python loops.
"""

import pandas as pd
import numpy as np
import logging

from ..utils import calculate_signal_type_stats

from .gap import _add_event_sequence_id


def _remove_common_periods(dict_data: dict) -> dict:
    """
    Removes common (overlapping) periods between time intervals of different keys.
    Exact port of v1's _remove_common_periods.
    """
    def is_overlap(period1, period2):
        return max(period1[0], period2[0]) < min(period1[1], period2[1])
    
    def adjust_periods(period1, period2):
        if is_overlap(period1, period2):
            if period1[0] < period2[0] and period1[1] > period2[1]:
                return [(period1[0], period2[0]), (period2[1], period1[1])]
            elif period2[0] <= period1[0] < period2[1]:
                return [(period2[1], period1[1])]
            elif period2[0] < period1[1] <= period2[1]:
                return [(period1[0], period2[0])]
        return [period1]
    
    keys = list(dict_data.keys())
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if i >= j:
                continue
            
            new_periods_key1 = []
            for period1 in dict_data[key1]:
                adjusted_periods = [period1]
                for period2 in dict_data[key2]:
                    temp = []
                    for p in adjusted_periods:
                        temp.extend(adjust_periods(p, period2))
                    adjusted_periods = temp
                new_periods_key1.extend(adjusted_periods)
            
            dict_data[key1] = new_periods_key1
    
    return dict_data


def _flatten_dict_value_by_grouped_indices(dict_data, key, grouped_indices):
    """
    Flattens specific groups of indices in list of lists.
    Exact port of v1's flatten_dict_value_by_grouped_indices.
    """
    if key not in dict_data:
        return dict_data[key]
    
    value = dict_data[key]
    flattened_result = []
    processed_indices = set()
    
    for group in grouped_indices:
        valid_indices = [i for i in group if 0 <= i < len(value)]
        flattened = []
        for i in valid_indices:
            if i not in processed_indices:
                flattened += value[i]
                processed_indices.add(i)
        if valid_indices:
            flattened_result.append(flattened)
    
    for i in range(len(value)):
        if i not in processed_indices:
            flattened_result.append(value[i])
    
    dict_data[key] = flattened_result
    return dict_data[key]


def extract_occupancy(cache) -> pd.DataFrame:
    """
    Extract cycle-level occupancy data matching v1 output exactly.
    Uses vectorized cycle assignment and groupby-based sequence pairing.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "front"
    df_events = cache.get_merged_events(event_sequence=(82, 81), detector_type=detector_type)
    
    if len(df_events) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'vehicle_traffic', 'occupancy')
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_events)
    
    if len(phase_nos) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'vehicle_traffic', 'occupancy')
        return result
    
    df_config = cache.get_config(detector_type)
    n_cycles = len(df_cycle)
    
    # Pre-compute tolerance windows (prev/next cycle boundaries)
    cycle_begins = df_cycle['cycleBegin'].values
    cycle_ends = df_cycle['cycleEnd'].values
    
    tol_begins = np.empty(n_cycles, dtype=cycle_begins.dtype)
    tol_ends = np.empty(n_cycles, dtype=cycle_ends.dtype)
    tol_begins[0] = cycle_begins[0]
    tol_begins[1:] = cycle_begins[:-1]
    tol_ends[-1] = cycle_ends[-1]
    tol_ends[:-1] = cycle_ends[1:]
    
    # Pre-build sequence pairs for each channel (vectorized)
    # Group events by phaseNo, channelNo — build sequence pairs once
    seq_pairs = {}  # {(phaseNo, channelNo): DataFrame with sequenceID, pair timestamps}
    
    for (pn, ch), grp in df_events.groupby(['phaseNo', 'channelNo']):
        if pn not in phase_nos:
            continue
        grp = grp.sort_values('timeStamp').reset_index(drop=True).copy()
        grp['sequenceID'] = _add_event_sequence_id(grp, [82, 81])
        
        # Extract valid pairs (exactly 2 rows per sequence) (numpy optimized)
        seq_sizes = grp['sequenceID'].value_counts()
        valid_seq_ids = seq_sizes[seq_sizes == 2].index
        grp_clean = grp[grp['sequenceID'].isin(valid_seq_ids)].sort_values(['sequenceID', 'timeStamp'])
        if len(grp_clean) > 0:
            seq_pairs[(pn, ch)] = (
                grp_clean['timeStamp'].values[::2].astype('int64'),
                grp_clean['timeStamp'].values[1::2].astype('int64')
            )
    
    # Build result row by row (lightweight — no DataFrame filtering per cycle)
    result_rows = []
    
    for i in range(n_cycles):
        row_dict = {
            'signalID': df_cycle.loc[i, 'signalID'],
            'cycleNo': df_cycle.loc[i, 'cycleNo'],
            'cycleBegin': df_cycle.loc[i, 'cycleBegin'],
            'cycleEnd': df_cycle.loc[i, 'cycleEnd'],
            'cycleLength': df_cycle.loc[i, 'cycleLength'],
        }
        
        # Initialize columns
        for phase_no in phase_nos:
            lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
            for lane_type in lane_types:
                channel_nos = cache.get_channels_for_phase_lane(
                    detector_type, phase_no, lane_type
                )
                row_dict[f"channelNos{phase_no}{lane_type}"] = channel_nos
                for st in ['green', 'yellow', 'redClearance', 'red', 'red5']:
                    row_dict[f"{st}OccupancyPhase{phase_no}{lane_type}"] = [
                        [np.nan]
                    ] * len(channel_nos)
        
        # Tolerance window for this cycle
        st_tol = pd.Timestamp(tol_begins[i])
        et_tol = pd.Timestamp(tol_ends[i])
        st_tol_ns = st_tol.value
        et_tol_ns = et_tol.value
        
        for phase_no in phase_nos:
            lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
            signal_types = ['green', 'yellow', 'redClearance', 'red', 'red5']
            
            # Pre-filter channel pairs for this phase to the tolerance window once per cycle per phase!
            phase_pairs_in_tol = {}
            for (pn, ch), (det_ons, det_offs) in seq_pairs.items():
                if pn == phase_no:
                    mask = (det_ons >= st_tol_ns) & (det_offs <= et_tol_ns)
                    phase_pairs_in_tol[ch] = (det_ons[mask], det_offs[mask])
            
            for signal_type in signal_types:
                if signal_type == 'red5':
                    timestamps = cache.get_signal_timestamps(i, 'redClearance', phase_no)
                    if timestamps is None:
                        continue
                    timestamps = [(timestamps[0][0],
                                   timestamps[0][0] + pd.Timedelta(seconds=5))]
                else:
                    timestamps = cache.get_signal_timestamps(i, signal_type, phase_no)
                    if timestamps is None:
                        continue
                
                for lane_type in lane_types:
                    df_config_lane = df_config[
                        (df_config['phaseNo'] == phase_no) &
                        (df_config['laneType'] == lane_type)
                    ].reset_index(drop=True)
                    
                    if 'laneNoFromLeft' in df_config_lane.columns:
                        lane_nos = df_config_lane['laneNoFromLeft'].unique().tolist()
                    else:
                        lane_nos = [1]
                    
                    indices_to_flatten = []
                    
                    for lane_no in lane_nos:
                        if 'laneNoFromLeft' in df_config_lane.columns:
                            ch_config = df_config_lane[
                                df_config_lane['laneNoFromLeft'] == lane_no
                            ]
                        else:
                            ch_config = df_config_lane
                        
                        channel_nos = ch_config['channelNo'].unique().tolist()
                        channel_nos = [ch for ch in channel_nos if pd.notna(ch)]
                        
                        if not channel_nos:
                            continue
                        
                        dict_occupancy_channel = {ch: [] for ch in channel_nos}
                        
                        for channel_no in channel_nos:
                            pairs = phase_pairs_in_tol.get(channel_no)
                            if pairs is None or len(pairs[0]) == 0:
                                continue
                            
                            det_ons, det_offs = pairs
                            
                            for start_time, end_time in timestamps:
                                if pd.isna(start_time) or pd.isna(end_time):
                                    continue
                                
                                start_time_ns = start_time.value
                                end_time_ns = end_time.value
                                
                                # Find pairs overlapping with signal window
                                mask_over = (det_ons <= end_time_ns) & (det_offs >= start_time_ns)
                                ons_over = det_ons[mask_over]
                                offs_over = det_offs[mask_over]
                                
                                for on_ns, off_ns in zip(ons_over, offs_over):
                                    max_st = max(on_ns, start_time_ns)
                                    min_et = min(off_ns, end_time_ns)
                                    if max_st < min_et:
                                        dict_occupancy_channel[channel_no].append(
                                            (pd.Timestamp(max_st), pd.Timestamp(min_et))
                                        )
                        
                        # Remove common periods across channels
                        if len(lane_nos) > 1:
                            dict_occupancy_channel = _remove_common_periods(
                                dict_occupancy_channel
                            )
                        
                        all_channel_nos = row_dict[f"channelNos{phase_no}{lane_type}"]
                        
                        for key_ch in dict_occupancy_channel.keys():
                            occupancies = [np.nan]
                            for st_time, en_time in dict_occupancy_channel[key_ch]:
                                time_diff = abs(round(
                                    (en_time - st_time).total_seconds(), 4
                                ))
                                occupancies.append(time_diff)
                            
                            if key_ch in all_channel_nos:
                                idx = all_channel_nos.index(key_ch)
                                current = list(row_dict[
                                    f"{signal_type}OccupancyPhase{phase_no}{lane_type}"
                                ])
                                current[idx] = occupancies
                                row_dict[
                                    f"{signal_type}OccupancyPhase{phase_no}{lane_type}"
                                ] = current
                        
                        if len(channel_nos) > 1:
                            indices = [
                                all_channel_nos.index(ch) for ch in channel_nos
                                if ch in all_channel_nos
                            ]
                            indices_to_flatten.append(indices)
                    
                    if indices_to_flatten:
                        row_dict[f"{signal_type}OccupancyPhase{phase_no}{lane_type}"] = (
                            _flatten_dict_value_by_grouped_indices(
                                row_dict,
                                f"{signal_type}OccupancyPhase{phase_no}{lane_type}",
                                indices_to_flatten
                            )
                        )
        
        result_rows.append(row_dict)
    
    result = pd.DataFrame(result_rows)
    
    if len(result) > 0:
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        
        columns = [col for col in result.columns if 'Occupancy' in col]
        result = calculate_signal_type_stats(result, columns, include_sum=True)
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'occupancy')
    
    logging.info(f"Occupancy extracted: {len(result)} cycles")
    return result


