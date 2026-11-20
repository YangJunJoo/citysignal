"""
Shockwave feature extractor — vectorized.

Uses cached merged events and per-cycle red intervals, including a 60s buffer.
Inner _calculate_shockwave_variables logic is kept as-is since it's
inherently sequential (finding first >=5s occupation).
"""

import pandas as pd
import numpy as np
import logging


def extract_shockwave(cache) -> pd.DataFrame:
    """
    Extract shockwave data matching v1 output exactly.
    Filter merged detector events within each red interval plus 60 seconds.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "back"
    
    # Keep events outside cycle bounds: the red interval filter includes a buffer.
    df_all_events = cache.get_merged_events(event_sequence=(81, 82), detector_type=detector_type)
    
    if len(df_all_events) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', 'shockwave')
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_all_events)
    df_config = cache.get_config(detector_type)
    n_cycles = len(df_cycle)
    
    # Get red signal type intervals  
    df_intervals = cache.build_signal_type_intervals()
    df_intervals_red = df_intervals[
        (df_intervals['signal_type'] == 'red') &
        (df_intervals['phase_no'].isin(phase_nos))
    ].copy()
    
    # Step 2: Process per (cycle, phase, laneType) group using vectorized lookup
    result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
    
    # Pre-compute all column names
    for phase_no in phase_nos:
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        for lane_type in lane_types:
            _init_nan(result, phase_no, lane_type)
    
    # Convert intervals to dict for fast lookup
    red_intervals_dict = {}
    for _, row in df_intervals_red.iterrows():
        key = (int(row['cycle_idx']), int(row['phase_no']))
        if key not in red_intervals_dict:
            red_intervals_dict[key] = []
        red_intervals_dict[key].append((row['interval_start'], row['interval_end']))
    
    # Process each (cycle, phase, lane_type)
    for phase_no in phase_nos:
        lane_types = cache.get_lane_types_for_phase(detector_type, phase_no)
        
        for lane_type in lane_types:
            channel_nos = cache.get_channels_for_phase_lane(
                detector_type, phase_no, lane_type
            )
            L_d = _get_stop_bar_distance(df_config, phase_no, lane_type, channel_nos)
            
            if L_d is None or L_d <= 0:
                continue
            
            for ci in range(n_cycles):
                red_ts = red_intervals_dict.get((ci, phase_no))
                if red_ts is None:
                    continue
                
                for t_red_on, t_red_off in red_ts:
                    if pd.isna(t_red_on) or pd.isna(t_red_off):
                        continue
                    
                    key_v1 = f"v1Phase{phase_no}{lane_type}"
                    if not pd.isna(result.loc[ci, key_v1]):
                        continue  # Already computed
                    
                    # Filter events for these channels in [t_red_on, t_red_off + 60s]
                    t_red_off_buffer = t_red_off + pd.Timedelta(seconds=60)
                    ch_events = df_all_events[
                        (df_all_events['phaseNo'] == phase_no) &
                        (df_all_events['channelNo'].isin(channel_nos)) &
                        (df_all_events['timeStamp'] >= t_red_on) &
                        (df_all_events['timeStamp'] <= t_red_off_buffer)
                    ].sort_values('timeStamp').reset_index(drop=True)
                    
                    if len(ch_events) == 0:
                        continue
                    
                    sw = _calculate_shockwave_variables(ch_events, t_red_on, t_red_off, L_d)
                    
                    if sw.get('v1') is not None:
                        result.loc[ci, f"v1Phase{phase_no}{lane_type}"] = sw['v1']
                        result.loc[ci, f"v2Phase{phase_no}{lane_type}"] = sw.get('v2', np.nan)
                        result.loc[ci, f"L_maxPhase{phase_no}{lane_type}"] = sw.get('L_max', np.nan)
                        result.loc[ci, f"SPhase{phase_no}{lane_type}"] = sw.get('S', np.nan)
                        result.loc[ci, f"t_queue_onPhase{phase_no}{lane_type}"] = sw.get('t_queue_on', np.nan)
                        result.loc[ci, f"t_queue_offPhase{phase_no}{lane_type}"] = sw.get('t_queue_off', np.nan)
    
    # Add date
    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', 'shockwave')
    
    logging.info(f"Shockwave extracted: {len(result)} cycles")
    return result


def _init_nan(result_df, phase_no, lane_type):
    """Initialize shockwave columns with NaN for a phase-lane."""
    for prefix in ['v1', 'v2', 'L_max', 'S', 't_queue_on', 't_queue_off']:
        result_df[f"{prefix}Phase{phase_no}{lane_type}"] = np.nan


def _get_stop_bar_distance(df_config, phase_no, lane_type, channel_nos):
    """Get stopBarDistance from config for the first channel."""
    if not channel_nos:
        return None
    mask = (
        (df_config['phaseNo'] == phase_no) & 
        (df_config['laneType'] == lane_type) & 
        (df_config['channelNo'] == channel_nos[0])
    )
    rows = df_config[mask]
    if len(rows) > 0 and 'stopBarDistance' in rows.columns:
        val = rows['stopBarDistance'].iloc[0]
        return val if pd.notna(val) else None
    return None


def _calculate_shockwave_variables(df_events, t_red_on, t_red_off, L_d):
    """
    Calculate shockwave variables from detector events during a red period.
    Exact port of v1's _calculate_shockwave_variables.
    """
    result = {
        't_queue_on': None, 't_queue_off': None,
        'v1': None, 'v2': None, 'L_max': None, 'S': None
    }
    
    df = df_events.sort_values('timeStamp').reset_index(drop=True)
    
    # Normalize event codes
    ec_map = {82: 'detector_on', 81: 'detector_off'}
    df['normalized_event'] = df['eventCode'].map(ec_map)
    df = df[df['normalized_event'].isin(['detector_on', 'detector_off'])].reset_index(drop=True)
    
    if len(df) == 0:
        return result
    
    # Find continuous detector occupation >= 5 seconds
    events = df['normalized_event'].values
    timestamps = df['timeStamp'].values
    
    t_queue_on = None
    t_queue_off = None
    
    for i in range(len(df) - 1):
        if events[i] == 'detector_on' and events[i + 1] == 'detector_off':
            duration = (timestamps[i + 1] - timestamps[i]) / np.timedelta64(1, 's')
            if duration >= 5.0:
                t_queue_on = pd.Timestamp(timestamps[i])
                t_queue_off = pd.Timestamp(timestamps[i + 1])
                break
    
    if t_queue_on is None:
        return result
    
    result['t_queue_on'] = t_queue_on
    result['t_queue_off'] = t_queue_off
    
    t_red_duration = (t_red_off - t_red_on).total_seconds()
    
    # v1 = L_d / (t_queue_on - t_red_on)
    time_diff_v1 = (t_queue_on - t_red_on).total_seconds()
    v1 = L_d / time_diff_v1 if time_diff_v1 > 0 else None
    
    # v2 = L_d / (t_queue_off - t_red_off)
    time_diff_v2 = (t_queue_off - t_red_off).total_seconds()
    v2 = L_d / time_diff_v2 if time_diff_v2 > 0 else None
    
    result['v1'] = v1
    result['v2'] = v2
    
    # L_max
    if v1 is not None and v2 is not None and v2 > 0:
        if v1 < 0:
            denominator = v2 - v1
        else:
            denominator = v1 + v2
        result['L_max'] = (v1 * v2 * t_red_duration) / denominator if denominator > 0 else None
    
    # S = (L_max * t_red_duration) / 2
    if result['L_max'] is not None and t_red_duration > 0:
        result['S'] = (result['L_max'] * t_red_duration) / 2.0
    
    return result
