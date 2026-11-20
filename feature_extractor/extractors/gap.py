"""
Gap (stopbar gap and PET) feature extractor — vectorized.

Gap: Time difference between consecutive unoccupied detector intervals
     during each signal type period, using front (stopbar) detectors.
PET: Same concept but using back (setback) detectors, per-channel/per-laneType.

Key optimizations over the loop-based version:
  - Pre-builds OFF/ON and ON/OFF sequence pairs ONCE per (phase, channel)
  - Uses numpy arrays for interval overlap checks instead of per-pair DataFrame filtering
  - Processes cycles via pre-indexed pair tables instead of full event re-scan
"""

import pandas as pd
import numpy as np
import logging

from ..utils import calculate_signal_type_stats


def _add_event_sequence_id(df_event, valid_event_sequence):
    """
    Assign sequence IDs to events based on consecutive matching event code pairs.
    Exact port of v1's add_event_sequence_id, accelerated with numpy array access.
    """
    df_event = df_event.sort_values(by=["timeStamp", "eventCode", "channelNo"]).reset_index(drop=True)
    
    # Extract event codes as numpy array — avoids iterrows() overhead
    event_codes = df_event["eventCode"].values.astype(np.int64)
    valid_seq = list(valid_event_sequence)
    seq_len = len(valid_seq)
    
    sequence_id = 1
    n = len(event_codes)
    sequence_ids = np.empty(n, dtype=np.int64)
    current_seq = []
    
    for i in range(n):
        ec = int(event_codes[i])
        
        candidate = current_seq + [ec]
        if len(candidate) <= seq_len and candidate == valid_seq[:len(candidate)]:
            current_seq = candidate
        else:
            sequence_id += 1
            current_seq = [ec]
            if len(current_seq) > seq_len or current_seq != valid_seq[:len(current_seq)]:
                sequence_id += 1
                current_seq = [ec]
        
        sequence_ids[i] = sequence_id
        
        if len(current_seq) == seq_len:
            sequence_id += 1
            current_seq = []
    
    return sequence_ids.tolist()






def _build_sequence_pairs(df_events, phase_nos, seq_pattern):
    """
    Pre-build event sequence pairs for all (phase, channel) combos.
    
    Returns dict: {(phaseNo, channelNo): np.ndarray of shape (N_pairs, 2)}
      where each row is [timestamp_first, timestamp_second] in int64 nanoseconds.
    """
    pairs_dict = {}
    
    for (pn, ch), grp in df_events.groupby(['phaseNo', 'channelNo']):
        if pn not in phase_nos:
            continue
        grp = grp.sort_values(['timeStamp', 'eventCode', 'channelNo']).reset_index(drop=True).copy()
        grp['sequenceID'] = _add_event_sequence_id(grp, seq_pattern)
        
        # Keep only complete pairs (pandas optimized)
        seq_sizes = grp['sequenceID'].value_counts()
        valid_seq_ids = seq_sizes[seq_sizes == 2].index
        grp_clean = grp[grp['sequenceID'].isin(valid_seq_ids)].sort_values(['sequenceID', 'timeStamp'])
        if len(grp_clean) > 0:
            starts = grp_clean['timeStamp'].values[::2].astype('int64')
            ends = grp_clean['timeStamp'].values[1::2].astype('int64')
            pairs_dict[(pn, ch)] = np.column_stack([starts, ends])
        else:
            pairs_dict[(pn, ch)] = np.empty((0, 2), dtype=np.int64)
    
    return pairs_dict


def extract_gap(cache, with_stopbar: bool = True) -> pd.DataFrame:
    """
    Extract gap/PET data matching v1 output exactly.
    
    Key optimization: pre-builds OFF→ON and ON→OFF pair lookup tables once,
    then does numpy array comparisons instead of per-cycle DataFrame filtering.
    """
    df_cycle = cache.df_vehicle_cycle
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    detector_type = "front" if with_stopbar else "back"
    df_events = cache.get_merged_events(event_sequence=(81, 82), detector_type=detector_type)
    
    extraction_type = "Gap" if with_stopbar else "PET"
    feature_name = 'gap' if with_stopbar else 'pet'
    
    if len(df_events) == 0:
        result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
        cache.save_feature(result, 'vehicle_traffic', feature_name)
        return result
    
    phase_nos = cache.get_valid_phase_nos(df_events)
    df_config = cache.get_config(detector_type)
    n_cycles = len(df_cycle)
    
    # Pre-compute tolerance windows (prev/next cycle boundaries)
    cycle_begins_arr = df_cycle['cycleBegin'].values.astype('int64')
    cycle_ends_arr = df_cycle['cycleEnd'].values.astype('int64')
    
    # Pre-build signal type intervals from cache
    df_intervals = cache.build_signal_type_intervals()
    
    # Build interval lookup: {(cycle_idx, phase_no, signal_type): list of (start_ns, end_ns)}
    interval_lookup = {}
    for _, row in df_intervals.iterrows():
        key = (int(row['cycle_idx']), int(row['phase_no']), row['signal_type'])
        if key not in interval_lookup:
            interval_lookup[key] = []
        interval_lookup[key].append((
            pd.Timestamp(row['interval_start']).value,
            pd.Timestamp(row['interval_end']).value,
        ))
    
    if with_stopbar:
        # ===== STOPBAR GAP =====
        # V1 algorithm (faithfully replicated):
        #   Per cycle:
        #     1. Filter all events to tolerance window [prev_cycle_begin, next_cycle_end]
        #     2. Per-channel: assign sequence IDs [81,82], keep only complete pairs'
        #        RAW EVENTS (with sequenceID dropped)
        #     3. Concat kept events across channels, sort by timestamp
        #     4. Re-assign sequence IDs [81,82] on the combined stream
        #        → creates cross-channel OFF→ON pairs
        #     5. Per signal type (±1min buffer): check each pair against
        #        per-channel ON→OFF occupation spans, compute gap
        
        # Get all channel numbers per phase from events
        phase_channels = {}
        for (pn, ch), _ in df_events.groupby(['phaseNo', 'channelNo']):
            if pn not in phase_nos:
                continue
            if pn not in phase_channels:
                phase_channels[pn] = set()
            phase_channels[pn].add(ch)
        for pn in phase_channels:
            phase_channels[pn] = sorted(phase_channels[pn])

        # Pre-build per-phase combined OFF->ON intervals and per-channel ON->OFF intervals
        phase_off_intervals = {} # phase_no -> (det_offs_all, det_ons_all)
        phase_on_intervals = {}  # phase_no -> {ch -> (starts, ends)}
        
        for phase_no in phase_nos:
            channels = phase_channels.get(phase_no, [])
            if not channels:
                continue
            
            # 1. Per-channel complete [81, 82] sequences for the whole day
            dfs_off = []
            on_intervals = {}
            
            # Filter events to this phase once
            phase_events = df_events[df_events['phaseNo'] == phase_no]
            if len(phase_events) == 0:
                continue
                
            for ch in channels:
                # OFF->ON (81->82)
                ch_events = phase_events[phase_events['channelNo'] == ch] \
                    .sort_values(['timeStamp', 'eventCode', 'channelNo']).reset_index(drop=True).copy()
                if len(ch_events) > 0:
                    ch_events['sequenceID'] = _add_event_sequence_id(ch_events, [81, 82])
                    seq_sizes = ch_events['sequenceID'].value_counts()
                    valid_seq_ids = seq_sizes[seq_sizes == 2].index
                    ch_filtered = ch_events[ch_events['sequenceID'].isin(valid_seq_ids)].reset_index(drop=True).drop(columns='sequenceID')
                    if len(ch_filtered) > 0:
                        dfs_off.append(ch_filtered)
                
                # ON->OFF (82->81)
                ch_on_events = phase_events[phase_events['channelNo'] == ch] \
                    .sort_values(['timeStamp', 'eventCode', 'channelNo']).reset_index(drop=True).copy()
                if len(ch_on_events) > 0:
                    ch_on_events['sequenceID'] = _add_event_sequence_id(ch_on_events, [82, 81])
                    seq_sizes = ch_on_events['sequenceID'].value_counts()
                    valid_seq_ids = seq_sizes[seq_sizes == 2].index
                    df_clean = ch_on_events[ch_on_events['sequenceID'].isin(valid_seq_ids)].reset_index(drop=True)
                    if len(df_clean) > 0:
                        starts = df_clean['timeStamp'].values[::2].astype('int64')
                        ends = df_clean['timeStamp'].values[1::2].astype('int64')
                        on_intervals[ch] = (starts, ends)
            
            phase_on_intervals[phase_no] = on_intervals
            
            if not dfs_off:
                continue
                
            # 2. Concat, sort, re-sequence across channels
            df_phase_off = pd.concat(dfs_off, axis=0, ignore_index=True)
            df_phase_off = df_phase_off.sort_values(['timeStamp', 'eventCode', 'channelNo']).reset_index(drop=True).copy()
            df_phase_off['sequenceID'] = _add_event_sequence_id(df_phase_off, [81, 82])
            
            # Keep only complete pairs
            seq_sizes = df_phase_off['sequenceID'].value_counts()
            valid_seq_ids = seq_sizes[seq_sizes == 2].index
            df_phase_off_clean = df_phase_off[df_phase_off['sequenceID'].isin(valid_seq_ids)].reset_index(drop=True)
            
            if len(df_phase_off_clean) > 0:
                df_phase_off_sorted = df_phase_off_clean.sort_values(['sequenceID', 'timeStamp'])
                det_offs_all = df_phase_off_sorted['timeStamp'].values[::2].astype('int64')
                det_ons_all = df_phase_off_sorted['timeStamp'].values[1::2].astype('int64')
                phase_off_intervals[phase_no] = (det_offs_all, det_ons_all)
        
        result_rows = []
        
        for i in range(n_cycles):
            row_dict = {
                'signalID': df_cycle.loc[i, 'signalID'],
                'cycleNo': df_cycle.loc[i, 'cycleNo'],
                'cycleBegin': df_cycle.loc[i, 'cycleBegin'],
                'cycleEnd': df_cycle.loc[i, 'cycleEnd'],
                'cycleLength': df_cycle.loc[i, 'cycleLength'],
            }
            
            # Tolerance window (v1: prev_cycle_begin to next_cycle_end)
            if i == 0:
                tol_begin = df_cycle.loc[0, 'cycleBegin']
                tol_end = df_cycle.loc[min(1, n_cycles - 1), 'cycleEnd']
            elif i == n_cycles - 1:
                tol_begin = df_cycle.loc[i - 1, 'cycleBegin']
                tol_end = df_cycle.loc[i, 'cycleEnd']
            else:
                tol_begin = df_cycle.loc[i - 1, 'cycleBegin']
                tol_end = df_cycle.loc[i + 1, 'cycleEnd']
            
            tol_begin_ns = tol_begin.value
            tol_end_ns = tol_end.value
            
            for phase_no in phase_nos:
                for sig_type in ['green', 'yellow', 'redClearance', 'red']:
                    row_dict[f"{sig_type}GapPhase{phase_no}"] = [np.nan]
                
                channels = phase_channels.get(phase_no, [])
                if not channels:
                    continue
                
                if phase_no not in phase_off_intervals:
                    continue
                
                det_offs_all, det_ons_all = phase_off_intervals[phase_no]
                on_intervals = phase_on_intervals.get(phase_no, {})
                
                # Filter to tolerance window
                mask_tol = (det_offs_all >= tol_begin_ns) & (det_ons_all <= tol_end_ns)
                det_offs_tol = det_offs_all[mask_tol]
                det_ons_tol = det_ons_all[mask_tol]
                
                if len(det_offs_tol) == 0:
                    continue
                
                # Process each signal type
                signal_types = ['green', 'yellow', 'redClearance', 'red']
                for signal_type in signal_types:
                    intervals = interval_lookup.get((i, phase_no, signal_type))
                    if not intervals:
                        continue
                    
                    gaps = [np.nan]
                    
                    for sig_start_ns, sig_end_ns in intervals:
                        sig_start_ts = pd.Timestamp(sig_start_ns)
                        sig_end_ts = pd.Timestamp(sig_end_ns)
                        buf_start_ts = sig_start_ts - pd.Timedelta(minutes=1)
                        buf_end_ts = sig_end_ts + pd.Timedelta(minutes=1)
                        buf_start_ns = buf_start_ts.value
                        buf_end_ns = buf_end_ts.value
                        
                        # Filter to buffer window using numpy!
                        mask_buf = (det_offs_tol >= buf_start_ns) & (det_ons_tol <= buf_end_ns)
                        det_offs = det_offs_tol[mask_buf]
                        det_ons = det_ons_tol[mask_buf]
                        
                        if len(det_offs) == 0:
                            continue
                        
                        for det_off_ns, det_on_ns in zip(det_offs, det_ons):
                            det_off_ts = pd.Timestamp(det_off_ns)
                            det_on_ts = pd.Timestamp(det_on_ns)
                            
                            # Skip if pair is outside signal window or spans it
                            if det_on_ts < sig_start_ts or det_off_ts > sig_end_ts:
                                continue
                            if det_off_ts < sig_start_ts and det_on_ts > sig_end_ts:
                                continue
                            
                            # V1 occupation check: filter ON→OFF events per channel
                            # to signal-type ±1min buffer, check complete sequences (numpy optimized)
                            is_on = False
                            for ch in channels:
                                if ch not in on_intervals:
                                    continue
                                starts, ends = on_intervals[ch]
                                mask_on = (starts >= buf_start_ns) & (ends <= buf_end_ns) & \
                                          (starts < det_off_ns) & (ends > det_on_ns)
                                if np.any(mask_on):
                                    is_on = True
                                    break
                            
                            if is_on:
                                is_on = False
                                continue
                            
                            # Compute gap: clipped overlap with signal type window
                            max_st = max(det_off_ns, sig_start_ns)
                            min_et = min(det_on_ns, sig_end_ns)
                            
                            time_diff = round((min_et - max_st) / 1e9, 4)
                            if time_diff != 0:
                                gaps.append(time_diff)
                    
                    row_dict[f"{signal_type}GapPhase{phase_no}"] = gaps
            
            result_rows.append(row_dict)
    
    else:
        # ===== PET MODE =====
        # Pre-build OFF→ON (81→82) pairs per (phase, channel)
        off_on_pairs = _build_sequence_pairs(df_events, phase_nos, [81, 82])
        
        result_rows = []
        
        for i in range(n_cycles):
            cycle_begin_ns = cycle_begins_arr[i]
            cycle_end_ns = cycle_ends_arr[i]
            
            row_dict = {
                'signalID': df_cycle.loc[i, 'signalID'],
                'cycleNo': df_cycle.loc[i, 'cycleNo'],
                'cycleBegin': df_cycle.loc[i, 'cycleBegin'],
                'cycleEnd': df_cycle.loc[i, 'cycleEnd'],
                'cycleLength': df_cycle.loc[i, 'cycleLength'],
            }
            
            for phase_no in phase_nos:
                lane_types = df_config[df_config['phaseNo'] == phase_no]['laneType'].unique().tolist()
                
                for lane_type in lane_types:
                    ch_nos = df_config[
                        (df_config['phaseNo'] == phase_no) & 
                        (df_config['laneType'] == lane_type)
                    ]['channelNo'].unique().tolist()
                    
                    row_dict[f"channelNos{phase_no}{lane_type}"] = ch_nos
                    for sig_type in ['green', 'yellow', 'redClearance', 'red']:
                        row_dict[f"{sig_type}PETPhase{phase_no}{lane_type}"] = [[np.nan]] * len(ch_nos)
            
            for phase_no in phase_nos:
                signal_types = ['green', 'yellow', 'redClearance', 'red']
                lane_types = df_config[df_config['phaseNo'] == phase_no]['laneType'].unique().tolist()
                
                for signal_type in signal_types:
                    intervals = interval_lookup.get((i, phase_no, signal_type))
                    if not intervals:
                        continue
                    
                    for lane_type in lane_types:
                        ch_nos = df_config[
                            (df_config['phaseNo'] == phase_no) & 
                            (df_config['laneType'] == lane_type)
                        ]['channelNo'].unique().tolist()
                        
                        for channel_no in ch_nos:
                            pairs = off_on_pairs.get((phase_no, channel_no), np.empty((0, 2), dtype=np.int64))
                            if len(pairs) == 0:
                                continue
                            
                            gaps = [np.nan]
                            
                            for sig_start_ns, sig_end_ns in intervals:
                                # Filter pairs within signal window
                                mask = (pairs[:, 0] >= sig_start_ns) & \
                                       (pairs[:, 1] <= sig_end_ns)
                                filtered = pairs[mask]
                                
                                # Also within the cycle
                                mask2 = (filtered[:, 0] >= cycle_begin_ns) & \
                                        (filtered[:, 1] <= cycle_end_ns)
                                filtered = filtered[mask2]
                                
                                for p_idx in range(len(filtered)):
                                    det_off_ns = filtered[p_idx, 0]
                                    det_on_ns = filtered[p_idx, 1]
                                    
                                    if det_off_ns > sig_end_ns or det_on_ns < sig_start_ns:
                                        continue
                                    
                                    time_diff = round((det_on_ns - det_off_ns) / 1e9, 4)
                                    gaps.append(time_diff)
                            
                            ch_col = f"channelNos{phase_no}{lane_type}"
                            pet_col = f"{signal_type}PETPhase{phase_no}{lane_type}"
                            
                            if ch_col in row_dict:
                                ch_list = row_dict[ch_col]
                                if channel_no in ch_list:
                                    idx = ch_list.index(channel_no)
                                    row_dict[pet_col][idx] = gaps
            
            result_rows.append(row_dict)
    
    result = pd.DataFrame(result_rows)
    
    # Add date
    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    # Calculate statistics (Min/Max/Avg/Std)
    stat_cols = [col for col in result.columns if 'Gap' in col or 'PET' in col]
    stat_cols = [col for col in stat_cols if not col.startswith('channelNos')]
    result = calculate_signal_type_stats(result, stat_cols)
    
    # Save
    cache.save_feature(result, 'vehicle_traffic', feature_name)
    
    logging.info(f"{extraction_type} extracted: {len(result)} cycles")
    return result
