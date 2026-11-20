"""
Pedestrian feature extractors — vectorized.

Includes:
- extract_pedestrian_activity: Binary flag for pedestrian actuations per cycle/phase
  - Uses cycle windows to check event presence
- extract_pedestrian_delay: Computes delay from button press to walk begin
"""

import pandas as pd
import numpy as np
import logging


def extract_pedestrian_activity(cache) -> pd.DataFrame:
    """
    Extract pedestrian activity (binary flags) per cycle/phase.
    Uses pre-indexed cycle lookup instead of per-cycle DataFrame filtering.
    """
    df_ped_cycle = cache.df_pedestrian_cycle
    df_veh_cycle = cache.df_vehicle_cycle
    
    if len(df_ped_cycle) == 0:
        return pd.DataFrame()
    
    # Get events [45, 90]
    df_events = cache.get_events((45, 90))
    df_events['eventCode'] = df_events['eventCode'].astype(int)
    
    if len(df_events) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'pedestrian_traffic', 'activity')
        return result
    
    # Phase nos for pedestrian (2, 4, 6, 8)
    phase_nos = sorted(set(df_events['eventParam'].unique()).intersection({2, 4, 6, 8}))
    event_codes = [45, 90]
    
    # Pre-compute: assign events to ped cycles using searchsorted
    cycle_nos = sorted(df_ped_cycle['cycleNo'].unique())
    vehicle_cycle_nos = set(df_veh_cycle['cycleNo'].unique()) if len(df_veh_cycle) > 0 else set()
    
    # Pre-index ped cycle data
    ped_cycle_dict = {}
    for _, row in df_ped_cycle.iterrows():
        ped_cycle_dict[row['cycleNo']] = {
            'signalID': row['signalID'],
            'cycleBegin': row['cycleBegin'],
            'cycleEnd': row['cycleEnd'],
        }
    
    # Pre-index vehicle cycles
    veh_cycle_dict = {}
    for _, row in df_veh_cycle.iterrows():
        veh_cycle_dict[row['cycleNo']] = {
            'cycleBegin': row['cycleBegin'],
            'cycleEnd': row['cycleEnd'],
        }
    
    # Pre-sort events
    event_ts = df_events['timeStamp'].values
    event_params = df_events['eventParam'].values.astype(int)
    event_codes_arr = df_events['eventCode'].values.astype(int)
    sort_idx = np.argsort(event_ts)
    event_ts = event_ts[sort_idx]
    event_params = event_params[sort_idx]
    event_codes_arr = event_codes_arr[sort_idx]
    
    rows = []
    
    for cycle_no in cycle_nos:
        cd = ped_cycle_dict.get(cycle_no)
        if cd is None:
            continue
        
        signal_id_val = cd['signalID']
        cycle_begin = cd['cycleBegin']
        cycle_end = cd['cycleEnd']
        
        # Determine prev cycle window
        if (cycle_no - 1) in vehicle_cycle_nos:
            prev = veh_cycle_dict[cycle_no - 1]
            cycle_begins = [prev['cycleBegin'], cycle_begin]
            cycle_ends = [prev['cycleEnd'], cycle_end]
        else:
            cycle_begins = [cycle_begin - pd.Timedelta("1.5min"), cycle_begin]
            cycle_ends = [cycle_begin, cycle_end]
        
        row = {
            'signalID': signal_id_val,
            'cycleNo': cycle_no,
            'cycleBegin': cycle_begin,
            'cycleEnd': cycle_end,
        }
        
        for phase_no in phase_nos:
            for ec in event_codes:
                row[f"pedestrianActivity{ec}Phase{phase_no}"] = 0
                row[f"pedestrianActivity{ec}Phase{phase_no}CurrCycle"] = 0
                row[f"pedestrianActivity{ec}Phase{phase_no}PrevCycle"] = 0
        
        cycle_types = ["Prev", "Curr"]
        
        for idx, (start_time, end_time) in enumerate(zip(cycle_begins, cycle_ends)):
            # Binary search for events in window
            start_ns = pd.Timestamp(start_time).value
            end_ns = pd.Timestamp(end_time).value
            
            i_start = np.searchsorted(event_ts, start_ns, side='left')
            i_end = np.searchsorted(event_ts, end_ns, side='right')
            
            if i_start >= i_end:
                continue
            
            window_params = event_params[i_start:i_end]
            window_codes = event_codes_arr[i_start:i_end]
            
            for phase_no in phase_nos:
                phase_mask = window_params == phase_no
                if not np.any(phase_mask):
                    continue
                
                phase_codes = window_codes[phase_mask]
                
                for ec in event_codes:
                    ec_mask = phase_codes == ec
                    if not np.any(ec_mask):
                        continue
                    
                    # Presence is unchanged by deduplication of a nonempty set.
                    row[f"pedestrianActivity{ec}Phase{phase_no}{cycle_types[idx]}Cycle"] = 1
                    row[f"pedestrianActivity{ec}Phase{phase_no}"] = 1
                    break  # Preserve the original event-code precedence.
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    if len(result) > 0:
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    cache.save_feature(result, 'pedestrian_traffic', 'activity')
    
    logging.info(f"Pedestrian activity extracted: {len(result)} cycles")
    return result


def extract_pedestrian_delay(cache) -> pd.DataFrame:
    """
    Extract pedestrian delay per cycle.
    Uses pre-indexed cycle lookup instead of per-cycle DataFrame filtering.
    """
    df_ped_cycle = cache.df_pedestrian_cycle
    df_veh_cycle = cache.df_vehicle_cycle
    
    if len(df_ped_cycle) == 0:
        result = pd.DataFrame()
        cache.save_feature(result, 'pedestrian_traffic', 'delay')
        return result
    
    df_events = cache.get_events((45, 90))
    df_events = df_events.drop_duplicates(subset=['timeStamp', 'eventParam']).reset_index(drop=True)
    
    phase_nos = sorted(set(df_events['eventParam'].unique()).intersection({2, 4, 6, 8}))
    
    cycle_nos = sorted(df_ped_cycle['cycleNo'].unique())
    vehicle_cycle_nos = set(df_veh_cycle['cycleNo'].unique()) if len(df_veh_cycle) > 0 else set()
    
    # Pre-index cycles
    ped_cycle_dict = {}
    for _, row in df_ped_cycle.iterrows():
        key = row['cycleNo']
        if key not in ped_cycle_dict:
            ped_cycle_dict[key] = []
        ped_cycle_dict[key].append(row.to_dict())
    
    veh_cycle_dict = {}
    for _, row in df_veh_cycle.iterrows():
        veh_cycle_dict[row['cycleNo']] = {
            'cycleBegin': row['cycleBegin'],
            'cycleEnd': row['cycleEnd'],
        }
    
    # Pre-sort events for binary search
    event_ts = df_events['timeStamp'].values
    event_params = df_events['eventParam'].values
    sort_idx = np.argsort(event_ts)
    event_ts = event_ts[sort_idx]
    event_params = event_params[sort_idx]
    
    rows = []
    
    for cycle_no in cycle_nos:
        cycle_rows = ped_cycle_dict.get(cycle_no, [])
        if not cycle_rows:
            continue
        
        first_row = cycle_rows[0]
        signal_id = first_row['signalID']
        cycle_begin = first_row['cycleBegin']
        cycle_end = first_row['cycleEnd']
        
        if (cycle_no - 1) in vehicle_cycle_nos:
            prev = veh_cycle_dict[cycle_no - 1]
            cycle_begins = [prev['cycleBegin'], cycle_begin]
            cycle_ends = [prev['cycleEnd'], cycle_end]
        else:
            cycle_begins = [cycle_begin - pd.Timedelta("1.5min"), cycle_begin]
            cycle_ends = [cycle_begin, cycle_end]
        
        row = {
            'signalID': signal_id,
            'cycleNo': cycle_no,
            'cycleBegin': cycle_begin,
            'cycleEnd': cycle_end
        }
        
        for phase_no in phase_nos:
            row[f"pedestrianDelayPhase{phase_no}"] = 0
            row[f"pedestrianDelayPhase{phase_no}CurrCycle"] = 0
            row[f"pedestrianDelayPhase{phase_no}PrevCycle"] = 0
        
        # Get events in full tolerance window using binary search
        tol_start_ns = pd.Timestamp(cycle_begins[0]).value
        tol_end_ns = pd.Timestamp(cycle_ends[-1]).value
        
        i_start = np.searchsorted(event_ts, tol_start_ns, side='left')
        i_end = np.searchsorted(event_ts, tol_end_ns, side='right')
        
        if i_start >= i_end:
            rows.append(row)
            continue
        
        window_ts = event_ts[i_start:i_end]
        window_params = event_params[i_start:i_end]
        
        cycle_types = ["Prev", "Curr"]
        
        for idx_ct, (start_time, end_time) in enumerate(zip(cycle_begins, cycle_ends)):
            for phase_no in phase_nos:
                phase_mask = window_params == phase_no
                if not np.any(phase_mask):
                    continue
                
                phase_ts = pd.to_datetime(window_ts[phase_mask])
                button_press_times = phase_ts.tolist()
                
                if not button_press_times:
                    continue
                
                button_press_time = None
                
                ped_cycle_nos_set = set(df_ped_cycle['cycleNo'].unique())
                if (cycle_no - 1) in ped_cycle_nos_set:
                    prev_ped = [r for r in ped_cycle_dict.get(cycle_no - 1, [])
                                if r.get('phaseNo') == phase_no]
                    
                    if prev_ped and 'pedestrianClearanceEnd' in prev_ped[0]:
                        clearance_end = prev_ped[0]['pedestrianClearanceEnd']
                        if pd.notna(clearance_end):
                            for t in button_press_times:
                                if t > clearance_end:
                                    button_press_time = t
                                    break
                
                if button_press_time is None:
                    button_press_time = min(button_press_times)
                
                # Get walk begin times
                phase_ped_rows = [r for r in cycle_rows if r.get('phaseNo') == phase_no]
                
                for pr in phase_ped_rows:
                    if 'pedestrianWalkBegin' not in pr:
                        continue
                    walk_begin = pr['pedestrianWalkBegin']
                    
                    st = max(button_press_time, start_time)
                    et = min(walk_begin, end_time)
                    
                    if et > st:
                        delay = round((et - st).total_seconds(), 4)
                        row[f"pedestrianDelayPhase{phase_no}{cycle_types[idx_ct]}Cycle"] = delay
                        row[f"pedestrianDelayPhase{phase_no}"] += delay
        
        rows.append(row)
    
    result = pd.DataFrame(rows)
    
    if len(result) > 0:
        result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    cache.save_feature(result, 'pedestrian_traffic', 'delay')
    
    logging.info(f"Pedestrian delay extracted: {len(result)} cycles")
    return result
