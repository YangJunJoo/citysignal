"""Versioned vehicle states from adjacent observed events; disjoint cycle windows.

Naive input timestamps are already local wall time. No timezone offset is added.
Missing transitions remain unknown; no cycle-wide red inference is performed.
"""
from bisect import bisect_right
import re
import numpy as np
import pandas as pd

PROFILE_VERSION = 'observed-state-4-mapping-evidence'
COLORS = ('green', 'yellow', 'redClearance', 'red')
DEFAULT_BARRIERS = {1: 2, 2: 2, 5: 2, 6: 2, 3: 1, 4: 1, 7: 1, 8: 1}
EVENT_KEYS = {1: ('greenBegin',), 8: ('greenEnd', 'yellowBegin'),
              10: ('yellowEnd', 'redClearanceBegin'), 11: ('redClearanceEnd',)}
FORWARD = {(1, 8), (8, 10), (10, 11)}
TIME_COLS = [color + end for color in COLORS for end in ('Begin', 'End')]


class ProfileQualityError(ValueError):
    pass


def build_pedestrian_phase_profile(events, signal_id, date_str):
    """Build walk/clearance sequences from observed 21, 22 and 23 events.

    Partial and repeated sequences stay visible with correctSequenceFlag=0;
    they cannot become complete pedestrian cycles downstream.
    """
    codes = (21, 22, 23)
    keys = {21: ('pedestrianWalkBegin',),
            22: ('pedestrianWalkEnd', 'pedestrianClearanceBegin'),
            23: ('pedestrianClearanceEnd', 'pedestrianDontWalkBegin')}
    data = events.loc[events.eventCode.isin(codes),
                      ['timeStamp', 'eventCode', 'eventParam']].copy()
    data.timeStamp = pd.to_datetime(data.timeStamp)
    if data.timeStamp.isna().any():
        raise ProfileQualityError('missing_signal_timestamp')
    data = data.sort_values(['timeStamp', 'eventCode'], kind='stable')
    rows = []
    for phase, group in data.groupby('eventParam', sort=True):
        current, observed = {}, []
        for timestamp, code in group[['timeStamp', 'eventCode']].itertuples(index=False, name=None):
            if observed and tuple(observed + [code]) != codes[:len(observed) + 1]:
                current['correctSequenceFlag'] = int(tuple(observed) == codes)
                rows.append(current)
                current, observed = {}, []
            if not current:
                current = {'signalID': str(signal_id), 'phaseNo': int(phase)}
            for key in keys[code]:
                current[key] = timestamp
            observed.append(code)
        if current:
            current['correctSequenceFlag'] = int(tuple(observed) == codes)
            rows.append(current)
    times = [key for values in keys.values() for key in values]
    result = pd.DataFrame(rows).reindex(columns=['signalID', 'phaseNo', 'correctSequenceFlag', *times, 'date'])
    for column in times:
        result[column] = pd.to_datetime(result[column])
    result['date'] = pd.Timestamp(date_str).date()
    if not result.empty:
        result['_sort'] = result[times].bfill(axis=1).iloc[:, 0]
        # Retain the original timestamp-only sort, including ordering of ties.
        result = result.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    return result


def build_pedestrian_cycle_profile(phases, vehicle_cycles, signal_id, date_str):
    """Assign complete pedestrian sequences to half-open vehicle cycle windows."""
    if phases.empty or vehicle_cycles.empty:
        return pd.DataFrame()
    complete = phases.loc[phases.correctSequenceFlag.eq(1)].copy()
    if complete.empty:
        return pd.DataFrame()
    result = pd.merge_asof(
        complete.sort_values('pedestrianWalkBegin'),
        vehicle_cycles[['cycleNo', 'cycleBegin', 'cycleEnd']].sort_values('cycleBegin'),
        left_on='pedestrianWalkBegin', right_on='cycleBegin', direction='backward')
    result = result.loc[(result.pedestrianWalkBegin >= result.cycleBegin) &
                        (result.pedestrianWalkBegin < result.cycleEnd)].copy()
    result['signalID'] = str(signal_id)
    result['date'] = pd.Timestamp(date_str).date()
    return result.sort_values(['cycleNo', 'phaseNo']).reset_index(drop=True)


def build_vehicle_phase_profile(events, signal_id, date_str, barrier_map=None):
    barriers = DEFAULT_BARRIERS if barrier_map is None else barrier_map
    data = events.loc[events.eventCode.isin(EVENT_KEYS), ['timeStamp', 'eventCode', 'eventParam']].copy()
    data.timeStamp = pd.to_datetime(data.timeStamp)
    if data.timeStamp.isna().any():
        raise ProfileQualityError('missing_signal_timestamp')
    data = data.drop_duplicates().sort_values(['eventParam', 'timeStamp', 'eventCode'], kind='stable')
    rows = []
    for phase, group in data.groupby('eventParam', sort=True):
        current = {}; previous = None; observed = []
        for timestamp, code in group[['timeStamp', 'eventCode']].itertuples(index=False, name=None):
            if current and (previous, code) not in FORWARD:
                if previous == 11 and code == 1:
                    current['redBegin'] = current['redClearanceEnd']
                    current['redEnd'] = timestamp
                current['correctSequenceFlag'] = int(observed == [1, 8, 10, 11])
                rows.append(current)
                current = {}; observed = []
            if not current:
                current = {'signalID': str(signal_id), 'phaseNo': int(phase),
                           'barrierNo': int(barriers.get(int(phase), 0))}
            for key in EVENT_KEYS[code]:
                current[key] = timestamp
            observed.append(code); previous = code
        if current:
            current['correctSequenceFlag'] = int(observed == [1, 8, 10, 11])
            rows.append(current)
    columns = ['signalID', 'phaseNo', 'correctSequenceFlag', 'barrierNo', *TIME_COLS, 'date']
    result = pd.DataFrame(rows).reindex(columns=columns)
    for col in TIME_COLS:
        result[col] = pd.to_datetime(result[col])
    result['date'] = pd.Timestamp(date_str).date()
    if not result.empty:
        result['_sort'] = result[TIME_COLS].bfill(axis=1).iloc[:, 0]
        result = result.sort_values(['_sort', 'phaseNo'], kind='stable').drop(columns='_sort').reset_index(drop=True)
    result = _respect_termination_events(events, result)
    result.attrs.update(profile_version=PROFILE_VERSION, interval_closure='left',
                        timestamp_basis='America/New_York wall time',
                        state_evidence='observed starts, ends and reset guards; no bridging missing transitions')
    return result


def _respect_termination_events(events, profile):
    profile=profile.copy()
    endpoints={color:{} for color in COLORS}
    relevant=events.loc[events.eventCode.isin([0,1,7,8,9,10,11,12]),['eventParam','timeStamp','eventCode']].drop_duplicates().sort_values(['eventParam','timeStamp','eventCode'])
    relevant.timeStamp = pd.to_datetime(relevant.timeStamp)
    pairs={(1,7):'green',(1,8):'green',(8,9):'yellow',(8,10):'yellow',(10,11):'redClearance',(11,0):'red',(11,1):'red'}
    for ph,g in relevant.groupby('eventParam'):
        previous=None
        for code,timestamp in g[['eventCode','timeStamp']].itertuples(index=False,name=None):
            # Inactive after end-red-clearance does not change the red indication.
            if previous is not None and previous[0]==11 and code==12:continue
            if previous:
                old,a=previous;color=pairs.get((old,code))
                if color and a<timestamp:endpoints[color][(int(ph),a)]=timestamp
            previous=(code,timestamp)
    original_red_start=profile.redClearanceEnd.copy()
    for color in COLORS:
        starts=original_red_start if color=='red' else profile[color+'Begin']
        ends=pd.to_datetime([endpoints[color].get((int(ph),a),pd.NaT) for ph,a in zip(profile.phaseNo,starts)])
        profile[color+'End']=ends
        if color=='red':profile['redBegin']=starts.where(pd.notna(ends))
    profile['correctSequenceFlag']=(profile.correctSequenceFlag.eq(1)&profile[['greenEnd','yellowEnd','redClearanceEnd']].notna().all(axis=1)).astype(int)
    return profile


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _intersects(a, b):
    a, b = _union(a), _union(b); i = j = 0
    while i < len(a) and j < len(b):
        if max(a[i][0], b[j][0]) < min(a[i][1], b[j][1]):
            return True
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return False


def build_vehicle_cycle_profile(phases, signal_id, date_str, start_barrier_no=1, conflict_policy='reject'):
    if conflict_policy not in ('reject', 'quarantine'):
        raise ValueError('Unknown barrier conflict policy')
    if phases.empty:
        return pd.DataFrame()
    if phases.attrs.get('profile_version') != PROFILE_VERSION:
        raise ProfileQualityError('stale_phase_profile: regenerate vehicle phase from raw events')
    if start_barrier_no not in (1, 2):
        raise ValueError('start_barrier_no must be 1 or 2')
    known = {}
    greens = {1: [], 2: []}
    for phase, group in phases.groupby('phaseNo'):
        for color in COLORS:
            pairs = [(a, b) for a, b in group[[color+'Begin', color+'End']].itertuples(index=False, name=None)
                     if pd.notna(a) and pd.notna(b) and a < b]
            known[int(phase), color] = pairs
            if color == 'green':
                barrier = int(group.barrierNo.iloc[0])
                if barrier in greens: greens[barrier].extend(pairs)
    # Quarantine whole conflicting green spans, never just their intersection.
    # Preserve observed states; omit every cycle touching uncertain operation.
    left, right = _union(greens[1]), _union(greens[2])
    i = j = 0; quarantine = []
    while i < len(left) and j < len(right):
        if max(left[i][0], right[j][0]) < min(left[i][1], right[j][1]):
            quarantine.extend((left[i], right[j]))
        if left[i][1] < right[j][1]: i += 1
        else: j += 1
    if quarantine and conflict_policy == 'reject':
        raise ProfileQualityError('inconsistent_barrier_mapping: observed greens cross configured barriers')
    starts = phases.loc[phases.barrierNo.isin([1, 2]) & phases.greenBegin.notna()]
    bounds = []; previous = None
    for timestamp, group in starts.groupby('greenBegin', sort=True):
        barriers = set(group.barrierNo)
        if len(barriers) != 1:
            if conflict_policy == 'reject':
                raise ProfileQualityError('ambiguous_barrier_start: multiple barriers at one timestamp')
            quarantine.append((timestamp, timestamp + pd.Timedelta(1, unit='ns')))
            previous = None  # Require a fresh observed barrier transition.
            continue
        barrier = next(iter(barriers))
        if barrier == start_barrier_no and previous == 3 - start_barrier_no:
            bounds.append(timestamp)
        previous = barrier
    if len(bounds) < 2:
        return pd.DataFrame()
    rows = [{'signalID': str(signal_id), 'cycleNo': i+1, 'cycleBegin': a, 'cycleEnd': b,
             'cycleLength': (b-a).total_seconds(), 'date': pd.Timestamp(date_str).date()}
            for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    for (phase, color), intervals in known.items():
        column = f'{color}Phase{phase}'
        for row in rows: row[column] = []
        for start, end in intervals:
            i = max(0, bisect_right(bounds, start)-1)
            while i < len(rows) and bounds[i] < end:
                a, b = max(start, bounds[i]), min(end, bounds[i+1])
                if a < b: rows[i][column].append((a, b))
                i += 1
    for row in rows:
        for phase in sorted(phases.phaseNo.unique()):
            all_known = sorted((a, b) for color in COLORS for a, b in row[f'{color}Phase{phase}'])
            unknown = []; cursor = row['cycleBegin']
            for start, end in all_known:
                if start < cursor:
                    raise ProfileQualityError(f'overlapping_states: phase {phase}')
                if cursor < start: unknown.append((cursor, start))
                cursor = end
            if cursor < row['cycleEnd']: unknown.append((cursor, row['cycleEnd']))
            row[f'unknownPhase{phase}'] = unknown
    result = pd.DataFrame(rows)
    quarantine = _union(quarantine)
    omitted = np.zeros(len(result), dtype=bool)
    for a, b in quarantine:
        omitted |= ((result.cycleBegin < b) & (result.cycleEnd > a)).to_numpy()
    candidate_count = len(result)
    if omitted.any():
        result = result.loc[~omitted].copy().reset_index(drop=True)
        result['cycleNo'] = np.arange(1, len(result) + 1)
    result.attrs.update(barrier_conflict_policy=conflict_policy,
                        candidate_cycle_count=candidate_count,
                        quarantined_cycle_count=int(omitted.sum()),
                        quarantine_windows=quarantine,
                        coverage_scope='retained_cycle_windows_only' if quarantine else 'observed_cycle_window',
                        barrier_consistency='checked_on_retained_windows')
    result.attrs.update(profile_version=PROFILE_VERSION, interval_closure='left',
                        timestamp_basis='America/New_York wall time',
                        boundary_policy='mapped barrier transition; simultaneous starts grouped',
                        boundary_coverage='first and last incomplete cycles excluded')
    validate_cycle_profile(result)
    return result


def validate_cycle_profile(frame):
    if frame.empty or frame.attrs.get('profile_version') != PROFILE_VERSION:
        return
    begin, end = frame.cycleBegin, frame.cycleEnd
    if begin.isna().any() or end.isna().any() or (begin >= end).any() or not begin.is_monotonic_increasing:
        raise ProfileQualityError('invalid_cycle_boundaries')
    if (begin < end.cummax().shift()).any():
        raise ProfileQualityError('overlapping_cycles')
    phases = sorted({int(m.group(1)) for col in frame for m in [re.fullmatch(r'greenPhase(\d+)', col)] if m})
    for row in frame.itertuples(index=False):
        for phase in phases:
            spans = sorted((a,b) for color in (*COLORS, 'unknown') for a,b in getattr(row, f'{color}Phase{phase}'))
            cursor = row.cycleBegin
            for a,b in spans:
                if a != cursor or not a < b or b > row.cycleEnd:
                    raise ProfileQualityError('states_and_unknown_must_partition_cycle')
                cursor = b
            if cursor != row.cycleEnd:
                raise ProfileQualityError('state_coverage_gap')


def apply_state_quality(frame, cycles, feature_name):
    """Keep known SPaT durations, but do not present partial-cycle traffic estimates as complete."""
    if frame.empty or cycles.attrs.get('profile_version') != PROFILE_VERSION:
        return frame
    frame.attrs.update(profile_version=PROFILE_VERSION, interval_closure='left')
    for key in ('coverage_scope', 'candidate_cycle_count', 'quarantined_cycle_count'):
        if key in cycles.attrs: frame.attrs[key] = cycles.attrs[key]
    if feature_name == 'spat':
        return frame
    unknown = {}
    for col in cycles:
        match = re.fullmatch(r'unknownPhase(\d+)', col)
        if match:
            unknown[int(match.group(1))] = dict(zip(cycles.cycleNo, cycles[col].map(bool)))
    masked = 0
    meta = {'signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength', 'date'}
    for col in frame:
        if col in meta or str(col).startswith('channelNos') or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        match = re.search(r'Phase(\d+)', str(col))
        maps = [unknown.get(int(match.group(1)), {})] if match else list(unknown.values())
        mask = pd.Series(False, index=frame.index)
        for mapping in maps:
            mask |= frame.cycleNo.map(mapping).fillna(False).astype(bool)
        masked += int(mask.sum())
        if mask.any(): frame.loc[mask, col] = np.nan
    frame.attrs['state_quality_masked_values'] = masked
    frame.attrs['state_quality_policy'] = 'NaN for phase-dependent metrics when that phase has unknown time in cycle'
    return frame


def before_interval_end(cache, values, ends):
    """New profiles use [begin,end); legacy files retain their historical contract."""
    if getattr(cache, 'df_vehicle_cycle', pd.DataFrame()).attrs.get('interval_closure') == 'left':
        return values < ends
    return values <= ends
