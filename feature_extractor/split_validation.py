"""Compare a split signal's events with measured evidence from its raw source.

Event times have already been normalized by the checked runner: aware source
values become America/New_York wall times, while naive values remain unchanged.
No calendar-day interpretation is inferred from the input filename.
"""
from collections import Counter
from collections.abc import Mapping
from numbers import Integral

import pandas as pd

_LOCAL_ZONE = 'America/New_York'


def _timestamp(value):
    if value is None:
        raise ValueError('source_timestamp_evidence_missing')
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('source_timestamp_evidence_invalid') from exc
    if pd.isna(timestamp):
        raise ValueError('source_timestamp_evidence_invalid')
    return timestamp


def _local_naive(timestamp):
    return timestamp.tz_convert(_LOCAL_ZONE).tz_localize(None) if timestamp.tzinfo is not None else timestamp


def _local_bounds(first, last):
    if (first.tzinfo is None) != (last.tzinfo is None):
        raise ValueError('source_timestamp_evidence_invalid: mixed timestamp bases')
    if last < first:
        raise ValueError('source_timestamp_evidence_invalid: reversed observed interval')
    if first.tzinfo is None:
        return first, last
    # A fall-back interval may end earlier on the wall clock than it started.
    # Include hour boundaries (and their preceding nanosecond) so the observed
    # local envelope also includes the first and second occurrences of that hour.
    first_utc, last_utc = first.tz_convert('UTC'), last.tz_convert('UTC')
    points = [first, last]
    for boundary in pd.date_range(first_utc.ceil('h'), last_utc.floor('h'), freq='h'):
        points.append(boundary)
        before = boundary - pd.Timedelta(1, unit='ns')
        if before >= first_utc:
            points.append(before)
    local_points = [_local_naive(point) for point in points]
    return min(local_points), max(local_points)


def _count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('source_signal_evidence_invalid: counts must be nonnegative integers')
    return int(value)


def validate_split_source(events, source) -> None:
    """Raise ValueError when row counts, observed times, or hourly counts differ.

    ``source['expected_signal']`` must contain the raw-source signal row count
    and hourly counts. UTC hours that collapse to the same Eastern wall-clock
    hour at the fall DST transition are summed before comparison.
    """
    expected = source.get('expected_signal') if isinstance(source, Mapping) else None
    if expected is None:
        raise ValueError('source_signal_evidence_missing')
    if not isinstance(expected, Mapping) or 'row_count' not in expected or not isinstance(expected.get('hourly_counts'), Mapping):
        raise ValueError('source_signal_evidence_invalid')
    expected_rows = _count(expected['row_count'])
    if not isinstance(events, pd.DataFrame) or 'timeStamp' not in events:
        raise ValueError('split_timestamp_column_missing')
    if len(events) != expected_rows:
        raise ValueError(f'source_signal_row_count_mismatch: expected={expected_rows}, actual={len(events)}')

    first, last = _timestamp(source.get('min_timestamp')), _timestamp(source.get('max_timestamp'))
    lower, upper = _local_bounds(first, last)
    expected_hours = Counter()
    for label, count in expected['hourly_counts'].items():
        timestamp = _timestamp(label)
        if (timestamp.tzinfo is None) != (first.tzinfo is None):
            raise ValueError('source_signal_evidence_invalid: hourly timestamp base differs from observed bounds')
        count = _count(count)
        if count:
            expected_hours[_local_naive(timestamp).floor('h')] += count
    if sum(expected_hours.values()) != expected_rows:
        raise ValueError('source_signal_evidence_invalid: hourly counts do not sum to row_count')
    try:
        times = pd.to_datetime(events['timeStamp'], errors='raise')
        if times.isna().any():
            raise ValueError('missing event time')
        if times.dt.tz is not None:
            times = times.dt.tz_convert(_LOCAL_ZONE).dt.tz_localize(None)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError('split_timestamp_invalid') from exc
    if ((times < lower) | (times > upper)).any():
        raise ValueError('source_time_range_mismatch: split events fall outside raw-source observed timestamps')
    actual_hours = Counter(times.dt.floor('h').value_counts().to_dict())
    if actual_hours != expected_hours:
        mismatched = sorted(hour.isoformat() for hour in actual_hours.keys() | expected_hours.keys()
                            if actual_hours[hour] != expected_hours[hour])
        raise ValueError('source_signal_hourly_counts_mismatch: ' + ', '.join(mismatched))
