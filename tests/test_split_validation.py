"""Raw/split identity checks independent of filename calendar-day assumptions."""
import pandas as pd
import pytest

from feature_extractor.split_validation import validate_split_source


def events(*times):
    return pd.DataFrame({'timeStamp': pd.to_datetime(list(times))})


def source(hours, *, first='2026-09-06T00:00:00+00:00', last='2026-09-06T23:59:59+00:00', rows=None):
    return {'min_timestamp': first, 'max_timestamp': last,
            'timestamp_basis': 'UTC' if pd.Timestamp(first).tzinfo else 'unconfirmed_wall_time',
            'expected_signal': {'row_count': sum(hours.values()) if rows is None else rows,
                                'hourly_counts': hours}}


def test_utc_observations_match_previous_eastern_calendar_day():
    evidence = source({'2026-09-06T00:00:00+00:00': 1, '2026-09-06T23:00:00+00:00': 1})
    validate_split_source(events('2026-09-05T20:00:00', '2026-09-06T19:59:59'), evidence)


def test_naive_raw_observations_are_not_assumed_utc():
    evidence = source({'2026-09-06T00:00:00': 1, '2026-09-06T23:00:00': 1},
                      first='2026-09-06T00:00:00', last='2026-09-06T23:59:59')
    validate_split_source(events('2026-09-06T00:00:00', '2026-09-06T23:59:59'), evidence)


def test_wrong_date_content_is_rejected_even_with_matching_row_count():
    evidence = source({'2026-09-06T12:00:00+00:00': 2})
    with pytest.raises(ValueError, match='source_time_range_mismatch'):
        validate_split_source(events('2026-08-26T08:00:00', '2026-08-26T08:30:00'), evidence)


def test_split_row_count_must_match_signal_specific_raw_count():
    evidence = source({'2026-09-06T12:00:00+00:00': 2})
    with pytest.raises(ValueError, match='source_signal_row_count_mismatch'):
        validate_split_source(events('2026-09-06T08:00:00'), evidence)


def test_same_total_different_hourly_distribution_is_rejected():
    evidence = source({'2026-09-06T12:00:00+00:00': 1, '2026-09-06T13:00:00+00:00': 1})
    with pytest.raises(ValueError, match='source_signal_hourly_counts_mismatch'):
        validate_split_source(events('2026-09-06T08:00:00', '2026-09-06T08:30:00'), evidence)


def test_fall_dst_repeated_eastern_hour_sums_distinct_utc_hours():
    evidence = source({'2026-11-01T05:00:00+00:00': 2, '2026-11-01T06:00:00+00:00': 1},
                      first='2026-11-01T04:00:00+00:00', last='2026-11-01T08:00:00+00:00')
    validate_split_source(events('2026-11-01T01:05:00', '2026-11-01T01:15:00', '2026-11-01T01:55:00'), evidence)


def test_partial_fall_back_interval_has_repeated_local_hour_envelope():
    evidence = source({'2026-11-01T05:00:00+00:00': 1, '2026-11-01T06:00:00+00:00': 1},
                      first='2026-11-01T05:30:00+00:00', last='2026-11-01T06:20:00+00:00')
    validate_split_source(events('2026-11-01T01:59:00', '2026-11-01T01:01:00'), evidence)


def test_spring_dst_gap_does_not_invent_a_missing_eastern_hour():
    evidence = source({'2026-03-08T06:00:00+00:00': 1, '2026-03-08T07:00:00+00:00': 1},
                      first='2026-03-08T06:00:00+00:00', last='2026-03-08T08:00:00+00:00')
    validate_split_source(events('2026-03-08T01:15:00', '2026-03-08T03:15:00'), evidence)


def test_zero_count_hour_entries_do_not_create_phantom_events():
    evidence = source({'2026-09-06T12:00:00+00:00': 1, '2026-09-06T13:00:00+00:00': 0})
    validate_split_source(events('2026-09-06T08:15:00'), evidence)


def test_nanosecond_bounds_are_preserved():
    evidence = source({'2026-09-06T12:00:00+00:00': 1},
                      first='2026-09-06T12:00:00.123456789+00:00',
                      last='2026-09-06T12:00:00.123456789+00:00')
    validate_split_source(events('2026-09-06T08:00:00.123456789'), evidence)
    with pytest.raises(ValueError, match='source_time_range_mismatch'):
        validate_split_source(events('2026-09-06T08:00:00.123456788'), evidence)


@pytest.mark.parametrize('evidence', [{}, {'expected_signal': None}])
def test_missing_signal_evidence_cannot_validate(evidence):
    with pytest.raises(ValueError, match='^source_signal_evidence_missing$'):
        validate_split_source(events('2026-09-06T08:00:00'), evidence)


def test_inconsistent_raw_hourly_evidence_is_rejected():
    evidence = source({'2026-09-06T12:00:00+00:00': 1}, rows=2)
    with pytest.raises(ValueError, match='source_signal_evidence_invalid'):
        validate_split_source(events('2026-09-06T08:00:00', '2026-09-06T08:30:00'), evidence)


def test_mixed_source_timezone_evidence_is_rejected():
    evidence = source({'2026-09-06T12:00:00': 1})
    with pytest.raises(ValueError, match='source_signal_evidence_invalid'):
        validate_split_source(events('2026-09-06T08:00:00'), evidence)


def test_missing_event_timestamp_is_rejected():
    evidence = source({'2026-09-06T12:00:00+00:00': 1})
    with pytest.raises(ValueError, match='split_timestamp_invalid'):
        validate_split_source(pd.DataFrame({'timeStamp': [pd.NaT]}), evidence)
