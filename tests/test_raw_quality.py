from pathlib import Path
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_downloader.quality import (
    SourceContract, assess_parquet, assess_collection, classify_missing_dates,
)


def write_raw(tmp_path, name, timestamps, signals=None):
    path = tmp_path / name
    pq.write_table(pa.table({'timeStamp': timestamps, 'signalID': signals or ['1001'] * len(timestamps),
                            'eventCode': [81] * len(timestamps), 'eventParam': [1] * len(timestamps)}), path)
    return path


def utc_contract():
    return SourceContract(timestamp_timezone='UTC', day_timezone='UTC', evidence='test source specification')


def test_complete_day_and_zero_hour_counts(tmp_path):
    ts = pd.date_range('2026-08-26', periods=48, freq='30min', tz='UTC')
    path = write_raw(tmp_path, 'atspm-2026-8-26.parquet', ts)
    report = assess_parquet(path, contract=utc_contract())
    assert report['status'] == 'valid'
    assert report['expected_coverage_verified']
    assert report['row_count'] == report['valid_timestamp_count'] == 48
    assert len(report['hourly_counts']) == 24
    assert set(report['hourly_counts'].values()) == {2}


@pytest.mark.parametrize('d, hours', [('2026-03-08', 23), ('2026-11-01', 25)])
def test_dst_calendar_windows(tmp_path, d, hours):
    contract = SourceContract(timestamp_timezone='UTC', day_timezone='America/New_York', evidence='test source specification')
    begin, end = contract.expected_bounds(d)
    path = write_raw(tmp_path, f'atspm-{d}.parquet', pd.date_range(begin, end, freq='30min', inclusive='left'))
    report = assess_parquet(path, contract=contract)
    assert report['status'] == 'valid'
    assert report['expected_interval']['hours'] == hours
    assert len(report['hourly_counts']) == hours


def test_unknown_contract_does_not_assume_twenty_to_twenty(tmp_path):
    ts = pd.date_range('2026-08-25 20:00', periods=48, freq='30min')
    report = assess_parquet(write_raw(tmp_path, 'atspm-2026-8-26.parquet', ts))
    assert report['status'] == 'valid'
    assert report['expected_interval'] is None
    assert not report['expected_coverage_verified']
    assert report['contract_status'] == 'unconfirmed'


def test_source_fractional_seconds_and_invalid_timestamp(tmp_path):
    report = assess_parquet(write_raw(tmp_path, 'atspm-2026-8-26.parquet',
        ['08-26-2026 06:10:59.900', 'invalid', None, '08-26-2026 06:00:00.123456789']))
    assert report['invalid_timestamp_count'] == 2
    assert report['valid_timestamp_count'] == 2
    assert report['min_timestamp'].endswith('00.123456789')
    assert report['max_timestamp'].endswith('59.900000')
    assert report['status'] == 'source_incomplete'


def test_all_signal_gap_distinguished_from_quiet_signal(tmp_path):
    times = list(pd.date_range('2026-08-26', periods=48, freq='30min')) + [pd.Timestamp('2026-08-26 00:00')]
    report = assess_parquet(write_raw(tmp_path, 'atspm-2026-8-26.parquet', times, ['A']*48+['B']))
    assert report['status'] == 'valid'
    assert report['signal_count'] == 2
    assert report['signals']['B']['row_count'] == 1
    assert len(report['signals']['B']['hourly_counts']) == 1


@pytest.mark.parametrize('d,next_d,last,gap', [
    ('2026-8-26','2026-8-27','08-26-2026 06:10:59.900',64140.1),
    ('2026-9-1','2026-9-2','09-01-2026 06:26:59.900',63180.1),
])
def test_known_cross_file_gaps(tmp_path,d,next_d,last,gap):
    p1 = write_raw(tmp_path, f'atspm-{d}.parquet', [last])
    next_stamp = pd.Timestamp(next_d).strftime('%m-%d-%Y 00:00:00.000')
    p2 = write_raw(tmp_path, f'atspm-{next_d}.parquet', [next_stamp])
    result = assess_collection([p1,p2])
    assert result['adjacent_gaps'][0]['seconds'] == pytest.approx(gap)
    assert result['status'] == 'partial'
    assert not result['accepted_paths']


def test_date_boundary_overlap(tmp_path):
    first = write_raw(tmp_path, 'atspm-2026-8-26.parquet', ['08-27-2026 00:00:01'])
    second = write_raw(tmp_path, 'atspm-2026-8-27.parquet', ['08-27-2026 00:00:00'])
    result = assess_collection([first, second])
    assert len(result['overlaps']) == 1
    assert result['overlaps'][0]['seconds'] == -1
    assert not result['accepted_paths']


def test_internal_gap_and_expected_edges(tmp_path):
    path = write_raw(tmp_path, 'atspm-2026-8-26.parquet', pd.to_datetime(['2026-08-26 03:00','2026-08-26 05:00']))
    report = assess_parquet(path, contract=utc_contract())
    assert {'internal_gap','leading_gap','trailing_gap'} <= set(report['reason_codes'])
    assert report['hourly_counts']['2026-08-26T04:00:00+00:00'] == 0


def test_cache_invalidates_on_file_and_contract_changes(tmp_path):
    path = write_raw(tmp_path, 'atspm-2026-8-26.parquet', ['08-26-2026 00:00:00'])
    cache = tmp_path/'cache'
    first = assess_parquet(path, cache_dir=cache)
    assert assess_parquet(path, cache_dir=cache)['cache_hit']
    contracted = assess_parquet(path, cache_dir=cache, contract=utc_contract())
    assert not contracted['cache_hit']
    assert contracted['input_version'] != first['input_version']
    write_raw(tmp_path, path.name, ['08-26-2026 00:00:01'])
    replaced = assess_parquet(path, cache_dir=cache)
    assert not replaced['cache_hit']
    assert replaced['input_version'] != first['input_version']


def test_only_trailing_confirmed_404_is_pending():
    dates = ['2026-09-01','2026-09-02','2026-09-03','2026-09-04']
    results = classify_missing_dates(dates, ['2026-09-02'], {d:404 for d in dates})
    assert [r['status'] for r in results] == ['source_incomplete','source_pending','source_pending']
    assert classify_missing_dates(['2026-09-03'], [], {'2026-09-03':404})[0]['status'] == 'source_incomplete'
    assert classify_missing_dates(dates, ['2026-09-02'])[1]['status'] == 'source_incomplete'


def test_broken_parquet_fails_closed(tmp_path):
    path = tmp_path/'atspm-2026-8-26.parquet'
    path.write_bytes(b'not parquet')
    assert assess_parquet(path)['reason_codes'] == ['raw_read_failed']


def test_ambiguous_local_dst_timestamp_is_counted_invalid(tmp_path):
    contract = SourceContract(timestamp_timezone='America/New_York', day_timezone='America/New_York', evidence='test source')
    path = write_raw(tmp_path, 'atspm-2026-11-1.parquet', ['11-01-2026 01:30:00', '11-01-2026 03:00:00'])
    report = assess_parquet(path, contract=contract)
    assert report['invalid_timestamp_count'] == 1
    assert 'invalid_timestamp' in report['reason_codes']


def test_missing_event_fields_is_not_accepted(tmp_path):
    path = tmp_path/'atspm-2026-8-26.parquet'
    pq.write_table(pa.table({'timeStamp':['08-26-2026 00:00:00'], 'signalID':['1001']}),path)
    report = assess_parquet(path)
    assert 'missing_raw_columns' in report['reason_codes']
    assert report['status'] == 'source_incomplete'


def test_disconnected_manifest_windows_do_not_create_false_boundary_gaps(tmp_path):
    first = write_raw(tmp_path, 'atspm-2026-8-26.parquet', ['08-26-2026 23:59:59'])
    second = write_raw(tmp_path, 'atspm-2026-9-1.parquet', ['09-01-2026 00:00:00'])
    result = assess_collection([first,second], requested_dates=['2026-08-26','2026-09-01'])
    assert result['status'] == 'complete'
    assert not result['adjacent_gaps']
    assert len(result['accepted_paths']) == 2


@pytest.mark.parametrize('timestamp', ['02-30-2026 00:00:00','02-29-2026 00:00:00','08-26-2026 00:00:60'])
def test_impossible_calendar_or_second_is_not_silently_normalized(tmp_path,timestamp):
    path=write_raw(tmp_path,'atspm-2026-8-26.parquet',[timestamp,'08-26-2026 00:00:00'])
    report=assess_parquet(path)
    assert report['invalid_timestamp_count'] == 1
    assert report['valid_timestamp_count'] == 1
    assert 'invalid_timestamp' in report['reason_codes']
