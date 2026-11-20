"""Boundary contracts that must survive removal of redundant calculations."""
import pandas as pd
import pytest

from feature_extractor.data_loader import SignalDataCache
from feature_extractor.extractors.pedestrian import extract_pedestrian_activity
from feature_extractor.extractors.shockwave import extract_shockwave


def make_cache(tmp_path, offsets, codes, params):
    start = pd.Timestamp('2026-09-06')
    cache = object.__new__(SignalDataCache)
    cache.signal_id, cache.year, cache.month, cache.day = 'S', 2026, 9, 6
    cache.date_str, cache.district = '2026-09-06', 'fdot_d5'
    cache.cwd, cache.relative_production = str(tmp_path), 'production'
    cache.df_event = pd.DataFrame({
        'signalID': ['S'] * len(offsets),
        'timeStamp': pd.to_datetime([start + pd.Timedelta(seconds=s) for s in offsets]),
        'eventCode': pd.Series(codes, dtype='int64'),
        'eventParam': pd.Series(params, dtype='int64'),
    })
    cache.df_vehicle_cycle = pd.DataFrame({
        'signalID': ['S', 'S'], 'cycleNo': [1, 2],
        'cycleBegin': [start, start + pd.Timedelta(seconds=10)],
        'cycleEnd': [start + pd.Timedelta(seconds=10), start + pd.Timedelta(seconds=20)],
        'cycleLength': [10, 10],
        'redPhase2': [[(start, start + pd.Timedelta(seconds=10))],
                      [(start + pd.Timedelta(seconds=10), start + pd.Timedelta(seconds=20))]],
    })
    cache.df_pedestrian_cycle = cache.df_vehicle_cycle.copy(deep=True)
    cache.df_config_raw = pd.DataFrame({
        'phaseNo': [2], 'channelNo': [1], 'laneType': ['Through'],
        'stopBarDistance': [100], 'countBar': [1],
    })
    cache.vehicle_phase_nos = [2]
    cache._config_cache, cache._event_cache, cache._merged_cache = {}, {}, {}
    cache._cycle_index_cache, cache._signal_intervals_cache = {}, None
    return cache


def test_shared_cycle_boundary_counts_in_both_cycles(tmp_path):
    cache = make_cache(tmp_path, [-1, 0, 10, 20, 21], [81]*5, [1]*5)
    result = cache.assign_cycle_index((81,), 'back')
    start = pd.Timestamp('2026-09-06')
    assignments = {
        offset: sorted(result.loc[result.timeStamp == start + pd.Timedelta(seconds=offset), 'cycle_idx'].tolist())
        for offset in [-1, 0, 10, 20, 21]
    }
    assert assignments == {-1: [-1], 0: [0], 10: [0, 1], 20: [1], 21: [-1]}
    assert cache.assign_cycle_index((81,), 'back') is result


@pytest.mark.parametrize('offsets,expected', [
    ([-1], [[1, 0], [0, 0]]),
    ([0], [[1, 1], [1, 0]]),
    ([10], [[0, 1], [1, 1]]),
    ([20], [[0, 0], [0, 1]]),
    ([1, 1.1, 3.499, 3.5, 6], [[0, 1], [1, 0]]),
    ([10, 10, 10.1, 12.499, 12.5], [[0, 1], [1, 1]]),
])
def test_pedestrian_presence_preserves_windows_and_repeated_events(tmp_path, offsets, expected):
    cache = make_cache(tmp_path, offsets, [45]*len(offsets), [2]*len(offsets))
    result = extract_pedestrian_activity(cache)
    columns = ['pedestrianActivity45Phase2PrevCycle', 'pedestrianActivity45Phase2CurrCycle']
    assert result[columns].values.tolist() == expected
    assert result['pedestrianActivity45Phase2'].tolist() == [int(any(row)) for row in expected]
    assert result.filter(like='pedestrianActivity90').eq(0).all().all()


def test_pedestrian_empty_events_are_no_data(tmp_path):
    assert extract_pedestrian_activity(make_cache(tmp_path, [], [], [])).empty


@pytest.mark.parametrize('off_seconds,expected', [(70, True), (70.001, False)])
def test_shockwave_red_end_buffer_is_inclusive(tmp_path, off_seconds, expected):
    result = extract_shockwave(make_cache(tmp_path, [5, off_seconds], [82, 81], [1, 1]))
    if expected:
        assert result.loc[0, 'v1Phase2T'] == 20
        assert result.loc[0, 'v2Phase2T'] == 100 / 60
        assert result.loc[0, 't_queue_offPhase2T'] == pd.Timestamp('2026-09-06') + pd.Timedelta(seconds=70)
    else:
        assert result.loc[0, ['v1Phase2T', 'v2Phase2T', 't_queue_offPhase2T']].isna().all()


def test_shockwave_no_events_preserves_metadata_only_output(tmp_path):
    result = extract_shockwave(make_cache(tmp_path, [], [], []))
    assert len(result) == 2
    assert list(result) == ['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength', 'date']


def test_shockwave_events_outside_cycles_still_use_red_buffer(tmp_path):
    result = extract_shockwave(make_cache(tmp_path, [25, 35], [82, 81], [1, 1]))
    assert result['v1Phase2T'].tolist() == [100 / 25, 100 / 15]
    assert result['v2Phase2T'].tolist() == [100 / 25, 100 / 15]


def test_legacy_prev_next_mode_keeps_home_cycle_only(tmp_path):
    cache = make_cache(tmp_path, [-1, 0, 10, 20, 21], [81]*5, [1]*5)
    result = cache.assign_cycle_index((81,), 'back', tolerance='prev_next')
    assert result['cycle_idx'].tolist() == [-1, 0, 1, 1, -1]
    assert cache.assign_cycle_index((81,), 'back', tolerance='prev_next') is result
