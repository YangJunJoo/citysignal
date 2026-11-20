"""Failure isolation, dependency closure, and persistence safety for V2."""

from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from feature_extractor import pipeline
from feature_extractor.data_loader import SignalDataCache
from feature_extractor.feature_registry import resolve_features
from feature_extractor.result_types import FeatureResult


def frame(value=1):
    return pd.DataFrame({'cycleNo': [1], 'value': [value]})


def profiles():
    return {
        name: FeatureResult(name, 'success', data=frame(), row_count=1)
        for name in ('vehicle_phase_profile', 'vehicle_cycle_profile',
                     'pedestrian_phase_profile', 'pedestrian_cycle_profile')
    }


def cache_without_writes():
    return SimpleNamespace(save_feature=Mock(side_effect=AssertionError('unexpected output write')))


def test_exception_blocks_consumers_and_preserves_cause_while_independent_runs(monkeypatch):
    cache = cache_without_writes()

    def fail(_cache):
        try:
            raise ValueError('invalid detector 12')
        except ValueError as error:
            raise RuntimeError('headway calculation failed') from error

    conflict = Mock()
    monkeypatch.setattr(pipeline, 'extract_headway', fail)
    monkeypatch.setattr(pipeline, 'extract_conflict', conflict)
    monkeypatch.setattr(pipeline, 'extract_spat', lambda _cache: frame(7))

    results = pipeline.extract_features_checked(cache, ['conflict', 'spat'], profiles())

    assert results['headway'].status == 'failed'
    assert results['headway'].reason_code == 'extractor_exception'
    assert 'ValueError: invalid detector 12' in results['headway'].message
    assert 'RuntimeError: headway calculation failed' in results['headway'].message
    assert results['conflict'].status == 'blocked'
    assert results['conflict'].reason_code == 'dependency_failed'
    assert 'invalid detector 12' in results['conflict'].message
    assert results['spat'].status == 'success'
    assert results['spat'].row_count == 1
    conflict.assert_not_called()
    cache.save_feature.assert_not_called()


def test_checked_extraction_intercepts_legacy_saves_without_changing_cache(monkeypatch):
    cache = cache_without_writes()
    original_save = cache.save_feature

    def extract(proxy):
        result = frame()
        proxy.save_feature(result, 'vehicle_signal', 'spat')
        return result

    monkeypatch.setattr(pipeline, 'extract_spat', extract)
    result = pipeline.extract_features_checked(cache, ['spat'], profiles())['spat']

    assert result.status == 'success'
    assert result.output_file_path is None
    assert cache.save_feature is original_save
    original_save.assert_not_called()


def test_empty_result_is_no_data_and_blocks_secondary(monkeypatch):
    monkeypatch.setattr(pipeline, 'extract_headway', lambda cache: pd.DataFrame())
    secondary = Mock()
    monkeypatch.setattr(pipeline, 'extract_conflict', secondary)

    results = pipeline.extract_features_checked(cache_without_writes(), ['conflict'], profiles())

    assert results['headway'].status == 'no_data'
    assert results['headway'].reason_code == 'empty_result'
    assert results['headway'].row_count == 0
    assert results['conflict'].status == 'blocked'
    assert results['conflict'].reason_code == 'dependency_no_data'
    secondary.assert_not_called()


@pytest.mark.parametrize('status,reason_code', [
    ('failed', 'dependency_failed'),
    ('no_data', 'dependency_no_data'),
    ('not_applicable', 'dependency_not_applicable'),
    ('skipped', 'dependency_skipped'),
    ('blocked', 'dependency_failed'),
])
def test_profile_outcomes_block_dependents_without_becoming_not_applicable(monkeypatch, status, reason_code):
    prior = profiles()
    prior['vehicle_cycle_profile'] = FeatureResult(
        'vehicle_cycle_profile', status, 'profile_source', 'cycle source outcome')
    extractor = Mock()
    monkeypatch.setattr(pipeline, 'extract_spat', extractor)

    results = pipeline.extract_features_checked(cache_without_writes(), ['spat'], prior)

    assert results['vehicle_cycle_profile'] is prior['vehicle_cycle_profile']
    assert results['spat'].status == 'blocked'
    assert results['spat'].reason_code == reason_code
    extractor.assert_not_called()


def test_profiles_are_required_and_not_loaded_silently(monkeypatch):
    extractor = Mock()
    monkeypatch.setattr(pipeline, 'extract_spat', extractor)

    results = pipeline.extract_features_checked(cache_without_writes(), ['spat'])

    assert tuple(results) == resolve_features(['spat'])
    assert results['vehicle_phase_profile'].reason_code == 'missing_profile_result'
    assert results['vehicle_cycle_profile'].status == 'blocked'
    assert results['spat'].status == 'blocked'
    extractor.assert_not_called()


def test_explicit_applicability_is_the_only_not_applicable_path(monkeypatch):
    spat = Mock(return_value=pd.DataFrame())
    volume = Mock()
    headway = Mock()
    monkeypatch.setattr(pipeline, 'extract_spat', spat)
    monkeypatch.setattr(pipeline, 'extract_volume', volume)
    monkeypatch.setattr(pipeline, 'extract_headway', headway)

    results = pipeline.extract_features_checked(
        cache_without_writes(), ['spat', 'volume', 'headway'], profiles(),
        applicability={
            'spat': (None, 'detector applicability unknown'),
            'volume': (False, 'authoritative configuration confirms no volume detector'),
            'headway': (False, ''),
        })

    assert results['spat'].status == 'no_data'
    assert results['volume'].status == 'not_applicable'
    assert 'authoritative configuration' in results['volume'].message
    assert results['headway'].status == 'blocked'
    assert results['headway'].reason_code == 'applicability_unverified'
    spat.assert_called_once()
    volume.assert_not_called()
    headway.assert_not_called()


def test_secondary_selection_resolves_dependencies_once(monkeypatch):
    volume_frame, ped_frame = frame(3), frame(5)
    volume = Mock(return_value=volume_frame)
    pedestrian = Mock(return_value=ped_frame)
    secondary = Mock(return_value=frame(8))
    monkeypatch.setattr(pipeline, 'extract_volume', volume)
    monkeypatch.setattr(pipeline, 'extract_pedestrian_activity', pedestrian)
    monkeypatch.setattr(pipeline, 'extract_conflict_propensity', secondary)

    results = pipeline.extract_features_checked(
        cache_without_writes(), ['conflict_propensity', 'volume'], profiles())

    assert tuple(results) == resolve_features(['conflict_propensity', 'volume'])
    assert all(result.status == 'success' for result in results.values())
    volume.assert_called_once()
    pedestrian.assert_called_once()
    secondary.assert_called_once()
    assert secondary.call_args.args[1] is volume_frame
    assert secondary.call_args.args[2] is ped_frame


def test_conflict_headway_alias_has_one_canonical_execution(monkeypatch):
    headway = Mock(return_value=frame())
    conflict = Mock(return_value=frame())
    monkeypatch.setattr(pipeline, 'extract_headway', headway)
    monkeypatch.setattr(pipeline, 'extract_conflict', conflict)

    results = pipeline.extract_features_checked(
        cache_without_writes(), ['conflict', 'conflict_headway'], profiles())

    assert results['conflict'].status == 'success'
    assert 'conflict_headway' not in results
    headway.assert_called_once()
    conflict.assert_called_once()


@pytest.mark.parametrize('data,expected_status', [(None, 'failed'), (pd.DataFrame(), 'no_data')])
def test_prior_success_requires_nonempty_dataframe(monkeypatch, data, expected_status):
    prior = profiles()
    prior['headway'] = FeatureResult('headway', 'success', data=data)
    headway = Mock()
    conflict = Mock()
    monkeypatch.setattr(pipeline, 'extract_headway', headway)
    monkeypatch.setattr(pipeline, 'extract_conflict', conflict)

    results = pipeline.extract_features_checked(cache_without_writes(), ['conflict'], prior)

    assert results['headway'].status == expected_status
    assert results['conflict'].status == 'blocked'
    headway.assert_not_called()
    conflict.assert_not_called()


def test_invalid_extractor_result_is_failed(monkeypatch):
    monkeypatch.setattr(pipeline, 'extract_spat', lambda cache: None)
    result = pipeline.extract_features_checked(cache_without_writes(), ['spat'], profiles())['spat']
    assert result.status == 'failed'
    assert 'expected pandas DataFrame' in result.message


def test_unknown_features_rejected_before_extraction():
    with pytest.raises(ValueError, match='Unknown feature'):
        pipeline.extract_features_checked(cache_without_writes(), ['misspelled'])


@pytest.fixture
def saving_cache(tmp_path):
    cache = object.__new__(SignalDataCache)
    cache.cwd = str(tmp_path)
    cache.relative_production = 'production'
    cache.district = 'fdot_d5'
    cache.signal_id = 'test-signal'
    cache.date_str = '2026-09-06'
    return cache


def target(cache):
    return Path(cache.get_output_path('vehicle_traffic', 'volume')) / f'{cache.date_str}.parquet'


def test_atomic_save_publishes_readable_exact_output(saving_cache):
    expected = pd.DataFrame({'cycleNo': [1], 'values': [[1.0, 2.0]],
                             'time': [pd.Timestamp('2026-09-06T12:00:00')]})
    saving_cache.save_feature(expected, 'vehicle_traffic', 'volume')
    path = target(saving_cache)
    pd.testing.assert_frame_equal(read_frame(path), expected, check_exact=True)
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize('failure_stage', ['serialize', 'validate', 'replace'])
def test_atomic_save_failure_preserves_previous_output_and_cleans_temporary(
        saving_cache, monkeypatch, failure_stage):
    prior = frame(11)
    saving_cache.save_feature(prior, 'vehicle_traffic', 'volume')
    path = target(saving_cache)
    original_bytes = path.read_bytes()
    replacement = frame(99)

    if failure_stage == 'serialize':
        def partial_write(self, destination, *args, **kwargs):
            Path(destination).write_bytes(b'partial parquet')
            raise OSError('injected disk error')
        monkeypatch.setattr(storage.pq, 'write_table', partial_write)
    elif failure_stage == 'validate':
        monkeypatch.setattr(storage, 'read_frame', lambda path: frame(-1))
    else:
        monkeypatch.setattr('feature_extractor.storage.os.replace',
                            Mock(side_effect=OSError('injected replace error')))

    with pytest.raises((OSError, AssertionError)):
        saving_cache.save_feature(replacement, 'vehicle_traffic', 'volume')

    assert path.read_bytes() == original_bytes
    assert list(path.parent.iterdir()) == [path]


def test_atomic_save_rejects_non_dataframe_without_creating_output(saving_cache):
    with pytest.raises(TypeError, match='DataFrame'):
        saving_cache.save_feature(None, 'vehicle_traffic', 'volume')
    assert not target(saving_cache).exists()


def test_legacy_conflict_headway_selection_keeps_dataframe_api(monkeypatch):
    cache = cache_without_writes()
    monkeypatch.setattr(pipeline, 'SignalDataCache', lambda **kwargs: cache)
    monkeypatch.setattr(pipeline, 'extract_headway', lambda cache: frame(1))
    conflict = Mock(return_value=frame(2))
    monkeypatch.setattr(pipeline, 'extract_conflict_headway', conflict)

    results = pipeline.extract_all_features(
        'test', 2026, 9, 6, '/tmp', 'interim', 'production', features=['conflict_headway'])

    assert list(results) == ['conflict_headway']
    pd.testing.assert_frame_equal(results['conflict_headway'], frame(2))
    conflict.assert_called_once()


def test_legacy_unsuppressed_exception_keeps_original_type(monkeypatch):
    monkeypatch.setattr(pipeline, 'SignalDataCache', lambda **kwargs: cache_without_writes())
    monkeypatch.setattr(pipeline, 'extract_spat', Mock(side_effect=ValueError('original error')))

    with pytest.raises(ValueError, match='original error'):
        pipeline.extract_all_features('test', 2026, 9, 6, '/tmp', 'interim', 'production',
                                      features=['spat'], suppress_errors=False)


def test_legacy_atomic_save_supports_nested_nan_roundtrip(saving_cache):
    expected = pd.DataFrame({'cycleNo': [1], 'headwaySamples': [[[float('nan')]]]})
    saving_cache.save_feature(expected, 'vehicle_traffic', 'volume')
    pd.testing.assert_frame_equal(read_frame(target(saving_cache)), expected,
                                  check_exact=False, rtol=0, atol=0)


def propensity_inputs():
    metadata = {
        'signalID': ['test', 'test'], 'cycleNo': [1, 2],
        'cycleBegin': pd.to_datetime(['2026-09-06T12:00:00', '2026-09-06T12:01:00']),
        'cycleEnd': pd.to_datetime(['2026-09-06T12:01:00', '2026-09-06T12:02:00']),
    }
    volume = pd.DataFrame(metadata).assign(greenVolumePhase2R=[1, 0])
    pedestrian = pd.DataFrame(metadata).assign(pedestrianActivityPhase4=[1, 1])
    return volume, pedestrian


def test_propensity_coalesces_matching_intervals_and_keeps_zero_flags():
    from feature_extractor.extractors.secondary import extract_conflict_propensity
    volume, pedestrian = propensity_inputs()
    result = extract_conflict_propensity(SimpleNamespace(save_feature=Mock()), volume, pedestrian)
    assert 'cycleBegin' in result and 'cycleEnd' in result
    assert not any(column.endswith(('_vol', '_ped')) for column in result.columns)
    assert result['conflictPropensityPhase2R_4'].tolist() == [1, 0]
    pd.testing.assert_series_equal(result['cycleBegin'], volume['cycleBegin'])


def test_propensity_rejects_conflicting_intervals():
    from feature_extractor.extractors.secondary import extract_conflict_propensity
    volume, pedestrian = propensity_inputs()
    pedestrian.loc[0, 'cycleBegin'] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match='Conflicting cycleBegin'):
        extract_conflict_propensity(SimpleNamespace(save_feature=Mock()), volume, pedestrian)


def test_propensity_rejects_duplicate_cycle_keys():
    from feature_extractor.extractors.secondary import extract_conflict_propensity
    volume, pedestrian = propensity_inputs()
    pedestrian = pd.concat([pedestrian, pedestrian.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        extract_conflict_propensity(SimpleNamespace(save_feature=Mock()), volume, pedestrian)


def test_propensity_actual_activity_prefix_is_not_guessed_into_new_measure():
    volume, pedestrian = propensity_inputs()
    pedestrian = pedestrian.rename(columns={'pedestrianActivityPhase4': 'pedestrianActivity45Phase4'})
    prior = profiles()
    prior['volume'] = FeatureResult('volume', 'success', data=volume, row_count=len(volume))
    prior['pedestrian_activity'] = FeatureResult('pedestrian_activity', 'success',
                                                data=pedestrian, row_count=len(pedestrian))
    result = pipeline.extract_features_checked(
        cache_without_writes(), ['conflict_propensity'], prior)['conflict_propensity']
    assert result.status == 'no_data'
    assert result.reason_code == 'missing_metric_inputs'
    assert 'pedestrianActivityPhase measures' in result.message
    assert result.row_count == 0
