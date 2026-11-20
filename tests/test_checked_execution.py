"""Regression tests for verified publication, reuse, and exact job manifests."""

from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from feature_extractor.checked_execution import _result_from_call, publish_frame, validate_frame
from feature_extractor.checked_runner import discover_jobs, execute_job, job_from_path
from feature_extractor.feature_registry import output_path, resolve_features
from feature_extractor.result_types import FeatureResult
from feature_extractor.run_store import RunStore, content_hash


def cycle_frame(signal='test-signal', count=0):
    return pd.DataFrame({
        'signalID': [signal], 'cycleNo': [1],
        'cycleBegin': [pd.Timestamp('2026-09-06T12:00:00')],
        'cycleEnd': [pd.Timestamp('2026-09-06T12:01:00')],
        'cycleLength': [60], 'date': [pd.Timestamp('2026-09-06').date()],
        'greenVolumePhase2T': [count],
    })


def test_valid_zero_count_is_success_and_published(tmp_path):
    data = cycle_frame(count=0)
    destination = output_path(tmp_path, 'volume', 'test-signal', '2026-09-06')
    result = _result_from_call('volume', lambda: data, destination, 'test-signal')

    assert result.status == 'success'
    assert result.row_count == 1
    publish_frame(result.data, 'volume', 'test-signal', destination)
    actual = read_frame(destination)
    pd.testing.assert_frame_equal(actual, data)
    assert actual['greenVolumePhase2T'].iat[0] == 0


def test_empty_dataframe_is_no_data_and_cannot_be_published(tmp_path):
    destination = tmp_path / 'empty.parquet'
    result = _result_from_call('volume', lambda: pd.DataFrame(), destination, 'test-signal')

    assert result.status == 'no_data'
    assert result.reason_code == 'empty_result'
    assert result.row_count == 0
    with pytest.raises(ValueError, match='Empty result'):
        publish_frame(result.data, 'volume', 'test-signal', destination)
    assert not destination.exists()


def test_staged_dataframe_is_read_when_legacy_returns_none(tmp_path):
    staged = tmp_path / 'stage.parquet'
    expected = cycle_frame()

    def legacy():
        write_frame(expected, staged)

    result = _result_from_call('volume', legacy, staged, 'test-signal')
    assert result.status == 'success'
    pd.testing.assert_frame_equal(result.data, expected)


def test_profile_exception_is_failed_with_original_message(tmp_path):
    def fail():
        raise ValueError('phase 8R cannot map to a barrier')

    result = _result_from_call('vehicle_phase_profile', fail, tmp_path / 'missing.parquet', 'test-signal')
    assert result.status == 'failed'
    assert 'ValueError: phase 8R cannot map to a barrier' in result.message


@pytest.mark.parametrize('failure_stage', ['serialize', 'validation', 'replace'])
def test_atomic_failure_preserves_previous_bytes_and_removes_temporary(tmp_path, monkeypatch, failure_stage):
    destination = tmp_path / 'volume.parquet'
    original = cycle_frame(count=1)
    publish_frame(original, 'volume', 'test-signal', destination)
    original_bytes = destination.read_bytes()

    if failure_stage == 'serialize':
        def fail_after_partial_write(self, path, **kwargs):
            Path(path).write_bytes(b'incomplete parquet')
            raise OSError('injected serialization error')
        monkeypatch.setattr(storage.pq, 'write_table', fail_after_partial_write)
    elif failure_stage == 'validation':
        monkeypatch.setattr(storage, 'read_frame', lambda _: cycle_frame(signal='wrong-signal'))
    else:
        monkeypatch.setattr('feature_extractor.storage.os.replace',
                            Mock(side_effect=OSError('injected replace error')))

    with pytest.raises((OSError, ValueError)):
        publish_frame(cycle_frame(count=9), 'volume', 'test-signal', destination)
    assert destination.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize('mutation,match', [
    (lambda data: data.drop(columns='signalID'), 'Missing output columns'),
    (lambda data: data.assign(signalID='other-signal'), 'does not match'),
    (lambda data: data.assign(cycleEnd=pd.Timestamp('2026-09-06T11:00:00')), 'Invalid cycle interval'),
])
def test_output_schema_or_identity_failure_is_not_success(tmp_path, mutation, match):
    result = _result_from_call('volume', lambda: mutation(cycle_frame()), tmp_path / 'output.parquet', 'test-signal')
    assert result.status == 'failed'
    assert match in result.message


def test_latest_failed_attempt_cannot_reuse_older_success(tmp_path):
    store = RunStore(tmp_path / 'ledger.sqlite3')
    run_id = store.start_run()
    destination = tmp_path / 'volume.parquet'
    publish_frame(cycle_frame(), 'volume', 'test-signal', destination)
    required = ('volume',)
    first = store.start_attempt(run_id, 'test-signal', '2026-09-06', 'input', 'config', 'v2', required)
    store.finish_attempt(first, 'complete', {'volume': FeatureResult(
        'volume', 'success', output_file_path=str(destination), row_count=1)})
    assert store.reusable('test-signal', '2026-09-06', 'input', 'config', 'v2', required)['attempt_id'] == first

    latest = store.start_attempt(run_id, 'test-signal', '2026-09-06', 'input', 'config', 'v2', required)
    store.finish_attempt(latest, 'partial', {'volume': FeatureResult('volume', 'failed', 'extractor_exception', 'failure')})

    assert store.latest('test-signal', '2026-09-06')['attempt_id'] == latest
    assert store.reusable('test-signal', '2026-09-06', 'input', 'config', 'v2', required) is None
    assert destination.is_file()


def test_changed_success_output_cannot_be_reused(tmp_path):
    store = RunStore(tmp_path / 'ledger.sqlite3')
    run_id = store.start_run()
    path = tmp_path / 'volume.parquet'
    publish_frame(cycle_frame(), 'volume', 'test-signal', path)
    attempt = store.start_attempt(run_id, 'test-signal', '2026-09-06', 'input', 'config', 'v2', ('volume',))
    store.finish_attempt(attempt, 'complete', {'volume': FeatureResult(
        'volume', 'success', output_file_path=str(path), row_count=1)})
    path.write_bytes(b'changed output')
    assert store.reusable('test-signal', '2026-09-06', 'input', 'config', 'v2', ('volume',)) is None


def test_exact_manifest_does_not_expand_signal_date_cross_product(tmp_path):
    jobs = []
    for signal, day in [('first-signal', 5), ('second-signal', 6)]:
        path = tmp_path / 'inputs' / signal / f'atspm-2026-9-{day}.parquet'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'input placeholder; parsing is not exercised here')
        job = job_from_path(path, ['spat'])
        job['config_hash'] = 'pinned-config'
        jobs.append(job)
    manifest = tmp_path / 'jobs.json'
    manifest.write_text(json.dumps({'jobs': jobs}))
    config = SimpleNamespace(jobs_file=str(manifest), features=None)

    selected = discover_jobs(config)

    assert [(job['signal_id'], job['date_str']) for job in selected] == [
        ('first-signal', '2026-09-05'), ('second-signal', '2026-09-06')]
    assert len(selected) == 2
    assert all(tuple(job['features']) == resolve_features(['spat']) for job in selected)


def test_manifest_rejects_mismatched_signal_identity(tmp_path):
    path = tmp_path / 'actual-signal' / 'atspm-2026-9-6.parquet'
    path.parent.mkdir()
    path.write_bytes(b'input placeholder')
    job = job_from_path(path, ['spat'])
    job.update(signal_id='wrong-signal', config_hash='pinned-config')
    manifest = tmp_path / 'jobs.json'
    manifest.write_text(json.dumps({'jobs': [job]}))
    with pytest.raises(ValueError, match='mismatched manifest'):
        discover_jobs(SimpleNamespace(jobs_file=str(manifest), features=None))


def test_invalid_config_is_recorded_before_any_extractor_runs(tmp_path, monkeypatch):
    input_path = tmp_path / 'inputs' / '1067' / 'atspm-2026-9-6.parquet'
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b'Input is intentionally never read when configuration is invalid.')
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    config_path = config_dir / '1067.csv'
    config_path.write_text('ID,Approach,Lane Type\n1,S ORLANDO DR,Through\n')
    store = RunStore(tmp_path / 'ledger.sqlite3')
    settings = {'db_path': store.path, 'run_id': store.start_run(), 'config_dir': str(config_dir),
                'output_dir': str(tmp_path / 'outputs'), 'engine': 'v2', 'overwrite': True,
                'sources': {'2026-09-06': {'status': 'valid'}}}
    job = job_from_path(input_path, ['spat'])
    job['config_hash'] = content_hash(config_path)
    extractor = Mock()
    monkeypatch.setattr('feature_extractor.checked_runner.execute_frames', extractor)

    outcome = execute_job(job, settings)

    assert outcome['status'] == 'partial'
    assert outcome['reason_code'] == 'missing_phase_column'
    assert all(result['status'] == 'blocked' for result in outcome['results'].values())
    assert store.latest('1067', '2026-09-06')['reason_code'] == 'missing_phase_column'
    extractor.assert_not_called()


@pytest.mark.parametrize('nested', [[float('nan')], [[float('nan')]], [(float('nan'), float('nan'))]])
def test_nested_nan_pickle_roundtrip_is_valid(tmp_path, nested):
    expected = cycle_frame()
    expected['headwaySamples'] = [nested]
    destination = tmp_path / 'headway.parquet'
    publish_frame(expected, 'headway', 'test-signal', destination)
    pd.testing.assert_frame_equal(read_frame(destination), expected,
                                  check_exact=False, rtol=0, atol=0)


def test_tiny_nested_value_change_is_rejected_by_serialization_check(tmp_path, monkeypatch):
    expected = cycle_frame()
    expected['headwaySamples'] = [[1.0, float('nan')]]
    changed = cycle_frame()
    changed['headwaySamples'] = [[1.0 + 1e-12, float('nan')]]
    monkeypatch.setattr(storage, 'read_frame', lambda path: changed)
    with pytest.raises((ValueError, AssertionError)):
        publish_frame(expected, 'headway', 'test-signal', tmp_path / 'headway.parquet')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('reason', ['cache_load_failed', 'stale_manifest',
                                    'input_changed_during_processing', 'dependency_publication_failed'])
def test_transient_blocked_attempt_is_retryable(tmp_path, reason):
    store = RunStore(tmp_path / 'ledger.sqlite3')
    run_id = store.start_run()
    attempt = store.start_attempt(run_id, 'test-signal', '2026-09-06', 'input', 'config', 'v2', ('volume',))
    store.finish_attempt(attempt, 'partial', {'volume': FeatureResult('volume', 'blocked', reason)})
    assert store.reusable('test-signal', '2026-09-06', 'input', 'config', 'v2', ('volume',)) is None


def test_profile_dependency_for_pedestrian_only_selection_includes_vehicle_cycle():
    selected = resolve_features(['pedestrian_activity'])
    assert 'vehicle_phase_profile' in selected
    assert 'vehicle_cycle_profile' in selected
    assert selected.index('vehicle_cycle_profile') < selected.index('pedestrian_cycle_profile')


def test_publication_failure_blocks_secondary_and_cleans_staging(tmp_path, monkeypatch):
    from feature_extractor import checked_execution
    from feature_extractor import data_loader, pipeline

    monkeypatch.setattr(checked_execution, 'build_vehicle_phase_profile',
                        lambda events, signal_id, date_str: pd.DataFrame({'signalID': [signal_id], 'phaseNo': [2]}))

    def cycles(phases, signal_id, date_str, conflict_policy):
        assert conflict_policy == 'quarantine'
        return cycle_frame(signal=signal_id)

    monkeypatch.setattr(checked_execution, 'build_vehicle_cycle_profile', cycles)
    monkeypatch.setattr(data_loader, 'SignalDataCache', lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(pipeline, 'extract_headway', lambda cache: cycle_frame())
    monkeypatch.setattr(pipeline, 'extract_conflict', lambda cache, source: cycle_frame())
    real_publish = checked_execution.publish_frame

    def publish(frame, name, signal_id, destination):
        if name == 'headway':
            raise OSError('injected publication failure')
        real_publish(frame, name, signal_id, destination)

    monkeypatch.setattr(checked_execution, 'publish_frame', publish)
    validation = SimpleNamespace(dataframe=pd.DataFrame({'phaseNo': [2]}), capabilities={})
    results = checked_execution._execute_cycle_frames(
        'test-signal', '2026-09-06', pd.DataFrame(), validation, tmp_path, ['conflict'])

    assert results['headway'].status == 'failed'
    assert results['headway'].reason_code == 'output_validation_failed'
    assert results['conflict'].status == 'blocked'
    assert results['conflict'].reason_code == 'dependency_publication_failed'
    assert not output_path(tmp_path, 'headway', 'test-signal', '2026-09-06').exists()
    assert not output_path(tmp_path, 'conflict', 'test-signal', '2026-09-06').exists()
    assert not list((tmp_path / '.staging').iterdir())


def test_partial_cli_return_summary_and_ledger_are_consistent(tmp_path, capsys):
    from feature_extractor.checked_runner import run_checked

    path = tmp_path / 'inputs' / '1067' / 'atspm-2026-9-6.parquet'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'Invalid config stops execution before this input is parsed.')
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    config_file = config_dir / '1067.csv'
    config_file.write_text('ID,Approach,Lane Type\n1,S ORLANDO DR,Through\n')
    job = job_from_path(path, ['spat'])
    job['config_hash'] = content_hash(config_file)
    manifest = tmp_path / 'jobs.json'
    manifest.write_text(json.dumps({'jobs': [job]}))
    output = tmp_path / 'output'
    summary_path = tmp_path / 'summary.json'
    config = SimpleNamespace(
        jobs_file=str(manifest), features=None, output_dir=str(output), event_dir=str(tmp_path / 'inputs'),
        config_dir=str(config_dir), config_decisions=None, quality_report=None, dry_run=False,
        overwrite=False, max_workers=1, run_id='partial-cli-test',
        summary_file=str(summary_path), lock_held=False,
    )

    return_code = run_checked(config)
    summary = json.loads(summary_path.read_text())
    store = RunStore(output / 'pipeline_runs.sqlite3')
    latest = store.latest('1067', '2026-09-06')
    with store.connect() as conn:
        run = dict(conn.execute('SELECT * FROM runs WHERE run_id=?', ('partial-cli-test',)).fetchone())

    assert return_code == 2
    assert summary['status'] == latest['status'] == run['status'] == 'partial'
    assert summary['counts'] == {'partial': 1}
    assert summary['outcomes'][0]['reason_code'] == latest['reason_code'] == 'missing_phase_column'
    assert json.loads(run['summary_json'])['skip_reasons'] == {'missing_phase_column': 1}
    assert all(result['status'] == 'blocked' for result in latest['results'].values())
    assert 'Run report:' in capsys.readouterr().out
