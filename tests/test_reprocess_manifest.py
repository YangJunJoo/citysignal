import csv
import importlib.util
import json
from pathlib import Path

from feature_extractor.feature_registry import resolve_features

spec = importlib.util.spec_from_file_location('manifest_builder', Path(__file__).resolve().parents[1] / 'scripts/build_reprocess_manifest.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_inputs(tmp_path):
    audit, events, configs = (tmp_path / name for name in ('audit', 'raw', 'configs'))
    for path in (audit, events / 'per_signal', configs):
        path.mkdir(parents=True)
    write_csv(audit / 'daily_summary.csv', ['date'], [{'date': '2026-09-01'}, {'date': '2026-09-02'}])
    return audit, events, configs


def source(events, signal, day):
    path = events / 'per_signal' / signal / f'atspm-2026-9-{day}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f'{signal}-{day}'.encode())
    return path


def config(configs, signal, text='Phase,Channel,Stopbar Distance,Lane Type\n2,4,0,Through\n'):
    (configs / f'{signal}.csv').write_text(text)


def test_exact_union_deduplicates_reasons_and_only_extends_reproduced_profiles(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    for signal in ('A', 'B'):
        config(configs, signal)
        for day in (1, 2, 3):
            source(events, signal, day)
    write_csv(audit / 'failed_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': 'A', 'date_str': '2026-09-01', 'error_message': 'profile failed'}])
    write_csv(audit / 'skipped_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': 'A', 'date_str': '2026-09-01', 'error_message': 'skipped_chattering_sensor_file_size_4.99_MB'}])
    write_csv(audit / 'latest_signal_outputs.csv', ['signal_id', 'date', 'run_outcome', 'nonempty_features', 'empty_features', 'missing_outputs'], [
        {'signal_id': 'A', 'date': '2026-09-01', 'run_outcome': 'reported_success', 'nonempty_features': '', 'empty_features': 'volume', 'missing_outputs': 'vehicle_cycle_profile'},
        {'signal_id': 'B', 'date': '2026-09-02', 'run_outcome': 'reported_success', 'nonempty_features': 'volume', 'empty_features': 'headway', 'missing_outputs': ''}])
    (audit / 'profile_error_reproduction.json').write_text(json.dumps([{'signal': 'A', 'error': 'reproduced'}]))
    result = builder.build_manifest(audit, events, configs)
    assert {(j['signal_id'], j['date_str']) for j in result['jobs']} == {('A', '2026-09-01'), ('A', '2026-09-02')}
    assert len(result['jobs'][0]['reasons']) == 4
    assert result['jobs'][0]['features'] == list(resolve_features())
    assert result['summary']['candidate_jobs'] == 2
    # No processing output is created by manifest construction.
    assert not (tmp_path / 'processed').exists()


def test_only_newly_usable_configs_reenter_missing_config_population(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    for signal in ('still-missing', 'new', 'invalid'):
        source(events, signal, 1)
    config(configs, 'new')
    config(configs, 'invalid', '\n')
    write_csv(audit / 'skipped_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': signal, 'date_str': '2026-09-01', 'error_message': 'no_config_file'}
        for signal in ('still-missing', 'new', 'invalid')])
    result = builder.build_manifest(audit, events, configs)
    assert [j['signal_id'] for j in result['jobs']] == ['new']
    assert result['summary']['deferred_missing_config_jobs'] == 2


def test_blocked_configs_and_source_dates_remain_visible_and_versions_are_pinned(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    path = source(events, 'A', 1)
    config(configs, 'A', 'Phase\n8R\n')
    write_csv(audit / 'failed_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': 'A', 'date_str': '2026-09-01', 'error_message': 'error'}])
    report = tmp_path / 'quality.json'
    report.write_text(json.dumps([{'files': [{'source_date': '2026-09-01', 'path': 'bulk.parquet',
        'status': 'valid', 'expected_coverage_verified': False}], 'rejected_paths': ['bulk.parquet']}]))
    before = path.read_bytes()
    result = builder.build_manifest(audit, events, configs, report)
    job = result['jobs'][0]
    assert job['preflight_status'] == 'blocked'
    assert set(job['blocked_reasons']) == {'invalid_phase_mapping', 'source_incomplete'}
    assert job['input_hash'] == builder.sha256_file(path)
    assert job['config_hash'] == builder.sha256_file(configs / 'A.csv')
    assert job['input_signature']
    assert job['unresolved_reasons'] == ['source_unverified']
    assert path.read_bytes() == before
    path.write_bytes(b'new source version')
    assert builder.build_manifest(audit, events, configs, report)['jobs'][0]['input_hash'] != job['input_hash']


def test_no_reintroduced_size_limit_and_missing_inputs_are_explicit(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    path = source(events, 'large', 1)
    path.write_bytes(b'0' * 3_200_001)
    config(configs, 'large')
    write_csv(audit / 'failed_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': 'gone', 'date_str': '2026-09-01', 'error_message': 'error'}])
    write_csv(audit / 'skipped_jobs.csv', ['signal_id', 'date_str', 'error_message'], [
        {'signal_id': 'large', 'date_str': '2026-09-01', 'error_message': 'skipped_chattering_sensor_file_size_3.20_MB'}])
    result = builder.build_manifest(audit, events, configs)
    assert result['jobs'][0]['signal_id'] == 'large'
    assert not result['jobs'][0]['blocked_reasons']
    assert result['blocked_candidates'][0]['reason_code'] == 'missing_input'


def test_optional_partial_outputs_selects_only_affected_features_and_dependencies(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    source(events, 'A', 2)
    config(configs, 'A')
    write_csv(audit / 'latest_signal_outputs.csv', ['signal_id', 'date', 'run_outcome', 'nonempty_features', 'empty_features', 'missing_outputs'], [
        {'signal_id': 'A', 'date': '2026-09-02', 'run_outcome': 'reported_success', 'nonempty_features': 'volume', 'empty_features': 'occupancy', 'missing_outputs': ''}])
    assert builder.build_manifest(audit, events, configs)['summary']['jobs'] == 0
    result = builder.build_manifest(audit, events, configs, include_partial_outputs=True)
    assert result['jobs'][0]['features'] == list(resolve_features(['occupancy']))
    assert 'volume' not in result['jobs'][0]['features']


def test_cli_writes_linked_quality_snapshot(tmp_path):
    audit, events, configs = manifest_inputs(tmp_path)
    output = tmp_path / 'result' / 'jobs.json'
    assert builder.main(['--audit-dir', str(audit), '--event-dir', str(events), '--config-dir', str(configs), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert Path(result['quality_report']).is_file()
    assert result['quality_report_hash'] == builder.sha256_file(result['quality_report'])


def test_quality_snapshot_preserves_rule_context_and_acceptance(tmp_path):
    report = tmp_path / 'quality.json'
    neighbor = {'path': 'neighbor.parquet', 'source_date': '2026-08-31', 'signature': {'size': 123}}
    report.write_text(json.dumps([
        {'rule_version': 'raw-quality-1', 'status': 'complete', 'files': [],
         'context_files': [neighbor], 'accepted_paths': ['good.parquet', 'later-rejected.parquet']},
        {'rule_version': 'raw-quality-1', 'status': 'partial', 'files': [],
         'rejected_paths': ['later-rejected.parquet'], 'adjacent_gaps': [{'seconds': 123}]}]))
    snapshot = builder._quality_snapshot(report)
    assert snapshot['rule_version'] == 'raw-quality-1'
    assert snapshot['status'] == 'partial'
    assert snapshot['context_files'] == [neighbor]
    assert snapshot['accepted_paths'] == ['good.parquet']
    assert snapshot['rejected_paths'] == ['later-rejected.parquet']
    assert snapshot['adjacent_gaps'] == [{'seconds': 123}]
