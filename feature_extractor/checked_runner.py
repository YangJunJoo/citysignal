"""Shared CLI, exact job manifests, completion checks and execution accounting."""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import ENGINE_ID
from .checked_execution import execute_frames
from .config_validator import validate_config
from .feature_registry import resolve_features
from .result_types import FeatureResult, completion_status
from .run_store import RunStore, content_hash, file_signature

DATE_PATTERN = re.compile(r'atspm-(\d{4})-(\d{1,2})-(\d{1,2})\.parquet$')


def job_from_path(path, features=None, reason='selected'):
    path = Path(path).resolve()
    match = DATE_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f'Invalid input filename: {path}')
    year, month, day = map(int, match.groups())
    return {'signal_id': path.parent.name, 'date_str': f'{year:04d}-{month:02d}-{day:02d}',
            'input_file': str(path), 'input_signature': file_signature(path),
            'features': list(resolve_features(features)), 'reason': reason}


def discover_jobs(config):
    if getattr(config, 'jobs_file', None):
        manifest = json.loads(Path(config.jobs_file).read_text())
        jobs = manifest['jobs'] if isinstance(manifest, dict) else manifest
        seen = set()
        for job in jobs:
            parsed = job_from_path(job['input_file'], job.get('features', config.features))
            key = (job['signal_id'], job['date_str'])
            if key != (parsed['signal_id'], parsed['date_str']) or key in seen:
                raise ValueError(f'Duplicate or mismatched manifest job: {key}')
            seen.add(key)
            if not job.get('input_signature') or not job.get('config_hash'):
                raise ValueError(f'Manifest must pin input_signature and config_hash: {key}')
            job['features'] = parsed['features']
        return jobs
    base = Path(config.event_dir)
    base = base / 'per_signal' if base.name != 'per_signal' and (base / 'per_signal').is_dir() else base
    jobs = []
    signal_dirs = [base / str(sig) for sig in config.signals] if config.signals else sorted(p for p in base.iterdir() if p.is_dir())
    for directory in signal_dirs:
        pattern = f'atspm-{config.year}-{config.month}-{config.day}.parquet' if all((config.year, config.month, config.day)) else 'atspm-*.parquet'
        for path in sorted(directory.glob(pattern)):
            job = job_from_path(path, config.features)
            parts = tuple(map(int, job['date_str'].split('-')))
            if any(wanted and wanted != actual for wanted, actual in zip((config.year, config.month, config.day), parts)):
                continue
            if getattr(config, 'start_date', None) and job['date_str'] < config.start_date:
                continue
            if getattr(config, 'end_date', None) and job['date_str'] > config.end_date:
                continue
            if getattr(config, 'auto_mode', False) and time.time() - path.stat().st_mtime > config.max_age_hours * 3600:
                continue
            jobs.append(job)
    return jobs


def _read_events(path, signal_id):
    frame = pd.read_parquet(path).rename(columns={'timestamp': 'timeStamp'})
    missing = {'timeStamp', 'eventCode', 'eventParam', 'signalID'} - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    if set(frame['signalID'].dropna().astype(str)) - {signal_id} or frame['signalID'].isna().any():
        raise ValueError('Input contains a different or missing signalID')
    if pd.api.types.is_string_dtype(frame['timeStamp']):
        parsed = pd.to_datetime(frame['timeStamp'], format='%m-%d-%Y %H:%M:%S.%f', errors='coerce')
        if parsed.isna().any():
            parsed = pd.to_datetime(frame['timeStamp'], format='mixed', errors='raise')
    else:
        parsed = pd.to_datetime(frame['timeStamp'], errors='raise')
    if parsed.isna().any():
        raise ValueError('Input contains invalid timestamps')
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert('America/New_York').dt.tz_localize(None)
    frame['timeStamp'] = parsed
    return frame.sort_values('timeStamp', kind='stable')


def _version(signature, decision, source):
    from .config_validator import RULES_VERSION
    evidence_path = ((decision or {}).get('controller_mapping') or {}).get('source_path')
    evidence_hash = content_hash(evidence_path) if evidence_path else None
    policy = json.dumps({'decision': decision, 'source': source, 'config_rules': RULES_VERSION,
                         'controller_source_hash': evidence_hash}, sort_keys=True)
    return signature + ':' + hashlib.sha256(policy.encode()).hexdigest()


def execute_job(job, settings):
    # A caller cannot select a different implementation or mislabel its outputs.
    settings = {**settings, 'engine': ENGINE_ID}
    store = RunStore(settings['db_path'])
    signal, date = job['signal_id'], job['date_str']
    selected = resolve_features(job.get('features'))
    config_path = Path(settings['config_dir']) / f'{signal}.csv'
    decision = settings.get('decisions', {}).get(signal)
    config_hash = content_hash(config_path)
    missing_input = not Path(job['input_file']).is_file()
    signature = 'missing_input' if missing_input else file_signature(job['input_file'])
    source = settings.get('sources', {}).get(date, {'status': 'unverified'})
    version = _version(signature, decision, source)
    stale = signature != job.get('input_signature', signature) or config_hash != job.get('config_hash', config_hash)
    if job.get('input_hash') and not missing_input:
        digest = hashlib.sha256()
        with open(job['input_file'], 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        stale = stale or digest.hexdigest() != job['input_hash']
    if not stale and not settings.get('overwrite'):
        reused = store.reusable(signal, date, version, config_hash, settings['engine'], selected)
        if reused:
            return {'signal_id': signal, 'date_str': date, 'status': reused['status'],
                    'reason_code': 'already_processed' if reused['status'] == 'complete' else 'unchanged_unresolved',
                    'results': reused['results'], 'attempt_id': reused['attempt_id'], 'reused': True}
    attempt = store.start_attempt(settings['run_id'], signal, date, version, config_hash, settings['engine'], selected)
    reason = ''
    results = {}
    try:
        validation = validate_config(config_path, signal_id=signal, decision=decision, data_date=date)
        if missing_input:
            reason = 'input_missing'
        elif stale:
            reason = 'stale_manifest'
        elif validation.action == 'exclude':
            reason = 'explicit_exclusion'
        elif validation.status != 'valid' and validation.action != 'use':
            reason = validation.reason_code
        elif validation.action != 'use':
            reason = 'repair_pending'
        elif source.get('status') not in {'valid', 'unverified'}:
            reason = source.get('status', 'source_incomplete')
        if reason:
            status = 'skipped' if reason == 'explicit_exclusion' else 'blocked'
            results = {name: FeatureResult(name, status, reason, json.dumps(validation.to_dict(), ensure_ascii=False)[:4000]) for name in selected}
        else:
            events = _read_events(job['input_file'], signal)
            if events.empty:
                results = {name: FeatureResult(name, 'no_data', 'empty_input') for name in selected}
            else:
                if source.get('status') == 'valid' and source.get('signature'):
                    from .split_validation import validate_split_source
                    try:
                        validate_split_source(events, source)
                    except ValueError as exc:
                        results = {name: FeatureResult(name, 'blocked', 'split_source_mismatch', str(exc)) for name in selected}
                if not results:
                    def unchanged_before_publish():
                        if file_signature(job['input_file']) != signature or content_hash(config_path) != config_hash:
                            raise RuntimeError('input_changed_during_processing')
                        if source.get('signature'):
                            from data_downloader.quality import file_signature as raw_signature
                            if raw_signature(source['signature']['path']) != source['signature']:
                                raise RuntimeError('input_changed_during_processing')
                    results = execute_frames(signal, date, events, validation, settings['output_dir'], selected,
                                             before_publish=unchanged_before_publish, source_quality=source)
            if file_signature(job['input_file']) != signature or content_hash(config_path) != config_hash:
                reason = 'input_changed_during_processing'
                results = {name: FeatureResult(name, 'blocked', reason) for name in selected}
        status = completion_status(results, selected)
        cycle = results.get('vehicle_cycle_profile')
        if cycle is not None and cycle.reason_code in {
                'barrier_mapping_unverified', 'controller_mapping_invalid',
                'controller_mapping_outside_period', 'controller_mapping_stale',
                'phase_movement_unverified', 'phase_barrier_unverified',
                'no_observed_eight_phase'}:
            reason = cycle.reason_code
        if cycle is not None and cycle.data is not None and cycle.data.attrs.get('quarantined_cycle_count', 0):
            status, reason = 'partial', 'cycle_windows_quarantined'
            cycle.message = str(cycle.data.attrs['quarantined_cycle_count']) + ' candidate cycles quarantined; retained windows only'
        if source.get('quality_scope') == 'signal':
            status = 'partial'
            reason = reason or 'source_rows_quarantined'
        if any(r.data is not None and r.data.attrs.get('mapping_policy_version') and
               not r.data.attrs.get('barrier_mapping_assumed') for r in results.values()):
            status = 'partial'  # Lane snapshots do not prove historical applicability.
            reason = reason or 'lane_mapping_effective_date_unverified'
        # Manual split-only inputs can be extracted, but source coverage remains
        # explicitly unverified and cannot produce an overall complete result.
        if source.get('status') == 'unverified' and status == 'complete':
            status, reason = 'partial', 'source_unverified'
        store.finish_attempt(attempt, status, results, reason)
    except Exception as exc:
        reason, status = ('input_changed_during_processing' if str(exc) == 'input_changed_during_processing' else 'job_error'), 'partial'
        results = {name: FeatureResult(name, 'blocked' if reason == 'input_changed_during_processing' else 'failed', reason, f'{type(exc).__name__}: {exc}') for name in selected}
        store.finish_attempt(attempt, status, results, reason)
    return {'signal_id': signal, 'date_str': date, 'status': status, 'reason_code': reason,
            'results': {name: result.to_dict() for name, result in results.items()}, 'attempt_id': attempt, 'reused': False}


def quality_sources(report):
    from data_downloader.quality import file_signature as raw_signature, RULE_VERSION as raw_rule
    entries = report.get('files', [])
    context = report.get('context_files', [])
    stale_dates = set()
    for entry in entries + context:
        try:
            if entry.get('rule_version') != raw_rule or entry.get('signature') != raw_signature(entry['path']):
                stale_dates.add(entry.get('source_date'))
        except (OSError, KeyError):
            stale_dates.add(entry.get('source_date'))
    # A changed neighbour invalidates the adjacent-boundary judgement too.
    from datetime import date as calendar_date, timedelta
    affected_dates = set(stale_dates)
    for date in stale_dates - {None}:
        dt = calendar_date.fromisoformat(date)
        affected_dates.update((dt + timedelta(days=i)).isoformat() for i in (-1, 1))
    covered = {entry.get('path') for entry in entries + context}
    for entry in entries:
        day = calendar_date.fromisoformat(entry['source_date'])
        parent = Path(entry['path']).parent
        for offset in (-1, 0, 1):
            neighbour = day + timedelta(days=offset)
            candidate = parent / f'atspm-{neighbour.year}-{neighbour.month}-{neighbour.day}.parquet'
            if candidate.is_file() and str(candidate.resolve()) not in covered:
                affected_dates.add(entry['source_date'])
    rejected = set(report.get('rejected_paths', []))
    sources = {}
    for entry in entries:
        date = entry.get('source_date')
        if date:
            status = 'source_quality_stale' if date in affected_dates else ('source_incomplete' if entry.get('path') in rejected else entry['status'])
            sources[date] = {'status': status, 'signature': entry.get('signature'),
                             'input_version': entry.get('input_version'),
                             'rule_version': entry.get('rule_version'),
                             'min_timestamp': entry.get('min_timestamp'), 'max_timestamp': entry.get('max_timestamp'),
                             'timestamp_basis': entry.get('timestamp_basis'), 'signals': entry.get('signals', {}),
                             'reported_status': entry.get('status'), 'reason_codes': entry.get('reason_codes', []),
                             'row_count': entry.get('row_count'), 'invalid_signal_count': entry.get('invalid_signal_count'),
                             'assigned_signal_row_count': sum(x.get('row_count', 0) for x in entry.get('signals', {}).values())}
    return sources


def settings_for_job(settings, job):
    copied = dict(settings)
    from .source_quality import scope_source_to_signal
    source = scope_source_to_signal(
        settings.get('sources', {}).get(job['date_str'], {'status': 'unverified'}), job['signal_id'])
    copied['sources'] = {job['date_str']: source}
    return copied


@contextlib.contextmanager
def pipeline_lock(output_dir, already_held=False):
    if already_held:
        yield
        return
    lock_path = Path(output_dir).resolve().parent / '.pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run_checked(config):
    output = Path(config.output_dir or Path(config.event_dir) / 'processed').resolve()
    jobs = discover_jobs(config)
    decisions = json.loads(Path(config.config_decisions).read_text()) if getattr(config, 'config_decisions', None) else {}
    decisions = decisions.get('decisions', decisions) if isinstance(decisions, dict) else decisions
    if isinstance(decisions, list):
        decisions = {row['signal_id']: row for row in decisions}
    quality = json.loads(Path(config.quality_report).read_text()) if getattr(config, 'quality_report', None) else {}
    if not quality and jobs:
        from types import SimpleNamespace
        from uuid import uuid4
        from run_pipeline import source_report
        event_base = Path(config.event_dir).resolve()
        raw_dir = event_base.parent if event_base.name == 'per_signal' else event_base
        quality_dir = output / 'runs' / ('quality-' + uuid4().hex)
        quality_dir.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(raw_dir=str(raw_dir), processed_dir=str(output),
            start_date=None, end_date=None, selected_dates=sorted({j['date_str'] for j in jobs}),
            jobs_file=getattr(config, 'jobs_file', None), quality_report=None)
        quality, _ = source_report(args, quality_dir)
    sources = quality_sources(quality)
    if getattr(config, 'dry_run', False):
        for job in jobs:
            job.setdefault('config_hash', content_hash(Path(config.config_dir) / f"{job['signal_id']}.csv"))
            job['validation'] = validate_config(Path(config.config_dir) / f"{job['signal_id']}.csv", signal_id=job['signal_id'],
                                                decision=decisions.get(job['signal_id']), data_date=job['date_str']).to_dict()
            job['source_quality'] = sources.get(job['date_str'], {'status': 'unverified'})
        manifest = {'jobs': jobs, 'status': 'dry_run'}
        destination = Path(getattr(config, 'summary_file', None) or 'jobs_manifest.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f'Dry run: {len(jobs)} exact signal/date jobs → {destination}')
        return 0
    with pipeline_lock(output, getattr(config, 'lock_held', False)):
        store = RunStore(output / 'pipeline_runs.sqlite3')
        run_id = store.start_run(getattr(config, 'run_id', None))
        settings = {'db_path': store.path, 'run_id': run_id, 'config_dir': config.config_dir,
                    'output_dir': str(output), 'engine': ENGINE_ID,
                    'overwrite': config.overwrite, 'decisions': decisions, 'sources': sources}
        outcomes = []
        workers = config.max_workers or min(os.cpu_count() or 1, 8)
        try:
            if workers == 1:
                for job in jobs:
                    outcomes.append(execute_job(job, settings_for_job(settings, job)))
                    if len(outcomes) % 25 == 0:
                        print(f'Processed {len(outcomes)}/{len(jobs)} jobs', flush=True)
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(execute_job, job, settings_for_job(settings, job)): job for job in jobs}
                    for future in as_completed(futures):
                        outcomes.append(future.result())
                        if len(outcomes) % 25 == 0:
                            print(f'Processed {len(outcomes)}/{len(jobs)} jobs', flush=True)
            counts = Counter(item['status'] for item in outcomes)
            reasons = Counter(item['reason_code'] for item in outcomes if item['reason_code'])
            feature_states = Counter(result['status'] for item in outcomes for result in item['results'].values())
            feature_reasons = Counter(result['reason_code'] for item in outcomes for result in item['results'].values() if result['reason_code'])
            unresolved_source = quality.get('status') not in (None, 'valid', 'complete', 'no_work')
            previous_unresolved = 0
            if not jobs:
                conditions, parameters = ["status NOT IN ('complete','imported_unverified')"], []
                for field, operation, value in [('date_str', '>=', getattr(config, 'start_date', None)),
                                                ('date_str', '<=', getattr(config, 'end_date', None))]:
                    if value:
                        conditions.append(f'{field} {operation} ?')
                        parameters.append(value)
                if config.signals:
                    conditions.append('signal_id IN (' + ','.join('?' for _ in config.signals) + ')')
                    parameters.extend(config.signals)
                with store.connect() as conn:
                    previous_unresolved = conn.execute('SELECT COUNT(*) FROM latest_attempts WHERE ' + ' AND '.join(conditions), parameters).fetchone()[0]
                unresolved_source = unresolved_source or bool(previous_unresolved)
            status = 'partial' if counts['partial'] or unresolved_source else ('complete' if outcomes else 'no_work')
            summary = {'run_id': run_id, 'status': status, 'jobs': len(jobs), 'counts': dict(counts),
                       'skip_reasons': dict(reasons), 'feature_states': dict(feature_states),
                       'feature_reasons': dict(feature_reasons), 'outcomes': outcomes,
                       'source_status': quality.get('status', 'unverified'), 'previous_unresolved': previous_unresolved}
            store.finish_run(run_id, status, summary)
        except BaseException as exc:
            store.finish_run(run_id, 'failed', {'error': f'{type(exc).__name__}: {exc}'})
            raise
        destination = Path(getattr(config, 'summary_file', None) or output / 'runs' / f'{run_id}.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in summary.items() if k != 'outcomes'}, ensure_ascii=False))
        print(f'Run report: {destination}')
        return 2 if status == 'partial' else 0
