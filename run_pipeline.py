#!/usr/bin/env python3
"""ATSPM download → source validation → split → checked feature extraction."""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from data_downloader import data_downloader as downloader
from data_downloader import split_by_signal
from data_downloader.quality import assess_collection, source_date_from_path
from feature_extractor.checked_runner import pipeline_lock, run_checked


def generate_dates(start, end):
    start = datetime.strptime(start, '%Y-%m-%d')
    end = datetime.strptime(end, '%Y-%m-%d')
    if end < start:
        raise ValueError('End date precedes start date')
    return [start + timedelta(days=i) for i in range((end-start).days+1)]


def source_report(args, run_dir):
    from collections import Counter
    from data_downloader.quality import RULE_VERSION, classify_missing_dates, file_signature

    # An exact date list takes precedence, including an intentionally empty list.
    selected_dates = getattr(args, 'selected_dates', None)
    if selected_dates is not None:
        dates = sorted(set(selected_dates))
    elif getattr(args, 'jobs_file', None):
        manifest = json.loads(Path(args.jobs_file).read_text())
        jobs = manifest.get('jobs', []) if isinstance(manifest, dict) else manifest
        dates = sorted({j['date_str'] for j in jobs})
    elif getattr(args, 'start_date', None) and getattr(args, 'end_date', None):
        dates = [d.strftime('%Y-%m-%d') for d in generate_dates(args.start_date, args.end_date)]
    else:
        dates = None
    inspect_dates = None if dates is None else set(dates)
    if inspect_dates is not None:
        for label in dates:
            dt = datetime.strptime(label, '%Y-%m-%d')
            inspect_dates.update((dt + timedelta(days=offset)).strftime('%Y-%m-%d') for offset in (-1, 1))
    all_paths = list(Path(args.raw_dir).glob('atspm-*.parquet'))
    available_dates = {source_date_from_path(path) for path in all_paths}
    available_dates.discard(None)
    paths = [p for p in all_paths if inspect_dates is None or source_date_from_path(p) in inspect_dates]
    statuses = {}
    for label in dates or []:
        y, m, d = map(int, label.split('-'))
        sidecar = Path(args.raw_dir) / f'atspm-{y}-{m}-{d}.csv.download.json'
        if sidecar.exists():
            try:
                statuses[label] = json.loads(sidecar.read_text()).get('http_status')
            except (OSError, ValueError):
                # Corrupt/mid-write evidence cannot establish publication pending.
                statuses[label] = None
    if getattr(args, 'quality_report', None):
        report = json.loads(Path(args.quality_report).read_text())
        if report.get('rule_version') != RULE_VERSION:
            raise ValueError('Quality report uses an outdated raw-source rule version')
        context = {entry['path']: entry for entry in report.get('context_files', [])}
        context.update({entry['path']: entry for entry in report.get('files', [])})
        context = {path: entry for path, entry in context.items()
                   if inspect_dates is None or entry.get('source_date') in inspect_dates}
        for entry in report.get('missing', []):
            statuses.setdefault(entry['source_date'], entry.get('http_status'))
        new_context_paths = {str(path.resolve()) for path in paths} - set(context)
        if new_context_paths:
            raise ValueError('Stale raw-source quality report: unassessed adjacent/selected files ' + ', '.join(sorted(new_context_paths)))
        # Validate before splitting as well as before extraction. Neighbor changes
        # can invalidate a boundary conclusion while a selected file stays intact.
        for entry in context.values():
            try:
                fresh = entry.get('signature') == file_signature(entry['path'])
            except OSError:
                fresh = False
            if not fresh:
                raise ValueError(f"Stale raw-source quality report: {entry['path']}")
        covered_dates = {entry['source_date'] for entry in report.get('files', [])}
        if dates is not None and set(dates).intersection(available_dates) - covered_dates:
            raise ValueError('Quality report does not cover all selected available source dates')
    else:
        report = assess_collection(paths, dates, statuses,
            cache_dir=Path(args.processed_dir).parent / 'operations' / 'quality_cache')
        context = {entry['path']: entry for entry in report['files']}
    if dates is not None:
        report['files'] = [entry for entry in report['files'] if entry['source_date'] in dates]
        for key in ('adjacent_gaps', 'overlaps'):
            report[key] = [entry for entry in report.get(key, [])
                           if entry.get('left_date') in dates or entry.get('right_date') in dates]
        # The publication frontier is a cheap inventory of all raw filenames,
        # independent of the limited set requiring a content scan this run.
        available_dates.update(entry['source_date'] for entry in report['files'])
        report['missing'] = classify_missing_dates(dates, available_dates, statuses)
    report['context_files'] = [
        {key: entry.get(key) for key in ('path', 'source_date', 'signature', 'input_version', 'rule_version', 'contract')}
        for entry in context.values()
        if inspect_dates is None or entry.get('source_date') in inspect_dates]
    previously_rejected = set(report.get('rejected_paths', []))
    report['accepted_paths'] = [entry['path'] for entry in report['files']
                                if entry['status'] == 'valid' and entry['path'] not in previously_rejected]
    report['rejected_paths'] = [entry['path'] for entry in report['files']
                                if entry['status'] != 'valid' or entry['path'] in previously_rejected]
    report['reason_counts'] = dict(Counter(reason for entry in report['files'] for reason in entry.get('reason_codes', []))
                                  + Counter(entry['status'] for entry in report.get('missing', [])))
    report['status'] = 'partial' if report['rejected_paths'] or report.get('missing') else ('complete' if report['files'] else 'no_work')
    report['requested_dates'] = dates
    report['context_dates'] = sorted(inspect_dates) if inspect_dates is not None else None
    report['raw_directory'] = str(Path(args.raw_dir).resolve())
    report['publication_frontier'] = max(available_dates) if available_dates else None
    output = Path(run_dir) / 'raw_quality.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"Source quality: {report['status']}; accepted={len(report['accepted_paths'])}, rejected={len(report['rejected_paths'])}, missing={len(report['missing'])}", flush=True)
    return report, str(output)


def run_download(args, start_dt, end_dt):
    downloader.main(start_date=start_dt, end_date=end_dt, output_dir=args.raw_dir)


def run_split(args, report):
    paths = [Path(p) for p in report['accepted_paths']]
    if not paths:
        return True
    success = split_by_signal.run_processing(Path(args.raw_dir), Path(args.per_signal_dir),
        overwrite=args.overwrite_split, max_workers=args.split_workers, input_files=paths)
    missing = [str(p) for p in paths if not split_by_signal.check_file_processed(p, Path(args.per_signal_dir))[0]]
    if missing:
        print(f'Split postcondition failed: {missing}', file=sys.stderr)
    return success and not missing


def run_extract(args, start_dt=None, end_dt=None):
    from feature_extractor.config import load_signals_from_file
    config = SimpleNamespace(event_dir=args.per_signal_dir, config_dir=args.config_dir,
        output_dir=args.processed_dir, year=None, month=None, day=None,
        signals=load_signals_from_file(args.signals_file) if args.signals_file else args.signals,
        features=args.features, overwrite=args.overwrite_extract,
        max_workers=args.max_workers, jobs_file=args.jobs_file, quality_report=args.quality_report,
        config_decisions=args.config_decisions, start_date=start_dt, end_date=end_dt,
        run_id=None, summary_file=args.summary_file, dry_run=args.dry_run, lock_held=True,
        auto_mode=args.auto, max_age_hours=args.max_age_hours)
    return run_checked(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', nargs='+', choices=['download', 'split', 'extract', 'all'], default=['all'])
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--raw-dir', default='/data2/ATSPM/raw_data')
    parser.add_argument('--per-signal-dir', default='/data2/ATSPM/raw_data/per_signal')
    parser.add_argument('--processed-dir', default='/data2/ATSPM/processed_data')
    parser.add_argument('--config-dir', default='/data2/ATSPM/config_data/signal_config_2025/lanes')
    parser.add_argument('--max-workers', type=int, default=12)
    parser.add_argument('--split-workers', type=int, default=4)
    parser.add_argument('--overwrite-split', action='store_true')
    parser.add_argument('--overwrite-extract', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--max-age-hours', type=int, default=24)
    parser.add_argument('--signals', nargs='*')
    parser.add_argument('--signals-file')
    parser.add_argument('--features', nargs='+')
    parser.add_argument('--single-date', action='store_true', help='Compatibility option; exact date filters use the common executor')
    parser.add_argument('--jobs-file')
    parser.add_argument('--quality-report')
    parser.add_argument('--config-decisions')
    parser.add_argument('--summary-file')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--lock-held', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    steps = ['download', 'split', 'extract'] if 'all' in args.steps else args.steps
    if bool(args.start_date) != bool(args.end_date):
        parser.error('Provide both --start-date and --end-date')
    if args.start_date:
        generate_dates(args.start_date, args.end_date)
    if args.jobs_file and steps != ['extract']:
        parser.error('--jobs-file requires --steps extract')
    if args.dry_run and steps != ['extract']:
        parser.error('--dry-run requires --steps extract')
    with pipeline_lock(args.processed_dir, args.lock_held):
        run_dir = Path(args.processed_dir).parent / 'operations' / 'runs' / uuid4().hex
        run_dir.mkdir(parents=True)
        if 'download' in steps:
            run_download(args, args.start_date, args.end_date)
        report, args.quality_report = source_report(args, run_dir)
        rc = 2 if report['status'] == 'partial' else 0
        if 'split' in steps and not run_split(args, report):
            return 1
        if 'extract' in steps:
            extract_rc = run_extract(args, args.start_date, args.end_date)
            if extract_rc == 1:
                return 1
            rc = max(rc, extract_rc)
        status = 'partial' if rc == 2 else 'complete'
        (run_dir / 'pipeline_summary.json').write_text(json.dumps({'status': status, 'exit_code': rc, 'quality_report': args.quality_report}, indent=2))
        print(f'Pipeline {status}: {run_dir}', flush=True)
        return rc


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print('Another ATSPM pipeline owns the shared lock.', file=sys.stderr)
        raise SystemExit(75)
    except Exception as exc:
        print(f'Pipeline failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
