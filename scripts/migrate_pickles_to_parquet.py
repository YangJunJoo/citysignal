#!/usr/bin/env python3
"""Plan, verify, and commit a one-way migration of trusted local PKL datasets.

Conversion never deletes sources. The separate delete command requires every
planned conversion to be verified and checks both file hashes again. Historical
processing rows are preserved; the journal records their storage translation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
import pickle
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from feature_extractor import storage
from feature_extractor.checked_runner import pipeline_lock
from feature_extractor.archive_storage import (ArchivedObjectState, assert_artifacts_equal,
                                                read_artifact, write_artifact)


class _RetiredCacheState:
    """Inert destination for the state of a retired cache class."""
    pass


class _StateReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('citysignal_v2.data_loader', 'SignalDataCache'):
            return _RetiredCacheState
        return super().find_class(module, name)


def read_legacy_artifact(path):
    try:
        value = pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if exc.name != 'citysignal_v2':
            raise
        with Path(path).open('rb') as stream:
            value = _StateReader(stream).load()
    if isinstance(value, _RetiredCacheState):
        return ArchivedObjectState('citysignal_v2.data_loader.SignalDataCache', vars(value))
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def signature(path):
    stat = Path(path).stat()
    return f'{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}'


def plan_migration(roots, destination, exclude_paths=()):
    roots = [Path(root).resolve() for root in roots]
    excluded = [Path(path).resolve() for path in exclude_paths]
    entries, skipped, seen = [], [], set()
    for root in roots:
        if not root.is_dir() or root == Path('/'):
            raise ValueError(f'Invalid migration root: {root}')
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = [name for name in subdirs if name != '.git' and
                          not any((Path(directory) / name).resolve().is_relative_to(path) for path in excluded)]
            for name in names:
                if not name.endswith('.pkl'):
                    continue
                source = Path(directory) / name
                if source.is_symlink():
                    skipped.append(dict(path=str(source), reason='symlink'))
                    continue
                source = source.resolve()
                if source in seen:
                    continue
                seen.add(source)
                target = source.with_suffix('.parquet')
                entries.append(dict(source=str(source), target=str(target), fingerprint=fingerprint(source),
                                    target_existed=target.exists()))
    entries.sort(key=lambda row: row['source'])
    plan = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(), roots=list(map(str, roots)),
                exclude_paths=list(map(str, excluded)), files=entries, skipped=skipped,
                file_count=len(entries), source_bytes=sum(row['fingerprint'][2] for row in entries),
                existing_targets=sum(row['target_existed'] for row in entries))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: v for k, v in plan.items() if k not in {'files', 'skipped'}}, indent=2))
    return plan


def open_journal(path, plan_path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=FULL')
    conn.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute('''CREATE TABLE IF NOT EXISTS files (
        source TEXT PRIMARY KEY, target TEXT, state TEXT, source_signature TEXT,
        target_signature TEXT, record_json TEXT NOT NULL)''')
    plan_hash = digest(plan_path)
    previous = conn.execute('SELECT value FROM metadata WHERE key=?', ('plan_sha256',)).fetchone()
    if previous and previous[0] != plan_hash:
        raise ValueError('The journal belongs to a different immutable migration plan')
    conn.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('plan_sha256', plan_hash))
    conn.commit()
    return conn


def record_result(conn, record):
    conn.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?)',
                 (record['source'], record['target'], record['state'], record.get('source_signature'),
                  record.get('target_signature'), json.dumps(record, separators=(',', ':'))))


def convert_one(entry, journal_path):
    source, target = Path(entry['source']), Path(entry['target'])
    record = dict(entry, state='failed')
    try:
        if source.is_symlink() or source.resolve() != source or target.parent.resolve() != source.parent:
            raise ValueError('The planned path now resolves through a symlink')
        if fingerprint(source) != entry['fingerprint']:
            raise ValueError('Source changed after the plan was recorded')
        before = digest(source)
        old_signature = signature(source)
        frame = read_legacy_artifact(source)
        if target.exists():
            # An existing export is never silently overwritten.
            assert_artifacts_equal(frame, read_artifact(target))
        else:
            write_artifact(frame, target, provenance=dict(
                source_path=str(source), source_signature=old_signature,
                source_sha256=before, journal_path=str(Path(journal_path).resolve())))
        assert_artifacts_equal(frame, read_artifact(target))
        if fingerprint(source) != entry['fingerprint'] or digest(source) != before:
            raise ValueError('Source changed while it was being converted')
        record.update(state='verified', source_signature=old_signature, source_sha256=before,
                      target_signature=signature(target), target_sha256=digest(target),
                      target_bytes=target.stat().st_size,
                      artifact_type=type(frame).__name__,
                      rows=len(frame) if isinstance(frame, pd.DataFrame) else None,
                      columns=len(frame.columns) if isinstance(frame, pd.DataFrame) else None)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return record


def initialize_worker():
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def convert_plan(plan_path, journal_path, workers=8):
    plan = json.loads(Path(plan_path).read_text())
    conn = open_journal(journal_path, plan_path)
    previous = {source: json.loads(record) for source, record in conn.execute('SELECT source,record_json FROM files')}
    pending = []
    for entry in plan['files']:
        old = previous.get(entry['source'])
        if old and old['state'] in {'verified', 'deleted'} and Path(old['target']).is_file():
            if digest(old['target']) == old['target_sha256']:
                continue
        pending.append(entry)
    started, failures = time.monotonic(), 0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as executor:
            futures = {executor.submit(convert_one, entry, str(Path(journal_path).resolve())): entry for entry in pending}
            for number, future in enumerate(as_completed(futures), 1):
                record = future.result()
                record_result(conn, record)
                if record['state'] == 'failed':
                    failures += 1
                    if failures <= 25:
                        print('FAILED', record['source'], record.get('error'), flush=True)
                if number % 100 == 0:
                    conn.commit()
                if number % 500 == 0 or number == len(pending):
                    print(json.dumps(dict(completed=number, pending_total=len(pending), failures=failures,
                                          seconds=round(time.monotonic() - started, 1))), flush=True)
        conn.commit()
        return summarize(conn, plan, Path(journal_path).with_suffix('.summary.json'))
    finally:
        conn.close()


def delete_sources(plan_path, journal_path):
    """Delete only sources with durable conversion evidence and matching hashes."""
    plan = json.loads(Path(plan_path).read_text())
    conn = open_journal(journal_path, plan_path)
    records = {source: json.loads(record) for source, record in conn.execute('SELECT source,record_json FROM files')}
    missing = [entry['source'] for entry in plan['files'] if records.get(entry['source'], {}).get('state') not in {'verified', 'deleted'}]
    if missing:
        conn.close()
        raise ValueError(f'Deletion refused: {len(missing)} planned files lack verified conversion')
    dirty_directories = set()

    def commit_deletions():
        for parent in dirty_directories:
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        conn.commit()
        dirty_directories.clear()

    try:
        for number, entry in enumerate(plan['files'], 1):
            record = records[entry['source']]
            source, target = Path(entry['source']), Path(entry['target'])
            if not target.is_file() or digest(target) != record['target_sha256']:
                raise ValueError(f'Converted output changed; refusing deletion: {target}')
            if source.exists():
                if source.is_symlink() or fingerprint(source) != entry['fingerprint'] or digest(source) != record['source_sha256']:
                    raise ValueError(f'Original input changed; refusing deletion: {source}')
                source.unlink()
                dirty_directories.add(source.parent)
            record['state'] = 'deleted'
            record_result(conn, record)
            if number % 100 == 0:
                commit_deletions()
            if number % 1000 == 0 or number == len(plan['files']):
                print(json.dumps(dict(deleted=number, total=len(plan['files']))), flush=True)
        commit_deletions()
        return summarize(conn, plan, Path(journal_path).with_suffix('.summary.json'))
    finally:
        conn.close()


def summarize(conn, plan, destination):
    counts = dict(conn.execute('SELECT state,COUNT(*) FROM files GROUP BY state'))
    records = [json.loads(row[0]) for row in conn.execute('SELECT record_json FROM files')]
    report = dict(planned_files=plan['file_count'], states=counts, source_bytes=plan['source_bytes'],
                  target_bytes=sum(r.get('target_bytes', 0) for r in records if r['state'] != 'failed'),
                  failures=[r for r in records if r['state'] == 'failed'], skipped=plan['skipped'])
    Path(destination).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {'failures', 'skipped'}}, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'convert', 'delete'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--roots', nargs='+', type=Path)
    parser.add_argument('--exclude', action='append', type=Path, default=[])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--operational-output', type=Path,
                        help='Hold the same lock as extraction during conversion/deletion')
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        if not args.roots:
            parser.error('--roots is required for plan')
        plan_migration(args.roots, args.plan, args.exclude)
        return 0
    if not args.journal:
        parser.error('--journal is required for conversion and deletion')
    with pipeline_lock(args.operational_output) if args.operational_output else nullcontext():
        report = (convert_plan(args.plan, args.journal, args.workers) if args.mode == 'convert'
                  else delete_sources(args.plan, args.journal))
    return 1 if report['states'].get('failed', 0) else 0


if __name__ == '__main__':
    raise SystemExit(main())
