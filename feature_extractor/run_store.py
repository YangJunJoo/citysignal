"""Append-only execution ledger, separate from the preserved legacy processing log."""
from contextlib import contextmanager
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .result_types import RULE_VERSION


def now():
    return datetime.now(timezone.utc).isoformat()


def file_signature(path):
    stat = Path(path).stat()
    return f'{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}'


def content_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else 'missing'


class RunStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT,
                    status TEXT, summary_json TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    signal_id TEXT NOT NULL, date_str TEXT NOT NULL,
                    input_signature TEXT, config_hash TEXT, engine TEXT, rule_version TEXT,
                    required_json TEXT, status TEXT, reason_code TEXT,
                    started_at TEXT, finished_at TEXT);
                CREATE INDEX IF NOT EXISTS attempts_job ON attempts(signal_id,date_str,attempt_id);
                CREATE INDEX IF NOT EXISTS attempts_run ON attempts(run_id);
                CREATE TABLE IF NOT EXISTS feature_results (
                    attempt_id INTEGER NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
                    reason_code TEXT, message TEXT, output_file_path TEXT, row_count INTEGER,
                    output_signature TEXT, PRIMARY KEY(attempt_id,name));
                CREATE VIEW IF NOT EXISTS latest_attempts AS
                    SELECT a.* FROM attempts a WHERE a.attempt_id=(
                        SELECT MAX(b.attempt_id) FROM attempts b
                        WHERE b.signal_id=a.signal_id AND b.date_str=a.date_str);
            ''')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=120)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def start_run(self, run_id=None):
        run_id = run_id or uuid4().hex
        with self.connect() as conn:
            conn.execute('INSERT INTO runs VALUES (?,?,NULL,?,NULL)', (run_id, now(), 'running'))
        return run_id

    def finish_run(self, run_id, status, summary):
        with self.connect() as conn:
            conn.execute('UPDATE runs SET finished_at=?,status=?,summary_json=? WHERE run_id=?',
                         (now(), status, json.dumps(summary), run_id))

    def start_attempt(self, run_id, signal, date, signature, config_hash, engine, required):
        with self.connect() as conn:
            return conn.execute('''INSERT INTO attempts
                (run_id,signal_id,date_str,input_signature,config_hash,engine,rule_version,
                 required_json,status,reason_code,started_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (run_id, signal, date, signature, config_hash, engine, RULE_VERSION,
                 json.dumps(list(required)), 'running', '', now())).lastrowid

    def finish_attempt(self, attempt_id, status, results, reason_code=''):
        with self.connect() as conn:
            for result in results.values():
                signature = file_signature(result.output_file_path) if result.status == 'success' and result.output_file_path else None
                conn.execute('INSERT INTO feature_results VALUES (?,?,?,?,?,?,?,?)',
                             (attempt_id, result.name, result.status, result.reason_code,
                              result.message[:4000], result.output_file_path, result.row_count, signature))
            conn.execute('UPDATE attempts SET status=?,reason_code=?,finished_at=? WHERE attempt_id=?',
                         (status, reason_code, now(), attempt_id))

    def latest(self, signal, date):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM attempts WHERE signal_id=? AND date_str=? ORDER BY attempt_id DESC LIMIT 1',
                               (signal, date)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result['results'] = {r['name']: dict(r) for r in conn.execute(
                'SELECT * FROM feature_results WHERE attempt_id=?', (row['attempt_id'],))}
            from .storage import resolve_migrated_result
            for feature in result['results'].values():
                feature['output_file_path'], feature['output_signature'] = resolve_migrated_result(
                    feature['output_file_path'], feature['output_signature'])
            return result

    def reusable(self, signal, date, signature, config_hash, engine, features):
        row = self.latest(signal, date)
        if not row or row['status'] == 'running':
            return None
        if (row['input_signature'], row['config_hash'], row['engine'], row['rule_version']) != (
                signature, config_hash, engine, RULE_VERSION):
            return None
        if set(features) != set(json.loads(row['required_json'])):
            return None
        # Failures are retryable. Empty, excluded and configuration-blocked jobs are
        # terminal for this version but remain partial on subsequent invocations.
        if any(r['status'] == 'failed' or r['reason_code'] in {'cache_load_failed', 'stale_manifest', 'input_changed_during_processing', 'dependency_publication_failed'} for r in row['results'].values()):
            return None
        for result in row['results'].values():
            if result['status'] == 'success':
                try:
                    if file_signature(result['output_file_path']) != result['output_signature']:
                        return None
                except (OSError, TypeError):
                    return None
        return row
