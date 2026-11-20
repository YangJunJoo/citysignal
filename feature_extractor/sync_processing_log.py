#!/usr/bin/env python3
"""Inventory existing feature outputs as unverified in the separate run ledger.

File existence does not establish success. The legacy processing_log database is
preserved; only checked extraction may later create verified completion records.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.checked_runner import pipeline_lock
from feature_extractor.feature_registry import FEATURES
from feature_extractor.result_types import FeatureResult
from feature_extractor.run_store import RunStore, file_signature

FEATURE_PATHS = {name: spec.path for name, spec in FEATURES.items() if spec.scope == 'feature'}
PROFILE_PATHS = {name: spec.path for name, spec in FEATURES.items() if spec.scope == 'profile'}


def scan_disk(output_dir, year_month_prefix="2025-10"):
    """Return existing output paths by (signal, date, feature), without unpickling."""
    base = Path(output_dir) / "fdot_d5" / "feature_extraction"
    entries = {}
    for name, spec in FEATURES.items():
        root = base / spec.path
        if not root.is_dir():
            continue
        for path in root.glob("*/*"):
            if not path.is_file() or path.suffix != ".parquet":
                continue
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.stem):
                continue
            if year_month_prefix and not path.stem.startswith(year_month_prefix):
                continue
            key = (path.parent.name, path.stem, name)
            # Each logical output has one canonical Parquet file.
            if key not in entries:
                entries[key] = str(path)
    return entries


def _existing_attempts(db_path):
    if not Path(db_path).is_file():
        return {}
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        if "processing_log" in tables:
            raise ValueError("Use pipeline_runs.sqlite3 for imports; legacy processing_log must remain unchanged.")
        if "latest_attempts" not in tables:
            return {}
        return {(r[0], r[1]): {"status": r[2], "input_signature": r[3]}
                for r in conn.execute("SELECT signal_id,date_str,status,input_signature FROM latest_attempts")}


def sync(output_dir, db_path, raw_dir=None, config_dir=None, year_month_prefix="2025-10", dry_run=False):
    # Protect the inventory snapshot and latest-attempt check from concurrent
    # checked runs, using the same lock as extraction and manual backfills.
    with nullcontext() if dry_run else pipeline_lock(output_dir):
        return _sync_inventory(output_dir, db_path, raw_dir, config_dir, year_month_prefix, dry_run)


def _sync_inventory(output_dir, db_path, raw_dir=None, config_dir=None, year_month_prefix="2025-10", dry_run=False):
    """Append imported_unverified jobs; never override a checked attempt."""
    if Path(db_path).name == "processing_log.db":
        raise ValueError("Use pipeline_runs.sqlite3; the legacy processing log is read-only evidence.")
    disk = scan_disk(output_dir, year_month_prefix)
    existing = _existing_attempts(db_path)
    jobs = defaultdict(dict)
    for (signal, day, name), path in disk.items():
        jobs[(signal, day)][name] = path
    candidates = []
    for (signal, day), paths in sorted(jobs.items()):
        signature = hashlib.sha256(json.dumps({name: file_signature(path) for name, path in sorted(paths.items())}, sort_keys=True).encode()).hexdigest()
        previous = existing.get((signal, day))
        if previous and (previous["status"] != "imported_unverified" or previous["input_signature"] == signature):
            continue
        candidates.append((signal, day, paths, signature))
    summary = {"status": "imported_unverified", "feature_files": len(disk),
               "jobs_to_import": len(candidates), "features_to_import": sum(len(job[2]) for job in candidates),
               "dry_run": dry_run, "verified_success_created": 0}
    if candidates and not dry_run:
        store = RunStore(db_path)
        run_id = store.start_run()
        for signal, day, paths, signature in candidates:
            attempt_id = store.start_attempt(run_id, signal, day, signature, "unverified", "legacy-import", sorted(paths))
            results = {name: FeatureResult(name, "skipped", "legacy_unverified",
                       "Imported from disk inventory; schema, coverage and input/config versions have not been verified.",
                       output_file_path=path) for name, path in paths.items()}
            store.finish_attempt(attempt_id, "imported_unverified", results, "legacy_unverified")
        store.finish_run(run_id, "imported_unverified", summary)
        summary["run_id"] = run_id
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/data2/ATSPM/processed_data")
    parser.add_argument("--db-path", help="Separate pipeline_runs.sqlite3 ledger (defaults under output-dir)")
    parser.add_argument("--raw-dir", help="Accepted for compatibility; raw availability is checked by the pipeline")
    parser.add_argument("--config-dir", help="Accepted for compatibility; configuration is checked by the pipeline")
    parser.add_argument("--prefix", default="2025-10")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    sync(args.output_dir, args.db_path or str(Path(args.output_dir) / "pipeline_runs.sqlite3"),
         args.raw_dir, args.config_dir, args.prefix, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
