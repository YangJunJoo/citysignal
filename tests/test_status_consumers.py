"""Consumers must not recover obsolete success from legacy rows or file presence."""
import importlib.util
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from feature_extractor.feature_registry import output_path
from feature_extractor.result_types import FeatureResult
from feature_extractor.run_store import RunStore
from feature_extractor.sync_processing_log import sync


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / relative)
    instance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instance)
    return instance


def ledger(tmp_path):
    store = RunStore(tmp_path / "pipeline_runs.sqlite3")
    run = store.start_run()
    return store, run


def attempt(store, run, signal, status, feature_status="no_data"):
    identifier = store.start_attempt(run, signal, "2026-09-06", "source", "config", "v2", ["volume"])
    store.finish_attempt(identifier, status, {"volume": FeatureResult("volume", feature_status, "no_events")}, "no_events" if status != "complete" else "")
    return identifier


def dashboard(tmp_path, monkeypatch):
    dash = module("atspm_test_dashboard", "dashboard/dashboard_server.py")
    monkeypatch.setattr(dash, "RUN_DB_PATH", str(tmp_path / "pipeline_runs.sqlite3"))
    monkeypatch.setattr(dash, "DB_PATH", str(tmp_path / "processing_log.db"))
    monkeypatch.setattr(dash, "RAW_PER_SIGNAL_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(dash, "DOWNLOAD_LOG_PATH", str(tmp_path / "missing-download.json"))
    monkeypatch.setattr(dash, "SPLIT_CACHE_PATH", str(tmp_path / "missing-split.json"))
    return dash


def legacy_database(path, signals):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE processing_log (signal_id TEXT,date_str TEXT,feature_name TEXT,status TEXT)")
        conn.executemany("INSERT INTO processing_log VALUES (?,'2026-09-06','volume','success')", [(s,) for s in signals])


def test_dashboard_latest_failure_cannot_be_hidden_by_old_success(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    store, run = ledger(tmp_path)
    attempt(store, run, "A", "complete", "not_applicable")
    attempt(store, run, "A", "partial", "no_data")
    legacy_database(tmp_path / "processing_log.db", ["A", "B"])
    stats = dash.get_feature_statistics()
    assert stats["run_signals_success"] == 0
    assert stats["legacy_unverified_jobs"] == 1
    assert stats["job_status_counts"] == {"partial": 1}
    assert stats["job_reason_counts"] == {"no_events": 1}
    daily = dash.get_daily_status(["2026-09-06"])[0]
    assert daily["extraction"]["status"] == "partial"
    assert daily["extraction"]["success"] == 0
    assert daily["extraction"]["unverified"] == 1


def test_dashboard_never_calls_75_percent_complete(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    store, run = ledger(tmp_path)
    for signal in ["A", "B", "C", "D"]:
        attempt(store, run, signal, "failed" if signal == "D" else "complete", "failed" if signal == "D" else "not_applicable")
        raw = tmp_path / "raw" / signal / "atspm-2026-9-6.parquet"
        raw.parent.mkdir(parents=True)
        raw.touch()
    daily = dash.get_daily_status(["2026-09-06"])[0]
    assert daily["extraction"]["success"] == 3
    assert daily["extraction"]["failed"] == 1
    assert daily["extraction"]["status"] == "partial"
    assert daily["color"] != "green"


def test_dashboard_legacy_only_is_unverified_and_read_only(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    legacy_database(tmp_path / "processing_log.db", ["A", "B"])
    before = (tmp_path / "processing_log.db").read_bytes()
    stats = dash.get_feature_statistics()
    assert stats["total_features_success"] == 0
    assert stats["legacy_unverified_jobs"] == 2
    assert dash.get_daily_status(["2026-09-06"])[0]["extraction"]["status"] == "legacy_unverified"
    assert not (tmp_path / "pipeline_runs.sqlite3").exists()
    assert before == (tmp_path / "processing_log.db").read_bytes()


def test_partial_exit_is_not_retried(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    process = SimpleNamespace(wait=lambda: 2)
    state = dict(process=process, status="running", auto_resume=True, retry_count=0)
    with patch.object(dash.threading, "Thread") as thread:
        dash.monitor_pipeline_process(process, state, threading.Lock())
        thread.assert_not_called()
    assert state["status"] == "partial"


def test_inventory_is_unverified_idempotent_and_does_not_replace_checked_jobs(tmp_path):
    output = tmp_path / "processed"
    path = output_path(output, "vehicle_phase_profile", "A", "2026-09-06")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not validated by inventory")
    db = output / "pipeline_runs.sqlite3"
    dry = sync(output, db, year_month_prefix="2026-09", dry_run=True)
    assert dry["jobs_to_import"] == 1
    assert not db.exists()
    first = sync(output, db, year_month_prefix="2026-09")
    assert first["verified_success_created"] == 0
    store = RunStore(db)
    row = store.latest("A", "2026-09-06")
    assert row["status"] == "imported_unverified"
    assert row["results"]["vehicle_phase_profile"]["reason_code"] == "legacy_unverified"
    assert row["results"]["vehicle_phase_profile"]["status"] == "skipped"
    assert sync(output, db, year_month_prefix="2026-09")["jobs_to_import"] == 0
    run = store.start_run()
    checked_id = attempt(store, run, "A", "partial")
    path.write_bytes(b"changed after checked attempt")
    assert sync(output, db, year_month_prefix="2026-09")["jobs_to_import"] == 0
    assert store.latest("A", "2026-09-06")["attempt_id"] == checked_id


def test_inventory_refuses_legacy_database(tmp_path):
    import pytest
    legacy = tmp_path / "processing_log.db"
    legacy_database(legacy, ["A"])
    before = legacy.read_bytes()
    with pytest.raises(ValueError, match="legacy"):
        sync(tmp_path, legacy, dry_run=True)
    assert legacy.read_bytes() == before


def test_backlog_uses_checked_runner_and_returns_partial_exit(tmp_path):
    backlog = module("atspm_test_backlog", "scripts/run_historical_backlog.py")
    with patch.object(backlog, "run_checked", return_value=2) as run:
        assert backlog.main(["--start-date", "2026-09-01", "--end-date", "2026-09-06", "--dry-run", "--signals", "1067"]) == 2
    args = run.call_args.args[0]
    assert (args.start_date, args.end_date, args.signals) == ("2026-09-01", "2026-09-06", ["1067"])
    assert args.dry_run
    assert not hasattr(args, 'use_v2')


def test_equal_counts_for_different_signals_are_not_complete(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    store, run = ledger(tmp_path)
    attempt(store, run, "old-signal", "complete", "not_applicable")
    raw = tmp_path / "raw" / "new-signal" / "atspm-2026-9-6.parquet"
    raw.parent.mkdir(parents=True)
    raw.touch()
    result = dash.get_daily_status(["2026-09-06"])[0]["extraction"]
    assert result["status"] == "partial"
    assert result["pending"] == 1
    assert result["total"] == 2


def test_dashboard_latest_run_omits_large_result_summary(tmp_path, monkeypatch):
    dash = dashboard(tmp_path, monkeypatch)
    store, run = ledger(tmp_path)
    store.finish_run(run, 'partial', {'large_result_blob': 'x' * 10000})
    result = dash.get_feature_statistics()['latest_run']
    assert result['run_id'] == run
    assert result['status'] == 'partial'
    assert set(result) == {'run_id', 'started_at', 'finished_at', 'status'}
