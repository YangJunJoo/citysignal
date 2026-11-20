#!/usr/bin/env python3
"""Build exact audited reprocessing jobs without running or modifying outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.config_validator import validate_config
from feature_extractor.feature_registry import FEATURES, resolve_features
from feature_extractor.run_store import file_signature

MANIFEST_VERSION = "audited-reprocess-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _quality_snapshot(path):
    if path is None:
        return {"rule_version": None, "status": "no_work", "files": [], "context_files": [],
                "accepted_paths": [], "rejected_paths": [], "missing": []}
    payload = json.loads(Path(path).read_text())
    reports = payload if isinstance(payload, list) else [payload]
    versions = {report.get("rule_version") for report in reports if report.get("rule_version")}
    if len(versions) > 1:
        raise ValueError("Cannot merge quality reports with different rule versions")
    statuses = {report.get("status") for report in reports if report.get("status")}
    status = next(iter(statuses)) if len(statuses) == 1 else ("partial" if statuses else "unverified")
    files, context, accepted, rejected, missing = {}, {}, set(), set(), []
    adjacent_gaps, overlaps = [], []
    reason_counts = Counter()
    for report in reports:
        accepted.update(report.get("accepted_paths", []))
        for entry in report.get("context_files", []):
            context[entry["path"]] = entry
        adjacent_gaps.extend(report.get("adjacent_gaps", []))
        overlaps.extend(report.get("overlaps", []))
        reason_counts.update(report.get("reason_counts", {}))
        rejected.update(report.get("rejected_paths", []))
        missing.extend(report.get("missing", []))
        for entry in report.get("files", []):
            day = entry.get("source_date") or entry.get("date")
            if not day:
                continue
            # Preserve the fields that establish coverage and source identity;
            # hourly/per-signal histograms remain in the cited original report.
            compact = {key: entry[key] for key in (
                "source_date", "date", "status", "reason_codes", "path", "signature",
                "input_version", "rule_version", "expected_coverage_verified", "contract_status",
                "contract", "expected_interval", "warnings", "min_timestamp", "max_timestamp", "timestamp_basis", "signals") if key in entry}
            previous = files.get(day)
            if previous and previous.get("status") != "valid" and compact.get("status") == "valid":
                continue
            files[day] = compact
    for entry in files.values():
        if entry.get("path") in rejected:
            entry["status"] = "source_incomplete"
    return {"rule_version": next(iter(versions), None), "status": status,
            "files": [files[day] for day in sorted(files)],
            "context_files": [context[key] for key in sorted(context)],
            "accepted_paths": sorted(accepted - rejected), "rejected_paths": sorted(rejected),
            "adjacent_gaps": adjacent_gaps, "overlaps": overlaps, "reason_counts": dict(reason_counts),
            "missing": missing, "original_report": str(Path(path).resolve()),
            "original_report_hash": sha256_file(path)}


def build_manifest(audit_dir, event_dir, config_dir, quality_report=None, include_partial_outputs=False):
    audit_dir, event_dir, config_dir = map(Path, (audit_dir, event_dir, config_dir))
    base = event_dir / "per_signal" if event_dir.name != "per_signal" and (event_dir / "per_signal").is_dir() else event_dir
    dates = sorted({row["date"] for row in _rows(audit_dir / "daily_summary.csv")})
    for day in dates:
        date.fromisoformat(day)
    candidates = {}
    category_counts = Counter()
    deferred_configs = []
    audit_hashes = {}

    def evidence(filename, line, message=""):
        path = audit_dir / filename
        if filename not in audit_hashes and path.is_file():
            audit_hashes[filename] = sha256_file(path)
        return {"source": str(path.resolve()), "source_hash": audit_hashes.get(filename),
                "row": line, "message": message}

    def add(signal, day, reason, origin, requested=None):
        signal, day = str(signal), str(day)
        date.fromisoformat(day)
        if not signal or signal in {".", ".."} or "/" in signal or "\\" in signal:
            raise ValueError(f"Invalid signal identity in audit: {signal!r}")
        key = (signal, day)
        item = candidates.setdefault(key, {"signal_id": signal, "date_str": day,
            "reasons": set(), "features": set(), "evidence": []})
        item["reasons"].add(reason)
        item["features"].update(resolve_features(requested))
        item["evidence"].append(origin)
        category_counts[reason] += 1

    for line, row in enumerate(_rows(audit_dir / "failed_jobs.csv"), 2):
        add(row["signal_id"], row["date_str"], "previous_extraction_failed",
            evidence("failed_jobs.csv", line, row.get("error_message", "")))
    for line, row in enumerate(_rows(audit_dir / "skipped_jobs.csv"), 2):
        previous = row.get("error_message", "")
        # This recognizes historical audit evidence only. It neither measures
        # size nor applies an event-rate/size exclusion to current jobs.
        if previous.startswith("skipped_chattering_sensor_file_size_") or previous.startswith("file_size_exceeds_"):
            add(row["signal_id"], row["date_str"], "removed_size_rule",
                evidence("skipped_jobs.csv", line, previous))
        elif previous in {"no_config_file", "missing_config"}:
            validation = validate_config(config_dir / f"{row['signal_id']}.csv", signal_id=row["signal_id"], data_date=row["date_str"])
            if validation.status == "valid":
                add(row["signal_id"], row["date_str"], "configuration_now_available",
                    evidence("skipped_jobs.csv", line, previous))
            else:
                deferred_configs.append({"signal_id": row["signal_id"], "date_str": row["date_str"],
                    "reason_code": validation.reason_code, "config_hash": validation.config_hash or "missing"})
    for line, row in enumerate(_rows(audit_dir / "latest_signal_outputs.csv"), 2):
        if row.get("run_outcome") != "reported_success":
            continue
        if not row.get("nonempty_features", "").strip():
            add(row["signal_id"], row["date"], "reported_success_without_nonempty_features",
                evidence("latest_signal_outputs.csv", line))
        elif include_partial_outputs:
            affected = {name for field in ("empty_features", "missing_outputs")
                        for name in row.get(field, "").split(";") if name in FEATURES}
            # Optional features do not create new default-completion failures.
            affected.intersection_update(resolve_features())
            if affected:
                add(row["signal_id"], row["date"], "audited_incomplete_outputs",
                    evidence("latest_signal_outputs.csv", line), sorted(affected))
    reproduction = audit_dir / "profile_error_reproduction.json"
    if reproduction.is_file():
        for index, item in enumerate(json.loads(reproduction.read_text()), 1):
            signal = str(item.get("signal") or item.get("signal_id"))
            for day in dates:
                current = date.fromisoformat(day)
                path = base / signal / f"atspm-{current.year}-{current.month}-{current.day}.parquet"
                # Extend only reproduced profile errors and only to actual
                # audited signal-days, never every signal across a date range.
                if path.is_file():
                    add(signal, day, "reproduced_profile_error",
                        evidence("profile_error_reproduction.json", index, item.get("error", "")))
    snapshot = _quality_snapshot(quality_report)
    source_by_date = {entry.get("source_date") or entry.get("date"): entry for entry in snapshot["files"]}
    jobs, missing_inputs = [], []
    blocked_counts = Counter()
    for (signal, day), candidate in sorted(candidates.items()):
        current = date.fromisoformat(day)
        path = base / signal / f"atspm-{current.year}-{current.month}-{current.day}.parquet"
        reasons = sorted(candidate["reasons"])
        if not path.is_file():
            missing_inputs.append({"signal_id": signal, "date_str": day, "input_file": str(path.resolve()),
                                   "reason_code": "missing_input", "selection_reasons": reasons})
            blocked_counts["missing_input"] += 1
            continue
        signature = file_signature(path)
        digest = sha256_file(path)
        if signature != file_signature(path):
            raise ValueError(f"Input changed while manifest was being built: {path}")
        validation = validate_config(config_dir / f"{signal}.csv", signal_id=signal, data_date=day)
        blocked = []
        if validation.status != "valid":
            blocked.append(validation.reason_code)
        source = dict(source_by_date.get(day, {"status": "unverified", "reason_codes": ["source_unverified"]}))
        source["expected_signal"] = source.pop("signals", {}).get(signal)
        if source.get("status") not in {"valid", "unverified"}:
            blocked.append(source.get("status", "source_incomplete"))
        unresolved = []
        if source.get("status") == "unverified" or source.get("expected_coverage_verified") is False:
            unresolved.append("source_unverified")
        for reason in blocked:
            blocked_counts[reason] += 1
        job = {"signal_id": signal, "date_str": day, "input_file": str(path.resolve()),
               "input_signature": signature, "input_hash": digest,
               "config_hash": validation.config_hash or "missing", "config_validation": validation.to_dict(),
               "features": list(resolve_features(sorted(candidate["features"]))),
               "reason": ";".join(reasons), "reasons": reasons, "evidence": candidate["evidence"],
               "source_quality": source, "preflight_status": "blocked" if blocked else "ready",
               "blocked_reasons": blocked, "unresolved_reasons": unresolved}
        jobs.append(job)
    summary = {"candidate_jobs": len(candidates), "jobs": len(jobs),
               "ready_jobs": sum(job["preflight_status"] == "ready" for job in jobs),
               "blocked_jobs": sum(job["preflight_status"] == "blocked" for job in jobs),
               "missing_input_jobs": len(missing_inputs),
               "source_unverified_jobs": sum(bool(job["unresolved_reasons"]) for job in jobs),
               "deferred_missing_config_jobs": len(deferred_configs),
               "selection_counts_before_deduplication": dict(category_counts),
               "reason_counts": dict(Counter(reason for job in jobs for reason in job["reasons"])),
               "blocked_reason_counts": dict(blocked_counts),
               "unique_signals": len({job["signal_id"] for job in jobs}),
               "date_counts": dict(Counter(job["date_str"] for job in jobs)),
               "feature_counts": dict(Counter(name for job in jobs for name in job["features"]))}
    return {"manifest_version": MANIFEST_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
            "audit_dir": str(audit_dir.resolve()), "event_dir": str(event_dir.resolve()),
            "config_dir": str(config_dir.resolve()), "audited_dates": dates,
            "dry_run": True, "operational_outputs_modified": False,
            "include_partial_outputs": include_partial_outputs, "summary": summary,
            "jobs": jobs, "blocked_candidates": missing_inputs,
            "deferred_missing_configs": deferred_configs, "quality_snapshot": snapshot}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="scratch/pipeline_audit_20260909")
    parser.add_argument("--event-dir", default="/data2/ATSPM/raw_data")
    parser.add_argument("--config-dir", default="/data2/ATSPM/config_data/signal_config_2025/lanes")
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-report")
    parser.add_argument("--include-partial-outputs", action="store_true",
                        help="Also select individually missing/empty required features in audited nonempty signal-days")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.audit_dir, args.event_dir, args.config_dir,
                              args.quality_report, args.include_partial_outputs)
    output = Path(args.output)
    for source_dir in (Path(args.event_dir).resolve(), Path(args.config_dir).resolve()):
        if output.resolve() == source_dir or source_dir in output.resolve().parents:
            parser.error("--output must be outside operational event/configuration trees")
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = output.with_name(output.stem + "_quality.json")
    snapshot_path.write_text(json.dumps(manifest.pop("quality_snapshot"), indent=2, ensure_ascii=False) + "\n")
    manifest["quality_report"] = str(snapshot_path.resolve())
    manifest["quality_report_hash"] = sha256_file(snapshot_path)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
