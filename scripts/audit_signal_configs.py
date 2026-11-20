#!/usr/bin/env python3
"""Read-only configuration audit; writes review artifacts only to --output-dir."""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.config_validator import RULES_VERSION, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="/data2/ATSPM/config_data/signal_config_2025/lanes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signals", nargs="+")
    parser.add_argument("--data-date", help="Date for evaluating explicit exclusion decisions (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="Affected period recorded in repair proposals")
    parser.add_argument("--end-date", help="Affected period recorded in repair proposals")
    parser.add_argument("--decisions", help="JSON list of explicit action decisions")
    parser.add_argument("--barrier-map", help="JSON object mapping allowed phase numbers to barriers")
    parser.add_argument("--normalization-samples", type=int, default=3, help="Number of valid normalization examples to save")
    args = parser.parse_args(argv)
    config_dir, output = Path(args.config_dir), Path(args.output_dir)
    if not config_dir.is_dir():
        parser.error("--config-dir must be an existing directory")
    # Audit output must never overwrite source configuration files.
    if output.resolve() == config_dir.resolve() or config_dir.resolve() in output.resolve().parents:
        parser.error("--output-dir must be outside the source configuration tree")
    output.mkdir(parents=True, exist_ok=True)
    decisions = json.loads(Path(args.decisions).read_text()) if args.decisions else []
    by_signal = {}
    for decision in decisions:
        signal = str(decision.get("signal_id", ""))
        if signal in by_signal:
            parser.error(f"Multiple decisions for {signal}; provide one decision per signal and audit date")
        by_signal[signal] = decision
    barrier_map = json.loads(Path(args.barrier_map).read_text()) if args.barrier_map else None
    paths = [config_dir / f"{signal}.csv" for signal in args.signals] if args.signals else sorted(config_dir.glob("*.csv"))
    records, actions, normalized = [], [], []
    for path in paths:
        result = validate_config(path, signal_id=path.stem, barrier_map=barrier_map,
                                 decision=by_signal.get(path.stem), data_date=args.data_date)
        record = result.to_dict()
        records.append(record)
        actions.append({
            "signal_id": path.stem, "action": result.action, "status": result.status,
            "reason_code": result.reason_code, "source": str(path), "config_hash": result.config_hash,
            "start_date": args.start_date or args.data_date, "end_date": args.end_date or args.data_date,
            "reason": "; ".join(issue["message"] for issue in result.issues if issue["severity"] == "error") or "Configuration passed validation.",
            "decision": result.decision,
        })
        if result.status == "valid" and len(normalized) < max(args.normalization_samples, 0):
            original = path.read_text(encoding="utf-8-sig")
            canonical = result.dataframe.to_csv(index=False)
            if original != canonical:
                directory = output / "normalization_examples"
                directory.mkdir(exist_ok=True)
                destination = directory / path.name
                destination.write_text(canonical, encoding="utf-8")
                diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), canonical.splitlines(keepends=True),
                                fromfile=str(path), tofile=str(destination)))
                diff_path = destination.with_suffix(".diff")
                diff_path.write_text(diff, encoding="utf-8")
                normalized.append({"signal_id": path.stem, "source": str(path), "config_hash": result.config_hash,
                                   "candidate": str(destination), "diff": str(diff_path),
                                   "scope": "Meaning-preserving header/value normalization only; source untouched."})
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "rules_version": RULES_VERSION,
        "config_dir": str(config_dir), "data_date": args.data_date,
        "affected_period": {"start_date": args.start_date, "end_date": args.end_date},
        "total": len(records), "status_counts": dict(Counter(r["status"] for r in records)),
        "reason_counts": dict(Counter(r["reason_code"] for r in records)),
        "action_counts": dict(Counter(r["action"] for r in records)),
        "source_files_modified": False,
    }
    for filename, value in [("validation.json", {"summary": summary, "configs": records}),
                            ("action_decisions.json", actions), ("normalization_proposals.json", normalized)]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["signal_id", "status", "reason_code", "action", "config_hash", "row_count", "path", "rules_version"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if any(r["status"] != "valid" or r["action"] != "use" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
