#!/usr/bin/env python3
"""Run historical signal-days through the same checked pipeline as daily work."""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.checked_runner import run_checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-09-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--event-dir", default="/data2/ATSPM/raw_data")
    parser.add_argument("--config-dir", default="/data2/ATSPM/config_data/signal_config_2025/lanes")
    parser.add_argument("--output-dir", default="/data2/ATSPM/processed_data")
    parser.add_argument("--dry-run", action="store_true", help="Write an exact job manifest without processing outputs")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate jobs even when a verified version is reusable")
    parser.add_argument("--max-workers", type=int, default=38)
    parser.add_argument("--signals", nargs="+")
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--jobs-file")
    parser.add_argument("--quality-report")
    parser.add_argument("--config-decisions")
    parser.add_argument("--summary-file")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
            parser.error("--start-date must not follow --end-date")
    except ValueError:
        parser.error("Dates must use YYYY-MM-DD")
    args.year = args.month = args.day = None
    args.auto_mode = False
    args.lock_held = False
    # The common runner validates input/config/output versions and the complete
    # required feature set. Legacy success rows are never pre-filtered here.
    return run_checked(args)


if __name__ == "__main__":
    raise SystemExit(main())
