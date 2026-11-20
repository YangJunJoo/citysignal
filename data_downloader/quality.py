"""Streaming raw-source quality checks; no implicit date or timezone conversion.

A source contract is optional. Without an evidence-backed contract, only observed
coverage and cross-file gaps are assessed; expected daily coverage stays unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import os
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

RULE_VERSION = "raw-quality-3"
HOUR_NS = 3_600_000_000_000
MINUTE_NS = 60_000_000_000


@dataclass(frozen=True)
class SourceContract:
    timestamp_timezone: str | None = None
    day_timezone: str | None = None
    day_start: str = "00:00"
    date_offset_days: int = 0
    evidence: str | None = None
    gap_seconds: int = 3600

    def __post_init__(self):
        if self.gap_seconds < 60:
            raise ValueError("Minute-bin gap detection requires gap_seconds >= 60")
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", self.day_start):
            raise ValueError("day_start must be HH:MM")

    @property
    def confirmed(self):
        return bool(self.timestamp_timezone and self.day_timezone and self.evidence)

    def expected_bounds(self, source_date):
        if not self.confirmed or not source_date:
            return None
        d = date.fromisoformat(str(source_date)) + timedelta(days=self.date_offset_days)
        start = pd.Timestamp(f"{d.isoformat()} {self.day_start}").tz_localize(self.day_timezone)
        # Localize each calendar boundary independently: DST days contain 23/25 hours.
        end = pd.Timestamp(f"{(d + timedelta(days=1)).isoformat()} {self.day_start}").tz_localize(self.day_timezone)
        return start.tz_convert("UTC"), end.tz_convert("UTC")


def source_date_from_path(path):
    match = re.search(r"atspm-(\d{4})-(\d{1,2})-(\d{1,2})", Path(path).name)
    return date(*map(int, match.groups())).isoformat() if match else None


def file_signature(path):
    """Cheap replacement-sensitive fingerprint used together with the rule version."""
    p = Path(path)
    st = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        digest.update(handle.read(65536))
        if st.st_size > 65536:
            handle.seek(max(0, st.st_size - 65536))
            digest.update(handle.read(65536))
    return {"path": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "sample_sha256": digest.hexdigest()}


def _timestamp_array(values, contract):
    if pa.types.is_timestamp(values.type):
        timestamps = pc.cast(values, pa.timestamp("ns", tz=values.type.tz))
    elif pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
        # Native parser for the actual SunStore CSV format preserves fractional seconds.
        source_format = pc.fill_null(pc.match_substring_regex(
            values, r"^\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}(\.\d{1,9})?$"), False)
        base = pc.strptime(pc.utf8_slice_codeunits(values, 0, 19),
                           format="%m-%d-%Y %H:%M:%S", unit="ns", error_is_null=True)
        # strptime normalizes impossible calendar days and leap-second-like
        # inputs. Reject that normalization so invalid source values stay invalid.
        raw_day = pc.cast(pc.if_else(source_format, pc.utf8_slice_codeunits(values, 3, 5), "01"), pa.int64())
        raw_second = pc.cast(pc.if_else(source_format, pc.utf8_slice_codeunits(values, 17, 19), "00"), pa.int64())
        exact_calendar = pc.and_kleene(pc.equal(pc.day(base), raw_day), pc.equal(pc.second(base), raw_second))
        base = pc.if_else(exact_calendar, base, None)
        fraction = pc.if_else(source_format, pc.utf8_slice_codeunits(values, 20), "")
        fraction = pc.utf8_rpad(fraction, 9, padding="0")
        ns = pc.add(pc.cast(base, pa.int64()), pc.cast(fraction, pa.int64()))
        timestamps = pc.cast(pc.if_else(source_format, ns, None), pa.timestamp("ns"))
        # Supports typed/ISO fixtures and explicitly offset source timestamps too.
        remaining = pc.and_(pc.invert(source_format), pc.is_valid(values))
        if pc.any(remaining).as_py():
            parsed = pd.to_datetime(pc.filter(values, remaining).to_pandas(), errors="coerce", format="mixed")
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):
                parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
                if not contract.timestamp_timezone:
                    raise ValueError("Offset timestamps require an explicit timestamp timezone contract")
                if contract.timestamp_timezone != "UTC":
                    parsed = parsed.dt.tz_localize("UTC").dt.tz_convert(contract.timestamp_timezone).dt.tz_localize(None)
            arr = pc.fill_null(pc.cast(timestamps, pa.int64()), np.iinfo(np.int64).min).to_numpy(zero_copy_only=False).copy()
            arr[np.asarray(remaining)] = parsed.astype("datetime64[ns]").astype("int64").to_numpy()
            timestamps = pa.array(pd.to_datetime(arr, unit="ns"), type=pa.timestamp("ns"), from_pandas=True)
    else:
        raise ValueError(f"Unsupported timestamp type: {values.type}")
    if contract.timestamp_timezone and not timestamps.type.tz:
        # Ambiguous or nonexistent wall times are invalid, never guessed at DST.
        try:
            timestamps = pc.assume_timezone(timestamps, contract.timestamp_timezone,
                                            ambiguous="raise", nonexistent="raise")
        except pa.ArrowInvalid:
            localized = timestamps.to_pandas().dt.tz_localize(
                contract.timestamp_timezone, ambiguous='NaT', nonexistent='NaT')
            timestamps = pa.Array.from_pandas(localized)
    if timestamps.type.tz:
        timestamps = pc.cast(timestamps, pa.timestamp("ns", tz="UTC"))
    return timestamps


def _iso(ns, aware=False):
    ts = pd.Timestamp(int(ns), unit="ns")
    return (ts.tz_localize("UTC") if aware else ts).isoformat()


def _scan_batches(batches, columns, source_date, contract):
    normalized = {name.lower().replace("_", ""): name for name in columns}
    timestamp_col = normalized.get("timestamp")
    signal_col = normalized.get("signalid")
    if not timestamp_col or not signal_col:
        raise ValueError("Raw source requires timestamp and signalID columns")
    minutes = {}
    signal_counts = Counter()
    signal_hours = defaultdict(Counter)
    rows = invalid = invalid_signals = 0
    aware = bool(contract.timestamp_timezone)
    for batch in batches:
        ts = _timestamp_array(batch.column(timestamp_col), contract)
        aware = bool(ts.type.tz)
        signals = pc.cast(batch.column(signal_col), pa.string())
        counts = pc.value_counts(signals).to_pylist()
        for count in counts:
            if count['values'] is None or str(count['values']).strip() in ('', 'nan'):
                invalid_signals += count['counts']
            else:
                signal_counts[str(count['values'])] += count['counts']
        rows += len(batch)
        invalid += ts.null_count
        valid = pc.is_valid(ts)
        ints = pc.filter(pc.cast(ts, pa.int64()), valid)
        signals = pc.filter(signals, valid)
        if not len(ints):
            continue
        values = ints.to_numpy()
        table = pa.table({"minute": values // MINUTE_NS, "ts": ints})
        grouped = table.group_by("minute").aggregate([("ts", "count"), ("ts", "min"), ("ts", "max")])
        for item in grouped.to_pylist():
            key = item['minute']
            old = minutes.get(key)
            entry = (item['ts_count'], item['ts_min'], item['ts_max'])
            minutes[key] = entry if old is None else (old[0] + entry[0], min(old[1], entry[1]), max(old[2], entry[2]))
        grouped = pa.table({"signal": signals, "hour": values // HOUR_NS}).group_by(["signal", "hour"]).aggregate([("hour", "count")])
        for item in grouped.to_pylist():
            if item['signal'] is not None:
                signal_hours[item['signal']][item['hour']] += item['hour_count']
    ordered = sorted(minutes)
    hourly = Counter()
    for minute, (count, _, _) in minutes.items():
        hourly[minute // 60] += count
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        lo, hi = minutes[left][2], minutes[right][1]
        if (hi - lo) / 1e9 > contract.gap_seconds:
            gaps.append({"start": _iso(lo, aware), "end": _iso(hi, aware), "seconds": (hi-lo)/1e9, "reason_code": "internal_gap"})
    first = min((v[1] for v in minutes.values()), default=None)
    last = max((v[2] for v in minutes.values()), default=None)
    expected = contract.expected_bounds(source_date)
    bounds = None
    outside = 0
    if expected:
        lo, hi = (x.value for x in expected)
        bounds = {"start": expected[0].isoformat(), "end": expected[1].isoformat(), "hours": (hi-lo)/HOUR_NS}
        if first is not None:
            if (first-lo)/1e9 > contract.gap_seconds:
                gaps.append({"start": _iso(lo, True), "end": _iso(first, True), "seconds": (first-lo)/1e9, "reason_code": "leading_gap"})
            if (hi-last)/1e9 > contract.gap_seconds:
                gaps.append({"start": _iso(last, True), "end": _iso(hi, True), "seconds": (hi-last)/1e9, "reason_code": "trailing_gap"})
            # Contract boundaries are minute-aligned by design.
            outside = sum(v[0] for minute, v in minutes.items() if minute*MINUTE_NS < lo or minute*MINUTE_NS >= hi)
        hour_keys = range(lo//HOUR_NS, (hi+HOUR_NS-1)//HOUR_NS)
    elif ordered:
        hour_keys = range(ordered[0]//60, ordered[-1]//60+1)
    else:
        hour_keys = []
    reasons = []
    if not rows:
        reasons.append("empty_source")
    if invalid:
        reasons.append("invalid_timestamp")
    if invalid_signals:
        reasons.append("invalid_signal_id")
    if outside:
        reasons.append("outside_expected_interval")
    reasons.extend(sorted({g['reason_code'] for g in gaps}))
    # File-date sanity check works without asserting a timezone or midnight window.
    if first is not None and source_date and not expected:
        d = pd.Timestamp(source_date)
        if first < (d-pd.Timedelta(days=2)).value or last > (d+pd.Timedelta(days=2)).value:
            reasons.append("implausible_source_date")
    return {"status": "source_incomplete" if reasons else "valid", "reason_codes": reasons,
            "source_date": source_date, "row_count": rows, "valid_timestamp_count": rows-invalid,
            "invalid_timestamp_count": invalid, "invalid_signal_count": invalid_signals,
            "signal_count": len(signal_counts), "min_timestamp": _iso(first, aware) if first is not None else None,
            "max_timestamp": _iso(last, aware) if last is not None else None,
            "timestamp_basis": "UTC" if aware else "unconfirmed_wall_time",
            "expected_interval": bounds, "expected_coverage_verified": bool(expected) and not reasons,
            "contract_status": "confirmed" if contract.confirmed else "unconfirmed",
            "warnings": [] if contract.confirmed else ["source_date_timezone_contract_unconfirmed"],
            "outside_expected_interval_count": outside,
            "hourly_counts": { _iso(h*HOUR_NS, aware): hourly.get(h, 0) for h in hour_keys },
            "gaps": gaps,
            "signals": {signal: {"row_count": count, "hourly_counts": {_iso(h*HOUR_NS, aware): n for h,n in sorted(signal_hours[signal].items())}} for signal,count in sorted(signal_counts.items())}}


def assess_parquet(path, source_date=None, contract=None, cache_dir=None, batch_size=1_000_000):
    p = Path(path)
    contract = contract or SourceContract()
    source_date = source_date or source_date_from_path(p)
    signature = file_signature(p)
    cache_key = hashlib.sha256(json.dumps({"signature": signature, "rule": RULE_VERSION,
                      "contract": asdict(contract), "source_date": source_date}, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_dir)/f"{cache_key}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        cached['cache_hit'] = True
        return cached
    try:
        f = pq.ParquetFile(p)
        names = [n for n in f.schema_arrow.names if n.lower().replace('_','') in ('timestamp','signalid')]
        result = _scan_batches(f.iter_batches(batch_size=batch_size, columns=names), names, source_date, contract)
        result['schema'] = str(f.schema_arrow)
        result['schema_fields'] = {field.name.lower().replace('_', ''): str(field.type) for field in f.schema_arrow}
        missing_columns = sorted({'timestamp','signalid','eventcode','eventparam'} - set(result['schema_fields']))
        if missing_columns:
            result['status'] = 'source_incomplete'
            result['reason_codes'].append('missing_raw_columns')
            result['missing_columns'] = missing_columns
    except Exception as exc:
        result = {"status": "source_incomplete", "reason_codes": ["raw_read_failed"], "error": f"{type(exc).__name__}: {exc}", "source_date": source_date}
    result.update(path=str(p.resolve()), signature=signature, input_version=cache_key,
                  rule_version=RULE_VERSION, contract=asdict(contract), cache_hit=False)
    if signature != file_signature(p):
        result.update(status="source_incomplete", reason_codes=["source_changed_during_scan"])
        return result
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp = cache_path.with_suffix(f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(cache_path)
    return result


def classify_missing_dates(requested_dates, available_dates, http_statuses=None):
    available = set(available_dates)
    latest = max(available) if available else None
    http_statuses = http_statuses or {}
    result = []
    for d in sorted(set(requested_dates)-available):
        http = http_statuses.get(d)
        # Pending is supported only by a trailing 404 and an observed publication frontier.
        status = "source_pending" if http == 404 and latest and d > latest else "source_incomplete"
        result.append({"source_date": d, "status": status, "http_status": http,
                       "reason_code": "publication_pending" if status == "source_pending" else "source_missing"})
    return result


def assess_collection(paths, requested_dates=None, missing_statuses=None, contract=None, cache_dir=None):
    contract = contract or SourceContract()
    reports = [assess_parquet(path, contract=contract, cache_dir=cache_dir) for path in paths]
    reports.sort(key=lambda r: r.get('source_date') or r['path'])
    adjacent, overlaps = [], []
    for left, right in zip(reports, reports[1:]):
        if left.get('source_date') and right.get('source_date'):
            date_delta = (date.fromisoformat(right['source_date']) - date.fromisoformat(left['source_date'])).days
            if date_delta > 1:
                # Missing source dates are classified separately; disconnected
                # manifest windows do not invalidate their boundary files.
                continue
        if not left.get('max_timestamp') or not right.get('min_timestamp'):
            continue
        lo, hi = pd.Timestamp(left['max_timestamp']), pd.Timestamp(right['min_timestamp'])
        if left['timestamp_basis'] != right['timestamp_basis']:
            continue
        seconds = (hi-lo).total_seconds()
        record = {"left_date": left['source_date'], "right_date": right['source_date'], "start": lo.isoformat(), "end": hi.isoformat(), "seconds": seconds}
        if seconds > contract.gap_seconds:
            adjacent.append(dict(record, reason_code="adjacent_gap"))
            # The boundary itself is uncertain without a contract; quarantine both sides.
            for item in (left, right):
                item['status'] = 'source_incomplete'
                item['reason_codes'] = list(dict.fromkeys(item['reason_codes'] + ['adjacent_gap']))
        elif seconds < 0:
            overlaps.append(dict(record, reason_code="adjacent_overlap"))
            for item in (left, right):
                item['status'] = 'source_incomplete'
                item['reason_codes'] = list(dict.fromkeys(item['reason_codes'] + ['adjacent_overlap']))
    missing = classify_missing_dates(requested_dates or [], [r['source_date'] for r in reports], missing_statuses)
    accepted = [r['path'] for r in reports if r['status'] == 'valid']
    rejected = [r['path'] for r in reports if r['status'] != 'valid']
    return {"rule_version": RULE_VERSION, "status": "partial" if rejected or missing else ("complete" if reports else "no_work"),
            "files": reports, "missing": missing, "adjacent_gaps": adjacent, "overlaps": overlaps,
            "accepted_paths": accepted, "rejected_paths": rejected,
            "reason_counts": dict(Counter(reason for r in reports for reason in r['reason_codes']) + Counter(r['status'] for r in missing))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--contract', type=Path, help='JSON SourceContract with evidence-backed timezone/date rules')
    parser.add_argument('--http-statuses', type=Path, help='JSON date -> HTTP status for missing dates')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args(argv)
    if bool(args.start_date) != bool(args.end_date):
        parser.error('--start-date and --end-date must be provided together')
    contract = SourceContract(**json.loads(args.contract.read_text())) if args.contract else SourceContract()
    requested = pd.date_range(args.start_date, args.end_date).strftime('%Y-%m-%d').tolist() if args.start_date and args.end_date else None
    paths = [Path(p) for p in args.paths]
    if args.raw_dir:
        paths += [p for p in args.raw_dir.glob('atspm-*.parquet') if requested is None or source_date_from_path(p) in requested]
    statuses = json.loads(args.http_statuses.read_text()) if args.http_statuses else None
    report = assess_collection(paths, requested, statuses, contract, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('files', 'accepted_paths', 'rejected_paths')}, indent=2))
    return 2 if report['status'] == 'partial' else 0


if __name__ == '__main__':
    raise SystemExit(main())
