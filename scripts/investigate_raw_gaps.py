#!/usr/bin/env python3
"""Compare staged remote CSV candidates with retained raw Parquet and neighbors.

Read-only for operational data. This command never replaces source files. Use
--fetch to download candidates into the selected investigation output directory.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.csv as csv
import requests
from data_downloader.data_downloader import download_csv, _remote_version
from data_downloader.quality import SourceContract, RULE_VERSION, assess_parquet, _scan_batches, file_signature

BASE_URL = 'https://sunstore.cflsmartroads.com/stagedfiles/csvfiles/atspm-{date}.csv'


def url_date(d):
    return f'{d.year}-{d.month}-{d.day}'


def probe(url):
    try:
        with requests.head(url, timeout=(15, 30), allow_redirects=True, verify=False) as response:
            return {'url': url, 'http_status': response.status_code, 'headers': dict(response.headers),
                    'checked_at': datetime.now(timezone.utc).isoformat()}
    except requests.RequestException as exc:
        return {'url': url, 'http_status': None, 'error': f'{type(exc).__name__}: {exc}'}


def assess_csv_candidate(path, source_date, contract, cache_dir=None):
    signature = file_signature(path)
    key = hashlib.sha256(json.dumps({'signature': signature, 'rule': RULE_VERSION,
               'contract': asdict(contract), 'date': source_date}, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/f'csv-{key}.json' if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    reader = csv.open_csv(path, read_options=csv.ReadOptions(block_size=16*1024*1024),
        convert_options=csv.ConvertOptions(column_types={'signalID':pa.string(),'timeStamp':pa.string(),
                                                         'eventCode':pa.int64(),'eventParam':pa.int64()}))
    report = _scan_batches(reader, reader.schema.names, source_date, contract)
    report.update(path=str(Path(path).resolve()), signature=signature, schema=str(reader.schema),
                  schema_fields={field.name.lower().replace('_',''):str(field.type) for field in reader.schema},
                  input_version=key, rule_version=RULE_VERSION)
    if file_signature(path) != signature:
        raise ValueError('Candidate changed during scan')
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(report, indent=2))
    return report


def _boundary_gaps(previous, current, following, gap_seconds):
    import pandas as pd
    records = []
    for left, right in [(previous, current), (current, following)]:
        if not left or not right or not left.get('max_timestamp') or not right.get('min_timestamp'):
            continue
        start, end = pd.Timestamp(left['max_timestamp']), pd.Timestamp(right['min_timestamp'])
        seconds = (end-start).total_seconds()
        if seconds > gap_seconds:
            records.append({'left_date': left['source_date'], 'right_date': right['source_date'],
                            'start': start.isoformat(), 'end': end.isoformat(), 'seconds': seconds})
    return records


def compare_candidate(local, candidate, previous=None, following=None, gap_seconds=3600):
    old_gaps = _boundary_gaps(previous, local, following, gap_seconds)
    new_gaps = _boundary_gaps(previous, candidate, following, gap_seconds)
    local_hourly, new_hourly = local.get('hourly_counts', {}), candidate.get('hourly_counts', {})
    retained = all(new_hourly.get(hour, 0) >= count for hour, count in local_hourly.items())
    added_hours = sorted(h for h, n in new_hourly.items() if n > 0 and not local_hourly.get(h))
    same = all(local.get(k) == candidate.get(k) for k in ('row_count', 'min_timestamp', 'max_timestamp', 'hourly_counts', 'signals'))
    # Actual content must improve and all known gaps must close before publication.
    schema_compatible = local.get('schema_fields') == candidate.get('schema_fields') and bool(candidate.get('schema_fields'))
    recoverable = (schema_compatible and retained and bool(added_hours) and candidate['status'] == 'valid' and not new_gaps)
    return {'conclusion': 'recoverable' if recoverable else 'source_still_incomplete',
            'same_observed_statistics': same, 'schema_compatible': schema_compatible, 'retains_existing_hourly_counts': retained,
            'added_occupied_hours': added_hours, 'local_boundary_gaps': old_gaps,
            'candidate_boundary_gaps': new_gaps,
            'candidate_row_count': candidate['row_count'], 'local_row_count': local.get('row_count'),
            'replacement_performed': False,
            'coverage_limitation': 'Expected daily boundaries remain unconfirmed unless an evidence-backed contract is supplied.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('/data2/ATSPM/raw_data'))
    parser.add_argument('--dates', nargs='+', default=['2026-08-26', '2026-09-01'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args(argv)
    contract = SourceContract(**json.loads(args.contract.read_text())) if args.contract else SourceContract()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage, cache = args.output_dir/'staging', args.output_dir/'cache'
    stage.mkdir(exist_ok=True)
    results = []
    for label in args.dates:
        d = date.fromisoformat(label)
        neighbors = {}
        probes = {}
        for offset in [-1,0,1]:
            neighbor = d + timedelta(days=offset)
            n = url_date(neighbor)
            p = args.raw_dir/f'atspm-{n}.parquet'
            probes[neighbor.isoformat()] = probe(BASE_URL.format(date=n))
            neighbors[offset] = assess_parquet(p, contract=contract, cache_dir=cache) if p.exists() else None
        candidate_path = stage/f'atspm-{url_date(d)}.csv'
        evidence_path = Path(str(candidate_path)+'.download.json')
        existing_evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        current_version = _remote_version(probes[label].get('headers', {}))
        version_consistent = (probes[label].get('http_status') == 200 and any(current_version.values())
                              and all(existing_evidence.get(key) == value for key, value in current_version.items() if value))
        already_downloaded = (version_consistent and candidate_path.exists() and existing_evidence.get('status') == 'downloaded'
                              and candidate_path.stat().st_size == existing_evidence.get('byte_count'))
        downloaded = already_downloaded or (args.fetch and download_csv(BASE_URL.format(date=url_date(d)),candidate_path,max_retries=3))
        result = {'source_date': label, 'local': neighbors[0], 'remote_probes': probes,
                  'download_evidence': json.loads(evidence_path.read_text()) if evidence_path.exists() else None,
                  'conclusion': 'unavailable', 'replacement_performed': False}
        if downloaded and neighbors[0]:
            try:
                candidate = assess_csv_candidate(candidate_path, label, contract, cache)
                result['candidate'] = candidate
                result.update(compare_candidate(neighbors[0],candidate,neighbors[-1],neighbors[1],contract.gap_seconds))
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
        elif not downloaded:
            result['error'] = 'No complete validated candidate download available'
        results.append(result)
        (args.output_dir/f'investigation_{label}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('local','candidate','remote_probes','download_evidence')},indent=2),flush=True)
    report = {'investigated_at': datetime.now(timezone.utc).isoformat(), 'contract': asdict(contract), 'results': results}
    (args.output_dir/'recovery_investigation.json').write_text(json.dumps(report,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
