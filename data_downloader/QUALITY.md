Raw source quality and recovery
===============================

Run the batch scanner before placing raw files in a split/extraction manifest:

```sh
python -m data_downloader.quality --raw-dir /data2/ATSPM/raw_data \
  --start-date 2026-08-26 --end-date 2026-09-08 \
  --cache-dir /data2/ATSPM/operations/raw_quality_cache \
  --output /data2/ATSPM/operations/raw_quality.json
```

The library API is `assess_parquet(path, source_date=None, contract=None,
cache_dir=None)` and `assess_collection(paths, requested_dates=None,
missing_statuses=None, contract=None, cache_dir=None)`. The collection produces
`accepted_paths`, `rejected_paths`, separate missing-date states, reason counts,
and `complete` / `partial` / `no_work`. CLI exits 2 for partial and 0 otherwise;
execution errors exit nonzero. `accepted_paths` passed the configured checks;
`expected_coverage_verified` explicitly says whether expected daily boundaries
were also checked.

Every scan reads timestamps and signal IDs in bounded batches, counts invalid
values, records exact extrema including fractional seconds, hourly row counts
for the entire source and individual signals, and gaps between occupied minute
bins. The default gap threshold is 3,600 seconds. Quiet individual signals do
not make the entire source incomplete. Adjacent-file gaps and overlapping
coverage reject both boundary files until an evidence-backed date contract can
attribute the missing interval. Neither file size nor chattering controls
eligibility.

Cache identity includes resolved path, size, nanosecond mtime/ctime, first/last
64 KiB hashes, source date, contract, and rule version. Changes to any component
invalidate reuse. This is a replacement-sensitive signature, not a full content
hash. Files changed during scanning are rejected. Collection-boundary decisions
are recomputed, not cached in the independent per-file result.

Source date and timezone contract
---------------------------------

SunStore's public Controller Event Log metadata and field definitions were
checked on 2026-09-11. They describe the timestamp as record creation time but
do not specify the staged CSV filename/date boundary or timestamp timezone:

- https://sunstoreAPI.cflsmartroads.com:8081/trm/api/datasets/5566e980-dda3-44db-8aa9-d9736545aebe
- https://sunstoreAPI.cflsmartroads.com:8081/trm/api/datasets/5566e980-dda3-44db-8aa9-d9736545aebe/fields

The default contract therefore preserves naive timestamps and records
`contract_status=unconfirmed`. It checks internal and adjacent coverage without
asserting midnight-to-midnight or 20:00-to-20:00 semantics. Observed 20:00
boundaries are evidence about the files, not a verified permanent rule.

After obtaining an authoritative specification, pass `--contract contract.json`.
For example, a source **explicitly specified** as UTC timestamps grouped by
Eastern calendar day could use:

```json
{
  "timestamp_timezone": "UTC",
  "day_timezone": "America/New_York",
  "day_start": "00:00",
  "date_offset_days": 0,
  "evidence": "Reference to the confirmed source specification",
  "gap_seconds": 3600
}
```

This example is not the default SunStore contract. Calendar boundaries are
localized independently, so DST days span 23 or 25 hours. Ambiguous/nonexistent
naive local timestamps are invalid rather than guessed. Unconfirmed contracts
leave `expected_coverage_verified=false` even if no observed gap is found.

`missing_statuses` (or CLI `--http-statuses`) is a JSON mapping of ISO date to
HTTP status. Only confirmed 404 dates after an observed available publication
frontier become `source_pending`; internal missing dates, unknown responses,
and requests without any observed frontier remain `source_incomplete`.

Download integrity and recovery investigation
--------------------------------------------

`download_csv` stages bytes beside the requested target. It persists response
metadata in `<target>.download.json`, including failed HTTP status. Resume
requires a previous strong ETag or Last-Modified and sends If-Range. HTTP 206
must match start/end/total/Content-Length and the stored version. HTTP 200
restarts the staging file; HTTP 416 retries a full request. Invalid resumes are
discarded, incomplete bodies never publish, and an existing final target is
preserved on failure. Previously corrupt operational inputs are retained as
`.corrupt.<timestamp>` instead of deleted. CSV-to-Parquet conversion publishes
atomically after readable/nonempty integrity checks; temporal quality is a
separate manifest gate.

```sh
python scripts/investigate_raw_gaps.py \
  --dates 2026-08-26 2026-09-01 --raw-dir /data2/ATSPM/raw_data \
  --output-dir scratch/raw_quality_20260911 --fetch
```

This command probes each target and adjacent date, downloads whole candidates
to `staging`, validates the CSV contents in batches, and compares source schema,
row counts, exact time range, source/signal hourly counts and neighboring gaps.
It reports `recoverable`, `source_still_incomplete`, or `unavailable`. It never
replaces operational raw data. A candidate must preserve observed counts,
improve coverage, match schema, and close known gaps before it can be proposed
for replacement. Publication and downstream invalidation use the pipeline's
separate exact-job manifest workflow.
