# Checked pipeline operations

`scripts/run_daily_pipeline.sh` invokes `run_pipeline.py` for download → raw-source
quality checks → splitting → feature extraction. CLI, backlog, dashboard, and
direct-extraction entry points share the same completion rules. There are no
file-size thresholds or chattering-based exclusion gates.

## Configuration validation

```bash
python scripts/audit_signal_configs.py --help
```

Validation checks whitespace, headers, column structure, phase/channel values,
and barrier relationships. Known column names and numeric representations are
normalized in memory. Lane and detector identities are not invented. Valid
one-to-many channel relationships are retained with warnings for review.

Configuration actions are `use`, `repair_pending`, and `exclude`. Explicit
exclusions require a reason, source, and applicable date range. Repair-pending
or unusable configurations are blocked before extraction; usable profiles and
detector-specific capabilities are evaluated separately. Configuration SHA-256
and validation-rule versions are included in reuse decisions.

## Outputs and completion

The default model assumes phases 1–8 and rings `[1,2,3,4]` and `[5,6,7,8]`.
The existing barrier numbering is `{1,2,5,6}: 2` and `{3,4,7,8}: 1`.
Controller plans, reviewer records, and validity periods are not mandatory inputs
for this mode. An unconfirmed lane-configuration validity period does not block
the assumed model. Each output and audit records
`mapping_basis=assumed_eight_phase`, `barrier_mapping_assumed=true`, and
`barrier_mapping_verified=false`.

Observed phases are preserved. Phases outside 1–8 use barrier `0`, so they do not
define the default cycle boundaries. Observed phase states are retained even if
the phase is absent from the lane inventory. Lane/detector features use the actual
configuration; channel numbers and special phase identifiers are not guessed.
Pedestrian events retain their existing `eventParam` phase association.
Under the default assumption, conflicting green intervals quarantine affected
cycles. Explicit `controller_mapping` records are still checked for identity,
validity period, and source-content hashes.

`feature_extractor/feature_registry.py` defines names, paths, and dependencies for
four profiles and 15 features. The default required set contains 18 outputs;
`conflict_propensity` is optional and must be requested explicitly.
`conflict_headway` is an alias for `conflict`. PET is required by default.
Selecting a feature also selects its dependencies.

| Status | Meaning |
|---|---|
| `success` | A valid, nonempty output was checked and saved. Valid zero values count as success. |
| `failed` | Computation, output validation, or publication failed; the cause is retained. |
| `no_data` | Input/output is empty or required feature values are absent. |
| `not_applicable` | Explicit evidence establishes non-applicability, such as no configured detector of the required type. |
| `blocked` | Configuration, source quality, or dependency problems prevent execution. |
| `skipped` | An explicit exclusion applies, with supporting source and dates. |

A job is `complete` only when every required result is `success` or supported
`not_applicable`. A failed dependency blocks downstream features while independent
features continue. Empty data and exclusions do not satisfy overall completion.
The absence of pedestrian functionality is not inferred from lane CSVs alone.

Each job computes in an isolated staging directory. Schema, signal ID, row count,
cycle intervals, observed input bounds, and serialization round trips are checked
before per-file publication with `os.replace`. Existing files cannot turn a failed
new attempt into a successful one.

The legacy `processing_log.db` is preserved. **`pipeline_runs.sqlite3`** records
runs, attempts, and feature results, while `latest_attempts` exposes the latest
attempt per signal/date. Legacy file synchronization records
`imported_unverified`, not synthetic success. Reuse requires matching input,
configuration, policy, source-quality versions, and output signatures. An unchanged
`no_data` outcome can skip repeated computation while remaining incomplete.

The eight-phase default uses execution rule
`checked-extraction-6-eight-phase-default`. Outcomes blocked by
`barrier_mapping_unverified` under the previous rule are reevaluated rather than
reused unchanged.

Exit codes are `0=complete/no_work`, `2=partial`, `1=system failure`, and
`75=another process owns the shared lock`. The dashboard does not count partial
results as complete or automatically retry them as crashes.

## Source quality and recovery

Checks cover minimum/maximum timestamps, hourly totals and per-signal row counts,
invalid timestamps, internal gaps, and boundaries between adjacent files.
Results are cached by file signature and rule version; changes in neighboring
files also invalidate boundary conclusions. HTTP 404 publication delays are
`source_pending`; observed missing or incomplete source data is
`source_incomplete`.

The retained public source specification does not establish timestamp timezones
or source-file day boundaries. Therefore, a window such as 20:00–20:00 is not
treated as an established contract. Outputs retain `contract_status=unconfirmed`
and `expected_coverage_verified=false`. Passing observed-gap checks is separate
from proving complete expected coverage. An evidence-backed contract also enables
checks for 23/25-hour DST days.

Downloads validate response lengths, Content-Range, ETag, and Last-Modified.
Invalid resume responses restart safely; final files are not replaced before
validation. Recovery investigations use separate staging. Remote full CSVs for
2026-08-26 and 2026-09-01 were found to contain the same gaps as the retained raw
files, so the operational originals were preserved.

## Reprocess affected jobs

```bash
python scripts/build_reprocess_manifest.py \
  --audit-dir scratch/pipeline_audit_20260909 \
  --event-dir /data2/ATSPM/raw_data/per_signal \
  --config-dir /data2/ATSPM/config_data/signal_config_2025/lanes \
  --quality-report scratch/raw_quality_20260911/final_quality.json \
  --output scratch/reprocess_20260911/jobs.json

python run_pipeline.py \
  --steps extract \
  --jobs-file scratch/reprocess_20260911/jobs.json \
  --quality-report scratch/raw_quality_20260911/final_quality.json \
  --max-workers 12 \
  --summary-file scratch/reprocess_20260911/execution_summary.json
```

Building a manifest does not execute extraction. It records exact
`(signal_id, date_str)` pairs, requested features and dependencies, input signatures
and SHA-256, configuration SHA-256, selection evidence, and blocking reasons.
Signal lists and date lists are not expanded into a Cartesian product. Changed
inputs block stale manifests until reviewed. Use `--dry-run --summary-file PATH`
to save extraction-planning checks.

Known configuration problems and unrecovered sources remain in the manifest with
their blocking status. Create a new manifest for the affected signal/date after
obtaining a supported correction. The historical 2,015 jobs with missing
configurations should not be retried indiscriminately while those configurations
remain unavailable.

## Validation coverage

Regression tests cover configuration validation, result states and dependencies,
atomic-publication failures, exact manifests, HTTP resume behavior, gaps, DST,
invalid dates, dashboards, and legacy synchronization. Real-data checks include
all 18 required outputs for a valid signal and `no_data → blocked` propagation
when no cycles are available.

Some golden tests require retained raw inputs, configurations, and profiles for
1050/2025-09-13 and are skipped when those inputs are unavailable. Neither arbitrary
timezone changes nor reference-value edits are used to force comparisons to pass.
Fixtures write to temporary directories rather than operational output paths.
