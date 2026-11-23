# CitySignal

CitySignal reads signal events and intersection configurations to calculate vehicle
and pedestrian phase/cycle profiles and traffic metrics. The shared extraction
implementation lives in `feature_extractor/`, without separate v1/v2 directories.

The distribution and public Python import are named `citysignal`.
`src/citysignal/` exposes the public API and delegates to the shared implementation.

```text
src/citysignal/          # Public Python API and citysignal commands
feature_extractor/
├── extractors/          # SPaT, volume, headway, gap, occupancy, etc.
├── profiles.py          # Vehicle and pedestrian phase/cycle profiles
├── data_loader.py       # Cached inputs for each signal and date
├── pipeline.py          # Shared feature computation and dependencies
├── storage.py           # Checked Parquet I/O and nested-value restoration
├── checked_execution.py # Validation, staging, and atomic publication
├── checked_runner.py    # Job discovery, parallel execution, and run history
└── feature_registry.py  # Feature names, dependencies, and output paths
data_downloader/         # Downloads, splitting, source quality, and configurations
tests/                  # Regression/integration tests and reference data
docs/                   # Processing rules, operations, and handover guides
scripts/                # Operational and analysis commands
run_pipeline.py         # Download → quality checks → split → extract
```

## Installation and execution

Install into your Python environment from the project root:

```bash
python -m pip install -e '.[test]' --config-settings editable_mode=strict
python -m citysignal --help
citysignal-pipeline --help
python -m pytest tests -q
```

The installation provides `citysignal`, `citysignal-extract`, and
`citysignal-pipeline`. The previous `atspm-extract` and `atspm-pipeline` commands
remain available.

Use strict editable installation when the retired `citysignal/` virtual environment
is present: its directory name otherwise conflicts with the public import.
Keep the generated `build/` directory while using this installation.
For a regular installation, run `python -m pip install .`.

Use the dependencies in `pyproject.toml` for development. The root
`requirements.txt` records the previous operational environment.
The package currently supports pandas 2.x. pandas 3.x changes string/missing-value
dtypes and computation behavior, so it is excluded until output equivalence is
verified. The `.[reports]` extra is needed only to collect NOEMI configuration
reports through a browser; ordinary extraction does not require browser packages.

Operational entry points use `checked_runner` and `checked_execution`. They apply
source-quality checks, configuration validation, the project's eight-phase
assumption, and output validation before publication. Engine selection flags
for v1/v2 have been removed.

## Phase and cycle model

The default model assumes phases 1–8 and rings `[1,2,3,4]` and `[5,6,7,8]`.
It retains the existing barrier numbering:

- Phases `1, 2, 5, 6` belong to barrier `2`.
- Phases `3, 4, 7, 8` belong to barrier `1`.

Separate ring/barrier plans and validity-period records are not required.
The assumption itself, or an unconfirmed lane-configuration validity period,
does not block extraction.

Cycles use observed phases and states; unobserved phases or events are not
fabricated. Observed phases outside 1–8 remain in the output with barrier `0`
and do not define the default cycle boundaries. Lane/channel relationships come
from the actual configuration. Numeric phase assumptions do not create detector
mappings. Pedestrian events retain their `eventParam` phase identity.

Parquet attributes and mapping audits record
`mapping_basis=assumed_eight_phase`, `barrier_mapping_assumed=true`, and
`barrier_mapping_verified=false`. An explicitly supplied `controller_mapping`
is validated separately, including its validity period. Missing observed states,
conflicting cycle windows, raw-source gaps, and configuration errors remain
subject to quality checks.

## Data and validation

Operational inputs, results, and run history default to `/data2/ATSPM`; command
options can override these locations. Neither `citysignal/Lib/site-packages` nor
a local wheel is used as a source dependency. The retired `citysignal/` virtual
environment is preserved and excluded from Git.

Extracted outputs, internal staging data, and test reference data use **Parquet**.
Files use PyArrow's Parquet 2.6 format with ZSTD compression and retain the
signal/date directory layout. The writer determines column schemas, reads back
and validates a staged file, then publishes it atomically.

Read and write extracted profiles and features through the shared API:

```python
from citysignal.storage import read_frame, write_frame

frame = read_frame("/path/to/2025-09-13.parquet")
write_frame(frame, "/path/to/copy.parquet")
```

The `atspm.dataframe` footer contract preserves original dtypes, indexes, nested
containers, missing-value distinctions, and `DataFrame.attrs`. Time intervals
are stored as Arrow `list<struct<begin, end>>` and restored as
`list[tuple[Timestamp, Timestamp]]` for the existing computation code.
Nanosecond precision and local wall times are retained without arbitrary timezone
conversion. `read_frame` rejects missing or unsupported contracts. External raw
event Parquet files are read through the raw-data loader.

Historical bundles of multiple DataFrames and cache snapshots use Parquet archives
read with `citysignal.archive_storage.read_artifact`. DataFrames use actual Arrow
list/struct columns. Retired cache classes are restored as inert
`ArchivedObjectState` values containing a class name and attributes, without
executing the retired class code.

Use `read_frame` when reusing extracted results for computation:
`pd.read_parquet` alone does not restore all required attributes and Python
container types.

The one-time PKL migration uses
`scripts/migrate_pickles_to_parquet.py` in `plan → convert → delete` stages.
Deletion requires successful conversion and exact validation of every planned
file, followed by another SHA-256 check of both source and target. Failures,
collisions, or changed inputs preserve the source. The migration plan and journal
are retained under `scratch/parquet_migration_20261119/`.
Historical ledger rows remain intact; migration provenance and the journal resolve
their new output paths. Keep the journal for validation evidence and historical
result reuse.

Tests with small synthetic inputs run without operational data. Some regression
tests in `tests/golden_data` also require retained raw inputs, configurations, and
profiles; they are skipped when these inputs are unavailable. Review skip counts
alongside test results.

Compare saved reference and candidate results with the read-only tool:

```bash
python scripts/compare_reference_outputs.py \
  --reference-dir /path/to/reference \
  --candidate-dir /path/to/candidate \
  --signal 1050 \
  --date 2025-09-13 \
  --features spat volume
```

This command neither runs extraction nor overwrites result files. Original
comparison programs are preserved under `docs/archive/extraction_comparisons/`;
their former entry points now delegate to this read-only comparison tool.

New run history uses the engine identifier `vectorized`. Previous history is
preserved, and outputs recorded under an earlier engine identifier are not
automatically reused by a new run.

See the [handover guide](docs/handover/README.md) and
[checked pipeline operations](docs/checked_pipeline.md) for operational details.

## GitHub version control

The repository is `YangJunJoo/citysignal`. Git tracks code, documentation, and
Parquet reference data in `tests/golden_data/`. Operational data, generated
results, virtual environments, logs, and credentials are excluded by `.gitignore`.

On pushes and pull requests, GitHub Actions tests Python 3.10, 3.12, and 3.14,
builds a wheel and source distribution, and verifies installed-wheel imports and
commands in a separate environment. Tests requiring operational inputs may be
skipped on GitHub; real-output equivalence must also be checked in an environment
with those inputs. Build files and test results are available as workflow
artifacts. The workflow does not run operational extraction, create GitHub
Releases, or publish to PyPI.

The single version source is `__version__` in `src/citysignal/__init__.py`.
Tag verified commits with versions such as `v0.1.1` and record later changes in
separate commits. Repository documentation, comments, interface text, and commit
messages are written in English; see [Contributing](CONTRIBUTING.md).

## Priority extraction and monitoring

`scripts/run_extraction_campaign.py` processes the designated NPTOI priority dates,
then processes the remaining retained source files from newest to oldest.
It distinguishes the 72 requested local dates from four additional source days
that cover date boundaries.

A new campaign starts with an empty output directory. When relocating to the
operational path, preserve previous results in a backup and move generated results
and their ledger together. The current operational output is
`/data2/ATSPM/processed_data`.

Restarting with the same campaign directory skips dates already attempted.
`partial` and `failed` dates remain unresolved while the queue continues; they
are not counted as complete. The campaign holds the shared `.pipeline.lock`
throughout execution to prevent overlap with scheduled jobs.

```bash
python scripts/run_extraction_campaign.py \
  --campaign-dir /data2/ATSPM/operations/runs/nptoi_priority_20261120 \
  --processed-dir /data2/ATSPM/processed_data \
  --workers 16

python dashboard/campaign_server.py \
  --campaign-dir /data2/ATSPM/operations/runs/nptoi_priority_20261120 \
  --host 127.0.0.1 \
  --port 8050
```

The read-only dashboard at `http://localhost:8050/` refreshes every five seconds
with date progress, signal jobs, feature success/blocked counts, and live logs.
Use SSH or IDE port forwarding when accessing a remote server. Campaign state and
daily logs reside in the campaign directory; results and the execution ledger
reside in the output directory.
