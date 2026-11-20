# Daily Runs and Fixing Problems

[Repository Guide](README.md) · [Run Jobs and Read Results](session_02_run_and_read.md) · [Commands to Run](lab_guide.md)

## Automatic Daily Runs

```mermaid
flowchart TD
    S[Scheduled task on the server] --> W[scripts/run_daily_pipeline.sh]
    W --> LOCK{Get the lock to prevent overlapping runs}
    LOCK -->|Failure| EXIT[Exit 75]
    LOCK -->|Success| DATE[Check dates]
    DATE -->|Invalid range| BAD[Exit 64]
    DATE -->|Valid| P[run_pipeline.py --steps all]
    P --> LOG[operations/logs/daily_pipeline_*.log]
    P --> REPORT[Quality / Pipeline / Feature reports]
    P --> DB[Latest job results and measure status]
```

| Daily Script Default | Value |
|---|---|
| Date range | Last 14 days, through yesterday |
| Extraction workers | 24 |
| Split workers | 4 |
| Maximum date range | 60 days |
| Path | `/data2/ATSPM` |

Check the schedule and settings on the server. Having this script in the project does not mean it is scheduled to run.

## Find Where the Problem Starts

```mermaid
flowchart TD
    START[Incomplete job found] --> RAW{Source quality}
    RAW -->|source_pending| PENDING[Check late uploads / Download responses / Missing dates]
    RAW -->|source_incomplete| GAP[Check missing time periods and incomplete files]
    RAW -->|Accepted| CONFIG{Check settings}
    CONFIG -->|Invalid / repair_pending| MAP[Find supported settings for that date]
    CONFIG -->|Valid| PROFILE{Required timing tables}
    PROFILE -->|no_data| EVENTS[Check event times and how timing tables were built]
    PROFILE -->|success| FEATURE{Calculate and save measures}
    FEATURE -->|failed| ERROR[Check code errors / Table columns / File saving]
    FEATURE -->|blocked| DEP[Find the first required result that is missing]
    DEP --> PROFILE
```

## How One Missing Result Stops Other Calculations

```mermaid
flowchart LR
    VP[vehicle_phase_profile<br/>success] --> VC[vehicle_cycle_profile<br/>no_data: empty_result]
    VC --> V[volume<br/>blocked: dependency_no_data]
    VC --> O[occupancy<br/>blocked: dependency_no_data]
    VC --> S[spat<br/>blocked: dependency_no_data]
```

Start with the empty vehicle cycle table. Volume, occupancy, and SPaT need that table, so rerunning only those measures will not fix the problem.

## What Input Checks Can Tell You

```mermaid
flowchart LR
    RAW[Selected files and files from nearby days] --> OBS[Recorded times / Row counts / Gaps within and between files]
    OBS --> QUALITY[Check the data that is present]
    CONTRACT[Data source rules for timezone and when each day starts] --> COVERAGE[Check that the full expected time period is present]
    UNKNOWN[Source rules not confirmed] --> FLAGS[contract_status: unconfirmed<br/>expected_coverage_verified: false]
```

No gaps in the data you have does not prove that you have the whole expected day.

## Rerun Only the Affected Jobs

```mermaid
flowchart TD
    AUDIT[Recorded problems: Failed / Missing jobs] --> CAUSE[Fix input data / Settings / Calculation problems]
    CAUSE --> BUILD[build_reprocess_manifest.py]
    QUALITY[Latest input check report] --> BUILD
    INPUT[Current inputs / Settings] --> BUILD
    BUILD --> MANIFEST[jobs + blocked_candidates<br/>deferred_missing_configs + summary]
    MANIFEST --> REVIEW[Check jobs / Measures / File fingerprints / Reasons]
    REVIEW --> DRY[--jobs-file + --dry-run]
    DRY --> CHECK{Saved checks still match current inputs?}
    CHECK -->|No| BUILD
    CHECK -->|Yes| RUN[Run the same job list]
    RUN --> VERIFY[Check latest job results / Measure status / Input data / Saved files]
    VERIFY --> RESOLVED{Problems fixed?}
    RESOLVED -->|No| CAUSE
    RESOLVED -->|Yes| DONE[Record rerun results]
```

```mermaid
flowchart LR
    A[Job to rerun A] --> A1[DEMO-A / 2026-09-14]
    B[Job to rerun B] --> B1[DEMO-B / 2026-09-15]
    A1 --> M[Job list: Exactly two signal-date pairs]
    B1 --> M
    LIST[Two signals + Two dates] --> CROSS[Selecting both signals over both days can run four jobs]
```

Rebuild the job list when inputs or settings change. Do not edit file fingerprints to force old checks to pass. The list may have no jobs ready to run if input or settings problems remain.

## Daily Checks and Notes

| Check | Evidence |
|---|---|
| Did the run start, and which jobs did it select? | Server schedule, latest log, dates, number of parallel workers |
| Source status | `raw_quality.json`, missing dates, quality reasons |
| Incomplete jobs | Measure report, `latest_attempts` |
| Where the problem starts | Settings → Timing tables → Required earlier results → Calculation and saving |
| Jobs to rerun | Exact signals and dates, measures, current check results |
| Did the rerun work? | Latest job result, required measures, checked files |

```text
Run ID / Report:
Signal ID / Source-file date / Selected measures:
Source quality / Settings file fingerprint:
Latest status / First missing required result / Reason:
What was fixed and why / Exact jobs to rerun:
Check result / Person responsible / Next check:
```
