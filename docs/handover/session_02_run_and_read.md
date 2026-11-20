# Running Jobs and Reading Results

[Repository Guide](README.md) · [Commands to Run](lab_guide.md) · [Daily Runs and Fixing Problems](session_03_operations.md)

## Choose What to Run

```mermaid
flowchart TD
    START[Run request] --> SCOPE{Choose jobs}
    SCOPE -->|Calculate measures| FILTER[Date range + Signal list]
    SCOPE -->|Rerun specific jobs| MANIFEST[jobs-file: Signal-date pairs]
    FILTER --> FEATURES[Selected measures + Earlier results they need]
    MANIFEST --> FEATURES
    FEATURES --> PREFLIGHT[dry-run: Check data / Settings / Jobs]
    PREFLIGHT --> CHECK{Check selected jobs and inputs}
    CHECK -->|Issues found| FIX[Find and fix the problem]
    FIX --> PREFLIGHT
    CHECK -->|Ready to run| RUN[Run the same jobs]
    RUN --> REPORT[Read reports and latest job results]
```

| Option | Meaning |
|---|---|
| `--steps all` | Download, check data, split by signal, calculate measures |
| `--steps extract` | Calculate from existing signal files; also check original input data |
| `--start-date` / `--end-date` | Source date range |
| `--signals` | Choose signals to process |
| `--features` | Choose measures; earlier results they need are added automatically |
| `--jobs-file` | List of exact signals and dates, with input checks |
| `--processed-dir` | Folder for results and run history |
| `--summary-file` | Calculation report in JSON format |
| `--dry-run` | Check jobs and settings without calculating measures; still saves reports, check results, and a lock file |

Even if you choose one signal, input checks may read full daily files and files from the day before and after.

## Calculate, Check, and Save

```mermaid
sequenceDiagram
    participant CLI as run_pipeline.py
    participant Q as Check input data
    participant R as checked_runner
    participant E as checked_execution
    participant F as Output files
    participant DB as pipeline_runs.sqlite3
    CLI->>Q: Check selected dates and nearby days
    Q-->>CLI: raw_quality.json
    CLI->>R: Selected jobs + Input check results
    R->>E: Check settings and calculate measures
    E->>E: Calculate in a separate temporary folder
    E->>E: Check columns / Signal ID / Rows / Times / Save and reload
    E->>F: Move each checked file into place with os.replace
    E-->>R: Status and reason for each measure
    R->>DB: Save run history and results
    R-->>CLI: Calculation report
```

## Read the Run, Job, and Measure Status

```mermaid
flowchart TD
    EXIT[Program exit code] --> PIPE[Run reports / Logs]
    PIPE --> SOURCE[Source quality: source_status / raw_quality.json]
    PIPE --> JOB[Job status: signal_id + date_str]
    JOB --> FEATURE[Feature status: status / reason_code]
    FEATURE --> PATH[output_file_path / row_count]
    JOB --> LATEST[latest_attempts: Latest result for each job]
```

| Measure Status | Meaning |
|---|---|
| `success` | Result has usable rows, passed checks, and was saved; valid zero values are allowed |
| `no_data` | Input or result is empty, or required measure values are missing |
| `not_applicable` | A recorded reason confirms that this measure does not apply |
| `blocked` | Cannot run because input data, settings, or an earlier result has a problem |
| `failed` | Calculation, result checks, or saving failed |
| `skipped` | Left out for a recorded reason that applies to this date |

```mermaid
flowchart TD
    F[Required results for the selected job] --> C{All success or<br/>not_applicable with a recorded reason?}
    C -->|Yes| DONE[Calculation job complete]
    C -->|No| INCOMPLETE[Job incomplete]
    DONE --> SOURCE{Input data problems also fixed?}
    SOURCE -->|No| PARTIAL[Overall run may remain partial]
    OLD[An older result file exists] -.-> N[Latest attempt still needs to pass]
    ZERO[Valid zero values] --> SUCCESS[Can be success]
    EMPTY[Empty result] --> ND[no_data / Incomplete]
```

| Exit Code | What to Check |
|---|---|
| `0` | Read the report: work finished or no work was selected |
| `2` | `partial`: Check problems with input data and measures |
| `1` | Program failed; read logs if no report was saved |
| `75` | Another run is using the shared lock |

## Where to Find Reports and Results

```text
<processed-dir>/
├── pipeline_runs.sqlite3
└── fdot_d5/feature_extraction/.../<signal>/<date>.parquet

<parent-of-processed-dir>/operations/
├── quality_cache/
└── runs/<pipeline-run-id>/
    ├── raw_quality.json
    └── pipeline_summary.json

Path specified by --summary-file
└── Extraction summary JSON
```

## When Can an Earlier Result Be Used Again?

```mermaid
flowchart LR
    INPUT[Input file fingerprint] --> MATCH{All saved checks still match?}
    CONFIG[Settings file fingerprint] --> MATCH
    POLICY[Processing and quality-check versions] --> MATCH
    OUTPUT[Result file fingerprint] --> MATCH
    MATCH -->|Yes| REUSE[Use the saved result]
    MATCH -->|No| CALC[Calculate again]
    REUSE --> ND[Reused no_data remains incomplete]
```

The old `processing_log.db` is kept. `imported_unverified` means an old file was recorded but has not passed the current checks.
