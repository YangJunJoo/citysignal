# How Data Becomes Traffic Measures

[Repository Guide](README.md) · [Run Jobs and Read Results](session_02_run_and_read.md)

## From Input Files to Saved Results

```mermaid
flowchart TD
    CSV[Daily CSV from the data source] --> PAR[Daily Parquet]
    PAR --> Q{Check input data}
    ADJ[Files from the day before and after] --> Q
    Q -->|Accepted| SPLIT[Split by signal]
    Q -->|Missing / Incomplete| QR[Record problems / Stop affected jobs]
    SPLIT --> EV[Events for one signal]
    CONF[Signal settings CSV] --> CV{Check settings}
    CV -->|Valid| PROFILE[Phase and cycle timing tables]
    CV -->|Invalid / repair_pending| BLOCK[Stop calculation]
    EV --> PROFILE
    PROFILE --> FEATURE[Calculate selected measures]
    FEATURE --> VALID[Check columns / Times / Save and reload]
    VALID --> SAVE[Save checked Parquet files]
    SAVE --> LEDGER[Record each attempt and measure status]
    BLOCK --> LEDGER
```

## Match Events with Signal Settings

```mermaid
flowchart LR
    subgraph Events[Events]
        ID[signalID]
        TS[timeStamp]
        CODE[eventCode]
        PARAM[eventParam]
    end
    subgraph Config[Configuration]
        CH[Channel / channelNo]
        PH[Phase / phaseNo]
        LANE[laneType / stopBarDistance]
        BAR[barrierNo]
    end
    CODE --> TYPE[Find what eventParam refers to]
    PARAM --> TYPE
    TYPE -->|Detector event| CH
    TYPE -->|Phase event| PH
    CH --> PH
    PH --> PROFILE[Build phase and cycle timing tables]
    TS --> PROFILE
    ID --> PROFILE
    LANE --> FEATURE[Calculate traffic measures]
    BAR --> PROFILE
    PROFILE --> FEATURE
```

Read `eventCode` to know what `eventParam` means. If the file date and event date differ, that alone does not tell you the timezone or when the source's day starts.

## Which Timing Tables Are Built First?

```mermaid
flowchart LR
    VP[vehicle_phase_profile] --> VC[vehicle_cycle_profile]
    PP[pedestrian_phase_profile] --> PC[pedestrian_cycle_profile]
    VC --> PC
    VC --> V[Vehicle and signal measures]
    PC --> P[Pedestrian measures]
```

## What Each Measure Needs

An arrow means “this result is needed by the next calculation.” [feature_registry.py](../../feature_extractor/feature_registry.py) lists the exact names, required earlier results, and save locations.

```mermaid
flowchart TD
    VC[vehicle_cycle_profile] --> SP[spat]
    VC --> VO[volume]
    VC --> HW[headway]
    VC --> GA[gap]
    VC --> PET[pet]
    VC --> OC[occupancy]
    VC --> RR[red_running]
    VC --> DR[dilemma_running]
    VC --> SW[shockwave]
    VC --> SF[split_failure]
    SP --> SF
    OC --> SF
    VC --> CO[conflict]
    HW --> CO
    VC --> CP[conflict_pet]
    PET --> CP
    PC[pedestrian_cycle_profile] --> PA[pedestrian_activity]
    PC --> PD[pedestrian_delay]
    VC --> OPTIONAL[conflict_propensity / Optional]
    VO --> OPTIONAL
    PA --> OPTIONAL
```

| Selection                            | What Actually Runs                                             |
| ------------------------------------ | -------------------------------------------------------------- |
| Default                              | 4 timing tables + 14 traffic measures = 18 required results    |
| `--features spat volume occupancy` | Vehicle phase and cycle tables + selected measures = 5 results |
| `--features conflict_propensity`   | Optional measure and every earlier result it needs             |
| `conflict_headway`                 | Another name for`conflict`                                   |

## File Paths

```text
/data2/ATSPM/
├── raw_data/
│   ├── atspm-2026-9-6.parquet
│   └── per_signal/ORC-0003/atspm-2026-9-6.parquet
├── config_data/signal_config_2025/lanes/
│   └── ORC-0003.csv
└── processed_data/
    ├── pipeline_runs.sqlite3
    └── fdot_d5/feature_extraction/
        ├── signal_profile/
        │   ├── phase/{vehicle_signal,pedestrian_signal}/
        │   └── cycle/{vehicle_signal,pedestrian_signal}/
        └── feature/cycle/
            ├── vehicle_signal/spat/
            ├── vehicle_traffic/{volume,occupancy,...}/
            └── pedestrian_traffic/{activity,delay,conflict_propensity}/

Under each output directory: <signal_id>/<YYYY-MM-DD>.parquet
```

## How to Read a Result Row

```mermaid
flowchart LR
    ID[signalID] --> ROW[One row for one signal cycle]
    NO[cycleNo] --> ROW
    BEGIN[cycleBegin] --> ROW
    END[cycleEnd] --> ROW
    LENGTH[cycleLength] --> ROW
    ROW --> SP[SPaT: Time spent in each signal phase]
    ROW --> VO[Volume: Number of vehicles]
    ROW --> OC[Occupancy: How much time a detector is occupied]
```

Open the saved table to see all measure columns. Some cells may hold lists or other grouped values. A short preview only shows part of the table.
