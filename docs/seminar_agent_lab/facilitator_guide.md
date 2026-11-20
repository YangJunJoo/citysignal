# Facilitator Guide: AI-Agent ATSPM Lab

## Purpose and teaching boundary

Participants should practice specifying an analytical task, delegating execution, and judging the evidence returned by an agent. They do not need to edit Python or type terminal commands.

The prepared examples support vehicle phase observations linked to the supplied lane inventory. Historical effective dates of that inventory are unconfirmed. Actual ring/barrier plans are unavailable, so the current checked pipeline withholds cycle-based measures. Successful completion means a defensible, limited interpretation—not a complete set of traffic performance measures.

## Session plan

| Segment | Minutes | Facilitation notes |
|---|---:|---|
| Introduction | 10 | Explain phase versus detector channel, source date versus event date, and evidence-based interpretation. |
| Basic extraction | 20 | Have everyone execute one signal-date pair using the existing checked extractor. |
| Visualization and trace | 10 | Require a raw-event trace for one plotted interval. |
| Unresolved-phase case | 10 | Ask participants to test claims about phases above 8. |
| Debrief | 10 | Compare what was generated, what was supported, and what was withheld. |

## Prepare before the seminar

1. Provide an agent with local file and terminal access. A text-only assistant cannot perform this lab.
2. Use the current repository and an environment that can load the project's pandas/NumPy pickle files. This deployment uses the `atspm` environment.
3. Verify read access to the selected per-signal inputs, their corresponding bulk-source evidence, and configuration files.
4. Prepare a valid source-quality report with selected and adjacent-source coverage. Check signatures before the seminar. Avoid having every learner repeat a full bulk-file scan.
5. Assign a separate writable parent directory to each team, with an output directory beneath it. The extractor's lock is in the output directory's parent; putting every team's output directly under one shared parent would create lock contention.
6. Rehearse both prompts through actual extraction, publication, visualization, and summary inspection. Use fresh rehearsal folders.
7. Keep a reference report and plot available for discussion if a participant's agent connection fails. Identify these as reference artifacts, not as that participant's execution.

Do not disable source validation or use the internal `--lock-held` option to avoid contention. Prepare the workspace correctly instead. The lab requires no changes to production data, the production scheduler, or the shared extractor code.

## Deployment-specific setup card

These paths describe the current shared-server deployment. Replace them if you package the lab elsewhere. Give this card to the agent as context; participants do not need to operate the shell.

| Resource | Current location |
|---|---|
| Repository / working directory | `/home/ya195253/YJ/python_scripts/ATSPM` |
| Python interpreter | `/home/ya195253/anaconda3/envs/atspm/bin/python` |
| Per-signal event directory | `/data2/ATSPM/raw_data/per_signal` |
| Lane configuration directory | `/data2/ATSPM/config_data/signal_config_2025/lanes` |
| Basic event file | `/data2/ATSPM/raw_data/per_signal/1065/atspm-2024-5-1.parquet` |
| Problem event file | `/data2/ATSPM/raw_data/per_signal/1155/atspm-2025-9-7.parquet` |
| Candidate prepared quality report | `/data2/ATSPM/operations/runs/nptoi_mapping_20261030/raw_quality.json` |
| Example team parent | `/home/ya195253/YJ/python_scripts/ATSPM/scratch/seminar/group01` |
| Example attempt output | `/home/ya195253/YJ/python_scripts/ATSPM/scratch/seminar/group01/basic01/processed_data` |

The candidate quality report is a starting point, not a promise of current validity. Confirm its rule version, source signatures, and coverage. If stale, regenerate the required report once before the seminar using the normal validation path. Do not edit signatures to force acceptance.

For shared machines, use one extraction worker per participant and limit numerical-library threads in each participant environment to avoid oversubscription. Rehearse at the expected class size.

### Context prompt to configure a team

Replace the team identifier and quality-report path as appropriate, then give the following to the agent:

> This is a supervised ATSPM seminar workspace. Work from `/home/ya195253/YJ/python_scripts/ATSPM` using `/home/ya195253/anaconda3/envs/atspm/bin/python`. Your only writable area for this lab is `scratch/seminar/group01` under that repository; create separate attempt subfolders there.
>
> Read selected inputs under `/data2/ATSPM/raw_data/per_signal`, lane configurations under `/data2/ATSPM/config_data/signal_config_2025/lanes`, and the facilitator-approved quality report at `/data2/ATSPM/operations/runs/nptoi_mapping_20261030/raw_quality.json`. Check that the report is still valid for the selected source files. These shared resources are read-only.
>
> Use the repository's checked CLI, `feature_extractor.process_local_atspm`, with explicit signal/year/month/day selection, an explicit output folder and summary file, the supplied quality report, one worker, and only the requested feature. Select `vehicle_phase_profile` for the main lab. Record the actual command, exit status, result paths, and any limitations. Do not substitute an old PKL for a newly executed extraction.
>
> Do not download, process other signal-date pairs, change shared code or data, change cron, bypass locks or validation, or create inferred controller mappings. If a resource is unavailable, explain the blocker and continue with the evidence that is available.

## Expected observations

The following are reference observations from the existing corrected outputs checked on November 10, 2026. Row counts are comparison aids, not the sole acceptance criterion. Changes to code or inputs require a new rehearsal and updated reference notes.

| Example | Movement-linked phase IDs in saved output | Unmapped observed phase IDs | Saved phase rows |
|---|---|---|---:|
| 1065 / 2024-05-01 | 1–8 | 12 | 3,436 |
| 1155 / 2025-09-07 | 1–8 | 10, 12, 14, 16 | 4,481 |

The basic case also contains an unresolved phase. Use this to explain that a successful phase output can have limited scope. Do not describe the entire example as fully understood.

The saved mapped observations span:

- Basic example: 2024-04-30 20:00:02.400 through 2024-05-01 19:59:54.500.
- Problem example: 2025-09-06 20:00:38.200 through 2025-09-07 19:59:55.500.

These are observed bounds, not proof of uninterrupted coverage or a universal daily source contract. Ask the agent to inspect current metadata and raw evidence. The 09:00–09:10 basic viewing window falls within the saved observed range; check actual observations during rehearsal. Do not apply an offset merely to align event dates with file labels.

## Evidence participants should obtain

- A fresh execution record that selects one requested signal-date pair.
- A vehicle phase output in the team's own folder.
- A mapping-audit JSON and an unmapped-phase diagnostic file when relevant.
- A plot with a labeled timestamp basis, phase rows, and visible unknown time.
- A fully supported green interval, its supporting raw rows, and its movement mapping from the lane configuration.
- A report distinguishing successful phase extraction from historical mapping uncertainty and unavailable cycle interpretation.

The standard result path below the attempt output is:

`fdot_d5/feature_extraction/signal_profile/phase/vehicle_signal/<signal>/<source-date>.parquet`

Diagnostics are written below:

- `diagnostics/mapping_audit/<signal>/<source-date>.json`
- `diagnostics/unmapped_phase/<signal>/<source-date>.parquet`

The metadata value `barrierNo = 0` in these phase outputs is a placeholder for an unassigned barrier. It does not establish a physical “barrier 0.”

## How to explain execution statuses

A `success` feature has a published result. A `partial` overall job can still contain that successful feature because the lane inventory's historical applicability is unverified. In this pipeline, exit code 2 denotes a partial result and requires reading the report; it is not automatically a calculation crash.

A blocked result means the pipeline withheld the calculation for a stated reason. A `no_data` result means no qualifying result was available. Neither should be presented as a measured zero.

For the main task, participants request only `vehicle_phase_profile`, so a cycle result need not be present in the selected-feature report. The mapping audit still explains the lack of verified barrier evidence. If demonstrating a cycle-dependent request, inspect its current explicit blocking reason; never fabricate a successful metric for the lesson.

## Assessment rubric

Score each item 0 (missing/incorrect), 1 (partly supported), or 2 (clear and supported).

| Criterion | Maximum |
|---|---:|
| Correct signal/date scope and fresh execution evidence | 2 |
| Raw-event trace supporting one interval | 2 |
| Configuration evidence for the movement interpretation | 2 |
| Unknown/unmapped information handled explicitly | 2 |
| Accurate limitations and final conclusion | 2 |
| **Total** | **10** |

Do not award correctness simply because a plot looks plausible or a file exists.

## Common issues and responses

| Issue | Response |
|---|---|
| Agent only describes commands | Ask it to execute the prepared checked CLI and return the execution record. |
| Agent presents a reference result as a new run | Require the current summary, attempt output, command, and execution time. |
| Agent launches a full backfill | Interrupt that task and restore the one-pair scope. |
| Source-quality checks rescan large files | Check whether the prepared report was supplied and is valid. Resolve setup before continuing. |
| Lock contention | Check the team's parent-directory isolation and active lab runs; do not bypass the lock. |
| Agent calls every green overlap an error | Ask whether the overlap is within one phase's states or across distinct phases, and what configuration supports the claim. |
| Agent says phase 12 must be invalid or an overlap | Require event-type and configuration evidence. Numbers above 8 do not determine the role. |
| Agent fills a gap with red | Require unknown intervals to remain unknown and compare against raw transitions. |
| Agent treats a lane inventory as a dated controller plan | Require separate evidence for historical applicability and ring/barrier assignment. |

## Closing discussion

Ask each team to explain one result they can defend and one interpretation they withheld. Then ask: “What additional evidence would change your conclusion?”

Useful background: [Indiana Traffic Signal Hi Resolution Data Logger Enumerations](https://www.cflsmartroads.com/projects/smartsignals/ATSPM%20Detector%20Codes.pdf). Use the event definitions to distinguish phase, detector, and overlap parameters; the document does not establish a particular intersection's movement mapping.
