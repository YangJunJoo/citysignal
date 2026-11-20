# Extract and Validate Traffic Signal Events with an AI Agent

**Duration:** 60 minutes  
**Format:** Individual work or pairs; no coding required  
**Tools:** A file-and-terminal-enabled AI agent in a prepared ATSPM workspace

## Your task

Ask an AI agent to extract data for one intersection and one source date. Then require the agent to show how the result follows from the raw events and configuration.

By the end, you should be able to:

- Specify an extraction request with a clear signal ID, date, and output location.
- Check that the agent actually executed the extractor.
- Trace a reported signal-state interval to its raw events.
- Distinguish interpretable phase observations from unresolved observations.
- Explain why some requested results must be withheld.

A convincing answer contains evidence: file paths, event rows, configuration records, and the current execution report.

## Schedule

| Time | Activity |
|---|---|
| 0–10 min | Introduction, workspace, and input fields |
| 10–30 min | Task 1: Extract the basic example |
| 30–40 min | Task 2: Visualize and verify one interval |
| 40–50 min | Task 3: Investigate an unresolved-phase example |
| 50–60 min | Complete the worksheet and compare findings |

## Before you start

Your facilitator will provide:

- A prepared agent workspace and ATSPM execution environment.
- Read access to example event files and configuration files.
- A current, prepared source-quality report.
- Your assigned writable output folder.

Use only your assigned folder for reports and results. Participants can inspect the code, but they do not need to write or run commands themselves.

## Four fields to understand

| Field | Meaning |
|---|---|
| `signalID` | The intersection/controller identifier used by this dataset |
| `timeStamp` | The recorded event timestamp; its time basis must be checked |
| `eventCode` | What kind of event occurred |
| `eventParam` | The identifier associated with that event type |

**Read `eventCode` before interpreting `eventParam`.** For example, code 1 reports the start of a phase's green indication, while codes 81 and 82 report detector OFF and ON. The parameter identifies a phase in the first case and a detector channel in the second. A phase number and a detector channel number are different identifiers.

A phase number alone does not establish its direction, lane movement, or barrier assignment. Those interpretations require supporting configuration.

A source file date is also distinct from an event's calendar date. The example files contain observations spanning portions of two local calendar dates. Do not change timestamps simply to make them match the filename.

## Shared instructions for your agent

Paste the following instructions first. Your facilitator should already have supplied the workspace paths.

> You are helping me complete an ATSPM seminar lab. Use the prepared environment, source-quality report, and current checked extractor supplied by the facilitator.
>
> Write only inside my assigned lab folder. Treat shared event files, configurations, production outputs, and scheduled jobs as read-only. Process only the signal-date pair I request. Use one extraction worker. Do not download data or expand to other signals or dates. If a required resource or valid quality report is missing, report the blocker instead of bypassing validation.
>
> Execute the extractor to generate my result. Existing results may be used for comparison, but do not present them as a newly executed extraction. Keep an execution record with the command, exit status, summary path, and output paths. Give each attempt its own output folder.
>
> Distinguish observed facts, configuration-based interpretations, and unresolved information. Preserve unknown state intervals and unmapped phases as such. Do not infer a controller ring/barrier plan from phase numbers or from the absence of overlap.
>
> Start the authorized lab tasks without requesting routine confirmation. If a task would require changing shared resources or exceeding the stated scope, report that limitation and continue the parts within scope.

## Task 1 — Extract a specific intersection and source date

Use **signal 1065, source date 2024-05-01**.

### Prompt

> Extract vehicle phase-state data for signal 1065 from source date 2024-05-01 using the current checked ATSPM extractor.
>
> Before executing, inspect the selected input and lane configuration. Tell me which files you will use and what phase identities the configuration supports. Run only `vehicle_phase_profile` for this one signal-date pair and save a fresh result in my assigned folder.
>
> After execution, report the selected scope, command and exit status, feature-level status, overall job status, row count, and output paths. Identify any observed phases that could not be mapped to a movement. Explain the limits of the lane configuration's historical applicability.

### Check the response

- Does the execution record select exactly one signal-date pair?
- Did the agent request only the vehicle phase profile?
- Is the output in your own folder?
- Does the agent separate a successful phase feature from an overall `partial` job?
- Does it identify observations outside the configured movement mapping?

**Expected behavior:** A phase feature can succeed while the overall job remains `partial`. The existing lane inventory does not confirm its applicability on every historical date. Read the report before interpreting an exit code of 2 as a software failure.

## Task 2 — Visualize and verify

Use **09:00–09:10 on 2024-05-01 in the event timestamps' declared time basis**. Confirm that this window is present before plotting.

### Prompt

> Open the result you just generated. Check its timestamp metadata and available time range, and show a phase-state timeline for 2024-05-01 from 09:00 to 09:10. Label the time basis and use separate rows for phases. Show unresolved time as unknown rather than filling it with red or green. If the window is unavailable, explain why and select another observed ten-minute window.
>
> Choose one fully supported green interval. Show its exact start and end, the raw event rows that support those boundaries, and the configuration records connecting that phase to an approach and lane movement.
>
> Check for overlapping states within each phase in the displayed window. Explain what that check establishes and what it does not establish. Report which calendar dates occur in the source file and how they differ from the filename date.

### Check the response

- Are the plotted intervals from your newly generated result?
- Can you match one interval's boundaries to actual raw event rows?
- Does the approach/movement interpretation cite configuration records?
- Are missing or unconfirmed intervals visibly unknown?
- Does the agent avoid treating simultaneous green on different phases as an error by itself?

A valid green-end observation may be code 7 or code 8, depending on the recorded sequence. Ask the agent to explain the actual sequence it used. Code 7 does not, by itself, identify every subsequent indication.

## Task 3 — Investigate a problematic example

Use **signal 1155, source date 2025-09-07**.

### Prompt

> Extract `vehicle_phase_profile` for signal 1155 from source date 2025-09-07 into a separate attempt folder using the same checked procedure.
>
> Investigate observed phases 10, 12, 14, and 16. Show whether each phase appears in the raw phase events and whether the supplied lane or signal inventory establishes its movement. Use raw rows and configuration evidence in your explanation. Do not assume that a phase above 8 is invalid, an overlap, or an auxiliary phase.
>
> Show where interpretable phase results and unmapped diagnostic observations were saved. Explain whether the available evidence supports a controller ring/barrier assignment or cycle-based measures. Acknowledge what remains unknown instead of inventing a mapping.

### Check the response

The agent should preserve unresolved observations in diagnostics and explain why they are excluded from movement-level interpretation. A diagnostic record is evidence that something was observed; it is not an approved movement or cycle measure.

No verified controller ring/barrier plan is supplied for these examples. The current checked pipeline withholds cycle-based results. Missing evidence is an acceptable finding.

## Task 4 — Submit your findings

### Prompt

> Write a short report in my assigned folder covering both examples. Include the requested and executed scopes, extraction evidence, generated artifact paths, one verified raw-event trace, the unresolved phase identities, and the limits on interpretation. Separate successfully generated phase observations from withheld cycle-based measures. Do not claim the entire historical dataset or controller operation has been validated.

Complete the [participant worksheet](participant_worksheet.md), then present:

1. What did you extract?
2. What evidence supports one result?
3. What could you not interpret, and how did you handle it?

## Questions worth asking the agent

> Which file or record supports that statement?

> Did you calculate this result now, or read a previously generated file?

> Are you describing the source file date or the event timestamp date?

> What evidence identifies this phase's movement?

> What information would be needed to interpret the withheld result?

## Completion checklist

- [ ] I have an execution record and a newly generated phase output.
- [ ] I have a labeled timeline and one raw-event trace.
- [ ] I can explain a phase-to-movement mapping using configuration.
- [ ] I can explain the difference between a phase and a detector channel.
- [ ] I recorded unresolved observations and withheld measures.
- [ ] I stated the source-date, timestamp, and historical-configuration limitations.
