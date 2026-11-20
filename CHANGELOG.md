# Changelog

## 0.1.1

- Use the existing eight-phase, two-ring model by default, without requiring controller plans or effective dates.
- Record assumed mappings in Parquet metadata and mapping diagnostics without labelling them as verified.
- Run vehicle and pedestrian extraction through the shared implementation under this default model.
- Preserve auxiliary observed phases without using them to define the eight-phase cycle boundaries.
- Retain configuration, raw-source and observed-state quality checks; quarantine conflicting cycle windows.
- Invalidate reuse of the previous controller-evidence-only execution policy.

## 0.1.0

- Package the checked extractor and Parquet storage APIs as CitySignal.
- Add GitHub Actions tests, distribution builds and installed-wheel verification.
