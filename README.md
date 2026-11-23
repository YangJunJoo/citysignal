# CitySignal

CitySignal turns traffic signal event data into traffic measures, such as vehicle
counts, gaps between vehicles, and pedestrian waiting times. Results are saved
as Parquet files.

## Install

Requires Python 3.10 or newer. Run from the repository folder:

```bash
python -m pip install .
```

## Extract data

Provide event files in Parquet format and signal configuration files in CSV format.
Replace the example paths, signal ID, and date with your own:

```bash
citysignal-extract \
  --event-dir /path/to/events \
  --config-dir /path/to/configs \
  --output-dir /path/to/results \
  --signals 1050 \
  --year 2025 --month 9 --day 13
```

For all options, run `citysignal-extract --help`.
Use `citysignal-pipeline --help` for downloading and processing data together.

The default model assumes eight phases. Separate ring/barrier plans are not
required. Input and output checks help identify missing or conflicting data.

## Read results

Use the package reader to restore data types and saved metadata:

```python
from citysignal.storage import read_frame

frame = read_frame("/path/to/result.parquet")
```

## Development

```bash
python -m pip install -e '.[test]' --config-settings editable_mode=strict
python -m pytest tests -q
```

Some tests need local operational data and are skipped when it is unavailable.
GitHub Actions tests Python 3.10, 3.12, and 3.14 and builds the package.

## More information

- [Processing rules and operations](docs/checked_pipeline.md)
- [Handover guide](docs/handover/README.md)
- [Changes](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
