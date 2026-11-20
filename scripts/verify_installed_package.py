"""Verify a built CitySignal installation outside the source checkout.

Run with the installed environment's Python and ``-I`` so source files and
user-site packages cannot hide an incomplete wheel.
"""

from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import citysignal
from citysignal.archive_storage import read_artifact, write_artifact
from citysignal.storage import assert_frames_equal, read_frame, write_frame


def main():
    installed = distribution("citysignal")
    assert installed.version == citysignal.__version__
    files = {str(path) for path in installed.files}
    assert "citysignal/__init__.py" in files
    assert "feature_extractor/storage.py" in files
    assert "data_downloader/__init__.py" in files
    assert "run_pipeline.py" in files
    entry_points = {entry.name for entry in installed.entry_points
                    if entry.group == "console_scripts"}
    assert {"citysignal", "citysignal-extract", "citysignal-pipeline"} <= entry_points
    assert "/citysignal/Lib/" not in str(Path(citysignal.__file__).resolve())

    start = pd.Timestamp("2025-09-13T12:00:00.000000001")
    original = pd.DataFrame({
        "channels": [[1, 2], []],
        "headways": [[[float("nan"), 2.5], [3.0]], []],
        "intervals": [[(start, start + pd.Timedelta(1, unit="ns"))], []],
    })
    original.attrs["interval_closure"] = "left"
    with TemporaryDirectory(prefix="citysignal-install-check-") as directory:
        frame_path = Path(directory) / "frame.parquet"
        write_frame(original, frame_path)
        assert_frames_equal(original, read_frame(frame_path))
        archive_path = Path(directory) / "archive.parquet"
        write_artifact({"reference": original}, archive_path)
        assert_frames_equal(original, read_artifact(archive_path)["reference"])
    print(f"CitySignal {installed.version}: installed imports and Parquet APIs verified")


if __name__ == "__main__":
    main()
