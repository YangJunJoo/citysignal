"""Public API for the extractor's checked Parquet DataFrames."""

from feature_extractor.storage import (
    CONTRACT_KEY,
    SCHEMA_VERSION,
    assert_frames_equal,
    read_frame,
    write_frame,
)

__all__ = [
    "CONTRACT_KEY", "SCHEMA_VERSION", "assert_frames_equal", "read_frame", "write_frame",
]
