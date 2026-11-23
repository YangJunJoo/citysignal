"""CitySignal: checked ATSPM extraction and lossless Parquet storage.

The public package delegates to the shared extraction implementation in
``feature_extractor``. It never loads the retired citysignal virtual environment.
"""

__version__ = "0.1.2"

from feature_extractor import ENGINE_ID
from .storage import read_frame, write_frame

__all__ = ["ENGINE_ID", "__version__", "read_frame", "write_frame"]
