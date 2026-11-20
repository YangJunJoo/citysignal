"""Run checked feature extraction with ``python -m citysignal``."""

from .cli import extract

if __name__ == "__main__":
    raise SystemExit(extract())
