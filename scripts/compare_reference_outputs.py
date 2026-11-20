#!/usr/bin/env python3
"""Compare saved extraction outputs without running extractors or changing data."""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.feature_registry import FEATURES, output_path
from feature_extractor.storage import assert_frames_equal


def compare_frames(reference, candidate):
    """Compare every column, including timestamps, lists and missing values."""
    try:
        assert_frames_equal(reference, candidate)
    except AssertionError as exc:
        return str(exc)
    return ''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--signal', required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--features', nargs='+', choices=sorted(FEATURES), default=list(FEATURES))
    args = parser.parse_args(argv)
    mismatches = 0
    for name in args.features:
        reference = output_path(args.reference_dir, name, args.signal, args.date)
        candidate = output_path(args.candidate_dir, name, args.signal, args.date)
        missing = [str(path) for path in (reference, candidate) if not path.is_file()]
        if missing:
            difference = 'Missing output: ' + ', '.join(missing)
        else:
            difference = compare_frames(read_frame(reference), read_frame(candidate))
        if difference:
            mismatches += 1
            print(f'{name}: MISMATCH\n{difference}')
        else:
            print(f'{name}: MATCH')
    return 1 if mismatches else 0


if __name__ == '__main__':
    raise SystemExit(main())
