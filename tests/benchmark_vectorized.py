"""Benchmark the production V2 mapper and extractors with isolated outputs.

Input root must contain staged interim/ and production/ profile directories.
Example: python tests/benchmark_vectorized.py --input-root SNAPSHOT --signal 1050 --date 2025-09-13
Cold/warm equality checks cache reuse, not equivalence to V1. Optional reference
outputs must belong to the same input/config/profile snapshot.
"""

from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
import argparse
from datetime import date
import json
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extractor.data_loader import SignalDataCache
from feature_extractor.pipeline import _CalculationCache
from feature_extractor.extractors.volume import extract_volume
from feature_extractor.extractors.headway import extract_headway
from feature_extractor.extractors.red_running import extract_red_running, extract_dilemma_running
from feature_extractor.feature_registry import FEATURES, output_path


def timed(call):
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def benchmark(cache_factory, reference_dir=None):
    cache = cache_factory()
    mapped, cold = timed(lambda: cache.assign_cycle_index((81,), 'back'))
    reused, warm = timed(lambda: cache.assign_cycle_index((81,), 'back'))
    if mapped.empty:
        raise ValueError('Benchmark needs detector events and cycle profiles')
    assert reused is mapped
    # Independent inclusive-window reference; shared boundaries belong to both cycles.
    events = cache.get_merged_events((81,), 'back')
    counts = [int(events.timeStamp.between(row.cycleBegin, row.cycleEnd).sum())
              for row in cache.df_vehicle_cycle.itertuples()]
    actual = mapped.loc[mapped.cycle_idx >= 0].groupby('cycle_idx').size()
    assert actual.reindex(range(len(counts)), fill_value=0).tolist() == counts
    report = {'mapper': {'cold_seconds': cold, 'warm_seconds': warm,
                         'inclusive_cycle_counts_equal': True}}
    for name, extract in [('volume', extract_volume), ('headway', extract_headway),
                          ('red_running', extract_red_running), ('dilemma_running', extract_dilemma_running)]:
        cache = cache_factory()
        calculation = _CalculationCache(cache)
        first, cold = timed(lambda: extract(calculation))
        second, warm = timed(lambda: extract(calculation))
        pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=0, atol=0)
        row = {'cold_seconds': cold, 'warm_seconds': warm, 'rows': len(first),
               'cold_warm_equal': True, 'reference_equal': None}
        if reference_dir is not None:
            reference = output_path(reference_dir, name, cache.signal_id, cache.date_str)
            pd.testing.assert_frame_equal(read_frame(reference), first,
                                          check_exact=False, rtol=0, atol=0)
            row['reference_equal'] = True
        report[name] = row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--signal', required=True)
    parser.add_argument('--date', type=date.fromisoformat, required=True)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args()
    signal, date_str = args.signal, args.date.isoformat()
    # Copy only this signal-day's inputs. Never direct legacy extractor writes at the source.
    with tempfile.TemporaryDirectory(prefix='atspm-extractor-benchmark-') as temporary:
        root = Path(temporary)
        paths = [Path('interim/fdot_d5/event_data') / signal / f'{date_str}.parquet',
                 Path('interim/fdot_d5/signal_config') / f'{signal}.csv']
        paths += [output_path('production', name, signal, date_str)
                  for name, spec in FEATURES.items() if spec.scope == 'profile']
        for relative in paths:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.input_root / relative, destination)
        def cache_factory():
            return SignalDataCache(signal, args.date.year, args.date.month, args.date.day,
                                   str(root), 'interim', 'production')
        report = benchmark(cache_factory, args.reference_dir)
        print(json.dumps({'signal': signal, 'date': date_str,
                          'timing_scope': 'calculation only; input loading and publication excluded',
                          'results': report}, indent=2))


if __name__ == '__main__':
    main()
