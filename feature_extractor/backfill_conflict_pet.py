#!/usr/bin/env python3
"""
backfill_conflict_pet.py — Retroactively extract conflict_pet from existing PET Parquet files.

For each PET Parquet file on disk, counts PET values below each threshold (0.5–5.0s)
and saves the result as a conflict_pet Parquet file.

Usage:
    python backfill_conflict_pet.py                          # Full backfill
    python backfill_conflict_pet.py --signal 1245            # Single signal
    python backfill_conflict_pet.py --dry-run                # Preview only
    python backfill_conflict_pet.py --workers 32             # Parallel workers
    python backfill_conflict_pet.py --force                  # Overwrite existing
"""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame


import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

warnings.filterwarnings('ignore')

# ── Default paths ──
DEFAULT_BASE = "/data2/ATSPM/processed_data/fdot_d5/feature_extraction"
DEFAULT_PET_DIR = os.path.join(DEFAULT_BASE, "feature", "cycle", "vehicle_traffic", "pet")
DEFAULT_OUT_DIR = os.path.join(DEFAULT_BASE, "feature", "cycle", "vehicle_traffic", "conflict_pet")

THRESHOLDS = np.arange(0.5, 5.5, 0.5)


def _load_frame(path):
    return read_frame(path)


def process_one(args):
    """Process a single PET pkl → conflict_pet pkl."""
    pet_path, out_path = args
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df_pet = _load_frame(pet_path)

        if df_pet is None or len(df_pet) == 0:
            write_frame(pd.DataFrame(), out_path)
            return "empty"

        pet_cols = [c for c in df_pet.columns if 'PET' in c and df_pet[c].dtype == 'O']
        meta_cols = [c for c in ["signalID", "cycleNo", "cycleBegin", "cycleEnd", "cycleLength", "date"]
                     if c in df_pet.columns]

        if not pet_cols:
            write_frame(df_pet[meta_cols], out_path)
            return "no_pet_cols"

        # Build all conflict series, then concat in one shot
        series_list = []
        for col in pet_cols:
            for t in THRESHOLDS:
                conflict_col = col.replace('PET', f'ConflictPET{t}')

                def _count(vals, thresh=t):
                    if not isinstance(vals, list):
                        return 0
                    flat = []
                    for v in vals:
                        if isinstance(v, list):
                            flat.extend(v)
                        else:
                            flat.append(v)
                    return sum(1 for v in flat
                               if isinstance(v, (int, float)) and not np.isnan(v) and 0 < v < thresh)

                series_list.append(df_pet[col].apply(_count).rename(conflict_col))

        result = pd.concat([df_pet[meta_cols]] + series_list, axis=1)
        write_frame(result, out_path)
        return "ok"

    except Exception as e:
        return f"error:{e}"


def collect_tasks(pet_dir, out_dir, signal=None, force=False):
    """Scan PET directory and build list of (pet_path, out_path) tasks."""
    tasks = []
    if not os.path.exists(pet_dir):
        return tasks

    signals = [signal] if signal else sorted(os.listdir(pet_dir))
    for sig in signals:
        sig_dir = os.path.join(pet_dir, sig)
        if not os.path.isdir(sig_dir):
            continue
        out_sig_dir = os.path.join(out_dir, sig)
        for f in os.listdir(sig_dir):
            if not f.endswith('.parquet'):
                continue
            pet_path = os.path.join(sig_dir, f)
            out_path = os.path.join(out_sig_dir, f)
            if not force and os.path.exists(out_path):
                continue
            tasks.append((pet_path, out_path))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Backfill conflict_pet from existing PET Parquet files")
    parser.add_argument("--pet-dir", default=DEFAULT_PET_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--signal", default=None, help="Single signal ID")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"BACKFILL: conflict_pet from PET data")
    print(f"{'='*60}")
    print(f"  PET dir:  {args.pet_dir}")
    print(f"  Out dir:  {args.out_dir}")
    print(f"  Signal:   {args.signal or 'ALL'}")
    print(f"  Workers:  {args.workers}")
    print(f"  Force:    {args.force}")

    tasks = collect_tasks(args.pet_dir, args.out_dir, args.signal, args.force)
    print(f"  Tasks:    {len(tasks):,}")

    if not tasks:
        print("  Nothing to do!")
        return

    if args.dry_run:
        for p, o in tasks[:5]:
            sig = os.path.basename(os.path.dirname(p))
            print(f"    {sig}/{os.path.basename(p)}")
        if len(tasks) > 5:
            print(f"    ... and {len(tasks)-5:,} more")
        return

    t_start = time.time()
    counts = {"ok": 0, "empty": 0, "no_pet_cols": 0, "error": 0}
    done = 0

    with Pool(processes=args.workers) as pool:
        for status in pool.imap_unordered(process_one, tasks, chunksize=64):
            if status.startswith("error:"):
                counts["error"] += 1
            else:
                counts[status] = counts.get(status, 0) + 1
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done:,}/{len(tasks):,}] {rate:.0f} files/s, ETA {eta/60:.0f}m")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"  Total:    {len(tasks):,}")
    print(f"  Success:  {counts['ok']:,}")
    print(f"  Empty:    {counts['empty']:,}")
    print(f"  No cols:  {counts['no_pet_cols']:,}")
    print(f"  Errors:   {counts['error']:,}")
    print(f"  Time:     {elapsed:.0f}s ({len(tasks)/max(elapsed,1):.0f} files/s)")


if __name__ == "__main__":
    main()
