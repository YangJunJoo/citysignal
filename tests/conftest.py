"""
conftest.py — Shared pytest fixtures for feature_extractor tests.

Provides fixtures for SignalDataCache and golden (v1) reference data.
"""

from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame

import os
import sys
import shutil
from pathlib import Path
import pytest
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Golden data configuration
GOLDEN_SIGNAL = '1050'
GOLDEN_DATE = (2025, 9, 13)
GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden_data', '1050_2025-09-13')
DATA_ROOT = '/data2/ATSPM'


@pytest.fixture(scope='session')
def cache(tmp_path_factory):
    """Stage read-only operational inputs under a private test output root.

    Extractors save through the cache, so its production directory must never
    point at /data2/ATSPM. The raw source and existing profiles are only read;
    every generated feature is confined to pytest's temporary directory.
    """
    from feature_extractor.data_loader import SignalDataCache
    from feature_extractor.checked_runner import _read_events
    from feature_extractor.config_validator import validate_config
    from feature_extractor.feature_registry import output_path

    source_root = Path(DATA_ROOT)
    year, month, day = GOLDEN_DATE
    date_str = f'{year:04d}-{month:02d}-{day:02d}'
    source_events = source_root / 'raw_data' / 'per_signal' / GOLDEN_SIGNAL / f'atspm-{year}-{month}-{day}.parquet'
    source_config = source_root / 'config_data' / 'signal_config_2025' / 'lanes' / f'{GOLDEN_SIGNAL}.csv'
    profile_names = ('vehicle_phase_profile', 'pedestrian_phase_profile',
                     'vehicle_cycle_profile', 'pedestrian_cycle_profile')
    source_profiles = {name: output_path(source_root / 'processed_data', name, GOLDEN_SIGNAL, date_str)
                       for name in profile_names}
    missing = [str(path) for path in (source_events, source_config, *source_profiles.values()) if not path.is_file()]
    if missing:
        pytest.skip('Golden fixture requires local reference inputs: ' + ', '.join(missing))
    validation = validate_config(source_config, signal_id=GOLDEN_SIGNAL, data_date=date_str)
    if validation.status != 'valid':
        pytest.fail(f'Golden configuration is invalid: {validation.reason_code}')

    temporary_root = tmp_path_factory.mktemp('golden_1050_2025_09_13')
    interim = temporary_root / 'interim'
    production = temporary_root / 'production'
    events_path = interim / 'fdot_d5' / 'event_data' / GOLDEN_SIGNAL / f'{date_str}.parquet'
    events_path.parent.mkdir(parents=True)
    write_frame(_read_events(source_events, GOLDEN_SIGNAL), events_path)
    configs_path = interim / 'fdot_d5' / 'signal_config'
    configs_path.mkdir(parents=True)
    validation.dataframe.to_csv(configs_path / f'{GOLDEN_SIGNAL}.csv', index=False)
    for name, source in source_profiles.items():
        destination = output_path(production, name, GOLDEN_SIGNAL, date_str)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    return SignalDataCache(
        signal_id=GOLDEN_SIGNAL, year=year, month=month, day=day,
        cwd=str(temporary_root), relative_interim='interim',
        relative_production='production', config_dir=str(configs_path),
    )


@pytest.fixture(scope='session')
def golden_data():
    """Load all golden (v1) reference DataFrames."""
    data = {}
    if not os.path.isdir(GOLDEN_DIR):
        pytest.skip(f"Golden data not found: {GOLDEN_DIR}")
    for f in os.listdir(GOLDEN_DIR):
        if f.endswith('.parquet'):
            name = f.replace('.parquet', '')
            data[name] = read_frame(os.path.join(GOLDEN_DIR, f))
    return data
