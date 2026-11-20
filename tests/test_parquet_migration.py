import json
from pathlib import Path
import sqlite3

import pandas as pd
import pyarrow.parquet as pq
import pytest

from feature_extractor.result_types import FeatureResult
from feature_extractor.run_store import RunStore
from feature_extractor.storage import assert_frames_equal, read_frame
from scripts.migrate_pickles_to_parquet import (convert_plan, delete_sources,
                                                plan_migration)


def plan(tmp_path, sources):
    path = tmp_path / 'plan.json'
    plan_migration([sources], path)
    return path, tmp_path / 'journal.sqlite3'


def test_verified_migration_preserves_values_before_deleting_original(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    source = sources / 'sample.pkl'
    frame = pd.DataFrame({'values': [[float('nan'), 1.0]], 'count': [0]})
    frame.attrs['interval_closure'] = 'left'
    # PKL generation is confined to this migration fixture in pytest's /tmp.
    frame.to_pickle(source)
    path, journal = plan(tmp_path, sources)
    converted = convert_plan(path, journal, workers=1)
    assert converted['states'] == {'verified': 1}
    assert source.exists()
    assert_frames_equal(frame, read_frame(source.with_suffix('.parquet')))
    deleted = delete_sources(path, journal)
    assert deleted['states'] == {'deleted': 1}
    assert not source.exists()


def test_conversion_failure_prevents_any_source_deletion(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    pd.DataFrame({'x': [1]}).to_pickle(sources / 'valid.pkl')
    (sources / 'corrupt.pkl').write_bytes(b'not a pickle')
    path, journal = plan(tmp_path, sources)
    report = convert_plan(path, journal, workers=1)
    assert report['states'] == {'failed': 1, 'verified': 1}
    with pytest.raises(ValueError, match='lack verified'):
        delete_sources(path, journal)
    assert (sources / 'valid.pkl').is_file()
    assert (sources / 'corrupt.pkl').read_bytes() == b'not a pickle'


def test_changed_target_refuses_deletion(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    source = sources / 'sample.pkl'
    pd.DataFrame({'x': [1]}).to_pickle(source)
    path, journal = plan(tmp_path, sources)
    convert_plan(path, journal, workers=1)
    source.with_suffix('.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Converted output changed'):
        delete_sources(path, journal)
    assert source.is_file()


def test_old_execution_history_resolves_migration_and_detects_later_changes(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    source = sources / 'sample.pkl'
    pd.DataFrame({'x': [1]}).to_pickle(source)
    store = RunStore(tmp_path / 'runs.sqlite3')
    run = store.start_run()
    attempt = store.start_attempt(run, 'S', '2025-09-07', 'input', 'config', 'vectorized', ['volume'])
    store.finish_attempt(attempt, 'complete', {'volume': FeatureResult(
        'volume', 'success', output_file_path=str(source), row_count=1)})
    path, journal = plan(tmp_path, sources)
    convert_plan(path, journal, workers=1)
    delete_sources(path, journal)
    assert store.latest('S', '2025-09-07')['results']['volume']['output_file_path'] == str(source.with_suffix('.parquet'))
    assert store.reusable('S', '2025-09-07', 'input', 'config', 'vectorized', ['volume'])
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('SELECT output_file_path FROM feature_results').fetchone()[0] == str(source)
    target = source.with_suffix('.parquet')
    table = pq.read_table(target)
    changed = table.set_column(0, 'x', __import__('pyarrow').array([2]))
    pq.write_table(changed, target)
    assert store.reusable('S', '2025-09-07', 'input', 'config', 'vectorized', ['volume']) is None


def test_missing_value_override_does_not_hide_corrupted_physical_values(tmp_path):
    from feature_extractor.storage import write_frame
    path = tmp_path / 'sample.parquet'
    write_frame(pd.DataFrame({'x': pd.Series([float('nan')], dtype='object')}), path)
    table = pq.read_table(path)
    changed = table.set_column(0, 'x', __import__('pyarrow').array([42.0]))
    pq.write_table(changed, path)
    with pytest.raises(ValueError, match='contradicts missing-value'):
        read_frame(path)
