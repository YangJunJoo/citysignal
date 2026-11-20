
from feature_extractor import storage
from feature_extractor.storage import read_frame, write_frame
from types import SimpleNamespace

import pandas as pd
import pytest

from feature_extractor.checked_execution import _result_from_call
from feature_extractor.checked_runner import discover_jobs
from feature_extractor.feature_registry import resolve_features


def test_metadata_without_feature_values_is_no_data(tmp_path):
    data = pd.DataFrame({'signalID': ['S'], 'cycleNo': [1],
        'cycleBegin': [pd.Timestamp('2026-09-06T10:00:00')],
        'cycleEnd': [pd.Timestamp('2026-09-06T10:01:00')],
        'cycleLength': [60], 'date': ['2026-09-06']})
    result = _result_from_call('volume', lambda: data, tmp_path/'absent.parquet', 'S')
    assert result.status == 'no_data'
    assert result.reason_code == 'missing_metric_columns'


def test_empty_feature_selection_is_not_vacuously_complete():
    with pytest.raises(ValueError, match='At least one'):
        resolve_features([])


def test_nested_legacy_per_signal_directory_does_not_hide_inputs(tmp_path):
    base = tmp_path/'per_signal'
    (base/'per_signal').mkdir(parents=True)
    (base/'S').mkdir()
    (base/'S'/'atspm-2026-9-6.parquet').write_bytes(b'only discovery is tested')
    config = SimpleNamespace(event_dir=str(base), signals=['S'], year=2026, month=9, day=6,
                             features=['spat'], jobs_file=None)
    jobs = discover_jobs(config)
    assert [(job['signal_id'],job['date_str']) for job in jobs] == [('S','2026-09-06')]
