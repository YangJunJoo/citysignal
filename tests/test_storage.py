from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from feature_extractor.storage import (CONTRACT_KEY, assert_frames_equal, frame_to_table,
                                        read_frame, write_frame)


def test_time_intervals_use_structs_and_restore_nanoseconds_and_tuples(tmp_path):
    t = pd.Timestamp('2025-09-07T12:00:00.000000001')
    frame = pd.DataFrame({'greenPhase2': [[(t, t + pd.Timedelta(1, unit='ns'))], []]})
    path = tmp_path / 'intervals.parquet'
    write_frame(frame, path)
    field = pq.read_schema(path).field('greenPhase2')
    assert pa.types.is_list(field.type)
    assert pa.types.is_struct(field.type.value_type)
    assert field.type.value_type.field('begin').type == pa.timestamp('ns')
    result = read_frame(path)
    assert isinstance(result.greenPhase2.iloc[0], list)
    assert isinstance(result.greenPhase2.iloc[0][0], tuple)
    assert result.greenPhase2.iloc[0][0][0].value == t.value
    assert_frames_equal(frame, result)


def test_nested_empty_missing_and_mixed_numeric_values_are_distinct(tmp_path):
    frame = pd.DataFrame({'values': pd.Series([
        [], None, float('nan'), pd.NA, [None, float('nan'), pd.NA],
        [[1.0, float('nan')], []], [(float('nan'), float('nan'))],
        [1, 2.5], [2**60 + 1, 0.5]], dtype='object')})
    path = tmp_path / 'mixed.parquet'
    write_frame(frame, path)
    assert_frames_equal(frame, read_frame(path))


def test_metadata_retains_key_types_timestamps_and_quality_contract(tmp_path):
    t = pd.Timestamp('2025-09-07T12:00:00.000000001', tz='America/New_York')
    frame = pd.DataFrame({'value': [0.0]})
    frame.attrs = {'profile_version': 'observed', 'interval_closure': 'left',
                   'quarantine_windows': [(t, t + pd.Timedelta(1, unit='ns'))],
                   'barrier_map': {2: 1}, 'unknown': pd.NA, 'array': np.array([1, 2], dtype='int32')}
    path = tmp_path / 'metadata.parquet'
    write_frame(frame, path)
    assert CONTRACT_KEY in pq.read_schema(path).metadata
    result = read_frame(path)
    assert result.attrs['interval_closure'] == 'left'
    assert 2 in result.attrs['barrier_map']
    assert str(result.attrs['quarantine_windows'][0][0].tz) == 'America/New_York'
    assert_frames_equal(frame, result)


@pytest.mark.parametrize('index', [pd.RangeIndex(3, 9, 2, name='row'),
    pd.Index(['c', 'a', 'b'], name='row'),
    pd.date_range('2025-01-01', periods=3, freq='h', name='time'),
    pd.MultiIndex.from_tuples([('a', 2), ('a', 1), ('b', 1)], names=['signal', 'cycle'])])
def test_dtypes_indexes_and_signed_zero_survive(tmp_path, index):
    frame = pd.DataFrame({'float': [0.0, -0.0, np.nan],
        'integer': pd.array([1, None, 3], dtype='Int64'),
        'flag': pd.array([True, None, False], dtype='boolean'),
        'text': pd.array(['a', None, 'b'], dtype='string'),
        'category': pd.Categorical(['b', 'a', 'b'], categories=['b', 'a'], ordered=True),
        'date': [date(2025, 1, 1)] * 3}, index=index)
    path = tmp_path / 'types.parquet'
    write_frame(frame, path)
    result = read_frame(path)
    assert np.signbit(result['float'].iloc[1])
    assert_frames_equal(frame, result)


@pytest.mark.parametrize('shape', [(0,), (0, 2), (2, 2), ()])
def test_numpy_object_cells_keep_array_shape_and_dtype(tmp_path, shape):
    array = np.zeros(shape, dtype='int32')
    frame = pd.DataFrame({'arrays': [array]})
    path = tmp_path / 'arrays.parquet'
    write_frame(frame, path)
    result = read_frame(path)
    assert isinstance(result.arrays.iloc[0], np.ndarray)
    assert result.arrays.iloc[0].shape == shape
    assert_frames_equal(frame, result)


@pytest.mark.parametrize('frame', [pd.DataFrame(), pd.DataFrame(index=range(3)),
                                   pd.DataFrame({'values': pd.Series([], dtype='object')})])
def test_empty_results_preserve_shape_and_schema(tmp_path, frame):
    path = tmp_path / 'empty.parquet'
    write_frame(frame, path)
    assert_frames_equal(frame, read_frame(path))


def test_unknown_or_missing_contract_is_rejected(tmp_path):
    path = tmp_path / 'unknown.parquet'
    pq.write_table(pa.table({'x': [1]}), path)
    with pytest.raises(ValueError, match='Missing ATSPM'):
        read_frame(path)
    table = frame_to_table(pd.DataFrame({'x': [1]}))
    metadata = dict(table.schema.metadata)
    metadata[CONTRACT_KEY] = metadata[CONTRACT_KEY].replace(b'"version":1', b'"version":999')
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(ValueError, match='Unsupported'):
        read_frame(path)


def test_equality_rejects_small_nested_changes_missing_type_and_attrs():
    frame = pd.DataFrame({'values': [[1.0, float('nan')]]})
    for changed in [pd.DataFrame({'values': [[1.0 + 1e-12, float('nan')]]}),
                    pd.DataFrame({'values': [[1.0, None]]})]:
        with pytest.raises(AssertionError):
            assert_frames_equal(frame, changed)
    changed = frame.copy(deep=True)
    changed.attrs['interval_closure'] = 'left'
    with pytest.raises(AssertionError, match='attrs'):
        assert_frames_equal(frame, changed)


def test_repository_reference_datasets_are_contract_bearing_parquet():
    from pathlib import Path
    golden = Path(__file__).resolve().parent / 'golden_data' / '1050_2025-09-13'
    assert not list(golden.glob('*.pkl'))
    files = list(golden.glob('*.parquet'))
    assert files, 'The repository must retain its converted reference datasets'
    for path in files:
        assert isinstance(read_frame(path), pd.DataFrame)
