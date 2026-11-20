import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from feature_extractor.archive_storage import (ARCHIVE_KEY, ArchivedObjectState,
    assert_artifacts_equal, read_artifact, write_artifact)
from feature_extractor.storage import read_frame


def test_result_bundle_retains_distinct_frames_scalar_keys_and_shared_frames(tmp_path):
    first = pd.DataFrame({'count': [0, 2], 'samples': [[float('nan'), 1.0], []]})
    first.attrs['interval_closure'] = 'left'
    second = pd.DataFrame({'label': ['a'], 'time': [pd.Timestamp('2025-01-01T00:00:00.000000001')]})
    value = {'first': first, 'other': {'frame': second, 'same_frame': first},
             (81, 82): {'status': 'success', 'none': None}, 'empty': pd.DataFrame()}
    path = tmp_path / 'bundle.parquet'
    write_artifact(value, path)
    result = read_artifact(path)
    assert_artifacts_equal(value, result)
    assert result['first'] is result['other']['same_frame']
    schema = pq.read_schema(path)
    assert ARCHIVE_KEY in schema.metadata
    assert pa.types.is_large_list(schema.field('frame_0').type)
    assert pa.types.is_struct(schema.field('frame_0').type.value_type)
    with pytest.raises(ValueError, match='Missing ATSPM storage'):
        read_frame(path)


def test_cached_state_is_restored_without_importing_or_executing_old_class(tmp_path):
    value = ArchivedObjectState('retired.example.Cache', {
        'events': pd.DataFrame({'x': [1, 2]}),
        'array': np.array([[1, 2]], dtype='int32'),
        'other_array': np.array([4.0, float('nan')]), 'options': (True, 'S')})
    path = tmp_path / 'state.parquet'
    write_artifact(value, path)
    actual = read_artifact(path)
    assert_artifacts_equal(value, actual)
    assert isinstance(actual, ArchivedObjectState)
    assert actual.qualified_class == 'retired.example.Cache'


def test_scalar_only_bundle_uses_a_real_parquet_table(tmp_path):
    value = {'status': 'empty', 'count': 0, 'values': (None, float('nan'))}
    path = tmp_path / 'scalar.parquet'
    write_artifact(value, path)
    assert pq.read_table(path).num_rows == 1
    assert_artifacts_equal(value, read_artifact(path))
