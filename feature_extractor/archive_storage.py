"""Typed Parquet archives for historical result bundles and cached input state.

Every DataFrame is a named list-of-struct column. Small scalar/container metadata
describes the object tree. Retired Python classes are recorded as inert state;
their code is neither imported nor executed when an archive is restored.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import storage

ARCHIVE_KEY = b'atspm.archive'


@dataclass
class ArchivedObjectState:
    qualified_class: str
    attributes: dict


def _archive_table(value, provenance):
    arrays, names, memo = [], [], {}

    def frame_node(frame):
        if id(frame) in memo:
            return memo[id(frame)][1]
        name = f'frame_{len(arrays)}'
        table = storage.frame_to_table(frame)
        rows = pa.StructArray.from_arrays([c.combine_chunks() for c in table.columns], fields=list(table.schema))
        arrays.append(pa.LargeListArray.from_arrays(pa.array([0, len(frame)], type=pa.int64()), rows))
        names.append(name)
        node = dict(kind='dataframe', field=name, contract=json.loads(table.schema.metadata[storage.CONTRACT_KEY]))
        memo[id(frame)] = (frame, node)
        return node

    def node(value):
        if isinstance(value, pd.DataFrame):
            return frame_node(value)
        if isinstance(value, ArchivedObjectState):
            return dict(kind='object_state', qualified_class=value.qualified_class, attributes=node(value.attributes))
        if isinstance(value, dict):
            return dict(kind='dict', items=[[storage._pack(k), node(v)] for k, v in value.items()])
        if isinstance(value, (tuple, list)):
            return dict(kind=type(value).__name__, items=[node(v) for v in value])
        if isinstance(value, np.ndarray):
            series = pd.Series(value.reshape(-1))
            return dict(kind='ndarray', shape=list(value.shape), dtype=value.dtype.str,
                        frame=frame_node(pd.DataFrame({'value': series})))
        return dict(kind='scalar', value=storage._pack(value))

    tree = node(value)
    if not arrays:
        arrays, names = [pa.array([True])], ['__atspm_archive']
    contract = dict(version=1, tree=tree, provenance=provenance)
    return pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(
        {ARCHIVE_KEY: json.dumps(contract, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()})


def read_artifact(path):
    table = pq.ParquetFile(path).read()
    metadata = table.schema.metadata or {}
    if storage.CONTRACT_KEY in metadata:
        return storage.table_to_frame(table)
    if ARCHIVE_KEY not in metadata:
        raise ValueError('Missing ATSPM archive contract')
    contract = json.loads(metadata[ARCHIVE_KEY])
    if contract.get('version') != 1:
        raise ValueError('Unsupported ATSPM archive schema')
    memo = {}

    def restore(node):
        kind = node['kind']
        if kind == 'dataframe':
            name = node['field']
            if name not in memo:
                rows = table[name][0].values
                columns = [rows.field(i) for i in range(len(rows.type))]
                restored = pa.Table.from_arrays(columns, names=[field.name for field in rows.type])
                restored = restored.replace_schema_metadata({storage.CONTRACT_KEY: json.dumps(node['contract']).encode()})
                memo[name] = storage.table_to_frame(restored)
            return memo[name]
        if kind == 'object_state':
            return ArchivedObjectState(node['qualified_class'], restore(node['attributes']))
        if kind == 'dict':
            return {storage._unpack(k): restore(v) for k, v in node['items']}
        if kind in {'list', 'tuple'}:
            values = [restore(v) for v in node['items']]
            return tuple(values) if kind == 'tuple' else values
        if kind == 'ndarray':
            frame = restore(node['frame'])
            return frame['value'].to_numpy(dtype=node['dtype']).reshape(node['shape'])
        if kind == 'scalar':
            return storage._unpack(node['value'])
        raise ValueError(f'Unknown archive node: {kind}')

    return restore(contract['tree'])


def assert_artifacts_equal(expected, actual):
    if isinstance(expected, pd.DataFrame):
        storage.assert_frames_equal(expected, actual)
    elif isinstance(expected, ArchivedObjectState):
        assert isinstance(actual, ArchivedObjectState)
        assert expected.qualified_class == actual.qualified_class
        assert_artifacts_equal(expected.attributes, actual.attributes)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert storage._pack(list(expected)) == storage._pack(list(actual))
        for k, value in expected.items():
            assert_artifacts_equal(value, actual[k])
    elif isinstance(expected, (tuple, list)):
        assert type(expected) is type(actual) and len(expected) == len(actual)
        for left, right in zip(expected, actual):
            assert_artifacts_equal(left, right)
    else:
        assert storage._pack(expected) == storage._pack(actual)


def write_artifact(value, destination, *, provenance=None):
    if isinstance(value, pd.DataFrame):
        return storage.write_frame(value, destination, provenance=provenance)
    destination = Path(destination)
    if destination.suffix != '.parquet':
        raise ValueError('Archives must use the .parquet extension')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        pq.write_table(_archive_table(value, provenance), temporary, compression='zstd', version='2.6', compression_level=3)
        assert_artifacts_equal(value, read_artifact(temporary))
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
