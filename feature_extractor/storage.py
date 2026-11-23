"""Typed Parquet persistence with an explicit, reversible DataFrame contract.

The file contains Arrow values, not pickled Python objects. Container restoration,
missing-value distinctions, pandas dtypes and attrs are described in the footer.
"""
from datetime import date, datetime, timedelta
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
CONTRACT_KEY = b'atspm.dataframe'


def _pack(value):
    """Encode metadata without pickle or lossy default=str conversions."""
    if value is None:
        return ['none']
    if value is pd.NA:
        return ['NA']
    if value is pd.NaT:
        return ['NaT']
    if isinstance(value, np.ndarray):
        return ['array', value.dtype.str, list(value.shape), [_pack(v) for v in value.flat]]
    if isinstance(value, np.generic):
        if value.dtype.kind in 'mM':
            return ['numpy_time', value.dtype.str, str(value.view('i8'))]
        return ['numpy', value.dtype.str, _pack(value.item())]
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', str(value)]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, str):
        return ['str', value]
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, pd.Timestamp):
        return ['timestamp', str(value.value), str(value.tz) if value.tz else None, value.unit]
    if isinstance(value, pd.Timedelta):
        return ['timedelta', str(value.value)]
    if isinstance(value, datetime):
        return ['datetime', value.isoformat()]
    if isinstance(value, date):
        return ['date', value.isoformat()]
    if isinstance(value, timedelta):
        return ['python_timedelta', value.days, value.seconds, value.microseconds]
    if isinstance(value, dict):
        return ['dict', [[_pack(k), _pack(v)] for k, v in value.items()]]
    if isinstance(value, (tuple, list, set, frozenset)):
        return [type(value).__name__, [_pack(v) for v in value]]
    if isinstance(value, Path):
        return ['path', str(value)]
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def _unpack(value):
    kind, *args = value
    if kind in {'none', 'NA', 'NaT'}:
        return {'none': None, 'NA': pd.NA, 'NaT': pd.NaT}[kind]
    if kind in {'str', 'bool'}:
        return args[0]
    if kind == 'int':
        return int(args[0])
    if kind == 'float':
        return float.fromhex(args[0])
    if kind == 'bytes':
        return bytes.fromhex(args[0])
    if kind == 'timestamp':
        result = pd.Timestamp(int(args[0]), unit='ns', tz='UTC' if args[1] else None)
        if args[1]:
            result = result.tz_convert(args[1])
        return result.as_unit(args[2])
    if kind == 'timedelta':
        return pd.Timedelta(int(args[0]), unit='ns')
    if kind == 'datetime':
        return datetime.fromisoformat(args[0])
    if kind == 'date':
        return date.fromisoformat(args[0])
    if kind == 'python_timedelta':
        return timedelta(days=args[0], seconds=args[1], microseconds=args[2])
    if kind == 'path':
        return Path(args[0])
    if kind == 'numpy_time':
        return np.array(int(args[1]), dtype='int64').view(np.dtype(args[0]))[()]
    if kind == 'numpy':
        return np.array(_unpack(args[1]), dtype=np.dtype(args[0]))[()]
    if kind == 'array':
        items = [_unpack(v) for v in args[2]]
        result = np.empty(len(items), dtype=np.dtype(args[0]))
        for i, item in enumerate(items):
            result[i] = item
        return result.reshape(args[1])
    if kind == 'dict':
        return {_unpack(k): _unpack(v) for k, v in args[0]}
    if kind in {'list', 'tuple', 'set', 'frozenset'}:
        constructors = {'list': list, 'tuple': tuple, 'set': set, 'frozenset': frozenset}
        return constructors[kind](_unpack(v) for v in args[0])
    raise ValueError(f'Unknown metadata tag: {kind}')


def _missing(value):
    return (value is None or value is pd.NA or value is pd.NaT or
            isinstance(value, (float, np.floating)) and math.isnan(value) or
            isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value))


def _kind(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return 'sequence'
    if isinstance(value, dict):
        return 'mapping'
    if isinstance(value, (bool, np.bool_)):
        return 'bool'
    if isinstance(value, (int, np.integer)):
        return 'int'
    if isinstance(value, (float, np.floating)):
        return 'float'
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return 'timestamp'
    if isinstance(value, (pd.Timedelta, timedelta, np.timedelta64)):
        return 'duration'
    if isinstance(value, date):
        return 'date'
    if isinstance(value, str):
        return 'str'
    if isinstance(value, bytes):
        return 'bytes'
    raise TypeError(f'Unsupported object-column value: {type(value).__name__}')


def _container(value):
    if isinstance(value, np.ndarray):
        return ['ndarray', value.dtype.str]
    return [type(value).__name__]


def _items(value):
    return value.reshape(-1).tolist() if isinstance(value, np.ndarray) and value.ndim == 0 else list(value)


def _infer(values):
    valid = [v for v in values if not _missing(v)]
    if not valid:
        return {'kind': 'null'}
    kinds = {_kind(v) for v in valid}
    if kinds == {'int', 'float'} and all(not isinstance(v, (int, np.integer)) or int(float(v)) == int(v) for v in valid):
        return {'kind': 'float'}
    if len(kinds) != 1:
        return {'kind': 'variant', 'variants': [_infer([v for v in valid if _kind(v) == k]) for k in sorted(kinds)]}
    kind = next(iter(kinds))
    if kind == 'sequence':
        if all(isinstance(v, tuple) and len(v) == 2 and
               all(_missing(x) or _kind(x) == 'timestamp' for x in v) for v in valid):
            return {'kind': 'interval', 'begin': _infer([v[0] for v in valid]), 'end': _infer([v[1] for v in valid])}
        children = [x for v in valid for x in _items(v)]
        return {'kind': kind, 'container': _container(valid[0]), 'child': _infer(children)}
    if kind == 'mapping':
        keys = list(dict.fromkeys(k for v in valid for k in v))
        return {'kind': kind, 'keys': [_pack(k) for k in keys],
                'children': [_infer([v.get(k) for v in valid]) for k in keys]}
    if kind == 'timestamp':
        zones = {str(pd.Timestamp(v).tz) if pd.Timestamp(v).tz else None for v in valid}
        if len(zones) != 1:
            raise TypeError('A timestamp column cannot mix time zones')
        return {'kind': kind, 'tz': next(iter(zones)), 'python_type': type(valid[0]).__name__,
                'numpy_dtype': valid[0].dtype.str if isinstance(valid[0], np.datetime64) else None}
    if kind == 'duration':
        return {'kind': kind, 'python_type': type(valid[0]).__name__,
                'numpy_dtype': valid[0].dtype.str if isinstance(valid[0], np.timedelta64) else None}
    if kind == 'int' and min(valid) >= 0 and max(valid) > np.iinfo('int64').max:
        return {'kind': kind, 'unsigned': True}
    return {'kind': kind}


def _arrow_type(codec):
    kind = codec['kind']
    simple = {'null': pa.null(), 'bool': pa.bool_(), 'int': pa.int64(), 'float': pa.float64(),
              'str': pa.string(), 'bytes': pa.binary(), 'date': pa.date32(), 'duration': pa.duration('ns')}
    if kind in simple:
        if kind == 'int' and codec.get('unsigned'):
            return pa.uint64()
        return simple[kind]
    if kind == 'timestamp':
        return pa.timestamp('ns', tz=codec['tz'])
    if kind == 'sequence':
        return pa.list_(_arrow_type(codec['child']))
    if kind == 'interval':
        return pa.struct([('begin', _arrow_type(codec['begin'])), ('end', _arrow_type(codec['end']))])
    if kind == 'mapping':
        return pa.struct([(f'k{i}', _arrow_type(c)) for i, c in enumerate(codec['children'])])
    if kind == 'variant':
        return pa.struct([('__kind', pa.int16())] + [(f'v{i}', _arrow_type(c)) for i, c in enumerate(codec['variants'])])
    raise ValueError(kind)


def _encode(value, codec, overrides, path):
    if _missing(value):
        if value is not None:
            overrides.append([list(path), 'value', _pack(value)])
        return None
    kind = codec['kind']
    if kind == 'variant':
        index = next(i for i, c in enumerate(codec['variants']) if c['kind'] == _kind(value) or
                     c['kind'] == 'interval' and _kind(value) == 'sequence')
        return {'__kind': index, f'v{index}': _encode(value, codec['variants'][index], overrides, path)}
    if kind == 'sequence':
        container = _container(value)
        if container != codec['container']:
            overrides.append([list(path), 'container', container])
        if isinstance(value, np.ndarray):
            overrides.append([list(path), 'shape', list(value.shape)])
        return [_encode(v, codec['child'], overrides, (*path, i)) for i, v in enumerate(_items(value))]
    if kind == 'interval':
        return {key: _encode(v, codec[key], overrides, (*path, i)) for i, (key, v) in enumerate(zip(('begin', 'end'), value))}
    if kind == 'mapping':
        keys = [_unpack(k) for k in codec['keys']]
        if list(value) != keys:
            overrides.append([list(path), 'keys', _pack(list(value))])
        return {f'k{i}': _encode(value.get(k), c, overrides, (*path, i)) for i, (k, c) in enumerate(zip(keys, codec['children']))}
    if kind == 'float' and isinstance(value, (int, np.integer)):
        overrides.append([list(path), 'integer', None])
        return float(value)
    if kind in {'timestamp', 'duration'}:
        if type(value).__name__ != codec['python_type']:
            overrides.append([list(path), 'value', _pack(value)])
        return pd.Timestamp(value) if kind == 'timestamp' else pd.Timedelta(value)
    return value


def _decode(value, codec, overrides, path):
    extra = overrides.get(path, {})
    if 'value' in extra:
        original = _unpack(extra['value'])
        if _missing(original):
            if not _missing(value):
                raise ValueError('Stored value contradicts missing-value contract')
        elif codec['kind'] == 'timestamp':
            if pd.Timestamp(value) != pd.Timestamp(original):
                raise ValueError('Stored timestamp contradicts restoration contract')
        elif codec['kind'] == 'duration':
            if pd.Timedelta(value) != pd.Timedelta(original):
                raise ValueError('Stored duration contradicts restoration contract')
        else:
            raise ValueError('Unexpected non-missing restoration override')
        return original
    if value is None:
        return None
    kind = codec['kind']
    if kind == 'variant':
        index = value['__kind']
        return _decode(value[f'v{index}'], codec['variants'][index], overrides, path)
    if kind == 'sequence':
        result = [_decode(v, codec['child'], overrides, (*path, i)) for i, v in enumerate(value)]
        container = extra.get('container', codec['container'])
        if container[0] == 'tuple':
            return tuple(result)
        if container[0] == 'ndarray':
            array = np.empty(len(result), dtype=container[1]) if container[1] == '|O' else np.asarray(result, dtype=container[1])
            if container[1] == '|O':
                for i, item in enumerate(result):
                    array[i] = item
            return array.reshape(extra.get('shape', array.shape))
        return result
    if kind == 'interval':
        return tuple(_decode(value[k], codec[k], overrides, (*path, i)) for i, k in enumerate(('begin', 'end')))
    if kind == 'mapping':
        keys = [_unpack(k) for k in codec['keys']]
        result = {k: _decode(value[f'k{i}'], c, overrides, (*path, i)) for i, (k, c) in enumerate(zip(keys, codec['children']))}
        return {k: result[k] for k in _unpack(extra['keys'])} if 'keys' in extra else result
    if kind == 'timestamp':
        result = pd.Timestamp(value)
        if codec['python_type'] == 'datetime':
            return result.to_pydatetime()
        if codec['python_type'] == 'datetime64':
            return result.to_datetime64().astype(codec['numpy_dtype'])
        return result
    if kind == 'duration':
        result = pd.Timedelta(value)
        if codec['python_type'] == 'timedelta':
            return result.to_pytimedelta()
        if codec['python_type'] == 'timedelta64':
            return result.to_timedelta64().astype(codec['numpy_dtype'])
        return result
    return int(value) if 'integer' in extra else value


def _index_contract(index):
    if isinstance(index, pd.RangeIndex):
        return {'kind': 'range', 'start': index.start, 'stop': index.stop, 'step': index.step, 'name': _pack(index.name)}
    if isinstance(index, pd.MultiIndex):
        return {'kind': 'multi', 'values': _pack(list(index)), 'names': _pack(list(index.names))}
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return {'kind': type(index).__name__, 'values': _pack(list(index)), 'dtype': str(index.dtype),
                'name': _pack(index.name), 'freq': index.freqstr}
    return {'kind': 'index', 'values': _pack(list(index)), 'dtype': str(index.dtype), 'name': _pack(index.name)}


def _restore_index(contract):
    if contract['kind'] == 'range':
        return pd.RangeIndex(contract['start'], contract['stop'], contract['step'], name=_unpack(contract['name']))
    if contract['kind'] == 'multi':
        return pd.MultiIndex.from_tuples(_unpack(contract['values']), names=_unpack(contract['names']))
    if contract['kind'] in {'DatetimeIndex', 'TimedeltaIndex'}:
        constructor = getattr(pd, contract['kind'])
        return constructor(_unpack(contract['values']), dtype=contract['dtype'],
                           name=_unpack(contract['name']), freq=contract['freq'])
    return pd.Index(_unpack(contract['values']), dtype=contract['dtype'], name=_unpack(contract['name']), tupleize_cols=False)


def frame_to_table(frame, *, provenance=None):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Output must be a pandas DataFrame')
    contract = {'version': SCHEMA_VERSION, 'attrs': _pack(frame.attrs), 'index': _index_contract(frame.index),
                'columns': _index_contract(frame.columns), 'fields': [], 'allows_duplicate_labels': frame.flags.allows_duplicate_labels}
    if provenance is not None:
        contract['provenance'] = provenance
    arrays, names = [], []
    for i in range(len(frame.columns)):
        series = frame.iloc[:, i]
        name = str(frame.columns[i]) if isinstance(frame.columns[i], str) and frame.columns.is_unique else f'__column_{i}'
        spec = {'name': name, 'dtype': str(series.dtype)}
        if pd.api.types.is_object_dtype(series.dtype):
            values = series.tolist()
            codec = _infer(values)
            overrides = []
            encoded = [_encode(v, codec, overrides, (row,)) for row, v in enumerate(values)]
            arrays.append(pa.array(encoded, type=_arrow_type(codec), from_pandas=False))
            spec.update(codec=codec, overrides=overrides)
        else:
            arrays.append(pa.Array.from_pandas(series))
            if isinstance(series.dtype, pd.CategoricalDtype):
                spec['categories'] = _pack(series.cat.categories.tolist())
                spec['ordered'] = series.cat.ordered
            if isinstance(series.dtype, pd.StringDtype):
                spec['string_storage'] = series.dtype.storage
        contract['fields'].append(spec)
        names.append(name)
    if not arrays:
        arrays, names = [pa.nulls(len(frame))], ['__atspm_empty_frame']
    metadata = {CONTRACT_KEY: json.dumps(contract, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()}
    return pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(metadata)


def read_frame(path):
    """Read a contract-bearing output. Missing or unknown contracts fail closed."""
    table = pq.ParquetFile(path).read()
    return table_to_frame(table)


def table_to_frame(table):
    """Restore a DataFrame contract from a typed Arrow table."""
    encoded = (table.schema.metadata or {}).get(CONTRACT_KEY)
    if encoded is None:
        raise ValueError('Missing ATSPM storage contract')
    contract = json.loads(encoded)
    if contract.get('version') != SCHEMA_VERSION:
        raise ValueError(f'Unsupported ATSPM storage schema: {contract.get("version")}')
    columns = []
    for spec in contract['fields']:
        array = table[spec['name']]
        if 'codec' in spec:
            overrides = {}
            for path_keys, kind, value in spec['overrides']:
                overrides.setdefault(tuple(path_keys), {})[kind] = value
            values = [_decode(value, spec['codec'], overrides, (i,)) for i, value in enumerate(array.to_pylist())]
            series = pd.Series(values, dtype='object')
        elif 'categories' in spec:
            series = pd.Series(pd.Categorical(array.to_pylist(), categories=_unpack(spec['categories']), ordered=spec['ordered']))
        else:
            dtype = pd.StringDtype(storage=spec['string_storage']) if 'string_storage' in spec else spec['dtype']
            series = pd.Series(array.to_pandas()).astype(dtype)
        columns.append(series)
    frame = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=range(table.num_rows))
    frame.columns = _restore_index(contract['columns'])
    frame.index = _restore_index(contract['index'])
    frame.attrs = _unpack(contract['attrs'])
    frame.flags.allows_duplicate_labels = contract['allows_duplicate_labels']
    return frame


def _same(left, right):
    if _missing(left) or _missing(right):
        if left is right:
            return True
        if isinstance(left, (np.datetime64, np.timedelta64)) and isinstance(right, type(left)):
            return left.dtype == right.dtype and np.isnat(left) and np.isnat(right)
        return isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)) and math.isnan(left) and math.isnan(right)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and all(_same(a, b) for a, b in zip(left.flat, right.flat))
    if isinstance(left, dict):
        return isinstance(right, dict) and list(left) == list(right) and all(_same(left[k], right[k]) for k in left)
    if _kind(left) != _kind(right):
        return False
    if isinstance(left, (float, np.floating)) and left == 0 and right == 0:
        return bool(np.signbit(left)) == bool(np.signbit(right))
    return bool(left == right)


def assert_frames_equal(expected, actual):
    """Zero tolerance, with explicit nested-container and metadata checks."""
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=0, atol=0)
    if _pack(expected.attrs) != _pack(actual.attrs):
        raise AssertionError('DataFrame attrs changed during Parquet round trip')
    for i in range(len(expected.columns)):
        left, right = expected.iloc[:, i], actual.iloc[:, i]
        if pd.api.types.is_object_dtype(left.dtype):
            for row, (a, b) in enumerate(zip(left, right)):
                if not _same(a, b):
                    raise AssertionError(f'Object value changed in column {left.name!r}, row {row}')
        elif pd.api.types.is_float_dtype(left.dtype):
            a, b = left.to_numpy(dtype=float, na_value=np.nan), right.to_numpy(dtype=float, na_value=np.nan)
            valid = ~(np.isnan(a) | np.isnan(b))
            if not np.array_equal(np.signbit(a[valid]), np.signbit(b[valid])):
                raise AssertionError(f'Floating-point signs changed in {left.name!r}')


def write_frame(frame, destination, *, validator=None, provenance=None):
    """Validate, fsync and atomically publish one self-contained Parquet file."""
    destination = Path(destination)
    if destination.suffix != '.parquet':
        raise ValueError('ATSPM outputs must use the .parquet extension')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name, suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        pq.write_table(frame_to_table(frame, provenance=provenance), temporary, version='2.6', compression='zstd', compression_level=3)
        restored = read_frame(temporary)
        if validator is not None:
            validator(restored)
        assert_frames_equal(frame, restored)
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


def resolve_migrated_result(path, signature):
    """Resolve old ledger paths using verified migration provenance, not guessing.

    Historical ledger rows remain unchanged. Only an exact original signature
    can acquire the migrated file's current signature for completion reuse.
    """
    if not path or Path(path).suffix != '.pkl':
        return path, signature
    target = Path(path).with_suffix('.parquet')
    if not target.is_file():
        return path, signature
    try:
        encoded = (pq.read_schema(target).metadata or {}).get(CONTRACT_KEY)
        contract = json.loads(encoded) if encoded else {}
        provenance = contract.get('provenance', {})
        if contract.get('version') != SCHEMA_VERSION or provenance.get('source_path') != str(Path(path).resolve()):
            return path, signature
        journal = provenance.get('journal_path')
        if provenance.get('source_signature') == signature and journal and Path(journal).is_file():
            with sqlite3.connect(Path(journal).resolve().as_uri() + '?mode=ro', uri=True) as conn:
                record = conn.execute('SELECT target_signature FROM files WHERE source=? AND source_signature=? AND state IN (?,?)',
                                      (str(Path(path).resolve()), signature, 'verified', 'deleted')).fetchone()
            if record:
                signature = record[0]
        return str(target), signature
    except (OSError, ValueError, sqlite3.Error, pa.ArrowException):
        return path, signature
