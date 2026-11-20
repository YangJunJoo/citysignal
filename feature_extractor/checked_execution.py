"""Isolated profile/feature execution with verified, atomic output publication."""
import os
import tempfile
from pathlib import Path

import pandas as pd

from .feature_registry import FEATURES, output_path, resolve_features
from .result_types import FeatureResult
from . import storage
from .profiles import (build_vehicle_phase_profile, build_vehicle_cycle_profile,
                       build_pedestrian_phase_profile, build_pedestrian_cycle_profile)


class NoFeatureData(ValueError):
    pass


def validate_frame(frame, name, signal_id):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('Extractor did not produce a DataFrame')
    if frame.empty:
        return
    required = {'signalID', 'phaseNo'} if name.endswith('phase_profile') else {'signalID', 'cycleNo', 'cycleBegin', 'cycleEnd'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing output columns: {sorted(missing)}')
    if frame['signalID'].isna().any() or set(frame['signalID'].astype(str)) != {str(signal_id)}:
        raise ValueError('Output signalID does not match the job')
    if name == 'vehicle_cycle_profile':
        from feature_extractor.profiles import validate_cycle_profile
        validate_cycle_profile(frame)
    if not name.endswith('_profile'):
        metric_columns = [c for c in frame.columns if str(c) not in {'signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength', 'date'} and not str(c).startswith('channelNos')]
        if not metric_columns:
            raise NoFeatureData('Output contains cycle metadata but no feature columns')
    if 'cycleBegin' in frame:
        begin, end = pd.to_datetime(frame['cycleBegin']), pd.to_datetime(frame['cycleEnd'])
        if begin.isna().any() or end.isna().any() or (end < begin).any():
            raise ValueError('Invalid cycle interval')


def publish_frame(frame, name, signal_id, destination):
    """A failed write or verification leaves the previous output intact."""
    validate_frame(frame, name, signal_id)
    if frame.empty:
        raise ValueError('Empty result cannot be published as success')
    storage.write_frame(frame, destination, validator=lambda restored: validate_frame(restored, name, signal_id))


def _result_from_call(name, call, staged_output, signal_id):
    try:
        data = call()
        if not isinstance(data, pd.DataFrame) and staged_output.is_file():
            data = storage.read_frame(staged_output)
        validate_frame(data, name, signal_id)
        if name == 'vehicle_cycle_profile' and data.empty and data.attrs.get('quarantined_cycle_count', 0):
            return FeatureResult(name, 'blocked', 'no_valid_cycle_windows',
                                 'All candidate cycles intersect uncertain barrier operation.')
        return FeatureResult(name, 'no_data' if data.empty else 'success',
                             'empty_result' if data.empty else '', data=data, row_count=len(data))
    except NoFeatureData as exc:
        return FeatureResult(name, 'no_data', 'missing_metric_columns', str(exc))
    except Exception as exc:
        # Profile builders preserve quality errors in their messages.
        # Treat a controller mapping that needs evidence as blocked, not as a
        # generic extraction crash, while retaining the original explanation.
        if name == 'vehicle_cycle_profile':
            for reason in ('inconsistent_barrier_mapping', 'ambiguous_barrier_start'):
                if reason + ':' in str(exc):
                    return FeatureResult(name, 'blocked', reason, f'{type(exc).__name__}: {exc}')
        return FeatureResult(name, 'failed', 'extraction_error', f'{type(exc).__name__}: {exc}')


def execute_frames(signal_id, date_str, events, validation, output_dir, features=None, before_publish=None, source_quality=None):
    from .mapped_execution import execute_mapped_frames
    return execute_mapped_frames(signal_id, date_str, events, validation, output_dir,
                                 features, before_publish, source_quality)


def _execute_cycle_frames(signal_id, date_str, events, validation, output_dir, features=None, before_publish=None, source_quality=None, mapped_phase=None):
    selected = resolve_features(features)
    results = {}
    year, month, day = map(int, date_str.split('-'))
    staging_root = Path(output_dir) / '.staging'
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'{signal_id}-{date_str}-', dir=staging_root) as temporary:
        stage = Path(temporary)
        production = stage / 'production'
        event_path = stage / 'interim' / 'fdot_d5' / 'event_data' / signal_id / f'{date_str}.parquet'
        config_path = stage / 'interim' / 'fdot_d5' / 'signal_config' / f'{signal_id}.csv'
        event_path.parent.mkdir(parents=True)
        config_path.parent.mkdir(parents=True)
        storage.write_frame(events, event_path)
        validation.dataframe.to_csv(config_path, index=False)

        def stage_profile(name, frame):
            path = output_path(production, name, signal_id, date_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            storage.write_frame(frame, path)
            return frame

        def vehicle_phase():
            frame = (mapped_phase if mapped_phase is not None else
                     build_vehicle_phase_profile(events, signal_id, date_str))
            return stage_profile('vehicle_phase_profile', frame)

        def vehicle_cycle():
            phases = results['vehicle_phase_profile'].data
            frame = build_vehicle_cycle_profile(
                phases, signal_id, date_str,
                conflict_policy=('reject' if mapped_phase is not None and
                                 not mapped_phase.attrs.get('barrier_mapping_assumed') else 'quarantine'))
            if mapped_phase is not None:
                for key, value in phases.attrs.items():
                    frame.attrs.setdefault(key, value)
            return stage_profile('vehicle_cycle_profile', frame)

        calls = {
            'vehicle_phase_profile': vehicle_phase,
            'vehicle_cycle_profile': vehicle_cycle,
            'pedestrian_phase_profile': lambda: stage_profile(
                'pedestrian_phase_profile', build_pedestrian_phase_profile(events, signal_id, date_str)),
            'pedestrian_cycle_profile': lambda: stage_profile(
                'pedestrian_cycle_profile', build_pedestrian_cycle_profile(
                    results['pedestrian_phase_profile'].data,
                    results['vehicle_cycle_profile'].data, signal_id, date_str)),
        }
        if mapped_phase is not None and not mapped_phase.attrs.get('barrier_mapping_assumed'):
            for name in selected:
                if name.startswith('pedestrian_'):
                    results[name] = FeatureResult(name, 'blocked', 'pedestrian_mapping_unverified')
        capabilities = validation.capabilities.get('features', {})
        applicability = {name: (cap.get('applicable'), cap.get('reason_code', '')) for name, cap in capabilities.items()}
        for name in selected:
            cap = capabilities.get(name, {})
            if cap.get('applicable') is False:
                results[name] = FeatureResult(name, 'not_applicable', cap['reason_code'])
            elif cap.get('valid') is False:
                results[name] = FeatureResult(name, 'blocked', cap['reason_code'])
        for name in selected:
            if FEATURES[name].scope != 'profile' or name in results:
                continue
            deps = FEATURES[name].dependencies
            if any(results[d].status != 'success' for d in deps):
                results[name] = FeatureResult(name, 'blocked', 'dependency_unavailable', ','.join(deps))
                continue
            results[name] = _result_from_call(name, calls[name], output_path(production, name, signal_id, date_str), signal_id)
        try:
            from .data_loader import SignalDataCache
            from .pipeline import extract_features_checked
            cache = SignalDataCache(signal_id, year, month, day, str(stage), 'interim', 'production')
            results = extract_features_checked(cache, selected, prior_results=results, applicability=applicability)
        except Exception as exc:
            for name in selected:
                if name not in results:
                    results[name] = FeatureResult(name, 'blocked', 'cache_load_failed', f'{type(exc).__name__}: {exc}')
        if before_publish is not None:
            before_publish()
        for name in selected:
            result = results[name]
            if result.status in {'success', 'failed', 'no_data'} and any(results[d].status != 'success' for d in FEATURES[name].dependencies):
                result.status, result.reason_code = 'blocked', 'dependency_publication_failed'
                result.data = None
            if result.status == 'success':
                if mapped_phase is not None:
                    for key in ('mapping_policy_version', 'mapping_basis', 'barrier_mapping_assumed',
                                'barrier_mapping_verified', 'barrier_map', 'assumed_phase_numbers',
                                'assumed_ring_sequences', 'controller_evidence',
                                'lane_mapping_effective_date_verified', 'pedestrian_mapping_status',
                                'unmapped_observed_phases', 'auxiliary_observed_phases',
                                'config_source', 'config_sha256'):
                        if key in mapped_phase.attrs:
                            result.data.attrs[key] = mapped_phase.attrs[key]
                if source_quality:
                    result.data.attrs.update(source_quality_scope=source_quality.get('quality_scope', 'file'),
                        source_file_status=source_quality.get('source_file_status', source_quality.get('status')),
                        unassigned_source_rows=source_quality.get('excluded_unassigned_rows', 0),
                        source_expected_coverage_verified=False)
                destination = output_path(output_dir, name, signal_id, date_str)
                try:
                    if 'cycleBegin' in result.data and not events.empty:
                        begin, end = result.data['cycleBegin'], result.data['cycleEnd']
                        if (begin < events['timeStamp'].min()).any() or (end > events['timeStamp'].max()).any():
                            raise ValueError('Output cycle interval exceeds observed input time range')
                    publish_frame(result.data, name, signal_id, destination)
                    result.output_file_path = str(destination)
                    result.row_count = len(result.data)
                except NoFeatureData as exc:
                    result.status, result.reason_code, result.message = 'no_data', 'missing_metric_columns', str(exc)
                    result.row_count = 0
                except Exception as exc:
                    result.status, result.reason_code, result.message = 'failed', 'output_validation_failed', str(exc)
    return results
