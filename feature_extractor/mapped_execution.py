"""Publish checked results under the default eight-phase operating assumption.

Explicit reviewed controller plans remain supported. Diagnostic observations
are retained without inventing lane or detector identities.
"""
import json
from pathlib import Path

from feature_extractor.profiles import build_vehicle_phase_profile
from .feature_registry import output_path, resolve_features
from .mapping_policy import assess_mapping
from .result_types import FeatureResult


def diagnostic_path(root, kind, signal_id, date_str, suffix):
    return Path(root) / 'diagnostics' / kind / str(signal_id) / (date_str + suffix)


def execute_mapped_frames(signal_id, date_str, events, validation, output_dir,
                          features=None, before_publish=None, source_quality=None):
    from .checked_execution import publish_frame, _execute_cycle_frames
    selected = resolve_features(features)
    # Classify the observed identities before applying the selected model.
    raw_phase = build_vehicle_phase_profile(events, signal_id, date_str, barrier_map={})
    audit = assess_mapping(validation, signal_id, date_str, raw_phase.phaseNo.unique())
    assumed = audit['barrier_mapping_assumed']
    mapped = (raw_phase.copy() if assumed else
              raw_phase.loc[raw_phase.phaseNo.isin(audit['movement_mappings'])].copy())
    diagnostic = raw_phase.loc[~raw_phase.phaseNo.isin(audit['movement_mappings'])].copy()
    if audit['barrier_mapping_usable']:
        # Auxiliary IDs are preserved but do not define the eight-phase barriers.
        mapped['barrierNo'] = mapped.phaseNo.map(audit['barrier_map']).fillna(0).astype(int)
    mapped.attrs.update(audit)
    mapped.attrs['coverage_scope'] = ('observed_phases_under_eight_phase_assumption' if assumed
                                     else 'identified_phases_in_lane_inventory')
    diagnostic.attrs.update(audit, usage='phase_identity_without_lane_mapping' if assumed
                            else 'diagnostic_only_not_for_movement_metrics')
    if source_quality:
        attrs = dict(source_quality_scope=source_quality.get('quality_scope', 'file'),
                     source_file_status=source_quality.get('source_file_status', source_quality.get('status')),
                     unassigned_source_rows=source_quality.get('excluded_unassigned_rows', 0),
                     source_expected_coverage_verified=False)
        mapped.attrs.update(attrs)
        diagnostic.attrs.update(attrs)
    audit.update(mapped_phase_rows=len(mapped), diagnostic_phase_rows=len(diagnostic))
    if before_publish is not None:
        before_publish()
    if not diagnostic.empty:
        publish_frame(diagnostic, 'vehicle_phase_profile', signal_id,
                      diagnostic_path(output_dir, 'unmapped_phase', signal_id, date_str, '.parquet'))
    target = diagnostic_path(output_dir, 'mapping_audit', signal_id, date_str, '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(audit, ensure_ascii=False, default=str))
    temp.chmod(0o644)
    temp.replace(target)
    if audit['barrier_mapping_usable'] and not mapped.empty:
        results = _execute_cycle_frames(signal_id, date_str, events, validation, output_dir,
                                       selected, before_publish, source_quality,
                                       mapped_phase=mapped)
        return results
    results = {}
    for name in selected:
        if name == 'vehicle_phase_profile':
            if mapped.empty:
                reason = 'empty_result' if raw_phase.empty else 'phase_movement_unverified'
                results[name] = FeatureResult(name, 'no_data' if raw_phase.empty else 'blocked', reason)
            else:
                dest = output_path(output_dir, name, signal_id, date_str)
                publish_frame(mapped, name, signal_id, dest)
                results[name] = FeatureResult(name, 'success', data=mapped, output_file_path=str(dest),
                                              row_count=len(mapped), message=('Observed phases retained under eight-phase assumption.'
                                                  if assumed else 'Lane inventory mapping; historical applicability unverified.'))
        elif name.startswith('pedestrian_') and not assumed:
            results[name] = FeatureResult(name, 'blocked', 'pedestrian_mapping_unverified')
        else:
            cap = validation.capabilities.get('features', {}).get(name, {})
            reason = audit['cycle_block_reason'] or 'dependency_unavailable'
            if cap.get('valid') is False and name not in ('vehicle_cycle_profile', 'spat'):
                reason = cap['reason_code']
            # Applicability cannot stand in for a cycle the model cannot build.
            results[name] = FeatureResult(name, 'blocked', reason,
                'No cycle-based metric published when the selected phase model is unusable.')
    return results
