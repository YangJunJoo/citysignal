"""Separate observed phase identity, lane inventory and dated controller evidence.

A numerical phase is not a movement, and a lane inventory is not a ring plan.
No default barrier assignment is admissible for checked production outputs.
"""
from datetime import date
import hashlib
from pathlib import Path

import pandas as pd

MAPPING_POLICY_VERSION = 'movement-and-controller-evidence-1'


def movement_inventory(frame):
    """Only identify phases for which every listed row has a direction and type.

    Several lane types on one approach may share a phase. Multiple approaches
    require review; this helper does not infer that they are equivalent.
    """
    if not {'phaseNo', 'approach', 'laneType'}.issubset(frame):
        return {}, {}
    mapped, unresolved = {}, {}
    for phase, group in frame.dropna(subset=['phaseNo']).groupby('phaseNo'):
        phase = int(phase)
        if group[['approach', 'laneType']].isna().any().any():
            unresolved[phase] = 'incomplete_movement_identity'
        elif group.approach.nunique() != 1:
            unresolved[phase] = 'multiple_approaches_require_review'
        else:
            cols = [c for c in ('ID', 'approach', 'laneType', 'laneNoFromLeft') if c in group]
            mapped[phase] = group[cols].drop_duplicates().astype(object).where(group[cols].drop_duplicates().notna(), None).to_dict('records')
    return mapped, unresolved


def controller_evidence(validation, signal_id, date_str):
    """Validate an explicitly reviewed, dated record; never infer one from logs.

    The record is supplied as decision.controller_mapping. The current builder
    supports two barrier groups only. More complex plans remain unsupported.
    """
    record = (getattr(validation, 'decision', None) or {}).get('controller_mapping')
    if not record:
        return {}, 'barrier_mapping_unverified', {}
    evidence = dict(record)
    try:
        assert record['signal_id'] == str(signal_id)
        assert record['reviewed_by'].strip() and record['reviewed_at'].strip()
        assert record['cycle_method'] == 'two_barrier_observed_transitions'
        assert record['config_sha256'] == validation.config_hash
        first, last, target = (date.fromisoformat(record['valid_from']),
                               date.fromisoformat(record['valid_to']), date.fromisoformat(date_str))
        assert first <= last
        if not first <= target <= last:
            return {}, 'controller_mapping_outside_period', evidence
        source = Path(record['source_path'])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != record['source_sha256']:
            return {}, 'controller_mapping_stale', evidence
        barriers = {int(p): int(b) for p, b in record['barrier_map'].items()}
        assert len(barriers) == len(record['barrier_map']) and set(barriers.values()) == {1, 2}
        assert all(str(p) == str(key) and p > 0 and str(b) == str(value)
                   for (key, value), (p, b) in zip(record['barrier_map'].items(), barriers.items()))
        rings = record['ring_sequences']
        assert isinstance(rings, list) and len(rings) in (1, 2)
        phases = [p for ring in rings for p in ring]
        assert all(type(p) is int and p > 0 for p in phases)
        assert len(phases) == len(set(phases)) and set(phases) == set(barriers)
        for ring in rings:
            assert ring and set(barriers[p] for p in ring) == {1, 2}
            sides = [barriers[p] for p in ring]
            assert sum(a != b for a, b in zip(sides, sides[1:] + sides[:1])) == 2
        return barriers, '', evidence
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError, OSError):
        return {}, 'controller_mapping_invalid', evidence


def assess_mapping(validation, signal_id, date_str, observed_phases):
    movements, unresolved = movement_inventory(validation.dataframe)
    observed = set(map(int, observed_phases))
    unknown = sorted(observed - set(movements))
    barriers, reason, evidence = controller_evidence(validation, signal_id, date_str)
    if not reason and unknown:
        reason = 'phase_movement_unverified'
    if not reason and observed - set(barriers):
        reason = 'phase_barrier_unverified'
    return dict(mapping_policy_version=MAPPING_POLICY_VERSION,
                signal_id=str(signal_id), date=date_str,
                config_source=getattr(validation, 'path', None),
                config_sha256=getattr(validation, 'config_hash', None),
                movement_mapping_status='lane_inventory_only',
                lane_mapping_effective_date_verified=False,
                movement_mappings=movements,
                unresolved_config_phases=unresolved,
                observed_phases=sorted(observed), unmapped_observed_phases=unknown,
                configured_without_observed_phase=sorted(set(movements) - observed),
                barrier_map=barriers, controller_evidence=evidence,
                barrier_mapping_verified=not bool(reason), cycle_block_reason=reason,
                pedestrian_mapping_status='unverified',
                coverage_complete=False)
