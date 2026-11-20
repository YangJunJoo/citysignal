"""Qualify an individual signal without declaring its incomplete bulk file valid."""
from numbers import Integral

SIGNAL_QUALITY_VERSION = 'signal-source-quality-1'
# Only row-local defects can be isolated with existing per-signal evidence.
# Coverage gaps, schema errors, stale sources and unknown reasons stay blocked.
LOCALIZABLE_REASONS = frozenset({'invalid_signal_id', 'invalid_timestamp'})


def _count(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def scope_source_to_signal(source, signal_id):
    result = dict(source)
    evidence = result.pop('signals', {}).get(signal_id)
    result.update(expected_signal=evidence, signal_quality_version=SIGNAL_QUALITY_VERSION,
                  source_file_status=source.get('status'), quality_scope='file')
    reasons = set(source.get('reason_codes', []))
    if (source.get('status') != 'source_incomplete' or
            source.get('reported_status') != 'source_incomplete' or
            not reasons or not reasons.issubset(LOCALIZABLE_REASONS)):
        return result
    if not signal_id or signal_id.strip().lower() in ('nan', 'null', 'none'):
        return result
    if not isinstance(evidence, dict) or not isinstance(evidence.get('hourly_counts'), dict):
        return result
    rows = evidence.get('row_count')
    hourly = list(evidence['hourly_counts'].values())
    total, assigned, unassigned = (source.get(k) for k in ('row_count', 'assigned_signal_row_count', 'invalid_signal_count'))
    if not all(_count(n) for n in (rows, total, assigned, unassigned, *hourly)):
        return result
    if rows <= 0 or sum(hourly) != rows or assigned + unassigned != total:
        return result
    if not source.get('signature') or not source.get('min_timestamp') or not source.get('max_timestamp'):
        return result
    # execute_job must still match the split's identity, row count, time bounds,
    # and each hourly count to this evidence before publishing anything.
    result.update(status='valid', quality_scope='signal',
                  qualification_reason='identified_rows_verified_unassigned_rows_quarantined',
                  excluded_unassigned_rows=unassigned,
                  expected_coverage_verified=False)
    return result
