"""
test_features.py — Golden regression tests for V2 feature extractors.

Compares V2 extraction output against V1 golden data for signal 1050, 2025-09-13.
This date was verified as a 11/11 perfect V1≡V2 match.

Usage:
    pytest tests/test_features.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _compare_feature(v2_df, v1_df, feature_name, tol=0.01):
    """Compare V2 output against V1 golden data. Returns list of mismatches."""
    mismatches = []
    
    if len(v2_df) != len(v1_df):
        mismatches.append(f"row count: v2={len(v2_df)}, v1={len(v1_df)}")
    
    n_rows = min(len(v2_df), len(v1_df))
    common_cols = sorted(set(v2_df.columns) & set(v1_df.columns))
    
    for col in common_cols:
        v1_vals = v1_df[col].iloc[:n_rows]
        v2_vals = v2_df[col].iloc[:n_rows]
        
        if v1_vals.dtype == object or np.issubdtype(v1_vals.dtype, np.datetime64):
            continue
        
        try:
            a = v1_vals.values.astype(float)
            b = v2_vals.values.astype(float)
            nan_mismatch = (np.isnan(a) | np.isnan(b)) & ~(np.isnan(a) & np.isnan(b))
            diff = np.abs(a - b)
            diff[np.isnan(a) | np.isnan(b)] = 0
            n_bad = (diff > tol).sum() + nan_mismatch.sum()
            if n_bad > 0:
                mismatches.append(f"{col}: {n_bad} mismatches, max_diff={diff.max():.4f}")
        except (ValueError, TypeError):
            pass
    
    return mismatches


class TestFeatureExtraction:
    """Test each V2 feature extractor against V1 golden output."""
    
    def test_spat(self, cache, golden_data):
        from feature_extractor.extractors.spat import extract_spat
        v2 = extract_spat(cache)
        mm = _compare_feature(v2, golden_data['spat'], 'spat')
        assert not mm, f"spat mismatches: {mm}"
    
    def test_volume(self, cache, golden_data):
        from feature_extractor.extractors.volume import extract_volume
        v2 = extract_volume(cache)
        mm = _compare_feature(v2, golden_data['volume'], 'volume')
        assert not mm, f"volume mismatches: {mm}"
    
    def test_headway(self, cache, golden_data):
        from feature_extractor.extractors.headway import extract_headway
        v2 = extract_headway(cache)
        mm = _compare_feature(v2, golden_data['headway'], 'headway')
        assert not mm, f"headway mismatches: {mm}"
    
    def test_gap(self, cache, golden_data):
        from feature_extractor.extractors.gap import extract_gap
        v2 = extract_gap(cache)
        mm = _compare_feature(v2, golden_data['gap'], 'gap')
        assert not mm, f"gap mismatches: {mm}"
    
    def test_pet(self, cache, golden_data):
        from feature_extractor.extractors.gap import extract_gap
        v2 = extract_gap(cache, with_stopbar=False)
        mm = _compare_feature(v2, golden_data['pet'], 'pet')
        assert not mm, f"pet mismatches: {mm}"
    
    def test_occupancy(self, cache, golden_data):
        from feature_extractor.extractors.occupancy import extract_occupancy
        v2 = extract_occupancy(cache)
        mm = _compare_feature(v2, golden_data['occupancy'], 'occupancy')
        assert not mm, f"occupancy mismatches: {mm}"
    
    def test_red_running(self, cache, golden_data):
        from feature_extractor.extractors.red_running import extract_red_running
        v2 = extract_red_running(cache)
        mm = _compare_feature(v2, golden_data['red_running'], 'red_running')
        assert not mm, f"red_running mismatches: {mm}"
    
    def test_dilemma_running(self, cache, golden_data):
        from feature_extractor.extractors.red_running import extract_dilemma_running
        v2 = extract_dilemma_running(cache)
        mm = _compare_feature(v2, golden_data['dilemma_running'], 'dilemma_running')
        assert not mm, f"dilemma_running mismatches: {mm}"
    
    def test_shockwave(self, cache, golden_data):
        from feature_extractor.extractors.shockwave import extract_shockwave
        v2 = extract_shockwave(cache)
        mm = _compare_feature(v2, golden_data['shockwave'], 'shockwave')
        assert not mm, f"shockwave mismatches: {mm}"
    
    def test_pedestrian_activity(self, cache, golden_data):
        from feature_extractor.extractors.pedestrian import extract_pedestrian_activity
        v2 = extract_pedestrian_activity(cache)
        mm = _compare_feature(v2, golden_data['pedestrian_activity'], 'pedestrian_activity')
        assert not mm, f"pedestrian_activity mismatches: {mm}"
    
    def test_pedestrian_delay(self, cache, golden_data):
        from feature_extractor.extractors.pedestrian import extract_pedestrian_delay
        v2 = extract_pedestrian_delay(cache)
        mm = _compare_feature(v2, golden_data['pedestrian_delay'], 'pedestrian_delay')
        assert not mm, f"pedestrian_delay mismatches: {mm}"


class TestDataLoader:
    """Test SignalDataCache data loading and validation."""
    
    def test_cache_loads_data(self, cache):
        assert len(cache.df_event) > 0, "No events loaded"
        assert len(cache.df_vehicle_cycle) > 0, "No cycle profile loaded"
    
    def test_required_event_columns(self, cache):
        required = {'timeStamp', 'eventCode', 'eventParam'}
        assert required.issubset(set(cache.df_event.columns))
    
    def test_cycle_begin_sorted(self, cache):
        if 'cycleBegin' in cache.df_vehicle_cycle.columns:
            assert cache.df_vehicle_cycle['cycleBegin'].is_monotonic_increasing
    
    def test_signal_type_intervals(self, cache):
        intervals = cache.build_signal_type_intervals()
        assert len(intervals) > 0, "No signal type intervals built"
        assert set(intervals.columns) == {
            'cycle_idx', 'phase_no', 'signal_type', 'interval_start', 'interval_end'
        }
    
    def test_merged_events_have_config(self, cache):
        merged = cache.get_merged_events((81,), 'front')
        if len(merged) > 0:
            assert 'phaseNo' in merged.columns
            assert 'channelNo' in merged.columns


class TestSequenceID:
    """Test _add_event_sequence_id determinism and correctness."""
    
    def test_deterministic_sort(self, cache):
        """Same input should always produce the same sequence IDs."""
        from feature_extractor.extractors.gap import _add_event_sequence_id
        events = cache.get_merged_events((81, 82), 'front')
        if len(events) == 0:
            pytest.skip("No front detector events")
        
        # Pick first channel group
        first_key = list(events.groupby(['phaseNo', 'channelNo']).groups.keys())[0]
        grp = events.groupby(['phaseNo', 'channelNo']).get_group(first_key)
        
        ids1 = _add_event_sequence_id(grp.copy(), [81, 82])
        ids2 = _add_event_sequence_id(grp.copy(), [81, 82])
        assert ids1 == ids2, "Non-deterministic sequence ID assignment"
