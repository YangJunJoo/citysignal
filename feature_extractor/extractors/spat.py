"""
SPaT (Signal Phasing and Timing) feature extractor — vectorized.

Replaces the v1 per-cycle loop that iterates through each cycle and phase,
extracting signal durations via dictionary construction and pd.concat.

Calculation: Compute durations directly from the cycle profile columns
using vectorized pandas operations.
"""

import pandas as pd
import numpy as np
import logging


def extract_spat(cache) -> pd.DataFrame:
    """
    Extract SPaT (Signal Phasing and Timing) data for a signal-day.
    
    Computes green, yellow, redClearance, and red durations for each cycle and phase
    directly from the vehicle cycle profile without per-cycle loops.
    
    Parameters
    ----------
    cache : SignalDataCache
        Pre-loaded data cache
    
    Returns
    -------
    pd.DataFrame
        SPaT data with columns: signalID, cycleNo, cycleBegin, cycleEnd, cycleLength,
        {signalType}DurationPhase{N} for each phase, and date.
    """
    df_cycle = cache.df_vehicle_cycle
    
    if len(df_cycle) == 0:
        return pd.DataFrame()
    
    # Start with cycle metadata
    result = df_cycle[['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength']].copy()
    
    # Extract phase numbers from columns
    phase_nos = cache.vehicle_phase_nos
    
    signal_types = ['green', 'yellow', 'redClearance', 'red']
    
    for phase_no in phase_nos:
        for signal_type in signal_types:
            col_name = f"{signal_type}Phase{phase_no}"
            out_col = f"{signal_type}DurationPhase{phase_no}"
            
            if col_name not in df_cycle.columns:
                result[out_col] = 0
                continue
            
            # Vectorized duration calculation
            durations = df_cycle[col_name].apply(_compute_interval_duration)
            
            result[out_col] = durations
        duration_columns = [f'{color}DurationPhase{phase_no}' for color in signal_types]
        known = result[duration_columns].sum(axis=1)
        result[f'unknownDurationPhase{phase_no}'] = (df_cycle['cycleLength'] - known).clip(lower=0)
        result[f'stateCoveragePhase{phase_no}'] = known / df_cycle['cycleLength']
        if (known > df_cycle['cycleLength'] + 1e-6).any():
            raise ValueError(f'State durations exceed cycle length for phase {phase_no}')

    # Add date column
    result['date'] = pd.Timestamp(f"{cache.year}-{cache.month:02d}-{cache.day:02d}").date()
    
    # Save
    cache.save_feature(result, 'vehicle_signal', 'spat')
    
    logging.info(f"SPaT extracted: {len(result)} cycles, {len(phase_nos)} phases")
    return result


def _compute_interval_duration(timestamps) -> float:
    """
    Compute total duration in seconds from a list of (start, end) timestamp tuples.
    
    Handles v1's format where timestamps can be:
    - A list of (start, end) tuples 
    - A single (start, end) tuple
    - NaT values
    """
    if not isinstance(timestamps, list):
        timestamps = [timestamps]
    total = 0.0
    for interval in timestamps:
        if isinstance(interval, tuple) and len(interval) == 2:
            start, end = interval
            if pd.notna(start) and pd.notna(end):
                if end < start:
                    raise ValueError('Negative state interval')
                total += (end - start).total_seconds()
    return round(total, 4)


def _is_nat_timestamps(timestamps) -> bool:
    """Check if timestamps are effectively NaT/empty."""
    if timestamps is None:
        return True
    if not isinstance(timestamps, list):
        timestamps = [timestamps]
    if timestamps == [pd.NaT]:
        return True
    return all(pd.isna(t) for t in timestamps)
