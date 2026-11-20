"""
utils.py — Shared utility functions for ATSPM extractors.

Contains common calculations used across multiple feature extractors.
"""

import numpy as np
import pandas as pd


def calculate_signal_type_stats(df: pd.DataFrame, column_names: list,
                                 include_sum: bool = False) -> pd.DataFrame:
    """
    Calculate Min/Max/Avg/Std statistics for columns containing nested lists.
    Exact port of v1's _calculate_stats.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with list-valued columns
    column_names : list
        Column names to compute stats for (e.g., 'greenGapPhase2T')
    include_sum : bool
        If True, sum per-channel sublists before stats (occupancy).
        If False, flatten sublists (gap, headway).
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added Min/Max/Avg/Std columns
    """
    dict_stats = {}
    prefixes = ['green', 'yellow', 'redClearance', 'red']
    
    for col in column_names:
        prefix = next((p for p in prefixes if col.startswith(p)), None)
        if not prefix:
            continue
        
        min_col_name = col.replace(prefix, f"{prefix}Min")
        max_col_name = col.replace(prefix, f"{prefix}Max")
        avg_col_name = col.replace(prefix, f"{prefix}Avg")
        std_col_name = col.replace(prefix, f"{prefix}Std")
        
        if include_sum:
            df[col] = df[col].apply(
                lambda row: [
                    round(np.nansum(
                        lst if not isinstance(lst, float) or not np.isnan(lst) else [0]
                    ), 4)
                    for lst in row
                ]
            )
        else:
            df[col] = df[col].apply(
                lambda row: [item for sublist in row for item in sublist]
                if any(isinstance(sublist, list) for sublist in row) else row
            )
        
        dict_stats[min_col_name] = df[col].apply(
            lambda row: round(np.nanmin(row if not all(np.isnan(row)) else 0), 4)
        )
        dict_stats[max_col_name] = df[col].apply(
            lambda row: round(np.nanmax(row if not all(np.isnan(row)) else 0), 4)
        )
        dict_stats[avg_col_name] = df[col].apply(
            lambda row: round(np.nanmean(row if not all(np.isnan(row)) else 0), 4)
        )
        dict_stats[std_col_name] = df[col].apply(
            lambda row: round(np.nanstd(row if not all(np.isnan(row)) else 0), 4)
        )
    
    if dict_stats:
        df = pd.concat([df, pd.DataFrame(dict_stats)], axis=1)
    
    return df
