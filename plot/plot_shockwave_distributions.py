#!/usr/bin/env python3
"""
Script to generate distribution plots for shockwave feature Parquet files.

Usage:
    python plot_shockwave_distributions.py --signalID ORC-0003 --year 2025 --month 9 --day 7
"""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame


import argparse
import os
import sys
import math
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def construct_parquet_path(signal_id: str, year: int, month: int, day: int, 
                         base_dir: str = "/data2/ATSPM/processed_data/fdot_d5/feature_extraction") -> str:
    """
    Construct the path to the shockwave Parquet file.
    
    Parameters:
    -----------
    signal_id : str
        Signal ID (e.g., 'ORC-0003')
    year : int
        Year (e.g., 2025)
    month : int
        Month (1-12)
    day : int
        Day (1-31)
    base_dir : str
        Base directory for processed data
        
    Returns:
    --------
    str
        Full path to the Parquet file
    """
    date_str = f"{year}-{month:02d}-{day:02d}"
    parquet_path = os.path.join(
        base_dir,
        "feature",
        "cycle",
        "vehicle_traffic",
        "shockwave",
        signal_id,
        f"{date_str}.parquet"
    )
    return parquet_path


def generate_distribution_plots(parquet_path: str, output_dir: str = "plot") -> str:
    """
    Generate distribution plots for all variables in a Parquet file.
    
    Parameters:
    -----------
    parquet_path : str
        Path to the Parquet file
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    str
        Path to the output HTML file
    """
    # Load the Parquet file
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    df = read_frame(parquet_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract signal ID and date from path for output naming
    path_parts = parquet_path.split(os.sep)
    signal_id = path_parts[-2] if len(path_parts) >= 2 else "unknown"
    date_str = os.path.splitext(path_parts[-1])[0] if path_parts else "unknown"
    
    # Prepare traces
    traces = []
    
    for col in df.columns:
        series = df[col]
        title = col
        trace = None
        
        if is_datetime64_any_dtype(series):
            data = series.dropna()
            if len(data) > 0:
                trace = go.Histogram(
                    x=data, 
                    nbinsx=min(40, max(10, int(len(data) ** 0.5))), 
                    name=title, 
                    showlegend=False
                )
                trace.update(hovertemplate='%{x|%Y-%m-%d %H:%M:%S}<extra></extra>')
            else:
                print(f"Warning: Column {col} has no valid datetime data")
                continue
                
        elif is_numeric_dtype(series):
            data = series.dropna()
            if len(data) > 0:
                trace = go.Histogram(
                    x=data, 
                    nbinsx=min(60, max(10, int(len(data) ** 0.5))), 
                    name=title, 
                    showlegend=False
                )
                trace.update(hovertemplate='%{x:.3f}<extra></extra>')
            else:
                print(f"Warning: Column {col} has no valid numeric data")
                continue
        else:
            # Try to convert to numeric
            numeric = pd.to_numeric(series, errors='coerce')
            if numeric.notna().sum() > 0:
                data = numeric.dropna()
                trace = go.Histogram(
                    x=data, 
                    nbinsx=min(60, max(10, int(len(data) ** 0.5))), 
                    name=title, 
                    showlegend=False
                )
                trace.update(hovertemplate='%{x:.3f}<extra></extra>')
            else:
                # Categorical data
                counts = series.fillna('NA').astype(str).value_counts().sort_values(ascending=False)
                if len(counts) > 0:
                    top_n = counts.head(20)
                    trace = go.Bar(
                        x=top_n.index.tolist(), 
                        y=top_n.values.tolist(), 
                        name=title, 
                        showlegend=False
                    )
                    trace.update(hovertemplate='%{x}: %{y}<extra></extra>')
                else:
                    print(f"Warning: Column {col} has no valid data")
                    continue
        
        if trace is not None:
            traces.append((title, trace))
    
    if len(traces) == 0:
        raise ValueError("No valid traces generated from the data")
    
    # Create subplots
    cols = 3
    rows = math.ceil(len(traces) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[t[0] for t in traces])
    
    for idx, (_, trace) in enumerate(traces):
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=rows * 350, 
        width=1500, 
        title_text=f'{signal_id} {date_str} Feature Distributions', 
        bargap=0.05, 
        template='plotly_white'
    )
    
    # Save HTML file
    os.makedirs(output_dir, exist_ok=True)
    output_html = os.path.join(output_dir, f'shockwave_{signal_id}_{date_str}_distributions.html')
    fig.write_html(output_html)
    print(f'Distribution plot saved to {output_html}')
    
    return output_html


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate distribution plots for shockwave feature Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_shockwave_distributions.py --signalID ORC-0003 --year 2025 --month 9 --day 7
  python plot_shockwave_distributions.py --signalID ORC-0003 --year 2025 --month 9 --day 8 --base-dir /custom/path
        """
    )
    
    parser.add_argument(
        '--signalID',
        type=str,
        required=True,
        help='Signal ID (e.g., ORC-0003)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Year (e.g., 2025)'
    )
    
    parser.add_argument(
        '--month',
        type=int,
        required=True,
        help='Month (1-12)'
    )
    
    parser.add_argument(
        '--day',
        type=int,
        required=True,
        help='Day (1-31)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/data2/ATSPM/processed_data/fdot_d5/feature_extraction',
        help='Base directory for processed data (default: /data2/ATSPM/processed_data/fdot_d5/feature_extraction)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plot',
        help='Output directory for plots (default: plot)'
    )
    
    args = parser.parse_args()
    
    # Validate date
    try:
        import datetime
        datetime.date(args.year, args.month, args.day)
    except ValueError as e:
        print(f"Error: Invalid date - {e}", file=sys.stderr)
        sys.exit(1)
    
    # Construct pickle path
    parquet_path = construct_parquet_path(
        args.signalID, 
        args.year, 
        args.month, 
        args.day,
        args.base_dir
    )
    
    print(f"Looking for Parquet file: {parquet_path}")
    
    # Generate plots
    try:
        output_html = generate_distribution_plots(parquet_path, args.output_dir)
        print(f"\nSuccess! Generated distribution plot:")
        print(f"  HTML: {output_html}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error generating plots: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

