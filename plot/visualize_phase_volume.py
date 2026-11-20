"""
Visualize phase volume data for a specific phase number.

This script loads cycle-level vehicle traffic volume data, filters for a specific phase,
and visualizes the volume over time (by hour of day).

Usage:
    python plot/visualize_phase_volume.py \
        --file /data2/ATSPM/processed_data/fdot_d5/feature_extraction/feature/cycle/vehicle_traffic/volume/ORC-0003/2025-09-07.parquet \
        --phase 2
    
Note: Requires compatible NumPy/pandas/matplotlib environment.
"""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
import os
import re


def load_volume_data(file_path):
    """Load volume data from Parquet file."""
    return read_frame(file_path)


def find_phase_volume_columns(df, phase_no):
    """
    Find columns related to phase volume.
    
    Looks for columns like:
    - Phase2, Phase2_volume, volume_Phase2, etc.
    - Any column containing both 'phase' (case-insensitive) and the phase number
    """
    phase_cols = []
    phase_pattern = re.compile(r'phase\s*' + str(phase_no) + r'|Phase\s*' + str(phase_no), re.IGNORECASE)
    
    for col in df.columns:
        # Check if column contains phase number
        if phase_pattern.search(col):
            phase_cols.append(col)
        # Also check for exact matches like "Phase2", "Phase 2", etc.
        if re.match(rf'^Phase\s*{phase_no}$|^phase\s*{phase_no}$', col, re.IGNORECASE):
            phase_cols.append(col)
    
    return phase_cols


def visualize_phase_volume(file_path, phase_no=2, output_dir=None):
    """
    Visualize phase volume for a specific phase number.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
    phase_no : int
        Phase number to filter (default: 2)
    output_dir : str
        Directory to save the output image (default: plot/results)
    """
    # Load data
    print(f"Loading data from: {file_path}")
    data = load_volume_data(file_path)
    
    # Convert to DataFrame if it's not already
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # If it's a dict, try to find the DataFrame or convert
        if len(data) == 1:
            df = list(data.values())[0]
            if not isinstance(df, pd.DataFrame):
                print(f"Warning: Data is a dict with non-DataFrame values. Keys: {list(data.keys())}")
                df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
    else:
        print(f"Warning: Unexpected data type: {type(data)}")
        df = pd.DataFrame(data)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:20]}...")  # Show first 20 columns
    
    # Find phase-related volume columns
    phase_cols = find_phase_volume_columns(df, phase_no)
    print(f"\nFound phase {phase_no} columns: {phase_cols}")
    
    if not phase_cols:
        # Try alternative: look for columns containing volume and check all columns
        print("\nSearching for volume-related columns...")
        all_volume_cols = [col for col in df.columns if 'volume' in col.lower() or 'Volume' in col]
        print(f"All volume columns: {all_volume_cols[:20]}")
        
        # If still no match, show all columns for debugging
        print(f"\nAll columns in dataset:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        raise ValueError(f"No columns found matching phase {phase_no}")
    
    # Try to find a time column (cycleBegin, date, timestamp, etc.)
    time_cols = ['cycleBegin', 'date', 'timestamp', 'time', 'datetime', 'startTime', 'beginTime']
    time_col = None
    for tc in time_cols:
        if tc in df.columns:
            time_col = tc
            break
    
    if time_col is None:
        print("Warning: No time column found. Using index as time.")
        df['time'] = df.index
        time_col = 'time'
    else:
        print(f"Using time column: {time_col}")
    
    # Convert time column to datetime if it's not already
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df['datetime'] = df[time_col]
    else:
        try:
            df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        except:
            df['datetime'] = pd.to_datetime(df.index, errors='coerce')
    
    # Extract hour and minute from datetime
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute + df['datetime'].dt.second / 60.0
    
    # Create figure with grid layout similar to visualize_phase_assignment.py
    # X-axis: Minutes (0-60), Y-axis: Hour of day (0-23)
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Get primary phase volume column
    if not phase_cols:
        raise ValueError(f"No phase {phase_no} columns found")
    
    primary_col = phase_cols[0]  # Use first found column
    
    if primary_col not in df.columns:
        raise ValueError(f"Column {primary_col} not found in dataframe")
    
    # Filter valid data
    valid_df = df[df[primary_col].notna()].copy()
    
    if len(valid_df) == 0:
        print(f"Warning: No valid volume data found for phase {phase_no}")
        return
    
    # Create scatter plot with volume as color/size
    scatter = ax.scatter(valid_df['minute'], valid_df['hour'], 
                        c=valid_df[primary_col], 
                        s=100,  # Size of points
                        alpha=0.7,
                        cmap='YlOrRd',  # Yellow-Orange-Red colormap
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Volume', fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Minutes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hour of Day', fontsize=14, fontweight='bold')
    ax.set_title(f'Phase {phase_no} Volume by Time of Day\n(X-axis: Minutes, Y-axis: Hour)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}' for h in range(24)])
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 5))
    ax.set_xticklabels(range(0, 61, 5))
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    plt.tight_layout()
    
    # Print summary statistics
    hourly_volume = valid_df.groupby('hour')[primary_col].agg(['sum', 'mean', 'count']).reset_index()
    print(f"\n{'='*60}")
    print(f"Phase {phase_no} Volume Summary Statistics")
    print(f"{'='*60}")
    print(f"Total volume for the day: {hourly_volume['sum'].sum():.0f}")
    print(f"Average volume per hour: {hourly_volume['mean'].mean():.2f}")
    print(f"Peak hour: {hourly_volume.loc[hourly_volume['sum'].idxmax(), 'hour']:02d}:00")
    print(f"Peak hour volume: {hourly_volume['sum'].max():.0f}")
    print(f"Minimum hour volume: {hourly_volume['sum'].min():.0f}")
    print(f"Total cycles with data: {len(valid_df)}")
    print(f"{'='*60}\n")
    
    plt.tight_layout()
    
    # Determine output path
    if output_dir is None:
        output_dir = '/home/ya195253/YJ/python_scripts/ATSPM/plot/results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract date from filename or use current date
    base_name = os.path.basename(file_path).replace('.parquet', '')
    signal_id = os.path.basename(os.path.dirname(file_path))
    output_path = os.path.join(output_dir, f'{base_name}_{signal_id}_phase{phase_no}_volume.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize phase volume data')
    parser.add_argument('--file', type=str, 
                       default='/data2/ATSPM/processed_data/fdot_d5/feature_extraction/feature/cycle/vehicle_traffic/volume/ORC-0003/2025-09-07.parquet',
                       help='Path to Parquet file')
    parser.add_argument('--phase', type=int, default=2,
                       help='Phase number to visualize (default: 2)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ya195253/YJ/python_scripts/ATSPM/plot/results',
                       help='Directory to save output image')
    
    args = parser.parse_args()
    
    visualize_phase_volume(args.file, args.phase, args.output_dir)


if __name__ == "__main__":
    main()

