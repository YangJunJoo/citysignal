"""
Visualize phase assignment for a specific phase number.

This script loads phase profile data, filters for a specific phase number and
correctSequenceFlag == 1 (correct sequences only), and visualizes the begin and end times
of green, yellow, red clearance, and red signal states grouped by hour.

Begin and end times for each signal state are shown as colored line segments. Each segment
connects a begin time to its corresponding end time with the same color, and segments break
at hour boundaries so they only connect within the same hour.

X-axis: Minutes (0-60)
Y-axis: Hour of day (0-23)

Usage:
    python plot/visualize_phase_assignment.py \
        --file /data2/ATSPM/processed_data/fdot_d5/feature_extraction/signal_profile/phase/vehicle_signal/ORC-0003/2025-09-07.parquet \
        --phase 2
    
Note: Requires compatible NumPy/pandas/matplotlib environment.
If you encounter NumPy compatibility errors, try using a conda environment
or virtual environment with compatible package versions.
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


def load_phase_data(file_path):
    """Load phase data from Parquet file."""
    return read_frame(file_path)


def visualize_phase_assignment(file_path, phase_no=2, output_dir=None):
    """
    Visualize phase assignment for a specific phase number.
    
    Parameters:
    -----------
    file_path : str
        Path to the Parquet file
    phase_no : int
        Phase number to filter (default: 2)
    output_dir : str
        Directory to save the output image (default: same as input file)
    """
    # Load data
    print(f"Loading data from: {file_path}")
    df = load_phase_data(file_path)
    
    # Filter for specified phase
    df_phase = df[df['phaseNo'] == phase_no].copy()
    print(f"Filtered data for phase {phase_no}: {len(df_phase)} rows")
    
    # Filter for correctSequenceFlag == 1 (only show correct sequences)
    if 'correctSequenceFlag' in df_phase.columns:
        df_phase = df_phase[df_phase['correctSequenceFlag'] == 1].copy()
        print(f"Filtered data for correctSequenceFlag == 1: {len(df_phase)} rows")
    else:
        print("Warning: correctSequenceFlag column not found in data")
    
    if len(df_phase) == 0:
        print(f"No data found for phase {phase_no}")
        return
    
    # Convert datetime columns
    datetime_cols = ['greenBegin', 'greenEnd', 'yellowBegin', 'yellowEnd', 
                     'redClearanceBegin', 'redClearanceEnd', 'redBegin', 'redEnd']
    
    for col in datetime_cols:
        if col in df_phase.columns:
            df_phase[col] = pd.to_datetime(df_phase[col], errors='coerce')
    
    # Sort by greenBegin
    df_phase = df_phase.sort_values('greenBegin').reset_index(drop=True)
    
    # Extract hour and minute from greenBegin
    df_phase['hour'] = df_phase['greenBegin'].dt.hour
    df_phase['minute'] = df_phase['greenBegin'].dt.minute
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Signal state colors
    state_colors = {
        'green': 'green',
        'yellow': 'yellow',
        'redClearance': 'orange',
        'red': 'red'
    }
    
    # State column mappings
    state_columns = {
        'green': ('greenBegin', 'greenEnd'),
        'yellow': ('yellowBegin', 'yellowEnd'),
        'redClearance': ('redClearanceBegin', 'redClearanceEnd'),
        'red': ('redBegin', 'redEnd')
    }
    
    # Collect begin and end time pairs for each state
    state_intervals = {
        'green': [],
        'yellow': [],
        'redClearance': [],
        'red': []
    }
    
    # Collect begin and end times for each state
    for idx, row in df_phase.iterrows():
        for state_name, (begin_col, end_col) in state_columns.items():
            if pd.notna(row[begin_col]) and pd.notna(row[end_col]):
                begin_min = row[begin_col].minute + row[begin_col].second / 60.0
                end_min = row[end_col].minute + row[end_col].second / 60.0
                begin_hour = row[begin_col].hour
                end_hour = row[end_col].hour
                
                state_intervals[state_name].append({
                    'begin_time': begin_min,
                    'begin_hour': begin_hour,
                    'end_time': end_min,
                    'end_hour': end_hour
                })
    
    # Plot each phase state with lines connecting begin to end
    for state_name, state_color in state_colors.items():
        if len(state_intervals[state_name]) == 0:
            continue
        
        # Plot lines connecting begin to end for each interval
        # Lines only connect within the same hour, with temporary dots at hour boundaries
        for interval in state_intervals[state_name]:
            begin_hour = interval['begin_hour']
            end_hour = interval['end_hour']
            begin_time = interval['begin_time']
            end_time = interval['end_time']
            
            if begin_hour == end_hour:
                # Same hour: draw line directly from begin to end
                ax.plot([begin_time, end_time], 
                       [begin_hour, end_hour], 
                       color=state_color, 
                       linewidth=4, 
                       alpha=0.7,
                       zorder=1)
            else:
                # Different hours: break at hour boundaries with temporary dots
                # Line from begin to end of begin hour (minute 60)
                ax.plot([begin_time, 60.0], 
                       [begin_hour, begin_hour], 
                       color=state_color, 
                       linewidth=4, 
                       alpha=0.7,
                       zorder=1)
                # If there are intermediate hours, draw full-width lines for each
                for hour in range(begin_hour + 1, end_hour):
                    # Full-width line for intermediate hour
                    ax.plot([0.0, 60.0], 
                           [hour, hour], 
                           color=state_color, 
                           linewidth=4, 
                           alpha=0.7,
                           zorder=1)
                
                # Line from start of end hour (minute 0) to end
                ax.plot([0.0, end_time], 
                       [end_hour, end_hour], 
                       color=state_color, 
                       linewidth=4, 
                       alpha=0.7,
                       zorder=1)
                # (No markers at hour boundaries; visual separation handled by segment splits)
    
    # Formatting
    ax.set_xlabel('Minutes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hour', fontsize=14, fontweight='bold')
    ax.set_title(f'Phase {phase_no} Signal State Begin/End Times by Hour (Correct Sequences Only)\nConnected by Phase Color (Begin to End)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f'{h:02d}' for h in range(24)])
    ax.set_xlim(0, 60)
    ax.set_xticks(range(0, 61, 5))
    ax.set_xticklabels(range(0, 61, 5))
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=state_colors['green'], label='Green', alpha=0.8),
        Patch(facecolor=state_colors['yellow'], label='Yellow', alpha=0.8),
        Patch(facecolor=state_colors['redClearance'], label='Red Clearance', alpha=0.8),
        Patch(facecolor=state_colors['red'], label='Red', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Determine output path - use provided output_dir or default
    if output_dir is None:
        # Default to the specified output directory
        output_dir = '/home/ya195253/YJ/python_scripts/ATSPM/plot/results'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract date from filename or use current date
    base_name = os.path.basename(file_path).replace('.parquet', '')
    output_path = os.path.join(output_dir, f'{base_name}_phase{phase_no}_visualization.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Visualize phase assignment')
    parser.add_argument('--file', type=str, 
                       default='/data2/ATSPM/processed_data/fdot_d5/feature_extraction/signal_profile/phase/vehicle_signal/ORC-0003/2025-09-07.parquet',
                       help='Path to Parquet file')
    parser.add_argument('--phase', type=int, default=2,
                       help='Phase number to visualize (default: 2)')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ya195253/YJ/python_scripts/ATSPM/plot/results',
                       help='Directory to save output image (default: /home/ya195253/YJ/python_scripts/ATSPM/plot/results)')
    
    args = parser.parse_args()
    
    visualize_phase_assignment(args.file, args.phase, args.output_dir)


if __name__ == "__main__":
    main()
