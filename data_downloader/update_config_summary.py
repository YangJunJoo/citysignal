#!/usr/bin/env python3
"""
Update Config Summary Script
Updates the config_summary.csv file with signal information from downloaded configs or NOEMI.

This script can:
1. Extract signal IDs and SIIA IDs from downloaded intersection config files
2. Update the config_summary.csv file with new or updated entries
3. Merge with existing config_summary if available

Example usage:
conda activate atspm & python data_downloader/update_config_summary.py --config-dir data/signal_config_Dec_2025
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Support both direct script execution and python -m invocation.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from data_downloader.signal_config import load_config_summary


def extract_signal_info_from_configs(config_dir):
    """
    Extract signal information from downloaded intersection config files.
    
    Parameters:
    -----------
    config_dir : Path
        Directory containing downloaded config files (with intersection/ subdirectory)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with signal information extracted from config files
    """
    intersection_dir = config_dir / "intersection"
    
    if not intersection_dir.exists():
        logger.warning(f"Intersection directory not found: {intersection_dir}")
        return pd.DataFrame()
    
    signal_data = []
    
    # Get all intersection CSV files
    intersection_files = list(intersection_dir.glob("*.csv"))
    logger.info(f"Found {len(intersection_files)} intersection config files")
    
    for config_file in intersection_files:
        try:
            # Signal ID is the filename without extension
            signal_id = config_file.stem
            
            # Read the intersection config file
            # Intersection files are typically transposed (Field, Value format)
            df = pd.read_csv(config_file)
            
            # Extract SIIA ID and other info if available
            siia_id = None
            name = None
            global_id = None
            
            # Check if it's in transposed format (Field, Value)
            if 'Field' in df.columns and 'Value' in df.columns:
                df_transposed = df.set_index('Field').T
                
                # Try to extract SIIA ID
                if 'SIIA ID' in df_transposed.columns:
                    siia_id = df_transposed['SIIA ID'].iloc[0]
                elif 'SIIAID' in df_transposed.columns:
                    siia_id = df_transposed['SIIAID'].iloc[0]
                
                # Extract name/intersection name
                if 'Intersection Name' in df_transposed.columns:
                    name = df_transposed['Intersection Name'].iloc[0]
                elif 'Name' in df_transposed.columns:
                    name = df_transposed['Name'].iloc[0]
                
                # Extract global ID if available
                if 'Global ID' in df_transposed.columns:
                    global_id = df_transposed['Global ID'].iloc[0]
                    
            # If not transposed, check for direct columns
            elif 'siiaID' in df.columns or 'SIIA ID' in df.columns:
                siia_col = 'siiaID' if 'siiaID' in df.columns else 'SIIA ID'
                siia_id = df[siia_col].iloc[0] if len(df) > 0 else None
            
            signal_data.append({
                'globalID': global_id if global_id else f"770{signal_id}",  # Generate if not found
                'signalID': signal_id,
                'siiaID': siia_id if siia_id else None,
                'name': name if name else f"Signal {signal_id}",
                'detour': 'Yes',  # Default
                'icmPriority': None,
                'dataCollectionStatus': 'Unknown',
                'reconfigurationStatus': 'N/A',
                'latitude': None,
                'longitude': None
            })
            
        except Exception as e:
            logger.warning(f"Error processing {config_file}: {e}")
            continue
    
    if signal_data:
        df_new = pd.DataFrame(signal_data)
        logger.info(f"Extracted information for {len(df_new)} signals")
        return df_new
    else:
        logger.warning("No signal information extracted")
        return pd.DataFrame()


def update_config_summary(config_dir, output_path=None, merge_with_existing=True):
    """
    Update config_summary.csv with signal information from downloaded configs.
    
    Parameters:
    -----------
    config_dir : Path
        Directory containing downloaded config files
    output_path : Path, optional
        Path to save updated config_summary.csv. If None, saves to the canonical ATSPM config_data directory.
    merge_with_existing : bool
        If True, merge with existing config_summary.csv. If False, create new file.
    """
    # Extract signal info from downloaded configs
    df_new = extract_signal_info_from_configs(config_dir)
    
    if df_new.empty:
        logger.error("No signal information extracted. Cannot update config_summary.")
        return False
    
    # Try to load existing config_summary
    df_existing = pd.DataFrame()
    if merge_with_existing:
        try:
            df_existing = load_config_summary(converters={"signalID": str})
            logger.info(f"Loaded existing config_summary with {len(df_existing)} entries")
        except Exception as e:
            logger.info(f"Could not load existing config_summary: {e}. Creating new file.")
    
    # Merge dataframes
    if not df_existing.empty:
        # Merge on signalID, keeping existing data but updating with new info where available
        # Convert signalID to string for consistent merging
        df_existing['signalID'] = df_existing['signalID'].astype(str)
        df_new['signalID'] = df_new['signalID'].astype(str)
        
        # Merge: keep existing entries, add new ones, update siiaID if missing in existing
        df_merged = df_existing.copy()
        
        # Update existing entries with new siiaID if missing
        for idx, row in df_new.iterrows():
            signal_id = str(row['signalID'])
            existing_idx = df_merged[df_merged['signalID'] == signal_id].index
            
            if len(existing_idx) > 0:
                # Update existing entry if siiaID is missing
                if pd.isna(df_merged.loc[existing_idx[0], 'siiaID']) and pd.notna(row['siiaID']):
                    df_merged.loc[existing_idx[0], 'siiaID'] = row['siiaID']
                    logger.debug(f"Updated siiaID for signal {signal_id}")
            else:
                # Add new entry
                df_merged = pd.concat([df_merged, row.to_frame().T], ignore_index=True)
                logger.debug(f"Added new entry for signal {signal_id}")
        
        df_final = df_merged
    else:
        df_final = df_new
    
    # Sort by signalID
    df_final['signalID'] = df_final['signalID'].astype(str)
    df_final = df_final.sort_values('signalID', key=lambda values: values.str.casefold(), kind='stable').reset_index(drop=True)
    
    # Determine output path
    if output_path is None:
        output_path = Path(os.environ.get(
            "ATSPM_CONFIG_SUMMARY_PATH",
            "/data2/ATSPM/config_data/config_summary.csv",
        ))
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save updated config_summary
    df_final.to_csv(output_path, index=False)
    logger.info(f"Saved updated config_summary to {output_path}")
    logger.info(f"Total entries: {len(df_final)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Config Summary Update Summary:")
    print(f"  Total entries: {len(df_final)}")
    if not df_existing.empty:
        print(f"  Existing entries: {len(df_existing)}")
        print(f"  New entries: {len(df_final) - len(df_existing)}")
    print(f"  Output file: {output_path}")
    print("=" * 60)
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Update config_summary.csv with signal information from downloaded configs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update config_summary from downloaded configs
  python update_config_summary.py --config-dir data/signal_config_Dec_2025
  
  # Create new config_summary without merging
  python update_config_summary.py --config-dir data/signal_config_Dec_2025 --no-merge
  
  # Save to custom location
  python update_config_summary.py --config-dir data/signal_config_Dec_2025 --output data/config_summary_new.csv
        """
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        required=True,
        help='Directory containing downloaded signal config files (with intersection/ subdirectory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for config_summary.csv (default: /data2/ATSPM/config_data/config_summary.csv)'
    )
    
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='Do not merge with existing config_summary.csv (create new file)'
    )
    
    args = parser.parse_args()
    
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else None
    
    update_config_summary(
        config_dir=config_dir,
        output_path=output_path,
        merge_with_existing=not args.no_merge
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpdate interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

