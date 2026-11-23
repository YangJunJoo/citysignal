#!/usr/bin/env python3
"""
Signal Configuration Downloader Script
Downloads signal configuration data from NOEMI reports for all available signals.

Example usage:
conda activate atspm & python data_downloader/download_signal_configs.py
"""

import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Support both direct script execution and python -m invocation.
project_root = Path(__file__).resolve().parents[1]
script_dir = str(Path(__file__).resolve().parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from data_downloader.signal_config import load_config_summary
from data_downloader.signal_config import scrape_config_report

config_root = Path(os.environ.get("ATSPM_CONFIG_ROOT", "/data2/ATSPM/config_data"))


def get_all_signal_mappings():
    """
    Load signal ID to SIIA ID mapping from config_summary.csv.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: signalID, siiaID
    """
    try:
        df = load_config_summary()
        logger.info(f"Loaded {len(df)} signal mappings from config_summary.csv")
        return df[['signalID', 'siiaID']].dropna()
    except Exception as e:
        logger.error(f"Failed to load config_summary: {e}")
        raise


def download_signal_config(signal_id, siia_id, output_base_dir, max_retries=3):
    """
    Download configuration data for a single signal.
    
    Parameters:
    -----------
    signal_id : str
        Signal ID
    siia_id : str
        SIIA ID for accessing NOEMI report
    output_base_dir : Path
        Base output directory
    max_retries : int
        Maximum number of retry attempts
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    for attempt in range(1, max_retries + 1):
        try:
            saved = scrape_config_report(
                siia_id=str(siia_id), signal_id=str(signal_id), output_dir=output_base_dir)
            if saved:
                logger.debug(f"Downloaded {len(saved)} config files for signal {signal_id}")
                return True
            logger.warning(f"No config files found for signal {signal_id} after scraping")
            return False

        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Attempt {attempt}/{max_retries} failed for signal {signal_id}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download config for signal {signal_id} after {max_retries} attempts: {e}")
                return False
    
    return False


def main(output_dir_path=None):
    """Main function to download all signal configurations.
    
    Parameters:
    -----------
    output_dir_path : str or Path, optional
        Custom output directory path. If None, uses default.
    """
    # Configuration
    if output_dir_path:
        output_dir = Path(output_dir_path)
    else:
        output_dir = config_root / "signal_config_Dec_2025"
    
    print("=" * 60)
    print("Signal Configuration Downloader")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load signal mappings
    try:
        df_mappings = get_all_signal_mappings()
    except Exception as e:
        logger.error(f"Failed to load signal mappings: {e}")
        return
    
    # Convert signalID and siiaID to string for consistency
    df_mappings['signalID'] = df_mappings['signalID'].astype(str)
    df_mappings['siiaID'] = df_mappings['siiaID'].astype(str)
    
    # Remove duplicates (keep first occurrence)
    df_mappings = df_mappings.drop_duplicates(subset=['signalID'], keep='first')
    
    total_signals = len(df_mappings)
    print(f"Total signals to download: {total_signals}")
    print("-" * 60)
    
    # Statistics
    successful_downloads = 0
    failed_downloads = 0
    skipped_signals = []
    
    # Process each signal
    for idx, row in tqdm(df_mappings.iterrows(), total=total_signals, desc="Downloading configs"):
        signal_id = str(row['signalID'])
        siia_id = str(row['siiaID'])
        
        # Check if all config files already exist
        table_ids = ["intersection", "controller", "approach", "lanes", "signal"]
        all_exist = True
        for table_id in table_ids:
            config_file = output_dir / table_id / f"{signal_id}.csv"
            if not config_file.exists():
                all_exist = False
                break
        
        if all_exist:
            skipped_signals.append(signal_id)
            continue
        
        # Download configuration
        if download_signal_config(signal_id, siia_id, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Small delay to avoid overwhelming the server
        time.sleep(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Failed downloads: {failed_downloads}")
    print(f"  Skipped (already exist): {len(skipped_signals)}")
    print(f"  Total signals processed: {total_signals}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    if skipped_signals:
        print(f"\nSkipped signals (already downloaded): {len(skipped_signals)}")
        if len(skipped_signals) <= 20:
            print(f"  {', '.join(skipped_signals)}")
        else:
            print(f"  First 20: {', '.join(skipped_signals[:20])}...")
            print(f"  ... and {len(skipped_signals) - 20} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download signal configuration data from NOEMI reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all signal configs to default directory
  python download_signal_configs.py
  
  # Download to custom directory
  python download_signal_configs.py --output-dir data/signal_config_Dec_2025
  
  # Update config_summary after downloading
  python download_signal_configs.py --update-summary
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for downloaded configs (default: /data2/ATSPM/config_data/signal_config_Dec_2025)'
    )
    
    parser.add_argument(
        '--update-summary',
        action='store_true',
        help='Update config_summary.csv after downloading'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine output directory
        output_dir = args.output_dir if args.output_dir else None
        main(output_dir_path=output_dir)
        
        # Update config_summary if requested
        if args.update_summary:
            print("\n" + "=" * 60)
            print("Updating config_summary.csv...")
            print("=" * 60)
            
            try:
                # Import update function
                update_script_path = os.path.join(script_dir, "update_config_summary.py")
                if os.path.exists(update_script_path):
                    sys.path.insert(0, script_dir)
                    from update_config_summary import update_config_summary
                    
                    # Determine output directory used
                    final_output_dir = Path(args.output_dir) if args.output_dir else config_root / "signal_config_Dec_2025"
                    
                    update_config_summary(
                        config_dir=final_output_dir,
                        merge_with_existing=True
                    )
                else:
                    logger.warning("update_config_summary.py not found. Skipping summary update.")
            except Exception as e:
                logger.error(f"Failed to update config_summary: {e}")
                final_output_dir = Path(args.output_dir) if args.output_dir else config_root / "signal_config_Dec_2025"
                print(f"\nNote: You can manually update config_summary by running:")
                print(f"  python data_downloader/update_config_summary.py --config-dir {final_output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

