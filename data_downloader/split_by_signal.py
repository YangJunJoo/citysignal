#!/usr/bin/env python3
"""
Split Parquet Files by Signal ID
Splits parquet files in raw_data/ directory into per-signal parquet files.
Handles variable column names, empty files, and missing data gracefully.

Example usage:
conda activate atspm && python data_downloader/split_by_signal.py

the default output directory is /data2/ATSPM/raw_data/per_signal
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import logging
try:
    import schedule
except ImportError:
    schedule = None
import time
from datetime import datetime
import json
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_cache_file_path(output_dir):
    """Get the path to the processing cache file."""
    cache_dir = Path(output_dir) / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'processed_files_cache.json'


def load_processing_cache(output_dir):
    """Load the processing status cache from disk."""
    cache_file = get_cache_file_path(output_dir)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading cache file: {e}. Starting with empty cache.")
            return {}
    return {}


def save_processing_cache(cache, output_dir):
    """Save the processing status cache to disk."""
    cache_file = get_cache_file_path(output_dir)
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logger.warning(f"Error saving cache file: {e}")


def get_file_signature(file_path):
    """Get a signature for a file (path + mtime + size) to detect changes."""
    try:
        stat = file_path.stat()
        return {
            'path': str(file_path),
            'mtime': stat.st_mtime,
            'size': stat.st_size
        }
    except OSError:
        return None


def is_valid_split_output(output_path, expected_rows=None):
    """Return True only for a readable, non-empty, complete split parquet.

    The expected row count is available while checking/splitting a bulk file
    and prevents a valid-but-truncated parquet from being mistaken for a
    complete output after an interrupted run.
    """
    try:
        parquet_file = pq.ParquetFile(str(output_path))
        metadata = parquet_file.metadata
        if metadata is None or metadata.num_row_groups == 0 or metadata.num_rows == 0:
            return False
        if expected_rows is not None and metadata.num_rows != int(expected_rows):
            return False
        return True
    except Exception:
        return False


def is_file_cached_as_processed(file_path, cache):
    """Check if file is cached as processed and hasn't changed."""
    file_sig = get_file_signature(file_path)
    if not file_sig:
        return False
    
    file_key = file_sig['path']
    if file_key not in cache:
        return False
    
    cached_sig = cache[file_key]
    # Check if file has been modified (mtime or size changed)
    if (cached_sig.get('mtime') != file_sig['mtime'] or 
        cached_sig.get('size') != file_sig['size']):
        return False
    
    # File is cached and hasn't changed
    return cached_sig.get('processed', False)


def update_cache_processed(file_path, output_dir, cache):
    """Mark a file as processed in the cache."""
    file_sig = get_file_signature(file_path)
    if file_sig:
        cache[file_sig['path']] = {
            'mtime': file_sig['mtime'],
            'size': file_sig['size'],
            'processed': True,
            'timestamp': datetime.now().isoformat()
        }
        save_processing_cache(cache, output_dir)


def find_signal_id_column(df):
    """
    Find the signalID column with flexible name matching.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    str or None
        Column name for signalID, or None if not found
    """
    # Try common variations of signalID column name
    # Priority: check for normalized form first, then other variations
    possible_names = [
        'signalID', 'SignalID', 'signal_id', 'Signal_Id',
        'SIGNALID', 'SIGNAL_ID', 'SignalId', 'signal_Id'
    ]
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    # Case-insensitive search
    df_columns_lower = [col.lower() for col in df.columns]
    for name in possible_names:
        if name.lower() in df_columns_lower:
            # Return the actual column name (preserving case)
            idx = df_columns_lower.index(name.lower())
            return df.columns[idx]
    
    return None


def normalize_column_names(df):
    """
    Normalize column names to standard camelCase format for consistency.
    
    Normalizes to: signalID, timeStamp, eventCode, eventParam
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with normalized column names
    """
    df = df.copy()
    
    # Mapping of common variations to standard camelCase names
    column_mapping = {}
    
    # signalID variations - normalize to camelCase
    signal_id_col = find_signal_id_column(df)
    if signal_id_col and signal_id_col != 'signalID':
        column_mapping[signal_id_col] = 'signalID'
    
    # timeStamp variations - normalize to camelCase
    timestamp_variations = ['TimeStamp', 'timeStamp', 'timestamp', 'TIMESTAMP', 
                           'Time_Stamp', 'time_stamp', 'Timestamp']
    for var in timestamp_variations:
        if var in df.columns and var != 'timeStamp':
            column_mapping[var] = 'timeStamp'
            break
    
    # eventCode variations - normalize to camelCase
    eventcode_variations = ['EventCode', 'eventCode', 'event_code', 'EVENTCODE',
                           'Event_Code', 'event_code', 'Event_Code']
    for var in eventcode_variations:
        if var in df.columns and var != 'eventCode':
            column_mapping[var] = 'eventCode'
            break
    
    # eventParam variations - normalize to camelCase
    eventparam_variations = ['EventParam', 'eventParam', 'event_param', 'EVENTPARAM',
                            'Event_Param', 'event_param', 'Event_Param']
    for var in eventparam_variations:
        if var in df.columns and var != 'eventParam':
            column_mapping[var] = 'eventParam'
            break
    
    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
        logger.debug(f"Normalized columns: {column_mapping}")
    
    return df


def check_file_processed(input_file, output_base_dir):
    """
    Check if an input file has already been fully processed.
    
    This checks if output files exist for all signals in the input file.
    It's a lightweight check that reads only the SignalID column.
    
    Parameters:
    -----------
    input_file : Path
        Path to input parquet file
    output_base_dir : Path
        Base directory for output (will check per_signal subdirectory)
        
    Returns:
    --------
    tuple (bool, set)
        (is_fully_processed, set of missing signal_ids)
        is_fully_processed is True if all signals have output files
    """
    try:
        # Detect the SignalID column name from schema first (fast, no data read)
        import pyarrow.parquet as pq
        schema = pq.read_schema(input_file)
        col_names = schema.names
        
        # Find signal ID column from schema
        possible_names = [
            'signalID', 'SignalID', 'signal_id', 'Signal_Id',
            'SIGNALID', 'SIGNAL_ID', 'SignalId', 'signal_Id'
        ]
        signal_id_col = None
        for name in possible_names:
            if name in col_names:
                signal_id_col = name
                break
        if signal_id_col is None:
            # Case-insensitive fallback
            col_lower = {c.lower(): c for c in col_names}
            for name in possible_names:
                if name.lower() in col_lower:
                    signal_id_col = col_lower[name.lower()]
                    break
        
        if signal_id_col is None:
            return False, set()  # Can't check without SignalID
        
        # Read ONLY the SignalID column (fast - avoids loading entire multi-GB file)
        df = pd.read_parquet(input_file, columns=[signal_id_col])
        
        if df.empty:
            logger.debug(f"{input_file.name} is empty - marking as processed")
            return True, set()
        
        # Get unique signal IDs (same logic as process_parquet_file)
        # SignalID can be string or numeric - preserve original type
        try:
            rows_before_filtering = len(df)
            
            # STEP 1: Remove rows with null SignalID (same filtering as process_parquet_file)
            df = df.dropna(subset=[signal_id_col])
            
            # STEP 2: Also filter out empty strings if SignalID is string/object type
            if df[signal_id_col].dtype == 'object' or str(df[signal_id_col].dtype) == 'string':
                df = df[df[signal_id_col].astype(str).str.strip() != '']
            
            if df.empty:
                # File has no valid signal IDs after filtering - consider it processed
                # since no outputs would be created anyway
                # But log a warning if the file originally had data
                if rows_before_filtering > 0:
                    logger.debug(f"{input_file.name} has {rows_before_filtering:,} rows but ALL signalID values are null/empty - "
                               f"all rows filtered out, marking as processed (nothing to split)")
                else:
                    logger.debug(f"{input_file.name} has no valid SignalID values after filtering")
                return True, set()
            
            # Get unique signals that have valid data (same as what would be processed)
            # Preserve original type (string or numeric)
            unique_signals = set(df[signal_id_col].unique())
            
            # Filter out any remaining invalid signals (extra safety)
            signal_row_counts = df[signal_id_col].value_counts(dropna=False).to_dict()
            # Handle both string and numeric types
            def is_valid_signal_id(sig):
                if pd.isna(sig):
                    return False
                if isinstance(sig, str):
                    return sig.strip() != ''
                return True
            
            unique_signals = {sig for sig in unique_signals if is_valid_signal_id(sig)}
            
        except Exception as e:
            logger.debug(f"Error extracting signal IDs from {input_file.name}: {e}")
            return False, set()
        
        if not unique_signals:
            # No valid signals found - consider it processed
            return True, set()
        
        # Check output directory structure
        # If output_base_dir already ends with 'per_signal', use it directly
        if output_base_dir.name == 'per_signal' or str(output_base_dir).endswith('/per_signal'):
            per_signal_dir = output_base_dir
        else:
            per_signal_dir = output_base_dir / 'per_signal'
        if not per_signal_dir.exists():
            return False, unique_signals
        
        input_stem = input_file.stem
        output_filename = f"{input_stem}.parquet"
        
        # Check which signals have output files
        # Only check for signals that would actually be processed (have valid data)
        missing_signals = set()
        processed_signals = set()
        
        for signal_id in unique_signals:
            signal_dir = per_signal_dir / str(signal_id)
            output_path = signal_dir / output_filename
            
            if output_path.exists():
                expected_rows = signal_row_counts.get(signal_id)
                if is_valid_split_output(output_path, expected_rows=expected_rows):
                    processed_signals.add(signal_id)
                else:
                    # Missing footer, zero rows, or a row-count mismatch means a
                    # previous write was incomplete and must be repaired.
                    missing_signals.add(signal_id)
            else:
                # Output file doesn't exist - needs processing
                missing_signals.add(signal_id)
        
        # A file is fully processed if all valid signals have non-empty output files
        is_fully_processed = len(missing_signals) == 0 and len(processed_signals) > 0
        
        # If we found valid signals but none have outputs, it's not processed
        if unique_signals and len(processed_signals) == 0:
            is_fully_processed = False
        
        return is_fully_processed, missing_signals
        
    except Exception as e:
        logger.debug(f"Error checking if {input_file.name} is processed: {e}")
        return False, set()


def process_parquet_file(input_file, output_base_dir, overwrite=False):
    """
    Process a single parquet file and split it by SignalID.
    
    Parameters:
    -----------
    input_file : Path
        Path to input parquet file
    output_base_dir : Path
        Base directory for output (will create per_signal subdirectory)
    overwrite : bool
        Whether to overwrite existing files
        
    Returns:
    --------
    dict
        Statistics about the processing
    """
    stats = {
        'signals_processed': 0,
        'signals_skipped': 0,
        'total_records': 0,
        'errors': []
    }
    
    try:
        # Read parquet file
        logger.info(f"Reading file: {input_file.name}")
        df = pd.read_parquet(input_file)
        
        # Check if dataframe is empty
        if df.empty:
            logger.warning(f"File {input_file.name} is empty, skipping")
            stats['errors'].append(f"Empty file: {input_file.name}")
            return stats
        
        initial_total_records = len(df)
        stats['total_records'] = initial_total_records
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Find SignalID column
        signal_id_col = find_signal_id_column(df)
        if signal_id_col is None:
            error_msg = f"No SignalID column found in {input_file.name}. Available columns: {list(df.columns)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            return stats
        
        # SignalID can be string or numeric - preserve original type
        # Try to convert to numeric if possible (for numeric SignalIDs), but keep as string if needed
        rows_before_filtering = len(df)
        
        # Check if SignalID is already numeric type
        is_already_numeric = pd.api.types.is_numeric_dtype(df[signal_id_col])
        
        if not is_already_numeric:
            # Try to convert to numeric if possible (handles numeric SignalIDs stored as strings)
            try:
                numeric_converted = pd.to_numeric(df[signal_id_col], errors='coerce')
                # Only convert if most values are numeric (more than 50% success)
                non_null_count = numeric_converted.notna().sum()
                if non_null_count > len(df) * 0.5:
                    df[signal_id_col] = numeric_converted.astype('Int64')
                    logger.debug(f"Converted SignalID to numeric type in {input_file.name}")
                else:
                    # Keep as string if mostly non-numeric
                    logger.debug(f"Keeping SignalID as string type in {input_file.name}")
            except Exception as e:
                logger.debug(f"Keeping SignalID as original type in {input_file.name}: {e}")
        else:
            logger.debug(f"SignalID is already numeric type in {input_file.name}")
        
        # STEP 1: Remove rows with null SignalID BEFORE processing
        # This is critical - we only process rows that have valid signalID values
        df = df.dropna(subset=[signal_id_col])
        
        # STEP 2: Also filter out empty strings if SignalID is string/object type
        if df[signal_id_col].dtype == 'object' or str(df[signal_id_col].dtype) == 'string':
            df = df[df[signal_id_col].astype(str).str.strip() != '']
        
        # Log how many rows were removed
        rows_after_filtering = len(df)
        rows_removed = rows_before_filtering - rows_after_filtering
        
        if rows_removed > 0:
            logger.info(f"Filtered out {rows_removed:,} rows without valid SignalID from {input_file.name} "
                       f"({rows_removed/rows_before_filtering*100:.1f}% of rows). "
                       f"Processing {rows_after_filtering:,} rows with valid SignalID.")
        
        # Only skip if there are no rows left after filtering
        if df.empty:
            if rows_before_filtering > 0:
                logger.warning(f"File {input_file.name} has {rows_before_filtering:,} rows but ALL signalID values are null/empty. "
                             f"All rows were filtered out - nothing to process.")
                stats['errors'].append(f"All rows filtered: {input_file.name} has {rows_before_filtering:,} rows but no valid SignalID values")
            else:
                logger.warning(f"No valid SignalID values in {input_file.name}, skipping")
                stats['errors'].append(f"No valid SignalID values: {input_file.name}")
            return stats
        
        # Update stats to reflect only rows that will be processed
        stats['total_records'] = rows_after_filtering
        
        # Group by SignalID
        signal_groups = df.groupby(signal_id_col)
        
        # Create output directory structure
        # If output_base_dir already ends with 'per_signal', use it directly
        # Otherwise, create per_signal subdirectory
        if output_base_dir.name == 'per_signal' or str(output_base_dir).endswith('/per_signal'):
            per_signal_dir = output_base_dir
        else:
            per_signal_dir = output_base_dir / 'per_signal'
        per_signal_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each signal group
        for signal_id, signal_df in signal_groups:
            # Skip if signal_df is empty (shouldn't happen, but be safe)
            if signal_df.empty:
                logger.warning(f"Empty group for SignalID {signal_id} in {input_file.name}")
                continue
            
            # Create signal-specific directory
            signal_dir = per_signal_dir / str(signal_id)
            signal_dir.mkdir(parents=True, exist_ok=True)
            
            # Output filename: use input filename or create date-based name
            # Extract date from filename if possible (e.g., atspm-2025-8-1.parquet)
            input_stem = input_file.stem
            output_filename = f"{input_stem}.parquet"
            output_path = signal_dir / output_filename
            
            # Skip only a complete existing output. A non-empty but truncated
            # parquet may be left by an interrupted legacy run.
            if (output_path.exists() and not overwrite and
                    is_valid_split_output(output_path, expected_rows=len(signal_df))):
                logger.debug(f"Skipping existing file: {output_path}")
                stats['signals_skipped'] += 1
                continue
            
            temp_output_path = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                # Write beside the destination and publish atomically. An
                # interrupted write therefore never appears as a final output.
                signal_df.to_parquet(
                    temp_output_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                if not is_valid_split_output(temp_output_path, expected_rows=len(signal_df)):
                    raise ValueError(f"Split parquet validation failed: {temp_output_path}")
                os.replace(temp_output_path, output_path)
                stats['signals_processed'] += 1
                logger.debug(f"Wrote {len(signal_df)} records for SignalID {signal_id} to {output_path}")
            except Exception as e:
                error_msg = f"Error writing SignalID {signal_id} from {input_file.name}: {e}"
                logger.error(error_msg)
                stats['errors'].append(error_msg)
            finally:
                if temp_output_path.exists():
                    try:
                        temp_output_path.unlink()
                    except OSError:
                        pass
        
        logger.info(f"Processed {input_file.name}: {stats['signals_processed']} signals, "
                   f"{stats['signals_skipped']} skipped, {stats['total_records']} total records")
        
    except Exception as e:
        error_msg = f"Error processing {input_file.name}: {e}"
        logger.error(error_msg, exc_info=True)
        stats['errors'].append(error_msg)
    
    return stats


def run_processing(input_dir, output_dir, overwrite, max_workers=None, input_files=None):
    """
    Run the processing logic once.
    
    Parameters:
    -----------
    input_dir : Path
        Input directory containing parquet files
    output_dir : Path
        Output base directory
    overwrite : bool
        Whether to overwrite existing files
    max_workers : int, optional
        Number of bulk parquet files to split concurrently.
    input_files : iterable of path-like, optional
        Explicit bulk parquet files to process. When omitted, all top-level
        parquet files in input_dir are considered.
        
    Returns:
    --------
    bool
        True if processing was successful, False otherwise
    """
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Overwrite existing files: {overwrite}")
    logger.info("-" * 60)
    
    # Find all parquet files, or honor an explicit current-run manifest.
    if input_files is None:
        parquet_files = sorted(input_dir.glob('*.parquet'))
    else:
        parquet_files = []
        seen_files = set()
        for requested_file in input_files:
            parquet_file = Path(requested_file)
            file_key = str(parquet_file.resolve(strict=False))
            if file_key in seen_files:
                continue
            seen_files.add(file_key)
            if parquet_file.suffix.lower() != '.parquet' or not parquet_file.is_file():
                logger.error(f"Requested parquet file does not exist: {parquet_file}")
                return False
            parquet_files.append(parquet_file)
        parquet_files.sort(key=lambda path: str(path))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_dir}")
        return True  # Not an error, just nothing to process
    
    logger.info(f"Found {len(parquet_files)} parquet file(s) to check")
    logger.info("-" * 60)
    
    # Load processing cache
    cache = load_processing_cache(output_dir)
    
    # Check output directory before processing
    files_skipped = 0
    if not overwrite:
        logger.info("Checking output directory for already processed files...")
        files_to_process = []
        
        for parquet_file in tqdm(parquet_files, desc="Checking files"):
            # Neither a legacy cache entry nor one existing split output is
            # sufficient. Verify every expected signal and its exact row count.
            is_processed, missing_signals = check_file_processed(parquet_file, output_dir)
            
            if is_processed:
                files_skipped += 1
                # Update cache for next time
                update_cache_processed(parquet_file, output_dir, cache)
                logger.debug(f"Skipping {parquet_file.name} - already fully processed")
            else:
                if missing_signals:
                    logger.debug(f"{parquet_file.name} partially processed - {len(missing_signals)} signals missing")
                files_to_process.append(parquet_file)
        
        if files_skipped > 0:
            logger.info(f"Skipped {files_skipped} file(s) that are already fully processed")
        
        parquet_files = files_to_process
    
    if not parquet_files:
        logger.info("All files are already processed. Nothing to do.")
        return True
    
    logger.info(f"Processing {len(parquet_files)} parquet file(s)")
    logger.info("-" * 60)
    
    # Process each file
    total_stats = {
        'files_processed': 0,
        'files_skipped': files_skipped,
        'files_failed': 0,
        'total_signals_processed': 0,
        'total_signals_skipped': 0,
        'total_records': 0,
        'all_errors': []
    }
    
    if max_workers is None:
        max_workers = 1
        
    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info(f"Running split in parallel with {max_workers} worker processes...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_parquet_file, parquet_file, output_dir, overwrite): parquet_file
                for parquet_file in parquet_files
            }
            
            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files (parallel)"):
                parquet_file = future_to_file[future]
                try:
                    stats = future.result()
                    if stats['errors']:
                        total_stats['files_failed'] += 1
                        total_stats['all_errors'].extend(stats['errors'])
                    else:
                        total_stats['files_processed'] += 1
                        if not overwrite:
                            update_cache_processed(parquet_file, output_dir, cache)
                    
                    total_stats['total_signals_processed'] += stats['signals_processed']
                    total_stats['total_signals_skipped'] += stats['signals_skipped']
                    total_stats['total_records'] += stats['total_records']
                except Exception as e:
                    error_msg = f"Exception splitting {parquet_file.name}: {e}"
                    logger.error(error_msg, exc_info=True)
                    total_stats['files_failed'] += 1
                    total_stats['all_errors'].append(error_msg)
    else:
        for parquet_file in tqdm(parquet_files, desc="Processing files"):
            stats = process_parquet_file(parquet_file, output_dir, overwrite)
            
            if stats['errors']:
                total_stats['files_failed'] += 1
                total_stats['all_errors'].extend(stats['errors'])
            else:
                total_stats['files_processed'] += 1
                # Update cache for successfully processed files
                if not overwrite:
                    update_cache_processed(parquet_file, output_dir, cache)
            
            total_stats['total_signals_processed'] += stats['signals_processed']
            total_stats['total_signals_skipped'] += stats['signals_skipped']
            total_stats['total_records'] += stats['total_records']
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary:")
    print(f"  Files processed successfully: {total_stats['files_processed']}")
    if total_stats.get('files_skipped', 0) > 0:
        print(f"  Files skipped (already processed): {total_stats['files_skipped']}")
    print(f"  Files failed: {total_stats['files_failed']}")
    print(f"  Total signals processed: {total_stats['total_signals_processed']}")
    print(f"  Total signals skipped (already exist): {total_stats['total_signals_skipped']}")
    print(f"  Total records processed: {total_stats['total_records']:,}")
    
    if total_stats['all_errors']:
        print(f"\n  Errors encountered: {len(total_stats['all_errors'])}")
        print("\n  Error details:")
        for error in total_stats['all_errors'][:10]:  # Show first 10 errors
            print(f"    - {error}")
        if len(total_stats['all_errors']) > 10:
            print(f"    ... and {len(total_stats['all_errors']) - 10} more errors")
    
    print("=" * 60)
    
    # Return True if no failures, False otherwise
    return total_stats['files_failed'] == 0


def main():
    """Main function to split parquet files by SignalID."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split parquet files in raw_data/ by SignalID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all parquet files in default raw_data directory (verbose and overwrite enabled by default)
  python split_by_signal.py
  
  # Specify custom input and output directories
  python split_by_signal.py --input-dir /path/to/raw_data --output-dir /path/to/output
  
  # Skip existing files (disable overwrite)
  python split_by_signal.py --no-overwrite
  
  # Disable verbose logging
  python split_by_signal.py --no-verbose
  
  # Run automatically every 24 hours (default interval)
  python split_by_signal.py --schedule
  
  # Run automatically every 12 hours
  python split_by_signal.py --schedule --interval-hours 12
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/data2/ATSPM/raw_data',
        help='Input directory containing parquet files (default: /data2/ATSPM/raw_data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/data2/ATSPM/raw_data/per_signal',
        help='Output base directory (default: /data2/ATSPM/raw_data/per_signal)'
    )
    
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        dest='overwrite',
        help='Overwrite existing output files (default: disabled)'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_false',
        dest='overwrite',
        help='Do not overwrite existing output files (default)'
    )
    
    parser.set_defaults(verbose=False)
    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_false',
        dest='verbose',
        help='Disable verbose logging (default)'
    )
    
    parser.add_argument(
        '--schedule',
        action='store_true',
        default=True,
        help='Run automatically every 24 hours (default: run every day)'
    )
    
    parser.add_argument(
        '--interval-hours',
        type=int,
        default=24,
        help='Interval in hours for scheduled runs (default: 24)'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear the processing cache and recheck all files'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum worker processes for parallel splitting (default: 1, which is sequential)'
    )
    
    args = parser.parse_args()
    
    # Set logging level (verbose is True by default)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        # Default: ATSPM/raw_data relative to script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        input_dir = project_root / 'raw_data'
        # Also try absolute path from data_downloader.py
        if not input_dir.exists():
            input_dir = Path('/data2/ATSPM/raw_data')
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir
    
    # Handle cache clearing
    if args.clear_cache:
        cache_file = get_cache_file_path(output_dir)
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cache cleared: {cache_file}")
        else:
            print("No cache file found to clear.")
        return
    
    # If scheduled mode, set up the scheduler
    if args.schedule:
        if schedule is None:
            logger.error("Error: 'schedule' package is not installed. Install with: pip install schedule")
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("SCHEDULED MODE ENABLED")
        logger.info(f"Processing will run every {args.interval_hours} hours")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Create a wrapper function for scheduling
        def scheduled_job():
            logger.info(f"[SCHEDULED RUN] Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            success = run_processing(input_dir, output_dir, args.overwrite, max_workers=args.max_workers)
            if not success:
                logger.warning("[SCHEDULED RUN] Processing completed with errors")
            else:
                logger.info("[SCHEDULED RUN] Processing completed successfully")
            logger.info(f"[SCHEDULED RUN] Next run in {args.interval_hours} hours")
        
        # Schedule the job
        schedule.every(args.interval_hours).hours.do(scheduled_job)
        
        # Run once immediately
        logger.info("Running initial processing...")
        success = run_processing(input_dir, output_dir, args.overwrite, max_workers=args.max_workers)
        if not success:
            logger.warning("Initial processing completed with errors")
        
        # Then run on schedule
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")
            sys.exit(0)
    else:
        # Run once and exit
        success = run_processing(input_dir, output_dir, args.overwrite, max_workers=args.max_workers)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

