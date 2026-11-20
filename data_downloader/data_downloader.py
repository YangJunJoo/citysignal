#!/usr/bin/env python3
"""
Data Downloader Script
Downloads ATSPM CSV files from the server and converts them to Parquet format.

Example usage:
conda activate atspm & python data_downloader/data_downloader.py

the default output directory is /data2/ATSPM/raw_data
"""

import os
import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.util.retry import Retry
from urllib3.exceptions import IncompleteRead as Urllib3IncompleteRead
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def generate_date_range(start_date, end_date):
    """Generate a list of dates from start_date to end_date (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def format_date_for_url(date):
    """Format date as YYYY-M-D for URL."""
    return f"{date.year}-{date.month}-{date.day}"


def load_download_log(log_path):
    """Load the download log from JSON file."""
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If log file is corrupted, return empty dict
            return {}
    return {}


def save_download_log(log_path, log_data):
    """Save the download log to JSON file."""
    try:
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save download log: {e}")


def log_successful_download(log_path, date_str, url, csv_path, parquet_path):
    """Log a successful download and conversion."""
    log_data = load_download_log(log_path)
    log_data[date_str] = {
        'url': url,
        'csv_path': str(csv_path),
        'parquet_path': str(parquet_path),
        'downloaded_at': datetime.now().isoformat(),
        'status': 'completed'
    }
    save_download_log(log_path, log_data)


class DownloadIntegrityError(ValueError):
    """A response cannot safely be combined with the existing partial object."""


def _remote_version(headers):
    etag = headers.get("ETag") or headers.get("etag")
    modified = headers.get("Last-Modified") or headers.get("last-modified")
    return {"etag": etag, "last_modified": modified}


def _resume_validator(metadata):
    etag = metadata.get("etag")
    if etag and not etag.startswith("W/"):
        return etag
    return metadata.get("last_modified")


def _response_plan(response, resume_pos, metadata):
    """Return (write mode, start, total) only for a validated response."""
    headers = {k.lower(): v for k, v in response.headers.items()}
    if headers.get('content-encoding', 'identity').lower() != 'identity':
        raise DownloadIntegrityError("Encoded response cannot be byte-range validated")
    length = int(headers['content-length']) if 'content-length' in headers else None
    if response.status_code == 200:
        return 'wb', 0, length
    if response.status_code != 206:
        raise DownloadIntegrityError(f"Unexpected success HTTP status {response.status_code}")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get('content-range', ''))
    if not match:
        raise DownloadIntegrityError("Missing or malformed Content-Range")
    start, end, total = map(int, match.groups())
    if start != resume_pos or end < start or end >= total or end != total - 1:
        raise DownloadIntegrityError("Content-Range does not match requested suffix")
    if length is not None and length != end - start + 1:
        raise DownloadIntegrityError("Content-Length disagrees with Content-Range")
    if resume_pos:
        previous = _resume_validator(metadata)
        current = _remote_version(response.headers)
        key = 'etag' if metadata.get('etag') and not metadata['etag'].startswith('W/') else 'last_modified'
        if not previous or current.get(key) != previous:
            raise DownloadIntegrityError("Source validator changed or is unavailable during resume")
        if metadata.get('total_size') is not None and total != metadata['total_size']:
            raise DownloadIntegrityError("Source size changed during resume")
    return ('ab' if resume_pos else 'wb'), resume_pos, total


def download_csv(url, output_path, max_retries=50):
    """Download to staging, verify byte ranges/version, then atomically publish.

    Partial files from older versions (without a validator sidecar) restart.
    Existing final files are preserved on every failure. HTTP evidence is stored
    at <output>.download.json, including publication 404s.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(str(output_path) + '.tmp')
    meta_path = Path(str(temp_path) + '.json')
    evidence_path = Path(str(output_path) + '.download.json')
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1.2,
                           status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "HEAD"])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    evidence = {"url": url, "status": "failed"}

    def write_evidence(**values):
        evidence.update(values, checked_at=datetime.now().astimezone().isoformat())
        evidence_path.write_text(json.dumps(evidence, indent=2))

    def discard_partial():
        temp_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    try:
        for attempt in range(max_retries):
            response = None
            try:
                try:
                    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                except (OSError, ValueError):
                    metadata = {}
                resume_pos = temp_path.stat().st_size if temp_path.exists() else 0
                if resume_pos and (metadata.get('url') != url or not _resume_validator(metadata)):
                    discard_partial()
                    resume_pos = 0
                    metadata = {}
                headers = {'Accept-Encoding': 'identity'}
                if resume_pos:
                    headers.update(Range=f'bytes={resume_pos}-', **{'If-Range': _resume_validator(metadata)})
                response = session.get(url, headers=headers, timeout=(30, 1800), stream=True, verify=False)
                if response.status_code == 416:
                    response.close()
                    discard_partial()
                    resume_pos, metadata = 0, {}
                    response = session.get(url, headers={'Accept-Encoding': 'identity'},
                                           timeout=(30, 1800), stream=True, verify=False)
                write_evidence(http_status=response.status_code, headers=dict(response.headers))
                response.raise_for_status()
                mode, resume_pos, total_size = _response_plan(response, resume_pos, metadata)
                metadata = dict(_remote_version(response.headers), url=url, total_size=total_size)
                meta_path.write_text(json.dumps(metadata, indent=2))
                with temp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                size = temp_path.stat().st_size
                if total_size is not None and size != total_size:
                    raise ValueError(f"Download incomplete: {size} bytes, expected {total_size}")
                if size == 0:
                    raise DownloadIntegrityError("Empty download")
                os.replace(temp_path, output_path)
                meta_path.unlink(missing_ok=True)
                write_evidence(status='downloaded', byte_count=size, **metadata)
                return True
            except DownloadIntegrityError as exc:
                # Never append a rejected response to a previous partial object.
                discard_partial()
                write_evidence(error=str(exc))
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.Timeout, Urllib3IncompleteRead, ValueError) as exc:
                write_evidence(error=f"{type(exc).__name__}: {exc}")
            except requests.exceptions.RequestException as exc:
                write_evidence(error=f"{type(exc).__name__}: {exc}")
                discard_partial()
                return False
            finally:
                if response is not None:
                    response.close()
            if attempt < max_retries - 1:
                time.sleep(min(30, 2 ** min(attempt, 5)))
        return False
    finally:
        session.close()


def is_csv_corrupted(csv_path):
    """Check if CSV file is corrupted by trying to read it."""
    try:
        # Try to read just a few rows to check if file is valid
        df = pd.read_csv(str(csv_path), nrows=1, low_memory=False)
        # Also try to get file size - empty files might be considered corrupted
        if csv_path.stat().st_size == 0:
            return True
        return False
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, 
            Exception) as e:
        print(f"CSV file {csv_path.name} appears corrupted: {e}")
        return True


def is_parquet_corrupted(parquet_path):
    """Check if Parquet file is corrupted by trying to read it."""
    try:
        # Try to read metadata and a small sample
        parquet_file = pq.ParquetFile(str(parquet_path))
        # Check if file has metadata and can be read
        if parquet_file.metadata.num_rows == 0 or parquet_file.metadata.num_row_groups == 0:
            # Empty parquet file might be considered corrupted
            return True
        # Try to read a small sample from the first row group/column to verify data integrity
        parquet_file.read_row_group(0, columns=[parquet_file.schema.names[0]])
        return False
    except Exception as e:
        print(f"Parquet file {parquet_path.name} appears corrupted: {e}")
        return True


def csv_to_parquet(csv_path, parquet_path):
    """Convert CSV file to Parquet format."""
    try:
        # Read CSV file with low_memory=False to avoid DtypeWarning for mixed types
        df = pd.read_csv(str(csv_path), low_memory=False)
        
        # Find and normalize signalID column to handle mixed types
        # Try common variations of SignalID column name
        signal_id_col = None
        possible_names = [
            'SignalID', 'signalID', 'signal_id', 'Signal_Id',
            'SIGNALID', 'SIGNAL_ID', 'SignalId'
        ]
        
        for name in possible_names:
            if name in df.columns:
                signal_id_col = name
                break
        
        # Case-insensitive search if exact match not found
        if signal_id_col is None:
            df_columns_lower = [col.lower() for col in df.columns]
            for name in possible_names:
                if name.lower() in df_columns_lower:
                    idx = df_columns_lower.index(name.lower())
                    signal_id_col = df.columns[idx]
                    break
        
        # Normalize signalID column: preserve both numeric and string IDs
        if signal_id_col is not None:
            # Convert to string to handle mixed types (int + string like ORL-0265)
            df[signal_id_col] = df[signal_id_col].astype(str).str.strip()
            # Remove 'nan' strings from NaN conversion
            df = df[df[signal_id_col] != 'nan']
            # For purely numeric IDs, strip trailing .0 (from float conversion)
            df[signal_id_col] = df[signal_id_col].str.replace(r'\.0$', '', regex=True)
        
        # Write to Parquet format
        staged_path = Path(str(parquet_path) + '.converting.tmp')
        df.to_parquet(str(staged_path), engine='pyarrow', compression='snappy', index=False)
        if is_parquet_corrupted(staged_path):
            raise ValueError('Converted Parquet failed integrity validation')
        os.replace(staged_path, parquet_path)
        return True
    except Exception as e:
        print(f"Error converting {csv_path} to Parquet: {e}")
        return False


def main(start_date=None, end_date=None, output_dir=None):
    # Configuration
    base_url = "https://sunstore.cflsmartroads.com/stagedfiles/csvfiles/atspm-{date}.csv"
    if output_dir is None:
        output_dir = Path("/data2/ATSPM/raw_data")
    else:
        output_dir = Path(output_dir)
    log_path = output_dir / "download_log.json"
    
    current_date = datetime.now()  # Use current date dynamically
    
    # Date range: 2025-8-1 to current day minus 1 day
    if start_date is None:
        start_date = datetime(2025, 8, 1)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
    if end_date is None:
        end_date = current_date - timedelta(days=1)  # Current day minus 1 day (yesterday)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"Run started at: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Downloading files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output directory: {output_dir}")
    print(f"Download log: {log_path}")
    print("-" * 60)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load download log
    download_log = load_download_log(log_path)
    print(f"Loaded {len(download_log)} entries from download log")
    
    # Generate list of dates
    dates = generate_date_range(start_date, end_date)
    
    # Statistics
    successful_downloads = 0
    failed_downloads = 0
    successful_conversions = 0
    failed_conversions = 0
    skipped_files = 0
    corrupted_files_quarantined = 0
    
    # Process each date
    for date in tqdm(dates, desc="Processing dates"):
        date_str = format_date_for_url(date)
        url = base_url.format(date=date_str)
        
        # Temporary CSV file path
        csv_filename = f"atspm-{date_str}.csv"
        csv_path = output_dir / csv_filename
        
        # Parquet file path
        parquet_filename = f"atspm-{date_str}.parquet"
        parquet_path = output_dir / parquet_filename
        
        # Check if files exist and verify they're not corrupted
        csv_corrupted = False
        parquet_corrupted = False
        
        if csv_path.exists():
            if is_csv_corrupted(csv_path):
                csv_corrupted = True
                print(f"Corrupted CSV detected: {csv_path.name}, preserving as .corrupt and retrying...")
                try:
                    csv_path.rename(Path(str(csv_path) + '.corrupt.' + datetime.now().strftime('%Y%m%dT%H%M%S%f')))
                    # Also remove any temporary files
                    temp_csv = Path(str(csv_path) + '.tmp')
                    if temp_csv.exists():
                        os.remove(temp_csv)
                except Exception as e:
                    print(f"Warning: Could not remove corrupted CSV {csv_path.name}: {e}")
        
        if parquet_path.exists():
            if is_parquet_corrupted(parquet_path):
                parquet_corrupted = True
                print(f"Corrupted Parquet detected: {parquet_path.name}, preserving as .corrupt and retrying...")
                try:
                    parquet_path.rename(Path(str(parquet_path) + '.corrupt.' + datetime.now().strftime('%Y%m%dT%H%M%S%f')))
                except Exception as e:
                    print(f"Warning: Could not remove corrupted Parquet {parquet_path.name}: {e}")
        
        # If either file was corrupted, remove from log and retry
        if csv_corrupted or parquet_corrupted:
            corrupted_files_quarantined += 1
            if date_str in download_log:
                del download_log[date_str]
                save_download_log(log_path, download_log)
        
        # Skip if already logged as successful and parquet file exists and is not corrupted
        if date_str in download_log:
            log_entry = download_log[date_str]
            # Verify the parquet file still exists and is valid
            if parquet_path.exists() and log_entry.get('status') == 'completed' and not parquet_corrupted:
                skipped_files += 1
                continue
            # If logged but file missing or corrupted, remove from log and retry
            elif not parquet_path.exists() or parquet_corrupted:
                if not parquet_corrupted:
                    print(f"Warning: {date_str} was logged but parquet file missing, retrying...")
                del download_log[date_str]
                save_download_log(log_path, download_log)
        
        # Also skip if Parquet file exists and is valid (as backup check)
        if parquet_path.exists() and not parquet_corrupted:
            skipped_files += 1
            # Add to log if not already there
            if date_str not in download_log:
                log_successful_download(log_path, date_str, url, csv_path, parquet_path)
            continue
        
        # Download CSV file
        if download_csv(url, csv_path):
            successful_downloads += 1
            
            # Convert to Parquet
            if csv_to_parquet(csv_path, parquet_path):
                successful_conversions += 1
                
                # Log successful download and conversion
                log_successful_download(log_path, date_str, url, csv_path, parquet_path)
                
                # Remove CSV file after successful conversion
                if csv_path.exists():
                    os.remove(csv_path)
            else:
                failed_conversions += 1
        else:
            failed_downloads += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Failed downloads: {failed_downloads}")
    print(f"  Successful conversions: {successful_conversions}")
    print(f"  Failed conversions: {failed_conversions}")
    print(f"  Corrupted files quarantined: {corrupted_files_quarantined}")
    print(f"  Skipped (already exist): {skipped_files}")
    print(f"  Total dates processed: {len(dates)}")
    print(f"Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    import signal
    
    # Flag to handle graceful shutdown
    running = True
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        global running
        print("\n\nShutdown signal received. Finishing current run...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*60)
    print("Data Downloader - Continuous Mode")
    print("Will run every 24 hours to download new data")
    print("Press Ctrl+C to stop")
    print("="*60)
    print()
    
    # Run continuously: download -> sleep 24 hours -> repeat
    while running:
        try:
            # Run the download process
            main()
            
            if not running:
                break
            
            # Calculate next run time
            next_run = datetime.now() + timedelta(hours=24)
            print(f"\nNext download run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Sleeping for 24 hours...")
            print("-"*60)
            
            # Sleep for 24 hours (86400 seconds)
            # Sleep in smaller chunks to allow for graceful shutdown
            sleep_duration = 24 * 3600  # 24 hours in seconds
            chunk_size = 3600  # Check every hour if we should continue
            
            for _ in range(sleep_duration // chunk_size):
                if not running:
                    break
                time.sleep(chunk_size)
            
            # Sleep the remainder if still running
            if running:
                remainder = sleep_duration % chunk_size
                if remainder > 0:
                    time.sleep(remainder)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n\nUnexpected error occurred: {e}")
            print("Will retry in 24 hours...")
            if running:
                time.sleep(24 * 3600)
    
    print("\nData downloader stopped.")

