r"""
Process ATSPM event data from local directories with per_signal structure.
Adapted to work with parquet files organized by signal ID and custom config locations.

Update date: 2025-12-28

This script can process:
- Single parquet files for a specific date and signal
- All parquet files in per_signal directory structure (batch processing)
- Specific signal configurations or all available signals
- Multiple signals in parallel using multiprocessing
- Automated processing of new/updated files (for daily cron jobs)

Expected directory structure:
  /data2/ATSPM/raw_data/per_signal/{signal_id}/{YYYY-MM-DD}.parquet

Command line usage:
  # Process all files
  python feature_extractor/process_local_atspm.py \
  --event-dir /data2/ATSPM/raw_data/ \
  --config-dir /data2/ATSPM/config_data/signal_config_2025/lanes \
  --output-dir /data2/ATSPM/processed_data/
  --year 2025 --month 10

  # Process specific date and signals
  python feature_extractor/process_local_atspm.py \
  --event-dir /data2/ATSPM/raw_data/ \
  --config-dir /data2/ATSPM/config_data/signal_config_2025/lanes \
  --output-dir /data2/ATSPM/processed_data/ \
  --year 2025 --month 9 --day 7 \
  --signals ORC-0003

  # Automated mode: only process new/updated files (for cron jobs)
  python feature_extractor/process_local_atspm.py \
  --event-dir /data2/ATSPM/raw_data/ \
  --config-dir /data2/ATSPM/config_data/signal_config_2025/lanes \
  --output-dir /data2/ATSPM/processed_data/ \
  --auto --max-age-hours 24

Processing Log:
  The script maintains a SQLite database (processing_log.db) in the output directory
  that tracks processing status by signal_id, date, and feature. This allows:
  - Skipping already processed features on subsequent runs
  - Detecting when input files have been updated
  - Tracking processing history and statistics

For help with all options:
  python process_local_atspm.py --help
"""


import pandas as pd
import numpy as np
import os
import logging
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

# Add project root (parent of feature_extractor/) to sys.path for feature_extractor imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
import glob
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import sqlite3
import json
# Removed Lock import - SQLite handles concurrent access natively

# Suppress warnings
warnings.simplefilter('ignore', FutureWarning)  # Suppress all FutureWarnings including pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)  # DataFrame fragmentation
warnings.filterwarnings('ignore', message='.*DataFrame concatenation.*')
warnings.filterwarnings('ignore', message='.*behavior of DataFrame concatenation.*')
warnings.filterwarnings('ignore', message='.*empty or all-NA entries.*')
warnings.filterwarnings('ignore', message='.*highly fragmented.*')

# Try to import psutil for memory-aware worker calculation
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Warning will be shown in _calculate_optimal_workers() only in main process

# Import configuration module
from feature_extractor import ENGINE_ID
from feature_extractor.config import ProcessingConfig, ConfigParser

# Try to import tqdm for enhanced progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
    
    # Create a no-op tqdm class for silent mode
    class _NoOpTqdm:
        """A no-op version of tqdm that suppresses all output."""
        def __init__(self, iterable=None, *args, **kwargs):
            # Store the iterable if provided (tqdm can wrap an iterable)
            self.iterable = iterable
        def __iter__(self):
            # If wrapping an iterable, iterate over it; otherwise return self
            if self.iterable is not None:
                return iter(self.iterable)
            return self
        def __next__(self):
            # If no iterable, raise StopIteration
            raise StopIteration
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def update(self, n=1):
            pass
        def set_description(self, desc=None, refresh=True):
            pass
        def close(self):
            pass
    
    # Store original tqdm for restoration
    _original_tqdm = tqdm
except ImportError:
    TQDM_AVAILABLE = False
    _NoOpTqdm = None
    _original_tqdm = None


def _disable_tqdm_globally():
    """Disable tqdm globally by monkey-patching it with a no-op version."""
    if TQDM_AVAILABLE:
        import sys
        import os
        # Set environment variable that tqdm checks (if it supports it)
        os.environ['TQDM_DISABLE'] = '1'
        
        # Monkey-patch tqdm in the tqdm module
        try:
            import tqdm
            tqdm.tqdm = _NoOpTqdm
            # Also patch in sys.modules
            if 'tqdm' in sys.modules:
                sys.modules['tqdm'].tqdm = _NoOpTqdm
        except Exception:
            pass  # If patching fails, continue anyway


def _enable_tqdm_globally():
    """Re-enable tqdm globally by restoring the original."""
    if TQDM_AVAILABLE:
        import os
        import sys
        # Clear environment variable
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']
        
        # Restore original tqdm
        try:
            import tqdm
            tqdm.tqdm = _original_tqdm
            if 'tqdm' in sys.modules:
                sys.modules['tqdm'].tqdm = _original_tqdm
        except Exception:
            pass  # If restoration fails, continue anyway


class ProcessingLog:
    """Track processing status by signal_id, date, and feature using SQLite database."""
    
    def __init__(self, log_db_path=None):
        """
        Initialize processing log database.
        
        Parameters:
        -----------
        log_db_path : str, optional
            Path to SQLite database file. If None, uses default location in output_dir
        """
        if log_db_path is None:
            # Default location: use a standard location relative to script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_db_path = os.path.join(script_dir, "processing_log.db")
        
        self.log_db_path = log_db_path
        # Note: No lock needed - SQLite handles concurrent access with check_same_thread=False
        # SQLite uses file-level locking which works across processes
        self._init_database()
        
        # Connection pool for better performance (thread-local connections)
        self._local_conn = None
        self._batch_logs = []  # Batch logs to reduce I/O
        self._batch_size = 50  # Flush after 50 logs
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.log_db_path, check_same_thread=False, timeout=30.0)
        # Enable WAL mode for better concurrent performance
        conn.execute("PRAGMA journal_mode=WAL")
        # Optimize for performance
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        cursor = conn.cursor()
        
        # Create processing_log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL,
                date_str TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                status TEXT NOT NULL,
                input_file_path TEXT,
                input_file_mtime REAL,
                output_file_path TEXT,
                processing_time REAL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                UNIQUE(signal_id, date_str, feature_name)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_date_feature 
            ON processing_log(signal_id, date_str, feature_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_date 
            ON processing_log(signal_id, date_str)
        """)
        
        # Create index on timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON processing_log(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get a database connection (reuse if available, create new if needed)."""
        # For multiprocessing, each process needs its own connection
        # Use thread-local storage would be ideal, but for simplicity, create per call
        # The WAL mode and optimizations help with concurrent access
        return sqlite3.connect(self.log_db_path, check_same_thread=False, timeout=30.0)
    
    def flush_batch_logs(self):
        """Flush any pending batch logs to database."""
        if not self._batch_logs:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for log_entry in self._batch_logs:
                cursor.execute("""
                    INSERT OR REPLACE INTO processing_log 
                    (signal_id, date_str, feature_name, status, input_file_path, 
                     input_file_mtime, output_file_path, processing_time, error_message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, log_entry)
            
            conn.commit()
            conn.close()
            self._batch_logs = []
        except Exception as e:
            # If batch fails, log individually
            logging.warning(f"Batch log flush failed: {e}, logging individually")
            for log_entry in self._batch_logs:
                try:
                    self._log_feature_processing_direct(*log_entry)
                except Exception:
                    pass
            self._batch_logs = []
    
    def log_feature_processing(self, signal_id, date_str, feature_name, status, 
                              input_file_path=None, input_file_mtime=None,
                              output_file_path=None, processing_time=None,
                              error_message=None, use_batch=True):
        """
        Log a feature processing event.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date_str : str
            Date string in YYYY-MM-DD format
        feature_name : str
            Name of the feature being processed
        status : str
            Status: 'success', 'failed', 'skipped'
        input_file_path : str, optional
            Path to input file
        input_file_mtime : float, optional
            Modification time of input file
        output_file_path : str, optional
            Path to output file
        processing_time : float, optional
            Processing time in seconds
        error_message : str, optional
            Error message if status is 'failed'
        use_batch : bool
            If True, batch writes for better performance (default: True)
        """
        timestamp = datetime.now().isoformat()
        log_entry = (signal_id, date_str, feature_name, status, input_file_path,
                     input_file_mtime, output_file_path, processing_time, error_message, timestamp)
        
        if use_batch and hasattr(self, '_batch_logs'):
            self._batch_logs.append(log_entry)
            # Flush if batch is full
            if len(self._batch_logs) >= self._batch_size:
                self.flush_batch_logs()
        else:
            # Direct write (for critical logs or when batching disabled)
            self._log_feature_processing_direct(*log_entry)
    
    def _log_feature_processing_direct(self, signal_id, date_str, feature_name, status,
                                       input_file_path, input_file_mtime,
                                       output_file_path, processing_time, error_message, timestamp):
        """Direct database write (used for critical logs or when batching disabled)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processing_log 
                (signal_id, date_str, feature_name, status, input_file_path, 
                 input_file_mtime, output_file_path, processing_time, error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, date_str, feature_name, status, input_file_path,
                  input_file_mtime, output_file_path, processing_time, error_message, timestamp))
            conn.commit()
        finally:
            conn.close()
    
    def is_feature_processed(self, signal_id, date_str, feature_name, 
                            input_file_path=None, check_file_mtime=True):
        """
        Check if a feature has been successfully processed.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date_str : str
            Date string in YYYY-MM-DD format
        feature_name : str
            Name of the feature
        input_file_path : str, optional
            Path to input file to check modification time
        check_file_mtime : bool
            If True, also check if input file has been modified since last processing
        
        Returns:
        --------
        bool
            True if feature was successfully processed and file hasn't changed
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status, input_file_mtime FROM processing_log
                WHERE signal_id = ? AND date_str = ? AND feature_name = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (signal_id, date_str, feature_name))
            
            result = cursor.fetchone()
        finally:
            conn.close()
        
        if result is None:
            return False
        
        status, logged_mtime = result
        
        if status != 'success':
            return False
        
        # Check if input file has been modified
        if check_file_mtime and input_file_path and os.path.exists(input_file_path):
            try:
                current_mtime = os.path.getmtime(input_file_path)
                if logged_mtime is None or current_mtime > logged_mtime:
                    return False  # File has been updated
            except OSError:
                pass  # Can't check mtime, assume not processed
        
        return True
    
    def get_all_processed_features(self, signal_id, date_str):
        """
        Get all successfully processed features for a signal-date combination.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date_str : str
            Date string in YYYY-MM-DD format
        
        Returns:
        --------
        set
            Set of feature names that have been successfully processed
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT feature_name FROM processing_log
                WHERE signal_id = ? AND date_str = ? AND status = 'success'
            """, (signal_id, date_str))
            
            features = {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()
        
        return features
    
    def is_signal_date_fully_processed(self, signal_id, date_str, required_features, 
                                       input_file_path=None, check_file_mtime=True):
        """
        Check if all required features have been processed for a signal-date.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date_str : str
            Date string in YYYY-MM-DD format
        required_features : set or list
            Set of required feature names
        input_file_path : str, optional
            Path to input file to check modification time
        check_file_mtime : bool
            If True, also check if input file has been modified
        
        Returns:
        --------
        bool
            True if all required features have been successfully processed
        """
        processed_features = self.get_all_processed_features(signal_id, date_str)
        
        if not processed_features.issuperset(required_features):
            return False
        
        # Check file modification time if requested
        if check_file_mtime and input_file_path and os.path.exists(input_file_path):
            try:
                current_mtime = os.path.getmtime(input_file_path)
                # Get the most recent processing timestamp
                conn = self._get_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT MAX(input_file_mtime) FROM processing_log
                        WHERE signal_id = ? AND date_str = ? AND status = 'success'
                    """, (signal_id, date_str))
                    result = cursor.fetchone()
                finally:
                    conn.close()
                
                if result and result[0] and current_mtime > result[0]:
                    return False  # File has been updated
            except OSError:
                pass
        
        return True
    
    def get_processing_stats(self):
        """Get overall processing statistics from the log."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped
                FROM processing_log
            """)
            
            stats = cursor.fetchone()
        finally:
            conn.close()
        
        return {
            'total': stats[0] or 0,
            'successful': stats[1] or 0,
            'failed': stats[2] or 0,
            'skipped': stats[3] or 0
        }


class ProgressTracker:
    """Track and display progress for feature extraction steps."""
    
    def __init__(self, total_steps, signal_id, date_str="", use_tqdm=None, silent_mode=True):
        """
        Initialize progress tracker.
        
        Parameters:
        -----------
        total_steps : int
            Total number of steps to track
        signal_id : str
            Signal ID being processed
        date_str : str, optional
            Date string for display
        use_tqdm : bool, optional
            Whether to use tqdm progress bar. If None, auto-detect if tqdm is available
        silent_mode : bool, optional
            If True, suppress all progress output except signal start/completion
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.signal_id = signal_id
        self.date_str = date_str
        self.start_time = time.time()
        self.step_times = []
        # Respect the silent_mode parameter passed in
        self.silent_mode = silent_mode
        # Disable tqdm when in silent mode
        if silent_mode:
            self.use_tqdm = False
        else:
            self.use_tqdm = use_tqdm if use_tqdm is not None else (TQDM_AVAILABLE and not silent_mode)
        self.pbar = None
        
    def start_step(self, step_name):
        """Start a new step and print progress."""
        self.current_step += 1
        step_start_time = time.time()
        self.step_times.append(step_start_time)
        
        if self.silent_mode:
            # In silent mode, don't print anything for individual steps
            # Don't update progress bars or print anything
            return step_start_time
        
        if self.use_tqdm and self.pbar is not None:
            # Update tqdm progress bar with current step name
            self.pbar.set_description(f"Signal {self.signal_id} {self.date_str}: {step_name}")
            self.pbar.update(1)
        
        return step_start_time
    
    def end_step(self, step_start_time, step_name=""):
        """End current step and print completion info."""
        if self.silent_mode:
            # In silent mode, don't print anything for individual steps
            return
        
        # Feature-level logging removed as requested
        return
    
    def _format_time(self, seconds):
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def print_summary(self):
        """Print final processing summary."""
        total_time = time.time() - self.start_time
        
        # Progress output is suppressed in silent mode
        if not self.silent_mode:
            print(f"✓ Signal {self.signal_id} {self.date_str} completed in {self._format_time(total_time)}")
    
    def close(self):
        """Close the progress tracker and cleanup resources."""
        if self.use_tqdm and self.pbar:
            self.pbar.close()


class LocalATSPMProcessor:
    """Process ATSPM data from custom local paths with parquet files."""
    
    def __init__(self, 
                 event_data_dir, 
                 config_dir, 
                 output_dir=None,
                 year=None, month=None, day=None,
                 use_tqdm=None,
                 max_workers=None,
                 log_db_path=None,
                 features=None,
                 overwrite=False):
        """
        Initialize processor for parquet files in per_signal directory structure.
        
        Parameters:
        -----------
        event_data_dir : str
            Base directory containing per_signal subdirectories (e.g., "/data2/ATSPM/raw_data")
        config_dir : str
            Directory containing signal config CSV files
        output_dir : str, optional
            Output directory for results. If None, creates output in event_data_dir/processed
        year : int, optional
            Specific year to process. If None, processes all years found
        month : int, optional
            Specific month to process. If None, processes all months found
        day : int, optional
            Specific day to process. If None, processes all days found
        use_tqdm : bool, optional
            Whether to use tqdm progress bars. If None, auto-detect if tqdm is available
        max_workers : int, optional
            Maximum number of worker processes for parallel processing. If None, uses CPU count
        log_db_path : str, optional
            Path to processing log database. If None, uses default location
        features : list, optional
            List of specific features to extract. If None, extracts all features.
            Valid features: vehicle_phase_profile, pedestrian_phase_profile, vehicle_cycle_profile,
            pedestrian_cycle_profile, spat, volume, occupancy, split_failure, headway, conflict,
            conflict_pet, gap, pet, shockwave, red_running, dilemma_running, pedestrian_activity,
            pedestrian_delay, conflict_propensity
        """
        self.year = year
        self.month = month  
        self.day = day
        self.event_data_dir = event_data_dir
        self.config_dir = config_dir
        self.output_dir = output_dir or os.path.join(event_data_dir, "processed")
        self.use_tqdm = use_tqdm
        self.local_timezone = ZoneInfo("America/New_York")
        self.features = features  # Store requested features
        self.overwrite = overwrite  # Store overwrite flag
        
        # Resource constraints
        self.max_cpu = 48
        self.max_ram_gb = 300
        
        # Set default worker count (default is max_cpu = 48)
        if max_workers is None:
            self.max_workers = self.max_cpu
        else:
            self.max_workers = min(max_workers, self.max_cpu)
        # Cap inner feature worker threads to a small number to avoid oversubscription when nested
        self.feature_worker_limit = max(1, min(4, self.max_workers))
        
        # Print resource configuration
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Max CPU constraint:        {self.max_cpu}")
        print(f"Max RAM constraint:        {self.max_ram_gb} GB")
        print(f"Optimal worker processes:   {self.max_workers}")
        print(f"{'='*60}\n")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processing log
        if log_db_path is None:
            log_db_path = os.path.join(self.output_dir, "processing_log.db")
        self.processing_log = ProcessingLog(log_db_path)
        
        # Define feature name mapping (step names to feature identifiers)
        self.feature_name_mapping = {
            "Setup and initialization": "setup",
            "Vehicle phase profiles": "vehicle_phase_profile",
            "Pedestrian phase profiles": "pedestrian_phase_profile",
            "Vehicle cycle profiles": "vehicle_cycle_profile",
            "Pedestrian cycle profiles": "pedestrian_cycle_profile",
            "SPaT features": "spat",
            "Volume extraction": "volume",
            "Occupancy extraction": "occupancy",
            "Split failure extraction": "split_failure",
            "Headway extraction": "headway",
            "Conflict extraction": "conflict",
            "Conflict PET extraction": "conflict_pet",
            "Gap extraction": "gap",
            "PET extraction": "pet",
            "Shockwave extraction": "shockwave",
            "Red running extraction": "red_running",
            "Dilemma running extraction": "dilemma_running",
            "Pedestrian activity extraction": "pedestrian_activity",
            "Pedestrian delay extraction": "pedestrian_delay",
            "Conflict propensity extraction": "conflict_propensity",
            "Finalizing results": "finalization"
        }
        
        # Get required features for full processing
        self.required_features = set(self.feature_name_mapping.values())
        self.required_features.discard("setup")  # Setup is not a feature
        self.required_features.discard("finalization")  # Finalization is not a feature
        self.required_features.discard("conflict_propensity")  # Skip conflict propensity by default (causes NA boolean errors)
        
        # Define feature dependencies (which features require other features to be processed first)
        self.feature_dependencies = {
            "vehicle_cycle_profile": ["vehicle_phase_profile"],
            "pedestrian_cycle_profile": ["pedestrian_phase_profile", "vehicle_cycle_profile"],
            "split_failure": ["spat", "occupancy"],
            "conflict": ["headway"],
            "conflict_pet": ["pet"],
        }
        
        # Resolve requested features (include dependencies)
        if self.features is not None:
            self.selected_features = self._resolve_feature_dependencies(set(self.features))
        else:
            # Default: all features except conflict_propensity (causes NA boolean errors)
            all_features_except_conflict = set(self.feature_name_mapping.values())
            all_features_except_conflict.discard("setup")
            all_features_except_conflict.discard("finalization")
            all_features_except_conflict.discard("conflict_propensity")
            self.selected_features = all_features_except_conflict
        
        # Setup logging - only show errors
        logging.basicConfig(level=logging.CRITICAL, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Add filter to suppress specific warning messages
        class WarningFilter(logging.Filter):
            def filter(self, record):
                # Suppress "No continuous detector occupation period" warnings
                if "No continuous detector occupation period" in record.getMessage():
                    return False
                return True
        
        # Apply filter to root logger
        logging.getLogger().addFilter(WarningFilter())
        
        
        # Cache for processed index
        self._processed_index = None
    
    def _resolve_feature_dependencies(self, requested_features):
        """
        Resolve feature dependencies by adding required dependencies to the requested features.
        
        Parameters:
        -----------
        requested_features : set
            Set of requested feature names
        
        Returns:
        --------
        set
            Set of feature names including all dependencies
        """
        resolved = set(requested_features)
        added_deps = True
        
        # Iteratively add dependencies until no more are found
        while added_deps:
            added_deps = False
            for feature in list(resolved):
                if feature in self.feature_dependencies:
                    for dep in self.feature_dependencies[feature]:
                        if dep not in resolved:
                            resolved.add(dep)
                            added_deps = True
        
        return resolved
    
    def _get_steps_for_features(self, all_steps):
        """
        Filter steps based on selected features.
        
        Parameters:
        -----------
        all_steps : list
            List of all step names
        
        Returns:
        --------
        list
            List of step names to execute (filtered based on selected features)
        """
        if self.selected_features is None:
            # This shouldn't happen anymore (we set selected_features to a set), but keep for safety
            # Exclude conflict propensity step
            return [step for step in all_steps 
                   if self.feature_name_mapping.get(step) != "conflict_propensity"]
        
        # Always include setup, finalization, and initial steps 1-5 (required for all feature extraction)
        # Steps 1-5: Setup, Vehicle phase profiles, Pedestrian phase profiles, 
        #           Vehicle cycle profiles, Pedestrian cycle profiles
        filtered_steps = []
        
        # Check if step should be included
        for step_name in all_steps:
            feature_name = self.feature_name_mapping.get(step_name)
            
            # Always include setup, finalization, and initial steps 1-5 (required prerequisites)
            if feature_name in ["setup", "finalization", 
                               "vehicle_phase_profile", "pedestrian_phase_profile",
                               "vehicle_cycle_profile", "pedestrian_cycle_profile"]:
                filtered_steps.append(step_name)
                continue
            
            # Include step if its feature is in selected features
            if feature_name in self.selected_features:
                filtered_steps.append(step_name)
        
        return filtered_steps
        
    
    def _calculate_paths(self):
        """
        Calculate the paths for the extractor package when output_dir may be outside event_data_dir.
        
        Handles three cases:
        1. output_dir is inside event_data_dir: use event_data_dir as cwd, relative paths
        2. output_dir is outside but can be made relative: find common parent, use relative paths
        3. output_dir is completely separate: prefer event_data_dir as cwd, make output_dir relative
        
        Returns:
        --------
        tuple: (cwd, relative_interim_path, relative_production_path)
            cwd: The base working directory to use
            relative_interim_path: Relative path from cwd to event_data_dir (for input data)
            relative_production_path: Relative path from cwd to output_dir (for output data)
        """
        try:
            # Get absolute paths for comparison
            abs_event_dir = os.path.abspath(self.event_data_dir)
            abs_output_dir = os.path.abspath(self.output_dir)
            
            # Case 1: output_dir is a subdirectory of event_data_dir
            try:
                relative_production = os.path.relpath(abs_output_dir, abs_event_dir)
                # Check if it's actually a subdirectory (doesn't start with ..)
                if not relative_production.startswith('..'):
                    return abs_event_dir, "", relative_production
            except ValueError:
                pass  # Can't make relative, try other cases
            
            # Case 2: Try to find common parent directory
            try:
                # Find common path prefix
                common_path = os.path.commonpath([abs_event_dir, abs_output_dir])
                
                # Calculate relative paths from common parent
                rel_interim = os.path.relpath(abs_event_dir, common_path)
                rel_production = os.path.relpath(abs_output_dir, common_path)
                
                # Use common path as cwd
                return common_path, rel_interim, rel_production
                
            except (ValueError, OSError):
                # Case 3: Paths are on different drives (Windows) or completely separate
                # Prefer using event_data_dir as cwd and try to make output_dir relative
                try:
                    # Try making output_dir relative to event_data_dir (even if it goes up)
                    rel_production = os.path.relpath(abs_output_dir, abs_event_dir)
                    # Even if it starts with .., we can use it
                    return abs_event_dir, "", rel_production
                except ValueError:
                    # Different drives - try the reverse: use output_dir's parent as cwd
                    output_parent = os.path.dirname(abs_output_dir)
                    output_basename = os.path.basename(abs_output_dir)
                    
                    try:
                        rel_interim = os.path.relpath(abs_event_dir, output_parent)
                        return output_parent, rel_interim, output_basename
                    except ValueError:
                        # Last resort: paths are on completely different drives
                        # Warn user and use event_data_dir as cwd with a warning
                        logging.warning(
                            f"Cannot create relative path between event_data_dir ({abs_event_dir}) "
                            f"and output_dir ({abs_output_dir}). They appear to be on different drives. "
                            f"Using event_data_dir as base. Output may not be saved correctly."
                        )
                        # Try to use a relative path even if it goes up many levels
                        # This might work if the package supports .. in paths
                        try:
                            rel_production = os.path.relpath(abs_output_dir, abs_event_dir)
                            return abs_event_dir, "", rel_production
                        except ValueError:
                            # Absolute last resort - use event_data_dir and hope for the best
                            return abs_event_dir, "", "processed"
                    
        except Exception as e:
            # Fallback to default behavior
            logging.warning(f"Could not calculate relative paths: {e}. Using default paths.")
            return self.event_data_dir, "", "processed"
    
    def _convert_timestamp_to_local(self, series):
        """Return timezone-naive Eastern wall-clock timestamps.

        Staged raw timestamps are already Eastern wall-clock values when they
        are timezone-naive. Timezone-aware inputs are converted to Eastern
        Time before removing timezone metadata for downstream compatibility.
        """
        if series.dt.tz is None:
            return series
        return series.dt.tz_convert(self.local_timezone).dt.tz_localize(None)
    
    def _calculate_optimal_workers(self):
        """
        Calculate optimal number of workers based on RAM and CPU constraints.
        
        Estimates memory usage per worker and calculates max workers that fit
        within RAM and CPU limits.
        
        Returns:
        --------
        int
            Optimal number of worker processes
        """
        try:
            # Get available system resources
            cpu_count = mp.cpu_count()
            
            # Get RAM information if psutil is available
            if PSUTIL_AVAILABLE:
                available_ram_gb = psutil.virtual_memory().available / (1024**3)
                # Use the smaller of available RAM or max_ram_gb constraint
                usable_ram_gb = min(available_ram_gb, self.max_ram_gb)
            else:
                # If psutil not available, use the max_ram_gb constraint directly
                # Only show warning in main process (not in worker processes)
                try:
                    if mp.current_process().name == 'MainProcess':
                        logging.warning("psutil not available. Worker calculation will use CPU count only.")
                except (RuntimeError, AttributeError):
                    pass  # Suppress if multiprocessing not fully initialized
                usable_ram_gb = self.max_ram_gb
                available_ram_gb = None
            
            # Estimate memory per worker (conservative estimate)
            # Each worker needs to:
            # - Load parquet file (can be large, estimate 2-4 GB per file)
            # - Process with pandas (additional 1-2 GB overhead)
            # - Store intermediate results (1-2 GB)
            # Total: ~4-8 GB per worker, use 6 GB as average
            estimated_memory_per_worker_gb = 6.0
            
            # Calculate workers based on RAM constraint
            # Reserve 20% of RAM for system and other processes
            reserved_ram_gb = usable_ram_gb * 0.2
            available_for_workers_gb = usable_ram_gb - reserved_ram_gb
            workers_by_ram = int(available_for_workers_gb / estimated_memory_per_worker_gb)
            
            # Calculate workers based on CPU constraint
            workers_by_cpu = min(cpu_count, self.max_cpu)
            
            # Use the minimum of RAM-based and CPU-based worker counts
            optimal_workers = min(workers_by_ram, workers_by_cpu)
            
            # Ensure at least 1 worker, but not more than constraints
            optimal_workers = max(1, min(optimal_workers, self.max_cpu))
            
            logging.info(f"Resource calculation:")
            if available_ram_gb is not None:
                logging.info(f"  - Available RAM: {available_ram_gb:.1f} GB (using max {self.max_ram_gb} GB)")
            else:
                logging.info(f"  - Using RAM constraint: {self.max_ram_gb} GB (psutil not available)")
            logging.info(f"  - CPU count: {cpu_count} (capped at {self.max_cpu})")
            logging.info(f"  - Estimated memory per worker: {estimated_memory_per_worker_gb:.1f} GB")
            logging.info(f"  - Workers by RAM: {workers_by_ram}")
            logging.info(f"  - Workers by CPU: {workers_by_cpu}")
            logging.info(f"  - Optimal workers: {optimal_workers}")
            
            return optimal_workers
            
        except Exception as e:
            # Fallback to conservative default if calculation fails
            logging.warning(f"Could not calculate optimal workers: {e}. Using conservative default.")
            return min(8, mp.cpu_count(), self.max_cpu)
        
    def discover_parquet_files(self):
        """Discover all parquet files in the per_signal directory structure."""
        per_signal_dir = os.path.join(self.event_data_dir, "per_signal")
        
        if not os.path.exists(per_signal_dir):
            raise FileNotFoundError(f"per_signal directory not found in: {self.event_data_dir}")
        
        # Find all parquet files in per_signal subdirectories
        pattern = os.path.join(per_signal_dir, "*", "*.parquet")
        parquet_files = glob.glob(pattern)
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in: {per_signal_dir}")
            
        logging.info(f"Found {len(parquet_files)} parquet files in per_signal structure")
        
        # Count unique signal IDs in parquet files
        unique_signals_in_files = set()
        for filepath in parquet_files:
            try:
                signal_id = self.extract_signal_id_from_path(filepath)
                unique_signals_in_files.add(signal_id)
            except ValueError:
                logging.warning(f"Could not extract signal ID from path: {filepath}")
        
        print(f"✓ Found {len(unique_signals_in_files)} unique signals in parquet files")
        
        return sorted(parquet_files)
    
    def extract_date_from_filename(self, filepath):
        """Extract date information from parquet filename."""
        filename = os.path.basename(filepath)
        
        # Try different filename patterns
        patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})\.parquet',        # 2025-9-1.parquet
            r'(\d{4})(\d{2})(\d{2})\.parquet',               # 20250901.parquet
            r'atspm-(\d{4})-(\d{1,2})-(\d{1,2})\.parquet'   # atspm-2025-9-1.parquet
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                year, month, day = map(int, match.groups())
                return year, month, day
                
        raise ValueError(f"Could not extract date from filename: {filename}")
    
    def extract_signal_id_from_path(self, filepath):
        """Extract signal ID from the per_signal directory path."""
        # Path format: /data2/ATSPM/raw_data/per_signal/{signal_id}/{YYYY-MM-DD}.parquet
        path_parts = Path(filepath).parts
        per_signal_index = None
        
        for i, part in enumerate(path_parts):
            if part == "per_signal":
                per_signal_index = i
                break
                
        if per_signal_index is None or per_signal_index + 1 >= len(path_parts):
            raise ValueError(f"Could not extract signal ID from path: {filepath}")
            
        signal_id = path_parts[per_signal_index + 1]
        return signal_id
    
    def filter_files_by_date(self, parquet_files):
        """Filter parquet files based on year, month, day constraints."""
        if self.year is None and self.month is None and self.day is None:
            return parquet_files
            
        filtered_files = []
        for filepath in parquet_files:
            try:
                year, month, day = self.extract_date_from_filename(filepath)
                
                if self.year is not None and year != self.year:
                    continue
                if self.month is not None and month != self.month:
                    continue
                if self.day is not None and day != self.day:
                    continue
                    
                filtered_files.append(filepath)
            except ValueError:
                logging.warning(f"Could not parse date from {filepath}, skipping")
                continue
                
        return filtered_files
    
    def filter_files_by_signals(self, parquet_files, signal_ids):
        """Filter parquet files to only include specified signal IDs."""
        if signal_ids is None:
            return parquet_files
            
        filtered_files = []
        for filepath in parquet_files:
            try:
                signal_id = self.extract_signal_id_from_path(filepath)
                if signal_id in signal_ids:
                    filtered_files.append(filepath)
            except ValueError:
                logging.warning(f"Could not extract signal ID from {filepath}, skipping")
                continue
                
        return filtered_files
    
    def show_parquet_files_summary(self, files_to_process):
        """Display a summary of all parquet files that will be processed."""
        if not files_to_process:
            print("No parquet files will be processed.")
            return
            
        print(f"\n{'='*60}")
        print(f"PARQUET FILES PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files to process: {len(files_to_process)}")
        
        # Group by signal ID
        signal_groups = {}
        for filepath in files_to_process:
            try:
                signal_id = self.extract_signal_id_from_path(filepath)
                if signal_id not in signal_groups:
                    signal_groups[signal_id] = []
                signal_groups[signal_id].append(filepath)
            except ValueError:
                continue
        
        print(f"Unique signals with data: {len(signal_groups)}")
        print(f"{'='*60}\n")
    
    def load_event_data(self, filepath):
        """Load event data from a specific parquet file."""
        logging.info(f"Loading event data from: {filepath}")
        df = pd.read_parquet(filepath)
        
        # Ensure timestamp column is datetime
        if 'timeStamp' in df.columns:
            df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='mixed')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df = df.rename(columns={'timestamp': 'timeStamp'})
        else:
            raise ValueError("No timestamp column found. Expected 'timeStamp' or 'timestamp'")
        
        df['timeStamp'] = self._convert_timestamp_to_local(df['timeStamp'])
            
        logging.info(f"Loaded {len(df)} event records")
        return df
        
    def check_config_file_exists(self, signal_id):
        """
        Check if config file exists for a signal.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID to check
            
        Returns:
        --------
        bool
            True if config file exists, False otherwise
        """
        config_file = os.path.join(self.config_dir, f"{signal_id}.csv")
        return os.path.exists(config_file)
    
    def load_config_data(self, signal_id):
        """Load configuration data for a specific signal."""
        config_file = os.path.join(self.config_dir, f"{signal_id}.csv")
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        logging.info(f"Loading config for signal {signal_id}")
        df = pd.read_csv(config_file)
        
        # Convert float columns to int where applicable
        for col in df.columns:
            if df[col].dtype == 'float64':
                if df[col].dropna().apply(lambda x: x.is_integer()).all():
                    df[col] = df[col].astype('Int64')
                    
        return df
        
    def get_available_signals(self):
        """Get list of available signal IDs from config directory and per_signal structure."""
        # Get signals from config directory
        csv_files = [f for f in os.listdir(self.config_dir) if f.endswith('.csv')]
        config_signals = [f.replace('.csv', '') for f in csv_files]
        
        # Get signals from per_signal directory structure
        per_signal_dir = os.path.join(self.event_data_dir, "per_signal")
        per_signal_signals = []
        if os.path.exists(per_signal_dir):
            per_signal_signals = [d for d in os.listdir(per_signal_dir) 
                                if os.path.isdir(os.path.join(per_signal_dir, d))]
        
        # Combine and deduplicate
        all_signals = list(set(config_signals + per_signal_signals))
        
        # Print discovery summary
        print(f"✓ Available signals discovery:")
        print(f"  - Config directory signals: {len(config_signals)}")
        print(f"  - Per_signal directory signals: {len(per_signal_signals)}")
        print(f"  - Total unique signals found: {len(all_signals)}")
        
        return sorted(all_signals)
    
    def _process_single_file_worker(self, filepath, signal_ids=None, batch_size=None):
        from feature_extractor.checked_runner import execute_job, job_from_path
        from feature_extractor.run_store import RunStore
        job = job_from_path(filepath, self.features)
        if signal_ids and job['signal_id'] not in signal_ids:
            return job['signal_id'], job['date_str'], None
        store = RunStore(Path(self.output_dir) / 'pipeline_runs.sqlite3')
        run_id = store.start_run()
        result = execute_job(job, {'db_path': store.path, 'run_id': run_id,
            'config_dir': self.config_dir, 'output_dir': self.output_dir,
            'engine': ENGINE_ID, 'overwrite': self.overwrite})
        store.finish_run(run_id, result['status'], result)
        return job['signal_id'], job['date_str'], result['status'] == 'complete'


    def process_single_signal(self, signal_id, df_events, df_config, year, month, day,
                              input_file_path=None, input_file_mtime=None, batch_size=None):
        from feature_extractor.checked_execution import execute_frames
        from feature_extractor.config_validator import validate_config
        from feature_extractor.feature_registry import resolve_features
        from feature_extractor.result_types import FeatureResult, completion_status
        from feature_extractor.run_store import RunStore, file_signature
        date_str = f'{year:04d}-{month:02d}-{day:02d}'
        validation = validate_config(Path(self.config_dir) / f'{signal_id}.csv', signal_id=signal_id, data_date=date_str)
        selected = resolve_features(self.features)
        store = RunStore(Path(self.output_dir) / 'pipeline_runs.sqlite3')
        run_id = store.start_run()
        attempt = store.start_attempt(run_id, signal_id, date_str,
            file_signature(input_file_path) if input_file_path else 'in_memory_unverified',
            validation.config_hash, ENGINE_ID, selected)
        try:
            if validation.status != 'valid':
                results = {n: FeatureResult(n, 'blocked', validation.reason_code) for n in selected}
            else:
                results = execute_frames(signal_id, date_str, df_events, validation, self.output_dir,
                                         selected)
            status = completion_status(results, selected)
            store.finish_attempt(attempt, status, results)
            store.finish_run(run_id, status, {n: r.to_dict() for n, r in results.items()})
            return status == 'complete'
        except Exception as exc:
            results = {n: FeatureResult(n, 'failed', 'job_error', str(exc)) for n in selected}
            store.finish_attempt(attempt, 'partial', results, 'job_error')
            store.finish_run(run_id, 'partial', {'error': str(exc)})
            return False

                
            # Don't raise - allow processing to continue even if aggregation fails
            
                
    def _get_all_feature_directories(self):
        """Return all feature output directories that should be checked for completed runs."""
        base_path = os.path.join(
            self.output_dir,
            "fdot_d5",
            "feature_extraction"
        )
        
        # Define all feature directories based on README.md structure
        feature_dirs = [
            # Cycle-level features - Vehicle Signal
            os.path.join(base_path, "feature", "cycle", "vehicle_signal", "spat"),
            
            # Cycle-level features - Vehicle Traffic
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "conflict"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "conflict_pet"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "dilemma_running"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "gap"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "headway"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "occupancy"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "red_running"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "shockwave"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "split_failure"),
            os.path.join(base_path, "feature", "cycle", "vehicle_traffic", "volume"),
            
            # Cycle-level features - Pedestrian Traffic
            os.path.join(base_path, "feature", "cycle", "pedestrian_traffic", "activity"),
            os.path.join(base_path, "feature", "cycle", "pedestrian_traffic", "conflict_propensity"),
            os.path.join(base_path, "feature", "cycle", "pedestrian_traffic", "delay"),
            
            # Signal Profile features - Cycle Profiles
            os.path.join(base_path, "signal_profile", "cycle", "vehicle_signal"),
            os.path.join(base_path, "signal_profile", "cycle", "pedestrian_signal"),
            
            # Signal Profile features - Phase Profiles
            os.path.join(base_path, "signal_profile", "phase", "vehicle_signal"),
            os.path.join(base_path, "signal_profile", "phase", "pedestrian_signal"),
        ]
        
        return feature_dirs
    
    def _get_processed_feature_dir(self):
        """Return the canonical processed feature directory to detect completed runs (legacy method)."""
        # Keep this for backward compatibility, but now returns the first feature dir
        return self._get_all_feature_directories()[0]


    def _build_processed_index(self):
        # File presence is never evidence of successful extraction.
        self._processed_index = {}


    def _is_signal_date_already_processed(self, signal_id, year, month, day, input_file_path=None):
        # Reuse is validated by checked_runner with input, configuration, policy
        # and output signatures. Legacy filesystem/any-feature indexes cannot skip it.
        return False


    def process_all_parquet_files(self, signal_ids=None, use_multiprocessing=True):
        from feature_extractor.checked_runner import run_checked
        from uuid import uuid4
        report = Path(self.output_dir) / 'runs' / f'{uuid4().hex}.json'
        config = ProcessingConfig(event_dir=self.event_data_dir, config_dir=self.config_dir,
            output_dir=self.output_dir, year=self.year, month=self.month, day=self.day,
            signals=signal_ids, features=self.features, overwrite=self.overwrite,
            max_workers=self.max_workers if use_multiprocessing else 1,
            summary_file=str(report))
        run_checked(config)
        results = {}
        for row in json.loads(report.read_text())['outcomes']:
            results.setdefault(row['date_str'], {})[row['signal_id']] = row['status'] == 'complete'
        return results

    
    def detect_new_or_updated_files(self, signal_ids=None, max_age_hours=24):
        """
        Detect new or updated parquet files that need processing.
        
        Parameters:
        -----------
        signal_ids : list, optional
            List of signal IDs to check. If None, checks all available signals
        max_age_hours : int
            Maximum age in hours for files to be considered "new" (default: 24)
        
        Returns:
        --------
        list
            List of file paths that are new or have been updated
        """
        if signal_ids is None:
            signal_ids = self.get_available_signals()
        
        # Discover all parquet files
        all_parquet_files = self.discover_parquet_files()
        date_filtered_files = self.filter_files_by_date(all_parquet_files)
        signal_filtered_files = self.filter_files_by_signals(date_filtered_files, signal_ids)
        
        new_or_updated_files = []
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for filepath in signal_filtered_files:
            try:
                # Check file modification time
                if os.path.exists(filepath):
                    file_mtime = os.path.getmtime(filepath)
                    
                    # Check if file is new or recently updated
                    if file_mtime >= cutoff_time:
                        # Also check if it's already processed (with mtime check) unless overwrite is enabled
                        year, month, day = self.extract_date_from_filename(filepath)
                        signal_id = self.extract_signal_id_from_path(filepath)
                        
                        if self.overwrite or not self._is_signal_date_already_processed(
                            signal_id, year, month, day, input_file_path=filepath
                        ):
                            new_or_updated_files.append(filepath)
                    else:
                        # File is older, but check if it needs reprocessing
                        year, month, day = self.extract_date_from_filename(filepath)
                        signal_id = self.extract_signal_id_from_path(filepath)
                        
                        # Check if file was modified after last processing
                        # Use the existing method which checks both log and file mtime
                        # Skip check if overwrite is enabled
                        if self.overwrite or not self._is_signal_date_already_processed(
                            signal_id, year, month, day, input_file_path=filepath
                        ):
                            new_or_updated_files.append(filepath)
            except Exception as e:
                logging.warning(f"Error checking file {filepath}: {e}")
                continue
        
        return new_or_updated_files
    
    def process_all_signals(self, signal_ids=None):
        nested = self.process_all_parquet_files(signal_ids)
        return {signal: status for signals in nested.values() for signal, status in signals.items()}


def _has_processing_failures(result):
    """Return True when nested processing results contain an explicit failure."""
    if result is False:
        return True
    if isinstance(result, dict):
        if "error" in result:
            return True
        return any(_has_processing_failures(value) for value in result.values())
    if isinstance(result, (list, tuple, set)):
        return any(_has_processing_failures(value) for value in result)
    return False


def main():
    """All CLI modes use the same validated execution and completion contract."""
    from feature_extractor.checked_runner import run_checked
    try:
        args = ConfigParser.parse_args()
        return run_checked(ConfigParser.create_config_from_args(args))
    except BlockingIOError:
        print('Another ATSPM pipeline owns the shared lock.', file=sys.stderr)
        return 75
    except Exception as exc:
        logging.exception('Pipeline failed')
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
