"""
Configuration module for ATSPM processing.

This module contains configuration classes and argument parsing logic
that was previously embedded in the main processing script.
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional


def load_signals_from_file(file_path: str) -> List[str]:
    """Load signal IDs from a text file (one signal ID per line)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Signal list file not found: {file_path}")
    
    signals = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            signal_id = line.strip()
            if signal_id and not signal_id.startswith('#'):  # Skip empty lines and comments
                signals.append(signal_id)
    
    if not signals:
        raise ValueError(f"No valid signal IDs found in file: {file_path}")
    
    return signals


@dataclass
class ProcessingConfig:
    """Configuration class for ATSPM processing parameters."""
    
    # Required parameters
    event_dir: str
    config_dir: str
    
    # Optional parameters
    output_dir: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    signals: Optional[List[str]] = None
    
    # Processing mode flags
    batch_mode: bool = True
    single_date_mode: bool = False
    
    # Logging and progress options
    verbose: bool = False
    no_progress_bar: bool = False
    
    # Multiprocessing options
    max_workers: Optional[int] = None
    
    # Feature selection
    features: Optional[List[str]] = None
    
    # Overwrite mode
    overwrite: bool = False
    
    jobs_file: Optional[str] = None
    quality_report: Optional[str] = None
    config_decisions: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    run_id: Optional[str] = None
    summary_file: Optional[str] = None
    dry_run: bool = False
    lock_held: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        # Check if directories exist
        if not os.path.exists(self.event_dir):
            raise FileNotFoundError(f"Event directory does not exist: {self.event_dir}")
            
        if not os.path.exists(self.config_dir):
            raise FileNotFoundError(f"Config directory does not exist: {self.config_dir}")
        
        # Validate single date mode requirements
        if self.single_date_mode and (self.year is None or self.month is None or self.day is None):
            raise ValueError("Single date mode requires year, month, and day to be specified")
    
    @property
    def date_str(self) -> str:
        """Get formatted date string if date is specified."""
        if self.year and self.month and self.day:
            return f"{self.year}-{self.month:02d}-{self.day:02d}"
        return "all_dates"
    
    @property
    def signals_str(self) -> str:
        """Get formatted signals string."""
        if self.signals:
            return ", ".join(self.signals)
        return "all_available_signals"


class ConfigParser:
    """Parser for command line arguments and configuration creation."""
    
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        parser = argparse.ArgumentParser(
            description="Process ATSPM event data from local parquet files with per_signal structure",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=r"""
Examples:
  # Process all parquet files in per_signal directory structure
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs

  # Process specific date
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs --year 2025 --month 9 --day 1

  # Process specific signals only
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs --signals SIG001 SIG002

  # Process signals from a text file
  python feature_extractor/process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole --config-dir C:\\Users\\YJ\\Documents\\python_scripts\\atspm\\data\\signal_config_2025\\lanes --max-workers 6

  # Process with custom output directory and multiprocessing
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs --output-dir /path/to/output --max-workers 8

  # Extract only specific features
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs --features volume headway spat

  # Overwrite existing processed data
  python process_local_atspm.py --event-dir D:\\01_Data\\01_ATSPM\\Seminole\\per_signal --config-dir /path/to/configs --overwrite
            """
        )
        
        # Required arguments
        parser.add_argument(
            "--event-dir", 
            required=True,
            help="Base directory containing per_signal subdirectories with parquet event files"
        )
        
        parser.add_argument(
            "--config-dir", 
            default="/data2/ATSPM/config_data/signal_config_2025/lanes",
            help="Directory containing signal configuration CSV files (default: /data2/ATSPM/config_data/signal_config_2025/lanes)"
        )
        
        # Optional arguments
        parser.add_argument(
            "--output-dir",
            help="Output directory for processed results (default: event-dir/processed)"
        )
        
        parser.add_argument(
            "--year",
            type=int,
            help="Specific year to process (default: process all years)"
        )
        
        parser.add_argument(
            "--month",
            type=int,
            help="Specific month to process (default: process all months)"
        )
        
        parser.add_argument(
            "--day",
            type=int,
            help="Specific day to process (default: process all days)"
        )
        
        parser.add_argument(
            "--signals",
            nargs="*",
            help="Specific signal IDs to process (default: process all available signals)"
        )
        
        parser.add_argument(
            "--signals-file",
            help="File containing signal IDs to process (one signal ID per line)"
        )
        
        # Processing mode flags
        parser.add_argument(
            "--batch",
            action="store_true",
            help="Process all parquet files in the event directory (default behavior)"
        )
        
        parser.add_argument(
            "--single-date",
            action="store_true",
            help="Process single date (requires year, month, day to be specified)"
        )
        
        # Logging and progress options
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        parser.add_argument(
            "--no-progress-bar",
            action="store_true", default=True,
            help="Disable tqdm progress bars (use simple text progress instead)"
        )
        
        parser.add_argument(
            "--max-workers",
            type=int,
            help="Maximum number of worker processes for parallel processing (default: CPU count)"
        )
        
        parser.add_argument(
            "--auto",
            action="store_true",
            help="Automated mode: only process new or updated files (default: process all files)"
        )
        
        parser.add_argument(
            "--max-age-hours",
            type=int,
            default=24,
            help="Maximum age in hours for files to be considered 'new' in auto mode (default: 24)"
        )
        
        parser.add_argument(
            "--features",
            nargs="*",
            help="Specific features to extract (default: extract all features). "
                 "Valid features: vehicle_phase_profile, pedestrian_phase_profile, "
                 "vehicle_cycle_profile, pedestrian_cycle_profile, spat, volume, "
                 "occupancy, split_failure, headway, conflict, gap, pet, shockwave, "
                 "red_running, dilemma_running, pedestrian_activity, pedestrian_delay, "
                 "conflict_propensity, conflict_pet. Dependencies are included automatically."
        )
        
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing processed data even if it has already been processed (default: skip already processed files)"
        )
        
        
        for option in ('jobs-file', 'quality-report', 'config-decisions', 'start-date', 'end-date', 'run-id', 'summary-file'):
            parser.add_argument('--' + option)
        parser.add_argument('--dry-run', action='store_true', help='Validate and write exact job manifest without extraction')
        parser.add_argument('--lock-held', action='store_true', help=argparse.SUPPRESS)
        return parser
    
    @staticmethod
    def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = ConfigParser.create_parser()
        return parser.parse_args(args)
    
    @staticmethod
    def create_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
        """Create ProcessingConfig from parsed arguments."""
        # Determine processing mode
        batch_mode = args.batch or not args.single_date  # Default to batch mode
        single_date_mode = args.single_date
        
        # Handle signal list - prioritize signals-file over signals argument
        signals = args.signals
        if args.signals_file:
            if args.signals:
                print("Warning: Both --signals-file and --signals specified. Using signal IDs from file.")
            signals = load_signals_from_file(args.signals_file)
            print(f"Loaded {len(signals)} signal IDs from file: {args.signals_file}")
        
        config = ProcessingConfig(
            event_dir=args.event_dir,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            year=args.year,
            month=args.month,
            day=args.day,
            signals=signals,
            batch_mode=batch_mode,
            single_date_mode=single_date_mode,
            verbose=args.verbose,
            no_progress_bar=args.no_progress_bar,
            max_workers=args.max_workers,
            features=args.features,
            overwrite=args.overwrite
        )
        
        for name in ('jobs_file', 'quality_report', 'config_decisions', 'start_date', 'end_date', 'run_id', 'summary_file', 'dry_run', 'lock_held'):
            setattr(config, name, getattr(args, name, None))
        # Add auto mode attributes
        config.auto_mode = args.auto
        config.max_age_hours = args.max_age_hours
        
        # Mark if signals came from file for tracking
        if args.signals_file:
            config._signals_from_file = True
            
        return config
    
    @staticmethod
    def create_config_from_dict(config_dict: dict) -> ProcessingConfig:
        """Create ProcessingConfig from a dictionary."""
        return ProcessingConfig(**config_dict)


def load_config_from_file(config_file: str) -> ProcessingConfig:
    """Load configuration from a JSON or YAML file."""
    import json
    
    with open(config_file, 'r') as f:
        if config_file.endswith('.json'):
            config_dict = json.load(f)
        elif config_file.endswith(('.yml', '.yaml')):
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
        else:
            raise ValueError("Config file must be JSON (.json) or YAML (.yml, .yaml)")
    
    return ConfigParser.create_config_from_dict(config_dict)


def save_config_to_file(config: ProcessingConfig, config_file: str):
    """Save configuration to a JSON or YAML file."""
    import json
    
    config_dict = {
        'event_dir': config.event_dir,
        'config_dir': config.config_dir,
        'output_dir': config.output_dir,
        'year': config.year,
        'month': config.month,
        'day': config.day,
        'signals': config.signals,
        'batch_mode': config.batch_mode,
        'single_date_mode': config.single_date_mode,
        'verbose': config.verbose,
        'no_progress_bar': config.no_progress_bar,
        'max_workers': config.max_workers
    }
    
    with open(config_file, 'w') as f:
        if config_file.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        elif config_file.endswith(('.yml', '.yaml')):
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
        else:
            raise ValueError("Config file must be JSON (.json) or YAML (.yml, .yaml)")


# Example configuration files
EXAMPLE_CONFIG_JSON = """{
  "event_dir": "/path/to/parquet/files",
  "config_dir": "/path/to/configs",
  "output_dir": "/path/to/output",
  "year": 2025,
  "month": 9,
  "day": 1,
  "signals": ["SIG001", "SIG002"],
  "batch_mode": false,
  "single_date_mode": true,
  "verbose": true,
  "no_progress_bar": false,
  "max_workers": 4
}"""

EXAMPLE_CONFIG_YAML = """event_dir: /path/to/parquet/files
config_dir: /path/to/configs
output_dir: /path/to/output
year: 2025
month: 9
day: 1
signals:
  - SIG001
  - SIG002
batch_mode: false
single_date_mode: true
verbose: true
no_progress_bar: false
max_workers: 4"""
