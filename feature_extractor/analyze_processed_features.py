"""
Analyze the structure of processed ATSPM feature extraction Parquet files.

This script examines the column structure and content of Parquet files generated
by the ATSPM feature extraction pipeline. It can analyze individual files
or provide a comprehensive overview of all available feature types.

Usage:
    python analyze_processed_features.py
    python feature_extractor/analyze_processed_features.py --signal ORC-0003 --date 2025-09-07
    python feature_extractor/analyze_processed_features.py --all-features
"""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame


import pandas as pd
import numpy as np
import os
import glob
import argparse
from typing import Dict, List, Any
import warnings
import json
import hashlib
import sqlite3
warnings.filterwarnings('ignore')


class ATSPMFeatureAnalyzer:
    """Analyze the structure and content of ATSPM processed feature files."""
    
    def __init__(self, base_dir: str = r"/data2/ATSPM/processed_data/fdot_d5/feature_extraction", 
                 cache_dir: str = None):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing the processed feature files
        cache_dir : str, optional
            Directory to store analysis cache. If None, uses .analysis_cache in base_dir
        """
        self.base_dir = base_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(base_dir, '.analysis_cache')
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        self.feature_types = {
            # Cycle-level features
            'feature/cycle/vehicle_signal': ['spat'],
            'feature/cycle/vehicle_traffic': ['conflict', 'conflict_pet', 'dilemma_running', 'gap', 'headway', 'pet',
                                             'occupancy', 'red_running', 'split_failure', 'volume'],
            'feature/cycle/pedestrian_traffic': ['activity', 'conflict_propensity', 'delay'],
            # Signal profile features (files are directly in signal directories, not in subdirectories)
            'signal_profile/cycle/vehicle_signal': [''],  # Empty string means files are directly in signal dir
            'signal_profile/cycle/pedestrian_signal': [''],
            'signal_profile/phase/vehicle_signal': [''],
            'signal_profile/phase/pedestrian_signal': ['']
        }
        
    def get_available_signals(self) -> List[str]:
        """Get list of all available signal IDs."""
        signals = set()
        for feature_category, feature_list in self.feature_types.items():
            for feature_type in feature_list:
                # Handle empty feature_type (files directly in signal directory)
                if feature_type == '':
                    feature_dir = os.path.join(self.base_dir, feature_category)
                else:
                    feature_dir = os.path.join(self.base_dir, feature_category, feature_type)
                    
                if os.path.exists(feature_dir):
                    signal_dirs = [d for d in os.listdir(feature_dir) 
                                 if os.path.isdir(os.path.join(feature_dir, d))]
                    signals.update(signal_dirs)
        return sorted(list(signals))
    
    def get_available_dates(self, signal_id: str) -> List[str]:
        """Get list of available dates for a specific signal."""
        dates = set()
        for feature_category, feature_list in self.feature_types.items():
            for feature_type in feature_list:
                # Handle empty feature_type (files directly in signal directory)
                if feature_type == '':
                    feature_dir = os.path.join(self.base_dir, feature_category, signal_id)
                else:
                    feature_dir = os.path.join(self.base_dir, feature_category, feature_type, signal_id)
                    
                if os.path.exists(feature_dir):
                    parquet_files = glob.glob(os.path.join(feature_dir, "*.parquet"))
                    for parquet_file in parquet_files:
                        date = os.path.basename(parquet_file).replace('.parquet', '')
                        dates.add(date)
        return sorted(list(dates))
    
    def load_feature_file(self, signal_id: str, date: str, feature_category: str, feature_type: str) -> Any:
        """
        Load a Parquet file and return its content.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID (e.g., 'ORC-0006')
        date : str
            Date in YYYY-MM-DD format (e.g., '2025-09-01')
        feature_category : str
            Feature category (e.g., 'feature/cycle/vehicle_signal', 'signal_profile/cycle/vehicle_signal')
        feature_type : str
            Feature type (e.g., 'spat', 'volume', 'conflict', '' for direct files)
            
        Returns:
        --------
        Any
            Loaded data from Parquet file (usually DataFrame or dict)
        """
        # Handle empty feature_type (files directly in signal directory)
        if feature_type == '':
            file_path = os.path.join(self.base_dir, feature_category, signal_id, f"{date}.parquet")
        else:
            file_path = os.path.join(self.base_dir, feature_category, feature_type, signal_id, f"{date}.parquet")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            return read_frame(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_data_structure(self, data: Any, feature_name: str) -> Dict[str, Any]:
        """
        Analyze the structure of loaded data.
        
        Parameters:
        -----------
        data : Any
            Data loaded from Parquet file
        feature_name : str
            Name of the feature for identification
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results including type, shape, columns, etc.
        """
        analysis = {
            'feature_name': feature_name,
            'data_type': type(data).__name__,
            'structure': {}
        }
        
        if isinstance(data, pd.DataFrame):
            analysis['structure'] = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': dict(data.dtypes),
                'index_type': type(data.index).__name__,
                'has_nulls': data.isnull().sum().sum() > 0,
                'null_columns': data.isnull().sum()[data.isnull().sum() > 0].to_dict() if data.isnull().sum().sum() > 0 else {},
                'memory_usage': data.memory_usage(deep=True).sum(),
                'sample_data': data.head(3).to_dict() if len(data) > 0 else {}
            }
        elif isinstance(data, dict):
            analysis['structure'] = {
                'keys': list(data.keys()),
                'key_types': {k: type(v).__name__ for k, v in data.items()},
                'total_items': len(data)
            }
            
            # If dict contains DataFrames, analyze them
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    analysis['structure'][f'{key}_df_shape'] = value.shape
                    analysis['structure'][f'{key}_df_columns'] = list(value.columns)
        elif isinstance(data, (list, tuple)):
            analysis['structure'] = {
                'length': len(data),
                'element_type': type(data[0]).__name__ if len(data) > 0 else 'empty',
                'sample_elements': data[:3] if len(data) > 0 else []
            }
        else:
            analysis['structure'] = {
                'value': str(data)[:200] if len(str(data)) > 200 else str(data),
                'length': len(data) if hasattr(data, '__len__') else 'N/A'
            }
            
        return analysis
    
    def analyze_signal_date(self, signal_id: str, date: str, verbose: bool = True, 
                           force: bool = False) -> Dict[str, Any]:
        """
        Analyze all available features for a specific signal and date.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID to analyze
        date : str
            Date to analyze
        verbose : bool
            Whether to print detailed information
        force : bool
            If True, force re-analysis even if cached results exist
            
        Returns:
        --------
        Dict[str, Any]
            Complete analysis results
        """
        # Check cache first unless force is True
        cached_results = None
        if not force:
            cached_results = self._load_from_cache(signal_id, date)
            if cached_results is not None:
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"LOADING FROM CACHE: {signal_id} | DATE: {date}")
                    print(f"{'='*80}")
                    # Print summary from cache
                    if 'summary' in cached_results:
                        summary = cached_results['summary']
                        print(f"Total features attempted: {summary.get('total_features_attempted', 'N/A')}")
                        print(f"Successfully loaded: {summary.get('successful_loads', 'N/A')}")
                        print(f"Success rate: {summary.get('success_rate', 'N/A')}")
                    print(f"{'='*80}\n")
                return cached_results
        
        results = {
            'signal_id': signal_id,
            'date': date,
            'available_features': {},
            'summary': {}
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ANALYZING SIGNAL: {signal_id} | DATE: {date}")
            if force:
                # Check if cache exists to show message
                cache_path = self._get_cache_path(signal_id, date)
                if os.path.exists(cache_path):
                    print("(Force re-analysis - ignoring cache)")
            print(f"{'='*80}")
        
        total_features = 0
        successful_loads = 0
        
        for feature_category, feature_list in self.feature_types.items():
            category_results = {}
            
            for feature_type in feature_list:
                feature_name = f"{feature_category}/{feature_type}"
                total_features += 1
                
                try:
                    data = self.load_feature_file(signal_id, date, feature_category, feature_type)
                    if data is not None:
                        analysis = self.analyze_data_structure(data, feature_name)
                        category_results[feature_type] = analysis
                        successful_loads += 1
                        
                        if verbose:
                            self._print_feature_analysis(analysis)
                            
                except FileNotFoundError:
                    if verbose:
                        print(f"❌ {feature_name}: File not found")
                except Exception as e:
                    if verbose:
                        print(f"❌ {feature_name}: Error - {e}")
            
            if category_results:
                results['available_features'][feature_category] = category_results
        
        results['summary'] = {
            'total_features_attempted': total_features,
            'successful_loads': successful_loads,
            'success_rate': f"{successful_loads/total_features*100:.1f}%" if total_features > 0 else "0%"
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SUMMARY FOR {signal_id} ({date})")
            print(f"{'='*80}")
            print(f"Total features attempted: {total_features}")
            print(f"Successfully loaded: {successful_loads}")
            print(f"Success rate: {results['summary']['success_rate']}")
            print(f"{'='*80}\n")
        
        # Save to cache
        self._save_to_cache(results, signal_id, date)
        if verbose:
            print(f"💾 Analysis results cached")
        
        return results
    
    def _parse_column_structure(self, columns: List[str]) -> Dict[str, set]:
        """
        Parse column names into their components: phase_type, feature_name, phase_no, lane_type.
        
        Format: {phase_type}{feature_name}{phase_no}{lane_type}
        Example: greenConflict0.5Phase2T -> phase_type: green, feature_name: Conflict0.5, 
                 phase_no: Phase2, lane_type: T
        
        Parameters:
        -----------
        columns : List[str]
            List of column names
            
        Returns:
        --------
        Dict[str, set]
            Dictionary with keys: 'phase_types', 'feature_names', 'phase_numbers', 'lane_types'
        """
        import re
        
        # Basic columns to skip
        basic_columns = {'signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength', 'date', 'k'}
        
        phase_types = set()
        feature_names = set()
        phase_numbers = set()
        lane_types = set()
        
        # Known phase types (order matters - longer ones first)
        known_phase_types = ['redClearance', 'green', 'yellow', 'red', 'pedestrian']
        
        # Pattern to match phase numbers: Phase followed by digits and optional letters
        phase_pattern = r'(Phase\d+[A-Z]*)'
        
        # Pattern to match lane types at the end: T, R, L, or combinations
        lane_pattern = r'([TRL]+)$'
        
        # Pattern for channel numbers
        channel_pattern = r'^channelNos\d+[TRL]*'
        
        for col in columns:
            # Skip basic columns
            if col in basic_columns:
                continue
            
            # Check if it's a channel number column
            if re.match(channel_pattern, col):
                # Extract channel info and lane type
                match = re.match(r'^channelNos(\d+)([TRL]*)$', col)
                if match:
                    if match.group(2):
                        lane_types.add(match.group(2))
                feature_names.add('channelNos')
                continue
            
            original_col = col
            
            # Handle pedestrian cycle suffixes (CurrCycle, PrevCycle) before phase number extraction
            cycle_suffix = ''
            if 'CurrCycle' in col:
                cycle_suffix = 'CurrCycle'
                col = col.replace('CurrCycle', '')
            elif 'PrevCycle' in col:
                cycle_suffix = 'PrevCycle'
                col = col.replace('PrevCycle', '')
            
            # Extract phase number first (before removing lane type, in case phase number has letters)
            phase_match = re.search(phase_pattern, col)
            phase_no = ''
            if phase_match:
                phase_no = phase_match.group(1)
                phase_start = phase_match.start()
                phase_end = phase_match.end()
                
                # Check if lane type is after phase number
                remaining_after_phase = col[phase_end:]
                lane_match_after_phase = re.match(lane_pattern, remaining_after_phase)
                if lane_match_after_phase:
                    lane_type = lane_match_after_phase.group(1)
                    lane_types.add(lane_type)
                else:
                    # Check if lane type might be in the phase number itself (Phase2T -> Phase2 + T)
                    # Also check for pedestrian cycle indicators (Phase2C -> Phase2, but C is part of CurrCycle)
                    phase_no_clean = re.match(r'(Phase\d+)([TRL]+)$', phase_no)
                    if phase_no_clean and cycle_suffix == '':
                        phase_no = phase_no_clean.group(1)
                        lane_type = phase_no_clean.group(2)
                        lane_types.add(lane_type)
                
                phase_numbers.add(phase_no)
                # Remove phase number and lane type from col
                col = col[:phase_start]
            else:
                # No phase number, check for lane type at the end
                lane_match = re.search(lane_pattern, col)
                if lane_match:
                    lane_type = lane_match.group(1)
                    col = col[:-len(lane_type)]
                    lane_types.add(lane_type)
            
            # Extract phase type (must be at the start)
            phase_type = ''
            for pt in known_phase_types:
                if col.startswith(pt):
                    phase_type = pt
                    col = col[len(pt):]
                    phase_types.add(phase_type)
                    break
            
            # Whatever remains is the feature name
            if col:
                # Handle special cases for pedestrian features
                if 'pedestrian' in original_col.lower():
                    # Extract pedestrian-specific features
                    if 'Activity' in original_col:
                        feature_part = re.match(r'.*?(pedestrianActivity\d+)', original_col)
                        if feature_part:
                            feature_names.add(feature_part.group(1))
                        else:
                            feature_names.add(col)
                    elif 'Delay' in original_col:
                        feature_part = re.match(r'.*?(pedestrianDelay)', original_col)
                        if feature_part:
                            feature_names.add(feature_part.group(1))
                        else:
                            feature_names.add(col)
                    elif 'conflictPropensity' in original_col:
                        feature_names.add('conflictPropensity')
                    else:
                        feature_names.add(col)
                else:
                    # Normalize feature name - use lowercase for consistency
                    # But preserve camelCase for capitalized features
                    feature_name = col
                    # If it starts with uppercase and we have lowercase version, prefer lowercase
                    # This handles cases like "Volume" vs "volume"
                    if feature_name and feature_name[0].isupper():
                        # Check if there's a lowercase version elsewhere
                        feature_name_lower = feature_name[0].lower() + feature_name[1:] if len(feature_name) > 1 else feature_name.lower()
                        # Use the lowercase version for consistency
                        feature_names.add(feature_name_lower)
                    else:
                        feature_names.add(feature_name)
            elif not phase_type and not phase_no:
                # No phase type, no phase number - this might be a standalone feature
                feature_names.add(original_col)
        
        return {
            'phase_types': sorted(phase_types),
            'feature_names': sorted(feature_names),
            'phase_numbers': sorted(phase_numbers, key=lambda x: (len(x), x)),
            'lane_types': sorted(lane_types)
        }
    
    def _get_cache_path(self, signal_id: str, date: str) -> str:
        """
        Get the cache file path for a specific signal and date.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        str
            Path to cache file
        """
        # Create a hash of base_dir to ensure cache is specific to data location
        base_hash = hashlib.md5(self.base_dir.encode()).hexdigest()[:8]
        cache_filename = f"{signal_id}_{date}_{base_hash}.json"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_from_cache(self, signal_id: str, date: str) -> Dict[str, Any]:
        """
        Load analysis results from cache if they exist.
        
        Parameters:
        -----------
        signal_id : str
            Signal ID
        date : str
            Date in YYYY-MM-DD format
            
        Returns:
        --------
        Dict[str, Any] or None
            Cached results if found, None otherwise
        """
        cache_path = self._get_cache_path(signal_id, date)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                # If cache is corrupted, return None to re-analyze
                print(f"Warning: Could not load cache file {cache_path}: {e}")
                return None
        return None
    
    def _save_to_cache(self, results: Dict[str, Any], signal_id: str, date: str):
        """
        Save analysis results to cache.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results to cache
        signal_id : str
            Signal ID
        date : str
            Date in YYYY-MM-DD format
        """
        cache_path = self._get_cache_path(signal_id, date)
        try:
            # Convert numpy types and other non-JSON-serializable types
            cacheable_results = self._make_json_serializable(results)
            with open(cache_path, 'w') as f:
                json.dump(cacheable_results, f, indent=2, default=str)
        except Exception as e:
            # Don't fail if caching fails, just warn
            print(f"Warning: Could not save cache to {cache_path}: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Parameters:
        -----------
        obj : Any
            Object to convert
            
        Returns:
        --------
        Any
            JSON-serializable version of object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            # For DataFrames in sample_data, convert to dict
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _extract_unique_feature_patterns(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Extract unique feature patterns by removing phase numbers and direction indicators.
        
        Parameters:
        -----------
        columns : List[str]
            List of column names
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary mapping feature patterns to their occurrences
        """
        import re
        
        # Pattern to match and remove phase indicators (Phase1, Phase2, etc.)
        phase_pattern = r'Phase\d+[A-Z]*'
        
        # Pattern to match and remove direction indicators (T, R, L, TR, etc.)
        direction_pattern = r'[TRL]+$'
        
        # Pattern to match and remove channel numbers (channelNos\d+)
        channel_pattern = r'channelNos\d+'
        
        patterns = {}
        
        for col in columns:
            # Skip basic columns
            if col in ['signalID', 'cycleNo', 'cycleBegin', 'cycleEnd', 'cycleLength', 'date', 'k']:
                continue
                
            # Create a cleaned version of the column name
            cleaned = col
            
            # Remove phase indicators
            cleaned = re.sub(phase_pattern, '', cleaned)
            
            # Remove direction indicators from the end
            cleaned = re.sub(direction_pattern, '', cleaned)
            
            # Remove channel numbers
            cleaned = re.sub(channel_pattern, '', cleaned)
            
            # Clean up any double spaces or trailing underscores
            cleaned = cleaned.replace('__', '_').strip('_')
            
            # If cleaned is empty or very short, use the original
            if len(cleaned) < 3:
                cleaned = col
            
            if cleaned not in patterns:
                patterns[cleaned] = []
            patterns[cleaned].append(col)
        
        return patterns

    def _print_feature_analysis(self, analysis: Dict[str, Any]):
        """Print detailed analysis of a single feature."""
        # Clean up feature name display for empty feature_type
        feature_name = analysis['feature_name']
        if feature_name.endswith('/'):
            feature_name = feature_name.rstrip('/') + '/profile'
        print(f"\n📊 {feature_name}")
        print(f"   Type: {analysis['data_type']}")
        
        if analysis['data_type'] == 'DataFrame':
            structure = analysis['structure']
            print(f"   Shape: {structure['shape']}")
            print(f"   Total columns: {len(structure['columns'])}")
            print(f"   Memory usage: {structure['memory_usage']:,} bytes")
            
            # Parse column structure into components
            column_components = self._parse_column_structure(structure['columns'])
            
            # Show unique values for each component type
            if column_components['phase_types']:
                print(f"   🔵 Phase Types ({len(column_components['phase_types'])}):")
                print(f"      {', '.join(column_components['phase_types'])}")
            
            if column_components['feature_names']:
                print(f"   🟢 Feature Names ({len(column_components['feature_names'])}):")
                # Show feature names, wrapping if too many
                feature_list = column_components['feature_names']
                if len(feature_list) <= 10:
                    print(f"      {', '.join(feature_list)}")
                else:
                    # Show first 10, then count of remaining
                    print(f"      {', '.join(feature_list[:10])}")
                    print(f"      ... and {len(feature_list) - 10} more: {', '.join(feature_list[10:])}")
            
            if column_components['phase_numbers']:
                print(f"   🟡 Phase Numbers ({len(column_components['phase_numbers'])}):")
                print(f"      {', '.join(column_components['phase_numbers'])}")
            
            if column_components['lane_types']:
                print(f"   🔴 Lane Types ({len(column_components['lane_types'])}):")
                print(f"      {', '.join(column_components['lane_types'])}")
            
            # Always show all column names
            print(f"   📝 All columns ({len(structure['columns'])}):")
            # Group columns for better readability
            cols = structure['columns']
            if len(cols) <= 10:
                # Show all columns if less than 10
                for i, col in enumerate(cols, 1):
                    print(f"      {i:3d}. {col}")
            else:
                # Show first 5, then ..., then last 5
                for i, col in enumerate(cols[:5], 1):
                    print(f"      {i:3d}. {col}")
                print(f"      ... ({len(cols) - 10} more columns) ...")
                for i, col in enumerate(cols[-5:], len(cols) - 4):
                    print(f"      {i:3d}. {col}")
            
            if structure['has_nulls']:
                print(f"   ⚠️  Contains null values:")
                for col, null_count in structure['null_columns'].items():
                    print(f"      {col}: {null_count} nulls")
            else:
                print(f"   ✅ No null values")
                
        elif analysis['data_type'] == 'dict':
            structure = analysis['structure']
            print(f"   Dictionary with {structure['total_items']} keys: {structure['keys']}")
            
            # Show DataFrame information if any
            for key, value_type in structure['key_types'].items():
                if value_type == 'DataFrame':
                    shape_key = f'{key}_df_shape'
                    columns_key = f'{key}_df_columns'
                    if shape_key in structure:
                        print(f"   {key} (DataFrame): {structure[shape_key]} columns: {structure[columns_key]}")
        
        elif analysis['data_type'] in ['list', 'tuple']:
            structure = analysis['structure']
            print(f"   Length: {structure['length']}")
            print(f"   Element type: {structure['element_type']}")
    
    def get_all_features_overview(self, signal_id: str = None) -> Dict[str, Any]:
        """
        Get an overview of all available features across all signals or a specific signal.
        
        Parameters:
        -----------
        signal_id : str, optional
            If provided, only analyze this signal. Otherwise analyze all signals.
            
        Returns:
        --------
        Dict[str, Any]
            Overview of all features
        """
        signals_to_analyze = [signal_id] if signal_id else self.get_available_signals()
        
        overview = {
            'total_signals': len(signals_to_analyze),
            'feature_availability': {},
            'signal_summary': {}
        }
        
        print(f"\n{'='*80}")
        print(f"FEATURE AVAILABILITY OVERVIEW")
        print(f"{'='*80}")
        
        for feature_category, feature_list in self.feature_types.items():
            category_availability = {}
            
            for feature_type in feature_list:
                feature_name = f"{feature_category}/{feature_type}"
                available_signals = []
                
                feature_dir = os.path.join(self.base_dir, feature_category, feature_type)
                if os.path.exists(feature_dir):
                    all_signal_dirs = [d for d in os.listdir(feature_dir) 
                                     if os.path.isdir(os.path.join(feature_dir, d))]
                    
                    for sig in signals_to_analyze:
                        if sig in all_signal_dirs:
                            signal_feature_dir = os.path.join(feature_dir, sig)
                            parquet_files = glob.glob(os.path.join(signal_feature_dir, "*.parquet"))
                            if parquet_files:
                                available_signals.append(sig)
                
                category_availability[feature_type] = {
                    'available_signals': len(available_signals),
                    'total_signals': len(signals_to_analyze),
                    'coverage': f"{len(available_signals)/len(signals_to_analyze)*100:.1f}%" if signals_to_analyze else "0%"
                }
                
                print(f"📈 {feature_name}: {len(available_signals)}/{len(signals_to_analyze)} signals ({category_availability[feature_type]['coverage']})")
            
            overview['feature_availability'][feature_category] = category_availability
        
        return overview
    
    def export_analysis_to_csv(self, analysis_results: Dict[str, Any], output_file: str):
        """
        Export analysis results to CSV format.
        
        Parameters:
        -----------
        analysis_results : Dict[str, Any]
            Results from analyze_signal_date method
        output_file : str
            Path to output CSV file
        """
        rows = []
        
        for category, features in analysis_results['available_features'].items():
            for feature_type, analysis in features.items():
                row = {
                    'signal_id': analysis_results['signal_id'],
                    'date': analysis_results['date'],
                    'feature_category': category,
                    'feature_type': feature_type,
                    'data_type': analysis['data_type'],
                    'success': True
                }
                
                if analysis['data_type'] == 'DataFrame':
                    structure = analysis['structure']
                    row.update({
                        'shape_rows': structure['shape'][0],
                        'shape_cols': structure['shape'][1],
                        'columns': '|'.join(structure['columns']),
                        'has_nulls': structure['has_nulls'],
                        'memory_usage': structure['memory_usage']
                    })
                elif analysis['data_type'] == 'dict':
                    structure = analysis['structure']
                    row.update({
                        'dict_keys': '|'.join(structure['keys']),
                        'total_items': structure['total_items']
                    })
                
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            print(f"Analysis results exported to: {output_file}")
    
    def sync_cache_with_processing_log(self, processing_log_path: str, 
                                      output_dir: str = None,
                                      invalidate_only: bool = True,
                                      verbose: bool = True,
                                      limit: int = None,
                                      date_filter: str = None) -> Dict[str, Any]:
        """
        Sync analysis cache with processing log from process_local_atspm.py.
        
        This method reads the processing log database to find which signal/date
        combinations were successfully processed, and updates the cache accordingly.
        
        Parameters:
        -----------
        processing_log_path : str
            Path to processing_log.db SQLite database
        output_dir : str, optional
            Output directory from processing (to check file timestamps).
            If None, infers from base_dir
        invalidate_only : bool
            If True, only invalidate cache entries (delete them) for processed files.
            If False, also analyze and cache results for newly processed files.
        verbose : bool
            Whether to print progress information
        limit : int, optional
            Limit the number of combinations to process (useful for testing)
        date_filter : str, optional
            Only process dates matching this filter (e.g., '2025-09' for September 2025)
            
        Returns:
        --------
        Dict[str, Any]
            Summary of sync operation
        """
        if not os.path.exists(processing_log_path):
            raise FileNotFoundError(f"Processing log not found: {processing_log_path}")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SYNCING CACHE WITH PROCESSING LOG")
            print(f"{'='*80}")
            print(f"Processing log: {processing_log_path}")
            print(f"Mode: {'Invalidate only' if invalidate_only else 'Invalidate and analyze'}")
        
        # Connect to processing log database
        conn = sqlite3.connect(processing_log_path)
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = """
            SELECT DISTINCT signal_id, date_str, MAX(timestamp) as last_processed
            FROM processing_log
            WHERE status = 'success'
        """
        params = []
        
        if date_filter:
            query += " AND date_str LIKE ?"
            params.append(f"{date_filter}%")
        
        query += " GROUP BY signal_id, date_str ORDER BY signal_id, date_str"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        processed_combinations = cursor.fetchall()
        conn.close()
        
        total_found = len(processed_combinations)
        if verbose:
            print(f"Found {total_found} successfully processed signal/date combinations")
            if limit and total_found == limit:
                print(f"  (Limited to {limit} entries)")
            if date_filter:
                print(f"  (Filtered by date: {date_filter})")
        
        invalidated = []
        analyzed = []
        errors = []
        skipped = []
        
        # Show progress for large batches
        show_progress = verbose and len(processed_combinations) > 100
        if show_progress:
            print(f"\nProcessing {len(processed_combinations)} combinations...")
            print("(Showing progress every 100 items)")
        
        for idx, (signal_id, date_str, last_processed) in enumerate(processed_combinations, 1):
            if show_progress and idx % 100 == 0:
                print(f"  Progress: {idx}/{len(processed_combinations)} ({idx*100//len(processed_combinations)}%)")
            cache_path = self._get_cache_path(signal_id, date_str)
            
            # Check if cache exists
            cache_exists = os.path.exists(cache_path)
            
            # Check if output files exist and are newer than cache
            should_invalidate = False
            if output_dir:
                # Check if any output files exist for this signal/date
                output_files_exist = False
                for feature_category, feature_list in self.feature_types.items():
                    for feature_type in feature_list:
                        if feature_type == '':
                            file_path = os.path.join(self.base_dir, feature_category, signal_id, f"{date_str}.parquet")
                        else:
                            file_path = os.path.join(self.base_dir, feature_category, feature_type, signal_id, f"{date_str}.parquet")
                        
                        if os.path.exists(file_path):
                            output_files_exist = True
                            # If cache exists, check if output file is newer
                            if cache_exists:
                                output_mtime = os.path.getmtime(file_path)
                                cache_mtime = os.path.getmtime(cache_path)
                                if output_mtime > cache_mtime:
                                    should_invalidate = True
                                    break
                            else:
                                should_invalidate = True
                                break
                    if should_invalidate:
                        break
            else:
                # If no output_dir provided, check if any output files exist
                # Only invalidate if output files exist
                output_files_exist = False
                for feature_category, feature_list in self.feature_types.items():
                    for feature_type in feature_list:
                        if feature_type == '':
                            file_path = os.path.join(self.base_dir, feature_category, signal_id, f"{date_str}.parquet")
                        else:
                            file_path = os.path.join(self.base_dir, feature_category, feature_type, signal_id, f"{date_str}.parquet")
                        
                        if os.path.exists(file_path):
                            output_files_exist = True
                            if cache_exists:
                                output_mtime = os.path.getmtime(file_path)
                                cache_mtime = os.path.getmtime(cache_path)
                                if output_mtime > cache_mtime:
                                    should_invalidate = True
                                    break
                            else:
                                should_invalidate = True
                                break
                    if should_invalidate:
                        break
                
                if not output_files_exist:
                    # No output files exist, skip this combination
                    skipped.append((signal_id, date_str))
                    continue
            
            if should_invalidate or (not invalidate_only and not cache_exists):
                # Invalidate cache
                if cache_exists:
                    try:
                        os.remove(cache_path)
                        invalidated.append((signal_id, date_str))
                        if verbose:
                            print(f"  ✓ Invalidated cache: {signal_id} / {date_str}")
                    except Exception as e:
                        errors.append((signal_id, date_str, f"Failed to invalidate: {e}"))
                        if verbose:
                            print(f"  ✗ Error invalidating {signal_id} / {date_str}: {e}")
                
                # If not invalidate_only, analyze and cache
                if not invalidate_only:
                    try:
                        self.analyze_signal_date(signal_id, date_str, verbose=False, force=True)
                        analyzed.append((signal_id, date_str))
                        if verbose:
                            print(f"  ✓ Analyzed and cached: {signal_id} / {date_str}")
                    except Exception as e:
                        errors.append((signal_id, date_str, f"Failed to analyze: {e}"))
                        if verbose and not show_progress:
                            print(f"  ✗ Error analyzing {signal_id} / {date_str}: {e}")
        
        summary = {
            'total_processed': len(processed_combinations),
            'invalidated': len(invalidated),
            'analyzed': len(analyzed),
            'skipped': len(skipped),
            'errors': len(errors),
            'invalidated_list': invalidated,
            'analyzed_list': analyzed,
            'skipped_list': skipped,
            'errors_list': errors
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SYNC SUMMARY")
            print(f"{'='*80}")
            print(f"Total processed combinations: {summary['total_processed']}")
            print(f"Cache entries invalidated: {summary['invalidated']}")
            if summary['skipped'] > 0:
                print(f"Skipped (no output files): {summary['skipped']}")
            if not invalidate_only:
                print(f"Newly analyzed and cached: {summary['analyzed']}")
            if summary['errors'] > 0:
                print(f"Errors: {summary['errors']}")
            print(f"{'='*80}\n")
        
        return summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze ATSPM processed feature files')
    parser.add_argument('--signal', type=str, help='Signal ID to analyze (e.g., ORC-0006)')
    parser.add_argument('--date', type=str, help='Date to analyze (YYYY-MM-DD format)')
    parser.add_argument('--all-features', action='store_true', help='Show overview of all features')
    parser.add_argument('--export-csv', type=str, help='Export results to CSV file')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-analysis even if cached results exist')
    parser.add_argument('--base-dir', type=str, 
                       default=r"/data2/ATSPM/processed_data/fdot_d5/feature_extraction",
                       help='Base directory containing processed features')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to store analysis cache (default: .analysis_cache in base_dir)')
    parser.add_argument('--sync-cache', type=str, metavar='LOG_DB',
                       help='Sync cache with processing log database (path to processing_log.db)')
    parser.add_argument('--sync-analyze', action='store_true',
                       help='When syncing cache, also analyze and cache newly processed files (default: only invalidate)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory from processing (for timestamp comparison when syncing)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of entries to process when syncing (for testing)')
    parser.add_argument('--date-filter', type=str, default=None,
                       help='Filter by date prefix when syncing (e.g., 2025-09 for September 2025)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ATSPMFeatureAnalyzer(args.base_dir, cache_dir=args.cache_dir)
    
    if args.sync_cache:
        # Sync cache with processing log
        try:
            output_dir = args.output_dir
            if output_dir is None:
                # Try to infer from base_dir
                # base_dir is typically: /data2/ATSPM/processed_data/fdot_d5/feature_extraction
                # output_dir would be: /data2/ATSPM/processed_data
                if 'processed_data' in args.base_dir:
                    output_dir = args.base_dir.split('/feature_extraction')[0]
            
            analyzer.sync_cache_with_processing_log(
                args.sync_cache,
                output_dir=output_dir,
                invalidate_only=not args.sync_analyze,
                verbose=True,
                limit=args.limit,
                date_filter=args.date_filter
            )
        except Exception as e:
            print(f"Error syncing cache: {e}")
            return 1
    
    elif args.all_features:
        # Show overview of all features
        analyzer.get_all_features_overview(args.signal)
        
    elif args.signal and args.date:
        # Analyze specific signal and date
        try:
            results = analyzer.analyze_signal_date(args.signal, args.date, force=args.force)
            
            if args.export_csv:
                analyzer.export_analysis_to_csv(results, args.export_csv)
                
        except Exception as e:
            print(f"Error analyzing {args.signal} for {args.date}: {e}")
            
    else:
        # Interactive mode - show available options
        print("ATSPM Feature Analyzer")
        print("=" * 50)
        
        signals = analyzer.get_available_signals()
        print(f"Available signals: {len(signals)}")
        print(f"Sample signals: {signals[:10]}")
        
        if len(signals) > 10:
            print(f"... and {len(signals) - 10} more")
        
        print("\nUsage examples:")
        print("  python analyze_processed_features.py --signal ORC-0006 --date 2025-09-01")
        print("  python analyze_processed_features.py --all-features")
        print("  python analyze_processed_features.py --signal ORC-0006 --date 2025-09-01 --export-csv results.csv")
        print("  python analyze_processed_features.py --signal ORC-0006 --date 2025-09-01 --force")
        print("  python analyze_processed_features.py --sync-cache /data2/ATSPM/processed_data/processing_log.db")
        print("  python analyze_processed_features.py --sync-cache /data2/ATSPM/processed_data/processing_log.db --sync-analyze")
        print("  python analyze_processed_features.py --sync-cache /data2/ATSPM/processed_data/processing_log.db --date-filter 2025-09")
        print("  python analyze_processed_features.py --sync-cache /data2/ATSPM/processed_data/processing_log.db --limit 100")


if __name__ == "__main__":
    main()
