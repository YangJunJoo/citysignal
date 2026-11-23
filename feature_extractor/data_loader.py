"""
data_loader.py — Single-load cached data provider for ATSPM.

Replaces the v1 pattern where every feature extraction method calls _load_data()
independently, re-reading the same parquet/csv files 14+ times per signal-day.

This module loads all data once and provides pre-filtered, pre-merged views
to all feature extractors.
"""

import os
import logging
import pandas as pd
import numpy as np
from . import storage


def _float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 columns to Int64 (nullable integer) in a DataFrame."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            try:
                df[col] = df[col].astype('Int64')
            except (ValueError, TypeError):
                pass
    return df


def _abbreviate_directions(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Transform direction strings ('Left', 'Left Through') to abbreviations ('L', 'LT')."""
    df = df.copy()
    df[column_name] = df[column_name].apply(
        lambda value: ''.join(d[0].upper() for d in value.split()) if pd.notna(value) else value
    )
    return df


class SignalDataCache:
    """
    Load all data for a signal-day ONCE and provide pre-computed views.
    
    This replaces 14× _load_data() calls with a single initialization.
    Pre-computes:
      - Config filtered by detector type (back, front, count)
      - Events filtered by event code
      - Pre-merged event+config DataFrames
    """
    
    def __init__(self, signal_id: str, year: int, month: int, day: int,
                 cwd: str, relative_interim: str, relative_production: str,
                 config_dir: str = None):
        """
        Load and cache all data for a signal-day.
        
        Parameters
        ----------
        signal_id : str
            Signal identifier
        year, month, day : int
            Date components
        cwd : str
            Working directory root (uses `cwd`)
        relative_interim : str
            Relative path to interim data (maps to `relative_interim_database_dirpath`)
        relative_production : str
            Relative path to production data (maps to `relative_production_database_dirpath`)
        config_dir : str, optional
            Explicit directory containing signal config CSV files.
            If provided, overrides the default interim path lookup.
        """
        self.signal_id = signal_id
        self.year = year
        self.month = month
        self.day = day
        self.date_str = f"{year}-{month:02d}-{day:02d}"
        self.cwd = cwd
        self.relative_interim = relative_interim
        self.relative_production = relative_production
        self.config_dir = config_dir
        
        # District is hardcoded in v1 — keep consistent
        self.district = "fdot_d5"
        
        # Load raw data
        self.df_event = self._load_event_data()
        self.df_config_raw = self._load_config_data()
        self.df_vehicle_cycle = self._load_vehicle_cycle_profile()
        self.df_pedestrian_cycle = self._load_pedestrian_cycle_profile()
        
        # Pre-compute filtered views
        self._config_cache = {}       # {detector_type: filtered_config_df}
        self._event_cache = {}        # {event_seq_tuple: filtered_event_df}
        self._merged_cache = {}       # {(event_seq_tuple, detector_type): merged_df}
        
        # Pre-extract phase numbers from cycle profile columns
        self.vehicle_phase_nos = self._extract_phase_nos(self.df_vehicle_cycle)
        
        # Vectorization caches
        self._cycle_index_cache = {}      # {(event_seq, det_type): df_with_cycle_idx}
        self._signal_intervals_cache = None  # Built once on first call
        
        # Validate data integrity
        self._validate()
        
        logging.info(f"SignalDataCache initialized for {signal_id} on {self.date_str}: "
                     f"{len(self.df_event)} events, {len(self.df_vehicle_cycle)} cycles")
    
    def _validate(self):
        """
        Validate loaded data for integrity. Logs warnings for anomalies.
        Non-fatal: empty data may be valid for some signals/dates.
        """
        # Check for empty data
        if len(self.df_event) == 0:
            logging.warning(f"[{self.signal_id}] No event data for {self.date_str}")
            return
        
        # Required columns
        required_event_cols = {'timeStamp', 'eventCode', 'eventParam'}
        missing = required_event_cols - set(self.df_event.columns)
        if missing:
            logging.error(f"[{self.signal_id}] Missing event columns: {missing}")
        
        # Check cycle profile
        if len(self.df_vehicle_cycle) == 0:
            logging.warning(f"[{self.signal_id}] Empty vehicle cycle profile for {self.date_str}")
        elif 'cycleBegin' in self.df_vehicle_cycle.columns:
            begins = self.df_vehicle_cycle['cycleBegin']
            if not begins.is_monotonic_increasing:
                logging.warning(f"[{self.signal_id}] cycleBegin is not monotonically increasing")
        
        # Sanity check event count
        if len(self.df_event) < 10:
            logging.warning(f"[{self.signal_id}] Very few events ({len(self.df_event)}) for {self.date_str}")
        elif len(self.df_event) > 1_000_000:
            logging.warning(f"[{self.signal_id}] Unusually many events ({len(self.df_event)}) for {self.date_str}")

    def _build_path(self, base_type: str, *parts) -> str:
        """Build a file path from base type ('interim' or 'production') and parts."""
        if base_type == 'interim':
            base = os.path.join(self.cwd, self.relative_interim)
        else:
            base = os.path.join(self.cwd, self.relative_production)
        return os.path.join(base, *parts)
    
    def _load_event_data(self) -> pd.DataFrame:
        """Load the raw event data for this signal-day from Parquet."""
        # Internal staged events carry the same reversible storage contract.
        staged_path = self._build_path('interim', self.district, 'event_data',
                                       self.signal_id, f"{self.date_str}.parquet")
        if os.path.exists(staged_path):
            df = storage.read_frame(staged_path)
            df["eventCode"] = df["eventCode"].astype(int)
            return df
        
        # Fallback: parquet (process_local per_signal format)
        parquet_path = self._build_path('interim', self.district, 'per_signal',
                                         self.signal_id, f"{self.date_str}.parquet")
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if 'eventCode' in df.columns:
                df["eventCode"] = df["eventCode"].astype(int)
            return df
        
        logging.warning(f"Event data not found: {staged_path} (also tried per-signal parquet)")
        return pd.DataFrame()
    
    def _load_config_data(self) -> pd.DataFrame:
        """Load signal configuration data."""
        # If explicit config_dir was provided, use it
        if self.config_dir:
            path = os.path.join(self.config_dir, f"{self.signal_id}.csv")
        else:
            # Default: {interim}/{district}/signal_config/{signal_id}.csv
            path = self._build_path('interim', self.district, 'signal_config', 
                                    f"{self.signal_id}.csv")
        if not os.path.exists(path):
            logging.warning(f"Config data not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def _load_vehicle_cycle_profile(self) -> pd.DataFrame:
        """Load vehicle cycle profile data."""
        # Path: {production}/{district}/feature_extraction/signal_profile/cycle/vehicle_signal/{signal_id}/{date}.parquet
        path = self._build_path('production', self.district, 
                                'feature_extraction', 'signal_profile', 'cycle', 'vehicle_signal',
                                self.signal_id, f"{self.date_str}.parquet")
        if not os.path.exists(path):
            logging.warning(f"Vehicle cycle profile not found: {path}")
            return pd.DataFrame()
        return storage.read_frame(path)
    
    def _load_pedestrian_cycle_profile(self) -> pd.DataFrame:
        """Load pedestrian cycle profile data."""
        # Path: {production}/{district}/feature_extraction/signal_profile/cycle/pedestrian_signal/{signal_id}/{date}.parquet
        path = self._build_path('production', self.district,
                                'feature_extraction', 'signal_profile', 'cycle', 'pedestrian_signal',
                                self.signal_id, f"{self.date_str}.parquet")
        if not os.path.exists(path):
            logging.info(f"Pedestrian cycle profile not found (optional)")
            return pd.DataFrame()
        return storage.read_frame(path)
    
    @staticmethod
    def _extract_phase_nos(df_cycle: pd.DataFrame) -> list:
        """Extract unique phase numbers from cycle profile column names."""
        if len(df_cycle) == 0:
            return []
        phase_nos = set()
        for col in df_cycle.columns:
            if "Phase" in col:
                # Extract the digit(s) right after "Phase"
                idx = col.index("Phase") + len("Phase")
                num_str = ""
                while idx < len(col) and col[idx].isdigit():
                    num_str += col[idx]
                    idx += 1
                if num_str:
                    phase_nos.add(int(num_str))
        return sorted(phase_nos)
    
    def get_config(self, detector_type: str) -> pd.DataFrame:
        """
        Get config filtered by detector type and cached.
        
        Parameters
        ----------
        detector_type : str
            One of 'back', 'front', 'count'
        
        Returns
        -------
        pd.DataFrame
            Filtered config with abbreviated directions (for back/front),
            phaseNo not null, and Int64 types.
        """
        if detector_type in self._config_cache:
            return self._config_cache[detector_type]
        
        df = self.df_config_raw.copy()
        df = df[pd.notna(df["phaseNo"])].reset_index(drop=True)
        df = _float_to_int(df)
        
        # Apply detector type filter (matching v1 _filter_config_by_detector exactly)
        if detector_type == "back":
            # v1: Filter for back detector (stopBarDistance != 0)
            df = df[df["stopBarDistance"] != 0].reset_index(drop=True)
            # If multiple back detectors exist per phase, keep the farthest from stop bar
            for phase_no in df["phaseNo"].unique():
                df_phase = df[df["phaseNo"] == phase_no]
                back_distances = df_phase["stopBarDistance"].unique().tolist()
                if len(back_distances) > 1:
                    # Drop detectors at the minimum distance (keep farthest)
                    indices = df_phase[df_phase["stopBarDistance"] == min(back_distances)].index
                    df = df.drop(index=indices)
            df = df.reset_index(drop=True)
            df = _abbreviate_directions(df, "laneType")
        elif detector_type == "front":
            # v1: Filter for front detector (stopBarDistance == 0)
            df = df[df["stopBarDistance"] == 0].reset_index(drop=True)
            df = _abbreviate_directions(df, "laneType")
        elif detector_type == "count":
            if "countBar" in df.columns:
                df = df[df["countBar"] == 1].reset_index(drop=True)
            else:
                df = df.head(0)  # Empty if no countBar column
        
        self._config_cache[detector_type] = df
        return df
    
    def get_events(self, event_sequence: tuple) -> pd.DataFrame:
        """
        Get events filtered by event code sequence, cached.
        
        Parameters
        ----------
        event_sequence : tuple
            Tuple of event codes, e.g. (81,) or (82, 81)
        
        Returns
        -------
        pd.DataFrame
            Filtered and sorted event DataFrame
        """
        if event_sequence in self._event_cache:
            return self._event_cache[event_sequence]
        
        df = self.df_event[self.df_event["eventCode"].isin(event_sequence)].copy()
        df = df.sort_values(by=["timeStamp", "eventCode"]).reset_index(drop=True)
        
        self._event_cache[event_sequence] = df
        return df
    
    def get_merged_events(self, event_sequence: tuple, detector_type: str) -> pd.DataFrame:
        """
        Get events merged with config, cached.
        
        This replaces the repeated pattern in v1:
            df_event = filter_by_event_sequence(...)
            df_config = _filter_config_by_detector(...)
            df_event = pd.merge(df_event, df_config, on eventParam=channelNo)
        
        Parameters
        ----------
        event_sequence : tuple
            Tuple of event codes
        detector_type : str
            Detector type filter for config
        
        Returns
        -------
        pd.DataFrame
            Merged events+config with consistent types
        """
        cache_key = (event_sequence, detector_type)
        if cache_key in self._merged_cache:
            return self._merged_cache[cache_key]
        
        df_events = self.get_events(event_sequence)
        df_config = self.get_config(detector_type)
        
        if len(df_events) == 0 or len(df_config) == 0:
            self._merged_cache[cache_key] = pd.DataFrame()
            return self._merged_cache[cache_key]
        
        df_merged = pd.merge(df_events, df_config,
                             how="inner",
                             left_on=["eventParam"], right_on=["channelNo"])
        df_merged = _float_to_int(df_merged)
        
        self._merged_cache[cache_key] = df_merged
        return df_merged
    
    def get_valid_phase_nos(self, df_events_merged: pd.DataFrame) -> list:
        """
        Get phase numbers that exist in both the event data and cycle profile columns.
        
        Replicates v1 logic:
            phase_nos = sorted(df_event_id["phaseNo"].unique())
            phase_nos = [p for p in phase_nos if any(str(p) in col for col in df_cycle.columns)]
        """
        if len(df_events_merged) == 0 or len(self.df_vehicle_cycle) == 0:
            return []
        
        event_phases = sorted(df_events_merged["phaseNo"].unique())
        return [p for p in event_phases if p in self.vehicle_phase_nos]
    
    def get_lane_types_for_phase(self, detector_type: str, phase_no: int) -> list:
        """Get unique lane types for a phase from the config."""
        df_config = self.get_config(detector_type)
        return df_config[df_config["phaseNo"] == phase_no]["laneType"].unique().tolist()
    
    def get_channels_for_phase_lane(self, detector_type: str, phase_no: int, lane_type: str) -> list:
        """Get channel numbers for a specific phase and lane type."""
        df_config = self.get_config(detector_type)
        mask = (df_config["phaseNo"] == phase_no) & (df_config["laneType"] == lane_type)
        return df_config[mask]["channelNo"].unique().tolist()
    
    def get_signal_timestamps(self, cycle_idx: int, signal_type: str, phase_no: int):
        """
        Get signal type timestamps from cycle profile for a given cycle index and phase.
        
        Returns
        -------
        list of tuples or None
            List of (start, end) tuples, or None if no valid timestamps
        """
        col_name = f"{signal_type}Phase{phase_no}"
        if col_name not in self.df_vehicle_cycle.columns:
            return None
        
        timestamps = self.df_vehicle_cycle.loc[cycle_idx, col_name]
        
        if not isinstance(timestamps, list):
            timestamps = [timestamps]
        
        if (timestamps == [pd.NaT]) or all(pd.isna(t) for t in timestamps):
            return None
        
        return timestamps
    
    
    def assign_cycle_index(self, event_sequence: tuple, detector_type: str,
                           tolerance: str = 'exact') -> pd.DataFrame:
        """
        Assign each event to a vehicle cycle index using np.searchsorted.
        
        Parameters
        ----------
        event_sequence : tuple
            Event codes to filter by
        detector_type : str
            Detector type for config merge
        tolerance : str
            'exact' = inclusive bounds, duplicating shared-boundary events
            'prev_next' = legacy home-cycle assignment without boundary duplication;
                          this mode does not expand the window
        
        Returns
        -------
        pd.DataFrame
            Merged events with an additional 'cycle_idx' column (int, -1 if unassigned)
        """
        cache_key = (event_sequence, detector_type, tolerance)
        if cache_key in self._cycle_index_cache:
            return self._cycle_index_cache[cache_key]
        
        df_events = self.get_merged_events(event_sequence, detector_type)
        df_cycle = self.df_vehicle_cycle
        
        if len(df_events) == 0 or len(df_cycle) == 0:
            self._cycle_index_cache[cache_key] = pd.DataFrame()
            return self._cycle_index_cache[cache_key]
        
        df_out = df_events.copy()
        
        cycle_begins = df_cycle['cycleBegin'].values  # numpy datetime64 array
        cycle_ends = df_cycle['cycleEnd'].values
        event_ts = df_out['timeStamp'].values
        n_cycles = len(df_cycle)
        
        if tolerance in ('exact', 'prev_next'):
            # Assign to cycle where cycleBegin <= ts <= cycleEnd
            # searchsorted: find the rightmost cycle whose cycleBegin <= ts
            idx = np.searchsorted(cycle_begins, event_ts, side='right') - 1
            # Validate: event must be <= cycleEnd of assigned cycle
            valid = (idx >= 0) & (idx < n_cycles)
            if df_cycle.attrs.get('interval_closure') == 'left':
                valid[valid] &= event_ts[valid] < cycle_ends[idx[valid]]
            else:
                valid[valid] &= event_ts[valid] <= cycle_ends[idx[valid]]
            idx[~valid] = -1
            df_out['cycle_idx'] = idx
            
        if tolerance == 'exact' and df_cycle.attrs.get('interval_closure') != 'left':
            # V1 boundary fix: when cycleEnd[i] == cycleBegin[i+1], v1 counts
            # the event in BOTH cycles. Duplicate boundary events so they
            # appear in both the current and previous cycle.
            alt_idx = idx - 1
            can_dup = (alt_idx >= 0) & (alt_idx < n_cycles) & (idx >= 0)
            can_dup[can_dup] &= (event_ts[can_dup] >= cycle_begins[alt_idx[can_dup]]) & \
                                (event_ts[can_dup] <= cycle_ends[alt_idx[can_dup]])
            
            if np.any(can_dup):
                dup_rows = df_out[can_dup].copy()
                dup_rows['cycle_idx'] = alt_idx[can_dup]
                df_out = pd.concat([df_out, dup_rows], ignore_index=True)
            
        self._cycle_index_cache[cache_key] = df_out
        return df_out
    
    def build_signal_type_intervals(self) -> pd.DataFrame:
        """
        Flatten all (cycle_idx, phase_no, signal_type) windows from the vehicle 
        cycle profile into a single DataFrame for vectorized signal type assignment.
        
        Returns
        -------
        pd.DataFrame with columns:
            cycle_idx, phase_no, signal_type, interval_start, interval_end
        """
        if self._signal_intervals_cache is not None:
            return self._signal_intervals_cache
        
        df_cycle = self.df_vehicle_cycle
        if len(df_cycle) == 0:
            self._signal_intervals_cache = pd.DataFrame(
                columns=['cycle_idx', 'phase_no', 'signal_type', 'interval_start', 'interval_end']
            )
            return self._signal_intervals_cache
        
        rows = []
        signal_types = ['green', 'yellow', 'redClearance', 'red']
        
        for phase_no in self.vehicle_phase_nos:
            for signal_type in signal_types:
                col_name = f"{signal_type}Phase{phase_no}"
                if col_name not in df_cycle.columns:
                    continue
                
                # Process all cycles at once for this column
                col_data = df_cycle[col_name].values
                for cycle_idx, timestamps in enumerate(col_data):
                    if timestamps is None:
                        continue
                    
                    if not isinstance(timestamps, list):
                        timestamps = [timestamps]
                    
                    if not timestamps or all(
                        (isinstance(t, float) and np.isnan(t)) or t is pd.NaT or t is None
                        for t in timestamps
                    ):
                        continue
                    
                    # timestamps can be list of (start, end) tuples or list of values
                    if isinstance(timestamps[0], tuple):
                        for start, end in timestamps:
                            if pd.notna(start) and pd.notna(end):
                                rows.append((cycle_idx, phase_no, signal_type, start, end))
                    elif len(timestamps) == 2 and pd.notna(timestamps[0]) and pd.notna(timestamps[1]):
                        rows.append((cycle_idx, phase_no, signal_type, timestamps[0], timestamps[1]))
        
        if rows:
            result = pd.DataFrame(
                rows, columns=['cycle_idx', 'phase_no', 'signal_type', 'interval_start', 'interval_end']
            )
            result['interval_start'] = pd.to_datetime(result['interval_start'])
            result['interval_end'] = pd.to_datetime(result['interval_end'])
        else:
            result = pd.DataFrame(
                columns=['cycle_idx', 'phase_no', 'signal_type', 'interval_start', 'interval_end']
            )
        
        self._signal_intervals_cache = result
        return result
    
    def get_output_path(self, event_type: str, feature_name: str) -> str:
        """
        Build the output directory path for a feature.
        
        Parameters
        ----------
        event_type : str
            e.g. 'vehicle_traffic', 'vehicle_signal'  
        feature_name : str
            e.g. 'volume', 'headway'
        
        Returns
        -------
        str
            Full directory path for output
        """
        dirpath = os.path.join(
            self.cwd, self.relative_production, self.district,
            'feature_extraction', 'feature', 'cycle', event_type, feature_name, self.signal_id
        )
        return dirpath
    
    def save_feature(self, df: pd.DataFrame, event_type: str, feature_name: str):
        """Validate a temporary Parquet before atomically replacing the output.

        A serialization, verification, or replacement failure leaves any prior
        output intact. Empty DataFrames remain supported by this legacy API;
        checked extraction classifies them before the common executor saves.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Feature output must be a pandas DataFrame")
        dirpath = self.get_output_path(event_type, feature_name)
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, f"{self.date_str}.parquet")
        storage.write_frame(df, filepath)
        logging.info(f"{feature_name} saved: {filepath}")
