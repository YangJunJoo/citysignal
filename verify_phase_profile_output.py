#!/usr/bin/env python3
"""
Script to verify that phase profile output includes red phase columns
and is saved to the correct directory.
"""

from pathlib import Path as _StoragePath
import sys as _storage_sys
_storage_root = _StoragePath(__file__).resolve().parent
if _storage_root.name in {'feature_extractor', 'scripts', 'plot'}:
    _storage_root = _storage_root.parent
if str(_storage_root) not in _storage_sys.path:
    _storage_sys.path.insert(0, str(_storage_root))
from feature_extractor.storage import read_frame, write_frame


import os
import pandas as pd
import sys

def verify_phase_profile_output(event_data_dir, signal_id, year, month, day):
    """
    Verify phase profile output file exists and contains red phase columns.
    
    Parameters:
    -----------
    event_data_dir : str
        Base event data directory
    signal_id : str
        Signal ID to check
    year : int
        Year
    month : int
        Month
    day : int
        Day
    """
    date_str = f"{year}-{month:02d}-{day:02d}"
    district = "fdot_d5"
    
    # Expected output path for vehicle phase profile
    expected_path = os.path.join(
        event_data_dir,
        "processed",
        district,
        "feature_extraction",
        "signal_profile",
        "phase",
        "vehicle_signal",
        signal_id,
        f"{date_str}.parquet"
    )
    
    print(f"\n{'='*60}")
    print(f"VERIFYING PHASE PROFILE OUTPUT")
    print(f"{'='*60}")
    print(f"Signal ID: {signal_id}")
    print(f"Date: {date_str}")
    print(f"Expected path: {expected_path}")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(expected_path):
        print(f"❌ ERROR: Phase profile file not found at expected path!")
        print(f"   Path: {expected_path}")
        return False
    
    print(f"✓ Phase profile file exists")
    
    # Check file size
    file_size = os.path.getsize(expected_path)
    print(f"✓ File size: {file_size:,} bytes")
    
    if file_size == 0:
        print(f"❌ ERROR: Phase profile file is empty!")
        return False
    
    # Load and check the file
    try:
        df = read_frame(expected_path)
        print(f"✓ Successfully loaded phase profile")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check for required columns
        required_columns = [
            "signalID", "phaseNo", "greenBegin", "greenEnd",
            "yellowBegin", "yellowEnd", "redClearanceBegin", "redClearanceEnd"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ ERROR: Missing required columns: {missing_columns}")
            return False
        
        print(f"✓ All required columns present")
        
        # Check for red phase columns
        red_columns = ["redBegin", "redEnd"]
        has_red_columns = all(col in df.columns for col in red_columns)
        
        if has_red_columns:
            print(f"✓ Red phase columns found: {red_columns}")
            
            # Check if any red phases were computed
            has_red_data = df["redBegin"].notna().any() or df["redEnd"].notna().any()
            if has_red_data:
                red_count = df["redBegin"].notna().sum()
                print(f"✓ Red phase data found in {red_count} rows ({red_count/len(df)*100:.1f}%)")
            else:
                print(f"⚠ Warning: Red phase columns exist but all values are NaT")
        else:
            print(f"❌ ERROR: Red phase columns missing!")
            print(f"   Expected: {red_columns}")
            print(f"   Found columns: {list(df.columns)}")
            return False
        
        # Display sample data
        print(f"\n{'='*60}")
        print(f"SAMPLE DATA (first 3 rows)")
        print(f"{'='*60}")
        print(df[["signalID", "phaseNo", "greenBegin", "greenEnd", 
                  "yellowBegin", "yellowEnd", "redClearanceBegin", "redClearanceEnd",
                  "redBegin", "redEnd"]].head(3).to_string())
        
        print(f"\n{'='*60}")
        print(f"✓ VERIFICATION SUCCESSFUL")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load or verify phase profile: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python verify_phase_profile_output.py <event_data_dir> <signal_id> <year> <month> <day>")
        print("\nExample:")
        print("  python verify_phase_profile_output.py /data2/ATSPM/raw_data ORC-0003 2025 9 7")
        sys.exit(1)
    
    event_data_dir = sys.argv[1]
    signal_id = sys.argv[2]
    year = int(sys.argv[3])
    month = int(sys.argv[4])
    day = int(sys.argv[5])
    
    success = verify_phase_profile_output(event_data_dir, signal_id, year, month, day)
    sys.exit(0 if success else 1)

