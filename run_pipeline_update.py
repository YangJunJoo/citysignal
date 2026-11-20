#!/usr/bin/env python3
import sys
import os
import subprocess
import time

print("="*60)
print("Starting ATSPM Full Data Pipeline Update")
print("="*60)

# 1. Download Data
print("\n[Step 1/3] Downloading latest data...")
sys.path.insert(0, '/home/ya195253/YJ/python_scripts/ATSPM/data_downloader')
import data_downloader
try:
    data_downloader.main()
except Exception as e:
    print(f"Downloader finished with message: {e}")

# 2. Split Data
print("\n[Step 2/3] Splitting data by signal...")
import split_by_signal
from pathlib import Path
input_dir = Path("/data2/ATSPM/raw_data")
output_dir = Path("/data2/ATSPM/raw_data/per_signal")
split_by_signal.run_processing(input_dir, output_dir, overwrite=False)

# 3. Extract Features
print("\n[Step 3/3] Running Feature Extraction on new data...")
# Run with --auto and --max-age-hours 720 (30 days) to catch all the files we just split
cmd = [
    "python", 
    "/home/ya195253/YJ/python_scripts/ATSPM/feature_extractor/process_local_atspm.py",
    "--event-dir", "/data2/ATSPM/raw_data",
    "--config-dir", "/data2/ATSPM/config_data/signal_config_2025/lanes",
    "--output-dir", "/data2/ATSPM/processed_data",
    "--auto", 
    "--max-age-hours", "720"
]
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)

print("\n" + "="*60)
print("ATSPM Full Data Pipeline Update Completed")
print("="*60)
