#!/bin/bash
# Automated daily processing script for ATSPM feature extraction
# This script processes new or updated parquet files automatically
# Designed to be run via cron job every 24 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration - adjust these paths as needed
EVENT_DIR="/data2/ATSPM/raw_data"
CONFIG_DIR="/data2/ATSPM/config_data/signal_config_2025/lanes"
OUTPUT_DIR="/data2/ATSPM/processed_data"
MAX_AGE_HOURS=24  # Process files updated in last 24 hours

# Find Python executable
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Please install Python 3."
    exit 1
fi

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/automated_processing_$(date +%Y%m%d).log"

# Change to project root
cd "$PROJECT_ROOT"

# Run automated processing
echo "=========================================="
echo "Automated ATSPM Processing"
echo "Date: $(date)"
echo "=========================================="
echo ""

$PYTHON_CMD "$SCRIPT_DIR/process_local_atspm.py" \
    --event-dir "$EVENT_DIR" \
    --config-dir "$CONFIG_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --auto \
    --max-age-hours "$MAX_AGE_HOURS" \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Processing completed successfully"
else
    echo "✗ Processing failed with exit code $EXIT_CODE"
fi

echo "Log file: $LOG_FILE"
echo "=========================================="

exit $EXIT_CODE

