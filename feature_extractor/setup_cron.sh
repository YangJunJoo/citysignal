#!/bin/bash
# Setup script for daily PET extraction cron job
# This script helps set up a cron job to run PET extraction every 24 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/run_pet_extraction_daily.py"
CONFIG_FILE="$SCRIPT_DIR/pet_extraction_config.json"

# Find Python executable (try python3 first, then python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found. Please install Python 3."
    exit 1
fi

# Get full paths
PYTHON_SCRIPT_ABS="$(readlink -f "$PYTHON_SCRIPT")"
CONFIG_FILE_ABS="$(readlink -f "$CONFIG_FILE")"

# Create log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Cron job entry (runs daily at 2:00 AM)
# Format: minute hour day month weekday command
CRON_SCHEDULE="0 2 * * *"
CRON_COMMAND="cd \"$PROJECT_ROOT\" && $PYTHON_CMD \"$PYTHON_SCRIPT_ABS\" --config \"$CONFIG_FILE_ABS\" >> \"$LOG_DIR/cron.log\" 2>&1"

echo "=========================================="
echo "PET Extraction Daily Cron Setup"
echo "=========================================="
echo ""
echo "Script: $PYTHON_SCRIPT_ABS"
echo "Config: $CONFIG_FILE_ABS"
echo "Log dir: $LOG_DIR"
echo "Python: $PYTHON_CMD"
echo ""
echo "Cron schedule: Daily at 2:00 AM"
echo "Cron command:"
echo "  $CRON_COMMAND"
echo ""
echo "To add this cron job, run:"
echo "  crontab -e"
echo ""
echo "Then add this line:"
echo "  $CRON_SCHEDULE $CRON_COMMAND"
echo ""
echo "Or run this script with --install flag to install automatically:"
echo "  $0 --install"
echo ""

if [ "$1" == "--install" ]; then
    # Check if cron entry already exists
    if crontab -l 2>/dev/null | grep -q "$PYTHON_SCRIPT_ABS"; then
        echo "Cron job already exists. Removing old entry..."
        crontab -l 2>/dev/null | grep -v "$PYTHON_SCRIPT_ABS" | crontab -
    fi
    
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -
    echo "✓ Cron job installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l
elif [ "$1" == "--remove" ]; then
    # Remove cron job
    if crontab -l 2>/dev/null | grep -q "$PYTHON_SCRIPT_ABS"; then
        crontab -l 2>/dev/null | grep -v "$PYTHON_SCRIPT_ABS" | crontab -
        echo "✓ Cron job removed successfully!"
    else
        echo "No cron job found to remove."
    fi
elif [ "$1" == "--test" ]; then
    # Test run
    echo "Running test execution..."
    echo ""
    cd "$PROJECT_ROOT"
    $PYTHON_CMD "$PYTHON_SCRIPT_ABS" --config "$CONFIG_FILE_ABS"
else
    echo "Options:"
    echo "  --install  : Install the cron job automatically"
    echo "  --remove   : Remove the cron job"
    echo "  --test     : Run a test execution"
    echo ""
    echo "To manually edit crontab:"
    echo "  crontab -e"
fi

