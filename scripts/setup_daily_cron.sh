#!/bin/bash
###############################################################################
# Setup Daily ATSPM Pipeline Cron Job
# Installs a single cron task to trigger run_daily_pipeline.sh daily at 10:00 ET
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_daily_pipeline.sh"
CRON_LOG_DIR="/data2/ATSPM/operations/logs"

# Ensure the pipeline script is executable
chmod +x "${PIPELINE_SCRIPT}"

# The source is normally published after 03:00 ET, so 10:00 avoids a
# predictable one-day lag. The wrapper itself owns the shared pipeline lock.
CRON_SCHEDULE="0 10 * * *"
CRON_COMMAND="TZ=America/New_York /usr/bin/bash \"${PIPELINE_SCRIPT}\" >> \"${CRON_LOG_DIR}/cron_launcher.log\" 2>&1"

echo "=========================================="
echo "ATSPM Daily Pipeline Cron Setup"
echo "=========================================="
echo ""
echo "Script: $PIPELINE_SCRIPT"
echo "Cron schedule: Daily at 10:00 AM America/New_York"

if [ "${1:-}" == "--install" ]; then
    mkdir -p "${CRON_LOG_DIR}"
    # Check if cron entry already exists
    if crontab -l 2>/dev/null | grep -q "$(basename "$PIPELINE_SCRIPT")"; then
        echo "Cron job already exists. Removing old entry..."
        crontab -l 2>/dev/null | grep -v "$(basename "$PIPELINE_SCRIPT")" | crontab -
    fi
    
    # Add new cron job
    (crontab -l 2>/dev/null; echo "${CRON_SCHEDULE} ${CRON_COMMAND}") | crontab -
    echo "✓ Cron job installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep ATSPM || crontab -l | grep pipeline
elif [ "${1:-}" == "--remove" ]; then
    if crontab -l 2>/dev/null | grep -q "$(basename "$PIPELINE_SCRIPT")"; then
        crontab -l 2>/dev/null | grep -v "$(basename "$PIPELINE_SCRIPT")" | crontab -
        echo "✓ Cron job removed successfully!"
    else
        echo "No cron job found to remove."
    fi
else
    echo "Options:"
    echo "  --install  : Install the cron job automatically"
    echo "  --remove   : Remove the cron job"
    echo ""
    echo "To view current crontab: crontab -l"
fi
