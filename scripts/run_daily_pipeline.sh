#!/bin/bash
###############################################################################
# ATSPM Daily Automated Pipeline
# 
# Purpose: Runs routinely via cron to process the latest daily data.
# Config:
#   - Bounded recent-date window with explicit per-run manifests
#   - Shared lock prevents cron, dashboard, and manual backfill overlap
#   - Uses the canonical /data2 configuration and output trees
###############################################################################

set -u -o pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
# Conservative defaults; bounded environment overrides support manual backfill.
MAX_WORKERS="${ATSPM_MAX_WORKERS:-24}"
SPLIT_WORKERS="${ATSPM_SPLIT_WORKERS:-4}"
LOOKBACK_DAYS="${ATSPM_LOOKBACK_DAYS:-14}"
MAX_RANGE_DAYS="${ATSPM_MAX_RANGE_DAYS:-60}"

PROJECT_ROOT="/home/ya195253/YJ/python_scripts/ATSPM"
RAW_DATA_DIR="/data2/ATSPM/raw_data"
PER_SIGNAL_DIR="${RAW_DATA_DIR}/per_signal"
PROCESSED_DATA_DIR="/data2/ATSPM/processed_data"
CONFIG_DIR="/data2/ATSPM/config_data/signal_config_2025/lanes"
OPERATIONS_DIR="/data2/ATSPM/operations"
LOG_DIR="${OPERATIONS_DIR}/logs"
RUNS_DIR="${OPERATIONS_DIR}/runs"
LOCK_FILE="/data2/ATSPM/.pipeline.lock"
CONDA_BIN="/home/ya195253/anaconda3/bin/conda"
PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n atspm python3)

DATE_EXT=$(TZ=America/New_York date +%Y%m%dT%H%M%S)
RUN_DIR="${RUNS_DIR}/${DATE_EXT}_$$"
PIPELINE_LOG="${LOG_DIR}/daily_pipeline_${DATE_EXT}.log"
BULK_MANIFEST="${RUN_DIR}/bulk_files.txt"
LATEST_DATE_FILE="${RUN_DIR}/latest_date.txt"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${PIPELINE_LOG}"
}

# Use one lock for cron, dashboard-triggered runs, and manual backfills.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "Another ATSPM pipeline owns ${LOCK_FILE}; exiting without overlap."
    exit 75
fi

TODAY_ET=$(TZ=America/New_York date +%F)
DEFAULT_START=$(TZ=America/New_York date -d "${TODAY_ET} - ${LOOKBACK_DAYS} days" +%F)
DEFAULT_END=$(TZ=America/New_York date -d "${TODAY_ET} - 1 day" +%F)
START_DATE="${ATSPM_START_DATE:-${DEFAULT_START}}"
END_DATE="${ATSPM_END_DATE:-${DEFAULT_END}}"

if [[ ! "${START_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] ||
   [[ ! "${END_DATE}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    log "Invalid date override; expected YYYY-MM-DD."
    exit 64
fi

START_EPOCH=$(date -d "${START_DATE}" +%s) || exit 64
END_EPOCH=$(date -d "${END_DATE}" +%s) || exit 64
RANGE_DAYS=$(( (END_EPOCH - START_EPOCH) / 86400 + 1 ))
if (( RANGE_DAYS < 1 || RANGE_DAYS > MAX_RANGE_DAYS )); then
    log "Refusing date range of ${RANGE_DAYS} days (allowed: 1..${MAX_RANGE_DAYS})."
    exit 64
fi

log "============================================================"
log "🚀 ATSPM DAILY PIPELINE STARTED"
log "============================================================"
log "Date range: ${START_DATE} through ${END_DATE}"
log "Workers: extract=${MAX_WORKERS}, split=${SPLIT_WORKERS}"

# The Python entrypoint owns the common quality gate, exact date selection,
# completion states and summaries. This wrapper supplies the shared cron lock.
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/run_pipeline.py" \
    --steps all \
    --start-date "${START_DATE}" \
    --end-date "${END_DATE}" \
    --raw-dir "${RAW_DATA_DIR}" \
    --per-signal-dir "${PER_SIGNAL_DIR}" \
    --processed-dir "${PROCESSED_DATA_DIR}" \
    --config-dir "${CONFIG_DIR}" \
    --max-workers "${MAX_WORKERS}" \
    --split-workers "${SPLIT_WORKERS}" \
    --summary-file "${RUN_DIR}/feature_summary.json" \
    --lock-held >> "${PIPELINE_LOG}" 2>&1
FINAL_RC=$?
case "${FINAL_RC}" in
    0) log "ATSPM DAILY PIPELINE COMPLETE" ;;
    2) log "ATSPM DAILY PIPELINE PARTIAL: inspect source and feature reasons in the run reports" ;;
    *) log "ATSPM DAILY PIPELINE FAILED (rc=${FINAL_RC})" ;;
esac
log "Feature report: ${RUN_DIR}/feature_summary.json"
exit "${FINAL_RC}"
