#!/bin/bash
###############################################################################
# ATSPM Full Pipeline: Download → Split → Feature Extraction
# 
# Config:
#   - CPU: 12 workers (~20% of available)
#   - 12-hour progress reports
#   - Config logging to pipeline_config_*.log
#   - Uses conda atspm environment
#
# Usage:
#   nohup bash scripts/run_full_pipeline.sh &
#   tail -f logs/pipeline_*.log
###############################################################################

# ─── Configuration ──────────────────────────────────────────────────────────
TOTAL_CPU=$(nproc)
MAX_WORKERS=12

PROJECT_ROOT="/home/ya195253/YJ/python_scripts/ATSPM"
RAW_DATA_DIR="/data2/ATSPM/raw_data"
PER_SIGNAL_DIR="${RAW_DATA_DIR}/per_signal"
PROCESSED_DATA_DIR="/data2/ATSPM/processed_data"
CONFIG_DIR="/data2/ATSPM/config_data/signal_config_2025/lanes"
LOG_DIR="${PROJECT_ROOT}/logs"

# Ensure Conda is available (required for cron/nohup environments)
source ~/anaconda3/etc/profile.d/conda.sh || true

# Use conda atspm environment
PYTHON_CMD="conda run -n atspm python3"

PIPELINE_START=$(date +%s)
PIPELINE_START_STR=$(date '+%Y-%m-%d %H:%M:%S')
RUN_ID=$(date '+%m-%d-%Y@%H-%M-%S')

REPORT_INTERVAL=$((12 * 3600))

PIPELINE_LOG="${LOG_DIR}/pipeline_${RUN_ID}.log"
CONFIG_LOG="${LOG_DIR}/pipeline_config_${RUN_ID}.log"
PROGRESS_LOG="${LOG_DIR}/pipeline_progress_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"

# ─── Helpers ────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${PIPELINE_LOG}"
}

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${PROGRESS_LOG}" | tee -a "${PIPELINE_LOG}"
}

elapsed_time() {
    local diff=$(( $(date +%s) - $1 ))
    echo "$((diff/3600))h $((diff%3600/60))m"
}

write_progress_report() {
    local stage=$1
    log_progress "============================================================"
    log_progress "📊 PROGRESS REPORT (Elapsed: $(elapsed_time ${PIPELINE_START}))"
    log_progress "   Stage: ${stage}"
    log_progress "------------------------------------------------------------"

    local raw_count=$(ls "${RAW_DATA_DIR}"/*.parquet 2>/dev/null | wc -l)
    local latest_raw=$(ls -t "${RAW_DATA_DIR}"/*.parquet 2>/dev/null | head -1 | xargs -r basename)
    log_progress "   📥 Raw files: ${raw_count} (latest: ${latest_raw:-none})"

    local signal_count=$(ls "${PER_SIGNAL_DIR}" 2>/dev/null | wc -l)
    log_progress "   ✂️  Signals: ${signal_count} dirs"

    if [ -f "${PROCESSED_DATA_DIR}/processing_log.db" ]; then
        local fe_info=$(${PYTHON_CMD} -c "
import sqlite3
try:
    conn = sqlite3.connect('${PROCESSED_DATA_DIR}/processing_log.db')
    c = conn.cursor()
    c.execute(\"SELECT COUNT(DISTINCT date_str), COUNT(DISTINCT signal_id) FROM processing_log WHERE status='success'\")
    d, s = c.fetchone()
    c.execute(\"SELECT MAX(date_str) FROM processing_log WHERE status='success'\")
    mx = c.fetchone()[0]
    print(f'{d} dates x {s} signals, latest={mx}')
    conn.close()
except Exception as e: print(f'DB error: {e}')
" 2>/dev/null)
        log_progress "   🔧 Features: ${fe_info}"
    fi
    log_progress "============================================================"
}

# Background 12h reporter
start_reporter() {
    (
        while true; do
            sleep ${REPORT_INTERVAL}
            write_progress_report "$1"
        done
    ) &
    REPORTER_PID=$!
}

stop_reporter() {
    if [ -n "${REPORTER_PID:-}" ]; then
        kill ${REPORTER_PID} 2>/dev/null || true
        wait ${REPORTER_PID} 2>/dev/null || true
        unset REPORTER_PID
    fi
}

# ─── Config Log ─────────────────────────────────────────────────────────────

cat > "${CONFIG_LOG}" << CFGEOF
============================================================
ATSPM Full Pipeline Configuration
============================================================
Run ID:             ${RUN_ID}
Start Time:         ${PIPELINE_START_STR}
Host:               $(hostname)

--- System ---
Total CPUs:         ${TOTAL_CPU}
Workers (20%):      ${MAX_WORKERS}
RAM Total:          $(free -h | awk '/^Mem:/{print $2}')
RAM Available:      $(free -h | awk '/^Mem:/{print $7}')
Conda Env:          atspm
Python:             $(conda run -n atspm which python3 2>/dev/null)

--- Paths ---
Raw Data:           ${RAW_DATA_DIR}
Per-Signal:         ${PER_SIGNAL_DIR}
Processed:          ${PROCESSED_DATA_DIR}
Configs:            ${CONFIG_DIR} ($(ls "${CONFIG_DIR}" 2>/dev/null | wc -l) files)

--- Stage 1: Download ---
URL Pattern:        https://sunstore.cflsmartroads.com/.../atspm-{date}.csv
Date Range:         2025-08-01 ~ current-5

--- Stage 2: Split by Signal ---
Overwrite:          False

--- Stage 3: Feature Extraction ---
Engine:             V2 (vectorized)
Max Workers:        ${MAX_WORKERS}
Features:           All except conflict_propensity

--- Logs ---
Pipeline:           ${PIPELINE_LOG}
Config:             ${CONFIG_LOG}
Progress:           ${PROGRESS_LOG}
============================================================
CFGEOF

log "Config saved to ${CONFIG_LOG}"

# ─── STAGE 1: Download ─────────────────────────────────────────────────────

log "═══════════════ STAGE 1/3: DATA DOWNLOAD ═══════════════"
S1_START=$(date +%s)
start_reporter "Stage 1: Download"

${PYTHON_CMD} -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/data_downloader')
from data_downloader import main
main()
" >> "${PIPELINE_LOG}" 2>&1
S1_RC=$?

stop_reporter
log "Stage 1 done (rc=${S1_RC}) in $(elapsed_time ${S1_START})"
write_progress_report "Stage 1 Done → Stage 2"

# ─── STAGE 2: Split by Signal ──────────────────────────────────────────────

log "═══════════════ STAGE 2/3: SPLIT BY SIGNAL ═══════════════"
S2_START=$(date +%s)
start_reporter "Stage 2: Split"

${PYTHON_CMD} -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/data_downloader')
from split_by_signal import run_processing
from pathlib import Path
success = run_processing(Path('${RAW_DATA_DIR}'), Path('${PER_SIGNAL_DIR}'), overwrite=False)
sys.exit(0 if success else 1)
" >> "${PIPELINE_LOG}" 2>&1
S2_RC=$?

stop_reporter
log "Stage 2 done (rc=${S2_RC}) in $(elapsed_time ${S2_START})"
write_progress_report "Stage 2 Done → Stage 3"

log "═══════════════ STAGE 3/3: FEATURE EXTRACTION (Chunked by Date) ═══════════════"
S3_START=$(date +%s)
start_reporter "Stage 3: Features"

# Find all dates from parquet files and sort them chronologically
DATES_FILE="/tmp/atspm_dates_${RUN_ID}.txt"
${PYTHON_CMD} -c "
import glob, sys
from pathlib import Path

files = glob.glob('${RAW_DATA_DIR}/atspm-*.parquet')
dates = []
for f in files:
    basename = Path(f).stem
    if basename.startswith('atspm-'):
        d_str = basename[6:]
        parts = d_str.split('-')
        if len(parts) == 3:
            dates.append((int(parts[0]), int(parts[1]), int(parts[2])))
dates.sort()
for y, m, d in dates:
    print(f'{y} {m} {d}')
" > "${DATES_FILE}"

TOTAL_DAYS=$(wc -l < "${DATES_FILE}")
log "Found ${TOTAL_DAYS} unique dates to process."

CURRENT_DAY=1
while read year month day; do
    log "--- Processing Day ${CURRENT_DAY}/${TOTAL_DAYS} : ${year}-${month}-${day} ---"
    
    ${PYTHON_CMD} "${PROJECT_ROOT}/feature_extractor/process_local_atspm.py" \
        --event-dir "${RAW_DATA_DIR}" \
        --config-dir "${CONFIG_DIR}" \
        --output-dir "${PROCESSED_DATA_DIR}" \
        --max-workers ${MAX_WORKERS} \
        --no-progress-bar \
        --single-date \
        --year "${year}" \
        --month "${month}" \
        --day "${day}" \
        >> "${PIPELINE_LOG}" 2>&1
        
    FE_RC=$?
    if [ $FE_RC -ne 0 ] && [ $FE_RC -ne 1 ]; then
        log "⚠️ Worker crashed or failed on ${year}-${month}-${day} (rc=${FE_RC}). Continuing to next day..."
    fi
    
    CURRENT_DAY=$((CURRENT_DAY + 1))
done < "${DATES_FILE}"

rm -f "${DATES_FILE}"

stop_reporter
log "Stage 3 done in $(elapsed_time ${S3_START})"

# ─── Final ──────────────────────────────────────────────────────────────────

log "🎉 PIPELINE COMPLETE (Total: $(elapsed_time ${PIPELINE_START}))"
log "   Download rc=${S1_RC} | Split rc=${S2_RC} | Features rc=${S3_RC}"
write_progress_report "✅ PIPELINE COMPLETE"

echo "Config:   ${CONFIG_LOG}"
echo "Pipeline: ${PIPELINE_LOG}"
echo "Progress: ${PROGRESS_LOG}"
