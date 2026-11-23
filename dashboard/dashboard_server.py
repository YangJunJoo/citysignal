#!/usr/bin/env python3
"""
ATSPM Dashboard Backend Server
Queries the processing log database, runs/stops pipeline subprocesses,
and serves statistics via a lightweight JSON API.
"""

import os
import sys
import json
import sqlite3
import http.server
import socketserver
from datetime import datetime
import threading
import subprocess
import signal
import re
import glob
from collections import deque
import urllib.parse
import calendar
from pathlib import Path

# Find paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DB_PATH = "/data2/ATSPM/processed_data/processing_log.db"
RUN_DB_PATH = "/data2/ATSPM/processed_data/pipeline_runs.sqlite3"
RAW_PER_SIGNAL_DIR = "/data2/ATSPM/raw_data/per_signal"
DOWNLOAD_LOG_PATH = "/data2/ATSPM/raw_data/download_log.json"
SPLIT_CACHE_PATH = "/data2/ATSPM/raw_data/per_signal/.cache/processed_files_cache.json"
CONFIG_DIR = "/data2/ATSPM/config_data/signal_config_2025/lanes"
PIPELINE_LOCK_PATH = "/data2/ATSPM/.pipeline.lock"

# Global pipeline run state
pipeline_state = {
    "process": None,      # subprocess.Popen object
    "start_time": None,   # ISO timestamp string
    "steps": [],          # list of steps
    "start_date": None,   # YYYY-MM-DD
    "end_date": None,     # YYYY-MM-DD
    "status": "idle",     # "idle", "running", "completed", "failed", "killed", "resuming"
    "auto_resume": False, # Boolean flag to auto-resume on crash
    "retry_count": 0,     # Current retry attempts
    "cmd_args": []        # Saved command-line arguments to restart the process
}
pipeline_lock = threading.Lock()


def get_external_pipeline_pids():
    """Find all running run_pipeline.py processes on the system (excluding this process)."""
    pids = []
    try:
        output = subprocess.check_output(["pgrep", "-f", "run_pipeline.py"]).decode().strip()
        if output:
            for line in output.split('\n'):
                pid = int(line)
                if pid != os.getpid():
                    pids.append(pid)
    except Exception:
        pass
    return pids


def restart_pipeline_run():
    """Restarts the pipeline using the saved command-line arguments."""
    global pipeline_state, pipeline_lock
    
    with pipeline_lock:
        if pipeline_state["status"] != "resuming":
            return
            
        cmd = pipeline_state["cmd_args"]
        log_filepath = os.path.join(project_root, "pipeline_run.log")
        
        try:
            with open(log_filepath, "a") as f:
                f.write(f"\n\n=== [AUTO-RESUME] Attempt {pipeline_state['retry_count']}/3 at {datetime.now().isoformat()} ===\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
            
            log_file = open(log_filepath, "a")
            # Start background process in its own group
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                cwd=project_root
            )
            
            pipeline_state["process"] = proc
            pipeline_state["status"] = "running"
            pipeline_state["start_time"] = datetime.now().isoformat()
            
            # Start monitor thread
            t = threading.Thread(target=monitor_pipeline_process, args=(proc, pipeline_state, pipeline_lock))
            t.daemon = True
            t.start()
            print(f"[AUTO-RESUME] Successfully restarted pipeline (PID: {proc.pid})")
        except Exception as e:
            print(f"[AUTO-RESUME] Failed to restart pipeline: {e}")
            pipeline_state["status"] = "failed"
            pipeline_state["retry_count"] = 0


def monitor_pipeline_process(proc, state_dict, lock):
    """Monitor background pipeline execution and update state on completion or crash."""
    return_code = proc.wait()
    with lock:
        if state_dict["process"] == proc:
            state_dict["process"] = None
            if return_code == 0:
                state_dict["status"] = "completed"
                state_dict["retry_count"] = 0
            elif return_code == 2:
                state_dict["status"] = "partial"
                state_dict["retry_count"] = 0
            elif return_code in [-9, 9, 15, -15]:  # Explicitly killed by user
                state_dict["status"] = "killed"
                state_dict["retry_count"] = 0
            else:
                # Process crashed (unexpected exit code)
                if state_dict["auto_resume"] and state_dict["retry_count"] < 3:
                    state_dict["retry_count"] += 1
                    state_dict["status"] = "resuming"
                    print(f"Pipeline process crashed with code {return_code}. Triggering auto-resume attempt {state_dict['retry_count']}/3...")
                    threading.Thread(target=restart_pipeline_run).start()
                else:
                    state_dict["status"] = "failed"
                    state_dict["retry_count"] = 0


def tail_file(file_path, num_lines=200):
    """Retrieve the last N lines of a file efficiently."""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", errors="ignore") as f:
            return "".join(deque(f, num_lines))
    except Exception as e:
        return f"Error reading logs: {e}"


def get_days_in_month(month_str):
    """Get all date strings (YYYY-MM-DD) for a given month YYYY-MM."""
    try:
        year, month = map(int, month_str.split('-'))
        _, num_days = calendar.monthrange(year, month)
        dates = [f"{year:04d}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]
        return dates
    except Exception:
        return []


def extract_date_from_path(filepath):
    """Extract date string YYYY-MM-DD from parquet filepath/filename."""
    filename = os.path.basename(filepath)
    patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        r'(\d{4})(\d{2})(\d{2})',
        r'atspm-(\d{4})-(\d{1,2})-(\d{1,2})'
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            year, month, day = map(int, match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _read_connection(path):
    """Read dashboards without creating or migrating an operational database."""
    if not os.path.isfile(path):
        return None
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _daily_feature_counts():
    counts = {}
    checked = _read_connection(RUN_DB_PATH)
    try:
        if checked is not None:
            for row in checked.execute('SELECT date_str,status,COUNT(*) AS n FROM latest_attempts GROUP BY date_str,status'):
                counts.setdefault(row['date_str'], {})[row['status']] = row['n']
            if os.path.isfile(DB_PATH):
                checked.execute('ATTACH DATABASE ? AS legacy', (Path(DB_PATH).resolve().as_uri() + '?mode=ro',))
                for row in checked.execute('''SELECT l.date_str,COUNT(DISTINCT l.signal_id) AS n
                    FROM legacy.processing_log l WHERE NOT EXISTS
                    (SELECT 1 FROM latest_attempts a WHERE a.signal_id=l.signal_id AND a.date_str=l.date_str)
                    GROUP BY l.date_str'''):
                    counts.setdefault(row['date_str'], {})['legacy_unverified'] = row['n']
        else:
            legacy = _read_connection(DB_PATH)
            if legacy is not None:
                try:
                    for row in legacy.execute('SELECT date_str,COUNT(DISTINCT signal_id) AS n FROM processing_log GROUP BY date_str'):
                        counts.setdefault(row['date_str'], {})['legacy_unverified'] = row['n']
                finally:
                    legacy.close()
    finally:
        if checked is not None:
            checked.close()
    return counts


def get_feature_statistics():
    """Count latest checked outcomes; legacy feature rows cannot inflate success."""
    result = {
        'total_features_success': 0, 'total_features_failed': 0,
        'run_signals_success': 0, 'run_features_success': 0, 'run_features_failed': 0,
        'speed_signals_per_min': 0.0, 'elapsed_time': '0s', 'recent_activity': [],
        'daily_progress': [], 'feature_status_counts': {}, 'job_status_counts': {},
        'job_reason_counts': {}, 'feature_reason_counts': {}, 'latest_run': None,
        'legacy_unverified_jobs': 0,
    }
    daily_counts = _daily_feature_counts()
    result['legacy_unverified_jobs'] = sum(row.get('legacy_unverified', 0) + row.get('imported_unverified', 0) for row in daily_counts.values())
    conn = _read_connection(RUN_DB_PATH)
    if conn is None:
        return result
    try:
        for row in conn.execute('''SELECT f.status,COUNT(*) AS n FROM feature_results f
            JOIN latest_attempts a ON a.attempt_id=f.attempt_id GROUP BY f.status'''):
            result['feature_status_counts'][row['status']] = row['n']
        result['total_features_success'] = result['feature_status_counts'].get('success', 0)
        result['total_features_failed'] = result['feature_status_counts'].get('failed', 0)
        for row in conn.execute('SELECT status,COUNT(*) AS n FROM latest_attempts GROUP BY status'):
            result['job_status_counts'][row['status']] = row['n']
        for row in conn.execute('SELECT reason_code,COUNT(*) AS n FROM latest_attempts GROUP BY reason_code'):
            result['job_reason_counts'][row['reason_code'] or 'unspecified'] = row['n']
        for row in conn.execute('''SELECT f.reason_code,COUNT(*) AS n FROM feature_results f
            JOIN latest_attempts a ON a.attempt_id=f.attempt_id GROUP BY f.reason_code'''):
            result['feature_reason_counts'][row['reason_code'] or 'unspecified'] = row['n']
        run = conn.execute("SELECT run_id,started_at,finished_at,status FROM runs WHERE status != 'imported_unverified' ORDER BY started_at DESC LIMIT 1").fetchone()
        if run:
            result['latest_run'] = dict(run)
            result['run_signals_success'] = conn.execute("SELECT COUNT(*) FROM latest_attempts WHERE run_id=? AND status='complete'", (run['run_id'],)).fetchone()[0]
            for row in conn.execute('''SELECT f.status,COUNT(*) AS n FROM feature_results f
                JOIN latest_attempts a ON a.attempt_id=f.attempt_id WHERE a.run_id=? GROUP BY f.status''', (run['run_id'],)):
                if row['status'] == 'success':
                    result['run_features_success'] = row['n']
                elif row['status'] == 'failed':
                    result['run_features_failed'] = row['n']
            if run['started_at']:
                start = datetime.fromisoformat(run['started_at'])
                end = datetime.fromisoformat(run['finished_at']) if run['finished_at'] else datetime.now(start.tzinfo)
                elapsed = max((end - start).total_seconds(), 0)
                result['elapsed_time'] = f'{int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m'
                if elapsed:
                    result['speed_signals_per_min'] = round(result['run_signals_success'] / elapsed * 60, 2)
        for row in conn.execute('''SELECT a.signal_id,a.date_str,f.name,f.status,a.finished_at,
            f.reason_code,f.message FROM feature_results f JOIN attempts a ON a.attempt_id=f.attempt_id
            ORDER BY a.attempt_id DESC,f.name LIMIT 15'''):
            result['recent_activity'].append({'signal_id': row['signal_id'], 'date_str': row['date_str'],
                'feature_name': row['name'], 'status': row['status'], 'time': row['finished_at'] or '',
                'reason_code': row['reason_code'], 'error': row['message'] or ''})
        result['daily_progress'] = [{'date': day, 'signals': daily_counts[day].get('complete', 0)}
                                   for day in sorted(daily_counts)[-15:]]
        return result
    finally:
        conn.close()


def get_daily_status(requested_dates=None):
    """Aggregate progress status for Download, Split, and Feature Extraction stages by date."""
    # 1. Load download log
    download_statuses = {}
    try:
        if os.path.exists(DOWNLOAD_LOG_PATH):
            with open(DOWNLOAD_LOG_PATH, "r") as f:
                d_log = json.load(f)
                for k, v in d_log.items():
                    try:
                        dt = datetime.strptime(k, "%Y-%m-%d")
                        norm_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        parts = k.split('-')
                        if len(parts) == 3:
                            norm_date = f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                        else:
                            continue
                    download_statuses[norm_date] = v.get("status", "pending")
    except Exception:
        pass

    # 2. Load split cache
    split_statuses = {}
    try:
        if os.path.exists(SPLIT_CACHE_PATH):
            with open(SPLIT_CACHE_PATH, "r") as f:
                s_cache = json.load(f)
                for path, info in s_cache.items():
                    norm_date = extract_date_from_path(path)
                    if norm_date:
                        is_processed = info.get("processed", False)
                        split_statuses[norm_date] = "completed" if is_processed else "failed"
    except Exception:
        pass

    # 3. Each signal-day has exactly one latest checked state.
    db_stats = _daily_feature_counts()

    # Determine date range to compile
    if requested_dates is None:
        all_dates = sorted(list(set(list(download_statuses.keys()) + list(split_statuses.keys()) + list(db_stats.keys()))), reverse=True)
        requested_dates = all_dates[:20]  # Return last 20 active dates by default
    else:
        # Keep chronological order for calendar grid rendering
        requested_dates = sorted(requested_dates)

    progress_list = []
    for date_str in requested_dates:
        # Download Stage
        dl_status = download_statuses.get(date_str, "pending")
        if dl_status == "completed":
            dl_status_str = "completed"
        elif dl_status == "failed" or dl_status == "error":
            dl_status_str = "failed"
        else:
            dl_status_str = "pending"

        # Split Stage
        sp_status = split_statuses.get(date_str, "pending")

        # Extraction Stage
        ext_info = db_stats.get(date_str, {})
        success_count = ext_info.get("complete", 0)
        failed_count = ext_info.get("failed", 0)
        blocked_count = ext_info.get("blocked", 0)
        partial_count = ext_info.get("partial", 0)
        unverified_count = ext_info.get("legacy_unverified", 0) + ext_info.get("imported_unverified", 0)
        running_count = ext_info.get("running", 0)
        processed_count = sum(ext_info.values()) - unverified_count - running_count

        # Availability establishes the denominator; a percentage of failed or
        # unverified jobs can never establish daily completion.
        total_signals = 0
        raw_signal_ids = set()
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            filename = f"atspm-{dt.year}-{dt.month}-{dt.day}.parquet"
            matching_files = glob.glob(os.path.join(RAW_PER_SIGNAL_DIR, "*", filename))
            raw_signal_ids = {Path(path).parent.name for path in matching_files}
            total_signals = len(raw_signal_ids)
        except ValueError:
            pass
        total_signals = max(total_signals, sum(ext_info.values()))
        pending_count = max(total_signals - sum(ext_info.values()), 0)
        if success_count == total_signals and raw_signal_ids:
            conn = _read_connection(RUN_DB_PATH)
            try:
                complete_ids = {row[0] for row in conn.execute(
                    "SELECT signal_id FROM latest_attempts WHERE date_str=? AND status='complete'", (date_str,))} if conn else set()
            finally:
                if conn is not None:
                    conn.close()
            # Equal counts can still describe different signals.
            missing_ids = raw_signal_ids - complete_ids
            pending_count += len(missing_ids)
            total_signals += len(missing_ids)
        if total_signals > 0 and success_count == total_signals:
            ext_status_str = "completed"
        elif running_count:
            ext_status_str = "in_progress"
        elif processed_count:
            ext_status_str = "partial"
        elif unverified_count:
            ext_status_str = "legacy_unverified"
        else:
            ext_status_str = "pending"

        # Check if the date is in the range of the current running pipeline run
        is_active_date_run = False
        with pipeline_lock:
            if pipeline_state["status"] in ["running", "resuming"]:
                start_date_str = pipeline_state["start_date"]
                end_date_str = pipeline_state["end_date"]
                if start_date_str and end_date_str:
                    is_active_date_run = (start_date_str <= date_str <= end_date_str)
                elif start_date_str:
                    is_active_date_run = (date_str == start_date_str)

        # Categorize overall status into color class
        # Red: nothing done (download/split pending or failed)
        # Orange: download + split done, but extraction pending or incomplete
        # Green: every expected signal-day is verified complete.
        # Blue: extraction in progress or actively running in current pipeline range
        if ext_status_str == "completed" and not is_active_date_run:
            color = "green"
        elif is_active_date_run:
            color = "blue"
        elif ext_status_str == "completed":
            color = "green"
        elif ext_status_str == "in_progress":
            color = "blue"
        elif sp_status == "completed":
            color = "orange"
        else:
            color = "red"

        progress_list.append({
            "date": date_str,
            "download": dl_status_str,
            "split": sp_status,
            "extraction": {
                "status": ext_status_str,
                "processed": processed_count,
                "success": success_count,
                "failed": failed_count,
                "blocked": blocked_count,
                "partial": partial_count,
                "unverified": unverified_count,
                "pending": pending_count,
                "running": running_count,
                "status_counts": ext_info,
                "total": total_signals
            },
            "color": color
        })

    return progress_list


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=script_dir, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == '/api/pipeline/run':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
                
            steps = data.get('steps', ['download', 'split', 'extract'])
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            force_kill = data.get('force_kill', False)
            auto_resume = data.get('auto_resume', False)
            
            # Check for conflicting background processes
            external_pids = get_external_pipeline_pids()
            if external_pids:
                if force_kill:
                    for pid in external_pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except Exception:
                            pass
                else:
                    self.send_response(409)  # Conflict
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'error': 'Conflicting pipeline process is already running outside dashboard.',
                        'pids': external_pids
                    }).encode('utf-8'))
                    return
            
            with pipeline_lock:
                if pipeline_state["status"] == "running":
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Pipeline is already running.'}).encode('utf-8'))
                    return
                
                # Assemble Command
                pipeline_cmd = [
                    "/data2/ya195253/conda/envs/atspm/bin/python",
                    "-u",
                    os.path.join(project_root, "run_pipeline.py"),
                    "--steps"
                ] + steps
                
                if start_date:
                    pipeline_cmd.extend(["--start-date", start_date])
                if end_date:
                    pipeline_cmd.extend(["--end-date", end_date])
                pipeline_cmd.extend([
                    "--max-workers", "38",
                    "--config-dir", CONFIG_DIR,
                ])
                # Share the same non-blocking lock as cron and manual backfills.
                pipeline_cmd.append("--lock-held")
                cmd = ["flock", "-n", PIPELINE_LOCK_PATH] + pipeline_cmd
                
                # Overwrite logs file to start fresh
                log_filepath = os.path.join(project_root, "pipeline_run.log")
                try:
                    with open(log_filepath, "w") as f:
                        f.write(f"=== Pipeline Run Started at {datetime.now().isoformat()} ===\n")
                        f.write(f"Command: {' '.join(cmd)}\n\n")
                except Exception:
                    pass
                
                try:
                    log_file = open(log_filepath, "a")
                    # Start background process in its own group
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid,
                        cwd=project_root
                    )
                    
                    pipeline_state["process"] = proc
                    pipeline_state["status"] = "running"
                    pipeline_state["start_time"] = datetime.now().isoformat()
                    pipeline_state["steps"] = steps
                    pipeline_state["start_date"] = start_date
                    pipeline_state["end_date"] = end_date
                    pipeline_state["auto_resume"] = auto_resume
                    pipeline_state["retry_count"] = 0
                    pipeline_state["cmd_args"] = cmd
                    
                    # Monitor subprocess thread
                    t = threading.Thread(target=monitor_pipeline_process, args=(proc, pipeline_state, pipeline_lock))
                    t.daemon = True
                    t.start()
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'message': 'Pipeline started successfully.'}).encode('utf-8'))
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': f'Failed to start pipeline: {e}'}).encode('utf-8'))
                    
        elif path == '/api/pipeline/stop':
            with pipeline_lock:
                proc = pipeline_state["process"]
                if proc:
                    try:
                        # Terminate entire process group
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    pipeline_state["status"] = "killed"
                    pipeline_state["process"] = None
                    pipeline_state["retry_count"] = 0
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'message': 'Pipeline stopped successfully.'}).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'No active pipeline run to stop.'}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        
        if path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            try:
                response = get_feature_statistics()

                # 7. Date-by-date stages status matrix (customized for month)
                month_arg = query.get('month', [None])[0]
                if not month_arg:
                    # Default to current execution month or current calendar month
                    with pipeline_lock:
                        if pipeline_state["status"] == "running" and pipeline_state["start_date"]:
                            month_arg = pipeline_state["start_date"][:7]  # YYYY-MM
                        else:
                            month_arg = datetime.now().strftime("%Y-%m")
                            
                requested_dates = get_days_in_month(month_arg)
                dates_progress = get_daily_status(requested_dates)
                
                response.update({
                    'dates_progress': dates_progress,
                    'server_time': datetime.now().strftime('%H:%M:%S'),
                    'selected_month': month_arg,
                })

                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                
        elif path == '/api/pipeline/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            with pipeline_lock:
                elapsed_str = "0s"
                if pipeline_state["status"] == "running" and pipeline_state["start_time"]:
                    t1 = datetime.fromisoformat(pipeline_state["start_time"])
                    diff = (datetime.now() - t1).total_seconds()
                    hours = int(diff // 3600)
                    minutes = int((diff % 3600) // 60)
                    seconds = int(diff % 60)
                    if hours > 0:
                        elapsed_str = f"{hours}h {minutes}m"
                    elif minutes > 0:
                        elapsed_str = f"{minutes}m {seconds}s"
                    else:
                        elapsed_str = f"{seconds}s"
                        
                response = {
                    'status': pipeline_state['status'],
                    'start_time': pipeline_state['start_time'],
                    'steps': pipeline_state['steps'],
                    'start_date': pipeline_state['start_date'],
                    'end_date': pipeline_state['end_date'],
                    'elapsed_time': elapsed_str,
                    'auto_resume': pipeline_state['auto_resume'],
                    'retry_count': pipeline_state['retry_count']
                }
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        elif path == '/api/pipeline/logs':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            lines_arg = query.get('lines', ['200'])
            try:
                num_lines = int(lines_arg[0])
            except ValueError:
                num_lines = 200
                
            log_filepath = os.path.join(project_root, "pipeline_run.log")
            log_content = tail_file(log_filepath, num_lines)
            self.wfile.write(log_content.encode('utf-8'))
            
        else:
            super().do_GET()


def run_server(port=8050):
    handler = DashboardHandler
    while port < 8100:
        try:
            with socketserver.TCPServer(("", port), handler) as httpd:
                print("="*60)
                print(f"ATSPM Dashboard Web Server successfully started!")
                print(f"URL: http://localhost:{port}/dashboard.html")
                print("Press Ctrl+C to terminate the dashboard server")
                print("="*60)
                httpd.serve_forever()
                break
        except OSError:
            print(f"Port {port} in use, trying port {port + 1}...")
            port += 1


if __name__ == "__main__":
    try:
        port_num = 8050
        if len(sys.argv) > 1:
            try:
                port_num = int(sys.argv[1])
            except ValueError:
                pass
        run_server(port_num)
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        sys.exit(0)
