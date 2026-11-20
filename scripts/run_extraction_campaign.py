#!/usr/bin/env python3
"""Run priority dates, then newest-first dates into a separate fresh output tree."""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
RANGES = [('2024-04-30','2024-05-31'), ('2024-09-30','2024-10-15'),
          ('2025-01-31','2025-02-07'), ('2025-08-31','2025-09-15')]
TERMINAL = {'complete', 'partial', 'failed'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def dates_between(first, last):
    current, end = date.fromisoformat(first), date.fromisoformat(last)
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=1)


def priority_plan():
    targets = [day for first,last in RANGES for day in dates_between(first,last)]
    boundaries = [(date.fromisoformat(last)+timedelta(days=1)).isoformat() for _,last in RANGES]
    return targets, boundaries, sorted(set(targets+boundaries))


def newest_plan(raw_dir, exclude):
    available = set()
    for path in raw_dir.glob('atspm-*.parquet'):
        match = re.fullmatch(r'atspm-(\d{4})-(\d{1,2})-(\d{1,2})', path.stem)
        if match:
            available.add(date(*map(int, match.groups())).isoformat())
    if available:
        yesterday = (datetime.now(ZoneInfo('America/New_York')).date()-timedelta(days=1)).isoformat()
        available.update(dates_between(max(available), yesterday))
    return sorted(available-set(exclude), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--processed-dir', type=Path, required=True,
                        help='Separate empty output directory; never the existing operational result tree.')
    parser.add_argument('--raw-dir', type=Path, default=Path('/data2/ATSPM/raw_data'))
    parser.add_argument('--config-dir', type=Path, default=Path('/data2/ATSPM/config_data/signal_config_2025/lanes'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.processed_dir.resolve()
    canonical = Path('/data2/ATSPM/processed_data').resolve()
    if output == canonical or canonical in output.parents:
        parser.error('Use a separate output tree to preserve existing operational results.')
    campaign = args.campaign_dir.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign/'logs').mkdir(exist_ok=True)
    state_path = campaign/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['output_dir'] != str(output):
            parser.error('A resumed campaign must keep its original output directory.')
    else:
        if output.exists() and any(output.iterdir()):
            parser.error('A new campaign requires an empty output directory.')
        targets, boundaries, sources = priority_plan()
        state = {'created_at':now(), 'priority_ranges':RANGES, 'priority_target_dates':targets,
                 'boundary_source_dates':boundaries, 'priority_source_dates':sources,
                 'latest_source_dates':[], 'records':{}, 'phase':'priority', 'status':'starting',
                 'workers':args.workers, 'output_dir':str(output),
                 'campaign_dir':str(campaign), 'version':'0.1.1', 'current':None,
                 'ordering':'Priority source dates ascending; remaining source dates newest first.',
                 'completion_policy':'Dates are attempted once; partial/failed dates remain unresolved.'}
    unique = (campaign/'campaign.lock').open('a')
    fcntl.flock(unique, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state.update(pid=os.getpid(), status='waiting_for_pipeline_lock', updated_at=now())

    def publish():
        state['updated_at'] = now()
        state['date_counts'] = dict(Counter(r['status'] for r in state['records'].values()))
        atomic_json(state_path, state)

    publish()
    operational_lock = (output.parent/'.pipeline.lock').open('a')
    while True:
        try:
            fcntl.flock(operational_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            publish()
            time.sleep(10)
    child = None

    def stop(signum, frame):
        state['status'] = 'stopping'
        publish()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        state['status'] = 'interrupted'
        publish()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for phase in ('priority','latest'):
            if phase == 'latest' and not state['latest_source_dates']:
                state['latest_source_dates'] = newest_plan(args.raw_dir,state['priority_source_dates'])
            state['phase'] = phase
            for day in state[phase+'_source_dates']:
                key = phase+':'+day
                if state['records'].get(key,{}).get('status') in TERMINAL:
                    continue
                log_path = campaign/'logs'/(phase+'_'+day+'.log')
                summary_path = campaign/(phase+'_'+day+'.json')
                record = {'date':day,'phase':phase,'status':'running','started_at':now(),
                          'log_path':str(log_path),'summary_path':str(summary_path),
                          'boundary_source_day':day in state['boundary_source_dates']}
                state['records'][key] = record
                state['current'] = dict(record)
                state['status'] = 'running'
                publish()
                cmd = [sys.executable, '-u', str(PROJECT/'run_pipeline.py'), '--steps','all',
                       '--start-date',day,'--end-date',day,'--raw-dir',str(args.raw_dir),
                       '--per-signal-dir',str(args.raw_dir/'per_signal'),
                       '--processed-dir',str(output),'--config-dir',str(args.config_dir),
                       '--max-workers',str(args.workers),'--split-workers','1',
                       '--summary-file',str(summary_path),'--lock-held']
                env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           PYTHONUNBUFFERED='1')
                print(f'{now()} {phase}: {day}',flush=True)
                with log_path.open('a') as log:
                    child = subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,env=env,pass_fds=(operational_lock.fileno(),))
                    state['current']['pid'] = child.pid
                    while child.poll() is None:
                        publish()
                        time.sleep(5)
                    code = child.returncode
                record.update(finished_at=now(),exit_code=code,
                              status='complete' if code==0 else ('partial' if code==2 else 'failed'))
                if summary_path.exists():
                    try:
                        summary=json.loads(summary_path.read_text())
                        record['summary']={k:v for k,v in summary.items() if k!='outcomes'}
                    except (OSError,ValueError) as exc:
                        record['summary_error']=str(exc)
                state['current']=None
                publish()
                print(f'{now()} {phase}: {day} -> {record["status"]} (rc={code})',flush=True)
                child = None
            state[phase+'_attempts_finished_at'] = now()
            publish()
        state['status'] = ('finished_with_unresolved' if any(r['status']!='complete' for r in state['records'].values())
                           else 'complete')
        state['finished_at'] = now()
        publish()
    except BaseException:
        if state['status'] not in ('interrupted','stopping'):
            state['status']='failed'
            publish()
        raise
    finally:
        operational_lock.close()
        unique.close()


if __name__ == '__main__':
    main()
