#!/usr/bin/env python3
"""Read-only live dashboard for the extraction campaign; no pipeline controls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, urlparse

EXPECTED_JOBS = {}


def tail(path, limit=100_000):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2)
            stream.seek(max(0,stream.tell()-limit))
            return stream.read(limit).decode('utf-8',errors='replace')
    except OSError:
        return ''


def live_status(campaign):
    state=json.loads((campaign/'state.json').read_text())
    try:
        os.kill(state['pid'],0)
        process_state=Path(f'/proc/{state["pid"]}/status').read_text()
        state['supervisor_alive']='State:\tZ' not in process_state
    except (OSError,KeyError):
        state['supervisor_alive']=False
    state['observed_at']=datetime.now(timezone.utc).isoformat()
    current=state.get('current')
    database=Path(state['output_dir'])/'pipeline_runs.sqlite3'
    state['live']={'stage':'preparing','job_counts':{},'feature_counts':{},'reasons':{},'recent_jobs':[]}
    if not current:
        return state
    log=tail(Path(current['log_path']))
    state['live']['log_tail']=log[-30_000:]
    progress=re.findall(r'Processed (\d+)/(\d+) jobs',log)
    if progress:
        state['live']['expected_jobs']=int(progress[-1][1])
    if not database.exists():
        return state
    conn=sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=3)
    conn.row_factory=sqlite3.Row
    try:
        latest=conn.execute('SELECT run_id,started_at,status FROM runs ORDER BY rowid DESC LIMIT 1').fetchone()
        if not latest or latest['started_at']<current['started_at']:
            return state
        run=latest['run_id']
        state['live'].update(stage='extracting',run_id=run)
        if run not in EXPECTED_JOBS:
            year,month,day=map(int,current['date'].split('-'))
            inputs=Path('/data2/ATSPM/raw_data/per_signal')
            expected=set()
            for m in {str(month),f'{month:02d}'}:
                for d in {str(day),f'{day:02d}'}:
                    expected.update(inputs.glob(f'*/atspm-{year}-{m}-{d}.parquet'))
            EXPECTED_JOBS[run]=len(expected)
        if EXPECTED_JOBS[run]:
            state['live']['expected_jobs']=EXPECTED_JOBS[run]
        state['live']['job_counts']={r['status']:r['n'] for r in conn.execute(
            'SELECT status,COUNT(*) AS n FROM attempts WHERE run_id=? GROUP BY status',(run,))}
        state['live']['reasons']={r['reason_code']:r['n'] for r in conn.execute(
            "SELECT reason_code,COUNT(*) AS n FROM attempts WHERE run_id=? AND reason_code!='' GROUP BY reason_code ORDER BY n DESC",(run,))}
        state['live']['feature_counts']={r['status']:r['n'] for r in conn.execute(
            'SELECT f.status,COUNT(*) AS n FROM attempts a JOIN feature_results f ON f.attempt_id=a.attempt_id WHERE a.run_id=? GROUP BY f.status',(run,))}
        state['live']['recent_jobs']=[dict(r) for r in conn.execute(
            'SELECT signal_id,date_str,status,reason_code,finished_at FROM attempts WHERE run_id=? ORDER BY attempt_id DESC LIMIT 15',(run,))]
    finally:
        conn.close()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir',required=True,type=Path)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',default=8050,type=int)
    args=parser.parse_args()
    campaign=args.campaign_dir.resolve()
    page=Path(__file__).with_name('campaign.html')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request=urlparse(self.path)
            try:
                if request.path in ('/','/campaign.html'):
                    self.respond(page.read_bytes(),'text/html; charset=utf-8')
                elif request.path=='/api/status':
                    self.respond(json.dumps(live_status(campaign),ensure_ascii=False).encode(),'application/json; charset=utf-8')
                elif request.path=='/api/log':
                    state=json.loads((campaign/'state.json').read_text())
                    key=parse_qs(request.query).get('key',[''])[0]
                    record=state['records'].get(key) if key else state.get('current')
                    path=Path(record['log_path']) if record else campaign/'supervisor.log'
                    if not path.resolve().is_relative_to(campaign):
                        self.send_error(403)
                        return
                    self.respond(tail(path).encode(),'text/plain; charset=utf-8')
                else:
                    self.send_error(404)
            except Exception as exc:
                self.respond(json.dumps({'error':f'{type(exc).__name__}: {exc}'}).encode(),'application/json',503)

        def respond(self,body,content_type,status=200):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Campaign dashboard: http://{args.host}:{args.port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':
    main()
