#!/usr/bin/env python3
"""
run_all_models.py - Launch 4 imputation models on 4 separate GPUs.
Uses torch-spatiotemporal (tsl) for GRIN and SPIN,
custom implementations for IGNNK and PriSTI.

Usage:
  conda activate /data2/ya195253/envs/stgnn
  python scripts/run_all_models.py
"""

import os
import sys
import json
import time
import subprocess

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    'grin':   {'gpu': 4, 'script': os.path.join(SCRIPTS_DIR, 'train_grin.py')},
    'spin':   {'gpu': 5, 'script': os.path.join(SCRIPTS_DIR, 'train_spin.py')},
    'ignnk':  {'gpu': 6, 'script': os.path.join(SCRIPTS_DIR, 'train_ignnk.py')},
    'pristi': {'gpu': 7, 'script': os.path.join(SCRIPTS_DIR, 'train_pristi.py')},
}

LOG_DIR = '/data2/NPTOI/arterial/imputation_network/logs'
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    processes = {}
    for name, cfg in MODELS.items():
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu'])

        log_file = open(os.path.join(LOG_DIR, f'{name}.log'), 'w')
        print(f"Launching {name.upper()} on GPU {cfg['gpu']}...")

        p = subprocess.Popen(
            [sys.executable, cfg['script']],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes[name] = (p, log_file)

    print(f"\nAll {len(processes)} models launched. Monitoring...")

    while processes:
        for name in list(processes.keys()):
            p, log_file = processes[name]
            retcode = p.poll()
            if retcode is not None:
                log_file.close()
                status = "SUCCESS" if retcode == 0 else f"FAILED (code={retcode})"
                print(f"  [{name.upper()}] {status}")
                del processes[name]
        if processes:
            time.sleep(30)

    print("\nAll models completed!")

if __name__ == '__main__':
    main()
