"""Checked CLI for ATSPM event data organized by signal and source date.

Use ``python -m feature_extractor.process_local_atspm --help`` for options.
All execution modes delegate to checked_runner and its shared run ledger.
"""
import logging
import os
import sys
import warnings

import pandas as pd

# Support direct script execution as well as installed entry points.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Preserve the CLI warning policy.
warnings.simplefilter('ignore', FutureWarning)  # Suppress all FutureWarnings including pandas
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)  # DataFrame fragmentation
warnings.filterwarnings('ignore', message='.*DataFrame concatenation.*')
warnings.filterwarnings('ignore', message='.*behavior of DataFrame concatenation.*')
warnings.filterwarnings('ignore', message='.*empty or all-NA entries.*')
warnings.filterwarnings('ignore', message='.*highly fragmented.*')

from feature_extractor.config import ConfigParser


def main():
    """All CLI modes use the same validated execution and completion contract."""
    from feature_extractor.checked_runner import run_checked
    try:
        args = ConfigParser.parse_args()
        return run_checked(ConfigParser.create_config_from_args(args))
    except BlockingIOError:
        print('Another ATSPM pipeline owns the shared lock.', file=sys.stderr)
        return 75
    except Exception as exc:
        logging.exception('Pipeline failed')
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
