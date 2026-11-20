"""Compatibility entry point for read-only saved-output comparison.

The original historical program is preserved in docs/archive/extraction_comparisons.
Use --reference-dir, --candidate-dir, --signal and --date. No extraction or
production output replacement is performed by this command.
"""
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent
if _project_root.name == 'tests':
    _project_root = _project_root.parent
sys.path.insert(0, str(_project_root))
from scripts.compare_reference_outputs import main


if __name__ == '__main__':
    raise SystemExit(main())
