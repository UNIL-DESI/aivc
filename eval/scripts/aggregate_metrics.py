"""
Aggregate Metrics DVC pipeline runner script.

Invocable via DVC stage `aggregate_metrics`:
    python eval/scripts/aggregate_metrics.py
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from metrics.dvc_exporter import export_dvc_metrics

if __name__ == "__main__":
    export_dvc_metrics(eval_dir=EVAL_DIR)
