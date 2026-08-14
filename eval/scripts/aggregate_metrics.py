"""
Aggregate Metrics DVC pipeline runner script.

Invocable via DVC stage `aggregate_metrics`:
    python eval/scripts/aggregate_metrics.py [--profile PROFILE]
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from metrics.dvc_exporter import export_dvc_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate AIVC Evaluation Metrics across all benchmarks")
    parser.add_argument("--profile", type=str, default=None, help="Optional profile partition name")
    parser.add_argument("--eval-dir", type=str, default=None, help="Custom eval base directory")
    args, _ = parser.parse_known_args()

    eval_base = Path(args.eval_dir) if args.eval_dir else EVAL_DIR
    export_dvc_metrics(eval_dir=eval_base, profile=args.profile)
