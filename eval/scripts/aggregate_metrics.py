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


def consolidate_tool_interactions(eval_dir: Path) -> None:
    """Consolidate per-benchmark tool interaction files into master tool_interactions.jsonl."""
    metrics_dir = eval_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_file = metrics_dir / "tool_interactions.jsonl"
    interaction_files = list(metrics_dir.glob("*_tool_interactions.jsonl"))
    if not interaction_files:
        interaction_files = [
            metrics_dir / "swebench_cl_tool_interactions.jsonl",
            metrics_dir / "devbench_tool_interactions.jsonl",
            metrics_dir / "agentic_rag_tool_interactions.jsonl",
        ]
    seen_lines = set()
    collected = []
    for f in interaction_files:
        if f.exists() and f != out_file:
            try:
                with open(f, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        s = line.strip()
                        if s and s not in seen_lines:
                            seen_lines.add(s)
                            collected.append(s + "\n")
            except Exception:
                pass
    if collected:
        try:
            with open(out_file, "w", encoding="utf-8") as out_f:
                out_f.writelines(collected)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate AIVC Evaluation Metrics across all benchmarks")
    parser.add_argument("--profile", type=str, default=None, help="Optional profile partition name")
    parser.add_argument("--eval-dir", type=str, default=None, help="Custom eval base directory")
    args, _ = parser.parse_known_args()

    eval_base = Path(args.eval_dir) if args.eval_dir else EVAL_DIR
    consolidate_tool_interactions(eval_base)
    export_dvc_metrics(eval_dir=eval_base, profile=args.profile)

