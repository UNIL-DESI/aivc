"""
Unit tests for eval/metrics/dvc_exporter.py
"""

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from metrics.dvc_exporter import BenchmarkMetrics, DVCExporter, export_dvc_metrics


def test_dvc_exporter_dirs_created(tmp_path):
    eval_tmp = tmp_path / "eval"
    exporter = DVCExporter(eval_dir=eval_tmp)
    assert (eval_tmp / "metrics").exists()
    assert (eval_tmp / "plots").exists()


def test_dvc_exporter_consolidation_and_export(tmp_path):
    eval_tmp = tmp_path / "eval"
    metrics_dir = eval_tmp / "metrics"
    plots_dir = eval_tmp / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy dry_run_metrics.json
    dry_run_data = {
        "benchmark_name": "dry_run",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 5,
        "successful_tasks": 5,
        "pass_rate": 1.0,
        "token_cost": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_cost_usd": 0.00003,
            "completion_cost_usd": 0.000026,
            "total_cost_usd": 0.000056,
        },
        "metrics": {
            "exploration_overhead_ratio_eor": 0.25,
            "memory_utility_index_mui": 0.75,
            "cumulative_cost_savings_ratio_ccsr": 0.40,
        },
    }
    with open(metrics_dir / "dry_run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dry_run_data, f)

    exporter = DVCExporter(eval_dir=eval_tmp)
    json_path, csv_path = exporter.run()

    assert json_path.exists()
    assert csv_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert "overall_summary" in summary_data
    assert summary_data["total_benchmarks"] == 3
    assert summary_data["benchmarks"]["dry_run"]["total_tasks"] == 5
    assert summary_data["benchmarks"]["dry_run"]["pass_rate"] == 1.0

    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_content = f.read()

    assert "dry_run" in csv_content
    assert "swebench_cl" in csv_content
    assert "devbench" in csv_content
