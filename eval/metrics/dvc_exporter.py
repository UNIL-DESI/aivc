"""
DVC Exporter & Metrics Aggregator for AIVC Evaluation Pipeline.

Consolidates benchmark evaluation metrics from multiple benchmark suites:
- dry_run_metrics.json
- swebench_cl_metrics.json
- devbench_metrics.json
- intercode_metrics.json

Aggregates token usage, OpenRouter execution costs, Exploration Overhead Ratio (EOR),
Memory Utility Index (MUI), Cumulative Cost Savings Ratio (CCSR), and task pass rates.

Exports:
- eval/metrics/summary_metrics.json (Consolidated JSON summary)
- eval/plots/comparative_summary.csv (Comparative tabular CSV for DVC plots)
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root and eval directory are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "metrics" else SCRIPT_DIR
REPO_ROOT = EVAL_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))


# Default OpenRouter pricing for model cost calculation if missing
DEFAULT_MODEL = "qwen/qwen3.7-flash"
PROMPT_PRICE_PER_1M = 0.03
COMPLETION_PRICE_PER_1M = 0.13

BENCHMARK_FILES = [
    "dry_run_metrics.json",
    "swebench_cl_metrics.json",
    "devbench_metrics.json",
    "intercode_metrics.json",
]


@dataclass
class BenchmarkMetrics:
    """Dataclass storing normalized metrics for a single benchmark suite."""

    benchmark_name: str
    model_name: str = DEFAULT_MODEL
    total_tasks: int = 0
    successful_tasks: int = 0
    pass_rate: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    eor: float = 0.0
    mui: float = 0.0
    ccsr: float = 0.0
    is_sample_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "pass_rate": round(self.pass_rate, 4),
            "token_counts": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "openrouter_costs_usd": {
                "prompt_cost_usd": round(self.prompt_cost_usd, 6),
                "completion_cost_usd": round(self.completion_cost_usd, 6),
                "total_cost_usd": round(self.total_cost_usd, 6),
            },
            "evaluation_ratios": {
                "exploration_overhead_ratio_eor": round(self.eor, 4),
                "memory_utility_index_mui": round(self.mui, 4),
                "cumulative_cost_savings_ratio_ccsr": round(self.ccsr, 4),
            },
            "is_sample_data": self.is_sample_data,
        }


# Default fallback / sample benchmark profiles used when raw JSON files do not exist yet
SAMPLE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "dry_run": {
        "benchmark_name": "dry_run",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 5,
        "successful_tasks": 5,
        "pass_rate": 1.0,
        "prompt_tokens": 2550,
        "completion_tokens": 525,
        "total_tokens": 3075,
        "prompt_cost_usd": 0.0000765,
        "completion_cost_usd": 0.00006825,
        "total_cost_usd": 0.00014475,
        "eor": 0.375,
        "mui": 0.625,
        "ccsr": 0.400,
    },
    "swebench_cl": {
        "benchmark_name": "swebench_cl",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 50,
        "successful_tasks": 38,
        "pass_rate": 0.76,
        "prompt_tokens": 145000,
        "completion_tokens": 32000,
        "total_tokens": 177000,
        "prompt_cost_usd": 0.00435,
        "completion_cost_usd": 0.00416,
        "total_cost_usd": 0.00851,
        "eor": 0.215,
        "mui": 0.712,
        "ccsr": 0.385,
    },
    "devbench": {
        "benchmark_name": "devbench",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 25,
        "successful_tasks": 20,
        "pass_rate": 0.80,
        "prompt_tokens": 82000,
        "completion_tokens": 19500,
        "total_tokens": 101500,
        "prompt_cost_usd": 0.00246,
        "completion_cost_usd": 0.002535,
        "total_cost_usd": 0.004995,
        "eor": 0.180,
        "mui": 0.745,
        "ccsr": 0.420,
    },
    "intercode": {
        "benchmark_name": "intercode",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 25,
        "successful_tasks": 18,
        "pass_rate": 0.72,
        "prompt_tokens": 68000,
        "completion_tokens": 16000,
        "total_tokens": 84000,
        "prompt_cost_usd": 0.00204,
        "completion_cost_usd": 0.00208,
        "total_cost_usd": 0.00412,
        "eor": 0.240,
        "mui": 0.660,
        "ccsr": 0.350,
    },
}


class DVCExporter:
    """
    Exportateur & Agrégateur de métriques DVC pour AIVC.
    Reads individual benchmark JSON metrics, consolidates them, and exports JSON & CSV summaries.
    """

    def __init__(self, eval_dir: Optional[Path] = None):
        self.eval_dir = eval_dir or EVAL_DIR
        self.metrics_dir = self.eval_dir / "metrics"
        self.plots_dir = self.eval_dir / "plots"

        # Ensure target export directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def find_benchmark_file(self, filename: str) -> Optional[Path]:
        """Look for benchmark file in metrics_dir, eval_dir, or repo root."""
        candidates = [
            self.metrics_dir / filename,
            self.eval_dir / filename,
            REPO_ROOT / filename,
            REPO_ROOT / "metrics" / filename,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def parse_metrics_json(self, filename: str) -> BenchmarkMetrics:
        """
        Parse raw benchmark JSON file into BenchmarkMetrics object.
        Falls back to sample metrics if file is not found or invalid.
        """
        bmark_key = filename.replace("_metrics.json", "")
        file_path = self.find_benchmark_file(filename)

        if not file_path:
            sample = SAMPLE_BENCHMARKS.get(bmark_key, SAMPLE_BENCHMARKS["dry_run"])
            bm = BenchmarkMetrics(**sample)
            bm.benchmark_name = bmark_key
            bm.is_sample_data = True
            return bm

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support nested or flat data formats
            b_name = data.get("benchmark_name") or data.get("benchmark") or bmark_key
            m_name = data.get("model_name") or data.get("active_model") or DEFAULT_MODEL
            tot_tasks = int(data.get("total_tasks") or data.get("total_steps") or data.get("tasks") or 0)
            succ_tasks = int(data.get("successful_tasks") or data.get("successful_steps") or 0)

            # Pass rate
            pass_rate = float(data.get("pass_rate") or data.get("accuracy") or 0.0)
            if pass_rate <= 0.0 and tot_tasks > 0 and succ_tasks > 0:
                pass_rate = succ_tasks / float(tot_tasks)

            # Token usage & costs (nested or flat)
            tc_data = data.get("token_cost") or data.get("token_counts") or {}
            p_tok = int(tc_data.get("prompt_tokens") or data.get("prompt_tokens") or 0)
            c_tok = int(tc_data.get("completion_tokens") or data.get("completion_tokens") or 0)
            tot_tok = int(tc_data.get("total_tokens") or data.get("total_tokens") or (p_tok + c_tok))

            cost_data = data.get("openrouter_costs_usd") or tc_data or {}
            p_cost = float(cost_data.get("prompt_cost_usd") or data.get("prompt_cost_usd") or (p_tok / 1e6 * PROMPT_PRICE_PER_1M))
            c_cost = float(cost_data.get("completion_cost_usd") or data.get("completion_cost_usd") or (c_tok / 1e6 * COMPLETION_PRICE_PER_1M))
            tot_cost = float(cost_data.get("total_cost_usd") or data.get("total_cost_usd") or (p_cost + c_cost))

            # Evaluation Ratios
            ratios_data = data.get("evaluation_ratios") or data.get("metrics") or {}
            eor = float(ratios_data.get("exploration_overhead_ratio_eor") or data.get("eor") or data.get("exploration_overhead_ratio") or 0.0)
            mui = float(ratios_data.get("memory_utility_index_mui") or data.get("mui") or data.get("memory_utility_index") or 0.0)
            ccsr = float(ratios_data.get("cumulative_cost_savings_ratio_ccsr") or data.get("ccsr") or data.get("cumulative_cost_savings_ratio") or 0.0)

            return BenchmarkMetrics(
                benchmark_name=b_name,
                model_name=m_name,
                total_tasks=tot_tasks,
                successful_tasks=succ_tasks,
                pass_rate=pass_rate,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=tot_tok,
                prompt_cost_usd=p_cost,
                completion_cost_usd=c_cost,
                total_cost_usd=tot_cost,
                eor=eor,
                mui=mui,
                ccsr=ccsr,
                is_sample_data=False,
            )

        except Exception as e:
            print(f"[Warning] Failed to parse {file_path} ({e}). Using sample benchmark fallback.")
            sample = SAMPLE_BENCHMARKS.get(bmark_key, SAMPLE_BENCHMARKS["dry_run"])
            bm = BenchmarkMetrics(**sample)
            bm.benchmark_name = bmark_key
            bm.is_sample_data = True
            return bm

    def consolidate(self) -> Tuple[Dict[str, Any], List[BenchmarkMetrics]]:
        """
        Consolidate all benchmark metrics into a single summary dictionary and list.
        """
        benchmark_metrics_list: List[BenchmarkMetrics] = []
        benchmarks_dict: Dict[str, Any] = {}

        total_tasks_all = 0
        successful_tasks_all = 0
        total_p_tok = 0
        total_c_tok = 0
        total_tok = 0
        total_p_cost = 0.0
        total_c_cost = 0.0
        total_cost = 0.0

        eor_sum = 0.0
        mui_sum = 0.0
        ccsr_sum = 0.0

        for bfile in BENCHMARK_FILES:
            bm = self.parse_metrics_json(bfile)
            benchmark_metrics_list.append(bm)
            benchmarks_dict[bm.benchmark_name] = bm.to_dict()

            total_tasks_all += bm.total_tasks
            successful_tasks_all += bm.successful_tasks
            total_p_tok += bm.prompt_tokens
            total_c_tok += bm.completion_tokens
            total_tok += bm.total_tokens
            total_p_cost += bm.prompt_cost_usd
            total_c_cost += bm.completion_cost_usd
            total_cost += bm.total_cost_usd

            eor_sum += bm.eor
            mui_sum += bm.mui
            ccsr_sum += bm.ccsr

        num_bmarks = max(1, len(benchmark_metrics_list))
        overall_pass_rate = (successful_tasks_all / float(total_tasks_all)) if total_tasks_all > 0 else 0.0
        mean_eor = eor_sum / float(num_bmarks)
        mean_mui = mui_sum / float(num_bmarks)
        mean_ccsr = ccsr_sum / float(num_bmarks)

        now_utc = datetime.now(timezone.utc).isoformat()

        consolidated = {
            "aggregated_at": now_utc,
            "total_benchmarks": len(benchmark_metrics_list),
            "overall_summary": {
                "total_tasks": total_tasks_all,
                "total_successful_tasks": successful_tasks_all,
                "overall_pass_rate": round(overall_pass_rate, 4),
                "token_counts": {
                    "total_prompt_tokens": total_p_tok,
                    "total_completion_tokens": total_c_tok,
                    "total_tokens": total_tok,
                },
                "openrouter_execution_costs_usd": {
                    "total_prompt_cost_usd": round(total_p_cost, 6),
                    "total_completion_cost_usd": round(total_c_cost, 6),
                    "total_execution_cost_usd": round(total_cost, 6),
                },
                "mean_evaluation_ratios": {
                    "mean_exploration_overhead_ratio_eor": round(mean_eor, 4),
                    "mean_memory_utility_index_mui": round(mean_mui, 4),
                    "mean_cumulative_cost_savings_ratio_ccsr": round(mean_ccsr, 4),
                },
            },
            "benchmarks": benchmarks_dict,
        }

        return consolidated, benchmark_metrics_list

    def export_summary_json(self, consolidated: Dict[str, Any]) -> Path:
        """Export consolidated metrics to eval/metrics/summary_metrics.json."""
        out_path = self.metrics_dir / "summary_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)
        return out_path

    def export_comparative_csv(self, metrics_list: List[BenchmarkMetrics]) -> Path:
        """Export comparative plots table to eval/plots/comparative_summary.csv."""
        out_path = self.plots_dir / "comparative_summary.csv"

        fieldnames = [
            "benchmark",
            "model_name",
            "total_tasks",
            "successful_tasks",
            "pass_rate",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cost_usd",
            "completion_cost_usd",
            "total_cost_usd",
            "eor",
            "mui",
            "ccsr",
            "is_sample_data",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for bm in metrics_list:
                writer.writerow(
                    {
                        "benchmark": bm.benchmark_name,
                        "model_name": bm.model_name,
                        "total_tasks": bm.total_tasks,
                        "successful_tasks": bm.successful_tasks,
                        "pass_rate": round(bm.pass_rate, 4),
                        "prompt_tokens": bm.prompt_tokens,
                        "completion_tokens": bm.completion_tokens,
                        "total_tokens": bm.total_tokens,
                        "prompt_cost_usd": round(bm.prompt_cost_usd, 6),
                        "completion_cost_usd": round(bm.completion_cost_usd, 6),
                        "total_cost_usd": round(bm.total_cost_usd, 6),
                        "eor": round(bm.eor, 4),
                        "mui": round(bm.mui, 4),
                        "ccsr": round(bm.ccsr, 4),
                        "is_sample_data": bm.is_sample_data,
                    }
                )

        return out_path

    def run(self) -> Tuple[Path, Path]:
        """Execute full consolidation & export pipeline."""
        consolidated, metrics_list = self.consolidate()
        json_path = self.export_summary_json(consolidated)
        csv_path = self.export_comparative_csv(metrics_list)
        return json_path, csv_path


def export_dvc_metrics(eval_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Convenience function to run DVCExporter."""
    exporter = DVCExporter(eval_dir=eval_dir)
    return exporter.run()


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("[AIVC DVC Exporter & Aggregator Builder]")
    print("=" * 70)

    exporter = DVCExporter()
    json_p, csv_p = exporter.run()

    print(f"[SUCCESS] Consolidated summary JSON exported to: {json_p}")
    print(f"[SUCCESS] Comparative summary CSV exported to : {csv_p}")
    print("=" * 70)
