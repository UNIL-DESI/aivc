"""
DVC Exporter & Metrics Aggregator for AIVC Evaluation Pipeline.

Consolidates benchmark evaluation metrics from multiple benchmark suites:
- swebench_cl_metrics.json
- devbench_metrics.json
- agentic_rag_metrics.json

Aggregates token usage, OpenRouter execution costs, Exploration Overhead Ratio (EOR),
Memory Utility Index (MUI), Cumulative Cost Savings Ratio (CCSR), and task pass rates.
Supports profile-partitioned metric directories (e.g. eval/metrics/dry_run, eval/metrics/production).

Exports:
- eval/metrics/summary_metrics.json (Consolidated JSON summary)
- eval/plots/comparative_summary.csv (Comparative tabular CSV for DVC plots)
"""

import argparse
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
    "swebench_cl_metrics.json",
    "devbench_metrics.json",
    "agentic_rag_metrics.json",
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
    "agentic_rag": {
        "benchmark_name": "agentic_rag",
        "model_name": "qwen/qwen3.7-flash",
        "total_tasks": 30,
        "successful_tasks": 26,
        "pass_rate": 0.8667,
        "prompt_tokens": 95000,
        "completion_tokens": 22000,
        "total_tokens": 117000,
        "prompt_cost_usd": 0.00285,
        "completion_cost_usd": 0.00286,
        "total_cost_usd": 0.00571,
        "eor": 0.150,
        "mui": 0.790,
        "ccsr": 0.450,
    },
}


class DVCExporter:
    """
    Exportateur & Agrégateur de métriques DVC pour AIVC.
    Reads individual benchmark JSON metrics across SWE-bench-CL, DevBench, and Agentic RAG,
    consolidates them, and exports JSON & CSV summaries.
    Supports partition directories by evaluation profile.
    """

    def __init__(
        self,
        eval_dir: Optional[Path] = None,
        profile: Optional[str] = None,
        metrics_dir: Optional[Path] = None,
        plots_dir: Optional[Path] = None,
    ):
        self.eval_dir = eval_dir or EVAL_DIR
        self.profile = profile
        if metrics_dir is not None:
            self.metrics_dir = metrics_dir
        elif profile:
            self.metrics_dir = self.eval_dir / "metrics" / profile
        else:
            self.metrics_dir = self.eval_dir / "metrics"

        if plots_dir is not None:
            self.plots_dir = plots_dir
        elif profile:
            self.plots_dir = self.eval_dir / "plots" / profile
        else:
            self.plots_dir = self.eval_dir / "plots"

        # Ensure target export directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        (self.eval_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.eval_dir / "plots").mkdir(parents=True, exist_ok=True)

    def find_benchmark_file(self, filename: str) -> Optional[Path]:
        """Look for benchmark file in profile metrics dir, metrics_dir, eval_dir, or repo root."""
        candidates: List[Path] = [
            self.metrics_dir / filename,
        ]
        if self.profile:
            candidates.append(self.eval_dir / "metrics" / self.profile / filename)

        candidates.extend([
            self.eval_dir / "metrics" / filename,
            self.eval_dir / filename,
            REPO_ROOT / "eval" / "metrics" / filename,
            REPO_ROOT / filename,
            REPO_ROOT / "metrics" / filename,
        ])

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # Auto-discovery across any profile partition subdirectory in eval/metrics/
        base_metrics = self.eval_dir / "metrics"
        if base_metrics.exists() and base_metrics.is_dir():
            for sub in sorted(base_metrics.iterdir()):
                if sub.is_dir():
                    cand = sub / filename
                    if cand.exists() and cand.is_file():
                        return cand

        return None

    def parse_metrics_json(self, filename: str) -> BenchmarkMetrics:
        """
        Parse raw benchmark JSON file into BenchmarkMetrics object.
        Falls back to sample metrics if file is not found or invalid.
        """
        bmark_key = filename.replace("_metrics.json", "")
        file_path = self.find_benchmark_file(filename)

        if not file_path:
            sample = SAMPLE_BENCHMARKS.get(bmark_key, SAMPLE_BENCHMARKS["agentic_rag"])
            bm = BenchmarkMetrics(**sample)
            bm.benchmark_name = bmark_key
            bm.is_sample_data = True
            return bm

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Support nested or flat data formats
            summary_block = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
            resource_block = data.get("resource_consumption", {}) if isinstance(data.get("resource_consumption"), dict) else {}

            # Standardize benchmark identifier key
            b_name = bmark_key
            m_name = data.get("model_name") or data.get("active_model") or DEFAULT_MODEL
            tot_tasks = int(
                data.get("total_tasks")
                or data.get("total_steps")
                or data.get("tasks")
                or summary_block.get("total_instances")
                or summary_block.get("total_repos")
                or summary_block.get("total_phases_executed")
                or 0
            )
            succ_tasks = int(
                data.get("successful_tasks")
                or data.get("successful_steps")
                or summary_block.get("resolved_instances")
                or summary_block.get("completed_sdlc_repos")
                or 0
            )

            # Pass rate
            pass_rate = float(
                data.get("pass_rate")
                or data.get("accuracy")
                or summary_block.get("resolve_rate_pass_at_1")
                or summary_block.get("sdlc_completion_rate")
                or summary_block.get("phase_pass_rate")
                or 0.0
            )
            if pass_rate <= 0.0 and tot_tasks > 0 and succ_tasks > 0:
                pass_rate = succ_tasks / float(tot_tasks)

            # Token usage & costs (nested or flat)
            tc_data = data.get("token_cost") or data.get("token_counts") or {}
            p_tok = int(
                tc_data.get("prompt_tokens")
                or data.get("prompt_tokens")
                or summary_block.get("total_prompt_tokens")
                or resource_block.get("prompt_tokens")
                or 0
            )
            c_tok = int(
                tc_data.get("completion_tokens")
                or data.get("completion_tokens")
                or summary_block.get("total_completion_tokens")
                or resource_block.get("completion_tokens")
                or 0
            )
            tot_tok = int(
                tc_data.get("total_tokens")
                or data.get("total_tokens")
                or summary_block.get("total_tokens")
                or resource_block.get("total_tokens")
                or (p_tok + c_tok)
            )

            cost_data = data.get("openrouter_costs_usd") or tc_data or {}
            p_cost = float(
                cost_data.get("prompt_cost_usd")
                or data.get("prompt_cost_usd")
                or (p_tok / 1e6 * PROMPT_PRICE_PER_1M)
            )
            c_cost = float(
                cost_data.get("completion_cost_usd")
                or data.get("completion_cost_usd")
                or (c_tok / 1e6 * COMPLETION_PRICE_PER_1M)
            )
            tot_cost = float(
                cost_data.get("total_cost_usd")
                or data.get("total_cost_usd")
                or summary_block.get("total_cost_usd")
                or resource_block.get("aivc_total_cost_usd")
                or (p_cost + c_cost)
            )

            # Evaluation Ratios
            ratios_data = data.get("evaluation_ratios") or data.get("metrics") or {}
            eor = float(
                ratios_data.get("exploration_overhead_ratio_eor")
                or data.get("eor")
                or data.get("exploration_overhead_ratio")
                or summary_block.get("average_exploration_overhead_ratio_eor")
                or summary_block.get("avg_eor")
                or 0.0
            )
            mui = float(
                ratios_data.get("memory_utility_index_mui")
                or data.get("mui")
                or data.get("memory_utility_index")
                or summary_block.get("average_memory_utility_index_mui")
                or summary_block.get("avg_mui")
                or 0.0
            )
            ccsr = float(
                ratios_data.get("cumulative_cost_savings_ratio_ccsr")
                or data.get("ccsr")
                or data.get("cumulative_cost_savings_ratio")
                or summary_block.get("average_cumulative_cost_savings_ratio_ccsr")
                or summary_block.get("overall_ccsr")
                or 0.0
            )

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
            sample = SAMPLE_BENCHMARKS.get(bmark_key, SAMPLE_BENCHMARKS["agentic_rag"])
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
            bmark_key = bfile.replace("_metrics.json", "")
            bm = self.parse_metrics_json(bfile)
            benchmark_metrics_list.append(bm)
            benchmarks_dict[bmark_key] = bm.to_dict()

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
            "profile": self.profile or "default",
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
        """Export consolidated metrics to summary_metrics.json."""
        out_path = self.metrics_dir / "summary_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        # Also write root summary if partition is active
        root_summary = self.eval_dir / "metrics" / "summary_metrics.json"
        if root_summary != out_path:
            with open(root_summary, "w", encoding="utf-8") as f:
                json.dump(consolidated, f, indent=2)

        return out_path

    def export_comparative_csv(self, metrics_list: List[BenchmarkMetrics]) -> Path:
        """Export comparative plots table to comparative_summary.csv."""
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

        def _write_csv(target_p: Path) -> None:
            with open(target_p, "w", newline="", encoding="utf-8") as f:
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

        _write_csv(out_path)

        # Also write root CSV if partition is active
        root_csv = self.eval_dir / "plots" / "comparative_summary.csv"
        if root_csv != out_path:
            _write_csv(root_csv)

        return out_path

    def run(self) -> Tuple[Path, Path]:
        """Execute full consolidation & export pipeline."""
        consolidated, metrics_list = self.consolidate()
        json_path = self.export_summary_json(consolidated)
        csv_path = self.export_comparative_csv(metrics_list)
        return json_path, csv_path


def export_dvc_metrics(
    eval_dir: Optional[Path] = None,
    profile: Optional[str] = None,
    metrics_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Convenience function to run DVCExporter."""
    exporter = DVCExporter(
        eval_dir=eval_dir,
        profile=profile,
        metrics_dir=metrics_dir,
        plots_dir=plots_dir,
    )
    return exporter.run()


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AIVC DVC Exporter & Aggregator")
    parser.add_argument("--profile", type=str, default=None, help="Evaluation profile partition name")
    parser.add_argument("--eval-dir", type=str, default=None, help="Custom eval base directory")
    parsed = parser.parse_args()

    eval_base = Path(parsed.eval_dir) if parsed.eval_dir else None

    print("=" * 70)
    print("[AIVC DVC Exporter & Aggregator Builder]")
    print(f"Profile: {parsed.profile or 'default'}")
    print("=" * 70)

    exporter = DVCExporter(eval_dir=eval_base, profile=parsed.profile)
    json_p, csv_p = exporter.run()

    print(f"[SUCCESS] Consolidated summary JSON exported to: {json_p}")
    print(f"[SUCCESS] Comparative summary CSV exported to : {csv_p}")
    print("=" * 70)
