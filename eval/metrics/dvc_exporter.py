"""
DVC Exporter & Metrics Aggregator for AIVC Matrix Evaluation Pipeline.

Consolidates benchmark evaluation metrics from all 18 matrix combinations:
- 3 Benchmarks: SWE-bench-CL, DevBench, Agentic RAG
- 3 Models: google/gemini-3.7-flash, deepseek/deepseek-v4-pro, meta-models/Muse-Glimmer-30B
- 2 Arms: aivc, baseline

Aggregates token usage, OpenRouter/Together execution costs, Exploration Overhead Ratio (EOR),
Memory Utility Index (MUI), Cumulative Cost Savings Ratio (CCSR), NDCG, MRR, Tool Call Decay,
and comparative deltas (AIVC vs. Baseline).

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


DEFAULT_MODEL = "google/gemini-3.7-flash"
PROMPT_PRICE_PER_1M = 0.05
COMPLETION_PRICE_PER_1M = 0.20

STANDARD_BENCHMARKS = [
    "swebench_cl",
    "devbench",
    "agentic_rag",
]

BENCHMARK_FILES = [
    "dry_run_metrics.json",
    "swebench_cl_metrics.json",
    "devbench_metrics.json",
    "agentic_rag_metrics.json",
]


@dataclass
class BenchmarkMetrics:
    """Dataclass storing normalized metrics for a single benchmark matrix run."""

    benchmark_name: str
    model_name: str = DEFAULT_MODEL
    arm: str = "aivc"
    total_tasks: int = 0
    successful_tasks: int = 0
    pass_rate: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    baseline_estimated_cost_usd: float = 0.0
    eor: float = 0.0
    mui: float = 0.0
    ccsr: float = 0.0
    tool_call_decay: float = 0.0
    ndcg_at_1: float = 0.0
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    total_tool_calls: int = 0
    total_tool_interactions: int = 0
    is_sample_data: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "arm": self.arm,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "pass_rate": round(self.pass_rate, 4),
            "token_counts": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "costs_usd": {
                "prompt_cost_usd": round(self.prompt_cost_usd, 6),
                "completion_cost_usd": round(self.completion_cost_usd, 6),
                "total_cost_usd": round(self.total_cost_usd, 6),
                "baseline_estimated_cost_usd": round(self.baseline_estimated_cost_usd, 6),
            },
            "evaluation_ratios": {
                "exploration_overhead_ratio_eor": round(self.eor, 4),
                "memory_utility_index_mui": round(self.mui, 4),
                "cumulative_cost_savings_ratio_ccsr": round(self.ccsr, 4),
                "tool_call_decay": round(self.tool_call_decay, 4),
            },
            "retrieval_metrics": {
                "mean_reciprocal_rank_mrr": round(self.mrr, 4),
                "precision_at_1": round(self.precision_at_1, 4),
                "precision_at_3": round(self.precision_at_3, 4),
                "precision_at_5": round(self.precision_at_5, 4),
                "recall_at_1": round(self.recall_at_1, 4),
                "recall_at_3": round(self.recall_at_3, 4),
                "recall_at_5": round(self.recall_at_5, 4),
                "ndcg_at_1": round(self.ndcg_at_1, 4),
                "ndcg_at_3": round(self.ndcg_at_3, 4),
                "ndcg_at_5": round(self.ndcg_at_5, 4),
            },
            "tool_telemetry": {
                "total_tool_calls": self.total_tool_calls,
                "total_tool_interactions": self.total_tool_interactions,
            },
            "is_sample_data": self.is_sample_data,
        }


# Default fallback / sample benchmark profiles used when raw JSON files do not exist yet
SAMPLE_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "dry_run": {
        "benchmark_name": "dry_run",
        "model_name": DEFAULT_MODEL,
        "arm": "aivc",
        "total_tasks": 5,
        "successful_tasks": 5,
        "pass_rate": 1.0,
        "prompt_tokens": 2550,
        "completion_tokens": 525,
        "total_tokens": 3075,
        "prompt_cost_usd": 0.0001275,
        "completion_cost_usd": 0.000105,
        "total_cost_usd": 0.0002325,
        "eor": 0.375,
        "mui": 0.625,
        "ccsr": 0.400,
        "ndcg_at_5": 0.850,
        "mrr": 0.880,
    },
    "swebench_cl": {
        "benchmark_name": "swebench_cl",
        "model_name": DEFAULT_MODEL,
        "arm": "aivc",
        "total_tasks": 15,
        "successful_tasks": 12,
        "pass_rate": 0.80,
        "prompt_tokens": 45000,
        "completion_tokens": 12000,
        "total_tokens": 57000,
        "prompt_cost_usd": 0.00225,
        "completion_cost_usd": 0.00240,
        "total_cost_usd": 0.00465,
        "eor": 0.215,
        "mui": 0.712,
        "ccsr": 0.385,
        "ndcg_at_5": 0.785,
        "mrr": 0.820,
    },
    "devbench": {
        "benchmark_name": "devbench",
        "model_name": DEFAULT_MODEL,
        "arm": "aivc",
        "total_tasks": 15,
        "successful_tasks": 12,
        "pass_rate": 0.80,
        "prompt_tokens": 42000,
        "completion_tokens": 11500,
        "total_tokens": 53500,
        "prompt_cost_usd": 0.00210,
        "completion_cost_usd": 0.00230,
        "total_cost_usd": 0.00440,
        "eor": 0.180,
        "mui": 0.745,
        "ccsr": 0.420,
    },
    "agentic_rag": {
        "benchmark_name": "agentic_rag",
        "model_name": DEFAULT_MODEL,
        "arm": "aivc",
        "total_tasks": 15,
        "successful_tasks": 15,
        "pass_rate": 1.0,
        "prompt_tokens": 48000,
        "completion_tokens": 10000,
        "total_tokens": 58000,
        "prompt_cost_usd": 0.00240,
        "completion_cost_usd": 0.00200,
        "total_cost_usd": 0.00440,
        "eor": 0.120,
        "mui": 0.880,
        "ccsr": 0.450,
        "ndcg_at_5": 0.890,
        "mrr": 0.910,
    },
}


class DVCExporter:
    """
    Exportateur & Agrégateur de métriques DVC pour la matrice d'évaluation AIVC.
    Discovers individual benchmark JSON metrics, consolidates the full matrix,
    calculates comparative deltas, and exports summary JSON and comparative CSV tables.
    """

    def __init__(self, eval_dir: Optional[Path] = None):
        self.eval_dir = eval_dir or EVAL_DIR
        self.metrics_dir = self.eval_dir / "metrics"
        self.plots_dir = self.eval_dir / "plots"

        # Ensure target export directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def discover_metric_files(self) -> List[Path]:
        """Dynamically find all matrix metric JSON files in metrics_dir."""
        all_json = list(self.metrics_dir.glob("*_metrics.json"))
        # Exclude aggregate summary file
        matrix_files = [p for p in all_json if p.name != "summary_metrics.json"]

        # If no dynamic files found, fall back to standard benchmark files
        if not matrix_files:
            for bfile in BENCHMARK_FILES:
                c_path = self.metrics_dir / bfile
                if c_path.exists():
                    matrix_files.append(c_path)

        return sorted(matrix_files, key=lambda p: p.name)

    def parse_metrics_file(self, file_path: Path) -> BenchmarkMetrics:
        """Parse a single JSON metrics file into BenchmarkMetrics."""
        bmark_key = file_path.stem.replace("_metrics", "")
        filename_lower = file_path.name.lower()

        # Infer arm from filename if not specified inside JSON
        inferred_arm = "baseline" if ("baseline" in filename_lower or "naive" in filename_lower) else "aivc"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary_block = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
            resource_block = data.get("resource_consumption", {}) if isinstance(data.get("resource_consumption"), dict) else {}

            b_name = data.get("benchmark_name") or data.get("benchmark") or bmark_key
            # Clean benchmark name if it has model/arm suffix
            if "swebench_cl" in bmark_key:
                b_name = "swebench_cl"
            elif "devbench" in bmark_key:
                b_name = "devbench"
            elif "agentic_rag" in bmark_key:
                b_name = "agentic_rag"

            m_name = data.get("model_name") or data.get("active_model") or DEFAULT_MODEL
            arm = data.get("arm") or summary_block.get("arm") or inferred_arm
            if arm in ("naive", "baseline"):
                arm = "baseline"

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

            cost_data = data.get("openrouter_costs_usd") or data.get("costs_usd") or tc_data or {}
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
            base_est_cost = float(
                cost_data.get("baseline_estimated_cost_usd")
                or data.get("baseline_estimated_cost_usd")
                or summary_block.get("baseline_cost_usd")
                or resource_block.get("baseline_estimated_cost_usd")
                or 0.0
            )

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
            decay = float(
                ratios_data.get("tool_call_decay")
                or data.get("tool_call_decay")
                or summary_block.get("tool_call_decay")
                or 0.0
            )

            ret_data = data.get("retrieval_metrics") or {}
            ndcg1 = float(ret_data.get("ndcg_at_1") or data.get("ndcg_at_1") or 0.0)
            ndcg3 = float(ret_data.get("ndcg_at_3") or data.get("ndcg_at_3") or 0.0)
            ndcg5 = float(ret_data.get("ndcg_at_5") or data.get("ndcg_at_5") or 0.0)
            p1 = float(ret_data.get("precision_at_1") or data.get("precision_at_1") or 0.0)
            p3 = float(ret_data.get("precision_at_3") or data.get("precision_at_3") or 0.0)
            p5 = float(ret_data.get("precision_at_5") or data.get("precision_at_5") or 0.0)
            r1 = float(ret_data.get("recall_at_1") or data.get("recall_at_1") or 0.0)
            r3 = float(ret_data.get("recall_at_3") or data.get("recall_at_3") or 0.0)
            r5 = float(ret_data.get("recall_at_5") or data.get("recall_at_5") or 0.0)
            mrr = float(ret_data.get("mean_reciprocal_rank_mrr") or data.get("mrr") or 0.0)

            tool_telem = data.get("tool_telemetry") or {}
            tot_tool_calls = int(
                tool_telem.get("total_tool_calls")
                or summary_block.get("total_tool_calls")
                or data.get("total_tool_calls")
                or 0
            )
            tot_tool_interactions = int(
                tool_telem.get("total_tool_interactions")
                or summary_block.get("total_tool_interactions")
                or data.get("total_tool_interactions")
                or 0
            )

            return BenchmarkMetrics(
                benchmark_name=b_name,
                model_name=m_name,
                arm=arm,
                total_tasks=tot_tasks,
                successful_tasks=succ_tasks,
                pass_rate=pass_rate,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=tot_tok,
                prompt_cost_usd=p_cost,
                completion_cost_usd=c_cost,
                total_cost_usd=tot_cost,
                baseline_estimated_cost_usd=base_est_cost,
                eor=eor,
                mui=mui,
                ccsr=ccsr,
                tool_call_decay=decay,
                ndcg_at_1=ndcg1,
                ndcg_at_3=ndcg3,
                ndcg_at_5=ndcg5,
                precision_at_1=p1,
                precision_at_3=p3,
                precision_at_5=p5,
                recall_at_1=r1,
                recall_at_3=r3,
                recall_at_5=r5,
                mrr=mrr,
                total_tool_calls=tot_tool_calls,
                total_tool_interactions=tot_tool_interactions,
                is_sample_data=False,
            )

        except Exception as e:
            print(f"[Warning] Failed to parse {file_path} ({e}). Using sample fallback.")
            sample = SAMPLE_BENCHMARKS.get(bmark_key, SAMPLE_BENCHMARKS["dry_run"])
            bm = BenchmarkMetrics(**sample)
            bm.benchmark_name = bmark_key
            bm.is_sample_data = True
            return bm

    def consolidate(self) -> Tuple[Dict[str, Any], List[BenchmarkMetrics]]:
        """Consolidate all matrix benchmark metric files."""
        metric_files = self.discover_metric_files()
        metrics_list: List[BenchmarkMetrics] = []

        if metric_files:
            for mf in metric_files:
                metrics_list.append(self.parse_metrics_file(mf))
        else:
            for key, val in SAMPLE_BENCHMARKS.items():
                bm = BenchmarkMetrics(**val)
                bm.is_sample_data = True
                metrics_list.append(bm)

        # Compute comparative pairing: (benchmark, model) -> {aivc: bm, baseline: bm}
        pairings: Dict[Tuple[str, str], Dict[str, BenchmarkMetrics]] = {}
        for bm in metrics_list:
            pair_key = (bm.benchmark_name, bm.model_name)
            if pair_key not in pairings:
                pairings[pair_key] = {}
            pairings[pair_key][bm.arm] = bm

        comparative_analysis: List[Dict[str, Any]] = []
        for (bench, model), arms in pairings.items():
            aivc_bm = arms.get("aivc")
            base_bm = arms.get("baseline")

            if aivc_bm and base_bm:
                delta_ndcg5 = round(aivc_bm.ndcg_at_5 - base_bm.ndcg_at_5, 4)
                delta_mrr = round(aivc_bm.mrr - base_bm.mrr, 4)
                delta_pass_rate = round(aivc_bm.pass_rate - base_bm.pass_rate, 4)
                cost_savings_pct = (
                    round((1.0 - (aivc_bm.total_cost_usd / base_bm.total_cost_usd)) * 100.0, 2)
                    if base_bm.total_cost_usd > 0
                    else round(aivc_bm.ccsr * 100.0, 2)
                )
                delta_eor = round(base_bm.eor - aivc_bm.eor, 4)  # positive means AIVC reduced overhead

                comparative_analysis.append({
                    "benchmark": bench,
                    "model_name": model,
                    "aivc_pass_rate": aivc_bm.pass_rate,
                    "baseline_pass_rate": base_bm.pass_rate,
                    "delta_pass_rate": delta_pass_rate,
                    "aivc_ndcg_at_5": aivc_bm.ndcg_at_5,
                    "baseline_ndcg_at_5": base_bm.ndcg_at_5,
                    "delta_ndcg_at_5": delta_ndcg5,
                    "aivc_mrr": aivc_bm.mrr,
                    "baseline_mrr": base_bm.mrr,
                    "delta_mrr": delta_mrr,
                    "aivc_cost_usd": aivc_bm.total_cost_usd,
                    "baseline_cost_usd": base_bm.total_cost_usd,
                    "cost_savings_pct": cost_savings_pct,
                    "eor_reduction": delta_eor,
                    "aivc_mui": aivc_bm.mui,
                    "aivc_ccsr": aivc_bm.ccsr,
                })

        # Overall summary stats
        total_tasks_all = sum(bm.total_tasks for bm in metrics_list)
        succ_tasks_all = sum(bm.successful_tasks for bm in metrics_list)
        total_tok = sum(bm.total_tokens for bm in metrics_list)
        total_cost = sum(bm.total_cost_usd for bm in metrics_list)

        num_bmarks = max(1, len(metrics_list))
        mean_pass_rate = succ_tasks_all / float(total_tasks_all) if total_tasks_all > 0 else 0.0
        mean_eor = sum(bm.eor for bm in metrics_list) / float(num_bmarks)
        mean_mui = sum(bm.mui for bm in metrics_list) / float(num_bmarks)
        mean_ccsr = sum(bm.ccsr for bm in metrics_list) / float(num_bmarks)
        mean_ndcg5 = sum(bm.ndcg_at_5 for bm in metrics_list) / float(num_bmarks)
        mean_mrr = sum(bm.mrr for bm in metrics_list) / float(num_bmarks)

        now_utc = datetime.now(timezone.utc).isoformat()

        consolidated = {
            "aggregated_at": now_utc,
            "total_matrix_runs": len(metrics_list),
            "overall_summary": {
                "total_tasks": total_tasks_all,
                "total_successful_tasks": succ_tasks_all,
                "overall_pass_rate": round(mean_pass_rate, 4),
                "total_tokens": total_tok,
                "total_cost_usd": round(total_cost, 6),
                "mean_eor": round(mean_eor, 4),
                "mean_mui": round(mean_mui, 4),
                "mean_ccsr": round(mean_ccsr, 4),
                "mean_ndcg_at_5": round(mean_ndcg5, 4),
                "mean_mrr": round(mean_mrr, 4),
            },
            "comparative_analysis": comparative_analysis,
            "matrix_runs": [bm.to_dict() for bm in metrics_list],
        }

        return consolidated, metrics_list

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
            "arm",
            "total_tasks",
            "successful_tasks",
            "pass_rate",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cost_usd",
            "completion_cost_usd",
            "total_cost_usd",
            "baseline_estimated_cost_usd",
            "eor",
            "mui",
            "ccsr",
            "tool_call_decay",
            "ndcg_at_1",
            "ndcg_at_3",
            "ndcg_at_5",
            "precision_at_3",
            "recall_at_3",
            "mrr",
            "total_tool_calls",
            "total_tool_interactions",
            "is_sample_data",
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for bm in metrics_list:
                writer.writerow({
                    "benchmark": bm.benchmark_name,
                    "model_name": bm.model_name,
                    "arm": bm.arm,
                    "total_tasks": bm.total_tasks,
                    "successful_tasks": bm.successful_tasks,
                    "pass_rate": round(bm.pass_rate, 4),
                    "prompt_tokens": bm.prompt_tokens,
                    "completion_tokens": bm.completion_tokens,
                    "total_tokens": bm.total_tokens,
                    "prompt_cost_usd": round(bm.prompt_cost_usd, 6),
                    "completion_cost_usd": round(bm.completion_cost_usd, 6),
                    "total_cost_usd": round(bm.total_cost_usd, 6),
                    "baseline_estimated_cost_usd": round(bm.baseline_estimated_cost_usd, 6),
                    "eor": round(bm.eor, 4),
                    "mui": round(bm.mui, 4),
                    "ccsr": round(bm.ccsr, 4),
                    "tool_call_decay": round(bm.tool_call_decay, 4),
                    "ndcg_at_1": round(bm.ndcg_at_1, 4),
                    "ndcg_at_3": round(bm.ndcg_at_3, 4),
                    "ndcg_at_5": round(bm.ndcg_at_5, 4),
                    "precision_at_3": round(bm.precision_at_3, 4),
                    "recall_at_3": round(bm.recall_at_3, 4),
                    "mrr": round(bm.mrr, 4),
                    "total_tool_calls": bm.total_tool_calls,
                    "total_tool_interactions": bm.total_tool_interactions,
                    "is_sample_data": bm.is_sample_data,
                })

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
    **kwargs: Any,
) -> Tuple[Path, Path]:
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
