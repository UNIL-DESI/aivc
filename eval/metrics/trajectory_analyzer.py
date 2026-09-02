"""
Trajectory Analyzer & Evaluation Metrics Builder for AIVC.

This module implements quantitative metrics for benchmark evaluation:
- Exploration Overhead Ratio (EOR): Proportion of redundant/exploratory tool actions.
- Memory Utility Index (MUI): Utility and precision of recalled memory items.
- Cumulative Cost Savings Ratio (CCSR): Relative cost reduction of AIVC vs Baseline.
- TokenCostTracker: Token counting and OpenRouter API cost calculation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# Default pricing map per 1M tokens (USD)
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "qwen/qwen3.7-flash": {
        "prompt_price_per_1m": 0.03,
        "completion_price_per_1m": 0.13,
    },
    "deepseek/deepseek-v4-flash-0731": {
        "prompt_price_per_1m": 0.07,
        "completion_price_per_1m": 0.28,
    },
    "openai/gpt-5.6-luna-pro": {
        "prompt_price_per_1m": 1.50,
        "completion_price_per_1m": 6.00,
    },
    "google/gemini-3.6-flash": {
        "prompt_price_per_1m": 0.05,
        "completion_price_per_1m": 0.20,
    },
    "google/gemini-3.7-flash": {
        "prompt_price_per_1m": 0.05,
        "completion_price_per_1m": 0.20,
    },
    "anthropic/claude-sonnet-5": {
        "prompt_price_per_1m": 3.00,
        "completion_price_per_1m": 15.00,
    },
    "z-ai/glm-5.2": {
        "prompt_price_per_1m": 0.10,
        "completion_price_per_1m": 0.40,
    },
    "meta-models/Muse-Glimmer-30B": {
        "prompt_price_per_1m": 0.35,
        "completion_price_per_1m": 1.50,
        "batch_prompt_price_per_1m": 0.175,
        "batch_completion_price_per_1m": 0.75,
    },
}


EXPLORATION_TOOLS: Set[str] = {
    "grep_search",
    "list_dir",
    "view_file",
    "find_by_name",
    "search_web",
    "read_past_file_content",
}


@dataclass
class TokenCostTracker:
    """
    Tracks prompt & completion token usage and computes OpenRouter API cost.
    """
    model_name: str = "qwen/qwen3.7-flash"
    prompt_price_per_1m: float = 0.03
    completion_price_per_1m: float = 0.13
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.model_name in DEFAULT_PRICING:
            rates = DEFAULT_PRICING[self.model_name]
            self.prompt_price_per_1m = rates.get("prompt_price_per_1m", self.prompt_price_per_1m)
            self.completion_price_per_1m = rates.get("completion_price_per_1m", self.completion_price_per_1m)

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulate token counts."""
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)

    @property
    def total_tokens(self) -> int:
        """Return total token count."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def prompt_cost(self) -> float:
        """Cost for prompt tokens in USD."""
        return (self.prompt_tokens / 1_000_000.0) * self.prompt_price_per_1m

    @property
    def completion_cost(self) -> float:
        """Cost for completion tokens in USD."""
        return (self.completion_tokens / 1_000_000.0) * self.completion_price_per_1m

    @property
    def total_cost(self) -> float:
        """Total calculated cost in USD."""
        return self.prompt_cost + self.completion_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost_usd": round(self.prompt_cost, 6),
            "completion_cost_usd": round(self.completion_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
        }


def compute_eor(
    total_tool_calls: int,
    exploration_tool_calls: int,
) -> float:
    """
    Compute Exploration Overhead Ratio (EOR).
    EOR = Exploration Tool Calls / Total Tool Calls
    Returns 0.0 if total_tool_calls is 0.
    """
    if total_tool_calls <= 0:
        return 0.0
    return round(min(1.0, max(0.0, exploration_tool_calls / float(total_tool_calls))), 4)


def compute_mui(
    recalled_memories_count: int,
    used_memories_count: int,
    eor: float = 0.0,
) -> float:
    """
    Compute Memory Utility Index (MUI).
    MUI measures the utility and precision of recalled memory items.
    MUI = (Used Memories / Recalled Memories) * (1 - EOR)
    Returns 0.0 if recalled_memories_count is 0.
    """
    if recalled_memories_count <= 0:
        return 0.0
    recall_precision = min(1.0, max(0.0, used_memories_count / float(recalled_memories_count)))
    efficiency_factor = max(0.0, 1.0 - eor)
    return round(recall_precision * efficiency_factor, 4)


def compute_ccsr(
    baseline_cost: float,
    aivc_cost: float,
) -> float:
    """
    Compute Cumulative Cost Savings Ratio (CCSR).
    CCSR = (Baseline Cost - AIVC Cost) / Baseline Cost
    Returns 0.0 if baseline_cost <= 0.
    """
    if baseline_cost <= 0.0:
        return 0.0
    savings = (baseline_cost - aivc_cost) / float(baseline_cost)
    return round(savings, 4)


def extract_files_from_patch(patch_str: str) -> List[str]:
    """Extract modified/inspected file paths from git patch text."""
    if not patch_str:
        return []
    files: List[str] = []
    for line in patch_str.splitlines():
        line = line.strip()
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                f_path = parts[3].lstrip("b/").lstrip("a/")
                if f_path and f_path not in files:
                    files.append(f_path)
        elif line.startswith("--- a/") or line.startswith("+++ b/"):
            f_path = line[6:].strip()
            if f_path and f_path not in files and f_path != "/dev/null":
                files.append(f_path)
    return files


def compute_ndcg_at_k(
    retrieved_files: List[str],
    ground_truth_files: List[str],
    k: int = 5,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k (NDCG@k).
    Binary relevance: 1 if file in ground_truth_files, 0 otherwise.
    """
    import math

    def _norm(p: str) -> str:
        return p.strip().replace("\\", "/").lower().lstrip("./")

    norm_gt = set(_norm(f) for f in ground_truth_files if f.strip())
    if not norm_gt or k <= 0:
        return 0.0

    norm_retrieved = [_norm(f) for f in retrieved_files if f.strip()][:k]

    dcg = 0.0
    for i, rf in enumerate(norm_retrieved):
        rel = 1.0 if any(rf == gf or rf.endswith(gf) or gf.endswith(rf) for gf in norm_gt) else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_hits = min(k, len(norm_gt))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg <= 0.0:
        return 0.0
    return round(min(1.0, dcg / idcg), 4)


def compute_retrieval_metrics(
    retrieved_files: List[Any],
    ground_truth_files: List[Any],
    k_list: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Compute Precision@k, Recall@k, F1@k, NDCG@k, and MRR (Mean Reciprocal Rank).
    Normalizes file path comparisons (case-insensitive, forward slashes).
    """
    def _flatten_files(items: Any) -> List[str]:
        flat: List[str] = []
        if items is None:
            return flat
        if isinstance(items, (str, bytes)):
            s = items.decode("utf-8", errors="replace") if isinstance(items, bytes) else items
            s = s.strip()
            if s:
                flat.append(s)
        elif isinstance(items, (list, tuple, set)):
            for it in items:
                flat.extend(_flatten_files(it))
        elif isinstance(items, dict):
            for v in items.values():
                flat.extend(_flatten_files(v))
        return flat

    def _norm(p: str) -> str:
        return p.strip().replace("\\", "/").lower().lstrip("./")

    clean_gt = _flatten_files(ground_truth_files)
    clean_retrieved = _flatten_files(retrieved_files)

    norm_gt = set(_norm(f) for f in clean_gt if f.strip())
    norm_retrieved = [_norm(f) for f in clean_retrieved if f.strip()]

    metrics: Dict[str, float] = {}

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, rf in enumerate(norm_retrieved, 1):
        if any(rf == gf or rf.endswith(gf) or gf.endswith(rf) for gf in norm_gt):
            mrr = 1.0 / rank
            break
    metrics["mrr"] = round(mrr, 4)

    # Precision@k, Recall@k, F1@k, NDCG@k
    for k in k_list:
        top_k = norm_retrieved[:k]
        hits = 0
        for rf in top_k:
            if any(rf == gf or rf.endswith(gf) or gf.endswith(rf) for gf in norm_gt):
                hits += 1

        prec_k = hits / float(k) if k > 0 else 0.0
        rec_k = hits / float(len(norm_gt)) if norm_gt else (1.0 if hits > 0 else 0.0)
        f1_k = (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0
        ndcg_k = compute_ndcg_at_k(norm_retrieved, list(norm_gt), k=k)

        metrics[f"precision_at_{k}"] = round(min(1.0, prec_k), 4)
        metrics[f"recall_at_{k}"] = round(min(1.0, rec_k), 4)
        metrics[f"f1_at_{k}"] = round(min(1.0, f1_k), 4)
        metrics[f"ndcg_at_{k}"] = ndcg_k

    return metrics


@dataclass
class TrajectoryMetrics:
    total_steps: int = 0
    total_tool_calls: int = 0
    exploration_tool_calls: int = 0
    recalled_memories: int = 0
    used_memories: int = 0
    eor: float = 0.0
    mui: float = 0.0
    ccsr: float = 0.0
    tool_counts: Dict[str, int] = field(default_factory=dict)
    token_cost: Optional[TokenCostTracker] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "exploration_tool_calls": self.exploration_tool_calls,
            "recalled_memories": self.recalled_memories,
            "used_memories": self.used_memories,
            "exploration_overhead_ratio_eor": self.eor,
            "memory_utility_index_mui": self.mui,
            "cumulative_cost_savings_ratio_ccsr": self.ccsr,
            "tool_counts": self.tool_counts,
        }
        if self.token_cost:
            res["token_cost"] = self.token_cost.to_dict()
        return res


class TrajectoryAnalyzer:
    """
    Analyzes execution trajectories of LLM agent runs and calculates evaluation metrics.
    """

    def __init__(self, model_name: str = "qwen/qwen3.7-flash"):
        self.tracker = TokenCostTracker(model_name=model_name)

    def analyze(
        self,
        trajectory: List[Dict[str, Any]],
        baseline_cost: float = 0.0,
        recalled_memories_count: int = 0,
        used_memories_count: int = 0,
    ) -> TrajectoryMetrics:
        """
        Analyze trajectory step dictionaries.
        Each step dict can contain:
        - 'tool_calls': List[Dict[str, str]] or List[str]
        - 'prompt_tokens': int
        - 'completion_tokens': int
        - 'recalled_memories': int
        - 'used_memories': int
        """
        total_steps = len(trajectory)
        total_tool_calls = 0
        exploration_tool_calls = 0
        tool_counts: Dict[str, int] = {}

        step_recalled = recalled_memories_count
        step_used = used_memories_count

        for step in trajectory:
            # Token counting
            p_tok = step.get("prompt_tokens", 0)
            c_tok = step.get("completion_tokens", 0)
            self.tracker.add_usage(p_tok, c_tok)

            # Memory counters if present in step
            if "recalled_memories" in step:
                step_recalled += step["recalled_memories"]
            if "used_memories" in step:
                step_used += step["used_memories"]

            # Tool call extraction
            tool_calls = step.get("tool_calls", [])
            for tc in tool_calls:
                total_tool_calls += 1
                if isinstance(tc, str):
                    tool_name = tc
                elif isinstance(tc, dict):
                    tool_name = tc.get("name", "")
                else:
                    tool_name = ""
                if tool_name:
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                if tool_name in EXPLORATION_TOOLS:
                    exploration_tool_calls += 1

        eor = compute_eor(total_tool_calls, exploration_tool_calls)
        mui = compute_mui(step_recalled, step_used, eor)
        ccsr = compute_ccsr(baseline_cost, self.tracker.total_cost)

        return TrajectoryMetrics(
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            exploration_tool_calls=exploration_tool_calls,
            recalled_memories=step_recalled,
            used_memories=step_used,
            eor=eor,
            mui=mui,
            ccsr=ccsr,
            tool_counts=tool_counts,
            token_cost=self.tracker,
        )
