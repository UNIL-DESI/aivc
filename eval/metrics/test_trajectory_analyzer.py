import sys
from pathlib import Path

# Add repo root and eval directory to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from metrics.trajectory_analyzer import (
    TokenCostTracker,
    TrajectoryAnalyzer,
    compute_ccsr,
    compute_eor,
    compute_mui,
)



def test_token_cost_tracker():
    tracker = TokenCostTracker(model_name="qwen/qwen3.7-flash")
    assert tracker.prompt_price_per_1m == 0.03
    assert tracker.completion_price_per_1m == 0.13

    tracker.add_usage(1_000_000, 1_000_000)
    assert tracker.total_tokens == 2_000_000
    assert abs(tracker.prompt_cost - 0.03) < 1e-6
    assert abs(tracker.completion_cost - 0.13) < 1e-6
    assert abs(tracker.total_cost - 0.16) < 1e-6


def test_compute_eor():
    assert compute_eor(10, 3) == 0.3
    assert compute_eor(0, 0) == 0.0
    assert compute_eor(5, 5) == 1.0


def test_compute_mui():
    assert compute_mui(5, 4, eor=0.2) == 0.64
    assert compute_mui(0, 0, eor=0.1) == 0.0


def test_compute_ccsr():
    assert compute_ccsr(10.0, 2.0) == 0.8
    assert compute_ccsr(0.0, 1.0) == 0.0


def test_trajectory_analyzer():
    analyzer = TrajectoryAnalyzer(model_name="qwen/qwen3.7-flash")
    trajectory = [
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "tool_calls": ["grep_search", "write_to_file"],
            "recalled_memories": 2,
            "used_memories": 2,
        },
        {
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "tool_calls": ["view_file"],
            "recalled_memories": 1,
            "used_memories": 1,
        },
    ]
    metrics = analyzer.analyze(trajectory, baseline_cost=0.01, recalled_memories_count=0, used_memories_count=0)
    assert metrics.total_steps == 2
    assert metrics.total_tool_calls == 3
    assert metrics.exploration_tool_calls == 2
    assert metrics.recalled_memories == 3
    assert metrics.used_memories == 3
    assert metrics.token_cost.total_tokens == 450
