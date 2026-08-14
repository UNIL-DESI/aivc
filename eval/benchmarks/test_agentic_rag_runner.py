"""
Unit tests for Agentic RAG Continual Learning Benchmark Runner.
"""

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent

for p in [str(REPO_ROOT), str(EVAL_DIR), str(BENCHMARK_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config_loader import (
    get_model_pricing,
    load_env_file,
    load_models_config,
    load_params_yaml,
    resolve_benchmark_paths,
)
from aivc_prompt_template import (
    AIVC_AGENTIC_RAG_SYSTEM_PROMPT,
    AIVC_RAG_TOOLS_SCHEMA,
    NAIVE_AGENTIC_RAG_SYSTEM_PROMPT,
    NAIVE_RAG_TOOLS_SCHEMA,
    format_agentic_rag_prompt,
)
from benchmarks.agentic_rag_runner import (
    AIVCContinualEnvironment,
    AgenticRAGCheckpointManager,
    AgenticRAGRunner,
    compute_retrieval_metrics,
    export_agentic_rag_curves,
    export_agentic_rag_metrics,
    load_agentic_rag_dataset,
)


def test_config_loader():
    params = load_params_yaml()
    assert "eval" in params
    assert params["eval"]["limit"] >= 1

    models_cfg = load_models_config()
    assert "models" in models_cfg

    p_price, c_price, ctx = get_model_pricing("qwen/qwen3.7-flash", models_cfg)
    assert p_price == 0.03
    assert c_price == 0.13
    assert ctx > 0


def test_prompt_template_and_schemas():
    assert "AIVC" in AIVC_AGENTIC_RAG_SYSTEM_PROMPT
    assert "Stateless" in NAIVE_AGENTIC_RAG_SYSTEM_PROMPT

    aivc_tool_names = [t["function"]["name"] for t in AIVC_RAG_TOOLS_SCHEMA]
    assert "remember" in aivc_tool_names
    assert "recall" in aivc_tool_names
    assert "submit_answer" in aivc_tool_names

    naive_tool_names = [t["function"]["name"] for t in NAIVE_RAG_TOOLS_SCHEMA]
    assert "remember" not in naive_tool_names
    assert "grep_search" in naive_tool_names
    assert "submit_answer" in naive_tool_names

    user_prompt = format_agentic_rag_prompt(
        query_item={"query_id": "Q-TEST", "repo": "test/repo", "query": "Find the handler"},
        arm="aivc",
    )
    assert "Q-TEST" in user_prompt
    assert "AIVC Instruction" in user_prompt


def test_aivc_continual_environment():
    env = AIVCContinualEnvironment(arm="aivc")
    assert len(env.memories) == 0

    # Test remember
    rem_msg = env.remember(
        title="Auth Handlers",
        note="Discovered AuthenticationMiddleware and hashers.",
        read_files=["auth/middleware.py"],
        edited_files=["auth/hashers.py"],
    )
    assert "mem-0001" in rem_msg
    assert len(env.memories) == 1
    assert "auth/hashers.py" in env.file_snapshots

    # Test recall
    recall_res = env.recall("AuthenticationMiddleware")
    assert "mem-0001" in recall_res
    assert "Auth Handlers" in recall_res

    # Test consult memory
    consult_res = env.consult_memory("mem-0001")
    assert "Auth Handlers" in consult_res

    # Test file history metadata
    hist_res = env.get_file_history_metadata("auth/hashers.py")
    assert "mem-0001" in hist_res

    # Test stateless arm behavior
    naive_env = AIVCContinualEnvironment(arm="naive")
    assert "disabled" in naive_env.execute_tool("remember", {"title": "X", "note": "Y"}, {})


def test_retrieval_metrics():
    retrieved = ["django/contrib/auth/middleware.py", "django/contrib/sessions/middleware.py", "django/views/base.py"]
    ground_truth = ["django/contrib/auth/middleware.py", "django/contrib/sessions/middleware.py", "django/contrib/auth/hashers.py"]

    metrics = compute_retrieval_metrics(retrieved, ground_truth, k_list=(1, 3, 5))
    assert metrics["mrr"] == 1.0
    assert metrics["precision_at_1"] == 1.0
    assert metrics["precision_at_3"] == 0.6667
    assert metrics["recall_at_3"] == 0.6667


def test_checkpoint_manager(tmp_path):
    ckpt_file = tmp_path / "test_rag_checkpoint.jsonl"
    mgr = AgenticRAGCheckpointManager(ckpt_file)
    assert len(mgr.processed_ids) == 0

    record = {
        "episode_index": 1,
        "query_id": "RAG-TEST-01",
        "resolved": True,
        "status": "resolved",
    }
    mgr.save_episode(record)
    assert mgr.is_processed("RAG-TEST-01")

    # Reload
    mgr2 = AgenticRAGCheckpointManager(ckpt_file)
    assert mgr2.is_processed("RAG-TEST-01")
    records = mgr2.load_all_records()
    assert len(records) == 1
    assert records[0]["query_id"] == "RAG-TEST-01"


def test_agentic_rag_runner_dry_run(tmp_path):
    ckpt_file = tmp_path / "rag_ckpt.jsonl"
    metrics_file = tmp_path / "rag_metrics.json"
    curves_file = tmp_path / "rag_curves.csv"

    dataset, _ = load_agentic_rag_dataset(limit=2)
    assert len(dataset) == 2

    runner = AgenticRAGRunner(
        arm="aivc",
        model_name="qwen/qwen3.7-flash",
        dry_run=True,
    )

    records = []
    for idx, q in enumerate(dataset, 1):
        ep_rec = runner.run_episode(q, episode_index=idx, total_episodes=len(dataset))
        records.append(ep_rec)

    assert len(records) == 2
    assert records[0]["status"] == "resolved"
    assert records[1]["status"] == "resolved"
    assert records[1]["tool_call_decay_ratio"] <= records[0]["tool_call_decay_ratio"]

    # Export metrics & curves
    m_json = export_agentic_rag_metrics(records, metrics_file, arm="aivc", model_name="qwen/qwen3.7-flash", dataset_name="test_ds")
    export_agentic_rag_curves(records, curves_file)

    assert metrics_file.exists()
    assert curves_file.exists()
    assert m_json["summary"]["pass_rate"] == 1.0
    assert m_json["retrieval_metrics"]["precision_at_1"] == 1.0
