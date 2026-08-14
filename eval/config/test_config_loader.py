"""
Unit tests for AIVC Evaluation Configuration & Prompt Templates.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.config.config_loader import (
    load_models_registry,
    load_profile_yaml,
    resolve_config,
    add_eval_args,
    resolve_config_from_args,
    ModelSpec,
)
from eval.config.aivc_prompt_template import (
    AIVC_SYSTEM_PROMPT,
    AIVC_BENCHMARK_PROMPT,
    AIVC_CORE_TOOLS_SCHEMA,
    WORKSPACE_TOOLS_SCHEMA,
    get_benchmark_tools_schema,
    get_aivc_system_prompt,
)


def test_models_registry():
    registry = load_models_registry()
    assert "qwen/qwen3.7-flash" in registry
    assert "deepseek/deepseek-v4-flash-0731" in registry
    assert "google/gemini-3.7-flash" in registry
    assert "openai/gpt-5.6-luna-pro" in registry

    qwen = registry["qwen/qwen3.7-flash"]
    assert qwen.prompt_price_per_1m == 0.03
    assert qwen.completion_price_per_1m == 0.13
    assert qwen.context_window == 128000
    assert qwen.supports_tools is True
    assert qwen.slug == "qwen3.7-flash"

    # Test cost calculation: 100k prompt tokens, 10k completion tokens
    cost = qwen.compute_cost(100_000, 10_000)
    assert abs(cost - (0.003 + 0.0013)) < 1e-6


def test_dry_run_profile():
    cfg = resolve_config(profile="dry_run")
    assert cfg.profile == "dry_run"
    assert cfg.dry_run is True
    assert cfg.model == "qwen/qwen3.7-flash"
    assert cfg.limit == 15
    assert cfg.reset_checkpoint is True
    assert cfg.max_turns == 50
    assert cfg.max_tokens == 4096
    assert cfg.max_cost_per_instance_usd == 0.10

    paths = cfg.get_paths()
    assert "dry_run" in str(paths.checkpoints_dir)
    assert "dry_run" in str(paths.metrics_dir)


def test_production_profile():
    cfg = resolve_config(profile="production")
    assert cfg.profile == "production"
    assert cfg.dry_run is False
    assert "deepseek/deepseek-v4-flash-0731" in cfg.models
    assert "google/gemini-3.7-flash" in cfg.models
    assert "openai/gpt-5.6-luna-pro" in cfg.models
    assert cfg.limit == 273
    assert cfg.reset_checkpoint is False
    assert cfg.max_cost_per_instance_usd == 0.50

    paths = cfg.get_paths()
    assert "production" in str(paths.checkpoints_dir)
    assert "deepseek-v4-flash-0731" in str(paths.checkpoints_dir)


def test_production_model_override():
    cfg = resolve_config(profile="production", model="google/gemini-3.7-flash", limit=50)
    assert cfg.model == "google/gemini-3.7-flash"
    assert cfg.limit == 50
    paths = cfg.get_paths()
    assert "gemini-3.7-flash" in str(paths.checkpoints_dir)
    assert "gemini-3.7-flash" in str(paths.metrics_dir)


def test_cli_arg_resolution():
    parser = argparse.ArgumentParser()
    add_eval_args(parser)
    args = parser.parse_args(["--profile", "production", "--limit", "10", "--no-reset-checkpoint", "--max-cost", "0.25"])
    cfg = resolve_config_from_args(args)
    assert cfg.profile == "production"
    assert cfg.limit == 10
    assert cfg.reset_checkpoint is False
    assert cfg.max_cost_per_instance_usd == 0.25


def test_prompt_template_and_schemas():
    assert "AIVC — AI Version Control (Long-Term Memory)" in AIVC_SYSTEM_PROMPT
    assert "Recall Funnel" in AIVC_SYSTEM_PROMPT
    assert "REMEMBER OFTEN" in AIVC_SYSTEM_PROMPT
    assert "RECALL FIRST" in AIVC_SYSTEM_PROMPT

    tools = get_benchmark_tools_schema(include_workspace=True, include_bash=True)
    tool_names = [t["function"]["name"] for t in tools]
    assert "remember" in tool_names
    assert "recall" in tool_names
    assert "get_recent_memories" in tool_names
    assert "consult_memory" in tool_names
    assert "get_file_history_metadata" in tool_names
    assert "read_past_file_content" in tool_names
    assert "view_file" in tool_names
    assert "grep_search" in tool_names
    assert "list_dir" in tool_names
    assert "submit_patch" in tool_names
    assert "execute_command" in tool_names

    # Check harmonized parameter names: 'top_n' in recall and 'file_path' in get_file_history_metadata
    recall_tool = next(t for t in tools if t["function"]["name"] == "recall")
    assert "top_n" in recall_tool["function"]["parameters"]["properties"]

    file_hist_tool = next(t for t in tools if t["function"]["name"] == "get_file_history_metadata")
    assert "file_path" in file_hist_tool["function"]["parameters"]["properties"]

    read_past_tool = next(t for t in tools if t["function"]["name"] == "read_past_file_content")
    assert "file_path" in read_past_tool["function"]["parameters"]["properties"]


if __name__ == "__main__":
    test_models_registry()
    test_dry_run_profile()
    test_production_profile()
    test_production_model_override()
    test_cli_arg_resolution()
    test_prompt_template_and_schemas()
    print("All configuration tests passed successfully!")
