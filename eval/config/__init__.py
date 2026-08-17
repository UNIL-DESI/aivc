"""
AIVC Evaluation Configuration & System Prompt Package.

Centralizes configuration resolution, model pricing, system prompts,
and tool schemas for all evaluation runners.
"""

from .aivc_prompt_template import (
    AIVC_BENCHMARK_PROMPT,
    AIVC_CORE_TOOLS_SCHEMA,
    AIVC_DEVBENCH_SYSTEM_PROMPT,
    AIVC_SYSTEM_PROMPT,
    BASH_TOOL_SCHEMA,
    DEVBENCH_DELIVERABLE_TOOL_SCHEMA,
    WORKSPACE_TOOLS_SCHEMA,
    get_aivc_system_prompt,
    get_benchmark_tools_schema,
)
from .config_loader import (
    CONFIG_DIR,
    EVAL_DIR,
    REPO_ROOT,
    EvalProfileConfig,
    ModelSpec,
    PathConfig,
    add_eval_args,
    load_benchmark_config,
    load_env_file,
    load_models_registry,
    load_params_yaml,
    load_profile_yaml,
    load_yaml_file,
    resolve_config,
    resolve_config_from_args,
)

__all__ = [
    "AIVC_BENCHMARK_PROMPT",
    "AIVC_CORE_TOOLS_SCHEMA",
    "AIVC_SYSTEM_PROMPT",
    "BASH_TOOL_SCHEMA",
    "WORKSPACE_TOOLS_SCHEMA",
    "get_aivc_system_prompt",
    "get_benchmark_tools_schema",
    "CONFIG_DIR",
    "EVAL_DIR",
    "REPO_ROOT",
    "EvalProfileConfig",
    "ModelSpec",
    "PathConfig",
    "add_eval_args",
    "load_benchmark_config",
    "load_env_file",
    "load_models_registry",
    "load_params_yaml",
    "load_profile_yaml",
    "load_yaml_file",
    "resolve_config",
    "resolve_config_from_args",
]
