"""
AIVC Evaluation Configuration & System Prompt Package.

Centralizes configuration resolution, model pricing, system prompts,
tool schemas, and resilient inference clients for all evaluation runners.
"""

import os

# Guarantee deterministic, 100% local SQLite WAL execution during evaluations (disable cloud/Drive background sync)
os.environ.setdefault("AIVC_DISABLE_SYNC", "1")

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

try:
    from ..inference_client import (
        InferenceAPIError,
        InferenceAuthError,
        InferenceBadRequestError,
        InferenceClient,
        InferenceError,
        InferenceRateLimitError,
        InferenceTimeoutError,
        OpenRouterClient,
        sanitize_messages,
    )
except (ImportError, ValueError):
    try:
        from inference_client import (  # type: ignore
            InferenceAPIError,
            InferenceAuthError,
            InferenceBadRequestError,
            InferenceClient,
            InferenceError,
            InferenceRateLimitError,
            InferenceTimeoutError,
            OpenRouterClient,
            sanitize_messages,
        )
    except ImportError:
        from eval.inference_client import (  # type: ignore
            InferenceAPIError,
            InferenceAuthError,
            InferenceBadRequestError,
            InferenceClient,
            InferenceError,
            InferenceRateLimitError,
            InferenceTimeoutError,
            OpenRouterClient,
            sanitize_messages,
        )

__all__ = [
    "AIVC_BENCHMARK_PROMPT",
    "AIVC_CORE_TOOLS_SCHEMA",
    "AIVC_DEVBENCH_SYSTEM_PROMPT",
    "AIVC_SYSTEM_PROMPT",
    "BASH_TOOL_SCHEMA",
    "DEVBENCH_DELIVERABLE_TOOL_SCHEMA",
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
    "InferenceClient",
    "OpenRouterClient",
    "InferenceError",
    "InferenceAPIError",
    "InferenceAuthError",
    "InferenceBadRequestError",
    "InferenceRateLimitError",
    "InferenceTimeoutError",
    "sanitize_messages",
]
