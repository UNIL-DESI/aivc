"""
Configuration Loader for AIVC Evaluation Benchmarks.

Provides centralized configuration loading from:
1. params.yaml (DVC parameters & active profile)
2. eval/config/models_openrouter.yaml (Model specifications, context windows, pricing)
3. Environment variables / .env files (OPENROUTER_API_KEY, AIVC_STORAGE_ROOT)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional PyYAML support with robust fallback
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Default fallback pricing per 1M tokens in USD
DEFAULT_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "qwen/qwen3.7-flash": {
        "prompt_price_per_1m": 0.03,
        "completion_price_per_1m": 0.13,
        "context_window": 128000,
    },
    "deepseek/deepseek-v4-flash-0731": {
        "prompt_price_per_1m": 0.07,
        "completion_price_per_1m": 0.28,
        "context_window": 64000,
    },
    "openai/gpt-5.6-luna-pro": {
        "prompt_price_per_1m": 1.50,
        "completion_price_per_1m": 6.00,
        "context_window": 256000,
    },
    "google/gemini-3.7-flash": {
        "prompt_price_per_1m": 0.05,
        "completion_price_per_1m": 0.20,
        "context_window": 1000000,
    },
    "google/gemini-3.6-flash": {
        "prompt_price_per_1m": 0.05,
        "completion_price_per_1m": 0.20,
        "context_window": 1000000,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "prompt_price_per_1m": 0.88,
        "completion_price_per_1m": 0.88,
        "context_window": 128000,
    },
    "qwen/qwen-2.5-72b-instruct": {
        "prompt_price_per_1m": 0.35,
        "completion_price_per_1m": 0.40,
        "context_window": 128000,
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "prompt_price_per_1m": 0.18,
        "completion_price_per_1m": 0.18,
        "context_window": 128000,
    },
    "qwen/qwen-2.5-7b-instruct": {
        "prompt_price_per_1m": 0.15,
        "completion_price_per_1m": 0.15,
        "context_window": 128000,
    },
    "anthropic/claude-sonnet-5": {
        "prompt_price_per_1m": 3.00,
        "completion_price_per_1m": 15.00,
        "context_window": 200000,
    },
    "z-ai/glm-5.2": {
        "prompt_price_per_1m": 0.10,
        "completion_price_per_1m": 0.40,
        "context_window": 128000,
    },
    "meta-models/Muse-Glimmer-30B": {
        "prompt_price_per_1m": 0.35,
        "completion_price_per_1m": 1.50,
        "batch_prompt_price_per_1m": 0.175,
        "batch_completion_price_per_1m": 0.75,
        "context_window": 131072,
        "provider": "together",
    },
    "openai/gpt-oss-20b": {
        "prompt_price_per_1m": 0.20,
        "completion_price_per_1m": 0.60,
        "batch_prompt_price_per_1m": 0.10,
        "batch_completion_price_per_1m": 0.30,
        "context_window": 131072,
        "provider": "together",
        "role": "validation_together",
    },
}


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Parse .env file and return key-value mapping."""
    if env_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        env_path = repo_root / ".env"

    env_vars: Dict[str, str] = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    clean_k = k.strip()
                    clean_v = v.strip().strip("'\"")
                    env_vars[clean_k] = clean_v
                    if clean_k not in os.environ:
                        os.environ[clean_k] = clean_v
    return env_vars


def load_params_yaml(params_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load params.yaml from repository root."""
    if params_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        params_path = repo_root / "params.yaml"

    if not params_path.exists():
        return {
            "profile": "dry_run",
            "eval": {
                "limit": 30,
                "max_turns": 50,
                "max_tokens": 4096,
                "max_cost_per_instance_usd": 0.10,
                "model": "qwen/qwen3.7-flash",
            },
        }

    if HAS_YAML:
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # Simple fallback parser for params.yaml without PyYAML
    data: Dict[str, Any] = {"eval": {}}
    with open(params_path, "r", encoding="utf-8") as f:
        in_eval = False
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if line_str.startswith("profile:"):
                data["profile"] = line_str.split(":", 1)[1].strip().strip("'\"")
            elif line_str.startswith("eval:"):
                in_eval = True
            elif in_eval and ":" in line_str:
                k, v = line_str.split(":", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                try:
                    if "." in v:
                        data["eval"][k] = float(v)
                    else:
                        data["eval"][k] = int(v)
                except ValueError:
                    data["eval"][k] = v
    return data


def load_models_config(models_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load eval/config/models.yaml or models_openrouter.yaml configuration."""
    if models_path is None:
        eval_dir = Path(__file__).resolve().parent
        primary_path = eval_dir / "config" / "models.yaml"
        fallback_path = eval_dir / "config" / "models_openrouter.yaml"
        models_path = primary_path if primary_path.exists() else fallback_path

    if not models_path.exists():
        return {
            "active_model": "qwen/qwen3.7-flash",
            "models": DEFAULT_MODEL_PRICING,
        }

    if HAS_YAML:
        with open(models_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {
        "active_model": "qwen/qwen3.7-flash",
        "models": DEFAULT_MODEL_PRICING,
    }



def get_model_pricing(model_name: str, models_config: Optional[Dict[str, Any]] = None) -> Tuple[float, float, int]:
    """
    Return (prompt_price_per_1m, completion_price_per_1m, context_window) for a given model.
    """
    if models_config and "models" in models_config:
        m_info = models_config["models"].get(model_name)
        if m_info:
            p_price = float(m_info.get("prompt_price_per_1m", 0.03))
            c_price = float(m_info.get("completion_price_per_1m", 0.13))
            ctx = int(m_info.get("context_window", 128000))
            return p_price, c_price, ctx

    if model_name in DEFAULT_MODEL_PRICING:
        info = DEFAULT_MODEL_PRICING[model_name]
        return info["prompt_price_per_1m"], info["completion_price_per_1m"], int(info["context_window"])

    # Generic default for unknown models
    return 0.03, 0.13, 128000


def resolve_benchmark_paths(
    benchmark_name: str,
    model_name: str,
    arm: Optional[str] = None,
    profile: Optional[str] = None,
    eval_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Resolve checkpoint, metrics, and plot curves paths based on profile, model, and arm.
    """
    if eval_dir is None:
        eval_dir = Path(__file__).resolve().parent

    clean_model = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    arm_suffix = f"_{arm}" if arm else ""

    profile_sub = profile if profile and profile != "default" else ""

    if profile_sub:
        ckpt_dir = eval_dir / "checkpoints" / profile_sub
        metrics_dir = eval_dir / "metrics" / profile_sub
        plots_dir = eval_dir / "plots" / profile_sub
    else:
        ckpt_dir = eval_dir / "checkpoints"
        metrics_dir = eval_dir / "metrics"
        plots_dir = eval_dir / "plots"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = ckpt_dir / f"{benchmark_name}_{clean_model}{arm_suffix}_checkpoint.jsonl"
    metrics_file = metrics_dir / f"{benchmark_name}_{clean_model}{arm_suffix}_metrics.json"
    curves_file = plots_dir / f"{benchmark_name}{arm_suffix}_curves.csv"

    return {
        "checkpoint_path": ckpt_file,
        "metrics_path": metrics_file,
        "plots_path": curves_file,
    }
