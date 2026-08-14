"""
AIVC Evaluation Configuration Resolver & Loader.

Handles hierarchical configuration loading, profile resolution (dry_run vs production),
CLI argument parsing, and dataclass typing for benchmark runners.

Hierarchical Resolution Order:
1. Hardcoded Baseline Defaults
2. Repository `params.yaml` configuration
3. Profile YAML (`eval/config/profiles/{profile}.yaml`)
4. Explicit CLI arguments / Overrides
5. Model registry lookup from `eval/config/models.yaml`
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False


# Base Paths
CONFIG_DIR = Path(__file__).resolve().parent
EVAL_DIR = CONFIG_DIR.parent
REPO_ROOT = EVAL_DIR.parent


# ---------------------------------------------------------------------------
# 1. Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Specification and pricing metadata for an evaluation LLM."""
    model_id: str
    name: str
    provider: str = "openrouter"
    prompt_price_per_1m: float = 0.0
    completion_price_per_1m: float = 0.0
    context_window: int = 128000
    supports_tools: bool = True
    description: str = ""

    @property
    def slug(self) -> str:
        """Sanitized filesystem-friendly identifier (e.g. 'qwen3.7-flash')."""
        if "/" in self.model_id:
            return self.model_id.split("/")[-1]
        return re.sub(r"[^\w\-.]", "_", self.model_id)

    @property
    def full_slug(self) -> str:
        """Full namespaced slug replacing '/' with '_' (e.g. 'qwen_qwen3.7-flash')."""
        return self.model_id.replace("/", "_")

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total inference cost in USD for the given token counts."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_1m
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_1m
        return prompt_cost + completion_cost


@dataclass
class PathConfig:
    """Resolved directory paths for benchmark checkpoints, metrics, and plots."""
    checkpoints_dir: Path
    metrics_dir: Path
    plots_dir: Path
    scratch_dir: Path

    def ensure_dirs(self) -> None:
        """Create all required directories if they do not already exist."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalProfileConfig:
    """Complete, validated configuration for an evaluation run."""
    profile: str
    dry_run: bool
    model: str
    models: List[str] = field(default_factory=list)
    limit: Optional[int] = 15
    reset_checkpoint: bool = True
    max_turns: int = 50
    max_tokens: int = 4096
    max_cost_per_instance_usd: float = 0.10
    raw_paths: Dict[str, str] = field(default_factory=dict)
    model_spec: Optional[ModelSpec] = None
    registry: Dict[str, ModelSpec] = field(default_factory=dict)

    def get_paths(self, model_name: Optional[str] = None, base_dir: Optional[Path] = None) -> PathConfig:
        """
        Resolve templated paths with `{model_slug}` and return PathConfig.
        
        Args:
            model_name: Model identifier used to resolve {model_slug}.
            base_dir: Root directory for relative paths (defaults to REPO_ROOT).
        """
        root = base_dir or REPO_ROOT
        target_model = model_name or self.model
        spec = self.registry.get(target_model)
        slug = spec.slug if spec else (target_model.split("/")[-1] if "/" in target_model else target_model)

        def _resolve_template(template_str: str, default_sub: str) -> Path:
            val = template_str or default_sub
            formatted = val.format(model_slug=slug, model=target_model)
            p = Path(formatted)
            return p if p.is_absolute() else (root / p)

        checkpoints = _resolve_template(
            self.raw_paths.get("checkpoints_dir", f"eval/checkpoints/{self.profile}"),
            f"eval/checkpoints/{self.profile}",
        )
        metrics = _resolve_template(
            self.raw_paths.get("metrics_dir", f"eval/metrics/{self.profile}"),
            f"eval/metrics/{self.profile}",
        )
        plots = _resolve_template(
            self.raw_paths.get("plots_dir", f"eval/plots/{self.profile}"),
            f"eval/plots/{self.profile}",
        )
        scratch = _resolve_template(
            self.raw_paths.get("scratch_dir", f"eval/scratch/{self.profile}"),
            f"eval/scratch/{self.profile}",
        )

        return PathConfig(
            checkpoints_dir=checkpoints,
            metrics_dir=metrics,
            plots_dir=plots,
            scratch_dir=scratch,
        )

    def get_model_spec(self, model_name: Optional[str] = None) -> ModelSpec:
        """Get the ModelSpec for the active or specified model."""
        target_model = model_name or self.model
        if target_model in self.registry:
            return self.registry[target_model]
        return ModelSpec(
            model_id=target_model,
            name=target_model.split("/")[-1],
            prompt_price_per_1m=0.05,
            completion_price_per_1m=0.20,
            context_window=128000,
            supports_tools=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "profile": self.profile,
            "dry_run": self.dry_run,
            "model": self.model,
            "models": self.models,
            "limit": self.limit,
            "reset_checkpoint": self.reset_checkpoint,
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens,
            "max_cost_per_instance_usd": self.max_cost_per_instance_usd,
            "raw_paths": self.raw_paths,
            "model_spec": {
                "name": self.model_spec.name,
                "model_id": self.model_spec.model_id,
                "prompt_price_per_1m": self.model_spec.prompt_price_per_1m,
                "completion_price_per_1m": self.model_spec.completion_price_per_1m,
                "context_window": self.model_spec.context_window,
            } if self.model_spec else None,
        }


# ---------------------------------------------------------------------------
# 2. File Loaders & Parsers
# ---------------------------------------------------------------------------

def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file safely with a fallback for simple key-value structures."""
    if not path.exists():
        return {}

    if HAS_YAML and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"[config_loader] Error parsing YAML from {path}: {e}", file=sys.stderr)
                return {}

    # Basic fallback parsing if pyyaml is missing
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if val.lower() == "true":
                    data[key] = True
                elif val.lower() == "false":
                    data[key] = False
                elif val.isdigit():
                    data[key] = int(val)
                else:
                    data[key] = val
    return data


def load_models_registry(models_path: Optional[Path] = None) -> Dict[str, ModelSpec]:
    """Load all registered models from models.yaml or models_openrouter.yaml."""
    target_path = models_path or (CONFIG_DIR / "models.yaml")
    if not target_path.exists():
        # Fallback to models_openrouter.yaml if models.yaml doesn't exist
        alt_path = CONFIG_DIR / "models_openrouter.yaml"
        if alt_path.exists():
            target_path = alt_path

    raw_data = load_yaml_file(target_path)
    models_dict = raw_data.get("models", {})
    registry: Dict[str, ModelSpec] = {}

    for model_id, info in models_dict.items():
        if not isinstance(info, dict):
            continue
        spec = ModelSpec(
            model_id=model_id,
            name=info.get("name", model_id),
            provider=info.get("provider", "openrouter"),
            prompt_price_per_1m=float(info.get("prompt_price_per_1m", 0.0)),
            completion_price_per_1m=float(info.get("completion_price_per_1m", 0.0)),
            context_window=int(info.get("context_window", 128000)),
            supports_tools=bool(info.get("supports_tools", True)),
            description=str(info.get("description", "")),
        )
        registry[model_id] = spec

    return registry


def load_profile_yaml(profile_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a profile YAML configuration from eval/config/profiles/{profile_name}.yaml."""
    base_dir = config_dir or CONFIG_DIR
    profile_path = base_dir / "profiles" / f"{profile_name}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Evaluation profile '{profile_name}' not found at: {profile_path}")
    return load_yaml_file(profile_path)


def load_params_yaml(params_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load repository params.yaml if available."""
    target_path = params_path or (REPO_ROOT / "params.yaml")
    if target_path.exists():
        return load_yaml_file(target_path)
    return {}


# ---------------------------------------------------------------------------
# 3. Hierarchical Configuration Resolver
# ---------------------------------------------------------------------------

def resolve_config(
    profile: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    reset_checkpoint: Optional[bool] = None,
    max_turns: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_cost_per_instance_usd: Optional[float] = None,
    config_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    models_path: Optional[Path] = None,
) -> EvalProfileConfig:
    """
    Hierarchically resolve evaluation configuration across params.yaml, profile YAML, and overrides.
    """
    base_cfg_dir = config_dir or CONFIG_DIR
    models_reg = load_models_registry(models_path)

    # 1. Load root params.yaml
    params_data = load_params_yaml(params_path)
    params_eval = params_data.get("eval", {})

    # Determine active profile name
    selected_profile = profile or params_data.get("profile") or "dry_run"

    # 2. Load profile YAML
    profile_data = load_profile_yaml(selected_profile, base_cfg_dir)

    # 3. Hierarchical Merge: Profile -> Params.yaml -> CLI Overrides
    is_dry_run = profile_data.get("dry_run", selected_profile == "dry_run")
    
    # Model resolution
    resolved_model = (
        model
        or profile_data.get("model")
        or params_eval.get("model")
        or "qwen/qwen3.7-flash"
    )

    models_list = profile_data.get("models") or [resolved_model]
    if resolved_model not in models_list:
        models_list.insert(0, resolved_model)

    # Limit resolution
    resolved_limit: Optional[int]
    if limit is not None:
        resolved_limit = limit
    elif "limit" in profile_data:
        resolved_limit = profile_data["limit"]
    elif "limit" in params_eval:
        resolved_limit = params_eval["limit"]
    else:
        resolved_limit = 15 if is_dry_run else 273

    # Reset checkpoint resolution
    resolved_reset: bool
    if reset_checkpoint is not None:
        resolved_reset = reset_checkpoint
    elif "reset_checkpoint" in profile_data:
        resolved_reset = bool(profile_data["reset_checkpoint"])
    else:
        resolved_reset = True if is_dry_run else False

    # Max turns resolution
    resolved_max_turns = (
        max_turns
        if max_turns is not None
        else profile_data.get("max_turns", params_eval.get("max_turns", 50))
    )

    # Max tokens resolution
    resolved_max_tokens = (
        max_tokens
        if max_tokens is not None
        else profile_data.get("max_tokens", params_eval.get("max_tokens", 4096))
    )

    # Max cost per instance resolution
    resolved_max_cost = (
        max_cost_per_instance_usd
        if max_cost_per_instance_usd is not None
        else profile_data.get(
            "max_cost_per_instance_usd",
            params_eval.get("max_cost_per_instance_usd", 0.10 if is_dry_run else 0.50),
        )
    )

    raw_paths = profile_data.get("paths", {})

    # Model spec from registry
    model_spec = models_reg.get(resolved_model)
    if not model_spec:
        model_spec = ModelSpec(
            model_id=resolved_model,
            name=resolved_model.split("/")[-1],
            prompt_price_per_1m=0.05,
            completion_price_per_1m=0.20,
            context_window=128000,
            supports_tools=True,
        )

    return EvalProfileConfig(
        profile=selected_profile,
        dry_run=is_dry_run,
        model=resolved_model,
        models=models_list,
        limit=resolved_limit,
        reset_checkpoint=resolved_reset,
        max_turns=resolved_max_turns,
        max_tokens=resolved_max_tokens,
        max_cost_per_instance_usd=float(resolved_max_cost),
        raw_paths=raw_paths,
        model_spec=model_spec,
        registry=models_reg,
    )


# ---------------------------------------------------------------------------
# 4. CLI Argument Parser Integration
# ---------------------------------------------------------------------------

def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach standard evaluation configuration flags to an ArgumentParser."""
    group = parser.add_argument_group("AIVC Evaluation Configuration")
    group.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["dry_run", "production"],
        help="Evaluation execution profile (dry_run: 15 tasks/fast; production: 273 tasks/full paper).",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target OpenRouter model identifier (e.g. 'qwen/qwen3.7-flash', 'deepseek/deepseek-v4-flash-0731').",
    )
    group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of benchmark instances/episodes to run.",
    )
    group.add_argument(
        "--reset-checkpoint",
        action="store_true",
        default=None,
        help="Force clean checkpoint reset before execution.",
    )
    group.add_argument(
        "--no-reset-checkpoint",
        action="store_false",
        dest="reset_checkpoint",
        help="Disable checkpoint reset to enable crash recovery and resume.",
    )
    group.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum agent conversation turns per episode (default: 50).",
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum LLM generation tokens per response (default: 4096).",
    )
    group.add_argument(
        "--max-cost",
        type=float,
        dest="max_cost_per_instance_usd",
        default=None,
        help="Financial safety cap per benchmark instance in USD.",
    )
    return parser


def resolve_config_from_args(
    args: argparse.Namespace,
    config_dir: Optional[Path] = None,
) -> EvalProfileConfig:
    """Resolve EvalProfileConfig using parsed CLI namespace."""
    return resolve_config(
        profile=getattr(args, "profile", None),
        model=getattr(args, "model", None),
        limit=getattr(args, "limit", None),
        reset_checkpoint=getattr(args, "reset_checkpoint", None),
        max_turns=getattr(args, "max_turns", None),
        max_tokens=getattr(args, "max_tokens", None),
        max_cost_per_instance_usd=getattr(args, "max_cost_per_instance_usd", None),
        config_dir=config_dir,
    )


def load_benchmark_config(
    parser: Optional[argparse.ArgumentParser] = None,
    args: Optional[argparse.Namespace] = None,
    profile: Optional[str] = None,
    model: Optional[str] = None,
    limit: Optional[int] = None,
    reset_checkpoint: Optional[bool] = None,
    max_turns: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_cost_per_instance_usd: Optional[float] = None,
    config_dir: Optional[Path] = None,
    params_path: Optional[Path] = None,
    models_path: Optional[Path] = None,
) -> EvalProfileConfig:
    """
    Load and resolve evaluation configuration for benchmark runners.
    If args is provided, resolves from namespace.
    If parser is provided (and args is None), parses known CLI arguments.
    Otherwise resolves directly from provided parameters or defaults.
    """
    if args is not None:
        return resolve_config_from_args(args, config_dir=config_dir)
    if parser is not None:
        parsed_args, _ = parser.parse_known_args()
        return resolve_config_from_args(parsed_args, config_dir=config_dir)
    return resolve_config(
        profile=profile,
        model=model,
        limit=limit,
        reset_checkpoint=reset_checkpoint,
        max_turns=max_turns,
        max_tokens=max_tokens,
        max_cost_per_instance_usd=max_cost_per_instance_usd,
        config_dir=config_dir,
        params_path=params_path,
        models_path=models_path,
    )


# ---------------------------------------------------------------------------
# 5. CLI Inspection Utility
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or test AIVC Evaluation Configuration.")
    add_eval_args(parser)
    parsed_args = parser.parse_args()

    config = resolve_config_from_args(parsed_args)
    paths = config.get_paths()

    print("=" * 70)
    print(f"  AIVC Configuration Profile: [{config.profile.upper()}]")
    print("=" * 70)
    print(f"  Dry Run Mode        : {config.dry_run}")
    print(f"  Active Model        : {config.model}")
    print(f"  Supported Models    : {config.models}")
    print(f"  Instance Limit (N)  : {config.limit}")
    print(f"  Reset Checkpoint    : {config.reset_checkpoint}")
    print(f"  Max Turns / Episode : {config.max_turns}")
    print(f"  Max Tokens / Step   : {config.max_tokens}")
    print(f"  Max Cost / Instance : ${config.max_cost_per_instance_usd:.2f} USD")
    if config.model_spec:
        print(f"  Model Pricing (1M)  : Prompt: ${config.model_spec.prompt_price_per_1m:.2f} | Completion: ${config.model_spec.completion_price_per_1m:.2f}")
        print(f"  Context Window      : {config.model_spec.context_window:,} tokens")
    print("-" * 70)
    print("  Resolved Output Paths:")
    print(f"    Checkpoints : {paths.checkpoints_dir}")
    print(f"    Metrics     : {paths.metrics_dir}")
    print(f"    Plots       : {paths.plots_dir}")
    print(f"    Scratch     : {paths.scratch_dir}")
    print("=" * 70)
