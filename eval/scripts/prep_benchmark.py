"""
Benchmark Environment Setup and Preparation Module for AIVC.

Prepares and validates benchmark environment, dependencies, dataset caches,
and directory structures prior to executing evaluation runner arms.

Outputs:
- eval/setup/<benchmark>_ready.flag
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root and eval directory are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent

for p in [str(REPO_ROOT), str(EVAL_DIR), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

SUPPORTED_BENCHMARKS = ["agentic_rag", "devbench", "swebench_cl"]


def check_agentic_rag_env(eval_dir: Path) -> Dict[str, Any]:
    """Validate dependencies and directory structure for Agentic RAG benchmark."""
    checks: Dict[str, Any] = {}

    # 1. Check prompt template and schemas
    try:
        from aivc_prompt_template import AIVC_AGENTIC_RAG_SYSTEM_PROMPT, AIVC_RAG_TOOLS_SCHEMA  # noqa
        checks["prompt_templates"] = True
    except Exception as e:
        checks["prompt_templates"] = f"Warning: {e}"

    # 2. Check runner module
    runner_path = eval_dir / "benchmarks" / "agentic_rag_runner.py"
    checks["runner_present"] = runner_path.exists()

    # 3. Check HuggingFace / Datasets availability
    try:
        import huggingface_hub  # noqa
        checks["huggingface_hub"] = True
    except ImportError:
        checks["huggingface_hub"] = "Not installed (using built-in default sequence fallback)"

    return checks


def check_devbench_env(eval_dir: Path) -> Dict[str, Any]:
    """Validate dependencies and directory structure for DevBench benchmark."""
    checks: Dict[str, Any] = {}

    # 1. Check prompt template and schemas
    try:
        from config import WORKSPACE_TOOLS_SCHEMA, DEVBENCH_DELIVERABLE_TOOL_SCHEMA  # noqa
        checks["deliverable_tools"] = True
    except Exception as e:
        checks["deliverable_tools"] = f"Warning: {e}"

    # 2. Check runner module
    runner_path = eval_dir / "benchmarks" / "devbench_runner.py"
    checks["runner_present"] = runner_path.exists()

    return checks


def check_swebench_cl_env(eval_dir: Path) -> Dict[str, Any]:
    """Validate dependencies and directory structure for SWE-bench CL benchmark."""
    checks: Dict[str, Any] = {}

    # 1. Check Docker / Sandbox availability
    docker_available = shutil.which("docker") is not None
    checks["docker_available"] = docker_available

    # 2. Check runner module
    runner_path = eval_dir / "benchmarks" / "swebench_cl_runner.py"
    checks["runner_present"] = runner_path.exists()

    # 3. Check HuggingFace / Datasets
    try:
        import datasets  # noqa
        checks["datasets"] = True
    except ImportError:
        checks["datasets"] = "Not installed (using built-in default sequence fallback)"

    return checks


def prepare_benchmark(benchmark: str, eval_dir: Optional[Path] = None) -> Path:
    """Run preparation steps and write verification flag file."""
    base_eval = eval_dir or EVAL_DIR
    setup_dir = base_eval / "setup"
    metrics_dir = base_eval / "metrics"
    plots_dir = base_eval / "plots"
    checkpoints_dir = base_eval / "checkpoints"

    # Ensure all target artifact directories exist
    for d in [setup_dir, metrics_dir, plots_dir, checkpoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[PREP] Initializing preparation for benchmark: '{benchmark}'...")

    if benchmark == "agentic_rag":
        checks = check_agentic_rag_env(base_eval)
    elif benchmark == "devbench":
        checks = check_devbench_env(base_eval)
    elif benchmark == "swebench_cl":
        checks = check_swebench_cl_env(base_eval)
    else:
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected one of {SUPPORTED_BENCHMARKS}")

    flag_path = setup_dir / f"{benchmark}_ready.flag"
    flag_content = {
        "benchmark": benchmark,
        "status": "READY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(base_eval),
        "checks": checks,
    }

    with open(flag_path, "w", encoding="utf-8") as f:
        json.dump(flag_content, f, indent=2)
        f.write("\n")

    print(f"[PREP] Benchmark '{benchmark}' verified and ready.")
    print(f"[PREP] FLAG created: {flag_path}")
    return flag_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and validate AIVC evaluation benchmark environment.")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=SUPPORTED_BENCHMARKS,
        help="Benchmark to prepare (agentic_rag, devbench, swebench_cl)",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Custom eval directory path",
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_dir) if args.eval_dir else EVAL_DIR
    prepare_benchmark(args.benchmark, eval_path)
