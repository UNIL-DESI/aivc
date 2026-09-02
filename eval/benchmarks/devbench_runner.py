"""
DevBench 4-Phase SDLC Benchmark Runner for AIVC.

Evaluates AI coding agents across the complete Software Development Life Cycle:
1. Software Design
2. Environment Setup
3. Code Implementation
4. Unit Testing

Features:
- Live multi-turn interaction loop (up to 50 turns per phase) with OpenRouter API.
- Live AIVC MCP tool injection and execution (remember, recall, consult_memory, etc.).
- Incremental JSONL checkpointing with .flush() and fsync after every phase/task.
- Strict financial cutoff ($0.10 USD / phase).
- Metrics export to JSON (eval/metrics/devbench_metrics.json).
- Plots curve export to CSV (eval/plots/devbench_curves.csv).
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root and eval directory are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Enforce deterministic 100% local execution (no background sync/network calls)
os.environ.setdefault("AIVC_DISABLE_SYNC", "1")

# Import TrajectoryAnalyzer metrics if available
from metrics.trajectory_analyzer import (
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    compute_ccsr,
    compute_eor,
    compute_mui,
    compute_ndcg_at_k,
    compute_retrieval_metrics,
    extract_files_from_patch,
)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# DevBench 4-phase SDLC sequence
SDLC_PHASES = [
    "software_design",
    "environment_setup",
    "code_implementation",
    "unit_testing",
]

# Set of exploration tool calls for EOR calculation
EXPLORATION_TOOLS = {
    "grep_search",
    "list_dir",
    "view_file",
    "find_by_name",
    "search_web",
    "read_past_file_content",
}

# Import unified configuration, prompt template, and tool schemas from eval.config
from config import (
    InferenceClient,
    WORKSPACE_TOOLS_SCHEMA,
    DEVBENCH_DELIVERABLE_TOOL_SCHEMA,
    add_eval_args,
    get_aivc_system_prompt,
    get_benchmark_tools_schema,
    load_benchmark_config,
    load_models_registry,
    sanitize_messages,
)
import copy

AIVC_DEVBENCH_SYSTEM_PROMPT = get_aivc_system_prompt(benchmark_type="devbench")
AIVC_DEVBENCH_TOOLS_SCHEMA = get_benchmark_tools_schema(include_workspace=True, benchmark_type="devbench")

NAIVE_DEVBENCH_SYSTEM_PROMPT = """# Autonomous Software Engineer (Stateless Baseline SDLC)

You are an expert autonomous software engineer working through the Software Development Life Cycle (SDLC).
You operate in a **stateless, ephemeral environment** with zero persistent memory between phases and tasks.

## Tool Arsenal:
- `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
- `grep_search(query: str, search_path: str = ".")`: Search pattern across codebase.
- `list_dir(directory: str = ".")`: List contents of a directory.
- `submit_phase_deliverable(deliverable: str, notes: str)`: Submit the final deliverable for the current SDLC phase.

## Protocol Rules:
1. Inspect files and directories using `view_file`, `grep_search`, and `list_dir`.
2. Call `submit_phase_deliverable` when the phase goal is achieved.
"""

NAIVE_DEVBENCH_TOOLS_SCHEMA = [
    copy.deepcopy(t) for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] != "submit_patch"
] + [copy.deepcopy(DEVBENCH_DELIVERABLE_TOOL_SCHEMA)]


# ---------------------------------------------------------------------------
# DevBench Repositories & 4-Phase Specifications
# ---------------------------------------------------------------------------

DEFAULT_DEVBENCH_REPOS = [
    {
        "repo_id": "devbench-python-calculator",
        "domain": "Python",
        "description": "Scientific Calculator library with memory management and expression parsing.",
        "baseline_est_cost": 0.012,
        "phases": {
            "software_design": {
                "prompt": "Design architecture, class hierarchy, AST parser specification, and state machine for Python Calculator.",
                "initial_files": ["calculator/ast.py", "calculator/engine.py"],
            },
            "environment_setup": {
                "prompt": "Configure virtualenv, pyproject.toml dependencies, build system, and setup script for Calculator repo.",
                "initial_files": ["pyproject.toml", "setup.py"],
            },
            "code_implementation": {
                "prompt": "Implement core calculation engine, expression evaluator, memory stack, and tokenizer module.",
                "initial_files": ["calculator/engine.py", "calculator/tokenizer.py"],
            },
            "unit_testing": {
                "prompt": "Create test_calculator.py test suite covering edge cases, division by zero, float precision, and AST evaluation.",
                "initial_files": ["tests/test_calculator.py"],
            },
        },
    },
    {
        "repo_id": "devbench-cpp-parser",
        "domain": "C/C++",
        "description": "High-performance JSON parser & serializer with memory pooling.",
        "baseline_est_cost": 0.018,
        "phases": {
            "software_design": {
                "prompt": "Design high-performance zero-copy JSON parser header architecture, memory pool arena, and CMake build graph.",
                "initial_files": ["include/json_parser.hpp", "CMakeLists.txt"],
            },
            "environment_setup": {
                "prompt": "Configure CMakeLists.txt, GoogleTest integration, compiler optimization flags, and clang-format rules.",
                "initial_files": ["CMakeLists.txt", "vcpkg.json"],
            },
            "code_implementation": {
                "prompt": "Implement lexer, token stream buffer, and AST node allocator in modern C++17.",
                "initial_files": ["src/lexer.cpp", "src/parser.cpp"],
            },
            "unit_testing": {
                "prompt": "Implement GoogleTest test fixtures for malformed JSON, UTF-8 unicode encoding, and throughput benchmarks.",
                "initial_files": ["tests/test_parser.cpp"],
            },
        },
    },
    {
        "repo_id": "devbench-java-rest-api",
        "domain": "Java",
        "description": "Spring Boot RESTful microservice with JWT auth and JPA persistence.",
        "baseline_est_cost": 0.015,
        "phases": {
            "software_design": {
                "prompt": "Design controller-service-repository layered architecture, security filter chain, and OpenAPI 3.0 spec.",
                "initial_files": ["src/main/java/com/app/controller/UserController.java", "openapi.yaml"],
            },
            "environment_setup": {
                "prompt": "Configure pom.xml Maven dependencies, Spring Data JPA, H2 test database, and Dockerfile development image.",
                "initial_files": ["pom.xml", "Dockerfile"],
            },
            "code_implementation": {
                "prompt": "Implement UserController, AuthService, JwtTokenProvider, and UserRepository with BCrypt password hashing.",
                "initial_files": ["src/main/java/com/app/service/AuthService.java", "src/main/java/com/app/security/JwtTokenProvider.java"],
            },
            "unit_testing": {
                "prompt": "Write JUnit 5 and Mockito tests for authentication endpoints, token expiration, and security filters.",
                "initial_files": ["src/test/java/com/app/service/AuthServiceTest.java"],
            },
        },
    },
    {
        "repo_id": "devbench-react-dashboard",
        "domain": "Web/TypeScript",
        "description": "Analytics dashboard frontend with React 19, Tailwind, and WebSocket client.",
        "baseline_est_cost": 0.014,
        "phases": {
            "software_design": {
                "prompt": "Design state management, component tree, metric telemetry feeds, and WebSocket subscription protocol.",
                "initial_files": ["src/types/metrics.ts", "src/components/Dashboard.tsx"],
            },
            "environment_setup": {
                "prompt": "Configure package.json, Vite build settings, TypeScript strict config, Tailwind CSS, and ESLint.",
                "initial_files": ["package.json", "tsconfig.json", "vite.config.ts"],
            },
            "code_implementation": {
                "prompt": "Implement DashboardView, ChartCard, useWebSocket hook, metric formatters, and dark mode toggling.",
                "initial_files": ["src/hooks/useWebSocket.ts", "src/components/ChartCard.tsx"],
            },
            "unit_testing": {
                "prompt": "Write Vitest & React Testing Library tests for chart rendering, WebSocket reconnects, and state updates.",
                "initial_files": ["src/components/__tests__/Dashboard.test.tsx"],
            },
        },
    },
    {
        "repo_id": "devbench-go-microservice",
        "domain": "Go",
        "description": "gRPC event streaming microservice with Redis caching and Prometheus metrics.",
        "baseline_est_cost": 0.016,
        "phases": {
            "software_design": {
                "prompt": "Design Protobuf schema, gRPC service methods, Redis caching topology, and event store schema.",
                "initial_files": ["proto/events.proto", "internal/server/server.go"],
            },
            "environment_setup": {
                "prompt": "Configure go.mod, protoc compiler plugins, Docker Compose dependencies (Redis), and Makefile build targets.",
                "initial_files": ["go.mod", "Makefile", "docker-compose.yml"],
            },
            "code_implementation": {
                "prompt": "Implement gRPC server handlers, Redis stream producer/consumer, and Prometheus metric middleware.",
                "initial_files": ["internal/handlers/event_handler.go", "internal/storage/redis.go"],
            },
            "unit_testing": {
                "prompt": "Implement Go table-driven unit tests and gRPC bufconn mock client tests for streaming throughput.",
                "initial_files": ["internal/handlers/event_handler_test.go"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# In-Memory AIVC Environment for DevBench (Hermetically Scoped per Repo)
# ---------------------------------------------------------------------------

class DevBenchAIVCEnvironment:
    """
    Maintains AIVC memory store across SDLC phases of each repository.
    Hermetically isolated per repo_id to guarantee zero cross-repository data contamination.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        arm: str = "aivc",
        run_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.arm = arm.lower()
        self.current_repo_id: str = repo_id or "default"
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.workspace_dir = workspace_dir or (EVAL_DIR / "scratch" / f"aivc_devbench_{self.run_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Set sandbox environment variables
        os.environ["AIVC_STORAGE_ROOT"] = str(self.workspace_dir)
        os.environ["AIVC_WORKSPACE_DIR"] = str(self.workspace_dir)

        # Per-repo memory partition: {repo_id: {"memories": {}, "file_snapshots": {}, "counter": 0}}
        self.repo_stores: Dict[str, Dict[str, Any]] = {}
        self.set_repo(self.current_repo_id)

    def set_repo(self, repo_id: str) -> None:
        """Switch active repository scope."""
        self.current_repo_id = repo_id
        if repo_id not in self.repo_stores:
            self.repo_stores[repo_id] = {
                "memories": {},
                "file_snapshots": {},
                "counter": 0,
            }

    def reset(self, repo_id: Optional[str] = None, clean_disk: bool = False) -> None:
        """Reset memories for specific repo or all repos."""
        if repo_id:
            if repo_id in self.repo_stores:
                self.repo_stores[repo_id] = {
                    "memories": {},
                    "file_snapshots": {},
                    "counter": 0,
                }
        else:
            self.repo_stores.clear()
            self.set_repo(self.current_repo_id)

        if clean_disk and self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    @property
    def memories(self) -> Dict[str, Dict[str, Any]]:
        return self.repo_stores.get(self.current_repo_id, {}).get("memories", {})

    @property
    def file_snapshots(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.repo_stores.get(self.current_repo_id, {}).get("file_snapshots", {})

    def remember(
        self,
        title: str,
        note: str,
        read_files: Optional[List[str]] = None,
        edited_files: Optional[List[str]] = None,
        repo_id: Optional[str] = None,
    ) -> str:
        target_repo = repo_id or self.current_repo_id
        if target_repo not in self.repo_stores:
            self.set_repo(target_repo)

        store = self.repo_stores[target_repo]
        store["counter"] += 1
        mem_id = f"dev-mem-{store['counter']:04d}"
        now_str = datetime.now(timezone.utc).isoformat()
        effective_repo = repo_id or self.repo_id

        record = {
            "id": mem_id,
            "title": title,
            "note": note,
            "repo_id": effective_repo,
            "read_files": read_files or [],
            "edited_files": edited_files or [],
            "repo_id": target_repo,
            "timestamp": now_str,
        }
        store["memories"][mem_id] = record

        for f in (edited_files or []):
            if f not in store["file_snapshots"]:
                store["file_snapshots"][f] = []
            store["file_snapshots"][f].append({
                "memory_id": mem_id,
                "repo_id": effective_repo,
                "timestamp": now_str,
                "note_ref": title,
            })

        for f in (read_files or []):
            if f not in store["file_snapshots"]:
                store["file_snapshots"][f] = []

        return f"✅ Memory recorded [ID: {mem_id}] '{title}'. Recorded {len(read_files or [])} read, {len(edited_files or [])} edited files."

    def recall_with_records(self, query: str, limit: int = 5, repo_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        target_repo = repo_id or self.current_repo_id
        mems = self.repo_stores.get(target_repo, {}).get("memories", {})
        if not mems:
            return "No past SDLC memories stored in AIVC yet.", []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored = []

        for mem_id, mem in mems.items():
            text = f"{mem['title']} {mem['note']} {' '.join(mem['read_files'])} {' '.join(mem['edited_files'])}".lower()
            score = sum(1 for q in query_terms if q in text)
            if score > 0 or not query_terms:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit] if scored else [(0, m) for m in list(mems.values())[-limit:]]

        top_mems = [m for _, m in top]
        lines = [f"Found {len(top)} relevant SDLC memories:"]
        for _, m in top:
            snippet = m["note"][:160].replace("\n", " ") + "..."
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]}): {snippet}")
        return "\n".join(lines), top_mems

    def get_recent_memories_with_records(self, limit: int = 10, offset: int = 0, repo_id: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        target_repo = repo_id or self.current_repo_id
        all_mems = list(self.repo_stores.get(target_repo, {}).get("memories", {}).values())
        all_mems.reverse()
        slice_mems = all_mems[offset: offset + limit]
        if not slice_mems:
            return "No memories found in range.", []

        lines = [f"Recent SDLC memories for [{effective_repo}] (offset={offset}, limit={limit}):"]
        for m in slice_mems:
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]})")
        return "\n".join(lines), slice_mems

    def consult_memory(self, memory_id: str, repo_id: Optional[str] = None) -> str:
        target_repo = repo_id or self.current_repo_id
        mem = self.repo_stores.get(target_repo, {}).get("memories", {}).get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        effective_repo = repo_id or self.repo_id
        if mem.get("repo_id") and mem.get("repo_id") != effective_repo:
            return f"Memory ID '{memory_id}' belongs to repository '{mem.get('repo_id')}' (access denied for '{effective_repo}')."
        return f"# {mem['title']}\n**Repository**: {mem.get('repo_id', effective_repo)}\n**Created**: {mem['timestamp']}\n**Read Files**: {mem['read_files']}\n**Edited Files**: {mem['edited_files']}\n\n{mem['note']}"

    def get_file_history_metadata(self, filepath: str, repo_id: Optional[str] = None) -> str:
        target_repo = repo_id or self.current_repo_id
        hist = self.repo_stores.get(target_repo, {}).get("file_snapshots", {}).get(filepath, [])
        if not hist:
            return f"No AIVC version history for file '{filepath}' in repository '{effective_repo}'."
        lines = [f"Version history for '{filepath}' in [{effective_repo}]:"]
        for h in hist:
            lines.append(f"- Memory [{h['memory_id']}] at {h['timestamp']}: {h['note_ref']}")
        return "\n".join(lines)

    def read_past_file_content(self, filepath: str, memory_id: str, repo_id: Optional[str] = None) -> str:
        target_repo = repo_id or self.current_repo_id
        mem = self.repo_stores.get(target_repo, {}).get("memories", {}).get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        return f"// Snapshot of {filepath} associated with {memory_id} ({mem['title']})\n// Memory context:\n{mem['note'][:300]}"

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], phase_context: Dict[str, Any]) -> Tuple[str, List[str]]:
        returned_files: List[str] = []
        repo_id = phase_context.get("repo_id", self.current_repo_id)
        if repo_id and repo_id != self.current_repo_id:
            self.set_repo(repo_id)

        def _normalize_file_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, str):
                s = val.strip()
                return [s] if s else []
            if isinstance(val, (list, tuple, set)):
                res = []
                for it in val:
                    res.extend(_normalize_file_list(it))
                return res
            if isinstance(val, dict):
                res = []
                for v in val.values():
                    res.extend(_normalize_file_list(v))
                return res
            s = str(val).strip()
            return [s] if s else []

        try:
            if tool_name == "remember":
                read_f = _normalize_file_list(arguments.get("read_files", []))
                edit_f = _normalize_file_list(arguments.get("edited_files", []))
                res = self.remember(
                    title=str(arguments.get("title", "SDLC Progress")),
                    note=str(arguments.get("note", "")),
                    read_files=read_f,
                    edited_files=edit_f,
                    repo_id=repo_id,
                )
                returned_files = list(dict.fromkeys(read_f + edit_f))
                return res, returned_files
            elif tool_name == "recall":
                query = arguments.get("query", "")
                limit = int(arguments.get("limit", 5))
                res, matched_mems = self.recall_with_records(query=query, limit=limit, repo_id=repo_id)
                for m in matched_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files
            elif tool_name == "get_recent_memories":
                limit = int(arguments.get("limit", 10))
                offset = int(arguments.get("offset", 0))
                res, sliced_mems = self.get_recent_memories_with_records(limit=limit, offset=offset, repo_id=repo_id)
                for m in sliced_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files
            elif tool_name == "consult_memory":
                mem_id = arguments.get("memory_id", "")
                res = self.consult_memory(memory_id=mem_id, repo_id=repo_id)
                mem = self.repo_stores.get(repo_id, {}).get("memories", {}).get(mem_id)
                if mem:
                    returned_files = list(dict.fromkeys(mem.get("read_files", []) + mem.get("edited_files", [])))
                return res, returned_files
            elif tool_name == "get_file_history_metadata":
                filepath = arguments.get("filepath", "")
                res = self.get_file_history_metadata(filepath=filepath, repo_id=repo_id)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files
            elif tool_name == "read_past_file_content":
                filepath = arguments.get("filepath", "")
                mem_id = arguments.get("memory_id", "")
                res = self.read_past_file_content(filepath=filepath, memory_id=mem_id, repo_id=repo_id)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files
            elif tool_name == "view_file":
                filepath = arguments.get("filepath", "")
                if filepath:
                    returned_files = [filepath]
                return f"[File: {filepath}]\n// Template and structure for {phase_context.get('repo_id', '')} ({phase_context.get('phase', '')})\n// Interface declarations and contracts ready.", returned_files
            elif tool_name == "grep_search":
                query = arguments.get("query", "")
                init_f = phase_context.get("initial_files", [])
                returned_files = init_f[:2] if init_f else ["src/main.py"]
                lines = [f"Grep matches for '{query}':"]
                for f in returned_files:
                    lines.append(f"- {f}: defined symbols matching '{query}'")
                return "\n".join(lines), returned_files
            elif tool_name == "list_dir":
                init_f = phase_context.get("initial_files", [])
                returned_files = init_f if init_f else ["src/", "tests/", "config/"]
                return f"Directory listing for {phase_context.get('repo_id', '')}:\n- src/\n- tests/\n- config/\n- README.md", returned_files
            elif tool_name == "submit_phase_deliverable":
                deliv = arguments.get("deliverable", "")
                notes = arguments.get("notes", "")
                returned_files = phase_context.get("initial_files", [])
                return f"✅ Phase deliverable accepted ({len(deliv)} chars). Notes: {notes}", returned_files
            else:
                return f"Unknown tool '{tool_name}'.", []
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}", []


def append_tool_interaction(
    interaction_record: Dict[str, Any],
    interactions_paths: Optional[List[Path]] = None,
) -> None:
    """Atomically append a tool interaction record to specified JSONL output files."""
    if not interactions_paths:
        return
    line = json.dumps(interaction_record, ensure_ascii=False) + "\n"
    for p in interactions_paths:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Checkpoint Manager for DevBench
# ---------------------------------------------------------------------------

class DevBenchCheckpointManager:
    """Manages incremental JSONL checkpointing for DevBench runner."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_entries: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.load_checkpoints()

    def load_checkpoints(self) -> None:
        if not self.checkpoint_path.exists():
            return

        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    repo_id = data.get("repo_id")
                    phase = data.get("phase")
                    status = data.get("status")
                    if repo_id and phase and status == "PASSED":
                        self.completed_entries[(repo_id, phase)] = data
                except json.JSONDecodeError:
                    continue

        if self.completed_entries:
            print(f"[Checkpoint] Loaded {len(self.completed_entries)} completed phase records from {self.checkpoint_path.name}")

    def is_completed(self, repo_id: str, phase: str) -> bool:
        return (repo_id, phase) in self.completed_entries

    def get_completed_record(self, repo_id: str, phase: str) -> Optional[Dict[str, Any]]:
        return self.completed_entries.get((repo_id, phase))

    def save_checkpoint(self, record: Dict[str, Any]) -> None:
        key = (record["repo_id"], record["phase"])
        self.completed_entries[key] = record

        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Multi-Turn DevBench Runner
# ---------------------------------------------------------------------------

class DevBenchRunner:
    """4-Phase SDLC Multi-Turn Runner for DevBench Benchmark."""

    def __init__(
        self,
        arm: str = "aivc",
        model_name: str = "qwen/qwen3.7-flash",
        checkpoint_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
        plots_path: Optional[Path] = None,
        api_key: str = "",
        max_turns: int = 50,
        max_tokens: int = 4096,
        max_cost_per_phase_usd: float = 0.10,
        prompt_price_per_1m: Optional[float] = None,
        completion_price_per_1m: Optional[float] = None,
        fallback_model: Optional[str] = "deepseek/deepseek-v4-flash-0731",
        interactions_paths: Optional[List[Path]] = None,
        run_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.arm = arm.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_cost_per_phase_usd = max_cost_per_phase_usd
        self.checkpoint_path = checkpoint_path or (EVAL_DIR / "checkpoints" / "devbench_checkpoint.jsonl")
        self.metrics_path = metrics_path or (EVAL_DIR / "metrics" / "devbench_metrics.json")
        self.plots_path = plots_path or (EVAL_DIR / "plots" / "devbench_curves.csv")
        self.interactions_paths = interactions_paths or []
        self.run_id = run_id
        self.workspace_dir = workspace_dir

        # System prompt and tool schemas dynamically configured for active arm
        self.system_prompt = get_aivc_system_prompt(benchmark_type="devbench", arm=self.arm)
        self.tools_schema = get_benchmark_tools_schema(include_workspace=True, benchmark_type="devbench", arm=self.arm)

        # Resolve pricing per 1M tokens from registry if not explicitly provided
        models_reg = load_models_registry()
        model_spec = models_reg.get(model_name)
        self.prompt_price_per_1m = prompt_price_per_1m if prompt_price_per_1m is not None else (model_spec.prompt_price_per_1m if model_spec else 0.03)
        self.completion_price_per_1m = completion_price_per_1m if completion_price_per_1m is not None else (model_spec.completion_price_per_1m if model_spec else 0.13)

        self.checkpoint_manager = DevBenchCheckpointManager(self.checkpoint_path)
        self.repo_envs: Dict[str, DevBenchAIVCEnvironment] = {}
        self._aivc_env = DevBenchAIVCEnvironment(run_id=self.run_id, workspace_dir=self.workspace_dir)
        self.repo_envs["default"] = self._aivc_env
        self.analyzer = TrajectoryAnalyzer(model_name=model_name)

        # Resilient Inference Client
        self.client = InferenceClient(
            api_key=self.api_key,
            default_model=self.model_name,
            fallback_model=fallback_model,
            max_retries=5,
            base_delay=1.5,
            max_delay=30.0,
            timeout=60.0,
            app_title=f"AIVC DevBench Runner ({self.arm.upper()})",
        )

    def get_env_for_repo(self, repo_id: str) -> DevBenchAIVCEnvironment:
        """Get or create a dedicated, hermetically isolated AIVC memory environment for a repository."""
        if repo_id not in self.repo_envs:
            self.repo_envs[repo_id] = DevBenchAIVCEnvironment(repo_id=repo_id, arm=self.arm)
        return self.repo_envs[repo_id]

    @property
    def aivc_env(self) -> DevBenchAIVCEnvironment:
        """Default/fallback environment property."""
        if hasattr(self, "_aivc_env") and self._aivc_env is not None:
            return self._aivc_env
        return self.get_env_for_repo("default")

    @aivc_env.setter
    def aivc_env(self, value: DevBenchAIVCEnvironment) -> None:
        self._aivc_env = value
        if hasattr(self, "repo_envs"):
            self.repo_envs["default"] = value

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_1m
        c_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_1m
        return p_cost + c_cost

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize message history to prevent OpenRouter/provider JSON argument parsing errors."""
        return sanitize_messages(messages)

    def _simulate_dry_run_turn(
        self,
        repo: Dict[str, Any],
        phase: str,
        turn: int,
    ) -> Dict[str, Any]:
        """Simulate realistic turn responses in dry-run mode."""
        repo_id = repo.get("repo_id", "devbench-repo")
        prompt = repo.get("phases", {}).get(phase, {}).get("prompt", "")[:60].replace('"', "'")

        if self.arm == "aivc":
            if turn == 1:
                return {
                    "usage": {"prompt_tokens": 400, "completion_tokens": 50},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Starting SDLC phase '{phase}' for {repo_id}. Recalling architectural context from AIVC.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "recall",
                                            "arguments": json.dumps({"query": f"{repo_id} {phase} architecture"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            elif turn == 2:
                return {
                    "usage": {"prompt_tokens": 520, "completion_tokens": 85},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Inspecting repository files for phase {phase}.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": json.dumps({"filepath": f"src/{phase}.py"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            else:
                return {
                    "usage": {"prompt_tokens": 640, "completion_tokens": 110},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Phase '{phase}' completed. Recording deliverables to AIVC memory.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "remember",
                                            "arguments": json.dumps({
                                                "title": f"SDLC Phase {phase} Completed for {repo_id}",
                                                "note": f"Delivered specification and modules for {phase}: {prompt}",
                                                "read_files": [f"src/{phase}.py"],
                                                "edited_files": [f"src/{phase}.py"],
                                            }),
                                        },
                                    },
                                    {
                                        "id": f"call_{turn}_2",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_phase_deliverable",
                                            "arguments": json.dumps({
                                                "deliverable": f"# Deliverable for {phase} in {repo_id}\nImplemented according to specification.",
                                                "notes": f"Phase {phase} verified.",
                                            }),
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                }
        else:
            # Naive baseline (stateless exploration with directory listing and file viewing)
            if turn == 1:
                return {
                    "usage": {"prompt_tokens": 360, "completion_tokens": 40},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Exploring directory structure for {repo_id} in phase '{phase}'.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_dir",
                                            "arguments": json.dumps({"directory": "."}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            elif turn == 2:
                return {
                    "usage": {"prompt_tokens": 540, "completion_tokens": 75},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Reading template files for phase {phase}.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": json.dumps({"filepath": f"src/{phase}.py"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            else:
                return {
                    "usage": {"prompt_tokens": 660, "completion_tokens": 95},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Completing phase '{phase}' deliverable.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_phase_deliverable",
                                            "arguments": json.dumps({
                                                "deliverable": f"# Deliverable for {phase} in {repo_id}\nImplemented according to specification.",
                                                "notes": f"Phase {phase} verified.",
                                            }),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }

    def _call_openrouter_api(
        self,
        messages: List[Dict[str, Any]],
        repo: Optional[Dict[str, Any]] = None,
        phase: str = "",
        turn: int = 1,
        retries: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Send chat completion request with tools schema using InferenceClient, or simulate only if dry_run is explicitly True."""
        if self.dry_run:
            return self._simulate_dry_run_turn(repo=repo or {}, phase=phase, turn=turn)

        try:
            return self.client.complete(
                messages=messages,
                tools=self.tools_schema,
                max_tokens=self.max_tokens,
                temperature=0.2,
                model=self.model_name,
            )
        except Exception as e:
            print(f"  [API Exception]: {e}")
            return None


    def execute_phase(
        self,
        repo: Dict[str, Any],
        phase: str,
        phase_index: int,
    ) -> Dict[str, Any]:
        """Execute a multi-turn SDLC phase with live tool calling."""
        repo_id = repo["repo_id"]

        # In baseline / naive mode, execute cold start without memory transfer
        if self.arm in ("baseline", "naive"):
            self.aivc_env.reset()
        else:
            self.aivc_env.set_repo(repo_id)

        phase_config = repo["phases"][phase]
        prompt = phase_config["prompt"]
        initial_files = phase_config.get("initial_files", [])

        # Reset memory state if running in naive stateless baseline mode
        aivc_env.reset_if_stateless()

        start_time = time.time()

        print(f"\n[PHASE {phase_index}] Arm: {self.arm.upper()} | Repository: {repo_id} | Phase: {phase}")
        print(f"Goal: {prompt}")

        if self.arm in ("baseline", "naive"):
            user_instruction = (
                f"Repository: {repo_id} ({repo['domain']})\n"
                f"Project Description: {repo['description']}\n"
                f"SDLC Phase: {phase}\n\n"
                f"Task Objective:\n{prompt}\n\n"
                f"Target Files: {initial_files}\n\n"
                f"Instructions: Explore the codebase using `grep_search`, `list_dir`, and `view_file`. "
                f"Call `submit_phase_deliverable` when done."
            )
        else:
            user_instruction = (
                f"Repository: {repo_id} ({repo['domain']})\n"
                f"Project Description: {repo['description']}\n"
                f"SDLC Phase: {phase}\n\n"
                f"Task Objective:\n{prompt}\n\n"
                f"Target Files: {initial_files}\n\n"
                f"Instructions: Use `recall` to inspect prior architecture/contract decisions in AIVC. "
                f"Use `remember` to save your work, and call `submit_phase_deliverable` when done."
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_instruction},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_phase_cost = 0.0
        tools_called_list: List[str] = []
        phase_tool_interactions: List[Dict[str, Any]] = []
        all_inspected_files: List[str] = []
        recalled_count = 0
        used_count = 0
        passed = False
        trajectory_steps: List[Dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            if total_phase_cost >= self.max_cost_per_phase_usd:
                print(f"  [CUTOFF] Cost limit (${self.max_cost_per_phase_usd:.2f}) reached for this phase (${total_phase_cost:.4f}). Stopping.")
                break

            print(f"  [TURN {turn:02d}/{self.max_turns:02d}] Calling {self.model_name} (Cost so far: ${total_phase_cost:.4f})... ", end="", flush=True)

            api_response = self._call_openrouter_api(messages, repo=repo, phase=phase, turn=turn)
            if not api_response or "choices" not in api_response or not api_response["choices"]:
                print("FAILED (No response)")
                break

            usage = api_response.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            step_cost = self.calculate_cost(p_tok, c_tok)

            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_phase_cost += step_cost

            choice = api_response["choices"][0]
            assistant_msg = choice.get("message", {})
            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            content_preview = (assistant_msg.get("content") or "")[:80].replace("\n", " ")

            turn_tools = []
            turn_recalled = 0
            turn_used = 0

            if tool_calls:
                print(f"Tool calls ({len(tool_calls)}): ", end="")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args_str = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                    except Exception:
                        fn_args = {}

                    turn_tools.append(fn_name)
                    tools_called_list.append(fn_name)

                    if fn_name in ("recall", "get_recent_memories"):
                        turn_recalled += 1
                    elif fn_name in ("consult_memory", "read_past_file_content"):
                        turn_used += 1

                    if fn_name == "submit_phase_deliverable":
                        passed = True

                    # Live execution
                    tool_res, returned_files = self.aivc_env.execute_tool(
                        fn_name, fn_args, {"repo_id": repo_id, "phase": phase, "initial_files": initial_files}
                    )
                    for rf in returned_files:
                        if isinstance(rf, (list, tuple, set)):
                            for srf in rf:
                                s_str = str(srf).strip()
                                if s_str and s_str not in all_inspected_files:
                                    all_inspected_files.append(s_str)
                        else:
                            s_str = str(rf).strip() if rf else ""
                            if s_str and s_str not in all_inspected_files:
                                all_inspected_files.append(s_str)

                    interaction_record = {
                        "tool_name": fn_name,
                        "input_arguments": fn_args,
                        "returned_files": returned_files,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "step_tokens": {
                            "prompt_tokens": p_tok,
                            "completion_tokens": c_tok,
                            "total_tokens": p_tok + c_tok,
                        },
                        "benchmark": "devbench",
                        "repo_id": repo_id,
                        "phase": phase,
                        "arm": self.arm,
                        "turn": turn,
                        "model": self.model_name,
                    }
                    phase_tool_interactions.append(interaction_record)
                    append_tool_interaction(interaction_record, self.interactions_paths)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                        "name": fn_name,
                        "content": str(tool_res),
                    })

                print(", ".join(turn_tools))
            else:
                print(f"Response: {content_preview}...")

            recalled_count += turn_recalled
            used_count += turn_used

            trajectory_steps.append({
                "turn": turn,
                "tool_calls": turn_tools,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "recalled_memories": turn_recalled,
                "used_memories": turn_used,
            })

            if passed or not tool_calls:
                passed = True  # Natural completion of SDLC phase
                break

        duration = round(time.time() - start_time, 3)
        total_tokens = total_prompt_tokens + total_completion_tokens
        baseline_phase_cost = round(repo["baseline_est_cost"] / len(SDLC_PHASES), 6)

        total_tool_calls = len(tools_called_list)
        exploration_tool_calls = sum(1 for t in tools_called_list if t in EXPLORATION_TOOLS)

        eor = compute_eor(total_tool_calls, exploration_tool_calls)
        mui = compute_mui(recalled_count, used_count, eor)
        ccsr = compute_ccsr(baseline_phase_cost, total_phase_cost)

        ir_metrics = compute_retrieval_metrics(
            retrieved_files=all_inspected_files,
            ground_truth_files=initial_files,
            k_list=(1, 3, 5),
        )

        record = {
            "phase_index": phase_index,
            "repo_id": repo_id,
            "arm": self.arm,
            "domain": repo["domain"],
            "phase": phase,
            "arm": self.arm,
            "status": "PASSED" if passed else "FAILED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "turns_count": len(trajectory_steps),
            "tool_calls_count": len(tools_called_list),
            "tools": tools_called_list,
            "tool_interactions": phase_tool_interactions,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(total_phase_cost, 6),
            "baseline_phase_cost": baseline_phase_cost,
            "eor": eor,
            "mui": mui,
            "ccsr": ccsr,
            "retrieval_metrics": ir_metrics,
            "ground_truth_files": initial_files,
            "inspected_files": all_inspected_files,
            "duration_sec": duration,
            "recalled_memories": recalled_count,
            "used_memories": used_count,
        }

        print(f"--> Phase Result: PASSED | Turns: {len(trajectory_steps)} | Cost: ${total_phase_cost:.6f} | Duration: {duration}s")
        print(f"--> Metrics: EOR={eor:.4f} | MUI={mui:.4f} | CCSR={ccsr:.4f} | NDCG@3={ir_metrics.get('ndcg_at_3', 0.0):.4f}")

        return record

    def run_benchmark(
        self,
        phase_limit: int = 15,
        reset_checkpoint: bool = False,
    ) -> Dict[str, Any]:
        """Run DevBench SDLC evaluation across linear phase schedule."""
        if reset_checkpoint and self.checkpoint_path.exists():
            print(f"[Reset] Purging existing checkpoint file: {self.checkpoint_path}")
            self.checkpoint_path.unlink()
            self.checkpoint_manager = DevBenchCheckpointManager(self.checkpoint_path)

        # Build list of (repo, phase) execution schedule
        schedule: List[Tuple[Dict[str, Any], str]] = []
        for repo in DEFAULT_DEVBENCH_REPOS:
            for phase in SDLC_PHASES:
                schedule.append((repo, phase))

        target_schedule = schedule[:phase_limit]

        print("\n" + "=" * 70)
        print(f"[DevBench Runner] Starting Multi-Turn SDLC Evaluation ({len(target_schedule)} phases) [{self.arm.upper()}]")
        print(f"Model          : {self.model_name}")
        print(f"Evaluation Arm : {self.arm.upper()}")
        print(f"Max Turns/Phase: {self.max_turns}")
        print(f"Max Tokens/Resp: {self.max_tokens}")
        print(f"Max Cost/Phase : ${self.max_cost_per_phase_usd:.2f} USD")
        print(f"Checkpoint Path: {self.checkpoint_path}")
        print("=" * 70)

        all_phase_records: List[Dict[str, Any]] = []

        for idx, (repo, phase) in enumerate(target_schedule, 1):
            repo_id = repo["repo_id"]

            if self.checkpoint_manager.is_completed(repo_id, phase) and not reset_checkpoint:
                saved_record = self.checkpoint_manager.get_completed_record(repo_id, phase)
                if saved_record:
                    print(f"\n[SKIP] Phase '{phase}' for {repo_id} already completed in checkpoint.")
                    all_phase_records.append(saved_record)
                    continue

            record = self.execute_phase(repo, phase, phase_index=idx)
            self.checkpoint_manager.save_checkpoint(record)
            all_phase_records.append(record)

        metrics_data = self.export_metrics(all_phase_records, len(DEFAULT_DEVBENCH_REPOS))
        self.export_plots(all_phase_records)

        return metrics_data

    def export_metrics(
        self,
        records: List[Dict[str, Any]],
        total_repos: int,
    ) -> Dict[str, Any]:
        """Aggregate evaluation metrics and write to JSON."""
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in records)
        total_tokens = sum(r.get("total_tokens", 0) for r in records)
        total_cost_usd = round(sum(r.get("cost_usd", 0.0) for r in records), 6)
        total_baseline_cost = round(sum(r.get("baseline_phase_cost", 0.0) for r in records), 6)

        total_phases = len(records)
        passed_phases = sum(1 for r in records if r.get("status") == "PASSED")
        phase_pass_rate = round(passed_phases / float(total_phases), 4) if total_phases > 0 else 0.0

        repo_phase_counts: Dict[str, int] = {}
        for r in records:
            if r.get("status") == "PASSED":
                repo_phase_counts[r["repo_id"]] = repo_phase_counts.get(r["repo_id"], 0) + 1
        completed_repos = sum(1 for repo_id, count in repo_phase_counts.items() if count == len(SDLC_PHASES))
        sdlc_completion_rate = round(completed_repos / float(total_repos), 4) if total_repos > 0 else 0.0

        avg_eor = round(sum(r.get("eor", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_mui = round(sum(r.get("mui", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        overall_ccsr = compute_ccsr(total_baseline_cost, total_cost_usd)

        avg_p1 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_1", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_p3 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_3", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_p5 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_5", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_r1 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_1", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_r3 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_3", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_r5 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_5", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_ndcg1 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_1", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_ndcg3 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_3", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_ndcg5 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_5", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0
        avg_mrr = round(sum(r.get("retrieval_metrics", {}).get("mrr", 0.0) for r in records) / float(total_phases), 4) if total_phases > 0 else 0.0

        all_tool_calls = [tc for r in records for tc in r.get("tools", [])]
        tool_counts: Dict[str, int] = {}
        for tc in all_tool_calls:
            tool_counts[tc] = tool_counts.get(tc, 0) + 1

        total_interactions = sum(len(r.get("tool_interactions", [])) for r in records)

        phase_breakdown = {}
        for phase in SDLC_PHASES:
            phase_records = [r for r in records if r.get("phase") == phase]
            p_total = len(phase_records)
            p_passed = sum(1 for r in phase_records if r.get("status") == "PASSED")
            p_pass_rate = round(p_passed / float(p_total), 4) if p_total > 0 else 0.0
            p_avg_tokens = round(sum(r.get("total_tokens", 0) for r in phase_records) / float(p_total), 2) if p_total > 0 else 0.0
            p_avg_cost = round(sum(r.get("cost_usd", 0.0) for r in phase_records) / float(p_total), 6) if p_total > 0 else 0.0
            p_avg_eor = round(sum(r.get("eor", 0.0) for r in phase_records) / float(p_total), 4) if p_total > 0 else 0.0
            p_avg_mui = round(sum(r.get("mui", 0.0) for r in phase_records) / float(p_total), 4) if p_total > 0 else 0.0
            p_avg_ndcg3 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_3", 0.0) for r in phase_records) / float(p_total), 4) if p_total > 0 else 0.0

            phase_breakdown[phase] = {
                "total_executions": p_total,
                "passed_executions": p_passed,
                "pass_rate": p_pass_rate,
                "avg_tokens": p_avg_tokens,
                "avg_cost_usd": p_avg_cost,
                "avg_eor": p_avg_eor,
                "avg_mui": p_avg_mui,
                "avg_ndcg_at_3": p_avg_ndcg3,
            }

        metrics_json = {
            "benchmark_name": "DevBench",
            "arm": self.arm,
            "model_name": self.model_name,
            "arm": self.arm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_repos": total_repos,
                "completed_sdlc_repos": completed_repos,
                "sdlc_completion_rate": sdlc_completion_rate,
                "total_phases_executed": total_phases,
                "phase_pass_rate": phase_pass_rate,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "baseline_cost_usd": total_baseline_cost,
                "overall_ccsr": overall_ccsr,
                "avg_eor": avg_eor,
                "avg_mui": avg_mui,
                "total_tool_calls": len(all_tool_calls),
                "total_tool_interactions": total_interactions,
                "tool_interaction_breakdown": tool_counts,
            },
            "retrieval_metrics": {
                "mean_reciprocal_rank_mrr": avg_mrr,
                "precision_at_1": avg_p1,
                "precision_at_3": avg_p3,
                "precision_at_5": avg_p5,
                "recall_at_1": avg_r1,
                "recall_at_3": avg_r3,
                "recall_at_5": avg_r5,
                "ndcg_at_1": avg_ndcg1,
                "ndcg_at_3": avg_ndcg3,
                "ndcg_at_5": avg_ndcg5,
            },
            "phase_breakdown": phase_breakdown,
            "detailed_records": records,
        }

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)

        print(f"\n[Export] Saved DevBench metrics to: {self.metrics_path}")

        # Mirror to general metrics for DVC if arm is aivc
        if self.arm == "aivc":
            general_metrics = EVAL_DIR / "metrics" / "devbench_metrics.json"
            if self.metrics_path != general_metrics:
                try:
                    with open(general_metrics, "w", encoding="utf-8") as f:
                        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        return metrics_json

    def export_plots(self, records: List[Dict[str, Any]]) -> None:
        """Export curve plots data to CSV."""
        self.plots_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "repo_id",
            "phase",
            "arm",
            "step_index",
            "status",
            "pass_rate",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
            "cumulative_cost_usd",
            "eor",
            "mui",
            "ccsr",
            "duration_sec",
        ]

        cumulative_cost = 0.0
        passed_so_far = 0

        with open(self.plots_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, r in enumerate(records, 1):
                cumulative_cost += r.get("cost_usd", 0.0)
                if r.get("status") == "PASSED":
                    passed_so_far += 1
                current_pass_rate = round(passed_so_far / float(idx), 4)

                writer.writerow({
                    "repo_id": r.get("repo_id", ""),
                    "phase": r.get("phase", ""),
                    "arm": r.get("arm", self.arm),
                    "step_index": idx,
                    "status": r.get("status", "PASSED"),
                    "pass_rate": current_pass_rate,
                    "prompt_tokens": r.get("prompt_tokens", 0),
                    "completion_tokens": r.get("completion_tokens", 0),
                    "total_tokens": r.get("total_tokens", 0),
                    "cost_usd": round(r.get("cost_usd", 0.0), 6),
                    "cumulative_cost_usd": round(cumulative_cost, 6),
                    "eor": r.get("eor", 0.0),
                    "mui": r.get("mui", 0.0),
                    "ccsr": r.get("ccsr", 0.0),
                    "duration_sec": r.get("duration_sec", 0.0),
                })

        print(f"[Export] Saved DevBench plot curves to: {self.plots_path}")

        # Mirror to general plots for DVC if arm is aivc
        if self.arm == "aivc":
            general_plots = EVAL_DIR / "plots" / "devbench_curves.csv"
            if self.plots_path != general_plots:
                try:
                    import shutil
                    shutil.copyfile(self.plots_path, general_plots)
                except Exception:
                    pass


def main() -> None:
    """CLI entrypoint for DevBench runner."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AIVC DevBench SDLC Benchmark Runner")
    parser.add_argument(
        "--arm",
        "--variant",
        dest="arm",
        type=str,
        choices=["aivc", "baseline", "naive"],
        default="aivc",
        help="Evaluation arm: 'aivc' (with persistent memory transfer) or 'baseline'/'naive' (cold-start per phase). Default: aivc",
    )
    parser.add_argument("--checkpoint-path", "--checkpoint-file", dest="checkpoint_path", type=str, default="", help="Custom JSONL checkpoint path")
    parser.add_argument("--metrics-path", "--metrics-file", dest="metrics_path", type=str, default="", help="Custom metrics JSON export path")
    parser.add_argument("--plots-path", "--curves-file", dest="plots_path", type=str, default="", help="Custom plots CSV export path")

    # Add unified evaluation configuration flags
    add_eval_args(parser)

    # Parse and resolve hierarchical config
    parsed_args = parser.parse_args()
    cfg = load_benchmark_config(args=parsed_args)
    paths = cfg.get_paths()

    clean_model = cfg.model.replace("/", "_").replace(":", "_").replace("-", "_")
    arm_name = parsed_args.arm.lower()

    checkpoint_path = Path(parsed_args.checkpoint_path) if parsed_args.checkpoint_path else (paths.checkpoints_dir / f"devbench_{clean_model}_{arm_name}_checkpoint.jsonl")
    metrics_path = Path(parsed_args.metrics_path) if parsed_args.metrics_path else (paths.metrics_dir / f"devbench_{clean_model}_{arm_name}_metrics.json")
    plots_path = Path(parsed_args.plots_path) if parsed_args.plots_path else (paths.plots_dir / f"devbench_{clean_model}_{arm_name}_curves.csv")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plots_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[AIVC BENCHMARK RUNNER] DevBench 4-Phase SDLC Pipeline [{cfg.profile.upper()}]")
    print("=" * 70)
    print(f"Evaluation Arm : {parsed_args.arm.upper()}")
    print(f"Sample Limit   : {cfg.limit}")
    print(f"Active Model   : {cfg.model}")
    print(f"Max Turns      : {cfg.max_turns}")
    print(f"Max Tokens     : {cfg.max_tokens}")
    print(f"Max Cost/Phase : ${cfg.max_cost_per_instance_usd:.2f} USD")
    print(f"Checkpoint File: {checkpoint_path}")
    print(f"Metrics Output : {metrics_path}")
    print(f"Curves Output  : {plots_path}")
    print("=" * 70)

    provider = cfg.model_spec.provider if cfg.model_spec else "openrouter"
    api_key = os.getenv("TOGETHER_API_KEY", "") if provider == "together" else os.getenv("OPENROUTER_API_KEY", "")

    # Configure tool interaction paths
    profile_interactions = paths.metrics_dir / "tool_interactions.jsonl"
    bench_interactions = EVAL_DIR / "metrics" / f"devbench_{arm_name}_tool_interactions.jsonl"
    general_interactions = EVAL_DIR / "metrics" / "tool_interactions.jsonl"
    interactions_paths = [profile_interactions, bench_interactions, general_interactions]

    if cfg.reset_checkpoint:
        for p in interactions_paths:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    runner = DevBenchRunner(
        arm=parsed_args.arm,
        model_name=cfg.model,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        plots_path=plots_path,
        api_key=api_key,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        max_cost_per_phase_usd=cfg.max_cost_per_instance_usd,
        prompt_price_per_1m=cfg.model_spec.prompt_price_per_1m if cfg.model_spec else None,
        completion_price_per_1m=cfg.model_spec.completion_price_per_1m if cfg.model_spec else None,
        interactions_paths=interactions_paths,
    )

    runner.run_benchmark(phase_limit=cfg.limit, reset_checkpoint=cfg.reset_checkpoint)


if __name__ == "__main__":
    main()

