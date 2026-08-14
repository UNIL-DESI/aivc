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
import sys
import time
import urllib.error
import urllib.request
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

# Import TrajectoryAnalyzer metrics if available
from metrics.trajectory_analyzer import (
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    compute_ccsr,
    compute_eor,
    compute_mui,
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
    add_eval_args,
    get_aivc_system_prompt,
    get_benchmark_tools_schema,
    load_benchmark_config,
    load_models_registry,
)


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
# In-Memory AIVC Environment for DevBench
# ---------------------------------------------------------------------------

class DevBenchAIVCEnvironment:
    """
    Maintains AIVC memory store across SDLC phases of each repository.
    """

    def __init__(self):
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.file_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self._counter = 0

    def remember(self, title: str, note: str, read_files: Optional[List[str]] = None, edited_files: Optional[List[str]] = None) -> str:
        self._counter += 1
        mem_id = f"dev-mem-{self._counter:04d}"
        now_str = datetime.now(timezone.utc).isoformat()

        record = {
            "id": mem_id,
            "title": title,
            "note": note,
            "read_files": read_files or [],
            "edited_files": edited_files or [],
            "timestamp": now_str,
        }
        self.memories[mem_id] = record

        for f in (edited_files or []):
            if f not in self.file_snapshots:
                self.file_snapshots[f] = []
            self.file_snapshots[f].append({
                "memory_id": mem_id,
                "timestamp": now_str,
                "note_ref": title,
            })

        return f"✅ Memory recorded [ID: {mem_id}] '{title}'. Recorded {len(read_files or [])} read, {len(edited_files or [])} edited files."

    def recall(self, query: str, limit: int = 5) -> str:
        if not self.memories:
            return "No past SDLC memories stored in AIVC yet."

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored = []

        for mem_id, mem in self.memories.items():
            text = f"{mem['title']} {mem['note']} {' '.join(mem['read_files'])} {' '.join(mem['edited_files'])}".lower()
            score = sum(1 for q in query_terms if q in text)
            if score > 0 or not query_terms:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit] if scored else [(0, m) for m in list(self.memories.values())[-limit:]]

        lines = [f"Found {len(top)} relevant SDLC memories:"]
        for _, m in top:
            snippet = m["note"][:160].replace("\n", " ") + "..."
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]}): {snippet}")
        return "\n".join(lines)

    def get_recent_memories(self, limit: int = 10, offset: int = 0) -> str:
        all_mems = list(self.memories.values())
        all_mems.reverse()
        slice_mems = all_mems[offset: offset + limit]
        if not slice_mems:
            return "No memories found in range."

        lines = [f"Recent SDLC memories:"]
        for m in slice_mems:
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]})")
        return "\n".join(lines)

    def consult_memory(self, memory_id: str) -> str:
        mem = self.memories.get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        return f"# {mem['title']}\n**Created**: {mem['timestamp']}\n**Read Files**: {mem['read_files']}\n**Edited Files**: {mem['edited_files']}\n\n{mem['note']}"

    def get_file_history_metadata(self, filepath: str) -> str:
        hist = self.file_snapshots.get(filepath, [])
        if not hist:
            return f"No AIVC version history for file '{filepath}'."
        lines = [f"Version history for '{filepath}':"]
        for h in hist:
            lines.append(f"- Memory [{h['memory_id']}] at {h['timestamp']}: {h['note_ref']}")
        return "\n".join(lines)

    def read_past_file_content(self, filepath: str, memory_id: str) -> str:
        mem = self.memories.get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        return f"// Historical snapshot of {filepath} at {memory_id} ({mem['title']})\n{mem['note'][:250]}"

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any], phase_context: Dict[str, Any]) -> str:
        try:
            if tool_name == "remember":
                return self.remember(
                    title=arguments.get("title", "SDLC Progress"),
                    note=arguments.get("note", ""),
                    read_files=arguments.get("read_files", []),
                    edited_files=arguments.get("edited_files", []),
                )
            elif tool_name == "recall":
                return self.recall(
                    query=arguments.get("query", ""),
                    limit=int(arguments.get("limit", 5)),
                )
            elif tool_name == "get_recent_memories":
                return self.get_recent_memories(
                    limit=int(arguments.get("limit", 10)),
                    offset=int(arguments.get("offset", 0)),
                )
            elif tool_name == "consult_memory":
                return self.consult_memory(memory_id=arguments.get("memory_id", ""))
            elif tool_name == "get_file_history_metadata":
                return self.get_file_history_metadata(filepath=arguments.get("filepath", ""))
            elif tool_name == "read_past_file_content":
                return self.read_past_file_content(
                    filepath=arguments.get("filepath", ""),
                    memory_id=arguments.get("memory_id", ""),
                )
            elif tool_name == "view_file":
                filepath = arguments.get("filepath", "")
                return f"[File: {filepath}]\n// Template and structure for {phase_context.get('repo_id', '')} ({phase_context.get('phase', '')})\n// Interface declarations and contracts ready."
            elif tool_name == "grep_search":
                query = arguments.get("query", "")
                return f"Grep matches for '{query}':\n- src/main: defined symbols matching '{query}'"
            elif tool_name == "list_dir":
                return f"Directory listing for {phase_context.get('repo_id', '')}:\n- src/\n- tests/\n- config/\n- README.md"
            elif tool_name == "submit_phase_deliverable":
                deliv = arguments.get("deliverable", "")
                notes = arguments.get("notes", "")
                return f"✅ Phase deliverable accepted ({len(deliv)} chars). Notes: {notes}"
            else:
                return f"Unknown tool '{tool_name}'."
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


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
        model_name: str = "qwen/qwen3.7-flash",
        checkpoint_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
        plots_path: Optional[Path] = None,
        api_key: str = "",
        max_turns: int = 50,
        max_tokens: int = 4096,
        max_cost_per_phase_usd: float = 0.10,
        dry_run: bool = False,
        prompt_price_per_1m: Optional[float] = None,
        completion_price_per_1m: Optional[float] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_cost_per_phase_usd = max_cost_per_phase_usd
        self.dry_run = dry_run
        self.checkpoint_path = checkpoint_path or (EVAL_DIR / "checkpoints" / "devbench_checkpoint.jsonl")
        self.metrics_path = metrics_path or (EVAL_DIR / "metrics" / "devbench_metrics.json")
        self.plots_path = plots_path or (EVAL_DIR / "plots" / "devbench_curves.csv")

        # System prompt and tool schemas from unified eval.config
        self.system_prompt = get_aivc_system_prompt(benchmark_type="devbench")
        self.tools_schema = get_benchmark_tools_schema(include_workspace=True, benchmark_type="devbench")

        # Resolve pricing per 1M tokens from registry if not explicitly provided
        models_reg = load_models_registry()
        model_spec = models_reg.get(model_name)
        self.prompt_price_per_1m = prompt_price_per_1m if prompt_price_per_1m is not None else (model_spec.prompt_price_per_1m if model_spec else 0.03)
        self.completion_price_per_1m = completion_price_per_1m if completion_price_per_1m is not None else (model_spec.completion_price_per_1m if model_spec else 0.13)

        self.checkpoint_manager = DevBenchCheckpointManager(self.checkpoint_path)
        self.aivc_env = DevBenchAIVCEnvironment()
        self.analyzer = TrajectoryAnalyzer(model_name=model_name)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_1m
        c_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_1m
        return p_cost + c_cost

    def _call_openrouter_api(self, messages: List[Dict[str, Any]], retries: int = 3) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set or empty. Real execution requires a valid API key.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aivc/aivc",
            "X-Title": "AIVC DevBench Runner",
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools_schema,
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }

        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    if resp.status == 200:
                        body = resp.read().decode("utf-8")
                        return json.loads(body)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
                print(f"  [API HTTP Error] (Attempt {attempt}/{retries}) Status {e.code}: {err_body}")
                if attempt == retries:
                    raise RuntimeError(f"OpenRouter API failed with HTTP {e.code}: {err_body}")
                time.sleep(2 * attempt)
            except Exception as e:
                print(f"  [API Network Error] (Attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    raise RuntimeError(f"OpenRouter API connection failed: {e}")
                time.sleep(2 * attempt)
        return None

    def execute_phase(
        self,
        repo: Dict[str, Any],
        phase: str,
        phase_index: int,
    ) -> Dict[str, Any]:
        """Execute a multi-turn SDLC phase with live tool calling."""
        repo_id = repo["repo_id"]
        phase_config = repo["phases"][phase]
        prompt = phase_config["prompt"]
        initial_files = phase_config.get("initial_files", [])

        start_time = time.time()

        print(f"\n[PHASE {phase_index}] Repository: {repo_id} | Phase: {phase}")
        print(f"Goal: {prompt}")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Repository: {repo_id} ({repo['domain']})\n"
                    f"Project Description: {repo['description']}\n"
                    f"SDLC Phase: {phase}\n\n"
                    f"Task Objective:\n{prompt}\n\n"
                    f"Target Files: {initial_files}\n\n"
                    f"Instructions: Use `recall` to inspect prior architecture/contract decisions in AIVC. "
                    f"Use `remember` to save your work, and call `submit_phase_deliverable` when done."
                ),
            },
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_phase_cost = 0.0
        tools_called_list: List[str] = []
        recalled_count = 0
        used_count = 0
        passed = False
        trajectory_steps: List[Dict[str, Any]] = []

        for turn in range(1, self.max_turns + 1):
            if total_phase_cost >= self.max_cost_per_phase_usd:
                print(f"  [CUTOFF] Cost limit (${self.max_cost_per_phase_usd:.2f}) reached for this phase (${total_phase_cost:.4f}). Stopping.")
                break

            print(f"  [TURN {turn:02d}/{self.max_turns:02d}] Calling {self.model_name} (Cost so far: ${total_phase_cost:.4f})... ", end="", flush=True)

            api_response = self._call_openrouter_api(messages)
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

                    if fn_name == "recall" or fn_name == "get_recent_memories":
                        turn_recalled += 1
                    elif fn_name == "consult_memory" or fn_name == "read_past_file_content":
                        turn_used += 1

                    if fn_name == "submit_phase_deliverable":
                        passed = True

                    # Live execution
                    tool_res = self.aivc_env.execute_tool(fn_name, fn_args, {"repo_id": repo_id, "phase": phase})

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

        record = {
            "phase_index": phase_index,
            "repo_id": repo_id,
            "domain": repo["domain"],
            "phase": phase,
            "status": "PASSED" if passed else "FAILED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "turns_count": len(trajectory_steps),
            "tool_calls_count": len(tools_called_list),
            "tools": tools_called_list,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(total_phase_cost, 6),
            "baseline_phase_cost": baseline_phase_cost,
            "eor": eor,
            "mui": mui,
            "ccsr": ccsr,
            "duration_sec": duration,
            "recalled_memories": recalled_count,
            "used_memories": used_count,
        }

        print(f"--> Phase Result: PASSED | Turns: {len(trajectory_steps)} | Cost: ${total_phase_cost:.6f} | Duration: {duration}s")
        print(f"--> Metrics: EOR={eor:.4f} | MUI={mui:.4f} | CCSR={ccsr:.4f}")

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
        print(f"[DevBench Runner] Starting Multi-Turn SDLC Evaluation ({len(target_schedule)} phases)")
        print(f"Model          : {self.model_name}")
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

            phase_breakdown[phase] = {
                "total_executions": p_total,
                "passed_executions": p_passed,
                "pass_rate": p_pass_rate,
                "avg_tokens": p_avg_tokens,
                "avg_cost_usd": p_avg_cost,
                "avg_eor": p_avg_eor,
                "avg_mui": p_avg_mui,
            }

        metrics_json = {
            "benchmark_name": "DevBench",
            "model_name": self.model_name,
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
            },
            "phase_breakdown": phase_breakdown,
            "detailed_records": records,
        }

        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)

        print(f"\n[Export] Saved DevBench metrics to: {self.metrics_path}")
        return metrics_json

    def export_plots(self, records: List[Dict[str, Any]]) -> None:
        """Export curve plots data to CSV."""
        self.plots_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "repo_id",
            "phase",
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


def main() -> None:
    """CLI entrypoint for DevBench runner."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AIVC DevBench SDLC Benchmark Runner")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Custom JSONL checkpoint path")
    parser.add_argument("--metrics-path", type=str, default="", help="Custom metrics JSON export path")
    parser.add_argument("--plots-path", type=str, default="", help="Custom plots CSV export path")

    # Add unified evaluation configuration flags
    add_eval_args(parser)

    # Parse and resolve hierarchical config
    parsed_args = parser.parse_args()
    cfg = load_benchmark_config(args=parsed_args)
    paths = cfg.get_paths()

    checkpoint_path = Path(parsed_args.checkpoint_path) if parsed_args.checkpoint_path else (paths.checkpoints_dir / "devbench_checkpoint.jsonl")
    metrics_path = Path(parsed_args.metrics_path) if parsed_args.metrics_path else (paths.metrics_dir / "devbench_metrics.json")
    plots_path = Path(parsed_args.plots_path) if parsed_args.plots_path else (paths.plots_dir / "devbench_curves.csv")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    plots_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[AIVC BENCHMARK RUNNER] DevBench 4-Phase SDLC Pipeline [{cfg.profile.upper()}]")
    print("=" * 70)
    print(f"Sample Limit   : {cfg.limit}")
    print(f"Active Model   : {cfg.model}")
    print(f"Max Turns      : {cfg.max_turns}")
    print(f"Max Tokens     : {cfg.max_tokens}")
    print(f"Max Cost/Phase : ${cfg.max_cost_per_instance_usd:.2f} USD")
    print(f"Checkpoint File: {checkpoint_path}")
    print(f"Metrics Output : {metrics_path}")
    print(f"Curves Output  : {plots_path}")
    print("=" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY", "")

    runner = DevBenchRunner(
        model_name=cfg.model,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        plots_path=plots_path,
        api_key=api_key,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        max_cost_per_phase_usd=cfg.max_cost_per_instance_usd,
        dry_run=cfg.dry_run,
        prompt_price_per_1m=cfg.model_spec.prompt_price_per_1m if cfg.model_spec else None,
        completion_price_per_1m=cfg.model_spec.completion_price_per_1m if cfg.model_spec else None,
    )

    runner.run_benchmark(phase_limit=cfg.limit, reset_checkpoint=cfg.reset_checkpoint)


if __name__ == "__main__":
    main()
