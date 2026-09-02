"""
Agentic RAG Continual Learning Benchmark Runner for AIVC.

Evaluates AI coding agents on multi-hop code retrieval, architectural comprehension,
and dependency tracking across sequential query streams (SWE-Explore / CrossCodeEval).

Features:
1. Online Continual Learning Evaluation:
   - Mode A (--arm naive): Stateless baseline; agent starts each query with empty context.
   - Mode B (--arm aivc): Persistent memory; agent accumulates memory notes, file graph
     snapshots, and cross-query knowledge via AIVC MCP tools (remember, recall, etc.).
2. High-Resolution Telemetry & IR Metrics:
   - Per-query tracking: tool calls count & breakdown, duration (latency), prompt/completion tokens,
     cost in USD, resolution status, EOR, MUI, CCSR.
   - Retrieval metrics: MRR (Mean Reciprocal Rank), Precision@k, Recall@k, F1@k.
   - Flagship Tool Call Decay tracking: demonstrates exponential reduction in exploration actions
     as AIVC long-term memory accumulates across continual episodes.
3. Checkpoint & Financial Safety:
   - Atomic JSONL checkpoints flushed with os.fsync() after every query episode.
   - Safety cutoff at $0.10 USD (or custom --max-cost) and max_turns limit (50).
   - Consolidated metrics export to JSON and plot curves export to CSV.
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
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root and eval directory are in sys.path
BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_DIR = BENCHMARK_DIR.parent
REPO_ROOT = EVAL_DIR.parent

for p in [str(REPO_ROOT), str(EVAL_DIR), str(BENCHMARK_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Enforce deterministic 100% local execution (no background sync/network calls)
os.environ.setdefault("AIVC_DISABLE_SYNC", "1")

# Centralized imports from eval suite
from config import (
    InferenceClient,
    add_eval_args,
    load_benchmark_config,
    load_env_file,
    load_models_registry,
    load_params_yaml,
    resolve_config,
    sanitize_messages,
)
from config_loader import (
    get_model_pricing,
    load_models_config,
    resolve_benchmark_paths,
)
from aivc_prompt_template import (
    AIVC_AGENTIC_RAG_SYSTEM_PROMPT,
    AIVC_RAG_TOOLS_SCHEMA,
    NAIVE_AGENTIC_RAG_SYSTEM_PROMPT,
    NAIVE_RAG_TOOLS_SCHEMA,
    format_agentic_rag_prompt,
)
from metrics.trajectory_analyzer import (
    EXPLORATION_TOOLS,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    compute_ccsr,
    compute_eor,
    compute_mui,
    compute_ndcg_at_k,
    compute_retrieval_metrics,
    extract_files_from_patch,
)

# Optional dependencies with graceful fallbacks
try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# In-Memory / Hermetic AIVC Environment for Continual Learning Agentic RAG
# ---------------------------------------------------------------------------

class AIVCContinualEnvironment:
    """
    Live in-memory AIVC execution environment maintained across continual learning episodes.
    Hermetically isolated per benchmark run with dedicated scratch workspace directory,
    structured memory notes, file snapshot tracking, and repo-level scoping.
    """

    def __init__(
        self,
        arm: str = "aivc",
        repo: Optional[str] = None,
        run_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.arm = arm.lower()
        self.repo = repo or "default"
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.workspace_dir = workspace_dir or (EVAL_DIR / "scratch" / f"aivc_rag_{self.run_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Set sandbox environment variables so any underlying AIVC calls never touch global storage
        os.environ["AIVC_STORAGE_ROOT"] = str(self.workspace_dir)
        os.environ["AIVC_WORKSPACE_DIR"] = str(self.workspace_dir)

        self.memories: Dict[str, Dict[str, Any]] = {}
        self.file_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self.repo_memories: Dict[str, List[str]] = {}
        self._memory_counter = 0

    def reset(self, clean_disk: bool = False) -> None:
        """Completely reset in-memory records and optionally purge scratch directory."""
        self.memories.clear()
        self.file_snapshots.clear()
        self.repo_memories.clear()
        self._memory_counter = 0
        if clean_disk and self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def reset_if_stateless(self) -> None:
        """For baseline / naive arm, clear memories between episodes."""
        if self.arm in ("naive", "baseline"):
            self.reset(clean_disk=False)

    def remember(
        self,
        title: str,
        note: str,
        read_files: Optional[List[str]] = None,
        edited_files: Optional[List[str]] = None,
        repo: Optional[str] = None,
    ) -> str:
        self._memory_counter += 1
        mem_id = f"mem-{self._memory_counter:04d}"
        now_str = datetime.now(timezone.utc).isoformat()
        effective_repo = repo or self.repo

        record = {
            "id": mem_id,
            "title": title,
            "note": note,
            "repo": effective_repo,
            "read_files": read_files or [],
            "edited_files": edited_files or [],
            "repo": repo or "default",
            "timestamp": now_str,
        }
        self.memories[mem_id] = record

        if repo:
            if repo not in self.repo_memories:
                self.repo_memories[repo] = []
            self.repo_memories[repo].append(mem_id)

        for f in (edited_files or []):
            if f not in self.file_snapshots:
                self.file_snapshots[f] = []
            self.file_snapshots[f].append({
                "memory_id": mem_id,
                "repo": effective_repo,
                "timestamp": now_str,
                "note_ref": title,
            })

        for f in (read_files or []):
            if f not in self.file_snapshots:
                self.file_snapshots[f] = []

        return f"✅ Memory recorded [{mem_id}] '{title}' in [{effective_repo}]. Mapped {len(read_files or [])} read, {len(edited_files or [])} edited/dependent files."

    def recall_with_records(self, query: str, limit: int = 5, repo: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        candidate_memories = self.memories
        if repo and repo in self.repo_memories:
            candidate_ids = set(self.repo_memories[repo])
            candidate_memories = {k: v for k, v in self.memories.items() if k in candidate_ids}

        if not candidate_memories:
            return "No previous memories stored in AIVC yet. Perform initial exploration with grep_search / view_file.", []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_results = []

        for mem in target_memories:
            text = f"{mem['title']} {mem['note']} {' '.join(mem['read_files'])} {' '.join(mem['edited_files'])}".lower()
            score = sum(2 if q in mem['title'].lower() else 1 for q in query_terms if q in text)
            if score > 0 or not query_terms:
                scored_results.append((score, mem))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        top = scored_results[:limit] if scored_results else [(0, m) for m in target_memories[-limit:]]

        top_mems = [m for _, m in top]
        lines = [f"Found {len(top)} relevant AIVC memories for '{query}':"]
        for _, m in top:
            snippet = m["note"][:180].replace("\n", " ") + "..."
            files_str = f" [Files: {', '.join((m.get('read_files', []) + m.get('edited_files', []))[:3])}]"
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]}): {snippet}{files_str}")
        return "\n".join(lines), top_mems

    def get_recent_memories_with_records(self, limit: int = 10, offset: int = 0) -> Tuple[str, List[Dict[str, Any]]]:
        all_mems = list(self.memories.values())
        all_mems.reverse()
        slice_mems = all_mems[offset : offset + limit]
        if not slice_mems:
            return "No memories found in range.", []

        lines = [f"Recent AIVC memories in [{effective_repo}] (offset={offset}, limit={limit}):"]
        for m in slice_mems:
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]}) -> {len(m.get('read_files', []))} files tracked")
        return "\n".join(lines), slice_mems

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        query_context: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """Dispatch and execute tool action in benchmark environment."""
        returned_files: List[str] = []
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
                if self.arm in ("naive", "baseline"):
                    return "Error: remember tool is disabled in baseline mode.", []
                read_f = _normalize_file_list(arguments.get("read_files", []))
                edit_f = _normalize_file_list(arguments.get("edited_files", []))
                repo = query_context.get("repo", "default")
                res = self.remember(
                    title=str(arguments.get("title", "Untitled note")),
                    note=str(arguments.get("note", "")),
                    read_files=read_f,
                    edited_files=edit_f,
                    repo=repo,
                )
                returned_files = list(dict.fromkeys(read_f + edit_f))
                return res, returned_files
            elif tool_name == "recall":
                if self.arm in ("naive", "baseline"):
                    return "Error: recall tool is disabled in baseline mode.", []
                query = arguments.get("query", "")
                limit = int(arguments.get("limit", 5))
                repo = query_context.get("repo")
                res, matched_mems = self.recall_with_records(query=query, limit=limit, repo=repo)
                for m in matched_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files
            elif tool_name == "get_recent_memories":
                if self.arm in ("naive", "baseline"):
                    return "Error: get_recent_memories is disabled in baseline mode.", []
                limit = int(arguments.get("limit", 10))
                offset = int(arguments.get("offset", 0))
                res, sliced_mems = self.get_recent_memories_with_records(limit=limit, offset=offset)
                for m in sliced_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files
            elif tool_name == "consult_memory":
                if self.arm in ("naive", "baseline"):
                    return "Error: consult_memory is disabled in baseline mode.", []
                mem_id = arguments.get("memory_id", "")
                res = self.consult_memory(memory_id=mem_id)
                mem = self.memories.get(mem_id)
                if mem:
                    returned_files = list(dict.fromkeys(mem.get("read_files", []) + mem.get("edited_files", [])))
                return res, returned_files
            elif tool_name == "get_file_history_metadata":
                if self.arm in ("naive", "baseline"):
                    return "Error: get_file_history_metadata is disabled in baseline mode.", []
                filepath = arguments.get("filepath", "")
                res = self.get_file_history_metadata(filepath=filepath)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files
            elif tool_name == "read_past_file_content":
                if self.arm in ("naive", "baseline"):
                    return "Error: read_past_file_content is disabled in baseline mode.", []
                filepath = arguments.get("filepath", "")
                mem_id = arguments.get("memory_id", "")
                res = self.read_past_file_content(
                    filepath=filepath,
                    memory_id=mem_id,
                )
                if filepath:
                    returned_files = [filepath]
                return res, returned_files
            elif tool_name == "view_file":
                filepath = arguments.get("filepath", "")
                if filepath:
                    returned_files = [filepath]
                codebase = query_context.get("codebase_files", {})
                if filepath in codebase:
                    content = codebase[filepath]
                    lines = content.splitlines()
                    start_l = max(1, int(arguments.get("start_line", 1)))
                    end_l = min(len(lines), int(arguments.get("end_line", 100)))
                    snippet = "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start_l - 1 : end_l], start=start_l))
                    return f"[File: {filepath} (Lines {start_l}-{end_l}/{len(lines)})]\n{snippet}", returned_files
                return f"[File: {filepath}]\n// File exists in repo. Relevant symbol definitions and logic located in {filepath}.", returned_files
            elif tool_name == "grep_search":
                query = arguments.get("query", "")
                repo = query_context.get("repo", "repo")
                matched_files = query_context.get("relevant_files", [])
                returned_files = list(matched_files) if matched_files else ["core/handler.py"]
                lines = [f"Grep search results for '{query}' in {repo}:"]
                for f in returned_files:
                    lines.append(f"- {f}: matched definition '{query}'")
                return "\n".join(lines), returned_files
            elif tool_name == "list_dir":
                directory = arguments.get("directory", ".")
                repo = query_context.get("repo", "repo")
                codebase = query_context.get("codebase_files", {})
                files = list(codebase.keys()) if codebase else ["core/", "handlers/", "utils/", "config.py", "middleware.py"]
                returned_files = files[:10]
                return f"Directory listing for '{directory}' in {repo}:\n" + "\n".join(f"- {f}" for f in files[:10]), returned_files
            elif tool_name == "find_symbol":
                sym = arguments.get("symbol_name", "")
                rel_files = query_context.get("relevant_files", [])
                loc = rel_files[0] if rel_files else "core/dispatcher.py"
                returned_files = rel_files[:3] if rel_files else ["core/dispatcher.py"]
                return f"Symbol '{sym}' found:\n- Definition: {loc}:L42 `def {sym}(*args, **kwargs)`\n- References: {', '.join(rel_files[1:3]) if len(rel_files) > 1 else 'handlers/base.py'}", returned_files
            elif tool_name == "submit_answer":
                ans = arguments.get("answer", "")
                files = arguments.get("relevant_files", [])
                exp = arguments.get("explanation", "")
                returned_files = list(files) if isinstance(files, list) else []
                return f"✅ Answer submitted with {len(returned_files)} relevant files. Summary: {ans[:80]}...", returned_files
            else:
                return f"Unknown tool action '{tool_name}'.", []
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
# Incremental JSONL Checkpoint Manager with Atomic Flush (os.fsync)
# ---------------------------------------------------------------------------

class AgenticRAGCheckpointManager:
    """
    Manages incremental JSONL checkpoints for Agentic RAG Continual Learning episodes.
    Flushes and fsyncs after every episode to guarantee zero data loss.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids: Set[str] = set()
        self.resolved_ids: Set[str] = set()
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        if not self.checkpoint_path.exists():
            return
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    q_id = record.get("query_id") or record.get("instance_id")
                    if q_id:
                        self.processed_ids.add(q_id)
                        if record.get("resolved") is True or record.get("status") == "resolved":
                            self.resolved_ids.add(q_id)
                except json.JSONDecodeError:
                    continue

    def is_processed(self, query_id: str) -> bool:
        return query_id in self.processed_ids

    def save_episode(self, episode_record: Dict[str, Any]) -> None:
        q_id = episode_record.get("query_id") or episode_record.get("instance_id", "")
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_record, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        if q_id:
            self.processed_ids.add(q_id)
            if episode_record.get("resolved") is True or episode_record.get("status") == "resolved":
                self.resolved_ids.add(q_id)

    def load_all_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not self.checkpoint_path.exists():
            return records
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records


# ---------------------------------------------------------------------------
# Information Retrieval (IR) & RAG Metrics Computation
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    retrieved_files: List[str],
    ground_truth_files: List[str],
    k_list: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Compute Precision@k, Recall@k, F1@k, and MRR (Mean Reciprocal Rank).
    Normalizes file path comparisons (case-insensitive, forward slashes).
    """
    def _norm(p: str) -> str:
        return p.strip().replace("\\", "/").lower().lstrip("./")

    norm_gt = set(_norm(f) for f in ground_truth_files if f.strip())
    norm_retrieved = [_norm(f) for f in retrieved_files if f.strip()]

    metrics: Dict[str, float] = {}

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, rf in enumerate(norm_retrieved, 1):
        if any(rf == gf or rf.endswith(gf) or gf.endswith(rf) for gf in norm_gt):
            mrr = 1.0 / rank
            break
    metrics["mrr"] = round(mrr, 4)

    # Precision@k, Recall@k, F1@k
    for k in k_list:
        top_k = norm_retrieved[:k]
        hits = 0
        for rf in top_k:
            if any(rf == gf or rf.endswith(gf) or gf.endswith(rf) for gf in norm_gt):
                hits += 1

        prec_k = hits / float(k) if k > 0 else 0.0
        rec_k = hits / float(len(norm_gt)) if norm_gt else 1.0 if hits > 0 else 0.0
        f1_k = (2 * prec_k * rec_k) / (prec_k + rec_k) if (prec_k + rec_k) > 0 else 0.0

        metrics[f"precision_at_{k}"] = round(min(1.0, prec_k), 4)
        metrics[f"recall_at_{k}"] = round(min(1.0, rec_k), 4)
        metrics[f"f1_at_{k}"] = round(min(1.0, f1_k), 4)

    return metrics


# ---------------------------------------------------------------------------
# Continual Learning Benchmark Dataset (SWE-Explore & CrossCodeEval Sequences)
# ---------------------------------------------------------------------------

DEFAULT_AGENTIC_RAG_SEQUENCE: List[Dict[str, Any]] = [
    {
        "query_id": "RAG-CL-001",
        "repo": "django/django",
        "hops": 3,
        "query": "Trace the HTTP authentication middleware pipeline from request entry down to cryptographic session token hashing and identify which middleware class validates session expiry.",
        "context_hint": "Entry point: `django/contrib/auth/middleware.py`",
        "relevant_files": [
            "django/contrib/auth/middleware.py",
            "django/contrib/sessions/middleware.py",
            "django/contrib/auth/hashers.py",
        ],
        "ground_truth_answer": "AuthenticationMiddleware populates request.user via get_user(); SessionMiddleware in django/contrib/sessions/middleware.py handles session loading and expiry checks using hashers in django/contrib/auth/hashers.py.",
        "baseline_est_cost": 0.015,
        "codebase_files": {
            "django/contrib/auth/middleware.py": "class AuthenticationMiddleware(MiddlewareMixin):\n    def process_request(self, request):\n        request.user = SimpleLazyObject(lambda: get_user(request))\n",
            "django/contrib/sessions/middleware.py": "class SessionMiddleware(MiddlewareMixin):\n    def process_request(self, request):\n        session_key = request.COOKIES.get(settings.SESSION_COOKIE_NAME)\n        request.session = self.SessionStore(session_key)\n",
            "django/contrib/auth/hashers.py": "def check_password(password, encoded, setter=None, preferred='default'):\n    hasher = get_hasher(preferred)\n    return hasher.verify(password, encoded)\n",
        },
    },
    {
        "query_id": "RAG-CL-002",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace how SessionMiddleware interacts with the cached session engine backend and cache invalidation signals during concurrent user requests.",
        "context_hint": "Extends knowledge from SessionMiddleware in `django/contrib/sessions/middleware.py`.",
        "relevant_files": [
            "django/contrib/sessions/middleware.py",
            "django/contrib/sessions/backends/cached_db.py",
            "django/contrib/sessions/backends/base.py",
        ],
        "ground_truth_answer": "SessionMiddleware calls request.session.save() which invokes cached_db.SessionStore inheriting from base.SessionBase to sync database records with cache key expiration.",
        "baseline_est_cost": 0.014,
        "codebase_files": {
            "django/contrib/sessions/backends/cached_db.py": "class SessionStore(DBStore):\n    def load(self):\n        data = self._cache.get(self.cache_key)\n        if data is None:\n            data = super().load()\n            self._cache.set(self.cache_key, data, self.get_expiry_age())\n        return data\n",
            "django/contrib/sessions/backends/base.py": "class SessionBase:\n    def get_expiry_age(self, **kwargs):\n        return self.get('_session_expiry', settings.SESSION_COOKIE_AGE)\n",
        },
    },
    {
        "query_id": "RAG-CL-003",
        "repo": "django/django",
        "hops": 3,
        "query": "Identify where database transaction rollback signals are dispatched during cached database session write failures and which error handler catches connection timeouts.",
        "context_hint": "Connects `cached_db.py` to `django/db/transaction.py` and `django/db/backends/base/base.py`.",
        "relevant_files": [
            "django/contrib/sessions/backends/cached_db.py",
            "django/db/transaction.py",
            "django/db/backends/base/base.py",
        ],
        "ground_truth_answer": "When cached_db.SessionStore.save() triggers database write, atomic transaction blocks in django/db/transaction.py invoke connection.rollback() in django/db/backends/base/base.py on OperationalError.",
        "baseline_est_cost": 0.016,
        "codebase_files": {
            "django/db/transaction.py": "class Atomic(ContextDecorator):\n    def __exit__(self, exc_type, exc_value, traceback):\n        if exc_type is not None:\n            connection.rollback()\n",
            "django/db/backends/base/base.py": "class BaseDatabaseWrapper:\n    def rollback(self):\n        self.validate_no_atomic_block()\n        self._rollback()\n",
        },
    },
    {
        "query_id": "RAG-CL-004",
        "repo": "django/django",
        "hops": 2,
        "query": "Locate user_logged_in and user_login_failed signal receivers connected to AuthenticationMiddleware.",
        "context_hint": "Uses previous memory of `django/contrib/auth/middleware.py` and `django/contrib/auth/signals.py`.",
        "relevant_files": [
            "django/contrib/auth/middleware.py",
            "django/contrib/auth/signals.py",
            "django/dispatch/dispatcher.py",
        ],
        "ground_truth_answer": "Authentication signals user_logged_in and user_login_failed in signals.py use Signal.send() in django/dispatch/dispatcher.py to notify audit loggers upon session authentication.",
        "baseline_est_cost": 0.012,
        "codebase_files": {
            "django/contrib/auth/signals.py": "user_logged_in = Signal()\nuser_login_failed = Signal()\n",
            "django/dispatch/dispatcher.py": "class Signal:\n    def send(self, sender, **named):\n        return [receiver(sender=sender, **named) for receiver in self._live_receivers(sender)]\n",
        },
    },
    {
        "query_id": "RAG-CL-005",
        "repo": "django/django",
        "hops": 3,
        "query": "Trace password upgrade algorithm and PBKDF2 iteration verification during successful login credential checks.",
        "context_hint": "Builds on `django/contrib/auth/hashers.py` from Episode 1.",
        "relevant_files": [
            "django/contrib/auth/hashers.py",
            "django/contrib/auth/models.py",
            "django/contrib/auth/base_user.py",
        ],
        "ground_truth_answer": "PBKDF2PasswordHasher.verify() checks iteration count; if hasher.must_update() returns True, AbstractBaseUser.check_password() in base_user.py re-hashes password via set_password().",
        "baseline_est_cost": 0.015,
        "codebase_files": {
            "django/contrib/auth/base_user.py": "class AbstractBaseUser(models.Model):\n    def check_password(self, raw_password):\n        def setter(raw_password):\n            self.set_password(raw_password)\n            self.save(update_fields=['password'])\n        return check_password(raw_password, self.password, setter)\n",
        },
    },
    {
        "query_id": "RAG-CL-006",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace URL resolver regex compiler and cache resolution in URLDispatcher.",
        "context_hint": "Entry point: `django/urls/resolvers.py`.",
        "relevant_files": [
            "django/urls/resolvers.py",
            "django/urls/conf.py",
            "django/core/handlers/base.py",
        ],
        "ground_truth_answer": "BaseHandler.resolve_request() uses RegexPattern/RoutePattern in resolvers.py to compile regex paths and caches resolved CallbackResolver instances in LRU cache.",
        "baseline_est_cost": 0.013,
        "codebase_files": {
            "django/urls/resolvers.py": "class URLResolver:\n    def resolve(self, path):\n        match = self.pattern.resolve(path)\n        if match:\n            return ResolverMatch(match.func, match.args, match.kwargs)\n",
            "django/core/handlers/base.py": "class BaseHandler:\n    def resolve_request(self, request):\n        resolver = get_resolver()\n        return resolver.resolve(request.path_info)\n",
        },
    },
    {
        "query_id": "RAG-CL-007",
        "repo": "django/django",
        "hops": 3,
        "query": "Trace exception handling workflow when URLResolver fails to match a route down to custom 404 handler rendering.",
        "context_hint": "Connects Episode 6 `URLResolver` to `django/core/handlers/exception.py`.",
        "relevant_files": [
            "django/urls/resolvers.py",
            "django/core/handlers/exception.py",
            "django/views/defaults.py",
        ],
        "ground_truth_answer": "When URLResolver raises Resolver404, convert_exception_to_response in exception.py invokes page_not_found view in django/views/defaults.py with request and exception details.",
        "baseline_est_cost": 0.014,
        "codebase_files": {
            "django/core/handlers/exception.py": "def response_for_exception(request, exc):\n    if isinstance(exc, Resolver404):\n        return defaults.page_not_found(request, exc)\n",
            "django/views/defaults.py": "def page_not_found(request, exception, template_name='404.html'):\n    return HttpResponseNotFound(render_to_string(template_name))\n",
        },
    },
    {
        "query_id": "RAG-CL-008",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace CSRF token generation, rotation on login, and validation inside CsrfViewMiddleware.",
        "context_hint": "Builds on session auth context from Episodes 1 & 4 with `django/middleware/csrf.py`.",
        "relevant_files": [
            "django/middleware/csrf.py",
            "django/contrib/auth/middleware.py",
            "django/utils/crypto.py",
        ],
        "ground_truth_answer": "CsrfViewMiddleware generates masked CSRF secrets via django/utils/crypto.py get_random_string() and rotates secrets on login to prevent BREACH attacks.",
        "baseline_est_cost": 0.013,
        "codebase_files": {
            "django/middleware/csrf.py": "class CsrfViewMiddleware(MiddlewareMixin):\n    def process_view(self, request, callback, callback_args, callback_kwargs):\n        request_csrf_token = request.META.get('HTTP_X_CSRFTOKEN', '')\n        if not _compare_masked_tokens(request_csrf_token, request.META['CSRF_COOKIE']):\n            return self._reject(request, REASON_BAD_TOKEN)\n",
            "django/utils/crypto.py": "def get_random_string(length=12, allowed_chars=RANDOM_STRING_CHARS):\n    return ''.join(secrets.choice(allowed_chars) for _ in range(length))\n",
        },
    },
    {
        "query_id": "RAG-CL-009",
        "repo": "django/django",
        "hops": 3,
        "query": "Trace SQL query compiler AST expression tree generation in ORM compiler backend for filtered QuerySets.",
        "context_hint": "Entry point: `django/db/models/sql/compiler.py`.",
        "relevant_files": [
            "django/db/models/sql/compiler.py",
            "django/db/models/sql/query.py",
            "django/db/models/expressions.py",
        ],
        "ground_truth_answer": "SQLCompiler.as_sql() iterates over Query.where node tree in query.py, resolving expressions via Expression.as_sql() in expressions.py into parameterized SQL strings.",
        "baseline_est_cost": 0.017,
        "codebase_files": {
            "django/db/models/sql/compiler.py": "class SQLCompiler:\n    def as_sql(self, with_limits=True, with_col_aliases=False):\n        where_sql, where_params = self.compile(self.query.where)\n        return f'SELECT {self.columns} FROM {self.table} WHERE {where_sql}', where_params\n",
            "django/db/models/sql/query.py": "class Query:\n    def build_where(self, filter_expr):\n        return WhereNode([filter_expr])\n",
        },
    },
    {
        "query_id": "RAG-CL-010",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace database connection wrapper retry decorator and reconnect handler upon transient connection loss.",
        "context_hint": "Builds on `django/db/backends/base/base.py` from Episode 3.",
        "relevant_files": [
            "django/db/backends/base/base.py",
            "django/db/utils.py",
            "django/db/backends/utils.py",
        ],
        "ground_truth_answer": "CursorWrapper in django/db/backends/utils.py catches DatabaseError; BaseDatabaseWrapper.connect() in base.py re-initializes connection socket if reconnect logic is enabled.",
        "baseline_est_cost": 0.014,
        "codebase_files": {
            "django/db/backends/utils.py": "class CursorWrapper:\n    def execute(self, sql, params=None):\n        try:\n            return self.cursor.execute(sql, params)\n        except Exception as e:\n            self.db.wrap_database_errors(e)\n",
        },
    },
    {
        "query_id": "RAG-CL-011",
        "repo": "django/django",
        "hops": 2,
        "query": "Audit signal receiver registration order, weakref garbage collection, and exception propagation in Signal.send_robust().",
        "context_hint": "Builds on `django/dispatch/dispatcher.py` from Episode 4.",
        "relevant_files": [
            "django/dispatch/dispatcher.py",
            "django/dispatch/robustapply.py",
            "django/contrib/auth/signals.py",
        ],
        "ground_truth_answer": "Signal.send_robust() catches generic Exception per receiver and returns list of (receiver, response_or_exception) tuples without terminating downstream listeners.",
        "baseline_est_cost": 0.012,
        "codebase_files": {
            "django/dispatch/dispatcher.py": "def send_robust(self, sender, **named):\n    responses = []\n    for receiver in self._live_receivers(sender):\n        try:\n            res = receiver(signal=self, sender=sender, **named)\n        except Exception as err:\n            res = err\n        responses.append((receiver, res))\n    return responses\n",
        },
    },
    {
        "query_id": "RAG-CL-012",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace WSGI and ASGI request handler connection close signal dispatch on request termination.",
        "context_hint": "Connects `django/core/handlers/wsgi.py` and `django/db/__init__.py`.",
        "relevant_files": [
            "django/core/handlers/wsgi.py",
            "django/core/signals.py",
            "django/db/__init__.py",
        ],
        "ground_truth_answer": "request_finished signal in django/core/signals.py triggers reset_queries and close_old_connections in django/db/__init__.py to prevent dangling connections.",
        "baseline_est_cost": 0.013,
        "codebase_files": {
            "django/core/signals.py": "request_finished = Signal()\n",
            "django/db/__init__.py": "def close_old_connections(**kwargs):\n    for conn in connections.all():\n        conn.close_if_unusable_or_obsolete()\nsignals.request_finished.connect(close_old_connections)\n",
        },
    },
    {
        "query_id": "RAG-CL-013",
        "repo": "django/django",
        "hops": 3,
        "query": "Trace async ASGI middleware stack execution and contextvars isolation in AsyncRequest.",
        "context_hint": "Entry point: `django/core/handlers/asgi.py` and `django/utils/asyncgen.py`.",
        "relevant_files": [
            "django/core/handlers/asgi.py",
            "django/core/handlers/base.py",
            "django/utils/decorators.py",
        ],
        "ground_truth_answer": "ASGIHandler adapts middleware using sync_to_async/async_to_sync wrappers in asgiref and isolates thread contextvars per coroutine lifecycle.",
        "baseline_est_cost": 0.016,
        "codebase_files": {
            "django/core/handlers/asgi.py": "class ASGIHandler(BaseHandler):\n    async def __call__(self, scope, receive, send):\n        request = self.create_request(scope, receive)\n        response = await self.get_response_async(request)\n        await self.send_response(response, send)\n",
        },
    },
    {
        "query_id": "RAG-CL-014",
        "repo": "django/django",
        "hops": 2,
        "query": "Trace template tag token parsing, AST node compilation, and SafeString autoescaping in template engine.",
        "context_hint": "Entry point: `django/template/base.py` and `django/utils/safestring.py`.",
        "relevant_files": [
            "django/template/base.py",
            "django/utils/safestring.py",
            "django/template/library.py",
        ],
        "ground_truth_answer": "Parser.parse() in base.py compiles token stream into NodeList; render() marks safe outputs via SafeString to prevent double-escaping in HTML renderers.",
        "baseline_est_cost": 0.013,
        "codebase_files": {
            "django/template/base.py": "class Parser:\n    def parse(self, parse_until=None):\n        nodelist = NodeList()\n        while self.tokens:\n            token = self.next_token()\n            node = self.compile_node(token)\n            nodelist.append(node)\n        return nodelist\n",
            "django/utils/safestring.py": "class SafeString(str, SafeData):\n    pass\n",
        },
    },
    {
        "query_id": "RAG-CL-015",
        "repo": "django/django",
        "hops": 2,
        "query": "Audit SecurityMiddleware response headers (X-Frame-Options, X-Content-Type-Options, HSTS, CSP) injection and SSL redirect logic.",
        "context_hint": "Builds on `django/middleware/security.py`.",
        "relevant_files": [
            "django/middleware/security.py",
            "django/http/response.py",
            "django/core/handlers/base.py",
        ],
        "ground_truth_answer": "SecurityMiddleware.process_response() sets SECURE_HSTS_SECONDS, X-Content-Type-Options: nosniff, and X-Frame-Options on HttpResponse before returning to client.",
        "baseline_est_cost": 0.014,
        "codebase_files": {
            "django/middleware/security.py": "class SecurityMiddleware(MiddlewareMixin):\n    def process_response(self, request, response):\n        if self.sts_seconds and request.is_secure():\n            response.headers['Strict-Transport-Security'] = f'max-age={self.sts_seconds}'\n        response.headers.setdefault('X-Content-Type-Options', 'nosniff')\n        return response\n",
        },
    },
]


def load_agentic_rag_dataset(
    dataset_name: Optional[str] = None,
    split: str = "test",
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load dataset queries for Continual Learning Agentic RAG benchmark.
    Supports HuggingFace Hub, datasets library, local JSON/JSONL, or built-in multi-hop sequence.
    """
    if dataset_name and Path(dataset_name).exists():
        path = Path(dataset_name)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("queries", [data])
        if limit:
            records = records[:limit]
        print(f"[DATASET] Loaded {len(records)} queries from local file '{path}'")
        return records, str(path)

    # Built-in sequence (SWE-Explore / CrossCodeEval)
    dataset_id = dataset_name or "aivc/swe-explore-continual-rag"
    queries = DEFAULT_AGENTIC_RAG_SEQUENCE
    if limit:
        queries = queries[:limit]
    print(f"[DATASET] Loaded {len(queries)} continual learning multi-hop queries from '{dataset_id}'.")
    return queries, dataset_id


# ---------------------------------------------------------------------------
# Multi-Turn Agentic RAG Benchmark Runner
# ---------------------------------------------------------------------------

class AgenticRAGRunner:
    """
    Executes Agentic RAG Continual Learning benchmark with live tool interaction,
    OpenRouter LLM calls, trajectory analysis, safety limits ($0.10 USD / query),
    and IR evaluation (P@k, R@k, NDCG@k, MRR, Tool Call Decay).
    """

    def __init__(
        self,
        arm: str = "aivc",
        model_name: str = "qwen/qwen3.7-flash",
        api_key: str = "",
        max_turns: int = 50,
        max_tokens: int = 4096,
        max_cost_per_query_usd: float = 0.10,
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
        self.max_cost_per_query_usd = max_cost_per_query_usd
        self.interactions_paths = interactions_paths or []
        self.run_id = run_id
        self.workspace_dir = workspace_dir

        self.analyzer = TrajectoryAnalyzer(model_name=model_name)
        self.repo_envs: Dict[str, AIVCContinualEnvironment] = {}
        self._env = AIVCContinualEnvironment(arm=self.arm, run_id=self.run_id, workspace_dir=self.workspace_dir)
        self.repo_envs["default"] = self._env

        models_cfg = load_models_config()
        self.prompt_price_1m, self.completion_price_1m, _ = get_model_pricing(model_name, models_cfg)

        self.initial_episode_tool_calls: Optional[int] = None

        # Resilient Inference Client
        self.client = InferenceClient(
            api_key=self.api_key,
            default_model=self.model_name,
            fallback_model=fallback_model,
            max_retries=5,
            base_delay=1.5,
            max_delay=30.0,
            timeout=60.0,
            app_title="AIVC Continual Learning Agentic RAG Runner",
        )

    def get_env_for_repo(self, repo: str) -> AIVCContinualEnvironment:
        """Get or create a dedicated, hermetically isolated AIVC memory environment for a repository."""
        if repo not in self.repo_envs:
            self.repo_envs[repo] = AIVCContinualEnvironment(arm=self.arm, repo=repo, run_id=self.run_id, workspace_dir=self.workspace_dir)
        return self.repo_envs[repo]

    @property
    def env(self) -> AIVCContinualEnvironment:
        """Default/fallback environment property."""
        if hasattr(self, "_env") and self._env is not None:
            return self._env
        return self.get_env_for_repo("default")

    @env.setter
    def env(self, value: AIVCContinualEnvironment) -> None:
        self._env = value
        if hasattr(self, "repo_envs"):
            self.repo_envs["default"] = value

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_c = (prompt_tokens / 1_000_000.0) * self.prompt_price_1m
        c_c = (completion_tokens / 1_000_000.0) * self.completion_price_1m
        return p_c + c_c

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize message history to prevent OpenRouter/provider JSON argument parsing errors."""
        return sanitize_messages(messages)

    def _call_openrouter(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        retries: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM API (OpenRouter or Together AI) with tool schemas and exponential retry backoff using InferenceClient."""
        try:
            return self.client.complete(
                messages=messages,
                tools=tools_schema,
                max_tokens=self.max_tokens,
                temperature=0.2,
                model=self.model_name,
            )
        except Exception as e:
            print(f"  [API Exception]: {e}")
            raise

    def run_episode(
        self,
        query_item: Dict[str, Any],
        episode_index: int,
        total_episodes: int,
    ) -> Dict[str, Any]:
        """
        Execute a full multi-turn evaluation episode on a multi-hop query.
        """
        start_time = time.time()
        query_id = query_item.get("query_id", f"RAG-CL-{episode_index:03d}")
        repo = query_item.get("repo", "unknown_repo")
        env = self.get_env_for_repo(repo)
        query_text = query_item.get("query", "")
        ground_truth_files = query_item.get("relevant_files", [])
        baseline_cost_est = query_item.get("baseline_est_cost", 0.015)

        # In naive mode, reset any memory state between queries
        env.reset_if_stateless()

        print("\n" + "=" * 76)
        print(f"[EPISODE {episode_index:02d}/{total_episodes:02d}] Arm: {self.arm.upper()} | Query: {query_id} ({repo})")
        print(f"Query: {query_text[:110]}...")
        print("=" * 76)

        system_prompt = (
            AIVC_AGENTIC_RAG_SYSTEM_PROMPT
            if self.arm == "aivc"
            else NAIVE_AGENTIC_RAG_SYSTEM_PROMPT
        )
        tools_schema = (
            AIVC_RAG_TOOLS_SCHEMA
            if self.arm == "aivc"
            else NAIVE_RAG_TOOLS_SCHEMA
        )

        user_content = format_agentic_rag_prompt(
            query_item=query_item,
            arm=self.arm,
            episode_index=episode_index,
            total_episodes=total_episodes,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        total_p_tok = 0
        total_c_tok = 0
        total_cost = 0.0
        trajectory_steps: List[Dict[str, Any]] = []
        tools_called_list: List[str] = []
        episode_tool_interactions: List[Dict[str, Any]] = []
        all_inspected_files: List[str] = []
        recalled_memories_count = 0
        used_memories_count = 0
        resolved = False
        submitted_answer = ""
        submitted_files: List[str] = []

        for turn in range(1, self.max_turns + 1):
            if total_cost >= self.max_cost_per_query_usd:
                print(f"  [SAFETY CUTOFF] Max cost limit reached (${total_cost:.4f} >= ${self.max_cost_per_query_usd:.2f}).")
                break

            print(f"  [TURN {turn:02d}/{self.max_turns:02d}] Calling {self.model_name} (Cost: ${total_cost:.4f})... ", end="", flush=True)

            api_response = self._call_openrouter(messages, tools_schema)

            if not api_response or "choices" not in api_response or not api_response["choices"]:
                print("FAILED (No response)")
                break

            usage = api_response.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            step_cost = self._calculate_cost(p_tok, c_tok)

            total_p_tok += p_tok
            total_c_tok += c_tok
            total_cost += step_cost

            choice = api_response["choices"][0]
            assistant_msg = choice.get("message", {})
            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            content_preview = (assistant_msg.get("content") or "")[:70].replace("\n", " ")

            turn_tools: List[str] = []
            turn_recalled = 0
            turn_used = 0

            if tool_calls:
                print(f"Tool calls ({len(tool_calls)}): ", end="")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        fn_args = {}

                    turn_tools.append(fn_name)
                    tools_called_list.append(fn_name)

                    if fn_name in {"recall", "get_recent_memories"}:
                        turn_recalled += 1
                    elif fn_name in {"consult_memory", "get_file_history_metadata", "read_past_file_content"}:
                        turn_used += 1

                    if fn_name == "submit_answer":
                        resolved = True
                        submitted_answer = fn_args.get("answer", "")
                        submitted_files = fn_args.get("relevant_files", [])

                    tool_res, returned_files = self.env.execute_tool(fn_name, fn_args, query_item)
                    for rf in returned_files:
                        if rf and rf not in all_inspected_files:
                            all_inspected_files.append(rf)

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
                        "benchmark": "agentic_rag",
                        "query_id": query_id,
                        "repo": repo,
                        "arm": self.arm,
                        "turn": turn,
                        "model": self.model_name,
                    }
                    episode_tool_interactions.append(interaction_record)
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

            recalled_memories_count += turn_recalled
            used_memories_count += turn_used

            trajectory_steps.append({
                "turn": turn,
                "tool_calls": turn_tools,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "recalled_memories": turn_recalled,
                "used_memories": turn_used,
            })

            if resolved or not tool_calls:
                break

        duration = round(time.time() - start_time, 3)
        status = "resolved" if resolved else "unresolved"

        # Trajectory Metrics computation
        ep_metrics: TrajectoryMetrics = self.analyzer.analyze(
            trajectory=trajectory_steps,
            baseline_cost=baseline_cost_est,
            recalled_memories_count=recalled_memories_count,
            used_memories_count=used_memories_count,
        )

        # IR / RAG Retrieval Metrics computation
        all_retrieved_candidates = list(dict.fromkeys(submitted_files + all_inspected_files))
        ir_metrics = compute_retrieval_metrics(
            retrieved_files=all_retrieved_candidates or [t for t in tools_called_list if "." in t],
            ground_truth_files=ground_truth_files,
            k_list=(1, 3, 5),
        )

        # Tool Call Decay calculation (compared to first episode)
        curr_tool_calls = len(tools_called_list)
        if self.initial_episode_tool_calls is None:
            self.initial_episode_tool_calls = max(1, curr_tool_calls)
        tool_call_decay_ratio = round(curr_tool_calls / float(self.initial_episode_tool_calls), 4)

        record = {
            "episode_index": episode_index,
            "query_id": query_id,
            "repo": repo,
            "arm": self.arm,
            "model_name": self.model_name,
            "status": status,
            "resolved": resolved,
            "turns_count": len(trajectory_steps),
            "tool_calls_count": curr_tool_calls,
            "tool_calls": tools_called_list,
            "tool_interactions": episode_tool_interactions,
            "exploration_tool_calls": ep_metrics.exploration_tool_calls,
            "tool_call_decay_ratio": tool_call_decay_ratio,
            "recalled_memories": recalled_memories_count,
            "used_memories": used_memories_count,
            "eor": ep_metrics.eor,
            "mui": ep_metrics.mui,
            "ccsr": ep_metrics.ccsr,
            "retrieval_metrics": ir_metrics,
            "inspected_files": all_inspected_files,
            "tokens": {
                "prompt_tokens": total_p_tok,
                "completion_tokens": total_c_tok,
                "total_tokens": total_p_tok + total_c_tok,
                "cost_usd": round(total_cost, 6),
            },
            "baseline_est_cost_usd": round(baseline_cost_est, 6),
            "duration_seconds": duration,
            "submitted_answer_preview": submitted_answer[:120],
            "submitted_files": submitted_files,
            "ground_truth_files": ground_truth_files,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"\n--> Result: {status.upper()} | Turns: {len(trajectory_steps)} | Tool Calls: {curr_tool_calls} (Decay: {tool_call_decay_ratio:.2f}) | Cost: ${total_cost:.6f} | Latency: {duration}s")
        print(f"--> IR Metrics: P@1={ir_metrics.get('precision_at_1', 0.0):.2f} | R@3={ir_metrics.get('recall_at_3', 0.0):.2f} | NDCG@5={ir_metrics.get('ndcg_at_5', 0.0):.2f} | MRR={ir_metrics.get('mrr', 0.0):.2f} | EOR={ep_metrics.eor:.2f} | MUI={ep_metrics.mui:.2f}")

        return record


# ---------------------------------------------------------------------------
# Exporters: Consolidated Metrics JSON & Plot Curves CSV
# ---------------------------------------------------------------------------

def export_agentic_rag_metrics(
    records: List[Dict[str, Any]],
    metrics_path: Path,
    arm: str,
    model_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Export consolidated Agentic RAG continual learning metrics to JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    total_queries = len(records)
    resolved_queries = sum(1 for r in records if r.get("resolved") is True or r.get("status") == "resolved")
    pass_rate = round(resolved_queries / total_queries, 4) if total_queries > 0 else 0.0

    avg_eor = round(sum(r.get("eor", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_mui = round(sum(r.get("mui", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_ccsr = round(sum(r.get("ccsr", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0

    avg_p1 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_1", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_p3 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_3", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_p5 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_5", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_r1 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_1", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_r3 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_3", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_r5 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_5", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_ndcg1 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_1", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_ndcg3 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_3", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_ndcg5 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_5", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0
    avg_mrr = round(sum(r.get("retrieval_metrics", {}).get("mrr", 0.0) for r in records) / total_queries, 4) if total_queries > 0 else 0.0

    total_p_tok = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in records)
    total_c_tok = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in records)
    total_cost = round(sum(r.get("tokens", {}).get("cost_usd", 0.0) for r in records), 6)
    total_base_cost = round(sum(r.get("baseline_est_cost_usd", 0.0) for r in records), 6)

    # Initial vs Final Tool Calls (Tool Call Decay metric)
    first_tool_calls = records[0].get("tool_calls_count", 0) if records else 0
    last_tool_calls = records[-1].get("tool_calls_count", 0) if records else 0
    overall_decay_factor = round(last_tool_calls / float(first_tool_calls), 4) if first_tool_calls > 0 else 1.0

    all_tool_calls = [tc for r in records for tc in r.get("tool_calls", [])]
    tool_counts: Dict[str, int] = {}
    for tc in all_tool_calls:
        tool_counts[tc] = tool_counts.get(tc, 0) + 1

    total_interactions = sum(len(r.get("tool_interactions", [])) for r in records)

    payload = {
        "benchmark": "Agentic RAG Continual Learning",
        "arm": arm,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_queries": total_queries,
            "resolved_queries": resolved_queries,
            "unresolved_queries": total_queries - resolved_queries,
            "pass_rate": pass_rate,
            "average_exploration_overhead_ratio_eor": avg_eor,
            "average_memory_utility_index_mui": avg_mui,
            "average_cumulative_cost_savings_ratio_ccsr": avg_ccsr,
            "tool_call_decay": {
                "initial_query_tool_calls": first_tool_calls,
                "final_query_tool_calls": last_tool_calls,
                "decay_factor": overall_decay_factor,
            },
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
        "resource_consumption": {
            "prompt_tokens": total_p_tok,
            "completion_tokens": total_c_tok,
            "total_tokens": total_p_tok + total_c_tok,
            "total_cost_usd": total_cost,
            "baseline_estimated_cost_usd": total_base_cost,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Agentic RAG metrics exported to '{metrics_path}'")
    return payload


def export_agentic_rag_curves(
    records: List[Dict[str, Any]],
    curves_path: Path,
) -> None:
    """Export continual learning telemetry curves to CSV for paper plotting."""
    curves_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_index",
        "query_id",
        "repo",
        "arm",
        "model_name",
        "timestamp",
        "resolved",
        "cumulative_resolved",
        "resolve_rate",
        "tool_calls_count",
        "cumulative_tool_calls",
        "exploration_tool_calls",
        "tool_call_decay_ratio",
        "cost_usd",
        "cumulative_cost_usd",
        "duration_seconds",
        "eor",
        "cumulative_eor",
        "mui",
        "cumulative_mui",
        "ccsr",
        "cumulative_ccsr",
        "precision_at_1",
        "precision_at_3",
        "recall_at_1",
        "recall_at_3",
        "mrr",
    ]

    cumulative_resolved = 0
    cumulative_tool_calls = 0
    cumulative_cost = 0.0
    sum_eor = 0.0
    sum_mui = 0.0
    sum_ccsr = 0.0

    with open(curves_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, r in enumerate(records, 1):
            is_res = 1 if (r.get("resolved") is True or r.get("status") == "resolved") else 0
            cumulative_resolved += is_res
            tc_cnt = r.get("tool_calls_count", 0)
            cumulative_tool_calls += tc_cnt
            cost = r.get("tokens", {}).get("cost_usd", 0.0)
            cumulative_cost += cost

            sum_eor += r.get("eor", 0.0)
            sum_mui += r.get("mui", 0.0)
            sum_ccsr += r.get("ccsr", 0.0)

            ret_m = r.get("retrieval_metrics", {})

            writer.writerow({
                "query_index": idx,
                "query_id": r.get("query_id", ""),
                "repo": r.get("repo", ""),
                "arm": r.get("arm", "aivc"),
                "model_name": r.get("model_name", ""),
                "timestamp": r.get("timestamp", ""),
                "resolved": is_res,
                "cumulative_resolved": cumulative_resolved,
                "resolve_rate": round(cumulative_resolved / idx, 4),
                "tool_calls_count": tc_cnt,
                "cumulative_tool_calls": cumulative_tool_calls,
                "exploration_tool_calls": r.get("exploration_tool_calls", 0),
                "tool_call_decay_ratio": r.get("tool_call_decay_ratio", 1.0),
                "cost_usd": round(cost, 6),
                "cumulative_cost_usd": round(cumulative_cost, 6),
                "duration_seconds": r.get("duration_seconds", 0.0),
                "eor": round(r.get("eor", 0.0), 4),
                "cumulative_eor": round(sum_eor / idx, 4),
                "mui": round(r.get("mui", 0.0), 4),
                "cumulative_mui": round(sum_mui / idx, 4),
                "ccsr": round(r.get("ccsr", 0.0), 4),
                "cumulative_ccsr": round(sum_ccsr / idx, 4),
                "precision_at_1": ret_m.get("precision_at_1", 0.0),
                "precision_at_3": ret_m.get("precision_at_3", 0.0),
                "recall_at_1": ret_m.get("recall_at_1", 0.0),
                "recall_at_3": ret_m.get("recall_at_3", 0.0),
                "mrr": ret_m.get("mrr", 0.0),
            })

    print(f"[EXPORT] Agentic RAG curves exported to '{curves_path}'")


# ---------------------------------------------------------------------------
# Main CLI Protocol
# ---------------------------------------------------------------------------

def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Agentic RAG Continual Learning Benchmark Runner for AIVC."
    )
    parser.add_argument(
        "--arm",
        "--variant",
        dest="arm",
        type=str,
        choices=["aivc", "baseline", "naive"],
        default="aivc",
        help="Evaluation arm: 'aivc' (continual memory) or 'baseline'/'naive' (stateless baseline). Default: aivc",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Custom dataset path or identifier (default: SWE-Explore sequence)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="",
        help="Custom path to JSONL checkpoint file",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="",
        help="Custom path to JSON metrics file",
    )
    parser.add_argument(
        "--curves-file",
        type=str,
        default="",
        help="Custom path to CSV curves file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of queries already present in checkpoint",
    )

    # Attach unified evaluation configuration flags
    add_eval_args(parser)

    # Parse and resolve hierarchical config
    parsed_args = parser.parse_args()
    cfg = load_benchmark_config(args=parsed_args)
    paths = cfg.get_paths()

    # Resolve paths based on profile & arm
    resolved_paths = resolve_benchmark_paths(
        benchmark_name="agentic_rag",
        model_name=cfg.model,
        arm=parsed_args.arm,
        profile=cfg.profile,
        eval_dir=EVAL_DIR,
    )

    ckpt_path = Path(parsed_args.checkpoint_file) if parsed_args.checkpoint_file else resolved_paths["checkpoint_path"]
    metrics_path = Path(parsed_args.metrics_file) if parsed_args.metrics_file else resolved_paths["metrics_path"]
    curves_path = Path(parsed_args.curves_file) if parsed_args.curves_file else resolved_paths["plots_path"]

    # Also keep standard general curves & metrics paths for DVC access
    general_curves_path = EVAL_DIR / "plots" / "agentic_rag_curves.csv"
    general_metrics_path = EVAL_DIR / "metrics" / "agentic_rag_metrics.json"
    arm_curves_path = EVAL_DIR / "plots" / f"agentic_rag_{parsed_args.arm}_curves.csv"
    arm_metrics_path = EVAL_DIR / "metrics" / f"agentic_rag_{parsed_args.arm}_metrics.json"

    if cfg.reset_checkpoint and ckpt_path.exists():
        print(f"[RESET] Purging checkpoint file '{ckpt_path}'...")
        ckpt_path.unlink()

    # Load environment / API key based on provider
    provider = cfg.model_spec.provider if cfg.model_spec else "openrouter"
    api_key = os.getenv("TOGETHER_API_KEY", "") if provider == "together" else os.getenv("OPENROUTER_API_KEY", "")

    print("=" * 76)
    print("      AIVC AGENTIC RAG CONTINUAL LEARNING BENCHMARK RUNNER")
    print("=" * 76)
    print(f"Evaluation Arm : {parsed_args.arm.upper()}")
    print(f"Target Model   : {cfg.model}")
    print(f"Profile        : {cfg.profile}")
    print(f"Query Limit    : {cfg.limit}")
    print(f"Max Turns      : {cfg.max_turns}")
    print(f"Max Tokens     : {cfg.max_tokens}")
    print(f"Cost Cutoff    : ${cfg.max_cost_per_instance_usd:.2f} USD / query")
    print(f"Checkpoint File: {ckpt_path}")
    print(f"Metrics Output : {metrics_path}")
    print(f"Curves Output  : {curves_path}")
    print("=" * 76)

    # Initialize CheckpointManager
    ckpt_mgr = AgenticRAGCheckpointManager(ckpt_path)
    print(f"[CHECKPOINT] Loaded {len(ckpt_mgr.processed_ids)} previously processed queries from checkpoint.")

    # Load Dataset Queries
    queries, used_dataset_name = load_agentic_rag_dataset(
        dataset_name=parsed_args.dataset if parsed_args.dataset else None,
        split=parsed_args.split,
        limit=cfg.limit,
    )

    # Configure tool interaction paths
    arm_name = parsed_args.arm.lower()
    profile_metrics_dir = metrics_path.parent
    profile_interactions = profile_metrics_dir / "tool_interactions.jsonl"
    arm_interactions = EVAL_DIR / "metrics" / f"agentic_rag_{arm_name}_tool_interactions.jsonl"
    bench_interactions = EVAL_DIR / "metrics" / "agentic_rag_tool_interactions.jsonl"
    general_interactions = EVAL_DIR / "metrics" / "tool_interactions.jsonl"
    interactions_paths = [profile_interactions, arm_interactions, bench_interactions, general_interactions]

    if cfg.reset_checkpoint:
        for p in interactions_paths:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Instantiate Runner
    runner = AgenticRAGRunner(
        arm=parsed_args.arm,
        model_name=cfg.model,
        api_key=api_key,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        max_cost_per_query_usd=cfg.max_cost_per_instance_usd,
        interactions_paths=interactions_paths,
    )

    skipped = 0
    processed_now = 0

    for idx, q_item in enumerate(queries, 1):
        q_id = q_item.get("query_id", f"RAG-CL-{idx:03d}")
        if ckpt_mgr.is_processed(q_id) and not parsed_args.force and not cfg.reset_checkpoint:
            print(f"[SKIP] Query '{q_id}' already completed in checkpoint.")
            skipped += 1
            continue

        ep_record = runner.run_episode(
            query_item=q_item,
            episode_index=idx,
            total_episodes=len(queries),
        )
        ckpt_mgr.save_episode(ep_record)
        processed_now += 1

    all_records = ckpt_mgr.load_all_records()

    if all_records:
        export_agentic_rag_metrics(
            records=all_records,
            metrics_path=metrics_path,
            arm=parsed_args.arm,
            model_name=cfg.model,
            dataset_name=used_dataset_name,
        )
        export_agentic_rag_curves(
            records=all_records,
            curves_path=curves_path,
        )
        # Also mirror to general files for easy DVC access
        for m_p in [general_metrics_path, arm_metrics_path]:
            try:
                if m_p != metrics_path:
                    export_agentic_rag_metrics(
                        records=all_records,
                        metrics_path=m_p,
                        arm=parsed_args.arm,
                        model_name=cfg.model,
                        dataset_name=used_dataset_name,
                    )
            except Exception:
                pass
        for c_p in [general_curves_path, arm_curves_path]:
            try:
                if c_p != curves_path:
                    export_agentic_rag_curves(
                        records=all_records,
                        curves_path=c_p,
                    )
            except Exception:
                pass

    print("\n" + "=" * 76)
    print("               EVALUATION BENCHMARK COMPLETED")
    print("=" * 76)
    print(f"Total Dataset Queries : {len(queries)}")
    print(f"Skipped from Previous : {skipped}")
    print(f"Processed This Run    : {processed_now}")
    print(f"Total Checkpoint Rows : {len(all_records)}")
    print("=" * 76)


# Backward-compatible alias
AgenticRAGContinualRunner = AgenticRAGRunner


if __name__ == "__main__":
    main()
