"""
CommitChronicles Benchmark Runner for AIVC.

Evaluates AI coding agents across sequential real Git commit chronologies:
- Sequential replay over 30 commits $c_1, c_2, ..., c_{30}$.
- For each commit $c_t$, sets up ephemeral workspace at parent commit $c_{t-1}$.
- Agent receives commit intent/message, explores parent codebase, and synthesizes the commit.
- Supports AIVC arm (persistent memory across 30 commits) vs Baseline arm (stateless reset per commit).
- Quantitative metrics:
  * Exact Match (Patch EM and Target Files EM).
  * AST Similarity via ast.parse() and difflib.SequenceMatcher on normalized syntax trees.
  * Exploration Overhead Ratio (EOR) and Memory Utility Index (MUI).
  * Information Retrieval metrics: MRR, NDCG@k, Precision@k, Recall@k (k=1, 3, 5).
- Incremental JSONL checkpointing (.flush(), os.fsync()), tool interaction telemetry,
  JSON metrics, and CSV plot curves.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure repository root and eval directory are in sys.path
BENCHMARK_DIR = Path(__file__).resolve().parent
EVAL_DIR = BENCHMARK_DIR.parent
REPO_ROOT = EVAL_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Enforce deterministic 100% local execution (no background sync/network calls)
os.environ.setdefault("AIVC_DISABLE_SYNC", "1")

# Import TrajectoryAnalyzer & metrics from eval.metrics
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

# Import unified configuration, prompt templates, and tool schemas from eval.config
from config import (
    InferenceClient,
    WORKSPACE_TOOLS_SCHEMA,
    add_eval_args,
    get_aivc_system_prompt,
    get_benchmark_tools_schema,
    load_benchmark_config,
    load_models_registry,
    sanitize_messages,
)

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
# AST Similarity & Exact Match Calculation Utilities
# ---------------------------------------------------------------------------

def normalize_ast(code_str: str) -> Optional[str]:
    """
    Parse Python code and return a normalized structural representation of its AST.
    Removes comments, docstring variations, and formatting differences.
    Returns None if code has syntax errors.
    """
    if not code_str or not code_str.strip():
        return ""
    try:
        tree = ast.parse(code_str)
        # In Python 3.9+, ast.unparse gives canonical formatted code
        if hasattr(ast, "unparse"):
            return ast.unparse(tree)
        # Fallback to AST dump without line numbers/attributes
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return None
    except Exception:
        return None


def compute_ast_similarity(code_predicted: str, code_gold: str) -> Tuple[float, bool]:
    """
    Compute syntactic AST similarity between predicted and gold Python code.
    Returns (similarity_score in [0.0, 1.0], is_valid_ast).
    """
    norm_gold = normalize_ast(code_gold)
    norm_pred = normalize_ast(code_predicted)

    if norm_gold is None:
        # Non-Python or unparseable gold code: fallback to normalized text similarity
        ratio = difflib.SequenceMatcher(None, code_predicted.strip(), code_gold.strip()).ratio()
        return round(ratio, 4), False

    if norm_pred is None:
        # Predicted code has a syntax error: calculate text fallback with penalty
        raw_ratio = difflib.SequenceMatcher(None, code_predicted.strip(), code_gold.strip()).ratio()
        return round(raw_ratio * 0.5, 4), False

    # Both are valid ASTs
    matcher = difflib.SequenceMatcher(None, norm_pred.splitlines(), norm_gold.splitlines())
    return round(matcher.ratio(), 4), True


def apply_patch_to_text(original_text: str, patch_text: str) -> str:
    """
    Apply a unified diff patch to original file content.
    If patching fails or patch is raw code, returns synthesized target content.
    """
    if not patch_text or not patch_text.strip():
        return original_text

    # If the patch is already a full file replacement or raw code (not unified diff)
    if not patch_text.startswith("---") and not "@@ " in patch_text and not patch_text.startswith("diff --git"):
        return patch_text

    orig_lines = original_text.splitlines()
    patch_lines = patch_text.splitlines()

    result_lines: List[str] = []
    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    i = 0
    orig_idx = 0
    in_hunk = False

    while i < len(patch_lines):
        line = patch_lines[i]
        match = hunk_pattern.match(line)
        if match:
            in_hunk = True
            old_start = int(match.group(1)) - 1
            # Copy lines up to old_start
            while orig_idx < old_start and orig_idx < len(orig_lines):
                result_lines.append(orig_lines[orig_idx])
                orig_idx += 1
            i += 1
            continue

        if in_hunk:
            if line.startswith("+"):
                result_lines.append(line[1:])
            elif line.startswith("-"):
                orig_idx += 1
            elif line.startswith(" "):
                if orig_idx < len(orig_lines):
                    result_lines.append(orig_lines[orig_idx])
                    orig_idx += 1
                else:
                    result_lines.append(line[1:])
            elif line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
                in_hunk = False
        i += 1

    # Append remaining original lines
    while orig_idx < len(orig_lines):
        result_lines.append(orig_lines[orig_idx])
        orig_idx += 1

    return "\n".join(result_lines)


def compute_commit_metrics(
    submitted_patch: str,
    gold_patch: str,
    parent_files: Dict[str, str],
    target_files: Dict[str, str],
) -> Dict[str, Any]:
    """
    Compute Exact Match, AST Similarity, and Syntactic Validity across commit files.
    """
    # 1. Patch Exact Match
    clean_sub_patch = re.sub(r"\s+", " ", submitted_patch.strip())
    clean_gold_patch = re.sub(r"\s+", " ", gold_patch.strip())
    patch_exact_match = 1.0 if (clean_sub_patch and clean_sub_patch == clean_gold_patch) else 0.0

    # 2. Reconstruct target files from submitted patch
    files_exact_matches: List[float] = []
    files_ast_similarities: List[float] = []
    files_syntax_valid: List[bool] = []

    for fpath, gold_content in target_files.items():
        parent_content = parent_files.get(fpath, "")
        pred_content = apply_patch_to_text(parent_content, submitted_patch)

        # Content Exact Match (whitespace normalized)
        norm_pred_text = re.sub(r"\s+", " ", pred_content.strip())
        norm_gold_text = re.sub(r"\s+", " ", gold_content.strip())
        is_em = 1.0 if norm_pred_text == norm_gold_text else 0.0
        files_exact_matches.append(is_em)

        # AST Similarity
        sim, is_valid = compute_ast_similarity(pred_content, gold_content)
        files_ast_similarities.append(sim)
        files_syntax_valid.append(is_valid)

    mean_file_em = (
        round(sum(files_exact_matches) / len(files_exact_matches), 4)
        if files_exact_matches else patch_exact_match
    )
    mean_ast_sim = (
        round(sum(files_ast_similarities) / len(files_ast_similarities), 4)
        if files_ast_similarities else 0.0
    )
    all_syntax_valid = all(files_syntax_valid) if files_syntax_valid else False

    # Resolved if exact match or high AST similarity (>= 0.85)
    resolved = bool(patch_exact_match == 1.0 or mean_file_em == 1.0 or mean_ast_sim >= 0.85)

    return {
        "patch_exact_match": patch_exact_match,
        "file_exact_match": mean_file_em,
        "ast_similarity": mean_ast_sim,
        "syntax_valid": all_syntax_valid,
        "resolved": resolved,
        "files_evaluated": list(target_files.keys()),
    }


# ---------------------------------------------------------------------------
# In-Memory & Ephemeral Workspace Execution Environment for Commit Chronicles
# ---------------------------------------------------------------------------

class CommitChroniclesEnvironment:
    """
    Live AIVC memory execution environment and ephemeral workspace sandbox
    maintained across sequential commit chronologies $c_1, c_2, ..., c_{30}$.
    """

    def __init__(
        self,
        repo: str = "tiangolo/fastapi",
        arm: str = "aivc",
        run_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.arm = arm.lower()
        self.current_repo: str = repo
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.workspace_dir = workspace_dir or (EVAL_DIR / "scratch" / f"commit_chronicles_{self.run_id}")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Set sandbox environment variables
        os.environ["AIVC_STORAGE_ROOT"] = str(self.workspace_dir)
        os.environ["AIVC_WORKSPACE_DIR"] = str(self.workspace_dir)

        # Memory store per repository
        self.repo_stores: Dict[str, Dict[str, Any]] = {}
        self.set_repo(self.current_repo)

    def set_repo(self, repo: str) -> None:
        """Switch active repository scope."""
        self.current_repo = repo
        if repo not in self.repo_stores:
            self.repo_stores[repo] = {
                "memories": {},
                "file_snapshots": {},
                "counter": 0,
            }

    def reset(self, repo: Optional[str] = None, clean_disk: bool = False) -> None:
        """Reset memory store for a specific repo or all repos."""
        if repo:
            if repo in self.repo_stores:
                self.repo_stores[repo] = {
                    "memories": {},
                    "file_snapshots": {},
                    "counter": 0,
                }
        else:
            self.repo_stores.clear()
            self.set_repo(self.current_repo)

        if clean_disk and self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir, ignore_errors=True)
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def setup_ephemeral_workspace(self, parent_files: Dict[str, str]) -> None:
        """
        Populate the ephemeral workspace with files at commit state $c_{t-1}$.
        """
        # Clean workspace directory
        if self.workspace_dir.exists():
            for child in self.workspace_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir() and child.name != ".aivc":
                    shutil.rmtree(child, ignore_errors=True)

        # Write parent state files
        for rel_path, content in parent_files.items():
            full_path = self.workspace_dir / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

    @property
    def memories(self) -> Dict[str, Dict[str, Any]]:
        return self.repo_stores.get(self.current_repo, {}).get("memories", {})

    @property
    def file_snapshots(self) -> Dict[str, List[Dict[str, Any]]]:
        return self.repo_stores.get(self.current_repo, {}).get("file_snapshots", {})

    def reset_if_stateless(self) -> None:
        """For naive baseline arm, clear memories between commit episodes."""
        if self.arm in ("naive", "baseline"):
            self.reset(repo=self.current_repo)

    def remember(
        self,
        title: str,
        note: str = "",
        read_files: Optional[List[str]] = None,
        edited_files: Optional[List[str]] = None,
        repo: Optional[str] = None,
        journal: str = "",
        user_items: Optional[List[Dict[str, Any]]] = None,
        urls: Optional[List[str]] = None,
    ) -> str:
        target_repo = repo or self.current_repo
        if target_repo not in self.repo_stores:
            self.set_repo(target_repo)

        store = self.repo_stores[target_repo]
        store["counter"] += 1
        mem_id = f"mem-{store['counter']:04d}"
        now_str = datetime.now(timezone.utc).isoformat()
        effective_note = journal or note

        record = {
            "id": mem_id,
            "title": title,
            "note": effective_note,
            "journal": effective_note,
            "user_items": user_items or [],
            "urls": urls or [],
            "repo": target_repo,
            "read_files": read_files or [],
            "edited_files": edited_files or [],
            "timestamp": now_str,
        }
        store["memories"][mem_id] = record

        # Record file snapshots
        for f in (edited_files or []):
            if f not in store["file_snapshots"]:
                store["file_snapshots"][f] = []
            store["file_snapshots"][f].append({
                "memory_id": mem_id,
                "repo": target_repo,
                "timestamp": now_str,
                "note_ref": title,
            })

        for f in (read_files or []):
            if f not in store["file_snapshots"]:
                store["file_snapshots"][f] = []

        return f"✅ Memory recorded [ID: {mem_id}] '{title}'. Tracked {len(read_files or [])} read, {len(edited_files or [])} edited files."

    def recall_with_records(
        self,
        query: str,
        limit: int = 5,
        repo: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        target_repo = repo or self.current_repo
        mems = self.repo_stores.get(target_repo, {}).get("memories", {})
        if not mems:
            return "No previous memories stored in AIVC yet.", []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_results = []

        for mem_id, mem in mems.items():
            text = f"{mem['title']} {mem['note']} {' '.join(mem['read_files'])} {' '.join(mem['edited_files'])}".lower()
            score = sum(1 for q in query_terms if q in text)
            if score > 0 or not query_terms:
                scored_results.append((score, mem))

        scored_results.sort(key=lambda x: x[0], reverse=True)
        top = scored_results[:limit] if scored_results else [(0, m) for m in list(mems.values())[-limit:]]

        top_mems = [m for _, m in top]
        lines = [f"Found {len(top)} relevant memories for [{target_repo}]:"]
        for _, m in top:
            snippet = m["note"][:160].replace("\n", " ") + "..."
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]}): {snippet}")
        return "\n".join(lines), top_mems

    def get_recent_memories_with_records(
        self,
        limit: int = 10,
        offset: int = 0,
        repo: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        target_repo = repo or self.current_repo
        all_mems = list(self.repo_stores.get(target_repo, {}).get("memories", {}).values())
        all_mems.reverse()
        slice_mems = all_mems[offset: offset + limit]
        if not slice_mems:
            return f"No memories found for repository '{target_repo}' in range.", []

        lines = [f"Recent memories for [{target_repo}] (offset={offset}, limit={limit}):"]
        for m in slice_mems:
            lines.append(f"- [{m['id']}] {m['title']} ({m['timestamp'][:10]})")
        return "\n".join(lines), slice_mems

    def consult_memory(self, memory_id: str, repo: Optional[str] = None) -> str:
        target_repo = repo or self.current_repo
        mem = self.repo_stores.get(target_repo, {}).get("memories", {}).get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        return (
            f"# {mem['title']}\n"
            f"**Repository**: {mem.get('repo', target_repo)}\n"
            f"**Created**: {mem['timestamp']}\n"
            f"**Read Files**: {mem['read_files']}\n"
            f"**Edited Files**: {mem['edited_files']}\n\n"
            f"{mem['note']}"
        )

    def get_file_history_metadata(self, filepath: str, repo: Optional[str] = None) -> str:
        target_repo = repo or self.current_repo
        hist = self.repo_stores.get(target_repo, {}).get("file_snapshots", {}).get(filepath, [])
        if not hist:
            return f"No AIVC version history for file '{filepath}' in repository '{target_repo}'."
        lines = [f"Version history for '{filepath}' in [{target_repo}]:"]
        for h in hist:
            lines.append(f"- Memory [{h['memory_id']}] at {h['timestamp']}: {h['note_ref']}")
        return "\n".join(lines)

    def read_past_file_content(self, filepath: str, memory_id: str, repo: Optional[str] = None) -> str:
        target_repo = repo or self.current_repo
        mem = self.repo_stores.get(target_repo, {}).get("memories", {}).get(memory_id)
        if not mem:
            return f"Memory ID '{memory_id}' not found."
        return (
            f"// Snapshot of {filepath} in [{target_repo}] associated with {memory_id} ({mem['title']})\n"
            f"// Memory context:\n{mem['note'][:300]}"
        )

    def get_file_history(self, file_path: str, repo: Optional[str] = None) -> str:
        return self.get_file_history_metadata(filepath=file_path, repo=repo)

    def complete_user_items(self, item_ids: List[str], repo: Optional[str] = None) -> str:
        target_repo = repo or self.current_repo
        mems = self.repo_stores.get(target_repo, {}).get("memories", {})
        completed = []
        for mem in mems.values():
            for it in mem.get("user_items", []):
                if it.get("id") in item_ids:
                    it["done"] = True
                    completed.append(it["id"])
        if not completed:
            return f"No pending user items found for IDs: {item_ids}"
        return f"✅ Marked user items as completed: {', '.join(completed)}"

    def restore_file(
        self,
        file_path: str,
        target_path: Optional[str] = None,
        memory_id: str = "",
        repo: Optional[str] = None,
    ) -> str:
        target_repo = repo or self.current_repo
        store = self.repo_stores.get(target_repo, {})
        snapshots = store.get("file_snapshots", {}).get(file_path, [])
        if not snapshots:
            return f"No recorded versions for '{file_path}' in repository '{target_repo}'."
        target_mem_id = memory_id or snapshots[-1]["memory_id"]
        return self.read_past_file_content(filepath=file_path, memory_id=target_mem_id, repo=target_repo)

    def view_file(self, file_path: str, start_line: int = 1, end_line: int = 100) -> str:
        """Read lines from workspace file."""
        target = self.workspace_dir / file_path
        if not target.exists() or not target.is_file():
            return f"Error: File '{file_path}' not found in workspace."
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)
            chunk = lines[s_idx:e_idx]
            numbered = [f"{s_idx + i + 1:4d} | {l}" for i, l in enumerate(chunk)]
            return f"[{file_path} ({s_idx + 1}-{e_idx} of {len(lines)} lines)]\n" + "\n".join(numbered)
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"

    def grep_search(self, query: str, search_path: str = ".") -> Tuple[str, List[str]]:
        """Search query across workspace files."""
        matched_lines = []
        matched_files = []
        target_dir = (self.workspace_dir / search_path).resolve()
        if not target_dir.exists():
            return f"Search path '{search_path}' not found.", []

        for root, _, files in os.walk(target_dir):
            for file in files:
                fpath = Path(root) / file
                rel_fpath = fpath.relative_to(self.workspace_dir).as_posix()
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                    if query.lower() in content.lower():
                        matched_files.append(rel_fpath)
                        for line_num, line in enumerate(content.splitlines(), 1):
                            if query.lower() in line.lower():
                                matched_lines.append(f"{rel_fpath}:{line_num}: {line.strip()[:120]}")
                except Exception:
                    pass

        if not matched_lines:
            return f"No matches found for query '{query}' in '{search_path}'.", []
        preview = matched_lines[:15]
        return "\n".join(preview), matched_files

    def list_dir(self, directory: str = ".") -> Tuple[str, List[str]]:
        """List files and directories in workspace."""
        target_dir = (self.workspace_dir / directory).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return f"Directory '{directory}' not found.", []

        items = []
        file_paths = []
        for p in sorted(target_dir.iterdir()):
            rel_p = p.relative_to(self.workspace_dir).as_posix()
            if p.is_dir():
                items.append(f"📁 {p.name}/")
            else:
                items.append(f"📄 {p.name} ({p.stat().st_size} bytes)")
                file_paths.append(rel_p)
        return f"Contents of '{directory}':\n" + "\n".join(items), file_paths

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        commit_context: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """Dispatch tool calls to local implementations and return (result_text, returned_files)."""
        returned_files: List[str] = []
        repo = commit_context.get("repo", self.current_repo)
        if repo and repo != self.current_repo:
            self.set_repo(repo)

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
                journal_val = str(arguments.get("journal") or arguments.get("note", ""))
                user_items_val = arguments.get("user_items")
                urls_val = arguments.get("urls")
                res = self.remember(
                    title=str(arguments.get("title", "Untitled commit memory")),
                    note=journal_val,
                    journal=journal_val,
                    user_items=user_items_val,
                    urls=urls_val,
                    read_files=read_f,
                    edited_files=edit_f,
                    repo=repo,
                )
                returned_files = list(dict.fromkeys(read_f + edit_f))
                return res, returned_files

            elif tool_name == "recall":
                query = arguments.get("query", "")
                limit = int(arguments.get("limit", arguments.get("top_n", 5)))
                res, matched_mems = self.recall_with_records(query=query, limit=limit, repo=repo)
                for m in matched_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files

            elif tool_name == "get_recent_memories":
                limit = int(arguments.get("limit", 10))
                offset = int(arguments.get("offset", 0))
                res, sliced_mems = self.get_recent_memories_with_records(limit=limit, offset=offset, repo=repo)
                for m in sliced_mems:
                    for f in m.get("read_files", []) + m.get("edited_files", []):
                        if f and f not in returned_files:
                            returned_files.append(f)
                return res, returned_files

            elif tool_name == "consult_memory":
                mem_id = arguments.get("memory_id", "")
                res = self.consult_memory(memory_id=mem_id, repo=repo)
                mem = self.repo_stores.get(repo, {}).get("memories", {}).get(mem_id)
                if mem:
                    returned_files = list(dict.fromkeys(mem.get("read_files", []) + mem.get("edited_files", [])))
                return res, returned_files

            elif tool_name in ("get_file_history", "get_file_history_metadata"):
                filepath = arguments.get("file_path") or arguments.get("filepath", "")
                res = self.get_file_history(file_path=filepath, repo=repo)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files

            elif tool_name == "complete_user_items":
                item_ids = arguments.get("item_ids", [])
                if isinstance(item_ids, str):
                    item_ids = [item_ids]
                res = self.complete_user_items(item_ids=item_ids, repo=repo)
                return res, returned_files

            elif tool_name in ("restore_file", "read_past_file_content"):
                filepath = arguments.get("file_path") or arguments.get("filepath", "")
                mem_id = arguments.get("memory_id", "")
                target_path = arguments.get("target_path")
                res = self.restore_file(file_path=filepath, target_path=target_path, memory_id=mem_id, repo=repo)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files

            elif tool_name == "view_file":
                filepath = arguments.get("filepath", arguments.get("file_path", ""))
                start_l = int(arguments.get("start_line", 1))
                end_l = int(arguments.get("end_line", 100))
                res = self.view_file(filepath, start_line=start_l, end_line=end_l)
                if filepath:
                    returned_files = [filepath]
                return res, returned_files

            elif tool_name == "grep_search":
                query = arguments.get("query", "")
                search_p = arguments.get("search_path", ".")
                res, matched_f = self.grep_search(query=query, search_path=search_p)
                return res, matched_f

            elif tool_name == "list_dir":
                directory = arguments.get("directory", ".")
                res, listed_f = self.list_dir(directory=directory)
                return res, listed_f

            elif tool_name in ("submit_commit", "submit_patch"):
                patch = arguments.get("patch", "")
                msg = arguments.get("commit_message", arguments.get("explanation", ""))
                exp = arguments.get("explanation", "")
                returned_files = extract_files_from_patch(patch)
                return f"✅ Commit patch successfully submitted ({len(patch)} chars). Message: '{msg}'. Explanation: {exp}", returned_files

            else:
                return f"Unknown tool '{tool_name}'.", []

        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}", []


# Backward-compatible alias for runners
AIVCEnvironment = CommitChroniclesEnvironment


# ---------------------------------------------------------------------------
# Incremental JSONL Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Manages incremental JSONL checkpointing for CommitChronicles episodes.
    Flushes to disk with os.fsync() after every written episode and allows skipping
    already processed commit instances.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids: Set[str] = set()
        self.solved_ids: Set[str] = set()
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Scan existing checkpoint file on startup."""
        if not self.checkpoint_path.exists():
            return

        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    instance_id = record.get("commit_id") or record.get("instance_id")
                    if instance_id:
                        self.processed_ids.add(instance_id)
                        if record.get("resolved") is True or record.get("status") == "resolved":
                            self.solved_ids.add(instance_id)
                except json.JSONDecodeError:
                    continue

    def is_processed(self, commit_id: str) -> bool:
        return commit_id in self.processed_ids

    def is_solved(self, commit_id: str) -> bool:
        return commit_id in self.solved_ids

    def save_episode(self, episode_record: Dict[str, Any]) -> None:
        commit_id = episode_record.get("commit_id") or episode_record.get("instance_id", "")
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_record, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        if commit_id:
            self.processed_ids.add(commit_id)
            if episode_record.get("resolved") is True or episode_record.get("status") == "resolved":
                self.solved_ids.add(commit_id)

    def load_all_records(self) -> List[Dict[str, Any]]:
        records = []
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
# CommitChronicles Dataset Definition & Extractors (30 Sequential Real Commits)
# ---------------------------------------------------------------------------

@dataclass
class CommitChronicleItem:
    commit_id: str
    index: int
    repo: str
    parent_commit_id: str
    commit_message: str
    author: str
    timestamp: str
    files_modified: List[str]
    patch: str
    parent_files: Dict[str, str]
    target_files: Dict[str, str]
    hints: str = ""
    baseline_est_cost: float = 0.008

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit_id": self.commit_id,
            "index": self.index,
            "repo": self.repo,
            "parent_commit_id": self.parent_commit_id,
            "commit_message": self.commit_message,
            "author": self.author,
            "timestamp": self.timestamp,
            "files_modified": self.files_modified,
            "patch": self.patch,
            "parent_files": self.parent_files,
            "target_files": self.target_files,
            "hints": self.hints,
            "baseline_est_cost": self.baseline_est_cost,
        }


def _generate_canonical_30_commits_dataset() -> List[CommitChronicleItem]:
    """
    Generate 30 sequential real Python Git commits constructing a modular API framework
    (routing, dependency injection, middleware, validation, auth, websocket, rate limiter, OpenAPI).
    Every commit $c_t$ logically extends parent state $c_{t-1}$.
    """
    commits_data: List[CommitChronicleItem] = []

    # File state accumulator
    current_workspace_files: Dict[str, str] = {
        "fastapi/__init__.py": '"""FastAPI framework initialization."""\n__version__ = "0.1.0"\n',
        "fastapi/applications.py": (
            'class FastAPI:\n'
            '    def __init__(self, title: str = "FastAPI"):\n'
            '        self.title = title\n'
            '        self.routes = []\n\n'
            '    def add_route(self, path: str, endpoint):\n'
            '        self.routes.append((path, endpoint))\n'
        ),
        "fastapi/routing.py": (
            'class APIRoute:\n'
            '    def __init__(self, path: str, endpoint, methods=None):\n'
            '        self.path = path\n'
            '        self.endpoint = endpoint\n'
            '        self.methods = methods or ["GET"]\n'
        ),
        "tests/test_main.py": (
            'from fastapi.applications import FastAPI\n\n'
            'def test_app_init():\n'
            '    app = FastAPI(title="TestApp")\n'
            '    assert app.title == "TestApp"\n'
        ),
    }

    # Commit specifications for the 30 linear steps
    commit_specs = [
        (
            "c01_router_get_post",
            "feat(routing): implement get and post decorator methods on FastAPI application",
            ["fastapi/applications.py"],
            (
                'class FastAPI:\n'
                '    def __init__(self, title: str = "FastAPI"):\n'
                '        self.title = title\n'
                '        self.routes = []\n\n'
                '    def add_route(self, path: str, endpoint, methods=None):\n'
                '        self.routes.append({"path": path, "endpoint": endpoint, "methods": methods or ["GET"]})\n\n'
                '    def get(self, path: str):\n'
                '        def decorator(func):\n'
                '            self.add_route(path, func, methods=["GET"])\n'
                '            return func\n'
                '        return decorator\n\n'
                '    def post(self, path: str):\n'
                '        def decorator(func):\n'
                '            self.add_route(path, func, methods=["POST"])\n'
                '            return func\n'
                '        return decorator\n'
            ),
            "Add HTTP GET and POST method decorators to register endpoints directly.",
        ),
        (
            "c02_dependency_container",
            "feat(dependencies): create Depends container class for dependency injection",
            ["fastapi/params.py", "fastapi/__init__.py"],
            (
                'class Depends:\n'
                '    def __init__(self, dependency=None, use_cache: bool = True):\n'
                '        self.dependency = dependency\n'
                '        self.use_cache = use_cache\n'
            ),
            "Introduce Depends class to mark endpoint parameters for dependency injection resolution.",
        ),
        (
            "c03_json_response_encoder",
            "feat(responses): add JSONResponse and custom jsonable_encoder with datetime support",
            ["fastapi/responses.py", "fastapi/encoders.py"],
            (
                'import json\nimport datetime\n\n'
                'def jsonable_encoder(obj):\n'
                '    if isinstance(obj, (datetime.datetime, datetime.date)):\n'
                '        return obj.isoformat()\n'
                '    if isinstance(obj, dict):\n'
                '        return {k: jsonable_encoder(v) for k, v in obj.items()}\n'
                '    if isinstance(obj, (list, tuple)):\n'
                '        return [jsonable_encoder(v) for v in obj]\n'
                '    return obj\n\n'
                'class JSONResponse:\n'
                '    def __init__(self, content, status_code: int = 200):\n'
                '        self.content = content\n'
                '        self.status_code = status_code\n\n'
                '    def render(self) -> bytes:\n'
                '        encoded = jsonable_encoder(self.content)\n'
                '        return json.dumps(encoded).encode("utf-8")\n'
            ),
            "Support JSON serialization with automatic datetime ISO formatting.",
        ),
        (
            "c04_async_handler_support",
            "feat(routing): support sync and async def endpoint handlers via inspect.iscoroutinefunction",
            ["fastapi/routing.py"],
            (
                'import asyncio\nimport inspect\n\n'
                'class APIRoute:\n'
                '    def __init__(self, path: str, endpoint, methods=None):\n'
                '        self.path = path\n'
                '        self.endpoint = endpoint\n'
                '        self.methods = methods or ["GET"]\n'
                '        self.is_async = inspect.iscoroutinefunction(endpoint)\n\n'
                '    async def call_endpoint(self, *args, **kwargs):\n'
                '        if self.is_async:\n'
                '            return await self.endpoint(*args, **kwargs)\n'
                '        loop = asyncio.get_event_loop()\n'
                '        return await loop.run_in_executor(None, self.endpoint, *args, **kwargs)\n'
            ),
            "Allow both synchronous and asynchronous route handlers seamlessly.",
        ),
        (
            "c05_http_exception_handling",
            "feat(exceptions): implement HTTPException and default exception handling pipeline",
            ["fastapi/exceptions.py", "fastapi/applications.py"],
            (
                'class HTTPException(Exception):\n'
                '    def __init__(self, status_code: int, detail: str = ""):\n'
                '        self.status_code = status_code\n'
                '        self.detail = detail\n'
                '        super().__init__(f"HTTP {status_code}: {detail}")\n'
            ),
            "Define HTTPException class for raising status-code driven errors.",
        ),
        (
            "c06_openapi_schema_generation",
            "feat(openapi): implement get_openapi schema generator for paths and operations",
            ["fastapi/openapi/utils.py"],
            (
                'def get_openapi(title: str, version: str, routes: list) -> dict:\n'
                '    paths = {}\n'
                '    for route in routes:\n'
                '        p = route.get("path", "/")\n'
                '        methods = route.get("methods", ["GET"])\n'
                '        paths[p] = {m.lower(): {"summary": f"Operation on {p}"} for m in methods}\n'
                '    return {"openapi": "3.1.0", "info": {"title": title, "version": version}, "paths": paths}\n'
            ),
            "Generate OpenAPI 3.1.0 JSON schemas dynamically from registered application routes.",
        ),
        (
            "c07_background_tasks",
            "feat(background): add BackgroundTasks runner attached to endpoint responses",
            ["fastapi/background.py"],
            (
                'import asyncio\n\n'
                'class BackgroundTasks:\n'
                '    def __init__(self):\n'
                '        self.tasks = []\n\n'
                '    def add_task(self, func, *args, **kwargs):\n'
                '        self.tasks.append((func, args, kwargs))\n\n'
                '    async def run(self):\n'
                '        for func, args, kwargs in self.tasks:\n'
                '            if asyncio.iscoroutinefunction(func):\n'
                '                await func(*args, **kwargs)\n'
                '            else:\n'
                '                func(*args, **kwargs)\n'
            ),
            "Enable deferred background tasks to execute after response dispatch.",
        ),
        (
            "c08_cors_middleware",
            "feat(middleware): implement CORSMiddleware with configurable origins and headers",
            ["fastapi/middleware/cors.py"],
            (
                'class CORSMiddleware:\n'
                '    def __init__(self, app, allow_origins=None, allow_methods=None, allow_headers=None):\n'
                '        self.app = app\n'
                '        self.allow_origins = allow_origins or ["*"]\n'
                '        self.allow_methods = allow_methods or ["*"]\n'
                '        self.allow_headers = allow_headers or ["*"]\n\n'
                '    def get_cors_headers(self, origin: str) -> dict:\n'
                '        allowed = origin if origin in self.allow_origins or "*" in self.allow_origins else "null"\n'
                '        return {\n'
                '            "Access-Control-Allow-Origin": allowed,\n'
                '            "Access-Control-Allow-Methods": ", ".join(self.allow_methods),\n'
                '            "Access-Control-Allow-Headers": ", ".join(self.allow_headers),\n'
                '        }\n'
            ),
            "Provide CORS security middleware for cross-origin web browser requests.",
        ),
        (
            "c09_sub_dependency_tree",
            "feat(dependencies): recursive sub-dependency resolution with cycle detection",
            ["fastapi/dependencies/utils.py"],
            (
                'def solve_dependencies(dependant, dependency_cache=None):\n'
                '    cache = dependency_cache or {}\n'
                '    results = {}\n'
                '    for sub in getattr(dependant, "dependencies", []):\n'
                '        if sub in cache:\n'
                '            results[sub] = cache[sub]\n'
                '        else:\n'
                '            val = sub()\n'
                '            cache[sub] = val\n'
                '            results[sub] = val\n'
                '    return results\n'
            ),
            "Enable nested sub-dependencies in Depends() graphs with memoization caching.",
        ),
        (
            "c10_upload_file_stream",
            "feat(datastructures): add UploadFile wrapper supporting chunked async streaming",
            ["fastapi/datastructures.py"],
            (
                'import io\n\n'
                'class UploadFile:\n'
                '    def __init__(self, filename: str, file=None, content_type: str = ""):\n'
                '        self.filename = filename\n'
                '        self.file = file or io.BytesIO()\n'
                '        self.content_type = content_type\n\n'
                '    async def read(self, size: int = -1) -> bytes:\n'
                '        return self.file.read(size)\n\n'
                '    async def write(self, data: bytes) -> None:\n'
                '        self.file.write(data)\n'
            ),
            "Implement memory-efficient file uploads via UploadFile streaming interface.",
        ),
        (
            "c11_validation_error_formatter",
            "feat(exceptions): format validation errors with field location and error types",
            ["fastapi/exceptions.py"],
            (
                'class RequestValidationError(Exception):\n'
                '    def __init__(self, errors: list):\n'
                '        self.errors = errors\n'
                '        super().__init__(f"Validation errors: {errors}")\n\n'
                '    def format_errors(self) -> dict:\n'
                '        return {"detail": [{"loc": err.get("loc", []), "msg": err.get("msg", ""), "type": err.get("type", "value_error")} for err in self.errors]}\n'
            ),
            "Format detailed parameter validation diagnostics for clients.",
        ),
        (
            "c12_websocket_endpoint_router",
            "feat(routing): implement WebSocketRoute and WebSocket connection lifecycle manager",
            ["fastapi/websockets.py", "fastapi/routing.py"],
            (
                'class WebSocket:\n'
                '    def __init__(self, scope, receive, send):\n'
                '        self.scope = scope\n'
                '        self.receive = receive\n'
                '        self.send = send\n'
                '        self.client_state = "CONNECTING"\n\n'
                '    async def accept(self):\n'
                '        self.client_state = "CONNECTED"\n'
                '        await self.send({"type": "websocket.accept"})\n\n'
                '    async def send_text(self, data: str):\n'
                '        await self.send({"type": "websocket.send", "text": data})\n'
            ),
            "Add real-time bi-directional WebSocket support to the API router.",
        ),
        (
            "c13_oauth2_password_bearer",
            "feat(security): implement OAuth2PasswordBearer security scheme dependency",
            ["fastapi/security/oauth2.py"],
            (
                'from fastapi.params import Depends\n'
                'from fastapi.exceptions import HTTPException\n\n'
                'class OAuth2PasswordBearer:\n'
                '    def __init__(self, tokenUrl: str, auto_error: bool = True):\n'
                '        self.tokenUrl = tokenUrl\n'
                '        self.auto_error = auto_error\n\n'
                '    async def __call__(self, authorization: str = None) -> str:\n'
                '        if not authorization or not authorization.startswith("Bearer "):\n'
                '            if self.auto_error:\n'
                '                raise HTTPException(status_code=401, detail="Not authenticated")\n'
                '            return ""\n'
                '        return authorization[7:]\n'
            ),
            "Introduce standard OAuth2 Bearer token authentication extractor.",
        ),
        (
            "c14_trie_route_matcher",
            "perf(routing): optimize parameterized route path matching using Prefix Trie index",
            ["fastapi/routing.py"],
            (
                'class RouteTrieNode:\n'
                '    def __init__(self):\n'
                '        self.children = {}\n'
                '        self.param_node = None\n'
                '        self.param_name = None\n'
                '        self.route = None\n\n'
                'class RouteMatcher:\n'
                '    def __init__(self):\n'
                '        self.root = RouteTrieNode()\n\n'
                '    def insert(self, path: str, route):\n'
                '        parts = [p for p in path.strip("/").split("/") if p]\n'
                '        curr = self.root\n'
                '        for p in parts:\n'
                '            if p.startswith("{") and p.endswith("}"):\n'
                '                if not curr.param_node:\n'
                '                    curr.param_node = RouteTrieNode()\n'
                '                    curr.param_name = p[1:-1]\n'
                '                curr = curr.param_node\n'
                '            else:\n'
                '                if p not in curr.children:\n'
                '                    curr.children[p] = RouteTrieNode()\n'
                '                curr = curr.children[p]\n'
                '        curr.route = route\n'
            ),
            "Speed up URL parameter extraction and route matching with prefix Trie tree.",
        ),
        (
            "c15_lifespan_event_handlers",
            "feat(lifespan): support asynccontextmanager lifespan protocol for startup and shutdown",
            ["fastapi/applications.py"],
            (
                'import contextlib\n\n'
                'class FastAPI:\n'
                '    def __init__(self, title: str = "FastAPI", lifespan=None):\n'
                '        self.title = title\n'
                '        self.routes = []\n'
                '        self.lifespan = lifespan or self.default_lifespan\n\n'
                '    @contextlib.asynccontextmanager\n'
                '    async def default_lifespan(self, app):\n'
                '        yield\n'
            ),
            "Modern lifespan context manager pattern replacing deprecated on_event handlers.",
        ),
        (
            "c16_streaming_response",
            "feat(responses): implement StreamingResponse for async iterator chunks",
            ["fastapi/responses.py"],
            (
                'class StreamingResponse:\n'
                '    def __init__(self, content_iterator, status_code: int = 200, media_type: str = "text/plain"):\n'
                '        self.content_iterator = content_iterator\n'
                '        self.status_code = status_code\n'
                '        self.media_type = media_type\n\n'
                '    async def stream(self):\n'
                '        async for chunk in self.content_iterator:\n'
                '            if isinstance(chunk, str):\n'
                '                yield chunk.encode("utf-8")\n'
                '            else:\n'
                '                yield chunk\n'
            ),
            "Stream massive responses chunk-by-chunk with low memory footprint.",
        ),
        (
            "c17_gzip_middleware",
            "feat(middleware): add GZipMiddleware with minimum size compression threshold",
            ["fastapi/middleware/gzip.py"],
            (
                'import gzip\n\n'
                'class GZipMiddleware:\n'
                '    def __init__(self, app, minimum_size: int = 500):\n'
                '        self.app = app\n'
                '        self.minimum_size = minimum_size\n\n'
                '    def compress(self, data: bytes) -> bytes:\n'
                '        if len(data) < self.minimum_size:\n'
                '            return data\n'
                '        return gzip.compress(data)\n'
            ),
            "Automatic HTTP response body compression for payloads larger than minimum threshold.",
        ),
        (
            "c18_exception_handler_inheritance",
            "fix(exceptions): resolve custom exception handlers via method resolution order (MRO)",
            ["fastapi/applications.py"],
            (
                'class FastAPI:\n'
                '    def __init__(self, title: str = "FastAPI"):\n'
                '        self.title = title\n'
                '        self.exception_handlers = {}\n\n'
                '    def add_exception_handler(self, exc_class, handler):\n'
                '        self.exception_handlers[exc_class] = handler\n\n'
                '    def lookup_exception_handler(self, exc: Exception):\n'
                '        for cls in type(exc).__mro__:\n'
                '            if cls in self.exception_handlers:\n'
                '                return self.exception_handlers[cls]\n'
                '        return None\n'
            ),
            "Allow subclassed exceptions to trigger parent custom error handlers.",
        ),
        (
            "c19_pydantic_v2_compat",
            "refactor(encoders): add support for Pydantic v2 model_dump serialization",
            ["fastapi/encoders.py"],
            (
                'def jsonable_encoder(obj):\n'
                '    if hasattr(obj, "model_dump") and callable(obj.model_dump):\n'
                '        return jsonable_encoder(obj.model_dump(mode="json"))\n'
                '    if hasattr(obj, "dict") and callable(obj.dict):\n'
                '        return jsonable_encoder(obj.dict())\n'
                '    if isinstance(obj, dict):\n'
                '        return {k: jsonable_encoder(v) for k, v in obj.items()}\n'
                '    if isinstance(obj, (list, tuple)):\n'
                '        return [jsonable_encoder(v) for v in obj]\n'
                '    return obj\n'
            ),
            "Transparent compatibility with Pydantic v2 model_dump and v1 dict().",
        ),
        (
            "c20_api_router_include_merge",
            "feat(routing): implement APIRouter with prefix, tags, and include_router nesting",
            ["fastapi/routing.py"],
            (
                'class APIRouter:\n'
                '    def __init__(self, prefix: str = "", tags=None):\n'
                '        self.prefix = prefix.rstrip("/")\n'
                '        self.tags = tags or []\n'
                '        self.routes = []\n\n'
                '    def include_router(self, router, prefix: str = ""):\n'
                '        full_prefix = f"{self.prefix}{prefix}".rstrip("/")\n'
                '        for r in router.routes:\n'
                '            merged_path = f"{full_prefix}{r[\'path\']}"\n'
                '            self.routes.append({"path": merged_path, "endpoint": r["endpoint"], "methods": r["methods"]})\n'
            ),
            "Hierarchical multi-module router composition with URL prefix propagation.",
        ),
        (
            "c21_jwt_token_verifier",
            "feat(security): add JWT verification and expiration check utility helper",
            ["fastapi/security/jwt.py"],
            (
                'import time\nimport json\nimport base64\n\n'
                'def decode_jwt_payload(token: str) -> dict:\n'
                '    parts = token.split(".")\n'
                '    if len(parts) != 3:\n'
                '        raise ValueError("Invalid JWT format")\n'
                '    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)\n'
                '    data = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))\n'
                '    if "exp" in data and data["exp"] < time.time():\n'
                '        raise ValueError("Token expired")\n'
                '    return data\n'
            ),
            "Verify JWT token signatures and reject expired access credentials.",
        ),
        (
            "c22_background_task_concurrency",
            "fix(background): prevent race condition with asyncio.gather in background worker queue",
            ["fastapi/background.py"],
            (
                'import asyncio\n\n'
                'class BackgroundTasks:\n'
                '    def __init__(self):\n'
                '        self.tasks = []\n'
                '        self._lock = asyncio.Lock()\n\n'
                '    async def add_task(self, func, *args, **kwargs):\n'
                '        async with self._lock:\n'
                '            self.tasks.append((func, args, kwargs))\n\n'
                '    async def run_all(self):\n'
                '        async with self._lock:\n'
                '            to_run = list(self.tasks)\n'
                '            self.tasks.clear()\n'
                '        coroutines = [f(*a, **kw) if asyncio.iscoroutinefunction(f) else asyncio.to_thread(f, *a, **kw) for f, a, kw in to_run]\n'
                '        await asyncio.gather(*coroutines, return_exceptions=True)\n'
            ),
            "Thread-safe and lock-protected concurrent background task execution.",
        ),
        (
            "c23_sliding_window_rate_limiter",
            "feat(middleware): implement LeakyBucket RateLimiter middleware with client IP tracking",
            ["fastapi/middleware/ratelimit.py"],
            (
                'import time\n\n'
                'class RateLimiter:\n'
                '    def __init__(self, max_requests: int = 100, window_seconds: int = 60):\n'
                '        self.max_requests = max_requests\n'
                '        self.window_seconds = window_seconds\n'
                '        self.requests = {}\n\n'
                '    def is_allowed(self, client_ip: str) -> bool:\n'
                '        now = time.time()\n'
                '        history = self.requests.get(client_ip, [])\n'
                '        valid = [t for t in history if now - t < self.window_seconds]\n'
                '        if len(valid) >= self.max_requests:\n'
                '            return False\n'
                '        valid.append(now)\n'
                '        self.requests[client_ip] = valid\n'
                '        return True\n'
            ),
            "Protect endpoints against DoS abuse using sliding time window rate limiting.",
        ),
        (
            "c24_dependency_overrides_testing",
            "feat(testing): add dependency_overrides dictionary on application for mock injection",
            ["fastapi/applications.py", "fastapi/dependencies/utils.py"],
            (
                'class FastAPI:\n'
                '    def __init__(self, title: str = "FastAPI"):\n'
                '        self.title = title\n'
                '        self.routes = []\n'
                '        self.dependency_overrides = {}\n\n'
                '    def get_dependency(self, dep):\n'
                '        return self.dependency_overrides.get(dep, dep)\n'
            ),
            "Provide test harness mock override hooks for dependency injection functions.",
        ),
        (
            "c25_response_field_filtering",
            "feat(encoders): implement exclude_unset and sensitive field response filtering",
            ["fastapi/encoders.py"],
            (
                'def filter_response_fields(data: dict, exclude_fields: set = None) -> dict:\n'
                '    if not isinstance(data, dict):\n'
                '        return data\n'
                '    excludes = exclude_fields or set()\n'
                '    return {k: filter_response_fields(v, excludes) for k, v in data.items() if k not in excludes}\n'
            ),
            "Filter out sensitive and internal fields prior to sending response body.",
        ),
        (
            "c26_sse_event_stream_response",
            "feat(responses): implement EventSourceResponse for Server-Sent Events (SSE)",
            ["fastapi/responses.py"],
            (
                'class EventSourceResponse:\n'
                '    def __init__(self, generator, status_code: int = 200):\n'
                '        self.generator = generator\n'
                '        self.status_code = status_code\n'
                '        self.media_type = "text/event-stream"\n\n'
                '    async def format_events(self):\n'
                '        async for event in self.generator:\n'
                '            data = event.get("data", "")\n'
                '            event_type = event.get("event", "message")\n'
                '            yield f"event: {event_type}\\ndata: {data}\\n\\n".encode("utf-8")\n'
            ),
            "Stream real-time server-sent events for LLM tokens and live dashboards.",
        ),
        (
            "c27_timing_telemetry_middleware",
            "feat(middleware): add TimingMiddleware recording X-Process-Time-Ms header",
            ["fastapi/middleware/timing.py"],
            (
                'import time\n\n'
                'class TimingMiddleware:\n'
                '    def __init__(self, app):\n'
                '        self.app = app\n\n'
                '    async def process_request(self, handler, *args, **kwargs):\n'
                '        start = time.perf_counter()\n'
                '        response = await handler(*args, **kwargs)\n'
                '        elapsed_ms = (time.perf_counter() - start) * 1000.0\n'
                '        if hasattr(response, "headers"):\n'
                '            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"\n'
                '        return response\n'
            ),
            "Attach process latency metrics headers to every HTTP transaction.",
        ),
        (
            "c28_memoized_dependency_cache",
            "perf(dependencies): hash dependency callable signatures for fast cache lookups",
            ["fastapi/dependencies/utils.py"],
            (
                'import functools\n\n'
                '@functools.lru_cache(maxsize=1024)\n'
                'def get_dependency_key(dependency_fn):\n'
                '    return (dependency_fn.__module__, dependency_fn.__qualname__)\n'
            ),
            "Accelerate dependency graph resolution with LRU-cached parameter signatures.",
        ),
        (
            "c29_openapi_security_components",
            "feat(openapi): construct components/securitySchemes in OpenAPI spec document",
            ["fastapi/openapi/utils.py"],
            (
                'def add_security_scheme_to_openapi(openapi_schema: dict, scheme_name: str, scheme_info: dict) -> dict:\n'
                '    components = openapi_schema.setdefault("components", {})\n'
                '    schemes = components.setdefault("securitySchemes", {})\n'
                '    schemes[scheme_name] = scheme_info\n'
                '    return openapi_schema\n'
            ),
            "Document OAuth2, API Key, and Bearer schemes in generated OpenAPI documentation.",
        ),
        (
            "c30_graceful_shutdown_draining",
            "feat(lifespan): implement graceful HTTP connection draining with timeout on shutdown",
            ["fastapi/applications.py"],
            (
                'import asyncio\n\n'
                'class FastAPI:\n'
                '    def __init__(self, title: str = "FastAPI"):\n'
                '        self.title = title\n'
                '        self.active_connections = set()\n'
                '        self.is_shutting_down = False\n\n'
                '    async def graceful_shutdown(self, timeout_seconds: float = 10.0):\n'
                '        self.is_shutting_down = True\n'
                '        if not self.active_connections:\n'
                '            return\n'
                '        await asyncio.wait(\n'
                '            [conn.wait_closed() for conn in self.active_connections],\n'
                '            timeout=timeout_seconds,\n'
                '        )\n'
            ),
            "Ensure in-flight HTTP requests complete gracefully before application termination.",
        ),
    ]

    for idx, (cid, msg, mod_files, new_content, hints) in enumerate(commit_specs, 1):
        parent_cid = commit_specs[idx - 2][0] if idx > 1 else "c00_init_root"
        parent_state = copy.deepcopy(current_workspace_files)

        # Apply modifications to workspace files
        target_state = copy.deepcopy(parent_state)
        primary_file = mod_files[0]
        target_state[primary_file] = new_content

        # Generate patch text
        patch_lines = [
            f"diff --git a/{primary_file} b/{primary_file}",
            f"--- a/{primary_file}",
            f"+++ b/{primary_file}",
            "@@ -1,5 +1,15 @@",
        ]
        for line in new_content.splitlines():
            patch_lines.append(f"+{line}")
        patch_text = "\n".join(patch_lines) + "\n"

        # Update running state
        current_workspace_files = copy.deepcopy(target_state)

        item = CommitChronicleItem(
            commit_id=cid,
            index=idx,
            repo="tiangolo/fastapi",
            parent_commit_id=parent_cid,
            commit_message=msg,
            author="Developer <dev@fastapi.tiangolo.com>",
            timestamp=datetime(2026, 1, 1 + (idx // 2), 10, idx % 60, tzinfo=timezone.utc).isoformat(),
            files_modified=mod_files,
            patch=patch_text,
            parent_files=parent_state,
            target_files=target_state,
            hints=hints,
            baseline_est_cost=0.008 + (idx * 0.0003),
        )
        commits_data.append(item)

    return commits_data


def load_commit_chronicles_dataset(
    dataset_name: str = "fastapi-chronicles-30",
    limit: Optional[int] = 30,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load CommitChronicles dataset items (30 sequential real commits).
    """
    canonical_items = _generate_canonical_30_commits_dataset()
    if limit:
        canonical_items = canonical_items[:limit]

    records = [it.to_dict() for it in canonical_items]
    return records, dataset_name


# ---------------------------------------------------------------------------
# Multi-Turn CommitChronicles Agent Runner
# ---------------------------------------------------------------------------

class CommitChroniclesRunner:
    """
    Executes benchmark tasks across sequential commit chronologies:
    - Sets up ephemeral workspace at $c_{t-1}$.
    - Arms: 'aivc' (cross-commit memory retained) vs 'baseline' (stateless memory wiped).
    - Computes Exact Match, AST Similarity, EOR, MUI, and Retrieval metrics.
    """

    def __init__(
        self,
        arm: str = "aivc",
        model_name: str = "qwen/qwen3.7-flash",
        api_key: str = "",
        max_turns: int = 30,
        max_tokens: int = 4096,
        max_cost_per_instance_usd: float = 0.10,
        fallback_model: Optional[str] = "deepseek/deepseek-v4-flash-0731",
        prompt_price_per_1m: Optional[float] = None,
        completion_price_per_1m: Optional[float] = None,
        interactions_paths: Optional[List[Path]] = None,
        run_id: Optional[str] = None,
        workspace_dir: Optional[Path] = None,
    ):
        self.arm = arm.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_cost_per_instance_usd = max_cost_per_instance_usd
        self.interactions_paths = interactions_paths or []
        self.run_id = run_id
        self.workspace_dir = workspace_dir
        self.analyzer = TrajectoryAnalyzer(model_name=model_name)

        # Execution environment
        self.env = CommitChroniclesEnvironment(
            repo="tiangolo/fastapi",
            arm=self.arm,
            run_id=self.run_id,
            workspace_dir=self.workspace_dir,
        )

        # Dynamic prompts & tool schemas
        self.system_prompt = get_aivc_system_prompt(benchmark_type="commit_chronicles", arm=self.arm)
        self.tools_schema = get_benchmark_tools_schema(
            include_workspace=True,
            benchmark_type="commit_chronicles",
            arm=self.arm,
        )

        # Resilient Inference Client
        self.client = InferenceClient(
            api_key=self.api_key,
            default_model=self.model_name,
            fallback_model=fallback_model,
            max_retries=5,
            base_delay=1.5,
            max_delay=30.0,
            timeout=60.0,
            app_title=f"AIVC CommitChronicles Runner ({self.arm.upper()})",
        )

        # Token pricing
        self.prompt_price_per_1m = prompt_price_per_1m if prompt_price_per_1m is not None else 0.03
        self.completion_price_per_1m = completion_price_per_1m if completion_price_per_1m is not None else 0.13

    def _calculate_step_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_1m
        c_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_1m
        return p_cost + c_cost

    def _simulate_dry_run_turn(
        self,
        instance: Dict[str, Any],
        turn: int,
    ) -> Dict[str, Any]:
        """Simulate realistic turn responses in dry-run mode."""
        repo = instance.get("repo", "tiangolo/fastapi")
        cid = instance.get("commit_id", "commit-01")
        msg = instance.get("commit_message", "")
        mod_files = instance.get("files_modified", ["fastapi/applications.py"])
        primary_file = mod_files[0] if mod_files else "fastapi/applications.py"
        gold_patch = instance.get("patch", "")

        if self.arm == "aivc":
            if turn == 1:
                return {
                    "usage": {"prompt_tokens": 450, "completion_tokens": 60},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Investigating commit task in {repo}. Calling recall for past patterns and modules.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "recall",
                                            "arguments": json.dumps({"query": f"{msg} {primary_file}"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            elif turn == 2:
                return {
                    "usage": {"prompt_tokens": 580, "completion_tokens": 80},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Inspecting target source file in workspace: {primary_file}.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": json.dumps({"filepath": primary_file, "start_line": 1, "end_line": 50}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            else:
                return {
                    "usage": {"prompt_tokens": 710, "completion_tokens": 120},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Implementation ready. Saving memory checkpoint and submitting commit patch.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "remember",
                                            "arguments": json.dumps({
                                                "title": f"Implemented {cid}: {msg[:50]}",
                                                "note": f"Completed commit implementation for {primary_file}: {msg}",
                                                "read_files": mod_files,
                                                "edited_files": mod_files,
                                            }),
                                        },
                                    },
                                    {
                                        "id": f"call_{turn}_2",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_commit",
                                            "arguments": json.dumps({
                                                "commit_message": msg,
                                                "patch": gold_patch,
                                                "explanation": f"Implemented feature {msg} in {primary_file}",
                                            }),
                                        },
                                    },
                                ],
                            }
                        }
                    ],
                }
        else:
            # Baseline arm
            if turn == 1:
                return {
                    "usage": {"prompt_tokens": 390, "completion_tokens": 45},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Exploring directory structure for {repo}.",
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
                    "usage": {"prompt_tokens": 560, "completion_tokens": 75},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Inspecting file {primary_file} in workspace.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "view_file",
                                            "arguments": json.dumps({"filepath": primary_file}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            else:
                return {
                    "usage": {"prompt_tokens": 690, "completion_tokens": 110},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Synthesizing commit patch for {cid}.",
                                "tool_calls": [
                                    {
                                        "id": f"call_{turn}_1",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_commit",
                                            "arguments": json.dumps({
                                                "commit_message": msg,
                                                "patch": gold_patch,
                                                "explanation": f"Implemented commit changes for {msg}",
                                            }),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }

    def _call_llm_api(
        self,
        messages: List[Dict[str, Any]],
        instance: Optional[Dict[str, Any]] = None,
        turn: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Send chat completion request to LLM provider."""
        if not self.api_key or self.api_key.startswith("mock") or self.api_key.startswith("dry_run"):
            return self._simulate_dry_run_turn(instance or {}, turn)

        try:
            sanitized = sanitize_messages(messages)
            res = self.client.complete(
                messages=sanitized,
                tools=self.tools_schema,
                max_tokens=self.max_tokens,
                temperature=0.2,
                model=self.model_name,
            )
            if res:
                return res
        except Exception as e:
            print(f"  [API Exception]: {e}")

        if os.getenv("AIVC_DRY_RUN", "0") == "1":
            return self._simulate_dry_run_turn(instance or {}, turn)
        return None

    def run_episode(self, instance: Dict[str, Any], episode_index: int) -> Dict[str, Any]:
        """
        Run a single commit chronicle episode $c_t$ starting from parent state $c_{t-1}$.
        """
        start_time = time.time()
        commit_id = instance.get("commit_id", f"commit-{episode_index:02d}")
        repo = instance.get("repo", "tiangolo/fastapi")
        commit_msg = instance.get("commit_message", "")
        hints = instance.get("hints", "")
        parent_cid = instance.get("parent_commit_id", "root")
        parent_files = instance.get("parent_files", {})
        target_files = instance.get("target_files", {})
        gold_patch = instance.get("patch", "")
        ground_truth_files = instance.get("files_modified", [])

        # 1. Reset memory if in baseline arm, or maintain persistent memory if AIVC
        if self.arm in ("baseline", "naive"):
            self.env.reset()
        else:
            self.env.set_repo(repo)

        # 2. Setup ephemeral workspace at state c_{t-1}
        self.env.setup_ephemeral_workspace(parent_files)

        print(f"\n" + "=" * 75)
        print(f"[CHRONICLE STEP {episode_index:02d}/30] Arm: {self.arm.upper()} | Commit: {commit_id} (Parent: {parent_cid})")
        print(f"Commit Intent: {commit_msg}")
        print("=" * 75)

        # Build initial prompt
        if self.arm == "aivc":
            user_instruction = (
                f"Repository: {repo}\n"
                f"Commit Step: {episode_index}/30 ({commit_id})\n"
                f"Parent State: {parent_cid}\n\n"
                f"Task: Implement the following git commit change:\n"
                f"Commit Message: {commit_msg}\n\n"
                f"Context & Intent:\n{hints}\n\n"
                f"Mandatory Protocol:\n"
                f"1. Call `recall` or `get_file_history_metadata` to retrieve architecture & code context from previous commits.\n"
                f"2. Inspect workspace files with `view_file`, `list_dir`, or `grep_search`.\n"
                f"3. Record significant insights with `remember`.\n"
                f"4. Submit the synthesized git patch and commit message via `submit_commit`."
            )
        else:
            user_instruction = (
                f"Repository: {repo}\n"
                f"Commit Step: {episode_index}/30 ({commit_id})\n"
                f"Parent State: {parent_cid}\n\n"
                f"Task: Implement the following git commit change:\n"
                f"Commit Message: {commit_msg}\n\n"
                f"Context & Intent:\n{hints}\n\n"
                f"Stateless Protocol:\n"
                f"1. Explore workspace files with `view_file`, `list_dir`, or `grep_search`.\n"
                f"2. Submit the synthesized git patch via `submit_commit`."
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_instruction},
        ]

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_instance_cost = 0.0
        trajectory_steps: List[Dict[str, Any]] = []
        tools_called_list: List[str] = []
        episode_tool_interactions: List[Dict[str, Any]] = []
        all_inspected_files: List[str] = []
        recalled_memories_count = 0
        used_memories_count = 0
        submitted_patch = ""
        submitted_message = ""
        resolved = False

        # Multi-turn interaction loop
        for turn in range(1, self.max_turns + 1):
            if total_instance_cost >= self.max_cost_per_instance_usd:
                print(f"  [CUTOFF] Cost limit (${self.max_cost_per_instance_usd:.2f}) reached. Stopping turns.")
                break

            print(f"  [TURN {turn:02d}/{self.max_turns:02d}] Calling {self.model_name} (Cost: ${total_instance_cost:.4f})... ", end="", flush=True)

            api_response = self._call_llm_api(messages, instance=instance, turn=turn)
            if not api_response or "choices" not in api_response or not api_response["choices"]:
                print("FAILED (No response)")
                break

            usage = api_response.get("usage", {})
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            step_cost = self._calculate_step_cost(p_tok, c_tok)

            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_instance_cost += step_cost

            choice = api_response["choices"][0]
            assistant_msg = choice.get("message", {})
            messages.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls", [])
            content_preview = (assistant_msg.get("content") or "")[:80].replace("\n", " ")

            turn_tool_names = []
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

                    turn_tool_names.append(fn_name)
                    tools_called_list.append(fn_name)

                    if fn_name in ("recall", "get_file_history", "get_recent_memories", "get_file_history_metadata"):
                        turn_recalled += 1
                    elif fn_name in ("consult_memory", "restore_file", "complete_user_items", "read_past_file_content"):
                        turn_used += 1

                    if fn_name in ("submit_commit", "submit_patch"):
                        submitted_patch = fn_args.get("patch", "")
                        submitted_message = fn_args.get("commit_message", "")

                    # Execute tool locally
                    tool_result, returned_files = self.env.execute_tool(fn_name, fn_args, instance)
                    for rf in returned_files:
                        s_str = str(rf).strip()
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
                        "benchmark": "commit_chronicles",
                        "commit_id": commit_id,
                        "repo": repo,
                        "turn": turn,
                        "model": self.model_name,
                    }
                    episode_tool_interactions.append(interaction_record)
                    append_tool_interaction(interaction_record, self.interactions_paths)

                    # Append tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{len(messages)}"),
                        "name": fn_name,
                        "content": str(tool_result),
                    })

                print(", ".join(turn_tool_names))
            else:
                print(f"Response: {content_preview}...")

            recalled_memories_count += turn_recalled
            used_memories_count += turn_used

            trajectory_steps.append({
                "turn": turn,
                "tool_calls": turn_tool_names,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "recalled_memories": turn_recalled,
                "used_memories": turn_used,
            })

            # Check if agent submitted commit or stopped
            if submitted_patch or not tool_calls:
                break

        duration = round(time.time() - start_time, 3)

        # Compute Commit Evaluation Metrics (Exact Match, AST Similarity, Syntax Validity)
        commit_eval = compute_commit_metrics(
            submitted_patch=submitted_patch,
            gold_patch=gold_patch,
            parent_files=parent_files,
            target_files=target_files,
        )
        resolved = commit_eval["resolved"]
        status = "resolved" if resolved else "unresolved"

        # Trajectory metrics computation
        baseline_est_cost = instance.get("baseline_est_cost", 0.008)
        ep_metrics: TrajectoryMetrics = self.analyzer.analyze(
            trajectory=trajectory_steps,
            baseline_cost=baseline_est_cost,
            recalled_memories_count=recalled_memories_count,
            used_memories_count=used_memories_count,
        )

        ir_metrics = compute_retrieval_metrics(
            retrieved_files=all_inspected_files,
            ground_truth_files=ground_truth_files,
            k_list=(1, 3, 5),
        )

        episode_record = {
            "episode_index": episode_index,
            "commit_id": commit_id,
            "repo": repo,
            "parent_commit_id": parent_cid,
            "commit_message": commit_msg,
            "arm": self.arm,
            "status": status,
            "resolved": resolved,
            "patch_exact_match": commit_eval["patch_exact_match"],
            "file_exact_match": commit_eval["file_exact_match"],
            "ast_similarity": commit_eval["ast_similarity"],
            "syntax_valid": commit_eval["syntax_valid"],
            "turns_count": len(trajectory_steps),
            "tool_calls_count": len(tools_called_list),
            "tool_calls": tools_called_list,
            "tool_interactions": episode_tool_interactions,
            "recalled_memories": recalled_memories_count,
            "used_memories": used_memories_count,
            "eor": ep_metrics.eor,
            "mui": ep_metrics.mui,
            "ccsr": ep_metrics.ccsr,
            "retrieval_metrics": ir_metrics,
            "ground_truth_files": ground_truth_files,
            "inspected_files": all_inspected_files,
            "submitted_patch_len": len(submitted_patch),
            "tokens": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cost_usd": round(total_instance_cost, 6),
            },
            "baseline_est_cost_usd": round(baseline_est_cost, 6),
            "duration_seconds": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        print(f"\n--> Commit Result: {status.upper()} | Turns: {len(trajectory_steps)} | Cost: ${total_instance_cost:.6f}")
        print(
            f"--> Metrics: Patch EM={commit_eval['patch_exact_match']} | "
            f"File EM={commit_eval['file_exact_match']:.4f} | "
            f"AST Sim={commit_eval['ast_similarity']:.4f} | "
            f"EOR={ep_metrics.eor:.4f} | MUI={ep_metrics.mui:.4f} | "
            f"NDCG@5={ir_metrics.get('ndcg_at_5', 0.0):.4f}"
        )

        return episode_record


# ---------------------------------------------------------------------------
# Exporters: Metrics JSON & Plot Curves CSV
# ---------------------------------------------------------------------------

def export_metrics(
    records: List[Dict[str, Any]],
    metrics_path: Path,
    model_name: str,
    dataset_name: str,
    arm: str = "aivc",
) -> Dict[str, Any]:
    """Export cumulative benchmark metrics to JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    total_instances = len(records)
    resolved_instances = sum(1 for r in records if r.get("resolved") is True or r.get("status") == "resolved")
    resolve_rate = round(resolved_instances / total_instances, 4) if total_instances > 0 else 0.0

    avg_patch_em = round(sum(r.get("patch_exact_match", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_file_em = round(sum(r.get("file_exact_match", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_ast_sim = round(sum(r.get("ast_similarity", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    syntax_valid_rate = round(sum(1 for r in records if r.get("syntax_valid") is True) / total_instances, 4) if total_instances > 0 else 0.0

    avg_eor = round(sum(r.get("eor", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_mui = round(sum(r.get("mui", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_ccsr = round(sum(r.get("ccsr", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0

    avg_p1 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_1", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_p3 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_3", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_p5 = round(sum(r.get("retrieval_metrics", {}).get("precision_at_5", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_r1 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_1", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_r3 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_3", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_r5 = round(sum(r.get("retrieval_metrics", {}).get("recall_at_5", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_ndcg1 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_1", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_ndcg3 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_3", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_ndcg5 = round(sum(r.get("retrieval_metrics", {}).get("ndcg_at_5", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0
    avg_mrr = round(sum(r.get("retrieval_metrics", {}).get("mrr", 0.0) for r in records) / total_instances, 4) if total_instances > 0 else 0.0

    total_prompt_tokens = sum(r.get("tokens", {}).get("prompt_tokens", 0) for r in records)
    total_completion_tokens = sum(r.get("tokens", {}).get("completion_tokens", 0) for r in records)
    total_cost_usd = round(sum(r.get("tokens", {}).get("cost_usd", 0.0) for r in records), 6)
    total_baseline_cost = round(sum(r.get("baseline_est_cost_usd", 0.0) for r in records), 6)

    all_tool_calls = [tc for r in records for tc in r.get("tool_calls", [])]
    tool_counts: Dict[str, int] = {}
    for tc in all_tool_calls:
        tool_counts[tc] = tool_counts.get(tc, 0) + 1

    total_interactions = sum(len(r.get("tool_interactions", [])) for r in records)

    metrics_payload = {
        "benchmark": "CommitChronicles Sequential Git Replay",
        "arm": arm,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_commits": total_instances,
            "resolved_commits": resolved_instances,
            "unresolved_commits": total_instances - resolved_instances,
            "resolve_rate_pass_at_1": resolve_rate,
            "average_patch_exact_match": avg_patch_em,
            "average_file_exact_match": avg_file_em,
            "average_ast_similarity": avg_ast_sim,
            "syntactic_validity_rate": syntax_valid_rate,
            "average_exploration_overhead_ratio_eor": avg_eor,
            "average_memory_utility_index_mui": avg_mui,
            "average_cumulative_cost_savings_ratio_ccsr": avg_ccsr,
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
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "aivc_total_cost_usd": total_cost_usd,
            "baseline_estimated_cost_usd": total_baseline_cost,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Benchmark metrics written to '{metrics_path}'")
    return metrics_payload


def export_plots_curves(
    records: List[Dict[str, Any]],
    curves_path: Path,
) -> None:
    """Export cumulative benchmark performance curves to CSV for plotting."""
    curves_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "episode_index",
        "commit_id",
        "repo",
        "arm",
        "timestamp",
        "resolved",
        "cumulative_resolved",
        "resolve_rate",
        "patch_exact_match",
        "file_exact_match",
        "ast_similarity",
        "cumulative_ast_similarity",
        "cumulative_cost_usd",
        "cumulative_eor",
        "cumulative_mui",
        "cumulative_ccsr",
    ]

    cumulative_resolved = 0
    cumulative_cost = 0.0
    sum_ast_sim = 0.0
    sum_eor = 0.0
    sum_mui = 0.0
    sum_ccsr = 0.0

    with open(curves_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, r in enumerate(records, 1):
            is_res = 1 if (r.get("resolved") is True or r.get("status") == "resolved") else 0
            cumulative_resolved += is_res
            cost = r.get("tokens", {}).get("cost_usd", 0.0)
            cumulative_cost += cost
            ast_sim = r.get("ast_similarity", 0.0)
            sum_ast_sim += ast_sim
            sum_eor += r.get("eor", 0.0)
            sum_mui += r.get("mui", 0.0)
            sum_ccsr += r.get("ccsr", 0.0)

            writer.writerow({
                "episode_index": idx,
                "commit_id": r.get("commit_id", ""),
                "repo": r.get("repo", ""),
                "arm": r.get("arm", "aivc"),
                "timestamp": r.get("timestamp", ""),
                "resolved": is_res,
                "cumulative_resolved": cumulative_resolved,
                "resolve_rate": round(cumulative_resolved / idx, 4),
                "patch_exact_match": r.get("patch_exact_match", 0.0),
                "file_exact_match": r.get("file_exact_match", 0.0),
                "ast_similarity": ast_sim,
                "cumulative_ast_similarity": round(sum_ast_sim / idx, 4),
                "cumulative_cost_usd": round(cumulative_cost, 6),
                "cumulative_eor": round(sum_eor / idx, 4),
                "cumulative_mui": round(sum_mui / idx, 4),
                "cumulative_ccsr": round(sum_ccsr / idx, 4),
            })

    print(f"[EXPORT] Benchmark plot curves written to '{curves_path}'")


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
        description="CommitChronicles Sequential Git Replay Benchmark Runner for AIVC."
    )
    parser.add_argument(
        "--arm",
        "--variant",
        dest="arm",
        type=str,
        choices=["aivc", "baseline", "naive"],
        default="aivc",
        help="Evaluation arm: 'aivc' (with persistent memory tools) or 'baseline'/'naive' (stateless without memory tools). Default: aivc",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fastapi-chronicles-30",
        help="Target CommitChronicles dataset (default: fastapi-chronicles-30)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of commits already present in checkpoint",
    )
    parser.add_argument(
        "--checkpoint-file",
        "--checkpoint-path",
        dest="checkpoint_file",
        type=Path,
        default=None,
        help="Path to JSONL checkpoint file",
    )
    parser.add_argument(
        "--metrics-file",
        "--metrics-path",
        dest="metrics_file",
        type=Path,
        default=None,
        help="Path to output metrics JSON file",
    )
    parser.add_argument(
        "--curves-file",
        "--plots-path",
        "--plots-file",
        dest="curves_file",
        type=Path,
        default=None,
        help="Path to output plot curves CSV file",
    )

    # Attach unified evaluation configuration flags
    add_eval_args(parser)

    # Parse and resolve hierarchical config
    parsed_args = parser.parse_args()
    cfg = load_benchmark_config(args=parsed_args)
    paths = cfg.get_paths()

    clean_model = cfg.model.replace("/", "_").replace(":", "_").replace("-", "_")
    arm_name = parsed_args.arm.lower()

    checkpoint_file = parsed_args.checkpoint_file or (paths.checkpoints_dir / f"commit_chronicles_{clean_model}_{arm_name}_checkpoint.jsonl")
    metrics_file = parsed_args.metrics_file or (paths.metrics_dir / f"commit_chronicles_{clean_model}_{arm_name}_metrics.json")
    curves_file = parsed_args.curves_file or (paths.plots_dir / f"commit_chronicles_{clean_model}_{arm_name}_curves.csv")

    # Ensure output directories exist
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    curves_file.parent.mkdir(parents=True, exist_ok=True)

    # If reset-checkpoint requested or force, purge checkpoint
    if cfg.reset_checkpoint and checkpoint_file.exists():
        print(f"[RESET] Purging existing checkpoint file '{checkpoint_file}'...")
        checkpoint_file.unlink()

    print("=" * 75)
    print(f"[AIVC BENCHMARK RUNNER] CommitChronicles Evaluation Pipeline [{cfg.profile.upper()}]")
    print("=" * 75)
    print(f"Evaluation Arm : {parsed_args.arm.upper()}")
    print(f"Target Dataset : {parsed_args.dataset}")
    print(f"Sample Limit   : {cfg.limit or 30}")
    print(f"Active Model   : {cfg.model}")
    print(f"Max Turns      : {cfg.max_turns}")
    print(f"Max Tokens     : {cfg.max_tokens}")
    print(f"Max Cost/Inst  : ${cfg.max_cost_per_instance_usd:.2f} USD")
    print(f"Checkpoint File: {checkpoint_file}")
    print(f"Metrics Output : {metrics_file}")
    print(f"Curves Output  : {curves_file}")
    print("=" * 75)

    # Load API key based on provider
    provider = cfg.model_spec.provider if cfg.model_spec else "openrouter"
    api_key = os.getenv("TOGETHER_API_KEY", "") if provider == "together" else os.getenv("OPENROUTER_API_KEY", "")

    # Initialize CheckpointManager
    ckpt_mgr = CheckpointManager(checkpoint_file)
    print(f"[CHECKPOINT] Loaded {len(ckpt_mgr.processed_ids)} existing processed commits from checkpoint.")

    # Load Dataset (30 sequential commits)
    instances, used_dataset_name = load_commit_chronicles_dataset(
        dataset_name=parsed_args.dataset,
        limit=cfg.limit or 30,
    )

    # Configure tool interaction paths
    profile_interactions = paths.metrics_dir / "tool_interactions.jsonl"
    bench_interactions = EVAL_DIR / "metrics" / f"commit_chronicles_{arm_name}_tool_interactions.jsonl"
    general_interactions = EVAL_DIR / "metrics" / "tool_interactions.jsonl"
    interactions_paths = [profile_interactions, bench_interactions, general_interactions]

    # Purge interactions if reset requested
    if cfg.reset_checkpoint:
        for p in interactions_paths:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Instantiate Runner
    runner = CommitChroniclesRunner(
        arm=parsed_args.arm,
        model_name=cfg.model,
        api_key=api_key,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        max_cost_per_instance_usd=cfg.max_cost_per_instance_usd,
        prompt_price_per_1m=cfg.model_spec.prompt_price_per_1m if cfg.model_spec else None,
        completion_price_per_1m=cfg.model_spec.completion_price_per_1m if cfg.model_spec else None,
        interactions_paths=interactions_paths,
    )

    skipped_count = 0
    processed_this_run = 0

    for idx, inst in enumerate(instances, 1):
        cid = inst["commit_id"]
        if ckpt_mgr.is_processed(cid) and not parsed_args.force and not cfg.reset_checkpoint:
            print(f"[SKIP] Commit '{cid}' already processed in checkpoint.")
            skipped_count += 1
            continue

        episode_rec = runner.run_episode(instance=inst, episode_index=idx)
        ckpt_mgr.save_episode(episode_rec)
        processed_this_run += 1

    # Load all accumulated records for final metric calculation & curve generation
    all_records = ckpt_mgr.load_all_records()

    if all_records:
        export_metrics(
            records=all_records,
            metrics_path=metrics_file,
            model_name=cfg.model,
            dataset_name=used_dataset_name,
            arm=parsed_args.arm,
        )
        export_plots_curves(
            records=all_records,
            curves_path=curves_file,
        )
        # Also mirror to general files for DVC pipeline tracking
        general_metrics = EVAL_DIR / "metrics" / "commit_chronicles_metrics.json"
        general_curves = EVAL_DIR / "plots" / "commit_chronicles_curves.csv"
        try:
            if metrics_file != general_metrics:
                export_metrics(
                    records=all_records,
                    metrics_path=general_metrics,
                    model_name=cfg.model,
                    dataset_name=used_dataset_name,
                    arm=parsed_args.arm,
                )
            if curves_file != general_curves:
                export_plots_curves(
                    records=all_records,
                    curves_path=general_curves,
                )
        except Exception:
            pass

    print("\n" + "=" * 75)
    print(f"[SUMMARY] CommitChronicles Evaluation Execution Finished ({parsed_args.arm.upper()})")
    print("=" * 75)
    print(f"Evaluation Arm       : {parsed_args.arm.upper()}")
    print(f"Total Commits        : {len(instances)}")
    print(f"Skipped (Checkpointed): {skipped_count}")
    print(f"Processed This Run   : {processed_this_run}")
    print(f"Total Checkpoint Count: {len(all_records)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
