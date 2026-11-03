"""
Workspace: main entry point for the AIVC versioning engine.

Manages tracked files, creates commits, computes statuses, and handles
the untrack + GC lifecycle.
"""

from __future__ import annotations

import difflib
import fnmatch
import glob as _glob
import json
import logging
import os
import re
import sys
import time
import unicodedata
import uuid
import shutil
import tempfile

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import urllib.parse

from aivc.core.blob_store import BlobStore
from aivc.core.memory import Memory, FileChange, memory_from_dict, memory_to_dict, _compose_note
from aivc.core.diff import compute_diff
from aivc.core.index import CoreIndex
from aivc.config import (
    get_machine_id,
    to_portable_path,
    from_portable_path,
    get_home_dir,
    extract_brain_subpath,
    get_aivc_temp_dir,
    get_aivc_user_temp_dir,
    normalize_main_project_file,
)


def _extract_linked_file_paths(text: str) -> list[str]:
    """Extracts local file paths from text (file:/// URLs, Windows drive letters, ~/, AppData)."""
    if not text or not isinstance(text, str):
        return []

    paths: list[str] = []
    # 1. file:/// links (e.g. [name](file:///C:/...))
    for m in re.finditer(r"file:///(?:[A-Za-z]:[\\/]|/)?([^\s\)\"\'\`\<\>]+)", text):
        raw = m.group(0)
        raw = re.sub(r":L\d+(?:-L\d+)?$", "", raw)
        clean = raw[8:] if raw.startswith("file:///") else (raw[7:] if raw.startswith("file://") else raw)
        decoded = urllib.parse.unquote(clean)
        if re.match(r"^/[A-Za-z]:", decoded):
            decoded = decoded[1:]
        paths.append(decoded)

    # 2. Windows drive letters C:\... or C:/... (negative lookbehind on \w to avoid matching https:// or file:///)
    for m in re.finditer(r"(?<!\w)[A-Za-z]:[\\/][^\s\)\"\'\`\<\>]+", text):
        p = re.sub(r":L\d+(?:-L\d+)?$", "", m.group(0))
        paths.append(p)

    # 3. ~/ or .gemini or AppData paths
    for m in re.finditer(r"(?:~|\.gemini|AppData)[\\/][^\s\)\"\'\`\<\>]+", text):
        p = re.sub(r":L\d+(?:-L\d+)?$", "", m.group(0))
        paths.append(p)

    seen = set()
    unique_paths = []
    for p in paths:
        cleaned = p.rstrip(".,;:)\\]")
        if cleaned.startswith(("http://", "https://", "ftp://", "mailto:", "spark://")):
            continue
        norm = cleaned.replace("\\", "/").lower()
        if norm not in seen:
            seen.add(norm)
            unique_paths.append(cleaned)
    return unique_paths


def _is_same_project(mpf_a: str, mpf_b: str) -> bool:
    """Check if two main_project_file paths point to the same project."""
    if not mpf_a or not mpf_b:
        return False
    a = to_portable_path(mpf_a).lower().strip()
    b = to_portable_path(mpf_b).lower().strip()
    if a == b:
        return True
    if a.endswith("/" + b) or b.endswith("/" + a):
        return True
    name_a = PurePath(a).name
    name_b = PurePath(b).name
    if name_a and name_b and name_a == name_b and name_a.endswith(".md"):
        return True
    return False


def _resolve_file_path(path_str: str | Path) -> Path:
    """Resolve an input file path (direct path, portable path, or brain/aivc artifact) to an existing local Path.

    If the file exists under an artifact mirror (e.g. C:\\Users\\Public\\aivc\\artifacts\\...),
    and not yet under ~/.gemini/antigravity/brain/..., it automatically mirrors it to the brain
    location so that downstream sync and from_portable_path find it consistently.
    """
    clean_str = str(path_str).strip()
    if clean_str.startswith("file:///"):
        clean_str = clean_str[8:]
    elif clean_str.startswith("file://"):
        clean_str = clean_str[7:]
    clean_str = re.split(r":L?\d+", clean_str)[0]

    # 1. Direct path check on local filesystem
    direct_p = Path(clean_str)
    if direct_p.is_file() or direct_p.is_dir():
        sub = extract_brain_subpath(clean_str)
        if sub and direct_p.is_file():
            import aivc.config
            home = aivc.config.get_home_dir()
            brain_p = home / ".gemini" / "antigravity" / "brain" / Path(sub)
            if not brain_p.is_file() or (
                brain_p.stat().st_size != direct_p.stat().st_size
                or brain_p.stat().st_mtime < direct_p.stat().st_mtime
            ):
                try:
                    brain_p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(direct_p, brain_p)
                except OSError:
                    pass
        return direct_p

    # 2. Check from_portable_path resolution
    try:
        portable_cand = from_portable_path(path_str)
        if portable_cand.is_file() or portable_cand.is_dir():
            return portable_cand
    except Exception:
        portable_cand = None

    # 3. Artifact candidates lookup across standard temp and brain locations
    sub = extract_brain_subpath(clean_str)
    if not sub:
        try:
            sub = extract_brain_subpath(to_portable_path(path_str))
        except Exception:
            sub = None

    if sub:
        import aivc.config
        home = aivc.config.get_home_dir()
        temp_dir = aivc.config.get_aivc_temp_dir() / "artifacts"
        user_temp_dir = aivc.config.get_aivc_user_temp_dir() / "artifacts"
        brain_dir = home / ".gemini" / "antigravity" / "brain"
        brain_cli_dir = home / ".gemini" / "antigravity-cli" / "brain"

        candidates = [
            temp_dir / Path(sub),
            user_temp_dir / Path(sub),
            brain_dir / Path(sub),
            brain_cli_dir / Path(sub),
        ]
        for cand in candidates:
            if cand.is_file() or cand.is_dir():
                target_brain = brain_dir / Path(sub)
                if cand.is_file() and (
                    not target_brain.is_file()
                    or target_brain.stat().st_size != cand.stat().st_size
                    or target_brain.stat().st_mtime < cand.stat().st_mtime
                ):
                    try:
                        target_brain.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cand, target_brain)
                    except OSError:
                        pass
                return cand

    return portable_cand if portable_cand is not None else Path(clean_str)



@dataclass
class FileStatus:
    """Status of a single tracked file."""

    path: str
    """Relative path of the file."""

    current_size: int | None
    """Current size on disk in bytes. None if the file no longer exists."""

    history_size: int
    """Total bytes consumed by historical blobs associated with this file."""


class Workspace:
    """Orchestrates tracking, memories, GC, and status reporting.

    Disk layout under <storage_root>:
        workspace.json       — tracked files + head memory id (stored as head_commit_id)
        commits/
            <uuid>.json      — individual memory records
        blobs/               — managed by BlobStore
        refcounts.json       — managed by BlobStore
    """

    _WORKSPACE_FILE = "workspace.json"
    _COMMITS_DIR = "commits"

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root
        self._commits_dir = self._root / self._COMMITS_DIR
        self._workspace_path = self._root / self._WORKSPACE_FILE

        self._commits_dir.mkdir(parents=True, exist_ok=True)
        self._blob_store = BlobStore(self._root)
        self._index = CoreIndex(self._root)
        self._last_mtime = 0.0
        self._on_reload_callbacks = []
        self._last_modified_commit_ids: list[str] = []
        self._last_gc_drive_files: list[str] = []

        if self._workspace_path.exists():
            self._state = self._load_state()
            self._last_mtime = self._workspace_path.stat().st_mtime

            # Validate that head_commit_id actually exists on disk
            head_id = self._state.get("head_commit_id")
            if head_id is not None:
                if not self._memory_path(head_id).exists():
                    sys.stderr.write(
                        f"Warning: HEAD commit {head_id!r} not found in commits/. "
                        "Resetting head_commit_id to None.\n"
                    )
                    self._state["head_commit_id"] = None
                    self._save_state()
            
            # Migration: convert stored relative paths to absolute and richer format
            migrated = False
            new_tracked = {}
            for path_str, tracker_data in self._state.get("tracked_files", {}).items():
                p = Path(path_str)
                abs_path = str(p if p.is_absolute() else Path.cwd() / p)
                if abs_path != path_str:
                    migrated = True
                
                # Enrich format if it's just a string hash
                if isinstance(tracker_data, str) or tracker_data is None:
                    migrated = True
                    # Initially, we don't have mtime/size, so we just wrap the hash
                    new_tracked[abs_path] = {"hash": tracker_data, "mtime": None, "size": None}
                else:
                    new_tracked[abs_path] = tracker_data
            
            if migrated:
                self._state["tracked_files"] = new_tracked

            # Migration: delete watched_dirs if present (Chantier 1)
            if "watched_dirs" in self._state:
                del self._state["watched_dirs"]
                migrated = True
            
            if migrated:
                self._save_state()
        else:
            self._state: dict[str, Any] = {
                "tracked_files": {},
                "head_commit_id": None,
            }
            self._save_state()
            # _save_state updates _last_mtime

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        raw = json.loads(self._workspace_path.read_text(encoding="utf-8"))
        _required = {"tracked_files", "head_commit_id"}
        _missing = _required - raw.keys()
        if _missing:
            raise ValueError(
                f"workspace.json is corrupted — missing keys: {_missing}."
            )
        return raw

    def _save_state(self) -> None:
        self._workspace_path.write_text(
            json.dumps(self._state, indent=2), encoding="utf-8"
        )
        self._last_mtime = self._workspace_path.stat().st_mtime

    def register_reload_callback(self, callback: Any) -> None:
        """Register a callback to be executed when the state is reloaded from disk."""
        self._on_reload_callbacks.append(callback)

    def _reload_state_if_needed(self) -> bool:
        """Check if workspace.json has changed on disk and reload if so.
        
        Returns:
            True if state was reloaded, False otherwise.
        """
        if not self._workspace_path.exists():
            return False
        
        current_mtime = self._workspace_path.stat().st_mtime
        if current_mtime > self._last_mtime:
            self._state = self._load_state()
            self._last_mtime = current_mtime
            for cb in self._on_reload_callbacks:
                cb()
            return True
        return False

    def _memory_path(self, memory_id: str) -> Path:
        return self._commits_dir / f"{memory_id}.json"

    def _save_memory(self, memory: Memory) -> None:
        path = self._memory_path(memory.id)
        path.write_text(json.dumps(memory_to_dict(memory), indent=2), encoding="utf-8")

    def _load_memory(self, memory_id: str) -> Memory:
        path = self._memory_path(memory_id)
        if not path.exists():
            raise KeyError(f"Memory {memory_id!r} not found.")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Memory {memory_id!r} is empty.")
        return memory_from_dict(json.loads(text))

    def _is_hidden(self, path: Path) -> bool:
        """Check if any component of the path starts with a dot."""
        return any(part.startswith(".") for part in path.parts)

    def _expand_path(self, path_or_glob: str) -> tuple[list[str], int]:
        """Expand a path, directory, or glob pattern to a list of relative file paths.
        
        Ignores hidden files and directories (starting with '.').

        Returns:
            A tuple (list of paths, count of hidden files skipped).

        Raises:
            ValueError: if no files match.
        """
        p = Path(path_or_glob)
        hidden_count = 0
        all_files = []

        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    # Exclude the AIVC storage directory itself
                    if str(f).startswith(str(self._root)):
                        continue
                    # Check if hidden relative to the watched root or any parent segment
                    if self._is_hidden(f):
                        hidden_count += 1
                    else:
                        all_files.append(str(f))
        else:
            # For glob patterns, we rely on glob module but filter results
            # The `Path(m)` conversion is important to handle paths correctly across OS
            # and to use `_is_hidden` which expects a Path object.
            matches = _glob.glob(path_or_glob, recursive=True)
            for m in matches:
                mp = Path(m)
                if mp.is_file():
                    if self._is_hidden(mp):
                        hidden_count += 1
                    else:
                        all_files.append(str(mp))

        if not all_files and hidden_count == 0 and not p.is_dir():
            raise ValueError(
                f"No files found matching {path_or_glob!r}. "
                "Did you mean a file, directory, or glob pattern?"
            )
        return all_files, hidden_count
    
    def migrate_index(self) -> None:
        """Ensure all existing JSON memories are in the SQLite index.
        
        This is an explicit migration call, usually triggered via CLI.
        """
        self._index.migrate_from_json(self._commits_dir)

    def _resolve_tracked_path(self, path_str: str, allow_basename_fallback: bool = False) -> str:
        """Resolve a relative or basename path to a known tracked file path."""
        try:
            if _resolve_file_path(path_str).exists() or to_portable_path(path_str) in self._state["tracked_files"]:
                return path_str
        except Exception:
            if to_portable_path(path_str) in self._state["tracked_files"]:
                return path_str

        raw_s = path_str.strip().replace("\\", "/")
        is_rel = not (raw_s.startswith(("~", "/")) or (len(raw_s) >= 2 and raw_s[1] == ":"))
        rel_norm = raw_s.lstrip("./")

        if is_rel and rel_norm:
            vn_cand = Path.home() / "Documents" / "VoiceNotes" / rel_norm
            if vn_cand.exists():
                return str(vn_cand)

            for tp in self._state["tracked_files"]:
                if tp == rel_norm or tp.endswith("/" + rel_norm):
                    return tp

        if allow_basename_fallback:
            basename = PurePath(raw_s).name
            if basename:
                for tp in self._state["tracked_files"]:
                    if PurePath(tp.replace("\\", "/")).name == basename:
                        return tp

        return path_str

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _purge_file_history(self, file_path: str) -> None:
        """Remove a specific file's history, decrementing blob references and cleaning index/state."""
        self._reload_state_if_needed()
        abs_p = str(Path(file_path).resolve())

        # Collect memories referencing this file via the index (fast).
        affected_memory_ids = self._index.get_memories_touching_file(abs_p)
        with self._blob_store.batch():
            for mid in affected_memory_ids:
                memory = self._load_memory(mid)
                updated_changes = []
                for change in memory.changes:
                    if change.path == abs_p:
                        if change.blob_hash is not None:
                            self._blob_store.decrement_ref(change.blob_hash)
                    else:
                        updated_changes.append(change)
                
                if len(updated_changes) != len(memory.changes):
                    memory.changes = updated_changes
                    self._save_memory(memory)

        # Cleanup the index for this file.
        self._index.remove_file_changes(abs_p)
        self._state["tracked_files"].pop(abs_p, None)
        self._save_state()

    def get_tracked_paths(self) -> list[str]:
        """Return just the list of tracked file absolute paths (fast)."""
        self._reload_state_if_needed()
        return list(self._state["tracked_files"].keys())

    def remember(
        self,
        title: str,
        journal: str = "",
        tasks: list[dict[str, Any]] | None = None,
        user_items: list[dict[str, Any]] | None = None,
        edited_files: list[str] | None = None,
        read_files: list[str] | None = None,
        urls: list[str] | None = None,
        note: str | None = None,
        machine_id: str = "",
        consulted_files: list[str] | None = None,
        roadmap_items: list[dict[str, Any]] | None = None,
        main_project_file: str = "",
        completed_user_items: list[str] | None = None,
    ) -> Memory:
        """Create a new memory in 3-part format (title, journal, tasks/user_items) without local diffs/blobs.

        Args:
            title: Short title summarising the achievement.
            journal: Agent thoughts and activity log.
            tasks: Unified tasks list [{id, task, task_status, details, done}].
            user_items: Legacy alias for tasks [{id, task, details, done, kind}].
            edited_files: List of file paths modified or created.
            read_files: List of file paths consulted.
            urls: Optional list of URLs.
            note: Backward-compatibility parameter mapping to journal.
            machine_id: ID of the machine where the memory was created.
            consulted_files: Legacy parameter mapping to read_files.
            roadmap_items: Optional roadmap tasks recorded with task_status='new' / kind='roadmap'.
            main_project_file: Portable path to the main project note or file.
        """
        self._reload_state_if_needed()

        if main_project_file:
            main_project_file = normalize_main_project_file(main_project_file)

        if not journal and note:
            journal = note

        actual_urls = [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]
        actual_read_files = list(read_files) if read_files is not None else []
        if consulted_files:
            actual_read_files.extend(consulted_files)
        actual_edited_files = list(edited_files) if edited_files is not None else []

        actual_read_files = list(dict.fromkeys([self._resolve_tracked_path(f, allow_basename_fallback=True) for f in actual_read_files]))
        actual_edited_files = list(dict.fromkeys([self._resolve_tracked_path(f, allow_basename_fallback=False) for f in actual_edited_files]))

        memory_id = str(uuid.uuid4())
        now_ts = datetime.now(timezone.utc).isoformat()
        self._last_modified_commit_ids: list[str] = []

        # Merge input task collections
        raw_task_entries: list[tuple[dict[str, Any], bool]] = []
        if tasks:
            for it in tasks:
                if isinstance(it, dict):
                    raw_task_entries.append((it, False))
        if user_items:
            for it in user_items:
                if isinstance(it, dict):
                    raw_task_entries.append((it, False))
        if roadmap_items:
            for it in roadmap_items:
                if isinstance(it, dict):
                    raw_task_entries.append((it, True))
                elif isinstance(it, str):
                    raw_task_entries.append(({"task": it}, True))

        # Build existing task map across commits for lookup and event-sourcing, sorted chronologically
        existing_tasks: dict[str, dict[str, Any]] = {}
        commits_data: dict[Path, dict[str, Any]] = {}
        raw_commits: list[tuple[str, str, Path, dict[str, Any]]] = []
        for p in self._commits_dir.glob("*.json"):
            try:
                c_data = json.loads(p.read_text(encoding="utf-8"))
                c_ts = str(c_data.get("timestamp") or "")
                raw_commits.append((c_ts, p.name, p, c_data))
            except Exception:
                continue

        raw_commits.sort(key=lambda x: (x[0], x[1]))

        first_seen: dict[str, tuple[str, int]] = {}
        project_active_tasks: dict[str, dict[str, Any]] = {}

        task_is_recurring: dict[str, bool] = {}
        task_base_details: dict[str, str] = {}

        for c_ts, p_name, p, c_data in raw_commits:
            commits_data[p] = c_data
            c_mpf = c_data.get("main_project_file") or ""
            is_same_proj = bool(main_project_file and c_mpf and _is_same_project(c_mpf, main_project_file))

            for c_idx, c_it in enumerate(c_data.get("user_items", []), 1):
                if isinstance(c_it, dict):
                    c_id = c_it.get("id")
                    if c_id:
                        if c_id not in first_seen:
                            first_seen[c_id] = (c_it.get("created_at") or c_ts, c_idx)
                        existing_tasks[c_id] = c_it

                        if c_it.get("is_recurring"):
                            task_is_recurring[c_id] = True
                        if c_it.get("base_details"):
                            task_base_details[c_id] = c_it.get("base_details")
                        elif c_it.get("task_status") == "new" and c_it.get("details"):
                            dt = str(c_it.get("details")).strip()
                            if not (dt.startswith("[VALIDÉ") or dt.startswith("[ÉCARTÉ")):
                                task_base_details[c_id] = dt

                        if is_same_proj:
                            c_st = c_it.get("task_status")
                            c_dn = bool(c_it.get("done", False))
                            if c_st == "validated" or c_dn:
                                project_active_tasks.pop(c_id, None)
                            elif c_st in ("new", "await_validation"):
                                project_active_tasks[c_id] = c_it

            if is_same_proj:
                for cid in c_data.get("completed_user_items", []) or []:
                    project_active_tasks.pop(str(cid).strip(), None)

        norm_user_items: list[dict[str, Any]] = []
        status_updates_by_id: dict[str, tuple[str, bool]] = {}

        if completed_user_items:
            for cid in completed_user_items:
                c_id_clean = str(cid).strip()
                if c_id_clean:
                    status_updates_by_id[c_id_clean] = ("validated", True)
                    project_active_tasks.pop(c_id_clean, None)

        self._last_gc_drive_files = []

        for idx, (item, is_roadmap_arg) in enumerate(raw_task_entries, 1):
            raw_id = str(item.get("id") or "").strip()
            task_title = str(item.get("task", "")).strip()

            # Deterministic lookup: a task is identified EXCLUSIVELY by its exact task_id
            matched_existing_id = None
            if raw_id and raw_id in existing_tasks:
                matched_existing_id = raw_id
                prior_task = existing_tasks[raw_id]
                final_id = raw_id
            else:
                prior_task = None
                final_id = raw_id

            # Resolve task_status
            explicit_status = item.get("task_status")
            if explicit_status in ("new", "await_validation", "validated"):
                task_status = explicit_status
            elif item.get("done") is True:
                task_status = "validated"
            elif (
                is_roadmap_arg
                or item.get("kind") == "roadmap"
                or item.get("type") == "roadmap"
                or (final_id and final_id.startswith("r-"))
            ):
                task_status = "new"
            elif prior_task and prior_task.get("task_status"):
                task_status = prior_task["task_status"]
            else:
                task_status = "await_validation"

            done = (task_status == "validated")
            kind = "roadmap" if task_status == "new" else "deliverable"

            # Assign generated id if still None
            if not final_id:
                prefix = f"r-{memory_id[:6]}" if task_status == "new" else f"{memory_id[:6]}"
                final_id = f"{prefix}-{idx}"

            # Resolve task title (fallback to prior commit if empty)
            if not task_title and prior_task:
                task_title = str(prior_task.get("task", "")).strip()

            # Resolve details
            details = str(item.get("details", "")).strip()
            if not details and prior_task and prior_task.get("details"):
                details = str(prior_task.get("details", "")).strip()

            # Resolve created_at (preserve original created_at on update)
            created_at = item.get("created_at")
            if not created_at and final_id and final_id in first_seen:
                created_at = first_seen[final_id][0]
            if not created_at and prior_task and prior_task.get("created_at"):
                created_at = prior_task["created_at"]
            if not created_at:
                created_at = now_ts

            # Resolve is_recurring
            is_rec = bool(item.get("is_recurring", False))
            if not is_rec and final_id and task_is_recurring.get(final_id, False):
                is_rec = True

            # Resolve base_details
            base_det = item.get("base_details") or task_base_details.get(final_id, "")
            if not base_det and task_status == "new" and details and not (details.startswith("[VALIDÉ") or details.startswith("[ÉCARTÉ")):
                base_det = details

            # Attach edited_files & read_files to task item (strict replacement: no historical accumulation)
            item_edited = list(item.get("edited_files", []))
            if not item_edited and len(raw_task_entries) == 1 and actual_edited_files:
                item_edited = list(actual_edited_files)
            merged_edited = list(dict.fromkeys(item_edited))

            item_read = list(item.get("read_files", []))
            if not item_read and len(raw_task_entries) == 1 and actual_read_files:
                item_read = [f for f in actual_read_files if to_portable_path(f).lower() != to_portable_path(main_project_file).lower()]
            merged_read = list(dict.fromkeys(item_read))

            norm_item = {
                "id": final_id,
                "task": task_title,
                "details": details,
                "done": done,
                "task_status": task_status,
                "kind": kind,
                "created_at": created_at,
                "updated_at": now_ts,
                "edited_files": merged_edited,
                "read_files": merged_read,
                "is_recurring": is_rec,
                "base_details": base_det,
            }
            norm_user_items.append(norm_item)

            if matched_existing_id:
                status_updates_by_id[final_id] = (task_status, done)
            elif raw_id and prior_task:
                status_updates_by_id[raw_id] = (task_status, done)

        # Sort user_items chronologically: existing tasks by (first_seen_ts, first_seen_idx), new tasks after in submission order
        def _task_sort_key(entry: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
            submission_idx, t_item = entry
            t_id = t_item.get("id", "")
            if t_id in first_seen:
                fs_ts, fs_idx = first_seen[t_id]
                return (0, fs_ts, fs_idx)
            return (1, t_item.get("created_at") or now_ts, submission_idx)

        indexed_items = list(enumerate(norm_user_items, 1))
        indexed_items.sort(key=_task_sort_key)
        norm_user_items = [it for _, it in indexed_items]

        # Event-sourcing update on previous commits if status changed
        if status_updates_by_id:
            for p, c_data in commits_data.items():
                c_items = c_data.get("user_items", [])
                c_modified = False
                for c_it in c_items:
                    c_id = c_it.get("id")
                    if c_id in status_updates_by_id:
                        new_st, new_dn = status_updates_by_id[c_id]
                        if c_it.get("task_status") != new_st or c_it.get("done") != new_dn:
                            c_it["task_status"] = new_st
                            c_it["done"] = new_dn
                            c_it["updated_at"] = now_ts
                            c_modified = True
                if c_modified:
                    c_data["user_items"] = c_items
                    c_data["note"] = _compose_note(
                        c_data.get("journal", ""),
                        c_items,
                        fallback_note=c_data.get("note", ""),
                    )
                    try:
                        p.write_text(json.dumps(c_data, indent=2), encoding="utf-8")
                        self._last_modified_commit_ids.append(p.stem)
                    except Exception:
                        pass

        # Collect updated task ids from this update for Drive Garbage Collection
        updated_ids = set(status_updates_by_id.keys())
        for it in norm_user_items:
            if it.get("id"):
                updated_ids.add(it.get("id"))

        if updated_ids:
            # 1. Map all currently active paths across the workspace
            active_paths: set[str] = set()

            for p_c, c_data in commits_data.items():
                for c_it in c_data.get("user_items", []):
                    c_id = c_it.get("id")
                    if c_id in updated_ids:
                        continue
                    st = status_updates_by_id.get(c_id, (c_it.get("task_status"), False))[0]
                    if st in ("new", "await_validation"):
                        for lp in _extract_linked_file_paths(c_it.get("details", "")):
                            active_paths.add(to_portable_path(lp).lower())
                        for ep in c_it.get("edited_files", []):
                            active_paths.add(to_portable_path(ep).lower())
                        for rp in c_it.get("read_files", []):
                            active_paths.add(to_portable_path(rp).lower())

            for it in norm_user_items:
                if it.get("task_status") in ("new", "await_validation"):
                    for lp in _extract_linked_file_paths(it.get("details", "")):
                        active_paths.add(to_portable_path(lp).lower())
                    for ep in it.get("edited_files", []):
                        active_paths.add(to_portable_path(ep).lower())
                    for rp in it.get("read_files", []):
                        active_paths.add(to_portable_path(rp).lower())

            # 2. Gather candidate paths from updated tasks (especially prior commits)
            candidate_paths: list[str] = []

            for p_c, c_data in commits_data.items():
                commit_task_ids = {c_it.get("id") for c_it in c_data.get("user_items", []) if c_it.get("id")}
                if commit_task_ids & updated_ids:
                    for c_it in c_data.get("user_items", []):
                        if c_it.get("id") in updated_ids:
                            candidate_paths.extend(_extract_linked_file_paths(c_it.get("details", "")))
                            candidate_paths.extend(c_it.get("edited_files", []))
                            candidate_paths.extend(c_it.get("read_files", []))

                    # If this commit has no other tasks still in 'new' or 'await_validation'
                    commit_has_active_tasks = False
                    for c_it in c_data.get("user_items", []):
                        c_id = c_it.get("id")
                        if c_id in updated_ids:
                            matching_norm = next((it for it in norm_user_items if it.get("id") == c_id), None)
                            if matching_norm and matching_norm.get("task_status") in ("new", "await_validation"):
                                commit_has_active_tasks = True
                                break
                        else:
                            st = status_updates_by_id.get(c_id, (c_it.get("task_status"), False))[0]
                            if st in ("new", "await_validation"):
                                commit_has_active_tasks = True
                                break

                    if not commit_has_active_tasks:
                        for ch in c_data.get("changes", []):
                            if isinstance(ch, dict) and ch.get("action") in ("edited", "added", "modified"):
                                if ch.get("path"):
                                    candidate_paths.append(ch.get("path"))

            for it in norm_user_items:
                if it.get("task_status") == "validated":
                    candidate_paths.extend(_extract_linked_file_paths(it.get("details", "")))
                    candidate_paths.extend(it.get("edited_files", []))
                    candidate_paths.extend(it.get("read_files", []))

            # 3. Filter out VoiceNotes (sanctuarisé), web URLs, and active paths
            seen_gc = set()
            for cand in candidate_paths:
                cand_str = str(cand).strip()
                if cand_str.startswith(("http://", "https://", "ftp://", "mailto:", "spark://")):
                    continue
                p_norm = to_portable_path(cand).lower()
                if "voicenotes" in p_norm:
                    continue
                if p_norm in active_paths:
                    continue
                if p_norm not in seen_gc:
                    seen_gc.add(p_norm)
                    self._last_gc_drive_files.append(cand)

        # Strict validation
        for files_list, list_name in [(actual_read_files, "read_files"), (actual_edited_files, "edited_files")]:
            for path_str in files_list:
                p = _resolve_file_path(path_str)
                portable_path = to_portable_path(path_str)

                # Check for directories
                if p.is_dir():
                    raise ValueError(f"Validation error in {list_name}: '{path_str}' is a directory.")

                # Check for untracked non-existent files
                if not p.is_file() and portable_path not in self._state["tracked_files"]:
                    raise ValueError(f"Validation error in {list_name}: '{path_str}' is an untracked non-existent file.")

        changes = []
        seen_paths = set()

        # edited_files -> FileChange(path=portable_path, action="edited" or "deleted")
        for path_str in actual_edited_files:
            p = _resolve_file_path(path_str)
            portable_path = to_portable_path(path_str)
            action = "deleted" if (not p.is_file() and portable_path in self._state["tracked_files"]) else "edited"
            if portable_path not in seen_paths:
                seen_paths.add(portable_path)
                blob_hash = None
                if p.is_file():
                    try:
                        if p.stat().st_size <= 10 * 1024 * 1024:
                            data = p.read_bytes()
                            blob_hash = self._blob_store.store(data)
                    except OSError:
                        blob_hash = None
                changes.append(
                    FileChange(
                        path=portable_path,
                        action=action,
                        blob_hash=blob_hash,
                        bytes_added=0,
                        bytes_removed=0,
                    )
                )

        # read_files -> FileChange(path=portable_path, action="consulted")
        for path_str in actual_read_files:
            portable_path = to_portable_path(path_str)
            if portable_path not in seen_paths:
                seen_paths.add(portable_path)
                changes.append(
                    FileChange(
                        path=portable_path,
                        action="consulted",
                        blob_hash=None,
                        bytes_added=0,
                        bytes_removed=0,
                    )
                )

        if not changes and not actual_urls and not norm_user_items and not (journal and journal.strip()):
            raise RuntimeError(
                "No changes detected in tracked files, no files consulted, and no URLs provided. "
                "Nothing to remember."
            )

        full_note = _compose_note(journal, norm_user_items, fallback_note=journal or note or "")
        if not full_note.strip():
            full_note = title

        memory = Memory(
            id=memory_id,
            timestamp=now_ts,
            title=title,
            note=full_note,
            parent_id=self._state.get("head_commit_id"),
            changes=changes,
            machine_id=machine_id,
            urls=actual_urls,
            user_items=norm_user_items,
            completed_user_items=list(completed_user_items) if completed_user_items else [],
            journal=journal or (note if not norm_user_items else ""),
            main_project_file=main_project_file,
        )

        self._save_memory(memory)
        self._index.add_memory(memory)

        for c in changes:
            if c.action == "deleted":
                self._state["tracked_files"].pop(c.path, None)
            else:
                self._state["tracked_files"][c.path] = {"hash": c.blob_hash, "mtime": None, "size": None}

        self._state["head_commit_id"] = memory.id
        self._save_state()

        # Step: Auto-sync advance_tldr notes for Obsidian projects
        try:
            from aivc.advance_sync import sync_advance_tldr
            sync_advance_tldr(memory)
        except Exception as e:
            logger.warning("Failed to auto-sync advance_tldr in workspace.remember: %s", e)

        return memory

    def create_memory(
        self,
        title: str,
        note: str = "",
        read_files: list[str] | None = None,
        edited_files: list[str] | None = None,
        machine_id: str = "",
        consulted_files: list[str] | None = None,
        urls: list[str] | None = None,
        user_items: list[dict[str, Any]] | None = None,
        journal: str = "",
        roadmap_items: list[dict[str, Any]] | None = None,
    ) -> Memory:
        """Alias for remember for backwards compatibility."""
        return self.remember(
            title=title,
            journal=journal or note,
            user_items=user_items,
            edited_files=edited_files,
            read_files=read_files,
            urls=urls,
            note=note,
            machine_id=machine_id,
            consulted_files=consulted_files,
            roadmap_items=roadmap_items,
        )

    def task_commit(
        self,
        title: str,
        content: str,
        main_project_file: str,
        task_id: str | None = None,
        task_status: str = "await_validation",
        edited_files: list[str] | None = None,
        read_files: list[str] | None = None,
        urls: list[str] | None = None,
        machine_id: str = "",
        is_recurring: bool = False,
    ) -> Memory:
        """Atomic task commit (1 call = 1 task memory) with flat parameters."""
        task_item = {
            "id": task_id.strip() if task_id else None,
            "task": title,
            "details": content,
            "task_status": task_status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "edited_files": list(edited_files) if edited_files else [],
            "read_files": list(read_files) if read_files else [],
            "is_recurring": bool(is_recurring),
        }
        return self.remember(
            title=title,
            journal=content,
            main_project_file=main_project_file,
            user_items=[task_item],
            edited_files=edited_files,
            read_files=read_files,
            urls=urls,
            machine_id=machine_id,
        )

    def complete_user_items(self, item_ids: list[str]) -> list[str]:
        """Search across all memories for user_items matching item_ids, mark done=True, re-save.

        Returns:
            List of item IDs effectively marked done.
        """
        self._reload_state_if_needed()
        target_ids = set(item_ids)
        marked_done = []

        for p in self._commits_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            user_items = data.get("user_items", [])
            modified = False
            for item in user_items:
                if item.get("id") in target_ids and not item.get("done", False):
                    item["done"] = True
                    item["task_status"] = "validated"
                    item["updated_at"] = datetime.now(timezone.utc).isoformat()
                    marked_done.append(item["id"])
                    modified = True

            if modified:
                data["user_items"] = user_items
                data["note"] = _compose_note(
                    data.get("journal", ""),
                    user_items,
                    fallback_note=data.get("note", ""),
                )
                p.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return marked_done

    def compact_commits_json(self) -> int:
        """Removes the redundant 'note' key from on-disk commit JSONs that have journal or user_items.

        Returns:
            The number of commit JSON files compacted.
        """
        compacted = 0
        if not self._commits_dir.is_dir():
            return 0

        for p in self._commits_dir.glob("*.json"):
            if p.stat().st_size == 0:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            has_journal = bool(data.get("journal") and str(data.get("journal")).strip())
            has_user_items = bool(data.get("user_items"))
            if (has_journal or has_user_items) and "note" in data:
                data.pop("note", None)
                try:
                    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    compacted += 1
                except Exception:
                    pass

        return compacted

    def get_file_history(self, file_path: str) -> list[Memory]:
        """Return chronological list of memories touching file_path (portable match)."""
        self._reload_state_if_needed()
        raw_s = file_path.strip().replace("\\", "/")
        is_rel = not (raw_s.startswith(("~", "/")) or (len(raw_s) >= 2 and raw_s[1] == ":"))
        rel_norm = raw_s.lstrip("./")
        target_portable = to_portable_path(file_path)
        basename = PurePath(raw_s).name

        memory_ids = self._index.get_memories_touching_file(file_path)
        loaded_memories: list[Memory] = []
        for mid in memory_ids:
            try:
                loaded_memories.append(self._load_memory(mid))
            except KeyError:
                continue

        matching_memories: list[Memory] = []
        # Passe 1 : match exact ou suffixe relatif
        for mem in loaded_memories:
            for ch in mem.changes:
                p = to_portable_path(ch.path)
                if p == target_portable or (is_rel and rel_norm and (p == rel_norm or p.endswith("/" + rel_norm))):
                    matching_memories.append(mem)
                    break

        # Passe 2 : repli par basename si matching_memories est vide et basename non vide
        if not matching_memories and basename:
            for mem in loaded_memories:
                for ch in mem.changes:
                    if PurePath(to_portable_path(ch.path).replace("\\", "/")).name == basename:
                        matching_memories.append(mem)
                        break

        matching_memories.sort(key=lambda m: m.timestamp)
        return matching_memories

    def get_task_history(self, task_id: str) -> list[dict[str, Any]]:
        """Return chronological list of memories touching task_id."""
        self._reload_state_if_needed()
        clean_tid = str(task_id).strip()
        if not clean_tid:
            return []

        commits: list[dict[str, Any]] = []
        for p in self._commits_dir.glob("*.json"):
            c_data = json.loads(p.read_text(encoding="utf-8"))
            commits.append(c_data)

        commits.sort(key=lambda c: str(c.get("timestamp") or ""))

        history: list[dict[str, Any]] = []
        for c in commits:
            mid = c.get("id") or ""
            ts = c.get("timestamp") or ""
            m_title = c.get("title") or ""
            journal = c.get("journal") or c.get("note") or ""

            # Check user_items
            matched_item = None
            for it in c.get("user_items", []):
                if isinstance(it, dict) and str(it.get("id", "")).strip() == clean_tid:
                    matched_item = it
                    break

            completed_ids = [str(cid).strip() for cid in (c.get("completed_user_items") or [])]
            is_completed = clean_tid in completed_ids

            if matched_item or is_completed:
                t_title = matched_item.get("task", "") if matched_item else ""
                t_status = matched_item.get("task_status", "validated" if is_completed else "new") if matched_item else "validated"
                is_rec = bool(matched_item.get("is_recurring", False)) if matched_item else False
                details = matched_item.get("details", "") if matched_item else ""
                edited = list(matched_item.get("edited_files", [])) if matched_item else []
                read = list(matched_item.get("read_files", [])) if matched_item else []

                history.append({
                    "memory_id": mid,
                    "timestamp": ts,
                    "memory_title": m_title,
                    "task_title": t_title,
                    "task_status": t_status,
                    "is_recurring": is_rec,
                    "details": details,
                    "journal": journal,
                    "edited_files": edited,
                    "read_files": read,
                })

        return history

    def get_task_history_formatted(self, task_id: str) -> str:
        """Return human/LLM-readable formatted markdown string of task history."""
        hist = self.get_task_history(task_id)
        if not hist:
            return f"Aucun souvenir trouvé pour la tâche [{task_id}]."

        clean_tid = str(task_id).strip()
        lines = [f"# Historique de la tâche [{clean_tid}] ({len(hist)} souvenir(s))\n"]
        for i, item in enumerate(hist, 1):
            ts = item.get("timestamp", "")[:19]
            st = item.get("task_status", "")
            rec_tag = " (Récurrente / Evergreen)" if item.get("is_recurring") else ""
            lines.append(f"## {i}. [{ts}] {item.get('memory_title', '')} — Statut: `{st}`{rec_tag}")
            lines.append(f"- **Memory ID** : `{item.get('memory_id', '')}`")
            if item.get("task_title"):
                lines.append(f"- **Tâche** : {item.get('task_title')}")
            if item.get("details"):
                lines.append(f"- **Détails / Décisions** :\n{item.get('details')}")
            if item.get("journal"):
                lines.append(f"- **Journal de session** :\n{item.get('journal')}")
            if item.get("edited_files"):
                lines.append(f"- **Fichiers modifiés** : {', '.join(item.get('edited_files', []))}")
            if item.get("read_files"):
                lines.append(f"- **Fichiers consultés** : {', '.join(item.get('read_files', []))}")
            lines.append("")

        return "\n".join(lines).strip()

    def get_tracked_files_metadata(self) -> dict[str, dict]:
        """Return the current metadata (mtime, size) for all tracked files from state.
        
        This avoids expensive live disk stats on every call.
        """
        self._reload_state_if_needed()
        return self._state.get("tracked_files", {})

    def get_status(self) -> list[FileStatus]:
        """Return the status of all tracked files.

        For each file: current on-disk size and total history size in AIVC.
        Uses the index to efficiently recover historical blob hashes.
        """
        self._reload_state_if_needed()
        statuses = []
        
        # O(1) bulk fetch of all historical blob hashes by file path (SQLite)
        all_hashes = self._index.get_all_blob_hashes_by_file()
        blob_size_cache = {}
        
        for tracked_path in self._state["tracked_files"]:
            p = _resolve_file_path(tracked_path)
            
            # Current size: check disk for correctness (deleted file detection)
            current_size = None
            if p.exists() and p.is_file():
                try:
                    current_size = p.stat().st_size
                except (PermissionError, OSError):
                    current_size = None

            # History size: use bulk index map
            history_size = 0
            hashes = all_hashes.get(tracked_path, set())
            for h in hashes:
                if h not in blob_size_cache:
                    try:
                        blob_size_cache[h] = self._blob_store.get_size(h)
                    except KeyError:
                        # Blob is missing locally (likely due to metadata-only sync)
                        blob_size_cache[h] = 0
                history_size += blob_size_cache[h]

            statuses.append(
                FileStatus(
                    path=tracked_path,
                    current_size=current_size,
                    history_size=history_size,
                )
            )
        return statuses

    def get_memory(self, memory_id: str) -> Memory:
        """Load and return a memory by ID.

        Raises:
            KeyError: if the memory does not exist.
        """
        return self._load_memory(memory_id)

    def get_log(
        self,
        limit: int = 20,
        offset: int = 0,
        include_remote: bool = True,
        only_local: bool = False,
    ) -> list[Memory]:
        """Return up to `limit` memories in reverse chronological order.

        Uses CoreIndex for fast ordering and machine_id filtering across local
        and remote memories.
        """
        self._reload_state_if_needed()
        self._index.migrate_from_json(self._commits_dir)

        local_m_id = get_machine_id() if (only_local or not include_remote) else None
        commit_ids = self._index.get_recent_commits(
            limit=limit, offset=offset, only_local_machine_id=local_m_id
        )

        memories: list[Memory] = []
        for cid in commit_ids:
            try:
                memories.append(self._load_memory(cid))
            except KeyError:
                sys.stderr.write(
                    f"Warning: Commit {cid!r} present in index but not found on disk.\n"
                )
        return memories

    def find_child_memory(self, memory_id: str) -> Memory | None:
        """Find the memory that has *memory_id* as its parent.

        Uses the index for O(1) lookup. Since AIVC maintains a linear chain, 
        there is at most one child.
        """
        child_info = self._index.find_child(memory_id)
        if child_info:
            child_id, _ = child_info
            return self._load_memory(child_id)
        return None

    def read_file_at_memory(self, file_path: str, memory_id: str) -> bytes:
        """Read the content of a tracked file as it was at a specific memory.

        Scans memory history to find the most recent blob for `file_path`
        at or before `memory_id`.

        Raises:
            KeyError: if the memory or file is not found in history.
        """
        target_portable = to_portable_path(file_path)
        current_id: str | None = memory_id
        while current_id is not None:
            memory = self._load_memory(current_id)
            for change in memory.changes:
                if to_portable_path(change.path) == target_portable:
                    if change.action == "deleted":
                        raise KeyError(f"File {file_path!r} was deleted at memory {current_id!r}.")
                    if change.blob_hash is not None:
                        return self._blob_store.retrieve(change.blob_hash)
                    # If blob is None, fallback to local file on disk
                    local_p = _resolve_file_path(target_portable)
                    if local_p.exists():
                        return local_p.read_bytes()
            current_id = memory.parent_id

        raise KeyError(
            f"File {file_path!r} was not found in the history up to memory {memory_id!r}."
        )

    def auto_restore_brain_files(self, memory: Memory, sync_manager: Any = None, background_drive: bool = False) -> list[str]:
        """Automatically restore files under .gemini/antigravity/brain/ linked to this memory.

        If a file does not exist locally and its blob is available in BlobStore
        (or recoverable via other commits or sync_manager), it is restored to disk.
        Returns the list of restored local file paths.
        """
        import re
        import threading
        from aivc.config import extract_brain_subpath, get_home_dir

        restored: list[str] = []
        home = get_home_dir()

        # Collect candidate items: (subpath_under_brain, blob_hash_or_None)
        candidates: dict[str, str | None] = {}

        # 1. From changes
        for change in getattr(memory, "changes", []):
            path_str = getattr(change, "path", "")
            if not path_str:
                continue
            sub = extract_brain_subpath(path_str)
            if sub and sub not in candidates:
                candidates[sub] = getattr(change, "blob_hash", None)

        # 2. From edited_files and read_files
        for path_str in getattr(memory, "edited_files", []) + getattr(memory, "read_files", []):
            sub = extract_brain_subpath(path_str)
            if sub and sub not in candidates:
                candidates[sub] = None

        # 3. From user_items (task, details, and attached files)
        brain_regex = re.compile(
            r"(?:file:///?(?:[A-Za-z]:)?[\\/]|~/|[A-Za-z]:[\\/]|/)?(?:Users|home)?[^\"\'\`\s\)\>]*?(?:\.?gemini[\\/]antigravity(?:-cli)?[\\/]brain|aivc[\\/]artifacts)[\\/]([^\s\"\'\`\)\>:]+)",
            re.IGNORECASE,
        )
        for item in getattr(memory, "user_items", []) or []:
            if isinstance(item, dict):
                text = f"{item.get('task', '')} {item.get('details', '')}"
                for m in brain_regex.finditer(text):
                    raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw and raw not in candidates:
                        candidates[raw] = None
                for ep in item.get("edited_files", []) + item.get("read_files", []):
                    sub = extract_brain_subpath(ep)
                    if sub and sub not in candidates:
                        candidates[sub] = None

        # 4. From journal and note
        for text in (getattr(memory, "journal", ""), getattr(memory, "note", "")):
            if text:
                for m in brain_regex.finditer(text):
                    raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw and raw not in candidates:
                        candidates[raw] = None

        temp_artifacts_dir = get_aivc_temp_dir() / "artifacts"
        user_temp_artifacts_dir = get_aivc_user_temp_dir() / "artifacts"
        mirror_user_temp = (user_temp_artifacts_dir.resolve() != temp_artifacts_dir.resolve())
        drive_candidates_to_pull: list[tuple[str, Path, list[Path], list[str]]] = []
        cache_file = self._root / "missing_blobs_cache.json"
        for sub, blob_hash in candidates.items():
            local_path = home / ".gemini" / "antigravity" / "brain" / Path(sub)
            cli_path = home / ".gemini" / "antigravity-cli" / "brain" / Path(sub)
            temp_artifact_path = temp_artifacts_dir / Path(sub)
            user_temp_artifact_path = user_temp_artifacts_dir / Path(sub) if mirror_user_temp else None
            sub_posix = sub.replace("\\", "/")
            portable = f"~/.gemini/antigravity/brain/{sub_posix}"
            extra_targets = [p for p in (temp_artifact_path, cli_path, user_temp_artifact_path) if p is not None]

            # Purge corrupt 8-byte 'restored' artifact if present in local_path
            if local_path.is_file() and local_path.stat().st_size == 8:
                try:
                    if local_path.read_bytes() == b"restored":
                        local_path.unlink()
                except OSError:
                    pass

            # Mirror to target mirrors if local_path already exists and valid
            if local_path.is_file():
                for target in extra_targets:
                    is_corrupt = False
                    try:
                        if target.is_file() and target.stat().st_size == 8:
                            is_corrupt = (target.read_bytes() == b"restored")
                    except OSError:
                        pass

                    needs_mirror = (
                        not target.is_file()
                        or is_corrupt
                        or target.stat().st_size != local_path.stat().st_size
                        or target.stat().st_mtime < local_path.stat().st_mtime
                    )
                    if needs_mirror:
                        try:
                            if target.parent.is_file():
                                target.parent.unlink(missing_ok=True)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(local_path, target)
                        except OSError:
                            pass
                continue

            content: bytes | None = None

            # 1. Try retrieving directly from blob_store using blob_hash
            if blob_hash:
                try:
                    content = self._blob_store.retrieve(blob_hash)
                except KeyError:
                    content = None

            # 2. If not found, check if another commit indexed this exact file
            if content is None:
                try:
                    hashes = self._index.get_blob_hashes_for_file(portable)
                    for h in hashes:
                        try:
                            content = self._blob_store.retrieve(h)
                            if content:
                                break
                        except KeyError:
                            continue
                except Exception:
                    pass


            # 3. If still not found, try sync_manager (Google Drive) if available
            if content is None and sync_manager is not None:
                cache_file = self._root / "missing_blobs_cache.json"
                missing_cache: dict[str, float] = {}
                if cache_file.exists():
                    try:
                        missing_cache = json.loads(cache_file.read_text(encoding="utf-8"))
                    except Exception:
                        missing_cache = {}

                now = time.time()
                ttl = 86400.0  # 24 hours

                # Check if file path or blob_hash is negatively cached
                is_neg_cached = False
                cache_keys = [k for k in (portable, str(local_path), blob_hash) if k]
                for key in cache_keys:
                    if key in missing_cache:
                        if now - missing_cache[key] < ttl:
                            is_neg_cached = True
                            break

                if not is_neg_cached:
                    if background_drive:
                        drive_candidates_to_pull.append((portable, local_path, extra_targets, cache_keys))
                    else:
                        try:
                            if getattr(sync_manager, "enabled", False):
                                if local_path.parent.is_file():
                                    local_path.parent.unlink(missing_ok=True)
                                dest = sync_manager.pull_latest_file(portable, target_path=local_path)
                                if dest and dest.is_file():
                                    restored.append(str(local_path))
                                    for target in extra_targets:
                                        try:
                                            if target.parent.is_file():
                                                target.parent.unlink(missing_ok=True)
                                            target.parent.mkdir(parents=True, exist_ok=True)
                                            shutil.copy2(local_path, target)
                                        except OSError:
                                            pass
                                # Clean up cache if previously marked missing
                                updated = False
                                for key in cache_keys:
                                    if key in missing_cache:
                                        del missing_cache[key]
                                        updated = True
                                if updated:
                                    try:
                                        cache_file.write_text(json.dumps(missing_cache, indent=2), encoding="utf-8")
                                    except Exception:
                                        pass
                                continue
                            else:
                                is_neg_cached = True
                        except FileNotFoundError:
                            is_neg_cached = True

                    if is_neg_cached:
                        for key in cache_keys:
                            missing_cache[key] = now
                        try:
                            cache_file.write_text(json.dumps(missing_cache, indent=2), encoding="utf-8")
                        except Exception:
                            pass
            # Write content retrieved from blob_store to disk
            if content is not None:
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(content)
                    restored.append(str(local_path))
                    for target in extra_targets:
                        try:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(local_path, target)
                        except OSError:
                            pass
                except Exception:
                    pass

        # If any Drive candidates were deferred for background restore, start daemon worker
        if drive_candidates_to_pull and background_drive and sync_manager is not None:
            def _async_drive_worker(
                candidates_to_pull: list[tuple[str, Path, list[Path], list[str]]],
                mgr: Any,
                c_file: Path,
            ) -> None:
                cache_now = time.time()
                m_cache: dict[str, float] = {}
                if c_file.exists():
                    try:
                        m_cache = json.loads(c_file.read_text(encoding="utf-8"))
                    except Exception:
                        m_cache = {}

                for port_path, l_path, targets, c_keys in candidates_to_pull:
                    pull_ok = False
                    try:
                        if getattr(mgr, "enabled", False):
                            if l_path.parent.is_file():
                                l_path.parent.unlink(missing_ok=True)
                            dest = mgr.pull_latest_file(port_path, target_path=l_path)
                            if dest and dest.is_file():
                                pull_ok = True
                                for t_path in targets:
                                    try:
                                        if t_path.parent.is_file():
                                            t_path.parent.unlink(missing_ok=True)
                                        t_path.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(l_path, t_path)
                                    except OSError:
                                        pass
                                for k in c_keys:
                                    m_cache.pop(k, None)
                    except FileNotFoundError:
                        pull_ok = False

                    if not pull_ok:
                        for k in c_keys:
                            m_cache[k] = cache_now

                try:
                    c_file.write_text(json.dumps(m_cache, indent=2), encoding="utf-8")
                except Exception:
                    pass

            t = threading.Thread(
                target=_async_drive_worker,
                args=(list(drive_candidates_to_pull), sync_manager, cache_file),
                daemon=True,
                name="AivcAutoRestoreDrivePull",
            )
            self._last_restore_thread = t
            t.start()

        return restored


