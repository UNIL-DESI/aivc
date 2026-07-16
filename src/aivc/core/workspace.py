"""
Workspace: main entry point for the AIVC versioning engine.

Manages tracked files, creates commits, computes statuses, and handles
the untrack + GC lifecycle.
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aivc.core.blob_store import BlobStore
from aivc.core.memory import Memory, FileChange, memory_from_dict, memory_to_dict
from aivc.core.diff import compute_diff
from aivc.core.index import CoreIndex


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

        if self._workspace_path.exists():
            self._state = self._load_state()
            self._last_mtime = self._workspace_path.stat().st_mtime
            
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
        return memory_from_dict(json.loads(path.read_text(encoding="utf-8")))

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, path: str, ignores: list[str] | None = None) -> dict[str, Any]:
        """Add a file, directory, or glob pattern to tracking.

        If the path is a directory, it is automatically added to the continuous
        surveillance list (watched_dirs).

        Args:
            path: A file path, directory path, or glob pattern.
            ignores: Optional list of globs to ignore (only applicable if watching a dir).

        Returns:
            A dict with:
            - "newly_tracked": list of file paths newly added.
            - "hidden_skipped": count of hidden files ignored.

        Raises:
            ValueError: if no matching files are found.
        """
        self._reload_state_if_needed()
        p = Path(path).resolve()
        abs_p = str(p)

        # 1. Register for surveillance if it's a directory
        if p.is_dir():
            self._state["watched_dirs"][abs_p] = {"ignores": ignores or []}

        # 2. Expand and track existing files
        files, hidden_count = self._expand_path(path)
        newly_tracked = []
        import fnmatch

        for f in files:
            abs_f = str(Path(f).resolve())
            skip = False

            # Check explicit ignores
            if ignores:
                if any(fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(Path(f).name, pat) for pat in ignores):
                    skip = True

            # Check ignores from watched directories
            if not skip:
                for wdir, info in self._state.get("watched_dirs", {}).items():
                    if abs_f.startswith(wdir + os.sep):
                        w_ignores = info.get("ignores", [])
                        if any(fnmatch.fnmatch(abs_f, pat) or fnmatch.fnmatch(Path(abs_f).name, pat) for pat in w_ignores):
                            skip = True
                            break

            if skip:
                continue

            if abs_f not in self._state["tracked_files"]:
                # Initialize with richer format
                self._state["tracked_files"][abs_f] = {"hash": None, "mtime": None, "size": None}
                newly_tracked.append(abs_f)
        self._save_state()
        return {
            "newly_tracked": newly_tracked,
            "hidden_skipped": hidden_count
        }


    def untrack(self, path_or_glob: str) -> None:
        """Remove a file, directory, or glob from tracking and garbage-collect history.

        WARNING: This is a highly destructive operation. If a directory or glob
        is provided, ALL matching files currently tracked will have their history
        permanently erased from AIVC.
        This also removes the path from continuous surveillance if applicable.

        Raises:
            KeyError: if no matching files or watched directories are found.
        """
        self._reload_state_if_needed()
        abs_p = str(Path(path_or_glob).resolve())
        removed_watch = False

        if abs_p in self._state.get("watched_dirs", {}):
            del self._state["watched_dirs"][abs_p]
            removed_watch = True

        to_untrack = set()
        p_is_dir = Path(abs_p).is_dir()
        
        for tracked_file in self._state["tracked_files"]:
            if tracked_file == abs_p:
                to_untrack.add(tracked_file)
            elif p_is_dir and tracked_file.startswith(abs_p + os.sep):
                to_untrack.add(tracked_file)
            elif fnmatch.fnmatch(tracked_file, abs_p):
                to_untrack.add(tracked_file)

        if not to_untrack and not removed_watch:
            raise KeyError(f"Path {path_or_glob!r} is not tracked or watched.")

        for file_path in to_untrack:
            # Collect memories referencing this file via the index (fast).
            affected_memory_ids = self._index.get_memories_touching_file(file_path)
            with self._blob_store.batch():
                for mid in affected_memory_ids:
                    memory = self._load_memory(mid)
                    updated_changes = []
                    for change in memory.changes:
                        if change.path == file_path:
                            if change.blob_hash is not None:
                                self._blob_store.decrement_ref(change.blob_hash)
                        else:
                            updated_changes.append(change)
                    
                    if len(updated_changes) != len(memory.changes):
                        memory.changes = updated_changes
                        self._save_memory(memory)

            # Cleanup the index for this file.
            self._index.remove_file_changes(file_path)
            del self._state["tracked_files"][file_path]
            
        self._save_state()

    def get_tracked_paths(self) -> list[str]:
        """Return just the list of tracked file absolute paths (fast)."""
        self._reload_state_if_needed()
        return list(self._state["tracked_files"].keys())

    def create_memory(
        self,
        title: str,
        note: str,
        consulted_files: list[str] | None = None,
        machine_id: str = "",
    ) -> Memory:
        """Detect changes in tracked files and create a new memory.

        Args:
            title: Short title for the memory.
            note: Detailed Markdown note (the LLM's 'memory').
            consulted_files: Optional list of file paths that were consulted
                             but not modified. Files that exist on disk but
                             aren't tracked will be auto-tracked. Non-existent
                             files are silently skipped.
            machine_id: ID of the machine where the memory was created.

        Returns:
            The newly created Memory.

        Raises:
            RuntimeError: if no changes are detected and no files were consulted.
        """
        self._reload_state_if_needed()
        with self._blob_store.batch():
            changes = compute_diff(self._state["tracked_files"], self._blob_store)
            
            # Handle consulted files
            consulted_changes = []
            if consulted_files:
                for path in consulted_files:
                    abs_path = str(Path(path).resolve())
                    if abs_path not in self._state["tracked_files"]:
                        # Auto-track consulted files if they exist on disk
                        if Path(abs_path).is_file():
                            self._state["tracked_files"][abs_path] = {"hash": None, "mtime": None, "size": None}
                        else:
                            # File doesn't exist — skip silently
                            continue
                    
                    # Check if it was already modified/added/deleted.
                    # If it's already in 'changes', we don't add it as 'consulted'.
                    if any(c.path == abs_path for c in changes):
                        continue

                    consulted_changes.append(
                        FileChange(
                            path=abs_path,
                            action="consulted",
                            blob_hash=None,
                            bytes_added=0,
                            bytes_removed=0,
                        )
                    )
            
            all_changes = changes + consulted_changes

        if not all_changes:
            raise RuntimeError(
                "No changes detected in tracked files and no files consulted. "
                "Nothing to remember."
            )

        memory = Memory.create(
            title=title,
            note=note,
            parent_id=self._state["head_commit_id"],
            changes=all_changes,
            machine_id=machine_id,
        )
        self._save_memory(memory)
        self._index.add_memory(memory)

        # Update tracked_files with the new hashes and metadata.
        for change in changes:
            if change.action == "deleted":
                # Keep the file in tracking but mark hash as None (it might come back).
                self._state["tracked_files"][change.path] = {"hash": None, "mtime": None, "size": None}
            else:
                # Store new hash and capture current disk metadata
                p = Path(change.path)
                stat = p.stat()
                self._state["tracked_files"][change.path] = {
                    "hash": change.blob_hash,
                    "mtime": stat.st_mtime,
                    "size": stat.st_size
                }

        self._state["head_commit_id"] = memory.id
        self._save_state()
        return memory

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
        
        for abs_path in self._state["tracked_files"]:
            p = Path(abs_path)
            
            # Current size: check disk for correctness (deleted file detection)
            current_size = None
            if p.exists() and p.is_file():
                try:
                    current_size = p.stat().st_size
                except (PermissionError, OSError):
                    current_size = None

            # History size: use bulk index map
            history_size = 0
            hashes = all_hashes.get(abs_path, set())
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
                    path=abs_path,
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

    def get_log(self, limit: int = 20, offset: int = 0) -> list[Memory]:
        """Return up to `limit` memories in reverse chronological order.

        Traverses the memory chain via parent_id starting from HEAD.
        Skips the first `offset` memories for pagination support.
        """
        self._reload_state_if_needed()
        memories: list[Memory] = []
        current_id = self._state["head_commit_id"]
        skipped = 0
        while current_id is not None and len(memories) < limit:
            memory = self._load_memory(current_id)
            if skipped < offset:
                skipped += 1
            else:
                memories.append(memory)
            current_id = memory.parent_id
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
        # Walk the chain from the target memory backwards.
        current_id: str | None = memory_id
        while current_id is not None:
            memory = self._load_memory(current_id)
            for change in memory.changes:
                if change.path == file_path and change.blob_hash is not None:
                    return self._blob_store.retrieve(change.blob_hash)
            current_id = memory.parent_id

        raise KeyError(
            f"File {file_path!r} was not found in the history up to memory {memory_id!r}."
        )

