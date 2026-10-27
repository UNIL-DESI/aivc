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
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aivc.core.blob_store import BlobStore
from aivc.core.memory import Memory, FileChange, memory_from_dict, memory_to_dict, _compose_note
from aivc.core.diff import compute_diff
from aivc.core.index import CoreIndex
from aivc.config import get_machine_id, to_portable_path, from_portable_path


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
        user_items: list[dict[str, Any]] | None = None,
        edited_files: list[str] | None = None,
        read_files: list[str] | None = None,
        urls: list[str] | None = None,
        note: str | None = None,
        machine_id: str = "",
        consulted_files: list[str] | None = None,
    ) -> Memory:
        """Create a new memory in 3-part format (title, journal, user_items) without local diffs/blobs.

        Args:
            title: Short title summarising the achievement.
            journal: Agent thoughts and activity log.
            user_items: User-facing task items [{id, task, details, done}].
            edited_files: List of file paths modified or created.
            read_files: List of file paths consulted.
            urls: Optional list of URLs.
            note: Backward-compatibility parameter mapping to journal.
            machine_id: ID of the machine where the memory was created.
            consulted_files: Legacy parameter mapping to read_files.
        """
        self._reload_state_if_needed()

        if not journal and note:
            journal = note

        actual_urls = [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]
        actual_read_files = list(read_files) if read_files is not None else []
        if consulted_files:
            actual_read_files.extend(consulted_files)
        actual_edited_files = list(edited_files) if edited_files is not None else []

        memory_id = str(uuid.uuid4())
        norm_user_items = []
        if user_items:
            for idx, item in enumerate(user_items, 1):
                norm_user_items.append(
                    {
                        "id": item.get("id") or f"{memory_id[:6]}-{idx}",
                        "task": str(item.get("task", "")).strip(),
                        "details": str(item.get("details", "")).strip(),
                        "done": bool(item.get("done", False)),
                    }
                )

        # Strict validation
        for files_list, list_name in [(actual_read_files, "read_files"), (actual_edited_files, "edited_files")]:
            for path_str in files_list:
                p = from_portable_path(path_str)
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
            p = from_portable_path(path_str)
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            note=full_note,
            parent_id=self._state.get("head_commit_id"),
            changes=changes,
            machine_id=machine_id,
            urls=actual_urls,
            user_items=norm_user_items,
            journal=journal or (note if not norm_user_items else ""),
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

    def get_file_history(self, file_path: str) -> list[Memory]:
        """Return chronological list of memories touching file_path (portable match)."""
        self._reload_state_if_needed()
        target_portable = to_portable_path(file_path)
        matching_memories = []

        memory_ids = self._index.get_memories_touching_file(file_path)
        for mid in memory_ids:
            try:
                mem = self._load_memory(mid)
                for ch in mem.changes:
                    if to_portable_path(ch.path) == target_portable:
                        matching_memories.append(mem)
                        break
            except KeyError:
                continue

        matching_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return matching_memories

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
            p = from_portable_path(tracked_path)
            
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
                    local_p = from_portable_path(target_portable)
                    if local_p.exists():
                        return local_p.read_bytes()
            current_id = memory.parent_id

        raise KeyError(
            f"File {file_path!r} was not found in the history up to memory {memory_id!r}."
        )

    def auto_restore_brain_files(self, memory: Memory, sync_manager: Any = None) -> list[str]:
        """Automatically restore files under .gemini/antigravity/brain/ linked to this memory.

        If a file does not exist locally and its blob is available in BlobStore
        (or recoverable via other commits or sync_manager), it is restored to disk.
        Returns the list of restored local file paths.
        """
        restored: list[str] = []

        for change in getattr(memory, "changes", []):
            path_str = getattr(change, "path", "")
            if not path_str:
                continue

            # Normalize to portable path, then to current machine local path
            portable = to_portable_path(path_str)
            local_path = from_portable_path(portable)

            # Check if file is located under .gemini/antigravity/brain/
            posix_path = local_path.as_posix()
            if ".gemini/antigravity/brain" not in posix_path and not portable.startswith("~/.gemini/antigravity/brain"):
                continue

            # If file already exists locally on disk, nothing to restore
            if local_path.is_file():
                continue

            content: bytes | None = None
            blob_hash = getattr(change, "blob_hash", None)

            # 1. Try retrieving directly from blob_store using change.blob_hash
            if blob_hash:
                try:
                    content = self._blob_store.retrieve(blob_hash)
                except KeyError:
                    content = None

            # 2. If not found in change, check if another commit indexed this exact file or filename
            if content is None:
                try:
                    hashes = self._index.get_blob_hashes_for_file(portable)
                    if not hashes:
                        filename = local_path.name
                        cursor = self._index._conn.cursor()
                        cursor.execute(
                            "SELECT blob_hash FROM file_changes WHERE (path LIKE ? OR path LIKE ?) AND blob_hash IS NOT NULL LIMIT 1",
                            (f"%/{filename}", f"%\\{filename}")
                        )
                        row = cursor.fetchone()
                        if row and row[0]:
                            hashes = {row[0]}
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
                    try:
                        if getattr(sync_manager, "enabled", False):
                            dest = sync_manager.pull_latest_file(portable, target_path=local_path)
                            if dest and dest.is_file():
                                restored.append(str(local_path))
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
                    except Exception:
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
                except Exception:
                    pass

        return restored


