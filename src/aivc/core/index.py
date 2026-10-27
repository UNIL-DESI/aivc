"""
CoreIndex: SQLite-backed index for commit metadata and file changes.

This index sits in the core/ layer and provides O(1) or O(log N) access
to information that was previously loaded via heavy O(N) JSON scans.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path, PurePath
from typing import TYPE_CHECKING

from aivc.config import to_portable_path

if TYPE_CHECKING:
    from aivc.core.memory import Memory

_DB_FILE = "core_index.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    commit_id  TEXT PRIMARY KEY,
    parent_id  TEXT,
    timestamp  TEXT NOT NULL,
    title      TEXT NOT NULL,
    machine_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_commits_parent ON commits(parent_id);

CREATE TABLE IF NOT EXISTS file_changes (
    commit_id     TEXT NOT NULL REFERENCES commits(commit_id),
    path          TEXT NOT NULL,
    action        TEXT NOT NULL,
    blob_hash     TEXT,
    bytes_added   INTEGER NOT NULL,
    bytes_removed INTEGER NOT NULL,
    UNIQUE(commit_id, path)
);

CREATE INDEX IF NOT EXISTS idx_fc_path ON file_changes(path);
CREATE INDEX IF NOT EXISTS idx_fc_commit ON file_changes(commit_id);
"""


class CoreIndex:
    """Fast index for memory metadata and file changes.

    Persisted as ``{storage_root}/core_index.db`` (SQLite).
    """

    def __init__(self, storage_root: Path) -> None:
        """Initialize the SQLite index.

        Args:
            storage_root: Root directory where the index DB will be stored.
        """
        storage_root.mkdir(parents=True, exist_ok=True)
        self._db_path = storage_root / _DB_FILE
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_SCHEMA)
        try:
            self._conn.execute("ALTER TABLE commits ADD COLUMN machine_id TEXT")
        except Exception:
            pass
        self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def add_memory(self, memory: Memory) -> None:
        """Index a memory and its file changes.

        Idempotent: uses INSERT OR REPLACE.
        """
        self._execute(
            "INSERT OR REPLACE INTO commits (commit_id, parent_id, timestamp, title, machine_id) VALUES (?, ?, ?, ?, ?)",
            (memory.id, memory.parent_id, memory.timestamp, memory.title, getattr(memory, "machine_id", "")),
        )

        for fc in memory.changes:
            portable_path = to_portable_path(fc.path)
            self._execute(
                "INSERT OR REPLACE INTO file_changes (commit_id, path, action, blob_hash, bytes_added, bytes_removed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory.id, portable_path, fc.action, fc.blob_hash, fc.bytes_added, fc.bytes_removed),
            )
        self._conn.commit()

    def get_recent_commits(
        self, limit: int = 20, offset: int = 0, only_local_machine_id: str | None = None
    ) -> list[str]:
        """Return commit IDs in reverse chronological order.

        Args:
            limit: Maximum number of commit IDs to return.
            offset: Number of commit IDs to skip.
            only_local_machine_id: If provided, filter to commits matching this machine_id, NULL, or empty.

        Returns:
            List of commit_id strings.
        """
        if only_local_machine_id is not None:
            rows = self._execute(
                "SELECT commit_id FROM commits WHERE machine_id = ? OR machine_id IS NULL OR machine_id = '' ORDER BY timestamp DESC, commit_id DESC LIMIT ? OFFSET ?",
                (only_local_machine_id, limit, offset),
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT commit_id FROM commits ORDER BY timestamp DESC, commit_id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [r[0] for r in rows]

    def remove_file_changes(self, file_path: str) -> None:
        """Remove all file_change entries for a specific path (used for untrack)."""
        target_portable = to_portable_path(file_path)
        rows = self._execute("SELECT DISTINCT path FROM file_changes").fetchall()
        for (p,) in rows:
            if to_portable_path(p) == target_portable:
                self._execute("DELETE FROM file_changes WHERE path = ?", (p,))
        self._conn.commit()

    def get_blob_hashes_for_file(self, file_path: str) -> set[str]:
        """Return all unique blob hashes ever associated with this file."""
        target_portable = to_portable_path(file_path)
        rows = self._execute(
            "SELECT DISTINCT path, blob_hash FROM file_changes WHERE blob_hash IS NOT NULL"
        ).fetchall()
        return {h for p, h in rows if to_portable_path(p) == target_portable}

    def get_all_blob_hashes_by_file(self) -> dict[str, set[str]]:
        """Return a mapping of file_path -> set(blob_hashes) for all files."""
        rows = self._execute(
            "SELECT path, blob_hash FROM file_changes WHERE blob_hash IS NOT NULL"
        ).fetchall()
        result = {}
        for path, hash_ in rows:
            p_port = to_portable_path(path)
            if p_port not in result:
                result[p_port] = set()
            result[p_port].add(hash_)
        return result

    def find_child(self, memory_id: str) -> tuple[str, str] | None:
        """Find the child memory ID and title for a given parent ID."""
        row = self._execute(
            "SELECT commit_id, title FROM commits WHERE parent_id = ?", (memory_id,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def get_memories_touching_file(self, file_path: str) -> list[str]:
        """Return all memory IDs that recorded a change for this file."""
        raw_s = file_path.strip().replace("\\", "/")
        is_rel = not (raw_s.startswith(("~", "/")) or (len(raw_s) >= 2 and raw_s[1] == ":"))
        rel_norm = raw_s.lstrip("./")
        target_portable = to_portable_path(file_path)
        basename = PurePath(raw_s).name

        rows = self._execute(
            "SELECT DISTINCT commit_id, path FROM file_changes WHERE path = ? OR path = ? OR path LIKE ? OR path LIKE ?",
            (file_path, target_portable, f"%/{basename}", f"%\\{basename}"),
        ).fetchall()

        matching: list[str] = []
        # Passe 1 : match exact ou suffixe relatif
        for commit_id, path in rows:
            p = to_portable_path(path)
            if p == target_portable or (is_rel and rel_norm and (p == rel_norm or p.endswith("/" + rel_norm))):
                if commit_id not in matching:
                    matching.append(commit_id)

        # Passe 2 : repli par basename si et seulement si matching est vide et basename non vide
        if not matching and basename:
            for commit_id, path in rows:
                if PurePath(to_portable_path(path).replace("\\", "/")).name == basename:
                    if commit_id not in matching:
                        matching.append(commit_id)

        return matching

    def migrate_from_json(self, memories_dir: Path) -> int:
        """Load all JSON memories from memories_dir and index them if not already present.

        Returns:
            The number of newly indexed memories.
        """
        from aivc.core.memory import memory_from_dict

        # Optimization: Load all existing commit IDs into a set once
        existing_commits_rows = self._execute("SELECT commit_id FROM commits").fetchall()
        existing_commits = {r[0] for r in existing_commits_rows}

        new_count = 0
        for p in memories_dir.glob("*.json"):
            memory_id = p.stem

            if memory_id not in existing_commits:
                try:
                    memory = memory_from_dict(json.loads(p.read_text(encoding="utf-8")))
                    self.add_memory(memory)
                    new_count += 1
                except Exception:
                    # Skip corrupted or invalid memories during migration
                    continue
        return new_count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
