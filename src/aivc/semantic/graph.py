"""
CooccurrenceGraph: bipartite graph tracking file ↔ memory relationships.

Persisted in an SQLite database for O(1) inserts and efficient queries,
eliminating the full-JSON-rewrite bottleneck of the original design.

Schema:
    commit_nodes(commit_id TEXT PK, title TEXT, timestamp TEXT)
    file_nodes(file_path TEXT PK)
    edges(commit_id TEXT, file_path TEXT, UNIQUE(commit_id, file_path))

Edges are explicit rows: an edge exists between a file and a memory iff
a row appears in the ``edges`` table.
"""

from __future__ import annotations

import fnmatch
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aivc.core.memory import Memory

_DB_FILE = "cooccurrence_graph.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commit_nodes (
    commit_id TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS file_nodes (
    file_path TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS edges (
    commit_id TEXT NOT NULL REFERENCES commit_nodes(commit_id),
    file_path TEXT NOT NULL REFERENCES file_nodes(file_path),
    UNIQUE(commit_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_edges_commit ON edges(commit_id);
CREATE INDEX IF NOT EXISTS idx_edges_file   ON edges(file_path);
"""


class CooccurrenceGraph:
    """Bipartite graph of file nodes ↔ memory nodes.

    Persisted as ``{storage_root}/cooccurrence_graph.db`` (SQLite).
    """

    def __init__(self, storage_root: Path) -> None:
        """Load or create the co-occurrence graph.

        Args:
            storage_root: Directory shared with BlobStore / ChromaDB.
        """
        storage_root.mkdir(parents=True, exist_ok=True)
        self._db_path = storage_root / _DB_FILE
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def _executemany(self, sql: str, params: list[tuple]) -> None:
        self._conn.executemany(sql, params)

    # ------------------------------------------------------------------
    # Public API — Mutation
    # ------------------------------------------------------------------

    def add_memory(self, memory: "Memory") -> None:
        """Register a memory and its file edges in the graph.

        Idempotent: calling with the same memory twice is safe.

        Args:
            memory: The memory to add.
        """
        file_paths = [c.path for c in memory.changes]

        self._execute(
            "INSERT OR REPLACE INTO commit_nodes (commit_id, title, timestamp) VALUES (?, ?, ?)",
            (memory.id, memory.title, memory.timestamp),
        )

        if file_paths:
            self._executemany(
                "INSERT OR IGNORE INTO file_nodes (file_path) VALUES (?)",
                [(fp,) for fp in file_paths],
            )
            self._executemany(
                "INSERT OR IGNORE INTO edges (commit_id, file_path) VALUES (?, ?)",
                [(memory.id, fp) for fp in file_paths],
            )

        self._conn.commit()

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory and all its edges from the graph.

        Args:
            memory_id: UUID of the memory to remove.

        Raises:
            KeyError: if the memory_id is not in the graph.
        """
        row = self._execute(
            "SELECT 1 FROM commit_nodes WHERE commit_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory {memory_id!r} is not in the graph.")

        # Get files associated with this memory BEFORE deleting edges.
        file_paths = [
            r[0]
            for r in self._execute(
                "SELECT file_path FROM edges WHERE commit_id = ?", (memory_id,)
            ).fetchall()
        ]

        # Remove all edges for this memory.
        self._execute("DELETE FROM edges WHERE commit_id = ?", (memory_id,))

        # Remove the memory node.
        self._execute("DELETE FROM commit_nodes WHERE commit_id = ?", (memory_id,))

        # Clean up orphan file nodes (files that no longer have any edges).
        if file_paths:
            # Use chunks to avoid SQLite's parameter limit (usually 999).
            unique_paths = list(set(file_paths))
            for i in range(0, len(unique_paths), 900):
                chunk = unique_paths[i : i + 900]
                placeholders = ",".join("?" for _ in chunk)
                self._execute(
                    f"""
                    DELETE FROM file_nodes
                    WHERE file_path IN ({placeholders})
                      AND NOT EXISTS (
                          SELECT 1 FROM edges WHERE edges.file_path = file_nodes.file_path
                      )
                    """,
                    tuple(chunk),
                )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API — Queries
    # ------------------------------------------------------------------

    def get_memory_files(self, memory_id: str) -> list[str]:
        """Return the file paths associated with a memory.

        Args:
            memory_id: UUID of the memory.

        Raises:
            KeyError: if the memory is not in the graph.
        """
        row = self._execute(
            "SELECT 1 FROM commit_nodes WHERE commit_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory {memory_id!r} is not in the graph.")

        return [
            r[0]
            for r in self._execute(
                "SELECT file_path FROM edges WHERE commit_id = ?", (memory_id,)
            ).fetchall()
        ]

    def get_file_memories(self, file_path: str) -> list[str]:
        """Return the memory IDs that touched a given file.

        Args:
            file_path: Relative path of the file.

        Raises:
            KeyError: if the file is not in the graph.
        """
        row = self._execute(
            "SELECT 1 FROM file_nodes WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row is None:
            raise KeyError(f"File {file_path!r} is not in the graph.")

        return [
            r[0]
            for r in self._execute(
                "SELECT commit_id FROM edges WHERE file_path = ?", (file_path,)
            ).fetchall()
        ]

    def get_file_memories_with_metadata(self, file_path: str) -> list[dict]:
        """Return memories that touched a file, with full metadata.

        Args:
            file_path: Absolute path of the file.

        Returns:
            List of ``{"memory_id": str, "title": str, "timestamp": str}``
            ordered by timestamp descending (newest first).

        Raises:
            KeyError: if the file is not in the graph.
        """
        row = self._execute(
            "SELECT 1 FROM file_nodes WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row is None:
            raise KeyError(f"File {file_path!r} is not in the graph.")

        rows = self._execute(
            """
            SELECT c.commit_id, c.title, c.timestamp
            FROM edges e
            JOIN commit_nodes c ON e.commit_id = c.commit_id
            WHERE e.file_path = ?
            ORDER BY c.timestamp DESC
            """,
            (file_path,),
        ).fetchall()

        return [
            {"memory_id": r[0], "title": r[1], "timestamp": r[2]}
            for r in rows
        ]

    def get_memories_by_glob(self, pattern: str) -> list[str]:
        """Return memory IDs that touched any file matching a glob pattern.

        Matches the absolute file paths in the graph against the given glob pattern.

        Args:
            pattern: A glob pattern (e.g. "*/tests/*.py").

        Returns:
            A list of unique memory IDs.
        """
        all_files = [
            r[0] for r in self._execute("SELECT file_path FROM file_nodes").fetchall()
        ]

        matched_files = [fp for fp in all_files if fnmatch.fnmatch(fp, pattern)]

        if not matched_files:
            return []

        memory_ids = set()
        chunk_size = 900  # Safe limit for SQLite `IN` clause parameters

        for i in range(0, len(matched_files), chunk_size):
            chunk = tuple(matched_files[i : i + chunk_size])
            placeholders = ",".join("?" for _ in chunk)
            rows = self._execute(
                f"SELECT DISTINCT commit_id FROM edges WHERE file_path IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                memory_ids.add(r[0])

        return list(memory_ids)

    def get_related_files(
        self, file_path: str, top_n: int = 10
    ) -> list[tuple[str, int]]:
        """Return the files most frequently committed alongside ``file_path``.

        Co-occurrence count: for each memory that touches ``file_path``, count
        every other file that appears in the same memory.

        Args:
            file_path: The reference file.
            top_n: Maximum number of related files to return.

        Returns:
            A list of ``(other_file_path, cooccurrence_count)`` sorted
            descending by count.

        Raises:
            KeyError: if the file is not in the graph.
        """
        row = self._execute(
            "SELECT 1 FROM file_nodes WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row is None:
            raise KeyError(f"File {file_path!r} is not in the graph.")

        # Single SQL query: join edges → edges on commit_id, exclude self.
        rows = self._execute(
            """
            SELECT e2.file_path, COUNT(*) as cnt
            FROM edges e1
            JOIN edges e2 ON e1.commit_id = e2.commit_id
            WHERE e1.file_path = ? AND e2.file_path != ?
            GROUP BY e2.file_path
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (file_path, file_path, top_n),
        ).fetchall()

        return [(r[0], r[1]) for r in rows]

    def to_vis_data(self) -> dict:
        """Export the graph to a visualisation-friendly dict.

        Returns:
            A dict with two lists:
            - ``nodes``: each node has ``id``, ``label``, ``type`` (``"memory"``
              or ``"file"``).
            - ``edges``: each edge has ``source`` (memory_id) and
              ``target`` (file_path).
        """
        nodes = []
        edges_list = []

        for row in self._execute(
            "SELECT commit_id, title, timestamp FROM commit_nodes"
        ).fetchall():
            nodes.append(
                {
                    "id": row[0],
                    "label": row[1],
                    "type": "memory",
                    "timestamp": row[2],
                }
            )

        for row in self._execute("SELECT file_path FROM file_nodes").fetchall():
            nodes.append({"id": row[0], "label": row[0], "type": "file"})

        for row in self._execute(
            "SELECT commit_id, file_path FROM edges"
        ).fetchall():
            edges_list.append({"source": row[0], "target": row[1]})

        return {"nodes": nodes, "edges": edges_list}

    def get_file_node_data(self, connected_files: set[str] | None = None) -> list[dict]:
        """Return enriched data for file nodes (documents) for visualisation.

        Args:
            connected_files: If provided, only return nodes whose file_path
                             is in this set. Used by the dashboard to skip
                             isolated (0-edge) nodes that would overwhelm
                             the layout engine.

        Returns:
            A list of dicts: ``{"id": file_path, "label": file_name, "full_path": file_path, "memory_count": int, "directory": str}``
        """
        nodes = []
        rows = self._execute(
            """
            SELECT f.file_path, COUNT(e.commit_id) as cnt
            FROM file_nodes f
            LEFT JOIN edges e ON f.file_path = e.file_path
            GROUP BY f.file_path
            """
        ).fetchall()

        for file_path, count in rows:
            if connected_files is not None and file_path not in connected_files:
                continue
            directory = str(Path(file_path).parent)
            if directory == ".":
                directory = "/"
            nodes.append({
                "id": file_path,
                "label": Path(file_path).name,
                "full_path": file_path,
                "memory_count": count,
                "directory": directory,
            })
        return nodes

    def get_file_cooccurrences(self) -> list[dict]:
        """Return weighted edges between files that are modified together.

        Returns:
            A list of dicts: ``{"source": file_path_1, "target": file_path_2, "weight": int}``
        """
        rows = self._execute(
            """
            WITH valid_commits AS (
                SELECT commit_id
                FROM edges
                GROUP BY commit_id
                HAVING COUNT(file_path) <= 100
            )
            SELECT e1.file_path, e2.file_path, COUNT(*) as weight
            FROM edges e1
            JOIN edges e2 ON e1.commit_id = e2.commit_id
            JOIN valid_commits vc ON e1.commit_id = vc.commit_id
            WHERE e1.file_path < e2.file_path
            GROUP BY e1.file_path, e2.file_path
            ORDER BY weight DESC
            LIMIT 500
            """
        ).fetchall()

        return [{"source": r[0], "target": r[1], "weight": r[2]} for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


