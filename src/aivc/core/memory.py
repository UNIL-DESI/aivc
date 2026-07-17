"""
Memory data structures for AIVC.

A Memory is the atomic unit of memory: a short title + a detailed Markdown note
that will later be vectorized for semantic search.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FileChange:
    """Represents the state change of a single tracked file in a memory."""

    path: str
    """Relative path of the file."""

    action: str
    """One of 'added', 'modified', 'deleted'."""

    blob_hash: str | None
    """SHA-256 hash of the new blob. None when action is 'deleted'."""

    bytes_added: int
    """Size of the new blob in bytes. 0 when action is 'deleted'."""

    bytes_removed: int
    """Size of the previous blob in bytes. 0 when action is 'added'."""

    def __post_init__(self) -> None:
        _VALID_ACTIONS = {"added", "modified", "deleted", "consulted"}
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid FileChange action {self.action!r}. "
                f"Must be one of {_VALID_ACTIONS}."
            )
        if self.action in ("deleted",) and self.blob_hash is not None:
            raise ValueError(
                f"FileChange with action={self.action!r} must have blob_hash=None."
            )
        if self.action in ("added", "modified") and self.blob_hash is None:
            raise ValueError(
                f"FileChange with action={self.action!r} must have a blob_hash."
            )

    @property
    def size_delta(self) -> int:
        """Net byte change: positive means growth, negative means shrinkage."""
        return self.bytes_added - self.bytes_removed

    def format_impact(self) -> str:
        """Human-readable size impact, e.g. '+1.2 KB / -512 B'."""
        return f"+{_format_bytes(self.bytes_added)} / -{_format_bytes(self.bytes_removed)}"


@dataclass
class Memory:
    """A versioning memory — the atomic unit of LLM memory."""

    id: str
    """UUID v4 unique identifier."""

    timestamp: str
    """ISO 8601 UTC creation timestamp."""

    title: str
    """Short title summarising the achievement."""

    note: str
    """Detailed Markdown note — the full 'memory' to be vectorised later."""

    parent_id: str | None
    """ID of the parent memory, or None for the initial memory."""

    changes: list[FileChange] = field(default_factory=list)
    """List of file changes recorded in this memory."""

    machine_id: str = ""
    """ID of the machine where the memory was created (empty for local/legacy)."""

    @classmethod
    def create(
        cls,
        title: str,
        note: str,
        parent_id: str | None,
        changes: list[FileChange],
        machine_id: str = "",
    ) -> "Memory":
        """Factory: create a new Memory with a fresh UUID and current UTC timestamp."""
        if not title.strip():
            raise ValueError("Memory title cannot be empty.")
        if not note.strip():
            raise ValueError("Memory note cannot be empty.")
        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            note=note,
            parent_id=parent_id,
            changes=changes,
            machine_id=machine_id,
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def memory_to_dict(memory: Memory) -> dict[str, Any]:
    """Serialise a Memory to a JSON-compatible dict."""
    return {
        "id": memory.id,
        "timestamp": memory.timestamp,
        "title": memory.title,
        "note": memory.note,
        "parent_id": memory.parent_id,
        "machine_id": memory.machine_id,
        "changes": [
            {
                "path": c.path,
                "action": c.action,
                "blob_hash": c.blob_hash,
                "bytes_added": c.bytes_added,
                "bytes_removed": c.bytes_removed,
            }
            for c in memory.changes
        ],
    }


def memory_from_dict(data: dict[str, Any]) -> Memory:
    """Deserialise a Memory from a dict. Crashes on any missing or invalid field."""
    _required = {"id", "timestamp", "title", "note", "parent_id", "changes"}
    _missing = _required - data.keys()
    if _missing:
        raise ValueError(
            f"Cannot deserialise Memory — missing fields: {_missing}. "
            f"Got keys: {set(data.keys())}."
        )

    raw_changes: list[dict[str, Any]] = data["changes"]
    changes = []
    for i, rc in enumerate(raw_changes):
        _change_required = {"path", "action", "blob_hash", "bytes_added", "bytes_removed"}
        _missing_change = _change_required - rc.keys()
        if _missing_change:
            raise ValueError(
                f"Cannot deserialise FileChange[{i}] — missing fields: {_missing_change}."
            )
        changes.append(
            FileChange(
                path=rc["path"],
                action=rc["action"],
                blob_hash=rc["blob_hash"],
                bytes_added=rc["bytes_added"],
                bytes_removed=rc["bytes_removed"],
            )
        )

    return Memory(
        id=data["id"],
        timestamp=data["timestamp"],
        title=data["title"],
        note=data["note"],
        parent_id=data["parent_id"],
        machine_id=data.get("machine_id", ""),  # Default to empty for legacy
        changes=changes,
    )


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"

