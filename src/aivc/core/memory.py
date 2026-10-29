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
    """Relative or portable path of the file."""

    action: str
    """One of 'added', 'modified', 'deleted', 'consulted', 'edited'."""

    blob_hash: str | None = None
    """SHA-256 hash of the blob, or None (no blobs stored in new memories)."""

    bytes_added: int = 0
    """Size of the new blob in bytes."""

    bytes_removed: int = 0
    """Size of the previous blob in bytes."""

    def __post_init__(self) -> None:
        _VALID_ACTIONS = {"added", "modified", "deleted", "consulted", "edited", "roadmap"}
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"Invalid FileChange action {self.action!r}. "
                f"Must be one of {_VALID_ACTIONS}."
            )

    @property
    def size_delta(self) -> int:
        """Net byte change: positive means growth, negative means shrinkage."""
        return self.bytes_added - self.bytes_removed

    def format_impact(self) -> str:
        """Human-readable size impact, e.g. '+1.2 KB / -512 B'."""
        return f"+{_format_bytes(self.bytes_added)} / -{_format_bytes(self.bytes_removed)}"


def _compose_note(journal: str, user_items: list[dict[str, Any]] | None, fallback_note: str = "") -> str:
    """Compose the full note text combining journal and user_items for search/indexing."""
    parts = []
    if journal and journal.strip():
        parts.append(journal.strip())
    if user_items:
        items_lines = []
        for item in user_items:
            task = item.get("task", "").strip()
            details = item.get("details", "").strip()
            item_id = item.get("id", "").strip()
            id_prefix = f"[{item_id}] " if item_id else ""
            status_val = item.get("task_status")
            if status_val == "validated" or item.get("done"):
                status = "[x]"
            else:
                status = "[ ]"
            status_tag = f"({status_val}) " if status_val else ""
            line = f"- {status} {id_prefix}{status_tag}{task}"
            if details:
                line += f" : {details}"
            items_lines.append(line)
        if items_lines:
            parts.append("## Actions Utilisateur\n" + "\n".join(items_lines))
    if parts:
        return "\n\n".join(parts)
    return fallback_note


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
    """Detailed Markdown note — the full 'memory' to be vectorised."""

    parent_id: str | None
    """ID of the parent memory, or None for the initial memory."""

    changes: list[FileChange] = field(default_factory=list)
    """List of file changes recorded in this memory."""

    machine_id: str = ""
    """ID of the machine where the memory was created (empty for local/legacy)."""

    urls: list[str] = field(default_factory=list)
    """List of URLs associated with this memory."""

    user_items: list[dict[str, Any]] = field(default_factory=list)
    """List of user-facing task items [{id, task, details, done}]."""

    journal: str = ""
    """Agent thoughts and activity log."""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dict."""
        return memory_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Deserialise from dict."""
        return memory_from_dict(data)

    @classmethod
    def create(
        cls,
        title: str,
        note: str = "",
        parent_id: str | None = None,
        changes: list[FileChange] | None = None,
        machine_id: str = "",
        urls: list[str] | None = None,
        user_items: list[dict[str, Any]] | None = None,
        journal: str = "",
    ) -> "Memory":
        """Factory: create a new Memory with a fresh UUID and current UTC timestamp."""
        if not title.strip():
            raise ValueError("Memory title cannot be empty.")

        actual_user_items = list(user_items) if user_items else []
        actual_journal = journal.strip() if journal else ""

        full_note = _compose_note(actual_journal, actual_user_items, fallback_note=note)
        if not full_note.strip():
            raise ValueError("Memory note cannot be empty.")

        return cls(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            title=title,
            note=full_note,
            parent_id=parent_id,
            changes=changes or [],
            machine_id=machine_id,
            urls=list(urls) if urls else [],
            user_items=actual_user_items,
            journal=actual_journal or (note if not actual_user_items else ""),
        )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def memory_to_dict(memory: Memory) -> dict[str, Any]:
    """Serialise a Memory to a JSON-compatible dict."""
    edited_files = [c.path for c in memory.changes if c.action in ("edited", "added", "modified")]
    read_files = [c.path for c in memory.changes if c.action == "consulted"]
    return {
        "id": memory.id,
        "timestamp": memory.timestamp,
        "title": memory.title,
        "journal": memory.journal,
        "user_items": memory.user_items,
        "note": memory.note,
        "parent_id": memory.parent_id,
        "machine_id": memory.machine_id,
        "urls": list(memory.urls),
        "edited_files": edited_files,
        "read_files": read_files,
        "changes": [
            {
                "path": c.path,
                "action": c.action,
                **({"blob_hash": c.blob_hash} if c.blob_hash else {}),
            }
            for c in memory.changes
        ],
    }



def memory_from_dict(data: dict[str, Any]) -> Memory:
    """Deserialise a Memory from a dict. Crashes on any missing or invalid field."""
    _required = {"id", "timestamp", "title", "parent_id", "changes"}
    _missing = _required - data.keys()
    if _missing:
        raise ValueError(
            f"Cannot deserialise Memory — missing fields: {_missing}. "
            f"Got keys: {set(data.keys())}."
        )

    user_items = data.get("user_items", [])
    journal = data.get("journal") or data.get("note", "")
    note = data.get("note", "")
    full_note = _compose_note(journal, user_items, fallback_note=note)

    raw_changes: list[dict[str, Any]] = data["changes"]
    changes = []
    for i, rc in enumerate(raw_changes):
        if "path" not in rc or "action" not in rc:
            raise ValueError(
                f"Cannot deserialise FileChange[{i}] — missing path or action in {rc}."
            )
        changes.append(
            FileChange(
                path=rc["path"],
                action=rc["action"],
                blob_hash=rc.get("blob_hash"),
                bytes_added=rc.get("bytes_added", 0),
                bytes_removed=rc.get("bytes_removed", 0),
            )
        )

    return Memory(
        id=data["id"],
        timestamp=data["timestamp"],
        title=data["title"],
        note=full_note,
        parent_id=data["parent_id"],
        machine_id=data.get("machine_id", ""),  # Default to empty for legacy
        urls=data.get("urls", []),
        user_items=user_items,
        journal=journal,
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

