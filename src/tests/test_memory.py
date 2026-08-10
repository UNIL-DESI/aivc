"""Unit tests for Memory and FileChange dataclasses."""

import pytest
from aivc.core.memory import Memory, FileChange, memory_from_dict, memory_to_dict


def make_file_change(
    path: str = "src/foo.py",
    action: str = "modified",
    blob_hash: str = "abc123",
    bytes_added: int = 100,
    bytes_removed: int = 80,
) -> FileChange:
    return FileChange(
        path=path,
        action=action,
        blob_hash=blob_hash,
        bytes_added=bytes_added,
        bytes_removed=bytes_removed,
    )


def make_memory(**kwargs) -> Memory:
    defaults = dict(
        title="Add feature X",
        note="## Details\n\nThis is a detailed note.",
        parent_id=None,
        changes=[make_file_change()],
    )
    defaults.update(kwargs)
    return Memory.create(**defaults)


# ---------------------------------------------------------------------------
# FileChange validation
# ---------------------------------------------------------------------------

def test_file_change_invalid_action_crashes() -> None:
    with pytest.raises(ValueError, match="Invalid FileChange action"):
        FileChange(path="f.py", action="RENAMED", blob_hash="abc", bytes_added=0, bytes_removed=0)


def test_file_change_deleted_must_have_no_blob_hash() -> None:
    with pytest.raises(ValueError, match="blob_hash=None"):
        FileChange(path="f.py", action="deleted", blob_hash="abc", bytes_added=0, bytes_removed=0)


def test_file_change_added_must_have_blob_hash() -> None:
    with pytest.raises(ValueError, match="must have a blob_hash"):
        FileChange(path="f.py", action="added", blob_hash=None, bytes_added=50, bytes_removed=0)


def test_file_change_size_delta() -> None:
    fc = make_file_change(bytes_added=200, bytes_removed=150)
    assert fc.size_delta == 50


def test_file_change_format_impact() -> None:
    fc = make_file_change(bytes_added=1024, bytes_removed=512)
    impact = fc.format_impact()
    assert "1.0 KB" in impact
    assert "512 B" in impact


# ---------------------------------------------------------------------------
# Memory creation
# ---------------------------------------------------------------------------

def test_memory_create_generates_unique_ids() -> None:
    m1 = make_memory()
    m2 = make_memory()
    assert m1.id != m2.id


def test_memory_empty_title_crashes() -> None:
    with pytest.raises(ValueError, match="title cannot be empty"):
        Memory.create(title="  ", note="valid note", parent_id=None, changes=[])


def test_memory_empty_note_crashes() -> None:
    with pytest.raises(ValueError, match="note cannot be empty"):
        Memory.create(title="Valid title", note="   ", parent_id=None, changes=[])


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------

def test_memory_roundtrip() -> None:
    original = make_memory()
    restored = memory_from_dict(memory_to_dict(original))
    assert restored.id == original.id
    assert restored.title == original.title
    assert restored.note == original.note
    assert restored.parent_id == original.parent_id
    assert len(restored.changes) == len(original.changes)
    assert restored.changes[0].path == original.changes[0].path
    assert restored.changes[0].bytes_added == original.changes[0].bytes_added


def test_memory_from_dict_crashes_on_missing_field() -> None:
    d = memory_to_dict(make_memory())
    del d["note"]
    with pytest.raises(ValueError, match="missing fields"):
        memory_from_dict(d)


def test_memory_from_dict_crashes_on_missing_change_field() -> None:
    d = memory_to_dict(make_memory())
    del d["changes"][0]["bytes_added"]
    with pytest.raises(ValueError, match="missing fields"):
        memory_from_dict(d)


def test_memory_urls_creation_and_roundtrip() -> None:
    urls = ["https://example.com/spec", "https://arxiv.org/abs/1234.5678"]
    original = make_memory(urls=urls)
    assert original.urls == urls

    d = memory_to_dict(original)
    assert d["urls"] == urls

    restored = memory_from_dict(d)
    assert restored.urls == urls


def test_memory_from_dict_legacy_backwards_compat() -> None:
    d = memory_to_dict(make_memory())
    del d["urls"]  # Simulate legacy commit JSON without urls key
    restored = memory_from_dict(d)
    assert restored.urls == []

