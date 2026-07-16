"""Unit tests for the Workspace orchestrator."""

import pytest
from pathlib import Path
from aivc.core.workspace import Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "aivc_storage")


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# create_memory()
# ---------------------------------------------------------------------------

def test_create_memory_basic_cycle(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"v1")
    memory = ws.create_memory("Initial", "## v1\n\nFirst memory.", edited_files=[str(f)])
    assert memory.title == "Initial"
    assert len(memory.changes) == 1
    assert memory.changes[0].action == "added"
    assert memory.parent_id is None


def test_create_memory_second_links_to_first(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"v1")
    m1 = ws.create_memory("v1", "First.", edited_files=[str(f)])
    f.write_bytes(b"v2_changed_size")
    m2 = ws.create_memory("v2", "Second.", edited_files=[str(f)])
    assert m2.parent_id == m1.id


def test_create_memory_no_changes_crashes(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"stable")
    ws.create_memory("Initial", "First memory.", edited_files=[str(f)])
    # Nothing changed — second memory must crash.
    with pytest.raises(RuntimeError, match="No changes detected"):
        ws.create_memory("Empty", "Nothing to save.", edited_files=[str(f)])


def test_create_memory_modified_file(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"v1")
    ws.create_memory("v1", "First.", edited_files=[str(f)])
    f.write_bytes(b"v2 - much longer content here")
    memory = ws.create_memory("v2", "Modified.", edited_files=[str(f)])
    assert memory.changes[0].action == "modified"
    assert memory.changes[0].bytes_added > 0
    assert memory.changes[0].bytes_removed > 0


# ---------------------------------------------------------------------------
# untrack() + GC
# ---------------------------------------------------------------------------

def test_untrack_removes_file_from_tracking(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "a.py", b"content")
    ws.create_memory("add a", "note", edited_files=[str(f)])
    ws.untrack(str(f))
    statuses = ws.get_status()
    assert all(s.path != str(f) for s in statuses)


def test_untrack_unknown_file_crashes(ws: Workspace) -> None:
    with pytest.raises(KeyError):
        ws.untrack("not_tracked.py")


def test_untrack_gc_exclusive_blob(tmp_path: Path, ws: Workspace) -> None:
    """Untracking a file with a unique blob must delete that blob from disk."""
    f = _write(tmp_path / "solo.py", b"unique content abc")
    memory = ws.create_memory("add solo", "note", edited_files=[str(f)])
    blob_hash = memory.changes[0].blob_hash
    blob_path = (ws._root / "blobs" / blob_hash)
    assert blob_path.exists()

    ws.untrack(str(f))
    assert not blob_path.exists(), "Blob must be deleted when refcount reaches 0"


def test_untrack_gc_shared_blob_preserved(tmp_path: Path, ws: Workspace) -> None:
    """Two files with identical content share a blob. Untracking one must NOT delete it."""
    content = b"shared identical content"
    fa = _write(tmp_path / "a.py", content)
    fb = _write(tmp_path / "b.py", content)
    memory = ws.create_memory("add both", "note", edited_files=[str(fa), str(fb)])

    hashes = {m.path: m.blob_hash for m in memory.changes}
    assert hashes[str(fa)] == hashes[str(fb)], "Shared content must yield the same blob hash"
    shared_hash = hashes[str(fa)]

    ws.untrack(str(fa))
    blob_path = ws._root / "blobs" / shared_hash
    assert blob_path.exists(), "Shared blob must survive after untracking one referencing file"


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------

def test_get_status_reports_current_and_history_sizes(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "size.py", b"x" * 100)
    ws.create_memory("v1", "note", edited_files=[str(f)])
    f.write_bytes(b"y" * 200)
    ws.create_memory("v2", "note", edited_files=[str(f)])

    statuses = ws.get_status()
    st = next(s for s in statuses if s.path == str(f))
    assert st.current_size == 200
    assert st.history_size >= 100  # at least the old blob


def test_get_status_none_for_deleted_file(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "gone.py", b"content")
    ws.create_memory("add", "note", edited_files=[str(f)])
    f.unlink()
    statuses = ws.get_status()
    st = next(s for s in statuses if s.path == str(f))
    assert st.current_size is None


# ---------------------------------------------------------------------------
# get_log() & get_memory()
# ---------------------------------------------------------------------------

def test_get_log_returns_memories_in_reverse_order(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "log.py", b"v1")
    m1 = ws.create_memory("m1", "note", edited_files=[str(f)])
    f.write_bytes(b"v2_new_size")
    m2 = ws.create_memory("m2", "note", edited_files=[str(f)])
    f.write_bytes(b"v3_even_newer_size_here")
    m3 = ws.create_memory("m3", "note", edited_files=[str(f)])

    log = ws.get_log()
    assert [m.id for m in log] == [m3.id, m2.id, m1.id]


def test_get_memory_crashes_on_unknown_id(ws: Workspace) -> None:
    with pytest.raises(KeyError):
        ws.get_memory("00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# find_child_memory()
# ---------------------------------------------------------------------------

def test_find_child_memory_returns_correct_child(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "child.py", b"v1")
    m1 = ws.create_memory("v1", "note", edited_files=[str(f)])
    f.write_bytes(b"v2" * 10)
    m2 = ws.create_memory("v2", "note", edited_files=[str(f)])
    f.write_bytes(b"v3" * 20)
    m3 = ws.create_memory("v3", "note", edited_files=[str(f)])

    child1 = ws.find_child_memory(m1.id)
    assert child1 is not None
    assert child1.id == m2.id

    child2 = ws.find_child_memory(m2.id)
    assert child2 is not None
    assert child2.id == m3.id


def test_find_child_memory_head_returns_none(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "head.py", b"v1")
    m = ws.create_memory("v1", "note", edited_files=[str(f)])
    assert ws.find_child_memory(m.id) is None


# ---------------------------------------------------------------------------
# read_file_at_memory()
# ---------------------------------------------------------------------------

def test_read_file_at_memory_returns_correct_content(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "hist.py", b"version 1")
    m1 = ws.create_memory("v1", "note", edited_files=[str(f)])
    f.write_bytes(b"version 2")
    ws.create_memory("v2", "note", edited_files=[str(f)])

    content_at_m1 = ws.read_file_at_memory(str(f), m1.id)
    assert content_at_m1 == b"version 1"


def test_read_file_at_memory_crashes_if_not_found(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "other.py", b"content")
    m = ws.create_memory("add", "note", edited_files=[str(f)])
    with pytest.raises(KeyError):
        ws.read_file_at_memory("nonexistent.py", m.id)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_workspace_persists_and_reloads(tmp_path: Path) -> None:
    """Recreating a Workspace from the same storage_root must restore state."""
    storage = tmp_path / "storage"
    ws1 = Workspace(storage)
    f = _write(tmp_path / "persist.py", b"data")
    m = ws1.create_memory("Initial", "note", edited_files=[str(f)])

    ws2 = Workspace(storage)
    log = ws2.get_log()
    assert len(log) == 1
    assert log[0].id == m.id