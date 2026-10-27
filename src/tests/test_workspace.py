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
    assert memory.changes[0].action == "edited"
    assert memory.parent_id is None


def test_create_memory_second_links_to_first(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"v1")
    m1 = ws.create_memory("v1", "First.", edited_files=[str(f)])
    f.write_bytes(b"v2_changed_size")
    m2 = ws.create_memory("v2", "Second.", edited_files=[str(f)])
    assert m2.parent_id == m1.id


def test_create_memory_empty_crashes(ws: Workspace) -> None:
    with pytest.raises(RuntimeError, match="Nothing to remember"):
        ws.create_memory("Empty", "", edited_files=[], read_files=[])


def test_create_memory_modified_file(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "app.py", b"v1")
    ws.create_memory("v1", "First.", edited_files=[str(f)])
    f.write_bytes(b"v2 - much longer content here")
    memory = ws.create_memory("v2", "Modified.", edited_files=[str(f)])
    assert memory.changes[0].action == "edited"


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------

def test_get_status_reports_current_and_history_sizes(tmp_path: Path, ws: Workspace) -> None:
    from aivc.config import to_portable_path
    f = _write(tmp_path / "size.py", b"x" * 100)
    ws.create_memory("v1", "note", edited_files=[str(f)])
    f.write_bytes(b"y" * 200)
    ws.create_memory("v2", "note", edited_files=[str(f)])

    statuses = ws.get_status()
    st = next(s for s in statuses if to_portable_path(s.path) == to_portable_path(f))
    assert st.current_size == 200


def test_get_status_none_for_deleted_file(tmp_path: Path, ws: Workspace) -> None:
    from aivc.config import to_portable_path
    f = _write(tmp_path / "gone.py", b"content")
    ws.create_memory("add", "note", edited_files=[str(f)])
    f.unlink()
    statuses = ws.get_status()
    st = next(s for s in statuses if to_portable_path(s.path) == to_portable_path(f))
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


# ---------------------------------------------------------------------------
# Broken chain and missing HEAD recovery (Chantier 3)
# ---------------------------------------------------------------------------

def test_get_log_with_broken_chain(tmp_path: Path, ws: Workspace) -> None:
    # 1. Create a chain of memories: m1 -> m2 -> m3
    f = _write(tmp_path / "app.py", b"v1")
    m1 = ws.create_memory("m1", "First commit", edited_files=[str(f)])
    
    f.write_bytes(b"v2")
    m2 = ws.create_memory("m2", "Second commit", edited_files=[str(f)])
    
    f.write_bytes(b"v3")
    m3 = ws.create_memory("m3", "Third commit", edited_files=[str(f)])

    # Verify the complete chain
    log = ws.get_log()
    assert [m.id for m in log] == [m3.id, m2.id, m1.id]

    # 2. Physically remove the intermediate commit file (m2.json)
    m2_file = ws._commits_dir / f"{m2.id}.json"
    assert m2_file.exists()
    m2_file.unlink()

    # 3. get_log() should return all available commit files (m3 and m1) without KeyError when intermediate m2 is missing
    broken_log = ws.get_log()
    assert [m.id for m in broken_log] == [m3.id, m1.id]


def test_workspace_init_with_missing_head(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    ws = Workspace(storage)
    
    # 1. Create a clean memory to have a head_commit_id
    f = _write(tmp_path / "app.py", b"v1")
    m1 = ws.create_memory("m1", "First commit", edited_files=[str(f)])
    
    assert ws._state["head_commit_id"] == m1.id
    
    # 2. Physically delete the commit file corresponding to head_commit_id
    m1_file = ws._commits_dir / f"{m1.id}.json"
    assert m1_file.exists()
    m1_file.unlink()
    
    # 3. Re-instantiate the Workspace. It should auto-repair and set head_commit_id to None
    ws_reloaded = Workspace(storage)
    assert ws_reloaded._state["head_commit_id"] is None


def test_create_memory_with_urls_only(ws: Workspace) -> None:
    m = ws.create_memory("URL Note", "Just saving links.", urls=["  https://example.com  ", "https://arxiv.org  "])
    assert m.urls == ["https://example.com", "https://arxiv.org"]
    assert m.changes == []


def test_create_memory_sanitizes_urls(ws: Workspace) -> None:
    m = ws.create_memory("URL Note", "Clean links.", urls=["", "   ", "https://valid.com", None])  # type: ignore
    assert m.urls == ["https://valid.com"]


def test_create_memory_read_files(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "doc.txt", b"consulted content")
    m = ws.create_memory("Consult doc", "Read doc.", read_files=[str(f)])
    assert len(m.changes) == 1
    c = m.changes[0]
    assert c.action == "consulted"
    assert c.blob_hash is None


def test_remember_with_roadmap_items(tmp_path: Path, ws: Workspace) -> None:
    f = _write(tmp_path / "code.py", b"hello")
    m = ws.remember(
        "Memory with roadmap",
        journal="Did something",
        edited_files=[str(f)],
        user_items=[
            {"task": "Deliverable 1", "details": "Done deal", "kind": "deliverable"},
            {"task": "Roadmap item in user_items", "kind": "roadmap"},
        ],
        roadmap_items=[
            {"task": "Roadmap item from parameter", "details": "detail roadmap"},
        ],
    )
    assert len(m.user_items) == 3
    assert m.user_items[0]["kind"] != "roadmap"
    assert m.user_items[1]["kind"] == "roadmap"
    assert m.user_items[1]["id"].startswith("r-")
    assert m.user_items[2]["kind"] == "roadmap"
    assert m.user_items[2]["id"].startswith("r-")

    # Verify complete_user_items can complete roadmap items
    r_id = m.user_items[2]["id"]
    completed = ws.complete_user_items([r_id])
    assert completed == [r_id]
    reloaded = ws._load_memory(m.id)
    assert reloaded.user_items[2]["done"] is True

