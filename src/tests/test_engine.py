"""Integration tests for aivc.semantic.engine.SemanticEngine.

These tests require sentence-transformers and chromadb to be installed.
They are marked with ``requires_ml`` and will be skipped if ML deps are not
available in the current environment. Run inside the venv created by install.sh.
"""

from __future__ import annotations

import pytest
from pathlib import Path

pytestmark = pytest.mark.requires_ml


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def engine(tmp_path: Path):
    from aivc.semantic.engine import SemanticEngine
    e = SemanticEngine(tmp_path / "storage")
    yield e
    e.shutdown()


# ---------------------------------------------------------------------------
# create_memory → indexes + updates graph
# ---------------------------------------------------------------------------

def test_create_memory_returns_memory(engine, tmp_path) -> None:
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    memory = engine.create_memory("First memory", "Initialised x to 1.", edited_files=[str(f)], main_project_file=str(f))
    assert memory.id
    assert memory.title == "First memory"


def test_create_memory_is_indexed(engine, tmp_path) -> None:
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    memory = engine.create_memory("Indexed memory", "This note should be vectorised.", edited_files=[str(f)], main_project_file=str(f))
    assert engine.wait_until_indexed()
    assert engine._indexer.is_indexed(memory.id)


def test_create_memory_updates_graph(engine, tmp_path) -> None:
    from aivc.config import to_portable_path
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    memory = engine.create_memory("Graph memory", "This should appear in the graph.", edited_files=[str(f)], main_project_file=str(f))
    assert to_portable_path(f) in engine._graph.get_memory_files(memory.id)


def test_create_memory_raises_if_no_main_project_file(engine) -> None:
    with pytest.raises(ValueError, match="main_project_file is required and cannot be empty"):
        engine.create_memory("Empty", "", edited_files=[], read_files=[], main_project_file="")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_relevant_memory(engine, tmp_path) -> None:
    f = tmp_path / "sort.py"
    _write(f, "def quicksort(arr): pass")
    engine.create_memory(
        "Implement quicksort",
        "Implemented an in-place quicksort algorithm for sorting integer arrays.",
        edited_files=[str(f)],
        main_project_file=str(f),
    )
    assert engine.wait_until_indexed()
    results = engine.search("sorting algorithm", top_k=10, top_n=1)
    assert len(results) == 1
    assert "sort" in results[0].title.lower() or results[0].score > 0


# ---------------------------------------------------------------------------
# get_related_files
# ---------------------------------------------------------------------------

def test_get_related_files(engine, tmp_path) -> None:
    from aivc.config import to_portable_path
    fa = tmp_path / "a.py"
    fb = tmp_path / "b.py"
    _write(fa, "a = 1")
    _write(fb, "b = 2")
    engine.create_memory("Co-occur memory", "Both a and b changed.", edited_files=[str(fa), str(fb)], main_project_file=str(fa))
    related = dict(engine.get_related_files(str(fa)))
    assert to_portable_path(fb) in related


# ---------------------------------------------------------------------------
# Workspace pass-throughs
# ---------------------------------------------------------------------------

def test_get_log_returns_memories(engine, tmp_path) -> None:
    f = tmp_path / "x.py"
    _write(f, "x = 1")
    engine.create_memory("Log test", "First entry in log.", edited_files=[str(f)], main_project_file=str(f))
    log = engine.get_log()
    assert len(log) == 1
    assert log[0].title == "Log test"


# ---------------------------------------------------------------------------
# main_project_file validation & behavior
# ---------------------------------------------------------------------------

def test_main_project_file_behavior(engine, tmp_path: Path) -> None:
    from aivc.config import to_portable_path

    # (a) qu'un main_project_file inexistant sur le disque lève ValueError
    non_existent = tmp_path / "does_not_exist.md"
    with pytest.raises(ValueError, match="does not exist on disk"):
        engine.remember(
            title="Invalid project file",
            journal="Test journal",
            main_project_file=str(non_existent),
        )

    # (b) qu'un main_project_file absent de read_files et edited_files est automatiquement ajouté comme consulted dans changes
    project_file = tmp_path / "Project Master.md"
    _write(project_file, "# Project Master")
    other_file = tmp_path / "code.py"
    _write(other_file, "print('hello')")

    mem = engine.remember(
        title="Valid memory with auto-added project file",
        journal="Journal entry",
        edited_files=[str(other_file)],
        main_project_file=str(project_file),
    )

    portable_proj = to_portable_path(project_file)
    change_paths = {c.path: c.action for c in mem.changes}
    assert portable_proj in change_paths
    assert change_paths[portable_proj] == "consulted"

    # (c) que memory.to_dict()["main_project_file"] contient bien le chemin portable normalisé
    mem_dict = mem.to_dict()
    assert "main_project_file" in mem_dict
    assert mem_dict["main_project_file"] == portable_proj


def test_main_project_file_resolves_voicenotes_relative_path(engine, monkeypatch, tmp_path: Path) -> None:
    from aivc.config import to_portable_path
    vn_dir = tmp_path / "Documents" / "VoiceNotes"
    notes_dir = vn_dir / "notes"
    notes_dir.mkdir(parents=True)
    proj = notes_dir / "Test Project.md"
    _write(proj, "# Test Project")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    mem = engine.remember(
        title="Test VoiceNotes relative path",
        journal="Journal",
        main_project_file="notes/Test Project.md",
    )

    expected_portable = to_portable_path(proj)
    assert mem.main_project_file == expected_portable
    assert any(c.path == expected_portable for c in mem.changes)


def test_remember_auto_deduplicates_task_ids_by_text(engine, monkeypatch, tmp_path: Path) -> None:
    """Verify task with identical normalized text reuses existing id for the same project."""
    vn_dir = tmp_path / "Documents" / "VoiceNotes"
    notes_dir = vn_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    proj = notes_dir / "Project Delta.md"
    _write(proj, "# Project Delta")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    mpf = "~/Documents/VoiceNotes/notes/Project Delta.md"

    # 1. First commit defines a roadmap task with generated id
    m1 = engine.remember(
        title="Commit 1",
        journal="Journal 1",
        main_project_file=mpf,
        tasks=[{"task": "Run cluster benchmark on GPU", "task_status": "new"}],
    )
    assigned_id = m1.user_items[0]["id"]
    assert assigned_id.startswith("r-")

    # 2. Second commit provides same task text with no id and status await_validation
    m2 = engine.remember(
        title="Commit 2",
        journal="Journal 2",
        main_project_file=mpf,
        tasks=[{"task": "  run cluster   benchmark on GPU  ", "task_status": "await_validation"}],
    )
    assert len(m2.user_items) == 1
    # Automatically reuses assigned_id
    assert m2.user_items[0]["id"] == assigned_id
    assert m2.user_items[0]["task_status"] == "await_validation"