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
    memory = engine.create_memory("First memory", "Initialised x to 1.", edited_files=[str(f)])
    assert memory.id
    assert memory.title == "First memory"


def test_create_memory_is_indexed(engine, tmp_path) -> None:
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    memory = engine.create_memory("Indexed memory", "This note should be vectorised.", edited_files=[str(f)])
    assert engine.wait_until_indexed()
    assert engine._indexer.is_indexed(memory.id)


def test_create_memory_updates_graph(engine, tmp_path) -> None:
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    memory = engine.create_memory("Graph memory", "This should appear in the graph.", edited_files=[str(f)])
    assert str(f) in engine._graph.get_memory_files(memory.id)


def test_create_memory_raises_if_no_changes(engine, tmp_path) -> None:
    f = tmp_path / "test.py"
    _write(f, "x = 1")
    engine.create_memory("First", "Initial note.", edited_files=[str(f)])
    with pytest.raises(RuntimeError, match="No changes detected"):
        engine.create_memory("Second", "Should fail — nothing changed.", edited_files=[str(f)])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_finds_relevant_memory(engine, tmp_path) -> None:
    f = tmp_path / "sort.py"
    _write(f, "def quicksort(arr): pass")
    engine.create_memory(
        "Implement quicksort",
        "Implemented an in-place quicksort algorithm for sorting integer arrays.",
        edited_files=[str(f)]
    )
    assert engine.wait_until_indexed()
    results = engine.search("sorting algorithm", top_k=10, top_n=1)
    assert len(results) == 1
    assert "sort" in results[0].title.lower() or results[0].score > 0


# ---------------------------------------------------------------------------
# get_related_files
# ---------------------------------------------------------------------------

def test_get_related_files(engine, tmp_path) -> None:
    fa = tmp_path / "a.py"
    fb = tmp_path / "b.py"
    _write(fa, "a = 1")
    _write(fb, "b = 2")
    engine.create_memory("Co-occur memory", "Both a and b changed.", edited_files=[str(fa), str(fb)])
    related = dict(engine.get_related_files(str(fa)))
    assert str(fb) in related


# ---------------------------------------------------------------------------
# Workspace pass-throughs
# ---------------------------------------------------------------------------

def test_get_log_returns_memories(engine, tmp_path) -> None:
    f = tmp_path / "x.py"
    _write(f, "x = 1")
    engine.create_memory("Log test", "First entry in log.", edited_files=[str(f)])
    log = engine.get_log()
    assert len(log) == 1
    assert log[0].title == "Log test"