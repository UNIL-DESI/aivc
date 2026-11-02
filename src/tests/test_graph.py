"""Unit tests for aivc.semantic.graph.CooccurrenceGraph (SQLite backend)."""

from __future__ import annotations

import pytest
from pathlib import Path

from aivc.core.memory import Memory, FileChange


def _make_memory(
    memory_id: str,
    title: str = "Memory",
    file_paths: list[str] | None = None,
) -> Memory:
    if file_paths is None:
        file_paths = []
    changes = [
        FileChange(
            path=fp, action="modified", blob_hash="deadbeef",
            bytes_added=10, bytes_removed=5,
        )
        for fp in file_paths
    ]
    return Memory(
        id=memory_id,
        timestamp="2026-01-01T00:00:00+00:00",
        title=title,
        note="note",
        parent_id=None,
        changes=changes,
    )


@pytest.fixture
def graph(tmp_path: Path):
    from aivc.semantic.graph import CooccurrenceGraph
    return CooccurrenceGraph(tmp_path / "storage")


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------

def test_add_memory_registers_memory_node(graph) -> None:
    m = _make_memory("m1", file_paths=["a.py", "b.py"])
    graph.add_memory(m)
    # Verify via public API
    assert graph.get_memory_files("m1") == ["a.py", "b.py"] or \
           set(graph.get_memory_files("m1")) == {"a.py", "b.py"}


def test_add_memory_registers_file_nodes(graph) -> None:
    m = _make_memory("m1", file_paths=["a.py", "b.py"])
    graph.add_memory(m)
    assert "m1" in graph.get_file_memories("a.py")
    assert "m1" in graph.get_file_memories("b.py")


def test_add_memory_is_idempotent(graph) -> None:
    m = _make_memory("m1", file_paths=["a.py"])
    graph.add_memory(m)
    graph.add_memory(m)
    # File must still list memory only once.
    assert graph.get_file_memories("a.py").count("m1") == 1


def test_add_memory_persists_to_disk(tmp_path) -> None:
    from aivc.semantic.graph import CooccurrenceGraph
    storage = tmp_path / "storage"
    g1 = CooccurrenceGraph(storage)
    m = _make_memory("m1", file_paths=["a.py"])
    g1.add_memory(m)

    # Load a fresh instance from the same root — data must persist.
    g2 = CooccurrenceGraph(storage)
    assert set(g2.get_memory_files("m1")) == {"a.py"}


# ---------------------------------------------------------------------------
# remove_memory
# ---------------------------------------------------------------------------

def test_remove_memory_deletes_memory_node(graph) -> None:
    m = _make_memory("m1", file_paths=["a.py"])
    graph.add_memory(m)
    graph.remove_memory("m1")
    with pytest.raises(KeyError):
        graph.get_memory_files("m1")


def test_remove_memory_cleans_up_orphan_file_node(graph) -> None:
    m = _make_memory("m1", file_paths=["solo.py"])
    graph.add_memory(m)
    graph.remove_memory("m1")
    # "solo.py" has no remaining memories → must be removed.
    with pytest.raises(KeyError):
        graph.get_file_memories("solo.py")


def test_remove_memory_keeps_shared_file_node(graph) -> None:
    m1 = _make_memory("m1", file_paths=["shared.py", "only_m1.py"])
    m2 = _make_memory("m2", file_paths=["shared.py"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    graph.remove_memory("m1")
    # "shared.py" is still referenced by m2.
    assert "m2" in graph.get_file_memories("shared.py")


def test_remove_memory_raises_if_not_in_graph(graph) -> None:
    with pytest.raises(KeyError, match="not in the graph"):
        graph.remove_memory("nonexistent")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_get_memory_files(graph) -> None:
    m = _make_memory("m1", file_paths=["a.py", "b.py"])
    graph.add_memory(m)
    assert set(graph.get_memory_files("m1")) == {"a.py", "b.py"}


def test_get_file_memories(graph) -> None:
    m1 = _make_memory("m1", file_paths=["a.py"])
    m2 = _make_memory("m2", file_paths=["a.py"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    assert set(graph.get_file_memories("a.py")) == {"m1", "m2"}


def test_get_related_files_basic(graph) -> None:
    m1 = _make_memory("m1", file_paths=["a.py", "b.py", "c.py"])
    m2 = _make_memory("m2", file_paths=["a.py", "b.py"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    related = dict(graph.get_related_files("a.py"))
    # "b.py" appears in both memories alongside "a.py".
    assert related["b.py"] == 2
    # "c.py" appears only in m1.
    assert related.get("c.py", 0) == 1
    # "a.py" itself must not be in results.
    assert "a.py" not in related


def test_get_related_files_raises_for_unknown(graph) -> None:
    with pytest.raises(KeyError):
        graph.get_related_files("nonexistent.py")


# ---------------------------------------------------------------------------
# to_vis_data
# ---------------------------------------------------------------------------

def test_to_vis_data_structure(graph) -> None:
    m = _make_memory("m1", title="My memory", file_paths=["a.py"])
    graph.add_memory(m)
    data = graph.to_vis_data()
    assert "nodes" in data
    assert "edges" in data

    node_ids = {n["id"] for n in data["nodes"]}
    assert "m1" in node_ids
    assert "a.py" in node_ids

    edge_sources = {e["source"] for e in data["edges"]}
    assert "m1" in edge_sources

# ---------------------------------------------------------------------------
# get_memories_by_glob
# ---------------------------------------------------------------------------

def test_get_memories_by_glob_basic(graph) -> None:
    m1 = _make_memory("m1", file_paths=["/abs/path/src/a.py", "/abs/path/src/b.py"])
    m2 = _make_memory("m2", file_paths=["/abs/path/docs/readme.md"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    memories = graph.get_memories_by_glob("*.py")
    assert set(memories) == {"m1"}

def test_get_memories_by_glob_specific_dir(graph) -> None:
    m1 = _make_memory("m1", file_paths=["/usr/src/semantic/a.py"])
    m2 = _make_memory("m2", file_paths=["/usr/src/core/b.py"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    memories = graph.get_memories_by_glob("*/semantic/*.py")
    assert set(memories) == {"m1"}

def test_get_memories_by_glob_no_match(graph) -> None:
    m1 = _make_memory("m1", file_paths=["/src/a.py"])
    graph.add_memory(m1)
    assert graph.get_memories_by_glob("*.md") == []

def test_get_memories_by_glob_all_match(graph) -> None:
    m1 = _make_memory("m1", file_paths=["/src/a.py"])
    m2 = _make_memory("m2", file_paths=["/docs/b.md"])
    graph.add_memory(m1)
    graph.add_memory(m2)
    assert set(graph.get_memories_by_glob("*")) == {"m1", "m2"}


# ---------------------------------------------------------------------------
# Thread safety & SQLite concurrency (non-regression for SQLITE_MISUSE)
# ---------------------------------------------------------------------------

def test_concurrent_add_memory_thread_safety(graph) -> None:
    import threading

    errors = []
    memories = [
        _make_memory(f"mem_{i}", file_paths=[f"file_{i % 5}.py", f"shared_{i % 2}.txt"])
        for i in range(50)
    ]

    def worker(items):
        for m in items:
            try:
                graph.add_memory(m)
                _ = graph.get_indexed_commit_ids()
            except Exception as e:
                errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(memories[0:25],)),
        threading.Thread(target=worker, args=(memories[25:50],)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent operations failed with: {errors}"
    assert len(graph.get_indexed_commit_ids()) == 50
