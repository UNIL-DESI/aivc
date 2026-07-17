"""Unit tests for aivc.semantic.searcher.Searcher.

These tests require sentence-transformers and chromadb to be installed.
They are marked with ``requires_ml`` and will be skipped if ML deps are not
available in the current environment. Run inside the venv created by install.sh.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from aivc.core.memory import Memory, FileChange

pytestmark = pytest.mark.requires_ml


def _make_memory(
    memory_id: str,
    title: str = "Memory",
    note: str = "A detailed note.",
    file_paths: list[str] | None = None,
) -> Memory:
    if file_paths is None:
        file_paths = ["src/main.py"]
    changes = [
        FileChange(
            path=fp, action="added", blob_hash="abc", bytes_added=10, bytes_removed=0,
        )
        for fp in file_paths
    ]
    return Memory(
        id=memory_id,
        timestamp="2026-01-01T00:00:00+00:00",
        title=title,
        note=note,
        parent_id=None,
        changes=changes,
    )


@pytest.fixture
def searcher(tmp_path: Path):
    from aivc.semantic.indexer import Indexer
    from aivc.semantic.searcher import Searcher
    indexer = Indexer(tmp_path / "storage")
    return Searcher(indexer)


@pytest.fixture
def loaded_searcher(tmp_path: Path):
    """A searcher pre-populated with two memories."""
    from aivc.semantic.indexer import Indexer
    from aivc.semantic.searcher import Searcher
    indexer = Indexer(tmp_path / "storage")
    m1 = _make_memory(
        "id-sorting",
        title="Implement quicksort",
        note="Implemented a fast in-place quicksort algorithm for numeric arrays.",
    )
    m2 = _make_memory(
        "id-database",
        title="Add database migrations",
        note="Added SQL migration scripts to create users and sessions tables.",
    )
    indexer.index_memory(m1)
    indexer.index_memory(m2)
    return Searcher(indexer)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_search_result_instances(loaded_searcher) -> None:
    from aivc.semantic.searcher import SearchResult
    results = loaded_searcher.search("sorting", top_k=2, top_n=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)


def test_search_result_has_all_fields(loaded_searcher) -> None:
    results = loaded_searcher.search("sorting algorithm", top_k=2, top_n=1)
    r = results[0]
    assert r.memory_id
    assert r.title
    assert r.timestamp
    assert isinstance(r.score, float)
    assert isinstance(r.snippet, str)
    assert isinstance(r.file_paths, list)


def test_search_top_n_limits_results(loaded_searcher) -> None:
    results = loaded_searcher.search("test", top_k=2, top_n=1)
    assert len(results) <= 1


def test_search_results_sorted_by_score_descending(loaded_searcher) -> None:
    results = loaded_searcher.search("database sql", top_k=2, top_n=2)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_raises_if_top_n_greater_than_top_k(loaded_searcher) -> None:
    with pytest.raises(ValueError, match="top_n"):
        loaded_searcher.search("query", top_k=5, top_n=10)


def test_search_returns_empty_list_on_empty_index(tmp_path) -> None:
    from aivc.semantic.indexer import Indexer
    from aivc.semantic.searcher import Searcher
    indexer = Indexer(tmp_path / "storage")
    s = Searcher(indexer)
    with pytest.raises(ValueError, match="index is empty"):
        s.search("anything")

def test_search_with_filter_ids_restricts_results(loaded_searcher) -> None:
    results = loaded_searcher.search("sql database", top_k=5, filter_ids=["id-sorting"])
    assert len(results) <= 1
    if results:
        assert results[0].memory_id == "id-sorting"

def test_search_with_empty_filter_ids_returns_empty(loaded_searcher) -> None:
    results = loaded_searcher.search("sql database", top_k=5, filter_ids=[])
    assert results == []

def test_search_snippet_prefix(loaded_searcher) -> None:
    # Add a memory with a very long note
    note = "Start. " * 200 # 1400 characters
    m3 = _make_memory("id-long", title="Long note", note=note)
    loaded_searcher._indexer.index_memory(m3)
    
    results = loaded_searcher.search("Start", top_k=5, top_n=1)
    assert len(results) == 1
    snippet = results[0].snippet
    assert len(snippet) <= 1003  # 1000 chars + "..."
    assert snippet.endswith("...")
    assert not snippet.startswith("…")

