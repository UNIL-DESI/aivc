"""
Tests for the AIVC MCP server tools.

The SemanticEngine is always mocked — no ML dependencies are loaded.
These tests validate the formatting and orchestration logic of each tool.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Bootstrap: inject a fake SemanticEngine before importing server.py so that
# the module-level initialisation does NOT touch the filesystem or ML stack.
# ---------------------------------------------------------------------------

_FAKE_STORAGE = "/tmp/aivc_test_storage"

# Must be set before importing the server module.
os.environ["AIVC_STORAGE_ROOT"] = _FAKE_STORAGE

# Build a mock SemanticEngine class and patch at import time.
_mock_engine = MagicMock()
_mock_engine.get_index_queue_size.return_value = 0


def _import_server():
    """Import aivc.server with SemanticEngine fully mocked."""
    import aivc.server as srv
    if hasattr(srv, "_get_engine"):
        srv._get_engine = lambda: _mock_engine
        
    return srv


# We import once for all tests.
_server = _import_server()

# Grab the 6 active tool functions directly from the server module.
_remember = _server.remember
_recall = _server.recall
_consult_memory = _server.consult_memory
_get_file_history = _server.get_file_history
_complete_user_items = _server.complete_user_items
_restore_file = _server.restore_file


# ---------------------------------------------------------------------------
# Helpers to build fake domain objects
# ---------------------------------------------------------------------------

def _make_memory(
    memory_id="abc-123",
    title="Do something",
    journal="Detailed journal.",
    note="",
    timestamp="2026-03-18T12:00:00+00:00",
    parent_id=None,
    changes=None,
    user_items=None,
):
    """Build a minimal mock Memory."""
    from aivc.core.memory import Memory, FileChange, _compose_note
    fc = FileChange(
        path="src/foo.py",
        action="modified",
    )
    actual_user_items = user_items if user_items is not None else []
    full_note = note or _compose_note(journal, actual_user_items, fallback_note="Detailed note.")
    return Memory(
        id=memory_id,
        title=title,
        note=full_note,
        parent_id=parent_id,
        timestamp=timestamp,
        changes=changes if changes is not None else [fc],
        user_items=actual_user_items,
        journal=journal,
    )


def _make_search_result(memory_id="abc-123", title="Old memory", score=0.9):
    """Build a minimal mock SearchResult."""
    from aivc.semantic.searcher import SearchResult
    return SearchResult(
        memory_id=memory_id,
        title=title,
        timestamp="2026-03-18T10:00:00+00:00",
        score=score,
        snippet="short snippet",
        file_paths=["src/foo.py", "src/bar.py"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemember(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    async def test_returns_memory_id_and_files(self):
        _mock_engine.remember.return_value = _make_memory()
        result = await _remember("Do something", "Detailed journal.")
        self.assertIn("Memory successfully created", result)
        self.assertIn("abc-123", result)

    async def test_delegates_to_engine(self):
        _mock_engine.remember.return_value = _make_memory()
        await _remember("T", "J")
        _mock_engine.remember.assert_called_once_with(
            title="T",
            journal="J",
            user_items=[],
            read_files=[],
            edited_files=[],
            urls=[],
        )

    async def test_delegates_to_engine_with_consulted(self):
        _mock_engine.remember.return_value = _make_memory()
        await _remember("T", "J", read_files=["f1.py"])
        _mock_engine.remember.assert_called_once_with(
            title="T",
            journal="J",
            user_items=[],
            read_files=["f1.py"],
            edited_files=[],
            urls=[],
        )

    async def test_delegates_to_engine_with_user_items(self):
        _mock_engine.remember.return_value = _make_memory(
            user_items=[{"id": "item-1", "task": "Check token", "done": False}]
        )
        result = await _remember(
            "T",
            "J",
            user_items=[{"id": "item-1", "task": "Check token", "done": False}],
        )
        self.assertIn("Actions Utilisateur", result)
        self.assertIn("[item-1] Check token", result)

    async def test_delegates_to_engine_with_tasks(self):
        _mock_engine.remember.return_value = _make_memory(
            user_items=[{"id": "t-1", "task": "Task test", "done": False, "task_status": "new"}]
        )
        result = await _remember(
            "T",
            "J",
            tasks=[{"id": "t-1", "task": "Task test", "task_status": "new"}],
        )
        self.assertIn("Actions Utilisateur", result)
        self.assertIn("[t-1] Task test", result)

    async def test_remember_validation_errors(self):
        _mock_engine.remember.side_effect = ValueError("Validation error")
        with self.assertRaises(ValueError):
            await _remember("T", "J", read_files=["non_existent.txt"])
        _mock_engine.remember.side_effect = None

    async def test_runtime_error_propagates(self):
        _mock_engine.remember.side_effect = RuntimeError("Something failed")
        with self.assertRaises(RuntimeError):
            await _remember("T", "J")
        _mock_engine.remember.side_effect = None

    async def test_empty_changes_handled(self):
        _mock_engine.remember.return_value = _make_memory(changes=[])
        result = await _remember("T", "J")
        self.assertIn("Memory successfully created", result)


class TestRecall(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)
        _mock_engine.get_index_queue_size.return_value = 0

    async def test_returns_memory_list_without_note(self):
        _mock_engine.search.return_value = [_make_search_result()]
        result = await _recall("find something")
        self.assertIn("abc-123", result)
        self.assertIn("Old memory", result)
        self.assertIn("> short snippet", result)
        self.assertNotIn("Detailed journal", result)

    async def test_includes_aggregated_files(self):
        results = [
            _make_search_result("m1", "M1"),
            _make_search_result("m2", "M2"),
        ]
        _mock_engine.search.return_value = results
        result = await _recall("find")
        self.assertIn("foo.py", result)

    async def test_no_results_returns_graceful_message(self):
        _mock_engine.search.return_value = []
        result = await _recall("nothing")
        self.assertIn("Aucun souvenir trouvé", result)

    async def test_top_n_capped_at_20(self):
        _mock_engine.search.return_value = []
        await _recall("x", top_n=100)
        _mock_engine.search.assert_called_once_with("x", top_n=20, filter_glob="")


class TestConsultMemory(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)
        _mock_engine.find_child_memory.return_value = None

    def test_returns_full_journal(self):
        _mock_engine.get_memory.return_value = _make_memory(journal="# My Work\n\nDetails here.")
        result = _consult_memory("abc-123")
        self.assertIn("# My Work", result)
        self.assertIn("Details here", result)

    def test_renders_user_items(self):
        _mock_engine.get_memory.return_value = _make_memory(
            user_items=[
                {"id": "task-1", "task": "Do thing", "details": "Extra info", "done": False},
                {"id": "task-2", "task": "Done thing", "details": "", "done": True},
            ]
        )
        result = _consult_memory("abc-123")
        self.assertIn("Actions Utilisateur", result)
        self.assertIn("- [ ] `[task-1]` Do thing", result)
        self.assertIn("*Détails :* Extra info", result)
        self.assertIn("- [x] `[task-2]` Done thing", result)

    def test_renders_file_changes(self):
        _mock_engine.get_memory.return_value = _make_memory()
        result = _consult_memory("abc-123")
        self.assertIn("foo.py", result)

    def test_key_error_propagates(self):
        _mock_engine.get_memory.side_effect = KeyError("Memory not found")
        with self.assertRaises(KeyError):
            _consult_memory("bad-id")

    def test_shows_parent_context(self):
        parent = _make_memory(memory_id="p-123", title="Parent memory")
        child = _make_memory(memory_id="c-456", title="Child memory", parent_id="p-123")

        def side_effect(mid):
            if mid == "p-123":
                return parent
            if mid == "c-456":
                return child
            raise KeyError(mid)

        _mock_engine.get_memory.side_effect = side_effect

        result = _consult_memory("c-456")
        self.assertIn("⬆️ **Prev** : Parent memory (ID: p-123)", result)

    def test_shows_child_context(self):
        memory = _make_memory(memory_id="c-123", title="My memory")
        child = _make_memory(memory_id="next-456", title="Next memory")

        _mock_engine.get_memory.return_value = memory
        _mock_engine.find_child_memory.return_value = child

        result = _consult_memory("c-123")
        self.assertIn("⬇️ **Next** : Next memory (ID: next-456)", result)


class TestGetFileHistory(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_returns_history_with_open_items(self):
        mem = _make_memory(
            memory_id="mem-1",
            title="Refactor core",
            user_items=[{"id": "item-1", "task": "Check prod", "done": False}],
        )
        _mock_engine._workspace.get_file_history.return_value = [mem]

        result = _get_file_history("src/foo.py")
        self.assertIn("AIVC History for: `src/foo.py`", result)
        self.assertIn("Refactor core", result)
        self.assertIn("`[item-1]` Check prod", result)

    def test_empty_history_returns_cold_start(self):
        _mock_engine._workspace.get_file_history.return_value = []
        result = _get_file_history("unknown.txt")
        self.assertIn("Aucun souvenir trouvé", result)


class TestCompleteUserItems(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_completes_items_success(self):
        _mock_engine.complete_user_items.return_value = ["item-1", "item-2"]
        result = _complete_user_items(["item-1", "item-2"])
        self.assertIn("✅ Tâche(s) marquée(s) comme terminée(s)", result)
        self.assertIn("item-1, item-2", result)

    def test_empty_input(self):
        result = _complete_user_items([])
        self.assertIn("Aucun identifiant de tâche spécifié.", result)

    def test_no_items_modified(self):
        _mock_engine.complete_user_items.return_value = []
        result = _complete_user_items(["ghost-id"])
        self.assertIn("Aucune tâche trouvée", result)


class TestRestoreFile(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_restore_file_success(self):
        _mock_engine.pull_latest_file.return_value = "c:/dest/foo.py"
        result = _restore_file("foo.py")
        self.assertIn("✅ Fichier restauré avec succès", result)
        self.assertIn("c:/dest/foo.py", result)

    def test_restore_file_error(self):
        _mock_engine.pull_latest_file.side_effect = RuntimeError("Drive network error")
        result = _restore_file("foo.py")
        self.assertIn("❌ Échec de la restauration", result)
        self.assertIn("Drive network error", result)


if __name__ == "__main__":
    unittest.main()
