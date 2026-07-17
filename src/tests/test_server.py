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
        # We don't overwrite the function permanently if not needed, but we can patch it
        srv._get_engine = lambda: _mock_engine
        
    return srv


# We import once for all tests.
_server = _import_server()

# Grab the tool functions directly from the server module.
_remember = _server.remember
_recall = _server.recall
_consult_memory = _server.consult_memory
_get_recent_memories = _server.get_recent_memories
_get_file_history_metadata = _server.get_file_history_metadata
_read_past_file_content = _server.read_past_file_content
_get_status = _server.get_status




# ---------------------------------------------------------------------------
# Helpers to build fake domain objects
# ---------------------------------------------------------------------------

def _make_memory(
    memory_id="abc-123",
    title="Do something",
    note="Detailed note.",
    timestamp="2026-03-18T12:00:00+00:00",
    parent_id=None,
    changes=None,
):
    """Build a minimal mock Memory."""
    from aivc.core.memory import Memory, FileChange
    fc = FileChange(
        path="src/foo.py",
        action="modified",
        blob_hash="aaa",
        bytes_added=100,
        bytes_removed=50,
    )
    return Memory(
        id=memory_id,
        title=title,
        note=note,
        timestamp=timestamp,
        parent_id=parent_id,
        changes=changes if changes is not None else [fc],
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


def _make_file_status(path="src/foo.py", current_size=1024, history_size=4096):
    """Build a minimal mock FileStatus."""
    from aivc.core.workspace import FileStatus
    return FileStatus(path=path, current_size=current_size, history_size=history_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemember(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    async def test_returns_memory_id_and_files(self):
        _mock_engine.create_memory.return_value = _make_memory()
        result = await _remember("Do something", "Detailed note.")
        self.assertIn("Memory successfully created", result)

    async def test_delegates_to_engine(self):
        _mock_engine.create_memory.return_value = _make_memory()
        _mock_engine.get_tracked_paths.return_value = []
        await _remember("T", "N")
        _mock_engine.create_memory.assert_called_once_with("T", "N", read_files=[], edited_files=[])

    async def test_delegates_to_engine_with_consulted(self):
        from pathlib import Path
        _mock_engine.create_memory.return_value = _make_memory()
        _mock_engine.get_tracked_paths.return_value = [str(Path("f1.py").resolve())]
        await _remember("T", "N", read_files=["f1.py"])
        _mock_engine.create_memory.assert_called_once_with("T", "N", read_files=["f1.py"], edited_files=[])

    async def test_remember_validation_errors(self):
        _mock_engine.create_memory.side_effect = ValueError("Validation error")
        with self.assertRaises(ValueError):
            await _remember("T", "N", read_files=["non_existent.txt"])
        _mock_engine.create_memory.side_effect = None

    async def test_runtime_error_propagates(self):
        _mock_engine.create_memory.side_effect = RuntimeError("No changes detected")
        with self.assertRaises(RuntimeError):
            await _remember("T", "N")
        _mock_engine.create_memory.side_effect = None

    async def test_empty_changes_handled(self):
        _mock_engine.create_memory.return_value = _make_memory(changes=[])
        result = await _remember("T", "N")
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
        # Must NOT contain the full note content
        self.assertNotIn("Detailed note", result)

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
        self.assertIn("No matching memories", result)

    async def test_top_n_capped_at_20(self):
        _mock_engine.search.return_value = []
        await _recall("x", top_n=100)
        # top_n is capped to 20 before passing to engine
        _mock_engine.search.assert_called_once_with("x", top_n=20, filter_glob="")


class TestConsultMemory(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)
        # Default: no child found
        _mock_engine.find_child_memory.return_value = None

    def test_returns_full_note(self):
        _mock_engine.get_memory.return_value = _make_memory(note="# My Work\n\nDetails here.")
        result = _consult_memory("abc-123")
        self.assertIn("# My Work", result)
        self.assertIn("Details here", result)

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
            if mid == "p-123": return parent
            if mid == "c-456": return child
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

    def test_no_parent_no_child(self):
        _mock_engine.get_memory.return_value = _make_memory(parent_id=None)
        _mock_engine.find_child_memory.return_value = None
        
        result = _consult_memory("initial-id")
        self.assertNotIn("⬆️ **Prev**", result)
        self.assertNotIn("⬇️ **Next**", result)

    def test_both_parent_and_child(self):
        parent = _make_memory(memory_id="p-1", title="P")
        current = _make_memory(memory_id="curr", title="C", parent_id="p-1")
        child = _make_memory(memory_id="next", title="N")

        def side_effect(mid):
            if mid == "p-1": return parent
            if mid == "curr": return current
            raise KeyError(mid)

        _mock_engine.get_memory.side_effect = side_effect
        _mock_engine.find_child_memory.return_value = child

        result = _consult_memory("curr")
        self.assertIn("- ⬆️ **Prev** : P (ID: p-1)\n\n", result)
        self.assertIn("- ⬇️ **Next** : N (ID: next)\n\n", result)
        _mock_engine.get_memory.side_effect = None

    def test_chantier2_features(self):
        from aivc.core.memory import FileChange
        
        # Test 1: Navigation bullets and double newlines
        parent = _make_memory(memory_id="p-123", title="Parent Title")
        current = _make_memory(
            memory_id="curr-123", 
            title="Current Title", 
            parent_id="p-123",
            changes=[
                FileChange(path="src/modified.py", action="modified", blob_hash="mod", bytes_added=10, bytes_removed=5),
                FileChange(path="src/consulted.py", action="consulted", blob_hash="cons", bytes_added=0, bytes_removed=0),
                FileChange(path="src/new.py", action="added", blob_hash="new", bytes_added=100, bytes_removed=0)
            ]
        )
        child = _make_memory(memory_id="c-123", title="Child Title")

        def mock_get_memory(mid):
            if mid == "p-123": return parent
            if mid == "curr-123": return current
            raise KeyError(mid)

        _mock_engine.get_memory.side_effect = mock_get_memory
        _mock_engine.find_child_memory.return_value = child

        # Mock read_file_at_memory for diffing
        def mock_read_file(path, memory_id):
            if memory_id == "curr-123":
                if path == "src/modified.py":
                    return b"line1\nline2_modified\nline3\n"
                elif path == "src/new.py":
                    return b"new1\nnew2\n"
            elif memory_id == "p-123":
                if path == "src/modified.py":
                    return b"line1\nline2\nline3\n"
            raise KeyError(f"Not found: {path} @ {memory_id}")

        _mock_engine.read_file_at_memory.side_effect = mock_read_file

        result = _consult_memory("curr-123")

        # Verify Navigation bullets (double newlines format)
        self.assertIn("- ⬆️ **Prev** : Parent Title (ID: p-123)\n\n", result)
        self.assertIn("- ⬇️ **Next** : Child Title (ID: c-123)\n\n", result)

        # Verify Separation of modified vs. consulted files
        self.assertIn("### Fichiers modifiés", result)
        self.assertIn("### Fichiers consultés", result)

        # Verify Diff statistics
        # src/modified.py changed line2 -> line2_modified (1 addition, 1 deletion): (+1 -1)
        self.assertIn("(+1 -1)", result)
        # src/new.py is new (no parent): 2 additions: (+2 -0)
        self.assertIn("(+2 -0)", result)

        _mock_engine.get_memory.side_effect = None
        _mock_engine.read_file_at_memory.side_effect = None


class TestGetRecentMemories(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_paginates_correctly(self):
        memories = [_make_memory(memory_id=f"m{i}", title=f"Memory {i}") for i in range(15)]
        _mock_engine.get_log.return_value = memories
        _mock_engine.get_memory_files.return_value = ["src/foo.py"]

        result = _get_recent_memories(limit=5, offset=5)
        self.assertIn("Memory 5", result)
        self.assertNotIn("Memory 0", result)

    def test_empty_range_returns_graceful_message(self):
        _mock_engine.get_log.return_value = []
        result = _get_recent_memories()
        self.assertIn("No memories found", result)

    def test_limit_capped_at_50(self):
        _mock_engine.get_log.return_value = []
        _get_recent_memories(limit=200)
        # Must request offset+50 at most
        _mock_engine.get_log.assert_called_once_with(limit=50)


class TestGetFileHistoryMetadata(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_returns_memory_history(self):
        _mock_engine.get_file_memories.return_value = ["abc-123"]
        _mock_engine.get_memory.return_value = _make_memory()
        result = _get_file_history_metadata("src/foo.py")
        self.assertIn("src/foo.py", result)
        self.assertIn("Do something", result)

    def test_key_error_propagates(self):
        _mock_engine.get_file_memories.side_effect = KeyError("File not in graph")
        with self.assertRaises(KeyError):
            _get_file_history_metadata("src/unknown.py")

    def test_empty_memory_list(self):
        _mock_engine.get_file_memories.return_value = []
        result = _get_file_history_metadata("src/orphan.py")
        self.assertIn("No memories found", result)


class TestReadPastFileContent(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_returns_decoded_utf8(self):
        _mock_engine.read_file_at_memory.return_value = b"# Hello World\n"
        result = _read_past_file_content("src/foo.py", "abc-123", diff_against="none")
        self.assertEqual(result, "# Hello World\n")

    def test_resolves_relative_path_to_absolute(self):
        from pathlib import Path
        _mock_engine.read_file_at_memory.return_value = b"# Hello Resolved\n"
        expected_abs_path = str(Path("src/foo.py").resolve())
        
        result = _read_past_file_content("src/foo.py", "abc-123", diff_against="none")
        
        _mock_engine.read_file_at_memory.assert_called_once_with(expected_abs_path, "abc-123")
        self.assertEqual(result, "# Hello Resolved\n")

    def test_returns_diff_current(self):
        import os
        from unittest.mock import patch
        _mock_engine.read_file_at_memory.return_value = b"# Hello World\n"
        with patch('os.path.exists', return_value=False):
            result = _read_past_file_content("src/foo.py", "abc-123", diff_against="current")
            self.assertIn("```diff", result)
            self.assertIn("-# Hello World", result)

    def test_returns_diff_parent(self):
        _mock_engine.read_file_at_memory.side_effect = lambda path, mid: (
            b"# Parent World\n" if mid == "parent-123" else b"# Hello World\n"
        )
        parent_memory = _make_memory()
        parent_memory.id = "parent-123"
        current_memory = _make_memory()
        current_memory.id = "abc-123"
        current_memory.parent_id = "parent-123"
        
        def mock_get_memory(mid):
            if mid == "abc-123":
                return current_memory
            return parent_memory
            
        _mock_engine.get_memory.side_effect = mock_get_memory
        
        result = _read_past_file_content("src/foo.py", "abc-123", diff_against="parent")
        self.assertIn("```diff", result)
        self.assertIn("-# Parent World", result)
        self.assertIn("+# Hello World", result)
        
        _mock_engine.get_memory.side_effect = None
        _mock_engine.read_file_at_memory.side_effect = None

    def test_invalid_diff_against_raises_value_error(self):
        with self.assertRaises(ValueError):
            _read_past_file_content("src/foo.py", "abc-123", diff_against="invalid")

    def test_key_error_propagates(self):
        _mock_engine.read_file_at_memory.side_effect = KeyError("Not found")
        _mock_engine.get_memory.side_effect = KeyError("Not found")
        result = _read_past_file_content("src/foo.py", "bad-memory", diff_against="none")
        self.assertIn("ERROR:", result)
        _mock_engine.read_file_at_memory.side_effect = None
        _mock_engine.get_memory.side_effect = None


class TestGetStatus(unittest.TestCase):
    def setUp(self):
        _mock_engine.reset_mock(return_value=True, side_effect=True)

    def test_returns_tree_structure(self):
        _mock_engine.get_tracked_paths.return_value = ["/abs/src/foo.py"]
        result = _get_status()
        self.assertIn("📁 **Root**", result)

    def test_no_tracked_files(self):
        _mock_engine.get_tracked_paths.return_value = []
        result = _get_status()
        self.assertIn("No files are currently tracked", result)

    def test_missing_file_handled(self):
        _mock_engine.get_tracked_paths.return_value = ["/abs/src/foo.py"]
        result = _get_status()
        self.assertIn("📁 **Root**", result)


if __name__ == "__main__":
    unittest.main()
