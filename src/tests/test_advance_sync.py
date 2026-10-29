"""
Tests for automatic synchronization of advance_tldr notes from AIVC remember events.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.core.workspace import Workspace
from aivc.core.memory import Memory, FileChange
from aivc.advance_sync import (
    is_obsidian_related_memory,
    sync_advance_tldr,
    set_custom_sync_handler,
    find_project_memory_cli_path,
    extract_obsidian_project_notes,
)


def test_extract_obsidian_project_notes_uses_main_project_file_exclusively():
    mem = Memory(
        id="mem-exclusive",
        timestamp="2026-10-28T10:00:00Z",
        title="Test",
        note="Note",
        parent_id=None,
        changes=[FileChange(path="~/Documents/VoiceNotes/Other.md", action="consulted")],
        main_project_file="~/Documents/VoiceNotes/Main.md",
    )
    notes = extract_obsidian_project_notes(mem)
    assert notes == ["~/Documents/VoiceNotes/Main.md"]


def test_is_obsidian_related_memory_with_user_items():
    mem = Memory(
        id="mem-1",
        timestamp="2026-10-28T10:00:00Z",
        title="Test Memory",
        note="Note",
        parent_id=None,
        user_items=[{"id": "t-1", "task": "Do something", "task_status": "new"}],
    )
    assert is_obsidian_related_memory(mem) is True


def test_is_obsidian_related_memory_with_markdown_file():
    fc = FileChange(path="~/Documents/VoiceNotes/My Project.md", action="consulted")
    mem = Memory(
        id="mem-2",
        timestamp="2026-10-28T10:00:00Z",
        title="Test Memory 2",
        note="Note",
        parent_id=None,
        changes=[fc],
    )
    assert is_obsidian_related_memory(mem) is True


def test_is_obsidian_related_memory_with_code_only():
    fc = FileChange(path="~/Documents/code/aivc/src/main.py", action="edited")
    mem = Memory(
        id="mem-3",
        timestamp="2026-10-28T10:00:00Z",
        title="Refactor core engine",
        note="Refactored without user items",
        parent_id=None,
        changes=[fc],
    )
    assert is_obsidian_related_memory(mem) is False


def test_sync_advance_tldr_with_custom_handler():
    mock_handler = MagicMock(return_value=["/path/to/advance_tldr/🤖 Project.md"])
    set_custom_sync_handler(mock_handler)
    try:
        mem = Memory(
            id="mem-4",
            timestamp="2026-10-28T10:00:00Z",
            title="[Project] Progress",
            note="Note",
            parent_id=None,
            user_items=[{"id": "t-1", "task": "A task", "task_status": "await_validation"}],
        )
        res = sync_advance_tldr(mem)
        assert res == ["/path/to/advance_tldr/🤖 Project.md"]
        assert mock_handler.called
        assert getattr(mem, "_tldr_synced", False) is True

        # Second call should use cached result and not call handler again
        mock_handler.reset_mock()
        res2 = sync_advance_tldr(mem)
        assert res2 == ["/path/to/advance_tldr/🤖 Project.md"]
        assert not mock_handler.called
    finally:
        set_custom_sync_handler(None)


def test_workspace_remember_triggers_sync(tmp_path: Path, monkeypatch):
    """Verify Workspace.remember automatically triggers advance_tldr sync."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    synced_memories = []

    def mock_sync(memory, vault_dir=None):
        synced_memories.append(memory)
        return ["/fake/advance_tldr/🤖 Test.md"]

    set_custom_sync_handler(mock_sync)
    try:
        mem = ws.remember(
            title="[Test Project] Automated Test",
            journal="Testing automatic sync in workspace",
            user_items=[{"id": "t-1", "task": "Validate feature", "task_status": "await_validation"}],
        )
        assert len(synced_memories) == 1
        assert synced_memories[0].id == mem.id
        assert getattr(mem, "_tldr_synced", False) is True
        assert getattr(mem, "_tldr_synced_paths", []) == ["/fake/advance_tldr/🤖 Test.md"]
    finally:
        set_custom_sync_handler(None)


def test_project_memory_cli_path_discovery():
    """Verify project_memory_cli.py is found if VoiceNotes exists."""
    cli_path = find_project_memory_cli_path()
    # If VoiceNotes exists in standard location, it should be discovered
    expected_path = Path.home() / "Documents" / "VoiceNotes" / "_agents" / "scripts-for-skills" / "project_memory_cli.py"
    if expected_path.is_file():
        assert cli_path is not None
        assert cli_path.is_file()


def test_real_end_to_end_advance_tldr_sync_on_disk():
    """Verify real end-to-end sync updates advance_tldr on disk via project_memory_cli."""
    cli_path = find_project_memory_cli_path()
    if not cli_path or not cli_path.is_file():
        pytest.skip("VoiceNotes project_memory_cli.py not found on this machine.")
    from aivc.advance_sync import _load_project_memory_module
    mod = _load_project_memory_module()
    if not hasattr(mod, "sync_advance_tldr_for_memory"):
        pytest.skip("sync_advance_tldr_for_memory not yet implemented in project_memory_cli.py")

    mem = Memory(
        id="test-e2e-sync",
        timestamp="2026-10-28T10:00:00Z",
        title="[Advance] Le Conseil des Voleurs — E2E Test",
        note="Note",
        parent_id=None,
        main_project_file="~/Documents/VoiceNotes/notes/Le Conseil des Voleurs.md",
        changes=[FileChange(path="~/Documents/VoiceNotes/notes/Le Conseil des Voleurs.md", action="consulted")],
        user_items=[{"id": "r-efe056-4", "task": "Session 9", "task_status": "await_validation"}],
    )
    synced = sync_advance_tldr(mem)
    assert len(synced) >= 1
    tldr_path = Path(synced[0])
    assert tldr_path.is_file()
    assert "Le Conseil des Voleurs" in tldr_path.name
    content = tldr_path.read_text(encoding="utf-8")
    assert "# 🤖 Le Conseil des Voleurs" in content
    assert "Session 9" in content


def test_normalize_advance_tldr_title():
    from aivc.advance_sync import normalize_advance_tldr_title
    assert normalize_advance_tldr_title("🤖 Consolidation et Exploration Dream") == "Consolidation et Exploration Dream"
    assert normalize_advance_tldr_title("🤖 🤖 Double Robot") == "Double Robot"
    assert normalize_advance_tldr_title("Standard Project") == "Standard Project"


def test_extract_brain_subpath_with_cli_and_gui():
    from aivc.config import extract_brain_subpath
    p_gui = "file:///C:/Users/Jamet/.gemini/antigravity/brain/conv-123/doc.md"
    p_cli = "file:///C:/Users/Jamet/.gemini/antigravity-cli/brain/c56d3473-5382-403a-bf60-4151776c89b5/dream_report.md"
    assert extract_brain_subpath(p_gui) == "conv-123/doc.md"
    assert extract_brain_subpath(p_cli) == "c56d3473-5382-403a-bf60-4151776c89b5/dream_report.md"


def test_real_end_to_end_no_double_robot_sync():
    """Verify notes starting with 🤖 do not generate double 🤖 🤖 advance_tldr files."""
    cli_path = find_project_memory_cli_path()
    if not cli_path or not cli_path.is_file():
        pytest.skip("VoiceNotes project_memory_cli.py not found on this machine.")
    from aivc.advance_sync import _load_project_memory_module
    mod = _load_project_memory_module()
    if not hasattr(mod, "sync_advance_tldr_for_memory"):
        pytest.skip("sync_advance_tldr_for_memory not yet implemented in project_memory_cli.py")

    mem = Memory(
        id="test-e2e-no-double-robot",
        timestamp="2026-10-29T10:00:00Z",
        title="[Advance] 🤖 Consolidation et Exploration Dream — Test",
        note="Note",
        parent_id=None,
        main_project_file="~/Documents/VoiceNotes/notes/🤖 Consolidation et Exploration Dream.md",
        changes=[FileChange(path="~/Documents/VoiceNotes/notes/🤖 Consolidation et Exploration Dream.md", action="consulted")],
        user_items=[{"id": "r-dream-1", "task": "Test Double Robot Eradication", "task_status": "await_validation"}],
    )
    synced = sync_advance_tldr(mem)
    assert len(synced) >= 1
    tldr_path = Path(synced[0])
    assert tldr_path.is_file()
    assert tldr_path.name == "🤖 Consolidation et Exploration Dream.md"
    content = tldr_path.read_text(encoding="utf-8")
    assert "# 🤖 Consolidation et Exploration Dream" in content
    assert "# 🤖 🤖" not in content


