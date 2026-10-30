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


def test_sync_advance_tldr_with_dict():
    mock_handler = MagicMock(return_value=["/path/to/advance_tldr/🤖 Dict.md"])
    set_custom_sync_handler(mock_handler)
    try:
        mem_dict = {
            "id": "mem-dict-1",
            "timestamp": "2026-10-29T12:00:00Z",
            "title": "[Dict Project] Test",
            "note": "Note",
            "user_items": [{"id": "t-1", "task": "A task", "task_status": "new"}],
        }
        res = sync_advance_tldr(mem_dict)
        assert res == ["/path/to/advance_tldr/🤖 Dict.md"]
        assert mock_handler.called
        assert mem_dict.get("_tldr_synced") is True
        assert mem_dict.get("_tldr_synced_paths") == ["/path/to/advance_tldr/🤖 Dict.md"]

        # Second call uses cache
        mock_handler.reset_mock()
        res2 = sync_advance_tldr(mem_dict)
        assert res2 == ["/path/to/advance_tldr/🤖 Dict.md"]
        assert not mock_handler.called
    finally:
        set_custom_sync_handler(None)


def test_pull_memories_from_others_triggers_sync_advance_tldr(tmp_path: Path, monkeypatch):
    """Verify pull_memories_from_others triggers auto_restore and sync_advance_tldr for new commits."""
    import json
    from aivc.sync.drive import NativeDriveSyncManager

    storage_root = tmp_path / ".aivc" / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    manager = NativeDriveSyncManager(storage_root)
    manager.enabled = True
    manager.machine_id = "local-machine-id"

    mock_service = MagicMock()
    monkeypatch.setattr(manager, "_get_service", lambda: mock_service)
    monkeypatch.setattr(manager, "_get_root_folder_id", lambda: "root-folder-id")

    machine_folders_res = {"files": [{"id": "remote-machine-folder", "name": "remote-machine"}]}
    commits_folder_res = {"files": [{"id": "remote-commits-folder", "name": "commits"}]}
    memory_files_res = {"files": [{"id": "remote-mem-file-1", "name": "mem-123.json"}]}

    mock_service.files().list.return_value.execute.side_effect = [
        machine_folders_res,
        commits_folder_res,
        memory_files_res,
    ]

    remote_mem_content = {
        "id": "mem-123",
        "timestamp": "2026-10-29T12:00:00Z",
        "title": "[Sync Test Project] From Remote Drive",
        "note": "Downloaded memory",
        "main_project_file": "~/Documents/VoiceNotes/Remote Project.md",
        "user_items": [{"id": "t-remote-1", "task": "Validate remote sync", "task_status": "new"}],
    }

    def fake_download(file_id, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(remote_mem_content), encoding="utf-8")

    monkeypatch.setattr(manager, "_download_file", fake_download)

    auto_restore_called_with = []
    def fake_auto_restore(mem_data):
        auto_restore_called_with.append(mem_data)
        return []

    monkeypatch.setattr(manager, "auto_restore_brain_files_from_data", fake_auto_restore)

    synced_memories = []
    def fake_sync(mem, vault_dir=None):
        synced_memories.append(mem)
        return ["/fake/advance_tldr/🤖 Remote Project.md"]

    set_custom_sync_handler(fake_sync)
    try:
        pulled = manager.pull_memories_from_others()
        assert pulled == 1
        assert len(auto_restore_called_with) == 1
        assert auto_restore_called_with[0]["id"] == "mem-123"
        assert len(synced_memories) == 1
        assert synced_memories[0]["id"] == "mem-123"
        assert synced_memories[0].get("_tldr_synced") is True
        assert synced_memories[0].get("_tldr_synced_paths") == ["/fake/advance_tldr/🤖 Remote Project.md"]
    finally:
        set_custom_sync_handler(None)


def test_restore_brain_files_from_tldr_file_path(tmp_path: Path, monkeypatch):
    """Verify restore_brain_files_for_memory accepts a file path, scans it, and decodes %20 URLs."""
    from aivc.advance_sync import restore_brain_files_for_memory

    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    pulled_files = []
    mock_mgr = MagicMock()
    mock_mgr.enabled = True
    def fake_pull(portable, target_path=None):
        pulled_files.append((portable, target_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("restored", encoding="utf-8")
        return target_path

    mock_mgr.pull_latest_file = fake_pull

    tldr_file = tmp_path / "advance_tldr" / "🤖 Frontier.md"
    tldr_file.parent.mkdir(parents=True, exist_ok=True)
    tldr_file.write_text(
        "Link: [Doc](file:///C:/Users/test/.gemini/antigravity/brain/conv-abc/Analyse%20Équilibrage%20Test.md)\n"
        "Canvas: [Canvas](file:///C:/Users/test/.gemini/antigravity/brain/conv-abc/Frontieres.canvas)",
        encoding="utf-8",
    )

    restored = restore_brain_files_for_memory(tldr_file, sync_manager=mock_mgr)
    assert len(restored) == 2
    assert any("Analyse Équilibrage Test.md" in r for r in restored)
    assert any("Frontieres.canvas" in r for r in restored)


def test_restore_brain_files_scans_existing_advance_tldr_note(tmp_path: Path, monkeypatch):
    """Verify restore_brain_files_for_memory scans the existing advance_tldr note of a project."""
    from aivc.advance_sync import restore_brain_files_for_memory

    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    pulled_files = []
    mock_mgr = MagicMock()
    mock_mgr.enabled = True
    def fake_pull(portable, target_path=None):
        pulled_files.append((portable, target_path))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("restored", encoding="utf-8")
        return target_path

    mock_mgr.pull_latest_file = fake_pull

    # Pre-create an existing advance_tldr note on disk referencing a brain file
    tldr_note = tmp_path / ".gemini" / "antigravity" / "brain" / "advance_tldr" / "🤖 Frontier.md"
    tldr_note.parent.mkdir(parents=True, exist_ok=True)
    tldr_note.write_text(
        "Existing link: [PDF](file:///C:/Users/other/.gemini/antigravity/brain/conv-xyz/rules.pdf)",
        encoding="utf-8",
    )

    # Memory only references the main project file, no edited_files
    mem = {
        "id": "mem-frontier-new",
        "title": "[Advance] Frontier",
        "main_project_file": "~/Documents/VoiceNotes/notes/Frontier.md",
        "user_items": [],
    }

    restored = restore_brain_files_for_memory(mem, sync_manager=mock_mgr)
    assert len(restored) == 1
    assert "rules.pdf" in restored[0]




def test_backfill_missing_main_project_files(tmp_path: Path):
    """Verify deterministic backfill of main_project_file and avoidance of guesswork."""
    import json
    from aivc.advance_sync import backfill_missing_main_project_files

    # Create dummy vault
    vault = tmp_path / "VoiceNotes"
    (vault / "notes").mkdir(parents=True)
    (vault / "notes" / "Project Alpha.md").write_text("# Project Alpha", encoding="utf-8")
    (vault / "Organisation du cours Artificial Intelligence in Business 2026.md").write_text("# AIB", encoding="utf-8")

    # Create dummy commit storage
    storage = tmp_path / "storage"
    commits_dir = storage / "commits"
    commits_dir.mkdir(parents=True)

    # Commit 1: matches by [Advance] prefix
    c1 = {
        "id": "c1-uuid",
        "title": "[Advance] Project Alpha — Milestones and deliverable",
        "main_project_file": None,
        "user_items": [{"id": "t-1", "task": "Check alpha"}],
    }
    (commits_dir / "c1.json").write_text(json.dumps(c1), encoding="utf-8")

    # Commit 2: matches special alias AIB 2026
    c2 = {
        "id": "c2-uuid",
        "title": "AIB 2026 — Syllabus update",
        "main_project_file": None,
        "user_items": [{"id": "t-2", "task": "Update Moodle"}],
    }
    (commits_dir / "c2.json").write_text(json.dumps(c2), encoding="utf-8")

    # Commit 3: Ambiguous title without vault match -> must remain None
    c3 = {
        "id": "c3-uuid",
        "title": "Unrelated generic work",
        "main_project_file": None,
        "user_items": [{"id": "t-3", "task": "Random task"}],
    }
    (commits_dir / "c3.json").write_text(json.dumps(c3), encoding="utf-8")

    res = backfill_missing_main_project_files(storage_dir=storage, vault_dir=vault)
    assert res["status"] == "success"
    assert res["updated"] == 2

    d1 = json.loads((commits_dir / "c1.json").read_text(encoding="utf-8"))
    assert d1["main_project_file"] == "~/Documents/VoiceNotes/notes/Project Alpha.md"

    d2 = json.loads((commits_dir / "c2.json").read_text(encoding="utf-8"))
    assert d2["main_project_file"] == "~/Documents/VoiceNotes/Organisation du cours Artificial Intelligence in Business 2026.md"

    d3 = json.loads((commits_dir / "c3.json").read_text(encoding="utf-8"))
    assert d3.get("main_project_file") is None


def test_sync_advance_tldr_mirrors_to_legacy_location(tmp_path: Path, monkeypatch):
    """Verify sync_advance_tldr automatically mirrors generated tldr notes to ~/.gemini/antigravity/brain/advance_tldr/."""
    from aivc.advance_sync import sync_advance_tldr, set_custom_sync_handler

    monkeypatch.setattr("aivc.advance_sync.get_home_dir", lambda: tmp_path)

    # Fake custom handler returning a note in a temp location
    temp_note = tmp_path / "temp_advance_tldr" / "🤖 Project X.md"
    temp_note.parent.mkdir(parents=True, exist_ok=True)
    temp_note.write_text("# 🤖 Project X\n\nContent", encoding="utf-8")

    # Set custom handler
    set_custom_sync_handler(lambda mem, vault_dir=None: [str(temp_note)])
    try:
        mem = {
            "id": "mem-mirror-test",
            "title": "[Advance] Project X",
            "main_project_file": "~/Documents/VoiceNotes/notes/Project X.md",
            "user_items": [],
        }
        synced = sync_advance_tldr(mem)
        assert synced == [str(temp_note)]

        legacy_expected = tmp_path / ".gemini" / "antigravity" / "brain" / "advance_tldr" / "🤖 Project X.md"
        assert legacy_expected.is_file()
        assert legacy_expected.read_text(encoding="utf-8") == "# 🤖 Project X\n\nContent"
    finally:
        set_custom_sync_handler(None)


def test_rewrite_brain_paths_to_temp_artifacts():
    from aivc.advance_sync import rewrite_brain_paths_to_temp_artifacts

    fixed_temp = Path("C:/Users/Jamet/AppData/Local/Temp/aivc")

    # 1. Standard brain artifact link rewritten to temp artifacts
    input_url = "[📄 script.ps1](file:///C:/Users/Jamet/.gemini/antigravity/brain/59bb502e-fd05/script.ps1)"
    expected_url = "[📄 script.ps1](file:///C:/Users/Jamet/AppData/Local/Temp/aivc/artifacts/59bb502e-fd05/script.ps1)"
    assert rewrite_brain_paths_to_temp_artifacts(input_url, temp_dir=fixed_temp) == expected_url

    # 2. Link already on aivc/artifacts remains intact (zero filC:, zero file:///file:///)
    already_art = "[📄 script.ps1](file:///C:/Users/Jamet/AppData/Local/Temp/aivc/artifacts/59bb502e-fd05/script.ps1)"
    res_already = rewrite_brain_paths_to_temp_artifacts(already_art, temp_dir=fixed_temp)
    assert res_already == already_art
    assert "filC:" not in res_already
    assert "file:///file:///" not in res_already

    # 3. Bare local path in command rewritten to native temp artifacts path
    input_cmd = r"pytest C:\Users\Jamet\.gemini\antigravity-cli\brain\59bb502e-fd05\test_x.py"
    expected_local = r"pytest " + str(fixed_temp / "artifacts" / "59bb502e-fd05" / "test_x.py")
    assert rewrite_brain_paths_to_temp_artifacts(input_cmd, temp_dir=fixed_temp) == expected_local

    # 4. Corrupted link filC:\... is repaired to file:///C:/...
    corrupted = r"filC:\Users\Jamet\AppData\Local\Temp\aivc\artifacts\59bb502e-fd05\script.ps1"
    expected_repaired = "file:///C:/Users/Jamet/AppData/Local/Temp/aivc/artifacts/59bb502e-fd05/script.ps1"
    assert rewrite_brain_paths_to_temp_artifacts(corrupted, temp_dir=fixed_temp) == expected_repaired




