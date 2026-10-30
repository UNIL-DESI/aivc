"""
Unit and integration tests for Dream #49:
- Volet B: Compact commit JSONs without duplicate 'note' key (memory_to_dict, memory_from_dict, compact_commits_json).
- Volet C: Garbage collection on Google Drive of files linked to validated tasks (Workspace._last_gc_drive_files, DriveSyncManager.delete_remote_file).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.core.memory import Memory, memory_to_dict, memory_from_dict
from aivc.core.workspace import Workspace, _extract_linked_file_paths
from aivc.sync.drive import NativeDriveSyncManager


def test_volet_b_memory_to_dict_omits_redundant_note():
    """Volet B: memory_to_dict() should NOT contain 'note' if journal or user_items is present."""
    # 1. With journal and user_items
    mem1 = Memory.create(
        title="Commit avec journal et tâche",
        journal="Journal de la session.",
        user_items=[{"id": "t1", "task": "Tâche 1", "task_status": "new"}],
        main_project_file="notes/Projet.md",
    )
    d1 = memory_to_dict(mem1)
    assert "note" not in d1
    assert d1["journal"] == "Journal de la session."
    assert len(d1["user_items"]) == 1

    # Deserialisation preserves full composed note
    restored1 = memory_from_dict(d1)
    assert "Journal de la session." in restored1.note
    assert "Tâche 1" in restored1.note

    # 2. Legacy memory without journal nor user_items
    mem_legacy = Memory(
        id="legacy-id",
        timestamp="2026-10-28T10:00:00Z",
        title="Legacy commit",
        note="Old unformatted note content.",
        parent_id=None,
        changes=[],
    )
    d_legacy = memory_to_dict(mem_legacy)
    assert "note" in d_legacy
    assert d_legacy["note"] == "Old unformatted note content."

    restored_legacy = memory_from_dict(d_legacy)
    assert restored_legacy.note == "Old unformatted note content."


def test_volet_b_compact_commits_json(tmp_path: Path):
    """Volet B: compact_commits_json() strips 'note' key from existing JSONs."""
    commits_dir = tmp_path / "commits"
    commits_dir.mkdir(parents=True)

    # File 1: Has journal and redundant note
    c1 = {
        "id": "c1",
        "timestamp": "2026-10-28T10:00:00Z",
        "title": "Commit 1",
        "journal": "Agent thoughts here.",
        "note": "Agent thoughts here (redundant).",
        "changes": [],
    }
    (commits_dir / "c1.json").write_text(json.dumps(c1, indent=2), encoding="utf-8")

    # File 2: Has user_items and redundant note
    c2 = {
        "id": "c2",
        "timestamp": "2026-10-28T11:00:00Z",
        "title": "Commit 2",
        "user_items": [{"id": "t1", "task": "Task 1"}],
        "note": "Task 1 note.",
        "changes": [],
    }
    (commits_dir / "c2.json").write_text(json.dumps(c2, indent=2), encoding="utf-8")

    # File 3: Pure legacy (no journal, no user_items) -> should keep 'note'
    c3 = {
        "id": "c3",
        "timestamp": "2026-10-28T12:00:00Z",
        "title": "Legacy 3",
        "note": "Keep this note.",
        "changes": [],
    }
    (commits_dir / "c3.json").write_text(json.dumps(c3, indent=2), encoding="utf-8")

    ws = Workspace(tmp_path)
    count = ws.compact_commits_json()
    assert count == 2

    # Verify files on disk
    d1 = json.loads((commits_dir / "c1.json").read_text(encoding="utf-8"))
    assert "note" not in d1
    assert d1["journal"] == "Agent thoughts here."

    d2 = json.loads((commits_dir / "c2.json").read_text(encoding="utf-8"))
    assert "note" not in d2

    d3 = json.loads((commits_dir / "c3.json").read_text(encoding="utf-8"))
    assert "note" in d3


def test_volet_c_extract_linked_file_paths():
    """Volet C: _extract_linked_file_paths accurately parses file:/// URLs, paths, and brain files."""
    text = (
        "Livrable prêt : [sources_executant.md](file:///C:/Users/Jamet/.gemini/antigravity/brain/123/scratch/sources_executant.md:L10-L20) "
        "et fichier secondaire file:///C:/Users/Jamet/AppData/Local/Temp/aivc/artifacts/123/output.json "
        "ainsi que C:\\Users\\Jamet\\.gemini\\antigravity-cli\\brain\\123\\scratch\\notes.txt."
    )
    paths = _extract_linked_file_paths(text)
    assert any("sources_executant.md" in p for p in paths)
    assert any("output.json" in p for p in paths)
    assert any("notes.txt" in p for p in paths)
    # Line number anchors stripped
    assert not any(":L10-L20" in p for p in paths)


def test_volet_c_gc_drive_files_on_task_validation(tmp_path: Path):
    """Volet C: when tasks are validated, Workspace gathers linked files in _last_gc_drive_files."""
    vault_note = tmp_path / "notes" / "Projet.md"
    vault_note.parent.mkdir(parents=True)
    vault_note.write_text("# Projet", encoding="utf-8")

    # Scratch artifact file (not in VoiceNotes)
    scratch_file = tmp_path / "scratch" / "result.md"
    scratch_file.parent.mkdir(parents=True)
    scratch_file.write_text("Result data", encoding="utf-8")

    ws = Workspace(tmp_path)

    # 1. Commit 1: Task created in await_validation pointing to scratch_file
    m1 = ws.remember(
        title="Livrable soumis",
        journal="J'ai terminé le livrable.",
        tasks=[
            {
                "id": "t-await-1",
                "task": "Valider livrable",
                "details": f"Voir [result.md](file:///{scratch_file.as_posix()})",
                "task_status": "await_validation",
            }
        ],
        main_project_file=str(vault_note),
    )
    assert ws._last_gc_drive_files == []

    # 2. Commit 2: Task validated -> scratch_file must be marked for GC
    m2 = ws.remember(
        title="Validation livrable",
        journal="Livrable validé par l'utilisateur.",
        tasks=[
            {
                "id": "t-await-1",
                "task": "Valider livrable",
                "details": f"[VALIDÉ PAR HENRI] Voir [result.md](file:///{scratch_file.as_posix()})",
                "task_status": "validated",
            }
        ],
        main_project_file=str(vault_note),
    )
    # Resulting gc list should contain scratch_file, and NOT vault_note
    assert len(ws._last_gc_drive_files) >= 1
    assert any("result.md" in p for p in ws._last_gc_drive_files)
    assert not any("Projet.md" in p for p in ws._last_gc_drive_files)


def test_volet_c_delete_remote_file_mock(tmp_path: Path):
    """Volet C: delete_remote_file searches files/ hierarchy and issues delete/trash call."""
    mock_service = MagicMock()
    mock_files = mock_service.files.return_value
    mock_files.list.return_value.execute.return_value = {"files": [{"id": "file-to-delete-id"}]}
    mock_files.delete.return_value.execute.return_value = {}

    manager = NativeDriveSyncManager(storage_root=tmp_path)
    manager.enabled = True
    manager._get_service = MagicMock(return_value=mock_service)
    manager._folder_cache["root/files"] = "files-root-id"

    test_file_path = "~/.gemini/antigravity/brain/uuid-123/scratch/sources.md"
    deleted = manager.delete_remote_file(test_file_path)

    assert deleted is True
    mock_files.delete.assert_called()
