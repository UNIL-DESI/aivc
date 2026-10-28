"""
Tests for semantic Google Drive export (Volet A).
Covers clean_semantic_text, project name extraction, task lifecycle deduplication,
document formatting, and mock Drive API upload/pagination.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.sync.semantic_export import (
    clean_semantic_text,
    extract_project_name,
    collect_tasks_from_commits,
    format_live_document,
    format_archive_document,
    SemanticDriveExporter,
)


def test_clean_semantic_text_strips_markdown_links():
    raw = "Vérifier le livrable [sources_executant.md](file:///C:/Users/Jamet/.gemini/antigravity/brain/123/scratch/sources_executant.md) et continuer."
    cleaned = clean_semantic_text(raw)
    assert "[sources_executant.md]" not in cleaned
    assert "file:///" not in cleaned
    assert "sources_executant.md" in cleaned


def test_clean_semantic_text_strips_raw_file_paths():
    raw = "Modifications appliquées dans C:\\Users\\Jamet\\Documents\\code\\aivc\\src\\main.py et ~/Documents/VoiceNotes/AIVC.md avec succès."
    cleaned = clean_semantic_text(raw)
    assert "C:\\Users" not in cleaned
    assert "~/Documents" not in cleaned
    assert "Modifications appliquées dans" in cleaned
    assert "avec succès." in cleaned


def test_clean_semantic_text_strips_brain_paths():
    raw = "Consulter .gemini/antigravity/brain/abc-123/rapport.md pour les détails."
    cleaned = clean_semantic_text(raw)
    assert ".gemini/antigravity/brain" not in cleaned
    assert "rapport.md" in cleaned


def test_clean_semantic_text_paths_to_basename():
    # Windows path with line numbers
    raw1 = "Fichier modifié dans C:\\Users\\Jamet\\Documents\\code\\aivc\\src\\main.py:L10-L25 avec succès."
    assert clean_semantic_text(raw1) == "Fichier modifié dans main.py avec succès."

    # POSIX path and monorepo without trailing empty space before comma
    raw2 = "Dans c:/Users/Jamet/Documents/code/monorepo , branche main..."
    assert clean_semantic_text(raw2) == "Dans monorepo, branche main..."

    # Brain scratch path
    raw3 = "Voir .gemini/antigravity/brain/123-abc/scratch/test.py:L5 pour détails."
    assert clean_semantic_text(raw3) == "Voir test.py pour détails."


def test_clean_semantic_text_urls_to_domain():
    raw1 = "inspection de http://localhost:8088/test (qui renvoie 503)"
    cleaned1 = clean_semantic_text(raw1)
    assert cleaned1 == "inspection de localhost:8088 (qui renvoie 503)"
    assert "http" not in cleaned1

    raw2 = "Consulter https://domaine.com/page/subpage pour voir."
    cleaned2 = clean_semantic_text(raw2)
    assert cleaned2 == "Consulter domaine.com pour voir."
    assert "https" not in cleaned2


def test_clean_semantic_text_cleans_orphan_line_references():
    raw = "Source : :L48-L62 et fin."
    cleaned = clean_semantic_text(raw)
    assert ":L48-L62" not in cleaned
    assert cleaned == "Source : et fin."


def test_clean_semantic_text_preserves_utf8():
    raw = "Éradication complète du problème avec Vânia et Stéphane à Genève : déjà réglé !"
    cleaned = clean_semantic_text(raw)
    assert cleaned == "Éradication complète du problème avec Vânia et Stéphane à Genève : déjà réglé !"


def test_extract_project_name():
    # From notes/ path
    p1 = extract_project_name([{"path": "notes/Améliorer le système Antigravity Obsidian.md"}])
    assert p1 == "Projet : Améliorer le système Antigravity Obsidian"

    # From root VoiceNotes markdown
    p2 = extract_project_name(["C:/Users/Jamet/Documents/VoiceNotes/AIVC.md"])
    assert p2 == "Projet : AIVC"

    # From memory title
    p3 = extract_project_name([], memory_title="[Asharde JDR] Avancement scénario 4")
    assert p3 == "Projet : Asharde JDR"

    # Default
    p4 = extract_project_name([])
    assert p4 == "Projet Général"


def test_extract_project_name_fallbacks():
    # Generic read_files fallback to title with [Advance]
    p1 = extract_project_name(["notes/Roadmap.md"], memory_title="[Advance] Chibre — Exploration IS-MCTS")
    assert p1 == "Projet : Chibre"

    # Generic read_files fallback to title with [Roadmap]
    p2 = extract_project_name([], memory_title="[Roadmap] Antigravity Téléphone MCP")
    assert p2 == "Projet : Antigravity Téléphone MCP"

    # Fallback to "Roadmap synchronisée pour <Projet>"
    p3 = extract_project_name([], memory_title="Roadmap synchronisée pour Asharde")
    assert p3 == "Projet : Asharde"

    # Fallback to journal first line
    p4 = extract_project_name(
        ["notes/memoire_principale.md"],
        memory_title="Session de travail",
        journal="## Session Pomodoro Autonome `/advance` du 28/10/2026 — Refonte du site WordPress de SEM\nDetails...",
    )
    assert p4 == "Projet : Refonte du site WordPress de SEM"

    # Fallback to title with project: ...
    p5 = extract_project_name([], memory_title="Organisation du cours AIB 2026 : Clôture courriels")
    assert p5 == "Projet : Organisation du cours AIB 2026"


def test_format_live_document_no_duplicate_project_prefix():
    items = [
        {
            "id": "task-1",
            "task": "Test tâche",
            "project": "Projet : Test",
            "timestamp": "2026-10-28T12:00:00Z",
            "status": "await_validation",
            "details": "Explication claire sans chemins",
            "journal": "Récit de session",
        },
        {
            "id": "task-2",
            "task": "Autre tâche",
            "project": "Projet : Projet : Double",
            "timestamp": "2026-10-28T12:00:00Z",
            "status": "await_validation",
            "details": "",
            "journal": "",
        },
    ]
    doc = format_live_document("await_validation", items)
    assert "- **Projet** : Test" in doc
    assert "- **Projet** : Double" in doc
    assert "Projet : Projet :" not in doc
    assert "- **Projet** : Projet :" not in doc


def test_collect_tasks_from_commits_lifecycle(tmp_path: Path):
    commits_dir = tmp_path / "commits"
    commits_dir.mkdir(parents=True)

    # 1. Commit 1: Task created as "new"
    c1 = {
        "id": "mem-1",
        "timestamp": "2026-10-28T10:00:00Z",
        "title": "Initial planning",
        "read_files": ["notes/Projet Alpha.md"],
        "user_items": [
            {"id": "task-1", "task": "Créer la maquette", "details": "Maquette Figma", "task_status": "new"},
            {"id": "task-2", "task": "Écrire les specs", "details": "Specs V1", "task_status": "new"},
        ],
    }
    (commits_dir / "mem-1.json").write_text(json.dumps(c1), encoding="utf-8")

    # 2. Commit 2: Task 1 moves to "await_validation"
    c2 = {
        "id": "mem-2",
        "timestamp": "2026-10-28T11:00:00Z",
        "title": "Maquette terminée",
        "read_files": ["notes/Projet Alpha.md"],
        "journal": "J'ai terminé la maquette complète.",
        "user_items": [
            {"id": "task-1", "task": "Créer la maquette", "details": "Maquette prête à valider", "task_status": "await_validation"},
        ],
    }
    (commits_dir / "mem-2.json").write_text(json.dumps(c2), encoding="utf-8")

    # 3. Commit 3: Task 1 validated by user
    c3 = {
        "id": "mem-3",
        "timestamp": "2026-10-28T12:00:00Z",
        "title": "Validation utilisateur",
        "read_files": ["notes/Projet Alpha.md"],
        "user_items": [
            {"id": "task-1", "task": "Créer la maquette", "details": "Validé", "task_status": "validated"},
        ],
    }
    (commits_dir / "mem-3.json").write_text(json.dumps(c3), encoding="utf-8")

    tasks_current, history = collect_tasks_from_commits(commits_dir)

    # Current state
    assert tasks_current["task-1"]["status"] == "validated"
    assert tasks_current["task-2"]["status"] == "new"

    # History counts
    assert len(history["new"]) == 2  # task-1 and task-2
    assert len(history["await_validation"]) == 1  # task-1
    assert len(history["validated"]) == 1  # task-1


def test_format_live_document():
    items = [
        {
            "id": "task-1",
            "task": "Test tâche",
            "project": "Projet : Test",
            "timestamp": "2026-10-28T12:00:00Z",
            "status": "await_validation",
            "details": "Explication claire sans chemins",
            "journal": "Récit de session",
        }
    ]
    doc = format_live_document("await_validation", items)
    assert "# AIVC — Décisions & Livrables en Attente de Validation (Live_Await_Validation)" in doc
    assert "[task-1] Test tâche" in doc
    assert "Explication claire sans chemins" in doc
    assert "Récit de session" in doc


def test_semantic_drive_exporter_mock(tmp_path: Path):
    commits_dir = tmp_path / "commits"
    commits_dir.mkdir(parents=True)

    # Create dummy tasks
    items = []
    for i in range(35):
        items.append({"id": f"t-{i}", "task": f"Tâche numéro {i}", "task_status": "new"})
    c = {
        "id": "mem-bulk",
        "timestamp": "2026-10-28T10:00:00Z",
        "title": "Bulk Tasks",
        "user_items": items,
    }
    (commits_dir / "bulk.json").write_text(json.dumps(c), encoding="utf-8")

    # Mock Drive Service
    mock_service = MagicMock()
    mock_files = mock_service.files.return_value
    mock_files.list.return_value.execute.return_value = {"files": []}
    mock_files.create.return_value.execute.return_value = {"id": "mock-doc-id"}
    mock_files.update.return_value.execute.return_value = {"id": "mock-doc-id"}

    mock_manager = MagicMock()
    mock_manager.enabled = True
    mock_manager._get_root_folder_id.return_value = "mock-root-id"
    mock_manager._get_service.return_value = mock_service

    exporter = SemanticDriveExporter(drive_sync_manager=mock_manager, storage_root=tmp_path)
    result = exporter.export_all(batch_size=25)

    assert result["status"] == "success"
    assert result["synced_live"] == 3
    # 35 items paginated by 25: Archives_New should have 2 batches (001 and 002)
    # Archives_Await_Validation: 1 batch (empty list produces 1 empty/initial doc)
    # Archives_Validated: 1 batch
    assert result["synced_archives"] >= 4


def test_format_live_and_archive_document_validated_details():
    items = [
        {
            "id": "t-val-1",
            "task": "Intégration validée",
            "project": "Projet : AIVC",
            "timestamp": "2026-10-28T12:00:00Z",
            "status": "validated",
            "details": "[VALIDÉ PAR HENRI] Validation formelle reçue.",
            "journal": "",
        },
        {
            "id": "t-val-2",
            "task": "Piste écartée",
            "project": "Projet : AIVC",
            "timestamp": "2026-10-28T12:00:00Z",
            "status": "validated",
            "details": "[ÉCARTÉ PAR HENRI] Piste abandonnée au profit d'une alternative.",
            "journal": "",
        },
    ]
    live_doc = format_live_document("validated", items)
    assert "- **Décision & Détails** :\n[VALIDÉ PAR HENRI] Validation formelle reçue." in live_doc
    assert "- **Décision & Détails** :\n[ÉCARTÉ PAR HENRI] Piste abandonnée au profit d'une alternative." in live_doc

    archive_doc = format_archive_document("validated", items, 1, 1)
    assert "- **Décision & Détails** :\n[VALIDÉ PAR HENRI] Validation formelle reçue." in archive_doc
    assert "- **Décision & Détails** :\n[ÉCARTÉ PAR HENRI] Piste abandonnée au profit d'une alternative." in archive_doc


def test_semantic_drive_exporter_blocked_during_pytest(tmp_path: Path):
    from aivc.sync.semantic_export import trigger_semantic_export_async

    # Non-mock manager or default manager should be blocked
    real_like_manager = MagicMock()
    # If not a MagicMock instance check, let's test with a dummy object
    class FakeRealManager:
        enabled = True
        storage_root = tmp_path

    exporter = SemanticDriveExporter(drive_sync_manager=FakeRealManager(), storage_root=tmp_path)
    res = exporter.export_all()
    assert res["status"] == "blocked"
    assert res["reason"] == "pytest_active"

    # trigger_semantic_export_async should return immediately without thread execution
    # when under pytest
    trigger_semantic_export_async(tmp_path)


def test_upsert_gdoc_and_folder_deduplication(tmp_path: Path):
    mock_service = MagicMock()
    mock_files = mock_service.files.return_value

    # Simulate duplicates in list
    mock_files.list.return_value.execute.side_effect = [
        # for folder
        {"files": [{"id": "f-orig", "name": "TestFolder"}, {"id": "f-dup1", "name": "TestFolder"}]},
        # for doc
        {"files": [{"id": "d-orig", "name": "TestDoc"}, {"id": "d-dup1", "name": "TestDoc"}]},
    ]
    mock_files.update.return_value.execute.side_effect = [
        {"id": "trashed-folder"},
        {"id": "trashed-doc"},
        {"id": "d-orig"},
    ]

    mock_manager = MagicMock()
    mock_manager.enabled = True
    mock_manager._get_service.return_value = mock_service

    exporter = SemanticDriveExporter(drive_sync_manager=mock_manager, storage_root=tmp_path)
    folder_id = exporter._get_or_create_folder("TestFolder", "root")
    assert folder_id == "f-orig"
    # dup1 must be trashed
    mock_files.update.assert_any_call(fileId="f-dup1", body={"trashed": True})

    doc_id = exporter._upsert_gdoc("TestDoc", "root", "hello")
    assert doc_id == "d-orig"
    # dup1 must be trashed
    mock_files.update.assert_any_call(fileId="d-dup1", body={"trashed": True})

