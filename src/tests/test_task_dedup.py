"""
Tests for automatic task deduplication and ID reuse across AIVC remember events and advance_tldr rendering.
"""

from pathlib import Path
import pytest

from aivc.core.workspace import (
    Workspace,
    _are_tasks_duplicate,
    _normalize_task_text,
    _extract_task_tokens,
)
from aivc.advance_sync import sync_advance_tldr


def test_are_tasks_duplicate_heuristics():
    # 1. Reformulations with dates/hashes/stopwords
    t1 = "Inspecter en 1 clic les 4 PDF ci-dessus générés par l'agent"
    t2 = "Inspecter en 1 clic les 4 PDF rectifiés au 29/10/2026 pour Yash"
    assert _are_tasks_duplicate(t1, t2) is True

    # 2. Significant prefix shared
    t3 = "Envoyer le courriel J-14 à Yash"
    t4 = "Envoyer le courriel J-14 à Yash avec pièces jointes et récapitulatif"
    assert _are_tasks_duplicate(t3, t4) is True

    # 3. High Jaccard similarity
    t5 = "Soumettre le dossier de subvention HEC"
    t6 = "Soumettre le dossier HEC"
    assert _are_tasks_duplicate(t5, t6) is True

    # 4. Distinct tasks remain distinct
    t_diff1 = "Rédiger l'abstract pour WWW 2027"
    t_diff2 = "Inspecter les 4 PDF"
    assert _are_tasks_duplicate(t_diff1, t_diff2) is False

    t_diff3 = "Préparer la réunion du 12/11/2026"
    t_diff4 = "Déployer la mise à jour sur le serveur de production"
    assert _are_tasks_duplicate(t_diff3, t_diff4) is False


def test_remember_deduplication_and_id_reuse(tmp_path: Path, monkeypatch):
    """Verify that two successive remember calls with reformulated task titles reuse the existing ID
    and produce only ONE active task in advance_tldr (the latest one).
    Also verify distinct tasks remain separate."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    proj_file = "~/Documents/VoiceNotes/notes/Dossier Ethique AAAI.md"

    # Commit 1: first task registration with ID 500601-1
    mem1 = ws.remember(
        title="[HEC Grant] Session 1",
        journal="Initial PDF inspection task",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "500601-1",
                "task": "Inspecter en 1 clic les 4 PDF ci-dessus générés par l'agent",
                "details": "PDFs à valider",
                "task_status": "await_validation",
            }
        ],
    )
    assert len(mem1.user_items) == 1
    assert mem1.user_items[0]["id"] == "500601-1"

    # Commit 2: reformulated task with a different raw_id b83702-1
    mem2 = ws.remember(
        title="[HEC Grant] Session 2",
        journal="Updated PDF inspection task",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "b83702-1",
                "task": "Inspecter en 1 clic les 4 PDF rectifiés au 29/10/2026 pour Yash",
                "details": "PDFs rectifiés après correction",
                "task_status": "await_validation",
            }
        ],
    )
    # The existing ID 500601-1 must be automatically reused
    assert len(mem2.user_items) == 1
    assert mem2.user_items[0]["id"] == "500601-1"
    assert "rectifiés" in mem2.user_items[0]["task"]

    # Commit 3: add a genuinely distinct task
    mem3 = ws.remember(
        title="[HEC Grant] Session 3",
        journal="Add a different milestone",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "500601-1",
                "task": "Inspecter en 1 clic les 4 PDF rectifiés au 29/10/2026 pour Yash",
                "details": "PDFs rectifiés",
                "task_status": "await_validation",
            },
            {
                "task": "Rédiger le rapport d'éthique pour le comité HEC",
                "details": "Section méthodologie et conformité RGPD",
                "task_status": "await_validation",
            },
        ],
    )
    assert len(mem3.user_items) == 2
    assert mem3.user_items[0]["id"] == "500601-1"
    assert mem3.user_items[1]["id"] != "500601-1"
    assert "rapport d'éthique" in mem3.user_items[1]["task"]


def test_advance_tldr_historical_dedup_end_to_end(tmp_path: Path, monkeypatch):
    """Verify that if historical commits contain duplicate tasks with different IDs
    (e.g. 500601-1 and b83702-1), project_memory_cli dedups them so only the latest remains."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    storage_root = tmp_path / ".aivc" / "storage"
    monkeypatch.setenv("PROJECT_MEMORY_AIVC_STORAGE_ROOT", str(storage_root))
    ws = Workspace(storage_root)

    proj_file = "~/Documents/VoiceNotes/notes/Dossier Ethique AAAI.md"

    # Simulate historical commit 1
    mem1 = ws.remember(
        title="[AAAI] Session 1",
        journal="First version",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "500601-1",
                "task": "Inspecter en 1 clic les 4 PDF ci-dessus",
                "details": "Details 1",
                "task_status": "await_validation",
            }
        ],
    )

    # Simulate historical commit 2 created with an arbitrary ID (b83702-1)
    # We write directly to commits to ensure legacy commits without write-side fix are tested
    import json
    c2_id = "b8370200-0000-0000-0000-000000000000"
    c2_data = {
        "id": c2_id,
        "timestamp": "2099-01-01T12:00:00Z",
        "title": "[AAAI] Session 2",
        "journal": "Second version",
        "note": "Note 2",
        "main_project_file": proj_file,
        "user_items": [
            {
                "id": "b83702-1",
                "task": "Inspecter en 1 clic les 4 PDF rectifiés au 29/10/2026 pour Yash",
                "details": "Details 2",
                "task_status": "await_validation",
                "done": False,
            }
        ],
    }
    c2_path = storage_root / "commits" / f"{c2_id}.json"
    c2_path.write_text(json.dumps(c2_data), encoding="utf-8")

    from aivc.advance_sync import _load_project_memory_module
    mod = _load_project_memory_module()
    if mod is None:
        pytest.skip("project_memory_cli module not available")

    # Invalidate cache
    if hasattr(mod, "_invalidate_aivc_cache"):
        mod._invalidate_aivc_cache()

    vault_dir = tmp_path / "Documents" / "VoiceNotes"
    vault_dir.mkdir(parents=True, exist_ok=True)
    note_dir = vault_dir / "notes"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "Dossier Ethique AAAI.md").write_text("# Dossier Ethique AAAI\n", encoding="utf-8")

    tldr_path, item_count = mod._sync_and_render_aivc_tldr(
        vault_root=str(vault_dir),
        rel_path="notes/Dossier Ethique AAAI.md",
        title="Dossier Ethique AAAI",
    )

    assert tldr_path is not None
    assert item_count == 1  # Exactly ONE active item, not 2!

    content = Path(tldr_path).read_text(encoding="utf-8")
    assert "b83702-1" in content or "500601-1" in content
    assert "rectifiés" in content  # Contains the most recent text
