"""
Tests unitaires pour la synchronisation AIVC :
- Restauration des artefacts issus de antigravity-cli/brain
- Conversion automatique des wikilinks en liens Markdown
- Prise en compte de completed_user_items
"""

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.config import extract_brain_subpath
from aivc.advance_sync import (
    convert_wikilinks_to_markdown_links,
    rewrite_brain_paths_to_temp_artifacts,
    restore_brain_files_for_memory,
)
from aivc.core.memory import Memory, memory_to_dict, memory_from_dict
from aivc.core.workspace import Workspace


def test_antigravity_cli_subpath_and_regex():
    """Vérifie que antigravity-cli/brain est correctement extrait et supporté."""
    raw_cli_path = "C:/Users/Jamet/.gemini/antigravity-cli/brain/84eb6fc4-test/Memo.md"
    sub = extract_brain_subpath(raw_cli_path)
    assert sub == "84eb6fc4-test/Memo.md"

    raw_cli_url = "file:///C:/Users/hjamet/.gemini/antigravity-cli/brain/conv-abc/diff.md"
    sub_url = extract_brain_subpath(raw_cli_url)
    assert sub_url == "conv-abc/diff.md"


def test_wikilinks_conversion_to_markdown_links():
    """Vérifie la conversion automatique de wikilinks [[...]] en liens Markdown [label](file:///...)."""
    # 1. Wikilink avec chemin Windows absolu et pipe label
    text1 = "Consulter [[C:\\Users\\Public\\aivc\\artifacts\\84eb6fc4\\Memo.md|📄 Ouvrir Memo.md]] pour la synthèse."
    converted1 = convert_wikilinks_to_markdown_links(text1)
    assert converted1 == "Consulter [📄 Ouvrir Memo.md](file:///C:/Users/Public/aivc/artifacts/84eb6fc4/Memo.md) pour la synthèse."
    assert "[[" not in converted1

    # 2. Wikilink sans pipe label
    text2 = "Voir [[C:/Users/Public/aivc/artifacts/123/file.md]] directement."
    converted2 = convert_wikilinks_to_markdown_links(text2)
    assert converted2 == "Voir [file.md](file:///C:/Users/Public/aivc/artifacts/123/file.md) directement."
    assert "[[" not in converted2

    # 3. Intégration dans rewrite_brain_paths_to_temp_artifacts
    text3 = "Lien: [[C:\\Users\\hjamet\\.gemini\\antigravity-cli\\brain\\conv-1\\file.md|Mon Document]]."
    rewritten = rewrite_brain_paths_to_temp_artifacts(text3)
    assert "[[" not in rewritten
    assert "[Mon Document](file:///" in rewritten
    assert "conv-1/file.md" in rewritten


def test_restore_brain_files_antigravity_cli_fallback(tmp_path: Path, monkeypatch):
    """Vérifie que la restauration cherche aussi antigravity-cli sur Drive si absent de antigravity."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    monkeypatch.setenv("AIVC_TEMP_DIR", str(tmp_path / "temp"))

    mock_mgr = MagicMock()
    mock_mgr.enabled = True

    # Simuler pull_latest_file : échoue si antigravity, réussit si antigravity-cli
    def fake_pull(file_path: str, target_path: Path | None = None) -> Path:
        if "antigravity-cli" in file_path:
            assert target_path is not None
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text("restored content from cli", encoding="utf-8")
            return target_path
        raise FileNotFoundError("Not found under regular antigravity")

    mock_mgr.pull_latest_file.side_effect = fake_pull

    memory_dict = {
        "id": "mem-cli-1",
        "title": "Test memory with cli artifact",
        "user_items": [
            {
                "id": "item-1",
                "task": "Test task",
                "details": "Check ~/.gemini/antigravity-cli/brain/84eb6fc4/doc.md",
            }
        ],
    }

    restored = restore_brain_files_for_memory(memory_dict, sync_manager=mock_mgr)
    expected_local = tmp_path / ".gemini" / "antigravity" / "brain" / "84eb6fc4" / "doc.md"
    assert str(expected_local) in restored
    assert expected_local.is_file()
    assert expected_local.read_text(encoding="utf-8") == "restored content from cli"


def test_completed_user_items_in_memory_and_workspace(tmp_path: Path, monkeypatch):
    """Vérifie que completed_user_items est sérialisé, désérialisé et pris en compte dans workspace."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    # 1. Vérification dataclass Memory
    mem = Memory.create(
        title="Commit 1",
        journal="Journal test",
        user_items=[{"id": "task-1", "task": "Task 1", "task_status": "await_validation"}],
        completed_user_items=["task-old-1", "task-old-2"],
        main_project_file="notes/Test.md",
    )
    assert mem.completed_user_items == ["task-old-1", "task-old-2"]

    d = memory_to_dict(mem)
    assert d.get("completed_user_items") == ["task-old-1", "task-old-2"]

    restored_mem = memory_from_dict(d)
    assert restored_mem.completed_user_items == ["task-old-1", "task-old-2"]

    # 2. Vérification workspace.remember avec completed_user_items
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    # Premier commit créant item-alpha
    mem1 = ws.remember(
        title="First commit",
        journal="Doing item-alpha",
        user_items=[{"id": "item-alpha", "task": "Do something", "task_status": "await_validation"}],
        main_project_file="notes/Test.md",
    )
    assert any(it.get("id") == "item-alpha" and it.get("task_status") == "await_validation" for it in mem1.user_items)

    # Deuxième commit marquant item-alpha comme complété via completed_user_items
    mem2 = ws.remember(
        title="Second commit",
        journal="Completed item-alpha",
        completed_user_items=["item-alpha"],
        main_project_file="notes/Test.md",
    )
    assert mem2.completed_user_items == ["item-alpha"]

    # Vérification que le premier commit a été mis à jour via event-sourcing
    reloaded_mem1 = ws._load_memory(mem1.id)
    alpha_item = next(it for it in reloaded_mem1.user_items if it.get("id") == "item-alpha")
    assert alpha_item.get("task_status") == "validated"
    assert alpha_item.get("done") is True
