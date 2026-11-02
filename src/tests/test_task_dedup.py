"""Tests for deterministic 100% task_id routing, atomic task_commit, and semantic export dedup."""

from pathlib import Path
import pytest

from aivc.core.workspace import Workspace
from aivc.sync.semantic_export import collect_tasks_from_commits


def test_deterministic_task_id_routing(tmp_path: Path, monkeypatch):
    """Verify that tasks are routed strictly by task_id:
    - Similar titles with different IDs remain distinct.
    - Same ID replaces in-situ, preserves created_at, and merges files.
    """
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    proj_file = "~/Documents/VoiceNotes/notes/Dossier Ethique AAAI.md"

    # 1. Two tasks with similar titles but different task_ids
    mem1 = ws.remember(
        title="Session 1",
        journal="Init",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "task-001",
                "task": "Inspecter en 1 clic les 4 PDF ci-dessus",
                "details": "Details 1",
                "task_status": "await_validation",
                "edited_files": ["~/file1.txt"],
            },
            {
                "id": "task-002",
                "task": "Inspecter en 1 clic les 4 PDF rectifiés au 29/10/2026 pour Yash",
                "details": "Details 2",
                "task_status": "await_validation",
                "edited_files": ["~/file2.txt"],
            },
        ],
    )
    assert len(mem1.user_items) == 2
    assert mem1.user_items[0]["id"] == "task-001"
    assert mem1.user_items[1]["id"] == "task-002"
    orig_created_at = mem1.user_items[0]["created_at"]

    # 2. Update task-001 with new title and new edited_files
    mem2 = ws.remember(
        title="Session 2",
        journal="Update task-001",
        main_project_file=proj_file,
        user_items=[
            {
                "id": "task-001",
                "task": "Inspecter en 1 clic les 4 PDF finaux",
                "details": "Details updated",
                "task_status": "await_validation",
                "edited_files": ["~/file3.txt"],
            }
        ],
    )
    assert len(mem2.user_items) == 1
    t1_updated = mem2.user_items[0]
    assert t1_updated["id"] == "task-001"
    assert t1_updated["task"] == "Inspecter en 1 clic les 4 PDF finaux"
    assert t1_updated["details"] == "Details updated"
    assert t1_updated["created_at"] == orig_created_at
    # Merged files (union)
    assert "~/file1.txt" in t1_updated["edited_files"]
    assert "~/file3.txt" in t1_updated["edited_files"]


def test_atomic_task_commit_and_drive_gc(tmp_path: Path, monkeypatch):
    """Verify that Workspace.task_commit creates a clean 1-task memory and triggers Drive GC on validation."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    proj_file = "~/Documents/VoiceNotes/notes/Dossier Ethique AAAI.md"
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "methodology.tex").write_text("methodology", encoding="utf-8")
    (paper_dir / "results.pdf").write_bytes(b"%PDF-dummy")
    (paper_dir / "intro.tex").write_text("intro", encoding="utf-8")

    # 1. Create a task via task_commit
    mem1 = ws.task_commit(
        title="Rédiger section méthodologie",
        content="Tu m'avais demandé d'écrire la section méthodologie. J'ai rédigé 3 paragraphes avec les métriques clés.",
        main_project_file=proj_file,
        task_id="meth-01",
        task_status="await_validation",
        edited_files=["~/paper/methodology.tex", "~/paper/results.pdf"],
        read_files=["~/paper/intro.tex"],
    )
    assert len(mem1.user_items) == 1
    t = mem1.user_items[0]
    assert t["id"] == "meth-01"
    assert t["task"] == "Rédiger section méthodologie"
    assert t["task_status"] == "await_validation"
    assert "~/paper/methodology.tex" in t["edited_files"]
    assert "~/paper/intro.tex" in t["read_files"]

    # 2. Mark validated via task_commit
    mem2 = ws.task_commit(
        title="Rédiger section méthodologie",
        content="[VALIDÉ PAR HENRI] Validé lors du point du matin.",
        main_project_file=proj_file,
        task_id="meth-01",
        task_status="validated",
    )
    assert len(mem2.user_items) == 1
    assert mem2.user_items[0]["task_status"] == "validated"
    assert mem2.user_items[0]["done"] is True

    # 3. Verify Drive GC collected candidate paths
    gc_files = getattr(ws, "_last_gc_drive_files", [])
    assert len(gc_files) > 0
    portable_gc = [f.replace("\\", "/") for f in gc_files]
    assert any("methodology.tex" in f or "results.pdf" in f for f in portable_gc)


def test_semantic_export_dedup_by_status(tmp_path: Path):
    """Verify that semantic export dedups tasks within history_by_status by task_id."""
    import json
    commits_dir = tmp_path / "commits"
    commits_dir.mkdir(parents=True)
    c1 = {
        "id": "c1",
        "timestamp": "2026-11-01T10:00:00Z",
        "user_items": [
            {"id": "t-1", "task": "Title 1", "task_status": "await_validation"}
        ],
    }
    c2 = {
        "id": "c2",
        "timestamp": "2026-11-01T11:00:00Z",
        "user_items": [
            {"id": "t-1", "task": "Title 1 Updated", "task_status": "await_validation"}
        ],
    }
    (commits_dir / "c1.json").write_text(json.dumps(c1), encoding="utf-8")
    (commits_dir / "c2.json").write_text(json.dumps(c2), encoding="utf-8")
    tasks_current, history_by_status = collect_tasks_from_commits(commits_dir)
    assert len(history_by_status["await_validation"]) == 1
    assert history_by_status["await_validation"][0]["task"] == "Title 1 Updated"
