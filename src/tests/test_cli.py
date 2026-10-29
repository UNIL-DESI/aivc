"""
Tests for the CLI module.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import os

from aivc.cli import main, _format_bytes

@pytest.fixture
def mock_engine():
    with patch("aivc.cli._get_engine") as mock_get:
        engine = MagicMock()
        mock_get.return_value = engine
        yield engine

def test_format_bytes():
    assert _format_bytes(None) == "missing"
    assert _format_bytes(500) == "500 B"
    assert _format_bytes(1024) == "1.0 KB"
    assert _format_bytes(10 * 1024**2) == "10.0 MB"

def test_cli_log(mock_engine, capsys):
    mock_engine.get_log.return_value = []
    with patch("sys.argv", ["aivc", "log"]):
        main()
    captured = capsys.readouterr()
    assert "No memories found" in captured.out

def test_cli_search(mock_engine, capsys):
    mock_engine.search.return_value = []
    with patch("sys.argv", ["aivc", "search", "test query"]):
        main()
    captured = capsys.readouterr()
    assert "No matching memories found" in captured.out



def test_cli_no_env_var(mock_engine, capsys):
    # Tests that omitting AIVC_STORAGE_ROOT now falls back gracefully
    mock_engine.get_log.return_value = []
    with patch.dict(os.environ, clear=True):
        if "AIVC_STORAGE_ROOT" in os.environ:
            del os.environ["AIVC_STORAGE_ROOT"]
            
        with patch("sys.argv", ["aivc", "log"]):
            main()
    
    captured = capsys.readouterr()
    assert "No memories found" in captured.out

def test_cmd_sync_push_disabled(capsys):
    with patch("aivc.config.get_aivc_config") as mock_cfg:
        mock_cfg.return_value = {"sync": {"enabled": False}}
        from aivc.cli import cmd_sync_push
        cmd_sync_push(MagicMock())

    captured = capsys.readouterr()
    assert "Cloud sync is currently disabled" in captured.out

def test_cmd_sync_push_no_token(capsys):
    with patch("aivc.config.get_aivc_config") as mock_cfg, \
         patch("aivc.config.get_token_path") as mock_token:
        mock_cfg.return_value = {"sync": {"enabled": True}}
        mock_token.return_value.exists.return_value = False
        from aivc.cli import cmd_sync_push
        cmd_sync_push(MagicMock())

    captured = capsys.readouterr()
    assert "Google Drive authentication missing" in captured.out

def test_cmd_sync_push_success_no_changes(capsys):
    with patch("aivc.config.get_aivc_config") as mock_cfg, \
         patch("aivc.config.get_token_path") as mock_token, \
         patch("aivc.sync.drive.NativeDriveSyncManager") as mock_mgr_cls:

        mock_cfg.return_value = {"sync": {"enabled": True}}
        mock_token.return_value.exists.return_value = True

        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.push_missing.return_value = {"memories_pushed": 0}

        from aivc.cli import cmd_sync_push
        cmd_sync_push(MagicMock())

    captured = capsys.readouterr()
    assert "Everything is up-to-date" in captured.out

def test_cmd_sync_push_success_with_changes(capsys):
    with patch("aivc.config.get_aivc_config") as mock_cfg, \
         patch("aivc.config.get_token_path") as mock_token, \
         patch("aivc.sync.drive.NativeDriveSyncManager") as mock_mgr_cls:

        mock_cfg.return_value = {"sync": {"enabled": True}}
        mock_token.return_value.exists.return_value = True

        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.push_missing.return_value = {"memories_pushed": 3}

        from aivc.cli import cmd_sync_push
        cmd_sync_push(MagicMock())

    captured = capsys.readouterr()
    assert "Sync Push complete" in captured.out
    assert "Pushed 3 missing memory/ies" in captured.out

def test_cmd_sync_push_error(capsys):
    with patch("aivc.config.get_aivc_config") as mock_cfg, \
         patch("aivc.config.get_token_path") as mock_token, \
         patch("aivc.sync.drive.NativeDriveSyncManager") as mock_mgr_cls:

        mock_cfg.return_value = {"sync": {"enabled": True}}
        mock_token.return_value.exists.return_value = True

        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.push_missing.side_effect = Exception("API Error")

        from aivc.cli import cmd_sync_push
        cmd_sync_push(MagicMock())

    captured = capsys.readouterr()
    assert "Error during sync push: API Error" in captured.out


def test_dream_inbox_registry_and_filtering(tmp_path: Path, monkeypatch, capsys):
    """Verify dream inbox registry persistence and document masking in list command."""
    import argparse
    import json
    import subprocess
    from aivc.cli import _load_dream_inbox_processed, _save_dream_inbox_processed, cmd_dream_inbox

    monkeypatch.setattr("aivc.config.get_aivc_dir", lambda: tmp_path)

    # 1. Initially registry is empty
    reg = _load_dream_inbox_processed()
    assert reg == {"processed": {}}

    # 2. Add an item and save
    reg["processed"]["doc-123"] = {
        "trashed_at": "2026-10-29T12:00:00Z",
        "method": "local_registry",
        "name": "Notes.docx",
    }
    _save_dream_inbox_processed(reg)

    reg_loaded = _load_dream_inbox_processed()
    assert "doc-123" in reg_loaded["processed"]

    # 3. Test cmd_dream_inbox list filtering
    mock_files = [
        {"Name": "Notes.docx", "ID": "doc-123", "ModTime": "2026-10-29T10:00:00Z"},
        {"Name": "Fresh.docx", "ID": "doc-456", "ModTime": "2026-10-29T11:00:00Z"},
    ]
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: json.dumps(mock_files))

    # Default list excludes doc-123
    args_list = argparse.Namespace(inbox_action="list", all=False)
    cmd_dream_inbox(args_list)
    out1 = capsys.readouterr().out
    assert "Fresh.docx" in out1
    assert "Notes.docx" not in out1

    # list --all includes doc-123 with status tag
    args_all = argparse.Namespace(inbox_action="list", all=True)
    cmd_dream_inbox(args_all)
    out2 = capsys.readouterr().out
    assert "Fresh.docx" in out2
    assert "Notes.docx" in out2
    assert "[déjà traité]" in out2
