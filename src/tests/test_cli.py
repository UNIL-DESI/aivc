"""
Tests for the CLI module.
"""

from unittest.mock import patch, MagicMock
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


def test_cli_status(mock_engine, capsys):
    mock_engine.get_status.return_value = []
    with patch("sys.argv", ["aivc", "status"]):
        main()
    captured = capsys.readouterr()
    assert "No files are currently tracked" in captured.out


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


def test_cli_track_single_file(mock_engine, capsys):
    mock_engine.track.return_value = {"newly_tracked": ["/home/user/project/src/app.py"], "hidden_skipped": 0}
    with patch("sys.argv", ["aivc", "track", "src/app.py"]):
        main()
    captured = capsys.readouterr()
    assert "Tracked 1 new file(s)" in captured.out
    mock_engine.track.assert_called_once_with("src/app.py", ignores=[])


def test_cli_track_already_tracked(mock_engine, capsys):
    mock_engine.track.return_value = {"newly_tracked": [], "hidden_skipped": 0}
    with patch("sys.argv", ["aivc", "track", "src/app.py"]):
        main()
    captured = capsys.readouterr()
    assert "No new files to track" in captured.out


def test_cli_no_env_var(mock_engine, capsys):
    # Tests that omitting AIVC_STORAGE_ROOT now falls back gracefully
    mock_engine.get_status.return_value = []
    with patch.dict(os.environ, clear=True):
        if "AIVC_STORAGE_ROOT" in os.environ:
            del os.environ["AIVC_STORAGE_ROOT"]
            
        with patch("sys.argv", ["aivc", "status"]):
            main()
    
    captured = capsys.readouterr()
    assert "No files are currently tracked" in captured.out


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
