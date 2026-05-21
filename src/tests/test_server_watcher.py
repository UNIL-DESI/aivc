"""Unit tests for the MCP Server Watcher logic."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["AIVC_STORAGE_ROOT"] = "/tmp/aivc_mock_root"
from aivc.server import AIVCWatcherHandler, start_background_watchers

def test_watcher_handler_ignores_hidden_files():
    mock_engine = MagicMock()
    
    # Use OS-native paths to prevent slash mismatch on Windows
    proj_dir = str(Path("/home/user/project").resolve())
    main_file = str(Path("/home/user/project/src/main.py").resolve())
    secret_file = str(Path("/home/user/project/src/.secret").resolve())
    git_config = str(Path("/home/user/project/.git/config").resolve())
    
    mock_engine.get_watched_dirs.return_value = {proj_dir: {"ignores": []}}
    handler = AIVCWatcherHandler(mock_engine, proj_dir)
    
    # Visible file
    event = MagicMock()
    event.is_directory = False
    event.src_path = main_file
    handler.on_created(event)
    mock_engine.track.assert_called_with(main_file)
    
    # Hidden file
    mock_engine.reset_mock()
    event.src_path = secret_file
    handler.on_created(event)
    mock_engine.track.assert_not_called()
    
    # File in hidden dir
    mock_engine.reset_mock()
    event.src_path = git_config
    handler.on_created(event)
    mock_engine.track.assert_not_called()

@patch("aivc.server._get_engine")
@patch("aivc.server._WATCHDOG_AVAILABLE", True)
@patch("aivc.server.Observer", create=True)
@patch("os.path.isdir")
def test_start_background_watchers(mock_isdir, mock_observer_cls, mock_get_engine):
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine
    
    mock_isdir.return_value = True
    watch_path = str(Path("/path/to/watch").resolve())
    mock_engine.get_watched_dirs.return_value = {
        watch_path: {"ignores": []}
    }
    
    mock_observer = MagicMock()
    mock_observer_cls.return_value = mock_observer
    
    start_background_watchers()
    
    # Startup sync called
    mock_engine.track.assert_called_with(watch_path)
    # Observer scheduled
    mock_observer.schedule.assert_called()
    # Observer started
    mock_observer.start.assert_called()
