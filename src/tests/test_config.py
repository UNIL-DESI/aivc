"""
Tests for the config module.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aivc.config import (
    get_aivc_config,
    save_aivc_config,
    get_machine_id,
    get_storage_root,
    get_credentials_path,
    get_token_path
)

@pytest.fixture
def temp_config_dir(tmp_path):
    config_dir = tmp_path / ".aivc"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    credentials_path = config_dir / "credentials.json"
    token_path = config_dir / "token.json"

    with patch("aivc.config._CONFIG_PATH", config_path), \
         patch("aivc.config._CREDENTIALS_PATH", credentials_path), \
         patch("aivc.config._TOKEN_PATH", token_path):
        yield config_dir

def test_get_aivc_config_no_file(temp_config_dir):
    # If the file doesn't exist, it should return an empty dict
    assert get_aivc_config() == {}

def test_get_aivc_config_valid_json(temp_config_dir):
    config_path = temp_config_dir / "config.json"
    data = {"machine_id": "test-machine"}
    config_path.write_text(json.dumps(data), encoding="utf-8")

    assert get_aivc_config() == data

def test_get_aivc_config_malformed_json(temp_config_dir):
    config_path = temp_config_dir / "config.json"
    config_path.write_text("invalid json {", encoding="utf-8")

    # It should return an empty dict on Exception
    assert get_aivc_config() == {}

def test_save_aivc_config(temp_config_dir):
    config_path = temp_config_dir / "config.json"
    data = {"machine_id": "new-machine"}

    save_aivc_config(data)

    assert config_path.exists()
    assert json.loads(config_path.read_text(encoding="utf-8")) == data

def test_get_machine_id_from_config(temp_config_dir):
    config_path = temp_config_dir / "config.json"
    data = {"machine_id": "configured-id"}
    config_path.write_text(json.dumps(data), encoding="utf-8")

    assert get_machine_id() == "configured-id"

def test_get_machine_id_fallback_to_hostname(temp_config_dir):
    # Ensure config is empty
    with patch("socket.gethostname", return_value="test-host"):
        with patch("os.path.exists", return_value=False): # Mock away WSL check
            assert get_machine_id() == "test-host"

def test_get_credentials_path(temp_config_dir):
    path = get_credentials_path()
    assert path == temp_config_dir / "credentials.json"

def test_get_token_path(temp_config_dir):
    path = get_token_path()
    assert path == temp_config_dir / "token.json"

def test_get_storage_root_env_set():
    with patch.dict(os.environ, {"AIVC_STORAGE_ROOT": "/tmp/aivc_root"}):
        assert get_storage_root() == Path("/tmp/aivc_root")

def test_get_storage_root_fallback_allowed():
    expected_path = Path.home() / ".aivc" / "storage"
    env_copy = os.environ.copy()
    env_copy.pop("AIVC_STORAGE_ROOT", None)
    with patch.dict(os.environ, env_copy, clear=True):
        assert get_storage_root(allow_fallback=True) == expected_path

def test_get_storage_root_no_fallback_exits():
    env_copy = os.environ.copy()
    env_copy.pop("AIVC_STORAGE_ROOT", None)
    with patch.dict(os.environ, env_copy, clear=True):
        with pytest.raises(SystemExit) as excinfo:
            get_storage_root(allow_fallback=False)

        assert "Environment variable 'AIVC_STORAGE_ROOT' is not set" in str(excinfo.value)
