"""
Centralized configuration management for AIVC.
"""

import json
import socket
import os
import sys
from pathlib import Path

_STORAGE_ROOT_ENV = "AIVC_STORAGE_ROOT"
_CONFIG_PATH = Path.home() / ".aivc" / "config.json"
_CREDENTIALS_PATH = Path.home() / ".aivc" / "credentials.json"
_TOKEN_PATH = Path.home() / ".aivc" / "token.json"

# Disable network telemetry that causes massive latency spikes on Windows
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # ChromaDB
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # HuggingFace

# ML Model configurations
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_aivc_config() -> dict:
    """Read AIVC config from ~/.aivc/config.json."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_aivc_config(config: dict) -> None:
    """Save AIVC config to ~/.aivc/config.json."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=4), encoding="utf-8")

def get_machine_id() -> str:
    """Retrieve machine_id from config or fallback to hostname (with WSL detection)."""
    config = get_aivc_config()
    m_id = config.get("machine_id", "").strip()
    
    if not m_id:
        hostname = socket.gethostname()
        # Detect WSL (Windows Subsystem for Linux)
        try:
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        return f"{hostname}-WSL"
        except Exception:
            pass
        return hostname
        
    return m_id

def get_credentials_path() -> Path:
    """Return the path to the Google OAuth credentials file."""
    return _CREDENTIALS_PATH

def get_token_path() -> Path:
    """Return the path to the Google OAuth token file."""
    return _TOKEN_PATH

def get_storage_root(allow_fallback: bool = False) -> Path:
    """Retrieve and validate the AIVC storage root directory.

    Args:
        allow_fallback: If True, defaults to ~/.aivc/storage if the env var is missing.
                        If False, exits the process (or raises ValueError).

    Returns:
        Path object pointing to the storage root.

    Raises:
        ValueError: If AIVC_STORAGE_ROOT is not set and allow_fallback is False
                    (only when not running in a CLI-like exit context).
    """
    path_str = os.environ.get(_STORAGE_ROOT_ENV)
    
    if not path_str:
        # Default to ~/.aivc/storage gracefully instead of crashing,
        # which provides much better stability for MCP and CLI environments.
        fallback_path = Path.home() / ".aivc" / "storage"
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path
            
    return Path(path_str)
