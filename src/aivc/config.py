"""
Centralized configuration management for AIVC.
"""

import json
import socket
import os
import sys
import re
from pathlib import Path

def get_home_dir() -> Path:
    try:
        return Path.home()
    except RuntimeError:
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            return Path(userprofile)
        home = os.environ.get("HOME")
        if home:
            return Path(home)
        homedrive = os.environ.get("HOMEDRIVE")
        homepath = os.environ.get("HOMEPATH")
        return Path(r"C:\Users\Jamet")


def get_aivc_temp_dir() -> Path:
    """Returns the base temp directory for AIVC runtime files (<temp>/aivc)."""
    env_dir = os.environ.get("AIVC_TEMP_DIR")
    if env_dir:
        return Path(env_dir)
    import tempfile
    return Path(tempfile.gettempdir()) / "aivc"


_STORAGE_ROOT_ENV = "AIVC_STORAGE_ROOT"
_CONFIG_PATH = get_home_dir() / ".aivc" / "config.json"
def get_aivc_dir() -> Path:
    return get_home_dir() / ".aivc"

_CREDENTIALS_PATH = get_aivc_dir() / "credentials.json"
_TOKEN_PATH = get_aivc_dir() / "token.json"


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
        fallback_path = get_home_dir() / ".aivc" / "storage"
        fallback_path.mkdir(parents=True, exist_ok=True)
        return fallback_path
            
    return Path(path_str)


def to_portable_path(path_str: str | Path) -> str:
    """Normalize a local filesystem path across different machines/OSes.

    Converts Windows/macOS/Linux user home directory paths to '~/...'.
    """
    if isinstance(path_str, Path):
        s = path_str.as_posix()
    else:
        s = str(path_str).strip().replace("\\", "/")

    # Strip file URI prefix if present
    if s.startswith("file:///"):
        s = s[8:]
    elif s.startswith("file://"):
        s = s[7:]

    # Strip trailing line numbers e.g. :L10-L20 if present
    if ":L" in s or re.search(r":\d+", s):
        s = re.split(r":L?\d+", s)[0]

    # If s contains a segment '/~/' (e.g. from concatenation with workspace root), extract the '~/' part
    if "/~/" in s:
        idx = s.rfind("/~/")
        s = s[idx + 1:]

    # Fix potential missing separator before .gemini: e.g. Users/alice.gemini -> Users/alice/.gemini
    s = re.sub(r"(/Users/[^/]+)\.(gemini)", r"\1/.\2", s, flags=re.IGNORECASE)
    s = re.sub(r"(/home/[^/]+)\.(gemini)", r"\1/.\2", s, flags=re.IGNORECASE)

    # Check for temporary aivc artifacts path
    sub_art = extract_brain_subpath(s)
    if sub_art and "aivc/artifacts/" in s.lower():
        return f"~/.gemini/antigravity/brain/{sub_art}"

    if s.startswith("~/"):
        return s
    if s == "~":
        return "~"

    # Try local home resolution first
    try:
        resolved = Path(path_str).resolve()
        home = get_home_dir().resolve()
        if resolved == home:
            return "~"
        try:
            rel = resolved.relative_to(home)
            return f"~/{rel.as_posix()}"
        except (ValueError, RuntimeError):
            pass
    except Exception:
        pass

    patterns = [
        r"^[A-Za-z]:/Users/[^/]+/(.*)$",
        r"^/?Users/[^/]+/(.*)$",
        r"^/?home/[^/]+/(.*)$",
    ]
    for pattern in patterns:
        m = re.match(pattern, s, re.IGNORECASE)
        if m:
            return f"~/{m.group(1)}"

    exact_patterns = [
        r"^[A-Za-z]:/Users/[^/]+/?$",
        r"^/?Users/[^/]+/?$",
        r"^/?home/[^/]+/?$",
    ]
    for pattern in exact_patterns:
        if re.match(pattern, s, re.IGNORECASE):
            return "~"

    return s


def extract_brain_subpath(raw_path: str | Path) -> str | None:
    """Extract relative path after .gemini/antigravity/brain/ from any path, URI, or link.

    Example:
        'file:///C:/Users/other/.gemini/antigravity/brain/conv-1/doc.md' -> 'conv-1/doc.md'
        'file:///C:/Users/alice/AppData/Local/Temp/aivc/artifacts/conv-1/doc.md' -> 'conv-1/doc.md'
    """
    if not raw_path:
        return None
    s = str(raw_path).strip().replace("\\", "/")
    if s.startswith("file:///"):
        s = s[8:]
    elif s.startswith("file://"):
        s = s[7:]

    # Strip line numbers
    if ":L" in s or re.search(r":\d+", s):
        s = re.split(r":L?\d+", s)[0]

    for marker in (".gemini/antigravity-cli/brain/", ".gemini/antigravity/brain/", "aivc/artifacts/"):
        idx = s.lower().find(marker)
        if idx != -1:
            rel = s[idx + len(marker):].strip().strip("/")
            return rel if rel else None
    return None



def from_portable_path(path_str: str | Path) -> Path:
    """Convert a portable path ('~/...') or foreign machine path to a local Path."""
    if isinstance(path_str, str):
        import re
        s_clean = path_str.replace("\\", "/")
        if "/~/" in s_clean:
            idx = s_clean.rfind("/~/")
            path_str = s_clean[idx + 1:]
        path_str = re.sub(r"([A-Za-z]:[\\/]Users[\\/][^\\/]+)\.(gemini)", r"\1\\.\2", str(path_str), flags=re.IGNORECASE)
        path_str = re.sub(r"(/Users/[^/]+)\.(gemini)", r"\1/.\2", str(path_str), flags=re.IGNORECASE)
    elif isinstance(path_str, Path):
        s_clean = path_str.as_posix()
        if "/~/" in s_clean:
            idx = s_clean.rfind("/~/")
            path_str = s_clean[idx + 1:]
    portable = to_portable_path(path_str)
    home = get_home_dir()
    if portable.startswith("~/"):
        rel_part = portable[2:].lstrip("/\\")
        return home / Path(rel_part)
    if portable == "~":
        return home
    return Path(portable)


def get_workspace_dir() -> Path:
    """Returns the active workspace or project vault directory.

    Checks AIVC_WORKSPACE_DIR, then PROJECT_MEMORY_VAULT_DIR.
    Otherwise scans direct subdirectories of ~/Documents for a project/notes workspace
    (e.g. containing 'notes' or '_agents'), falling back to ~/Documents.
    """
    env_ws = os.environ.get("AIVC_WORKSPACE_DIR") or os.environ.get("PROJECT_MEMORY_VAULT_DIR")
    if env_ws and str(env_ws).strip():
        return Path(env_ws).resolve()

    home = get_home_dir()
    docs = home / "Documents"
    if docs.is_dir():
        try:
            for sub in docs.iterdir():
                if sub.is_dir():
                    if (sub / "notes").is_dir() or (sub / "_agents").is_dir():
                        return sub.resolve()
        except Exception:
            pass
        return docs.resolve()
    return home.resolve()


def resolve_workspace_path(rel_path: str | Path, cwd: str | Path | None = None) -> Path | None:
    """Resolves a relative path against cwd, workspace root, or subdirectories of ~/Documents."""
    if not rel_path:
        return None
    raw_s = str(rel_path).strip().replace("\\", "/")
    if not raw_s:
        return None

    # If already absolute or portable home path
    if raw_s.startswith(("~/", "~\\")):
        p = from_portable_path(raw_s)
        if p.exists():
            return p.resolve()
        return p

    p_raw = Path(rel_path)
    if p_raw.is_absolute():
        return p_raw.resolve() if p_raw.exists() else p_raw

    rel_norm = raw_s.lstrip("./")

    # 1. Search in cwd if provided
    if cwd:
        try:
            cwd_path = Path(cwd)
            if cwd_path.is_dir() and (cwd_path / rel_norm).exists():
                return (cwd_path / rel_norm).resolve()
        except Exception:
            pass

    # 2. Search in workspace dir
    ws_dir = get_workspace_dir()
    if (ws_dir / rel_norm).exists():
        return (ws_dir / rel_norm).resolve()

    # 3. Search in direct subdirectories of ~/Documents
    docs = get_home_dir() / "Documents"
    if docs.is_dir():
        try:
            for sub in docs.iterdir():
                if sub.is_dir():
                    cand = sub / rel_norm
                    if cand.exists():
                        return cand.resolve()
        except Exception:
            pass

    return None


