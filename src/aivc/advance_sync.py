"""
Automatic synchronization of advance_tldr notes from AIVC remember events.
Integrates AIVC with project_memory_cli to keep advance_tldr/🤖 <Titre>.md updated.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from aivc.config import get_home_dir, to_portable_path

logger = logging.getLogger(__name__)

_PROJECT_MEMORY_MODULE: Any = None
_CUSTOM_SYNC_HANDLER: Any = None


def set_custom_sync_handler(handler: Any) -> None:
    """Allows tests or extensions to provide a custom synchronization handler."""
    global _CUSTOM_SYNC_HANDLER
    _CUSTOM_SYNC_HANDLER = handler


def get_vault_dir() -> Path:
    """Returns the Obsidian vault directory (VoiceNotes)."""
    env_vault = os.environ.get("PROJECT_MEMORY_VAULT_DIR")
    if env_vault:
        return Path(env_vault)
    home = get_home_dir()
    candidate = home / "Documents" / "VoiceNotes"
    return candidate


def find_project_memory_cli_path() -> Path | None:
    """Locates project_memory_cli.py script."""
    env_cli = os.environ.get("PROJECT_MEMORY_CLI_PATH")
    if env_cli and Path(env_cli).is_file():
        return Path(env_cli)

    vault = get_vault_dir()
    candidate = vault / "_agents" / "scripts-for-skills" / "project_memory_cli.py"
    if candidate.is_file():
        return candidate

    return None


def _load_project_memory_module() -> Any:
    """Loads and caches the project_memory_cli module."""
    global _PROJECT_MEMORY_MODULE
    if _PROJECT_MEMORY_MODULE is not None:
        return _PROJECT_MEMORY_MODULE

    if "project_memory_cli" in sys.modules:
        _PROJECT_MEMORY_MODULE = sys.modules["project_memory_cli"]
        return _PROJECT_MEMORY_MODULE

    cli_path = find_project_memory_cli_path()
    if not cli_path or not cli_path.is_file():
        return None

    try:
        spec = importlib.util.spec_from_file_location("project_memory_cli", str(cli_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _PROJECT_MEMORY_MODULE = mod
            return mod
    except Exception as e:
        logger.warning("Failed to load project_memory_cli from %s: %s", cli_path, e)

    return None


def is_obsidian_related_memory(memory: Any) -> bool:
    """Determines if a memory relates to the Obsidian vault or contains task items."""
    if isinstance(memory, dict):
        user_items = memory.get("user_items") or memory.get("tasks") or []
        read_files = memory.get("read_files") or []
        edited_files = memory.get("edited_files") or []
        changes = memory.get("changes") or []
        title = memory.get("title", "")
    else:
        user_items = getattr(memory, "user_items", []) or []
        read_files = getattr(memory, "read_files", []) or []
        edited_files = getattr(memory, "edited_files", []) or []
        changes = getattr(memory, "changes", []) or []
        title = getattr(memory, "title", "")

    # If tasks or user_items are present, synchronization is eligible
    if user_items:
        return True

    # Check for any .md file in read_files, edited_files or changes
    all_paths: list[str] = []
    for f in read_files:
        all_paths.append(f.get("path") if isinstance(f, dict) else str(f))
    for f in edited_files:
        all_paths.append(f.get("path") if isinstance(f, dict) else str(f))
    for c in changes:
        p = getattr(c, "path", None) if hasattr(c, "path") else (c.get("path") if isinstance(c, dict) else str(c))
        if p:
            all_paths.append(p)

    for p in all_paths:
        p_clean = str(p).strip().replace("\\", "/").lower()
        if p_clean.endswith(".md") or "voicenotes" in p_clean:
            return True

    # Check project title pattern [Project Title]
    if title and re.search(r"\[[^\]]+\]", title):
        return True

    return False


def restore_brain_files_for_memory(memory: Any, sync_manager: Any = None) -> list[str]:
    """Auto-restores files under .gemini/antigravity/brain/ referenced by memory if missing locally.
    Remaps any foreign machine paths to current PC's Path.home() / .gemini/antigravity/brain/...
    """
    from aivc.config import get_home_dir, extract_brain_subpath, get_storage_root
    from aivc.sync.drive import NativeDriveSyncManager

    home = get_home_dir()
    restored: list[str] = []

    # Prepare sync_manager if not provided
    mgr = sync_manager
    if mgr is None:
        try:
            storage_root = get_storage_root(allow_fallback=True)
            mgr = NativeDriveSyncManager(storage_root)
        except Exception:
            mgr = None

    if mgr is None or not getattr(mgr, "enabled", False):
        return []

    # Collect subpaths under brain/
    candidate_subpaths: set[str] = set()

    if isinstance(memory, dict):
        changes = memory.get("changes", []) or []
        edited_files = memory.get("edited_files", []) or []
        read_files = memory.get("read_files", []) or []
        user_items = memory.get("user_items", []) or memory.get("tasks", []) or []
        texts = [memory.get("journal", ""), memory.get("note", "")]
    else:
        changes = getattr(memory, "changes", []) or []
        edited_files = getattr(memory, "edited_files", []) or []
        read_files = getattr(memory, "read_files", []) or []
        user_items = getattr(memory, "user_items", []) or []
        texts = [getattr(memory, "journal", ""), getattr(memory, "note", "")]

    for c in changes:
        p = getattr(c, "path", None) if hasattr(c, "path") else (c.get("path") if isinstance(c, dict) else str(c))
        sub = extract_brain_subpath(p)
        if sub:
            candidate_subpaths.add(sub)

    for p in edited_files + read_files:
        p_val = p.get("path") if isinstance(p, dict) else str(p)
        sub = extract_brain_subpath(p_val)
        if sub:
            candidate_subpaths.add(sub)

    brain_regex = re.compile(
        r"(?:file:///?(?:[A-Za-z]:)?[\\/]|~/|[A-Za-z]:[\\/]|/)?(?:Users|home)?[^\"\'\`\s\)\>]*?\.?gemini[\\/]antigravity[\\/]brain[\\/]([^\s\"\'\`\)\>:]+)",
        re.IGNORECASE,
    )

    for item in user_items:
        if isinstance(item, dict):
            t = f"{item.get('task', '')} {item.get('details', '')}"
            for m in brain_regex.finditer(t):
                raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                if raw:
                    candidate_subpaths.add(raw)

    for txt in texts:
        if txt:
            for m in brain_regex.finditer(txt):
                raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                if raw:
                    candidate_subpaths.add(raw)

    for sub in candidate_subpaths:
        local_path = home / ".gemini" / "antigravity" / "brain" / Path(sub)
        if local_path.is_file():
            continue
        sub_posix = sub.replace("\\", "/")
        portable = f"~/.gemini/antigravity/brain/{sub_posix}"
        try:
            dest = mgr.pull_latest_file(portable, target_path=local_path)
            if dest and dest.is_file():
                restored.append(str(local_path))
        except Exception:
            pass

    return restored


def sync_advance_tldr(memory: Any, vault_dir: str | Path | None = None) -> list[str]:
    """Synchronizes advance_tldr/🤖 <Titre>.md for the given memory.

    Returns the list of generated or updated advance_tldr file paths.
    """
    if getattr(memory, "_tldr_synced", False):
        return getattr(memory, "_tldr_synced_paths", [])

    if _CUSTOM_SYNC_HANDLER is not None:
        try:
            res = _CUSTOM_SYNC_HANDLER(memory, vault_dir=vault_dir)
            setattr(memory, "_tldr_synced", True)
            setattr(memory, "_tldr_synced_paths", res or [])
            return res or []
        except Exception as e:
            logger.warning("Custom sync handler failed: %s", e)
            return []

    if not is_obsidian_related_memory(memory):
        return []

    # Restore missing brain files referenced by this memory or active tasks
    try:
        restore_brain_files_for_memory(memory)
    except Exception as e:
        logger.debug("auto-restore in sync_advance_tldr skipped: %s", e)



    mod = _load_project_memory_module()
    if mod is None:
        logger.debug("project_memory_cli module not available; skipping advance_tldr sync.")
        return []

    try:
        sync_fn = getattr(mod, "sync_advance_tldr_for_memory", None)
        if sync_fn is None:
            logger.warning("sync_advance_tldr_for_memory function not found in project_memory_cli.")
            return []

        vault_str = str(vault_dir) if vault_dir else str(get_vault_dir())
        synced_paths = sync_fn(memory, vault_dir=vault_str)
        setattr(memory, "_tldr_synced", True)
        setattr(memory, "_tldr_synced_paths", synced_paths or [])
        return synced_paths or []
    except Exception as e:
        logger.warning("Failed to synchronize advance_tldr for memory %s: %s", getattr(memory, "id", "unknown"), e)
        return []
