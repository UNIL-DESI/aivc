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
import shutil
import tempfile
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


def extract_obsidian_project_notes(memory: Any) -> list[str]:
    """Extracts Obsidian project note paths from a memory.

    If main_project_file is present, uses it exclusively.
    Otherwise scans read_files, edited_files, and changes for markdown notes.
    """
    if isinstance(memory, dict):
        mpf = memory.get("main_project_file") or ""
        read_files = memory.get("read_files") or []
        edited_files = memory.get("edited_files") or []
        changes = memory.get("changes") or []
    else:
        mpf = getattr(memory, "main_project_file", "") or ""
        read_files = getattr(memory, "read_files", []) or []
        edited_files = getattr(memory, "edited_files", []) or []
        changes = getattr(memory, "changes", []) or []

    if mpf and str(mpf).strip():
        return [str(mpf).strip()]

    notes: list[str] = []
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
            if str(p) not in notes:
                notes.append(str(p))

    return notes


def is_obsidian_related_memory(memory: Any) -> bool:
    """Determines if a memory relates to the Obsidian vault or contains task items."""
    if isinstance(memory, dict):
        mpf = memory.get("main_project_file") or ""
        user_items = memory.get("user_items") or memory.get("tasks") or []
        title = memory.get("title", "")
    else:
        mpf = getattr(memory, "main_project_file", "") or ""
        user_items = getattr(memory, "user_items", []) or []
        title = getattr(memory, "title", "")

    # If main_project_file is present, inspect it exclusively
    if mpf and str(mpf).strip():
        p_clean = str(mpf).strip().replace("\\", "/").lower()
        if p_clean.endswith(".md") or "voicenotes" in p_clean:
            return True

    # If tasks or user_items are present, synchronization is eligible
    if user_items:
        return True

    # Check extracted notes
    notes = extract_obsidian_project_notes(memory)
    if notes:
        for p in notes:
            p_clean = str(p).strip().replace("\\", "/").lower()
            if p_clean.endswith(".md") or "voicenotes" in p_clean:
                return True

    # Check project title pattern [Project Title]
    if title and re.search(r"\[[^\]]+\]", title):
        return True

    return False


def normalize_advance_tldr_title(title: str) -> str:
    """Normalizes project title for advance_tldr notes by stripping leading robot emojis."""
    return re.sub(r"^(?:🤖\s*)+", "", title).strip()


def restore_brain_files_for_memory(memory: Any, sync_manager: Any = None) -> list[str]:
    """Auto-restores files under .gemini/antigravity/brain/ referenced by memory if missing locally.
    Remaps any foreign machine paths to current PC's Path.home() / .gemini/antigravity/brain/...
    Supports memory objects, dicts, file paths, or advance_tldr notes.
    """
    import urllib.parse
    from aivc.config import get_home_dir, extract_brain_subpath, get_storage_root, get_aivc_temp_dir
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

    brain_regex = re.compile(
        r"(?:file:///?(?:[A-Za-z]:)?[\\/]|~/|[A-Za-z]:[\\/]|/)?(?:Users|home)?[^\"\'\`\s\)\>]*?(?:\.?gemini[\\/]antigravity(?:-cli)?[\\/]brain|aivc[\\/]artifacts)[\\/]([^\s\"\'\`\)\>:]+)",
        re.IGNORECASE,
    )

    if isinstance(memory, (str, Path)):
        p_mem = Path(memory)
        if p_mem.is_file():
            try:
                txt = p_mem.read_text(encoding="utf-8")
                for m in brain_regex.finditer(txt):
                    raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw:
                        candidate_subpaths.add(raw)
            except Exception:
                pass
        else:
            for m in brain_regex.finditer(str(memory)):
                raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                if raw:
                    candidate_subpaths.add(raw)
    else:
        if isinstance(memory, dict):
            changes = memory.get("changes", []) or []
            edited_files = memory.get("edited_files", []) or []
            read_files = memory.get("read_files", []) or []
            user_items = memory.get("user_items", []) or memory.get("tasks", []) or []
            texts = [memory.get("journal", ""), memory.get("note", "")]
            mpf = memory.get("main_project_file", "")
        else:
            changes = getattr(memory, "changes", []) or []
            edited_files = getattr(memory, "edited_files", []) or []
            read_files = getattr(memory, "read_files", []) or []
            user_items = getattr(memory, "user_items", []) or []
            texts = [getattr(memory, "journal", ""), getattr(memory, "note", "")]
            mpf = getattr(memory, "main_project_file", "")

        for c in changes:
            p = getattr(c, "path", None) if hasattr(c, "path") else (c.get("path") if isinstance(c, dict) else str(c))
            sub = extract_brain_subpath(p)
            if sub:
                candidate_subpaths.add(urllib.parse.unquote(sub))

        for p in edited_files + read_files:
            p_val = p.get("path") if isinstance(p, dict) else str(p)
            sub = extract_brain_subpath(p_val)
            if sub:
                candidate_subpaths.add(urllib.parse.unquote(sub))

        for item in user_items:
            if isinstance(item, dict):
                t = f"{item.get('task', '')} {item.get('details', '')}"
                for m in brain_regex.finditer(t):
                    raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw:
                        candidate_subpaths.add(raw)

        for txt in texts:
            if txt:
                for m in brain_regex.finditer(txt):
                    raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw:
                        candidate_subpaths.add(raw)

        if mpf:
            try:
                stem = Path(str(mpf)).stem
                title = normalize_advance_tldr_title(stem)
                tldr_temp = get_aivc_temp_dir() / "advance_tldr" / f"🤖 {title}.md"
                tldr_legacy = home / ".gemini" / "antigravity" / "brain" / "advance_tldr" / f"🤖 {title}.md"
                if home != Path.home() and tldr_legacy.is_file():
                    tldr_candidate = tldr_legacy
                elif tldr_temp.is_file():
                    tldr_candidate = tldr_temp
                else:
                    tldr_candidate = tldr_legacy
                if tldr_candidate.is_file():
                    tldr_txt = tldr_candidate.read_text(encoding="utf-8")
                    for m in brain_regex.finditer(tldr_txt):
                        raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                        if raw:
                            candidate_subpaths.add(raw)
            except Exception:
                pass

    temp_artifacts_dir = get_aivc_temp_dir() / "artifacts"
    for sub in candidate_subpaths:
        local_path = home / ".gemini" / "antigravity" / "brain" / Path(sub)
        temp_artifact_path = temp_artifacts_dir / Path(sub)

        # Mirror to temp_artifacts_dir if local_path already exists
        if local_path.is_file() and not temp_artifact_path.is_file():
            try:
                temp_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, temp_artifact_path)
            except OSError:
                pass

        if local_path.is_file():
            continue
        sub_posix = sub.replace("\\", "/")
        portable = f"~/.gemini/antigravity/brain/{sub_posix}"
        try:
            dest = mgr.pull_latest_file(portable, target_path=local_path)
            if dest and dest.is_file():
                restored.append(str(local_path))
                try:
                    temp_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(local_path, temp_artifact_path)
                except OSError:
                    pass
        except Exception:
            pass

    return restored


def sync_advance_tldr(memory: Any, vault_dir: str | Path | None = None) -> list[str]:
    """Synchronizes advance_tldr/🤖 <Titre>.md for the given memory.

    Returns the list of generated or updated advance_tldr file paths.
    """
    is_dict = isinstance(memory, dict)
    if is_dict:
        if memory.get("_tldr_synced", False):
            return memory.get("_tldr_synced_paths", [])
    else:
        if getattr(memory, "_tldr_synced", False):
            return getattr(memory, "_tldr_synced_paths", [])

    # Restore missing brain files referenced by this memory or active tasks
    try:
        restore_brain_files_for_memory(memory)
    except Exception as e:
        logger.debug("auto-restore in sync_advance_tldr skipped: %s", e)

    if _CUSTOM_SYNC_HANDLER is not None:
        try:
            res = _CUSTOM_SYNC_HANDLER(memory, vault_dir=vault_dir)
            if is_dict:
                memory["_tldr_synced"] = True
                memory["_tldr_synced_paths"] = res or []
            else:
                setattr(memory, "_tldr_synced", True)
                setattr(memory, "_tldr_synced_paths", res or [])
            if res:
                for sp in res:
                    try:
                        restore_brain_files_for_memory(sp)
                    except Exception:
                        pass
            return res or []
        except Exception as e:
            logger.warning("Custom sync handler failed: %s", e)
            return []

    if not is_obsidian_related_memory(memory):
        return []

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

        # Restore missing brain files referenced in newly generated/updated advance_tldr files
        if synced_paths:
            for sp in synced_paths:
                try:
                    restore_brain_files_for_memory(sp)
                except Exception:
                    pass

        if is_dict:
            memory["_tldr_synced"] = True
            memory["_tldr_synced_paths"] = synced_paths or []
        else:
            setattr(memory, "_tldr_synced", True)
            setattr(memory, "_tldr_synced_paths", synced_paths or [])
        return synced_paths or []
    except Exception as e:
        mem_id = memory.get("id", "unknown") if is_dict else getattr(memory, "id", "unknown")
        logger.warning("Failed to synchronize advance_tldr for memory %s: %s", mem_id, e)
        return []
