"""
Automatic synchronization of project notes and views from AIVC task_commit events.
Integrates AIVC with external project sync handlers to keep project views updated.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aivc.config import get_home_dir, to_portable_path, get_aivc_temp_dir, get_aivc_user_temp_dir

logger = logging.getLogger(__name__)

_PROJECT_MEMORY_MODULE: Any = None
_CUSTOM_SYNC_HANDLER: Any = None


def set_custom_sync_handler(handler: Any) -> None:
    """Allows tests or extensions to provide a custom synchronization handler."""
    global _CUSTOM_SYNC_HANDLER
    _CUSTOM_SYNC_HANDLER = handler


def get_workspace_dir() -> Path:
    """Returns the project workspace or vault directory."""
    from aivc.config import get_workspace_dir as _get_ws_dir
    return _get_ws_dir()


get_vault_dir = get_workspace_dir


def find_project_sync_cli_path() -> Path | None:
    """Locates project sync CLI script."""
    env_cli = os.environ.get("AIVC_PROJECT_SYNC_CLI_PATH") or os.environ.get("PROJECT_MEMORY_CLI_PATH")
    if env_cli and Path(env_cli).is_file():
        return Path(env_cli)

    ws_dir = get_workspace_dir()
    candidate = ws_dir / "_agents" / "scripts-for-skills" / "project_memory_cli.py"
    if candidate.is_file():
        return candidate

    return None


find_project_memory_cli_path = find_project_sync_cli_path


def _load_project_memory_module() -> Any:
    """Loads and caches the project sync module."""
    global _PROJECT_MEMORY_MODULE
    if _PROJECT_MEMORY_MODULE is not None:
        return _PROJECT_MEMORY_MODULE

    if "project_memory_cli" in sys.modules:
        _PROJECT_MEMORY_MODULE = sys.modules["project_memory_cli"]
        return _PROJECT_MEMORY_MODULE

    cli_path = find_project_sync_cli_path()
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


def extract_project_notes(memory: Any) -> list[str]:
    """Extracts project note paths from a memory.

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
        if p_clean.endswith(".md"):
            # Exclude temporary artifact directories
            if not any(k in p_clean for k in ("/brain/", "/temp/", "appdata", "scratch", ".aivc/artifacts")):
                if str(p) not in notes:
                    notes.append(str(p))

    return notes


extract_obsidian_project_notes = extract_project_notes


def is_project_related_memory(memory: Any) -> bool:
    """Determines if a memory relates to a project workspace or contains task items."""
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
        if p_clean.endswith(".md") and not any(k in p_clean for k in ("/brain/", "/temp/", "appdata", "scratch", ".aivc/artifacts")):
            return True

    # If tasks or user_items are present, synchronization is eligible
    if user_items:
        return True

    # Check extracted notes
    notes = extract_project_notes(memory)
    if notes:
        return True

    # Check project title pattern [Project Title]
    if title and re.search(r"\[[^\]]+\]", title):
        return True

    return False


is_obsidian_related_memory = is_project_related_memory


def normalize_project_tldr_title(title: str) -> str:
    """Normalizes project title for project view notes by stripping leading robot emojis."""
    return re.sub(r"^(?:🤖\s*)+", "", title).strip()


normalize_advance_tldr_title = normalize_project_tldr_title



def is_corrupt_test_artifact(path: Path) -> bool:
    """Detects 8-byte dummy 'restored' artifact written by unit tests."""
    try:
        if path.is_file() and path.stat().st_size == 8:
            return path.read_bytes() == b"restored"
    except Exception:
        pass
    return False


def convert_wikilinks_to_markdown_links(text: str, vault_dir: Path | str | None = None) -> str:
    """Convert Obsidian wikilinks [[target|label]] or [[target]] to standard markdown links.
    Ensures no [[...]] syntax remains in advance_tldr notes.
    """
    if not text or "[[" not in text:
        return text

    wikilink_pattern = re.compile(r"\[\[([^\]|\r\n]+)(?:\|([^\]\r\n]+))?\]\]")

    def _replace_wikilink(m: re.Match) -> str:
        target = m.group(1).strip()
        label = m.group(2).strip() if m.group(2) else None

        # 1. Already a file:/// URL
        if target.lower().startswith("file:///"):
            t_url = target.replace("\\", "/")
            lbl = label or Path(t_url).name
            return f"[{lbl}]({t_url})"

        # 2. Windows drive letter path e.g. C:\... or C:/...
        if re.match(r"^[A-Za-z]:[\\/]", target):
            posix_path = target.replace("\\", "/").rstrip("/")
            lbl = label or Path(target).name
            return f"[{lbl}](file:///{posix_path})"

        # 3. Portable home path ~/ or ~\
        if target.startswith("~/") or target.startswith("~\\"):
            posix_home = Path.home().as_posix()
            sub = target[2:].replace("\\", "/").lstrip("/")
            lbl = label or Path(target).name
            return f"[{lbl}](file:///{posix_home}/{sub})"

        # 4. Relative vault path
        base_v = Path(vault_dir) if vault_dir else get_workspace_dir()
        lbl = label or Path(target).stem
        clean_target = target.replace("\\", "/").lstrip("/")

        cand_direct = base_v / clean_target
        if cand_direct.exists():
            return f"[{lbl}](file:///{cand_direct.as_posix()})"

        if not clean_target.lower().endswith(".md"):
            cand_md = base_v / f"{clean_target}.md"
            if cand_md.exists():
                return f"[{lbl}](file:///{cand_md.as_posix()})"

            cand_notes_md = base_v / "notes" / f"{clean_target}.md"
            if cand_notes_md.exists():
                return f"[{lbl}](file:///{cand_notes_md.as_posix()})"

        cand_notes = base_v / "notes" / clean_target
        if cand_notes.exists():
            return f"[{lbl}](file:///{cand_notes.as_posix()})"

        target_file = f"{clean_target}.md" if not clean_target.lower().endswith(".md") else clean_target
        return f"[{lbl}](file:///{base_v.as_posix()}/{target_file})"

    return wikilink_pattern.sub(_replace_wikilink, text)


def restore_artifacts_for_memory(memory: Any, sync_manager: Any = None, background_drive: bool = False) -> list[str]:
    """Auto-restores files under .gemini/antigravity/brain/ referenced by memory if missing locally.
    Remaps any foreign machine paths to current PC's Path.home() / .gemini/antigravity/brain/...
    Supports memory objects, dicts, file paths, or project view notes.
    """
    import urllib.parse
    from aivc.config import get_home_dir, extract_brain_subpath, get_storage_root, get_aivc_temp_dir, get_aivc_user_temp_dir
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
                title = normalize_project_tldr_title(stem)
                tldr_temp = get_aivc_temp_dir() / "advance_tldr" / f"🤖 {title}.md"
                tldr_user_temp = get_aivc_user_temp_dir() / "advance_tldr" / f"🤖 {title}.md"
                tldr_legacy = home / ".gemini" / "antigravity" / "brain" / "advance_tldr" / f"🤖 {title}.md"
                tldr_cli = home / ".gemini" / "antigravity-cli" / "brain" / "advance_tldr" / f"🤖 {title}.md"
                tldr_files = [p for p in (tldr_temp, tldr_user_temp, tldr_legacy, tldr_cli) if p.is_file()]
                tldr_candidate = max(tldr_files, key=lambda p: p.stat().st_mtime) if tldr_files else tldr_legacy
                if tldr_candidate.is_file():
                    tldr_txt = tldr_candidate.read_text(encoding="utf-8")
                    for m in brain_regex.finditer(tldr_txt):
                        raw = urllib.parse.unquote(m.group(1)).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                        if raw:
                            candidate_subpaths.add(raw)
            except Exception:
                pass

    temp_artifacts_dir = get_aivc_temp_dir() / "artifacts"
    user_temp_artifacts_dir = get_aivc_user_temp_dir() / "artifacts"
    mirror_user_temp = (user_temp_artifacts_dir.resolve() != temp_artifacts_dir.resolve())
    drive_candidates: list[tuple[str, Path, list[Path]]] = []

    brain_roots = [
        home / ".gemini" / "antigravity" / "brain",
        home / ".gemini" / "antigravity-cli" / "brain",
    ]

    for sub in candidate_subpaths:
        sub_p = Path(sub)
        local_path = brain_roots[0] / sub_p
        cli_path = brain_roots[1] / sub_p
        temp_artifact_path = temp_artifacts_dir / sub_p
        user_temp_artifact_path = user_temp_artifacts_dir / sub_p if mirror_user_temp else None

        local_files = [local_path, cli_path]
        brain_candidates = [p for p in local_files if p.is_file()]
        if brain_candidates:
            existing_file = max(brain_candidates, key=lambda p: p.stat().st_mtime)
        elif temp_artifact_path.is_file():
            existing_file = temp_artifact_path
        elif user_temp_artifact_path is not None and user_temp_artifact_path.is_file():
            existing_file = user_temp_artifact_path
        else:
            existing_file = None

        target_list = [p for p in (local_files + [temp_artifact_path, user_temp_artifact_path]) if p is not None]

        if existing_file:
            # Sync across local brain roots and temp_artifacts_dir
            mirrored_to_local_path = False
            for target in target_list:
                if target == existing_file:
                    continue
                needs_copy = False
                if not target.is_file():
                    needs_copy = True
                else:
                    try:
                        if target.stat().st_size != existing_file.stat().st_size:
                            needs_copy = True
                        elif hashlib.sha256(target.read_bytes()).hexdigest() != hashlib.sha256(existing_file.read_bytes()).hexdigest():
                            needs_copy = True
                    except OSError:
                        needs_copy = True

                if needs_copy:
                    try:
                        if target.parent.is_file():
                            target.parent.unlink(missing_ok=True)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(existing_file, target)
                        if target == local_path:
                            mirrored_to_local_path = True
                    except OSError:
                        pass
            if mirrored_to_local_path:
                restored.append(str(local_path))
            continue

        sub_posix = sub.replace("\\", "/")
        extra_targets = [p for p in [temp_artifact_path, cli_path, user_temp_artifact_path] if p is not None]
        drive_candidates.append((sub_posix, local_path, extra_targets))

    if drive_candidates:
        if background_drive:
            def _async_advance_drive_worker(candidates: list[tuple[str, Path, list[Path]]], d_mgr: Any) -> None:
                for sub_posix, l_p, targets in candidates:
                    for port_p in (
                        f"~/.gemini/antigravity-cli/brain/{sub_posix}",
                        f"~/.gemini/antigravity/brain/{sub_posix}",
                        f"~/AppData/Local/Temp/aivc/artifacts/{sub_posix}",
                    ):
                        try:
                            if l_p.parent.is_file():
                                l_p.parent.unlink(missing_ok=True)
                            dest = d_mgr.pull_latest_file(port_p, target_path=l_p)
                            if dest and dest.is_file():
                                for t in targets:
                                    try:
                                        if t.parent.is_file():
                                            t.parent.unlink(missing_ok=True)
                                        t.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(l_p, t)
                                    except OSError:
                                        pass
                                break
                        except FileNotFoundError:
                            pass

            import threading
            t = threading.Thread(
                target=_async_advance_drive_worker,
                args=(list(drive_candidates), mgr),
                daemon=True,
                name="AivcProjectSyncDrivePull",
            )
            t.start()
        else:
            for sub_posix, local_path, targets in drive_candidates:
                dest = None
                for port_p in (
                    f"~/.gemini/antigravity-cli/brain/{sub_posix}",
                    f"~/.gemini/antigravity/brain/{sub_posix}",
                    f"~/AppData/Local/Temp/aivc/artifacts/{sub_posix}",
                ):
                    try:
                        if local_path.parent.is_file():
                            local_path.parent.unlink(missing_ok=True)
                        dest = mgr.pull_latest_file(port_p, target_path=local_path)
                        if dest and dest.is_file():
                            break
                    except FileNotFoundError:
                        pass
                if dest and dest.is_file():
                    restored.append(str(local_path))
                    for t in targets:
                        try:
                            if t.parent.is_file():
                                t.parent.unlink(missing_ok=True)
                            t.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(local_path, t)
                        except OSError:
                            pass

    return restored


restore_brain_files_for_memory = restore_artifacts_for_memory


def strip_legacy_file_lists(text: str) -> str:
    """Removes legacy automated file list lines from markdown task views:
    - ✏️ **Fichiers modifiés** : ...
    - 📖 **Fichiers consultés** : ...
    """
    if not text:
        return text
    pattern = re.compile(
        r"(?m)^[ \t]*-[ \t]*[✏️📖][ \t]*\*\*(?:Fichiers modifiés|Fichiers consultés)\*\*[ \t]*:.*(?:\n[ \t]{2,}.*)*\n?",
        re.UNICODE,
    )
    return pattern.sub("", text)


def linkify_unlinked_artifact_names(text: str, file_paths: list[Any]) -> str:
    """Detects filenames from edited_files or read_files mentioned in plain text
    and converts them to clickable Markdown links [📄 <filename>](file:///...).
    Ignores filenames already part of markdown links or code blocks.
    """
    if not text or not file_paths:
        return text

    import aivc.config as _cfg

    base_temp = _cfg.get_aivc_temp_dir()
    artifacts_dir = Path(base_temp) / "artifacts"
    artifacts_posix = artifacts_dir.as_posix().replace("\\", "/").lstrip("/")

    # 1. Mask existing markdown links [label](url)
    md_link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    masked_links: list[str] = []

    def _mask_link(m: re.Match) -> str:
        idx = len(masked_links)
        masked_links.append(m.group(0))
        return f"__AIVC_MD_LINK_{idx}__"

    masked_text = md_link_pattern.sub(_mask_link, text)

    # 2. Extract valid distinct targets (filename -> local_file_url)
    candidates: dict[str, str] = {}
    for item in file_paths:
        if not item:
            continue
        p_str = item.get("path") if isinstance(item, dict) else str(item)
        p_clean = str(p_str).strip().replace("\\", "/")
        if not p_clean:
            continue

        fname = Path(p_clean).name
        if not fname or len(fname) < 3 or "." not in fname:
            continue

        # Determine target file URL
        sub = _cfg.extract_brain_subpath(p_clean)
        if sub:
            target_url = f"file:///{artifacts_posix}/{sub.replace('\\', '/').lstrip('/')}"
        else:
            loc_p = _cfg.from_portable_path(p_clean)
            target_url = f"file:///{loc_p.as_posix().replace('\\', '/').lstrip('/')}"

        # Avoid collision on same filename
        if fname not in candidates:
            candidates[fname] = target_url

    if not candidates:
        return text

    # Sort candidates by length descending to match longest filenames first
    sorted_fnames = sorted(candidates.keys(), key=len, reverse=True)

    # Set of lower-case links already present in text
    existing_links_lower = [link.lower() for link in masked_links]

    for fname in sorted_fnames:
        fname_lower = fname.lower()
        # If filename already appears inside an existing Markdown link, skip
        if any(fname_lower in lk for lk in existing_links_lower):
            continue

        target_url = candidates[fname]
        esc = re.escape(fname)

        # Match either `fname` (with backticks) or bare fname with word/punctuation boundaries
        # Case A: with backticks `fname`
        new_masked, n_subs = re.subn(
            rf"`{esc}`",
            f"[📄 {fname}]({target_url})",
            masked_text,
            count=1,
        )
        if n_subs > 0:
            masked_text = new_masked
        else:
            # Case B: bare fname not preceded or followed by alphanumeric / path characters or inside another link
            bare_pattern = re.compile(
                rf"(?<![A-Za-z0-9_/\\])(?<!__AIVC_MD_LINK_){esc}(?![A-Za-z0-9_/\\])(?!__)",
            )
            masked_text = bare_pattern.sub(f"[📄 {fname}]({target_url})", masked_text, count=1)

    # 3. Unmask original markdown links
    for idx, original_link in enumerate(masked_links):
        masked_text = masked_text.replace(f"__AIVC_MD_LINK_{idx}__", original_link)

    return masked_text


def process_user_items_details(memory: Any) -> None:
    """Pre-processes task details in memory in-place:
    - rewrites brain artifact URLs to Public temp
    - rewrites user home paths to current machine
    - linkifies unlinked filenames from edited_files and read_files
    - strips legacy automated file lists
    """
    if isinstance(memory, dict):
        user_items = memory.get("user_items", []) or memory.get("tasks", []) or []
        edited_files = memory.get("edited_files", []) or []
        read_files = memory.get("read_files", []) or []
    else:
        user_items = getattr(memory, "user_items", []) or []
        edited_files = getattr(memory, "edited_files", []) or []
        read_files = getattr(memory, "read_files", []) or []

    all_memory_files = list(edited_files) + list(read_files)

    for item in user_items:
        if not isinstance(item, dict):
            continue
        details = item.get("details", "")
        if not details or not isinstance(details, str):
            continue

        item_files = list(all_memory_files)
        item_files.extend(item.get("edited_files", []) or [])
        item_files.extend(item.get("read_files", []) or [])

        # 1. Rewrite brain paths & user paths
        d_repaired = rewrite_artifact_paths_to_temp(details)

        # 2. Linkify unlinked artifact filenames
        d_linkified = linkify_unlinked_artifact_names(d_repaired, item_files)

        # 3. Strip legacy lists
        d_clean = strip_legacy_file_lists(d_linkified)

        if d_clean != details:
            item["details"] = d_clean


def extract_referenced_files_from_memory(memory: Any) -> list[str]:
    """Extracts local files or brain artifacts referenced in memory content/details/journal
    that should be pushed to Google Drive during synchronization.
    """
    import urllib.parse
    from aivc.config import extract_brain_subpath, from_portable_path

    if isinstance(memory, dict):
        user_items = memory.get("user_items", []) or memory.get("tasks", []) or []
        texts = [memory.get("journal", ""), memory.get("note", ""), memory.get("content", "")]
        read_files = memory.get("read_files", []) or []
    else:
        user_items = getattr(memory, "user_items", []) or []
        texts = [getattr(memory, "journal", ""), getattr(memory, "note", ""), getattr(memory, "content", "")]
        read_files = getattr(memory, "read_files", []) or []

    for item in user_items:
        if isinstance(item, dict):
            texts.append(item.get("task", ""))
            texts.append(item.get("details", ""))

    referenced_paths: set[str] = set()

    # Pattern for URLs and file paths in text
    url_file_pattern = re.compile(
        r"file://+([^\s\)\"\'\`<>]+)",
        re.IGNORECASE,
    )
    bare_path_pattern = re.compile(
        r"(?:[A-Za-z]:[\\/]|/(?:Users|home)[\\/]|~[\\/])[^\s\[\]\(\)\"\'\`<>]*?(?:\.?gemini[\\/]antigravity(?:-cli)?[\\/]brain|aivc[\\/]artifacts)[\\/]([^\s\[\]\(\)\"\'\`<>:]+)",
        re.IGNORECASE,
    )

    for txt in texts:
        if not txt or not isinstance(txt, str):
            continue

        for m in url_file_pattern.finditer(txt):
            raw_url = m.group(1).split(":")[0].strip()
            decoded = urllib.parse.unquote(raw_url)
            # Remove leading slash before Windows drive letter if present
            if re.match(r"^/[A-Za-z]:", decoded):
                decoded = decoded[1:]
            p = Path(decoded)
            if p.is_file():
                referenced_paths.add(str(p.resolve()))
            else:
                p_port = from_portable_path(decoded)
                if p_port.is_file():
                    referenced_paths.add(str(p_port.resolve()))

        for m in bare_path_pattern.finditer(txt):
            raw_p = m.group(0).split(":")[0].strip()
            decoded = urllib.parse.unquote(raw_p)
            p = Path(decoded)
            if p.is_file():
                referenced_paths.add(str(p.resolve()))
            else:
                p_port = from_portable_path(decoded)
                if p_port.is_file():
                    referenced_paths.add(str(p_port.resolve()))

    # Also check read_files: if any file is under brain/, include it
    for rf in read_files:
        rf_str = rf.get("path") if isinstance(rf, dict) else str(rf)
        if extract_brain_subpath(rf_str):
            p = from_portable_path(rf_str)
            if p.is_file():
                referenced_paths.add(str(p.resolve()))

    return sorted(list(referenced_paths))


def rewrite_artifact_paths_to_temp(text: str, temp_dir: Path | None = None) -> str:
    r"""Rewrites references to brain artifacts in markdown text to <temp>/aivc/artifacts/<subpath>.

    Handles:
    1. Repair of corrupted links: filC:\Users\... -> file:///C:/Users/...
    2. URLs: file:///.../(.gemini/antigravity/brain|aivc/artifacts)/<subpath> -> file:///<temp>/aivc/artifacts/<subpath>
    3. Bare local paths: C:\...\brain\<subpath> -> <temp>\aivc\artifacts\<subpath>
       Strict lookbehind prevents rewriting inside file:/// URLs (avoiding 'filC:' corruption).
    4. Foreign machine user home paths: file:///C:/Users/(Jamet|hjamet)/... -> file:///<local_home>/...
    5. Stripping of legacy file lists (- ✏️ **Fichiers modifiés** and - 📖 **Fichiers consultés**)
    """
    if not text:
        return text

    # Pre-step 0: convert any wikilinks [[...]] to markdown links
    text = convert_wikilinks_to_markdown_links(text)

    import aivc.config as _cfg

    base_temp = temp_dir if temp_dir is not None else _cfg.get_aivc_temp_dir()
    artifacts_dir = Path(base_temp) / "artifacts"
    artifacts_posix = artifacts_dir.as_posix().replace("\\", "/").lstrip("/")

    # Step 1a: Clean any accidental 'file:///e:///...' prefix corruption
    text = re.sub(r"file://+(?:e://+)+", "file:///", text, flags=re.IGNORECASE)

    # Step 1b: Repair corrupted 'filC:\...' (excluding 'file://' via (?![eE]://) and requiring single slash after drive colon)
    def _fix_corrupted_fil(m: re.Match) -> str:
        raw_p = m.group(1).replace("\\", "/")
        return f"file:///{raw_p}"

    text = re.sub(
        r"(?<![A-Za-z0-9_])fil(?![eE]://)([A-Za-z]:[\\/](?![\\/])[^\s\[\]\(\)\"\'\`<>]+)",
        _fix_corrupted_fil,
        text,
    )

    # Step 1c: Normalize any backslashes inside file:/// URLs to forward slashes
    text = re.sub(
        r"(file://+[A-Za-z]:[^\s\[\]\(\)\"\'\`<>]+)",
        lambda m: m.group(1).replace("\\", "/"),
        text,
    )

    # Step 2: Rewrite URLs to file:///<temp>/aivc/artifacts/<subpath>
    url_pattern = re.compile(
        r"file://+(?:[A-Za-z]:)?[^\s\[\]\(\)\"\'\`<>]*?(?:\.?gemini[\\/]antigravity(?:-cli)?[\\/]brain|aivc[\\/]artifacts)[\\/]([^\s\[\]\(\)\"\'\`<>:]+)",
        re.IGNORECASE,
    )

    def _replace_url(m: re.Match) -> str:
        raw_sub = m.group(1).replace("\\", "/").strip("/")
        return f"file:///{artifacts_posix}/{raw_sub}"

    text = url_pattern.sub(_replace_url, text)

    # Step 2b: Rewrite foreign user home paths in file:/// URLs to local user home
    local_home = _cfg.get_home_dir()
    local_home_posix = local_home.as_posix().replace("\\", "/").rstrip("/").lstrip("/")

    user_home_url_pattern = re.compile(
        r"file://+(?:[A-Za-z]:/Users/(?!Public(?:/|$))[^/\s\[\]\(\)\"\'\`<>]+|/home/[^/\s\[\]\(\)\"\'\`<>]+)/([^\s\[\]\(\)\"\'\`<>]+)",
        re.IGNORECASE,
    )

    def _replace_user_home_url(m: re.Match) -> str:
        sub_path = m.group(1).lstrip("/")
        return f"file:///{local_home_posix}/{sub_path}"

    text = user_home_url_pattern.sub(_replace_user_home_url, text)

    # Step 3: Rewrite bare local paths with strict lookbehind to avoid touching inside file:///
    local_pattern = re.compile(
        r"(?<![A-Za-z0-9_:/])(?<!file://)(?<!file:///)(?:[A-Za-z]:[\\/]|/(?:Users|home)[\\/]|~[\\/])[^\s\[\]\(\)\"\'\`<>]*?(?:\.?gemini[\\/]antigravity(?:-cli)?[\\/]brain|aivc[\\/]artifacts)[\\/]([^\s\[\]\(\)\"\'\`<>:]+)",
        re.IGNORECASE,
    )

    def _replace_local(m: re.Match) -> str:
        raw_sub = m.group(1)
        suffix = ""
        line_m = re.search(r"(:L?\d+(?:-L?\d+)?)$", raw_sub)
        if line_m:
            suffix = line_m.group(1)
            raw_sub = raw_sub[:-len(suffix)]

        sub_clean = raw_sub.replace("/", "\\").strip("\\")
        native_path = str(artifacts_dir / Path(sub_clean))
        return native_path + suffix

    text = local_pattern.sub(_replace_local, text)

    # Post-step safety: guarantee no remaining filC:\ can survive
    text = re.sub(
        r"(?<![A-Za-z0-9_])fil(?![eE]://)([A-Za-z]:[\\/](?![\\/])[^\s\[\]\(\)\"\'\`<>]+)",
        _fix_corrupted_fil,
        text,
    )
    text = convert_wikilinks_to_markdown_links(text)
    # Post-step: strip legacy automated file lists
    text = strip_legacy_file_lists(text)
    return text


rewrite_brain_paths_to_temp_artifacts = rewrite_artifact_paths_to_temp


def sync_project_views(
    memory: Any,
    workspace_dir: str | Path | None = None,
    vault_dir: str | Path | None = None,
) -> list[str]:
    """Synchronizes project view notes for the given memory.

    Returns the list of generated or updated view file paths.
    """
    effective_ws = workspace_dir or vault_dir
    home = get_home_dir()
    is_dict = isinstance(memory, dict)
    if is_dict:
        if memory.get("_tldr_synced", False):
            return memory.get("_tldr_synced_paths", [])
    else:
        if getattr(memory, "_tldr_synced", False):
            return getattr(memory, "_tldr_synced_paths", [])

    # Process user items details (linkify unlinked files, rewrite paths, strip legacy lists)
    process_user_items_details(memory)

    # Restore missing brain files referenced by this memory or active tasks
    try:
        restore_artifacts_for_memory(memory, background_drive=True)
    except Exception as e:
        logger.debug("auto-restore in sync_project_views skipped: %s", e)

    temp_adv_dir = get_aivc_temp_dir() / "advance_tldr"
    def _mirror_paths(synced_paths: list[str]) -> None:
        if not synced_paths:
            return
        import aivc.config as _cfg
        cur_home = _cfg.get_home_dir()
        cur_temp_adv_dir = _cfg.get_aivc_temp_dir() / "advance_tldr"
        cur_user_temp_adv_dir = _cfg.get_aivc_user_temp_dir() / "advance_tldr"
        cur_mirror_user_temp = (cur_user_temp_adv_dir.as_posix().lower() != cur_temp_adv_dir.as_posix().lower())

        for sp in synced_paths:
            try:
                p_sp = Path(sp)
                if p_sp.is_file():
                    try:
                        content = p_sp.read_text(encoding="utf-8")
                        repaired = rewrite_artifact_paths_to_temp(content)
                        if repaired != content:
                            p_sp.write_text(repaired, encoding="utf-8")
                    except Exception:
                        pass
                view_dir_name = p_sp.parent.name or "advance_tldr"
                stem = p_sp.stem
                title = normalize_project_tldr_title(stem)
                destinations = [
                    cur_home / ".gemini" / "antigravity" / "brain" / "advance_tldr" / f"🤖 {title}.md",
                    cur_home / ".gemini" / "antigravity-cli" / "brain" / "advance_tldr" / f"🤖 {title}.md",
                    cur_temp_adv_dir / f"🤖 {title}.md",
                ]
                if cur_mirror_user_temp:
                    destinations.append(cur_user_temp_adv_dir / f"🤖 {title}.md")

                if view_dir_name != "advance_tldr":
                    destinations.append(cur_home / ".gemini" / "antigravity" / "brain" / view_dir_name / f"🤖 {title}.md")
                    destinations.append(cur_home / ".gemini" / "antigravity-cli" / "brain" / view_dir_name / f"🤖 {title}.md")
                    destinations.append(get_aivc_temp_dir() / view_dir_name / f"🤖 {title}.md")
                    if cur_mirror_user_temp:
                        destinations.append(get_aivc_user_temp_dir() / view_dir_name / f"🤖 {title}.md")

                for legacy_dest in destinations:
                    if p_sp.is_file() and legacy_dest.as_posix().lower() != p_sp.as_posix().lower():
                        legacy_dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p_sp, legacy_dest)
            except Exception:
                pass
            try:
                restore_artifacts_for_memory(sp, background_drive=True)
            except Exception:
                pass

    if _CUSTOM_SYNC_HANDLER is not None:
        try:
            try:
                res = _CUSTOM_SYNC_HANDLER(memory, workspace_dir=effective_ws, vault_dir=effective_ws)
            except TypeError:
                res = _CUSTOM_SYNC_HANDLER(memory, vault_dir=effective_ws)
            if is_dict:
                memory["_tldr_synced"] = True
                memory["_tldr_synced_paths"] = res or []
            else:
                setattr(memory, "_tldr_synced", True)
                setattr(memory, "_tldr_synced_paths", res or [])
            _mirror_paths(res or [])
            return res or []
        except Exception as e:
            logger.warning("Custom sync handler failed: %s", e)
            return []

    if not is_project_related_memory(memory):
        return []

    mod = _load_project_memory_module()
    if mod is None:
        logger.debug("project sync module not available; skipping project view sync.")
        return []

    try:
        sync_fn = getattr(mod, "sync_project_views_for_memory", None) or getattr(mod, "sync_advance_tldr_for_memory", None)
        if sync_fn is None:
            logger.warning("sync function not found in project sync module.")
            return []

        target_ws_str = str(effective_ws) if effective_ws else str(get_workspace_dir())

        # Fallback resolution of main_project_file if not set on memory
        mpf = getattr(memory, "main_project_file", None) if not isinstance(memory, dict) else memory.get("main_project_file")
        if not mpf:
            all_c = getattr(memory, "changes", []) or [] if not isinstance(memory, dict) else memory.get("changes", [])
            for c in all_c:
                cp = getattr(c, "path", None) if hasattr(c, "path") else (c.get("path") if isinstance(c, dict) else str(c))
                cp_clean = str(cp).replace("\\", "/")
                if cp_clean.lower().endswith(".md"):
                    if not any(ex in cp_clean.lower() for ex in ("/brain/", "/temp/", "appdata", "scratch", ".aivc/artifacts", "_agents", ".git")):
                        if hasattr(memory, "main_project_file"):
                            try:
                                setattr(memory, "main_project_file", to_portable_path(cp))
                            except Exception:
                                pass
                        if isinstance(memory, dict):
                            memory["main_project_file"] = to_portable_path(cp)
                        break

        try:
            synced_paths = sync_fn(memory, workspace_dir=target_ws_str, vault_dir=target_ws_str)
        except TypeError:
            synced_paths = sync_fn(memory, vault_dir=target_ws_str)

        if synced_paths:
            _mirror_paths(synced_paths)

        if is_dict:
            memory["_tldr_synced"] = True
            memory["_tldr_synced_paths"] = synced_paths or []
        else:
            setattr(memory, "_tldr_synced", True)
            setattr(memory, "_tldr_synced_paths", synced_paths or [])
        return synced_paths or []
    except Exception as e:
        mem_id = memory.get("id", "unknown") if is_dict else getattr(memory, "id", "unknown")
        logger.warning("Failed to synchronize project views for memory %s: %s", mem_id, e)
        return []


sync_advance_tldr = sync_project_views
_mirror_advance_tldr_files = sync_project_views


def backfill_missing_main_project_files(
    storage_dir: Path | str | None = None,
    vault_dir: Path | str | None = None,
    workspace_dir: Path | str | None = None,
    dry_run: bool = False,
    custom_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deterministically identifies and backfills missing main_project_file in AIVC commit JSONs.

    Strict rules:
    - Anchors strictly to existing workspace notes (.md in notes/ or root) and explicit known aliases.
    - Zero guesswork: never falls back to cand_notes[0] or random guessing if ambiguous.
    """
    effective_ws = workspace_dir or vault_dir
    ws_dir = Path(effective_ws) if effective_ws else get_workspace_dir()
    if storage_dir:
        s_dir = Path(storage_dir)
    else:
        from aivc.config import get_storage_root
        s_dir = get_storage_root(allow_fallback=True)

    commits_dir = s_dir / "commits" if (s_dir / "commits").is_dir() else s_dir

    # 1. Index all valid workspace notes
    ws_notes = list(ws_dir.glob("notes/*.md")) + list(ws_dir.glob("*.md"))
    excluded_dirs = {"_agents", ".obsidian", ".git", "templates", "readwise", "attachments", "thumbnails"}

    note_by_stem_lower: dict[str, str] = {}
    note_by_rel_lower: dict[str, str] = {}

    for n in ws_notes:
        try:
            rel_posix = n.relative_to(ws_dir).as_posix()
        except ValueError:
            continue
        if any(ex in rel_posix.lower() for ex in excluded_dirs):
            continue
        try:
            rel_to_docs = n.relative_to(get_home_dir() / "Documents").as_posix()
            port = f"~/Documents/{rel_to_docs}"
        except ValueError:
            port = f"~/Documents/{ws_dir.name}/{rel_posix}"
        note_by_rel_lower[rel_posix.lower()] = port

        stem = n.stem.lower()
        clean_stem = re.sub(r"^(?:🤖\s*)+", "", stem).strip()
        note_by_stem_lower[clean_stem] = port
        note_by_stem_lower[stem] = port

    # Custom aliases from argument or environment variable
    special_aliases: dict[str, str] = {}
    if custom_aliases:
        special_aliases.update({k.lower(): v for k, v in custom_aliases.items()})
    env_aliases_json = os.environ.get("AIVC_PROJECT_ALIASES_JSON")
    if env_aliases_json:
        try:
            parsed = json.loads(env_aliases_json)
            if isinstance(parsed, dict):
                special_aliases.update({k.lower(): str(v) for k, v in parsed.items()})
        except Exception:
            pass

    updated_count = 0
    scanned_count = 0
    skipped_count = 0
    updated_commits = []

    for p in sorted(commits_dir.glob("*.json")):
        scanned_count += 1
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        mpf = d.get("main_project_file")
        if mpf:
            skipped_count += 1
            continue

        title = d.get("title", "")
        target_mpf = None

        # Step A: extract project name from title pattern [Tag] Project
        m_tag = re.match(
            r"^\[[^\]]+\]\s*(?:🤖\s*)?([^—\-\–\:]+)(?:[—\-\–\:].*)?$",
            title,
            re.IGNORECASE,
        )
        if m_tag:
            proj_candidate = m_tag.group(1).strip().lower()
            clean_cand = re.sub(r"^(?:🤖\s*)+", "", proj_candidate).strip()
            if clean_cand in special_aliases:
                target_mpf = special_aliases[clean_cand]
            elif clean_cand in note_by_stem_lower:
                target_mpf = note_by_stem_lower[clean_cand]

        # Step B: check prefix before dash/colon
        if not target_mpf and ("—" in title or " - " in title or ":" in title):
            delim = "—" if "—" in title else (" - " if " - " in title else ":")
            prefix = title.split(delim)[0].strip().lower()
            clean_prefix = re.sub(r"^(?:🤖\s*)+", "", prefix).strip()
            if clean_prefix in special_aliases:
                target_mpf = special_aliases[clean_prefix]
            elif clean_prefix in note_by_stem_lower:
                target_mpf = note_by_stem_lower[clean_prefix]

        # Step B2: check special aliases directly in title
        if not target_mpf and special_aliases:
            t_low = title.lower()
            for alias_key, alias_path in special_aliases.items():
                if alias_key in t_low:
                    target_mpf = alias_path
                    break

        # Step C: inspect read_files, edited_files, changes
        if not target_mpf:
            candidate_notes_in_commit: set[str] = set()
            all_files = (
                d.get("read_files", [])
                + d.get("edited_files", [])
                + [c.get("path") if isinstance(c, dict) else str(c) for c in d.get("changes", [])]
            )
            for f in all_files:
                path_str = f.get("path") if isinstance(f, dict) else str(f)
                p_clean = path_str.replace("\\", "/").strip()
                if p_clean.lower().endswith(".md"):
                    if not any(ex in p_clean.lower() for ex in ("/brain/", "/temp/", "appdata", "scratch", ".aivc/artifacts")):
                        stem = Path(p_clean).stem.lower()
                        clean_s = re.sub(r"^(?:🤖\s*)+", "", stem).strip()
                        if clean_s in note_by_stem_lower:
                            candidate_notes_in_commit.add(note_by_stem_lower[clean_s])

            if len(candidate_notes_in_commit) == 1:
                target_mpf = list(candidate_notes_in_commit)[0]
            elif len(candidate_notes_in_commit) > 1:
                for c_note in candidate_notes_in_commit:
                    stem = Path(c_note).stem.lower()
                    clean_s = re.sub(r"^(?:🤖\s*)+", "", stem).strip()
                    if clean_s in title.lower():
                        target_mpf = c_note
                        break

        # Step D: persist if determined
        if target_mpf:
            d["main_project_file"] = target_mpf
            if not dry_run:
                p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            updated_count += 1
            updated_commits.append({"id": d.get("id"), "title": title, "main_project_file": target_mpf})

    return {
        "status": "success",
        "scanned": scanned_count,
        "already_had_mpf": skipped_count,
        "updated": updated_count,
        "updated_commits": updated_commits,
    }
