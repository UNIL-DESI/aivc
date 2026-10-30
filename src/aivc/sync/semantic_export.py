"""
SemanticDriveExporter: Exports AIVC task states and memories to Google Drive as native Google Docs.

Maintains 3 Live Google Docs and 3 paginated Archive subfolders:
- Live_New, Live_Await_Validation, Live_Validated (25-30 most recent items)
- Archives_New, Archives_Await_Validation, Archives_Validated (Archive_<Status>_001, etc.)
Ensures 100% semantic text filtering: strips raw filesystem paths for audio readability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 25
MIME_GDOC = "application/vnd.google-apps.document"
MIME_FOLDER = "application/vnd.google-apps.folder"

# Regex patterns for semantic filtering
_LINK_REGEX = re.compile(r"\[([^\]]+)\]\((?:file:///[^\)]+|[a-zA-Z]:[^\)]+|~[^\)]+|https?://[^\)]+)\)")
_URL_REGEX = re.compile(r"https?://([^/\s\)\]\`\"\'\<\>]+)(?:/[^\s\)\]\`\"\'\<\>]*)?", re.IGNORECASE)
_LOCAL_PATH_REGEX = re.compile(
    r"(?:file:///(?:[a-zA-Z]:)?[\\/]?|[a-zA-Z]:[\\/]|/(?:Users|home|root|tmp|var|usr|etc|opt)[\\/]|(?:~|\.gemini|\.?antigravity|\.?brain)[\\/])"
    r"[^\s\"\'\`\<\>\)\]]+?(?::L\d+(?:-L\d+)?)?(?=[,\s\"\'\`\<\>\)\];]|$)",
    re.IGNORECASE,
)

GENERIC_PROJECT_NAMES = {
    "roadmap", "advance", "projet general", "projet général", "task",
    "scratch", "memoire_principale", "index_principal", "home",
    "general", "général"
}


def _path_replacer(match: re.Match) -> str:
    val = match.group(0).strip()
    val = re.sub(r":L\d+(?:-L\d+)?$", "", val, flags=re.IGNORECASE)
    val = val.replace("\\", "/").rstrip("/")
    basename = val.split("/")[-1].strip()
    return basename


def clean_semantic_text(text: str) -> str:
    """Strips file paths, local URLs, and formats Markdown for clear voice synthesis.

    - Replaces markdown links [label](path/url) with 'label'.
    - Replaces URLs with their domain name (without leaving orphan http/https).
    - Replaces raw local file paths (Windows, POSIX, brain paths) with their basename.
    - Strips orphan :Lxx-Lyy line references.
    - Cleans punctuation and whitespace.
    """
    if not text:
        return ""

    # 1. Replace markdown links [label](url_or_path) with just label
    cleaned = _LINK_REGEX.sub(r"\1", text)

    # 2. Replace URLs with domain
    cleaned = _URL_REGEX.sub(r"\1", cleaned)

    # 3. Replace local paths with basename
    cleaned = _LOCAL_PATH_REGEX.sub(_path_replacer, cleaned)

    # 4. Clean orphan line numbers like ":L48-L62"
    cleaned = re.sub(r":L\d+(?:-L\d+)?\b", "", cleaned, flags=re.IGNORECASE)

    # 5. Clean punctuation spacing & double colons
    cleaned = re.sub(r":\s*:", ":", cleaned)
    cleaned = re.sub(r"[ \t]+,", ",", cleaned)

    # 6. Clean up multiple empty lines and excessive spaces
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    result_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                result_lines.append("")
                prev_empty = True
        else:
            result_lines.append(line)
            prev_empty = False

    return "\n".join(result_lines).strip()


def _is_generic_project_name(name: str) -> bool:
    if not name:
        return True
    n = name.strip().lower()
    if n in GENERIC_PROJECT_NAMES:
        return True
    for g in GENERIC_PROJECT_NAMES:
        if n == g or n.startswith(f"{g} ") or n.startswith(f"{g}-") or n.startswith(f"{g}_"):
            return True
    return False


def extract_project_name(
    read_files: list[Any] | None,
    memory_title: str = "",
    journal: str = "",
    task_text: str = "",
    details_text: str = "",
    main_project_file: str = "",
) -> str:
    """Converts main_project_file or read_files notes reference into a readable project title.

    Examples:
        'notes/Améliorer le système Antigravity Obsidian.md' -> 'Projet : Améliorer le système Antigravity Obsidian'
        '[Asharde] Session 12' in title -> 'Projet : Asharde'
    """
    # 0. Priority: From main_project_file (absolute priority)
    if main_project_file and str(main_project_file).strip():
        p = str(main_project_file).strip().replace("\\", "/")
        name = None
        if "notes/" in p and p.endswith(".md"):
            name = p.split("notes/")[-1][:-3].strip()
        elif p.endswith(".md") and "voicenotes" in p.lower():
            name = p.split("/")[-1][:-3].strip()
        elif p.endswith(".md"):
            name = p.split("/")[-1][:-3].strip()
        elif "/" in p:
            name = p.split("/")[-1].strip()
        else:
            name = p.strip()

        if name:
            if not _is_generic_project_name(name):
                return f"Projet : {name}"
            return f"Projet : {name}"

    # 1. From read_files
    if read_files:
        for rf in read_files:
            path_str = rf.get("path") if isinstance(rf, dict) else str(rf)
            p = path_str.replace("\\", "/")
            name = None
            if "notes/" in p and p.endswith(".md"):
                name = p.split("notes/")[-1][:-3].strip()
            elif p.endswith(".md") and "voicenotes" in p.lower():
                name = p.split("/")[-1][:-3].strip()

            if name and not _is_generic_project_name(name):
                return f"Projet : {name}"

    # 2. From memory_title
    if memory_title:
        title = memory_title.strip()
        # Pattern 1: [Advance] <Projet> — ... or [Roadmap] <Projet> — ...
        m = re.match(r"^\[(?:Advance|Roadmap|Session|Project)\]\s*([^—–\:\n]+?)(?:\s+[-—–:]\s+.*|$)", title)
        if m and m.group(1).strip() and not _is_generic_project_name(m.group(1).strip()):
            return f"Projet : {m.group(1).strip()}"

        # Pattern 2: [Project Name] ...
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", title)
        if m:
            tag = m.group(1).strip()
            rest = m.group(2).strip()
            if not _is_generic_project_name(tag):
                return f"Projet : {tag}"
            elif rest:
                sub_m = re.match(r"^([^—–\:\n]+)", rest)
                if sub_m and not _is_generic_project_name(sub_m.group(1).strip()):
                    return f"Projet : {sub_m.group(1).strip()}"

        # Pattern 3: Roadmap synchronisée pour <Projet>
        m = re.search(r"Roadmap\s+synchronis[ée]e\s+pour\s+([^—–:\n\.\,]+)", title, re.IGNORECASE)
        if m and not _is_generic_project_name(m.group(1).strip()):
            return f"Projet : {m.group(1).strip()}"

        # Pattern 4: Session ... sur <Projet>
        m = re.search(r"Session\s+.*?\s+sur\s+([^—–:\n\.\,\(]+)", title, re.IGNORECASE)
        if m and not _is_generic_project_name(m.group(1).strip()):
            return f"Projet : {m.group(1).strip()}"

        # Pattern 5: <Projet> : ... or <Projet> — ...
        if " : " in title or " — " in title or " – " in title:
            first_part = re.split(r"\s+[:—–]\s+", title)[0].strip()
            if first_part and not _is_generic_project_name(first_part) and len(first_part) < 60:
                return f"Projet : {first_part}"

        # If title itself looks like a project name (short, non-generic)
        if not _is_generic_project_name(title) and len(title) < 50 and not title.startswith("Session"):
            return f"Projet : {title}"

    # 3. From first line of journal
    if journal:
        first_line = journal.strip().splitlines()[0] if journal.strip() else ""
        first_line = re.sub(r"^[#\*\-\s]+", "", first_line).strip()
        # Session ... sur `notes/<Projet>.md` or sur <Projet>
        m = re.search(r"(?:sur|pour)\s+`?(?:notes/)?([^`\(\n]+?)(?:\.md)?`?(?:\s*\(|\s+[-—–:]|$)", first_line, re.IGNORECASE)
        if m and not _is_generic_project_name(m.group(1).strip()):
            proj = m.group(1).strip()
            proj = re.split(r"\s+[:—–]\s+", proj)[0].strip()
            if not _is_generic_project_name(proj):
                return f"Projet : {proj}"

        # Journal Autonome — <Projet>
        m = re.search(r"(?:Journal|Session).*?[-—–:]\s*([^—–:\(\n]+)", first_line, re.IGNORECASE)
        if m and not _is_generic_project_name(m.group(1).strip()):
            return f"Projet : {m.group(1).strip()}"

    # 4. From task_text / details_text
    for txt in [task_text, details_text]:
        if txt:
            m = re.search(r"\[([^\]]+)\]", txt)
            if m and not _is_generic_project_name(m.group(1).strip()):
                return f"Projet : {m.group(1).strip()}"

    # 5. Fallback on title prefix before dash/colon/question mark
    if memory_title and not _is_generic_project_name(memory_title):
        first_chunk = re.split(r"\s+[-—–:]\s+|[—–\:\?]", memory_title)[0].strip()
        if first_chunk and not _is_generic_project_name(first_chunk):
            return f"Projet : {first_chunk}"

    return "Projet Général"


def collect_tasks_from_commits(commits_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Scans all commit JSON files and extracts the current task state and history by status.

    Returns:
        tasks_current: mapping of task_id -> latest task state
        history_by_status: mapping of 'new' | 'await_validation' | 'validated' -> chronological list
    """
    if not commits_dir.is_dir():
        return {}, {"new": [], "await_validation": [], "validated": []}

    commit_files = list(commits_dir.glob("*.json"))
    memories: list[dict[str, Any]] = []

    for f in commit_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            memories.append(data)
        except Exception:
            pass

    # Sort chronologically by timestamp
    memories.sort(key=lambda m: m.get("timestamp", ""))

    tasks_current: dict[str, dict[str, Any]] = {}
    history_by_status: dict[str, list[dict[str, Any]]] = {
        "new": [],
        "await_validation": [],
        "validated": [],
    }

    for m in memories:
        mpf = m.get("main_project_file", "")
        project_name = extract_project_name(
            m.get("read_files"),
            memory_title=m.get("title", ""),
            journal=m.get("journal", "") or m.get("note", ""),
            main_project_file=mpf,
        )
        user_items = m.get("user_items", []) or m.get("tasks", []) or []
        journal_text = clean_semantic_text(m.get("journal", "") or m.get("note", ""))
        timestamp = m.get("timestamp", "")
        memory_title = m.get("title", "")
        memory_id = m.get("id", "")

        for item in user_items:
            t_id = item.get("id") or item.get("task", "").strip()
            if not t_id:
                continue

            raw_status = item.get("task_status")
            if not raw_status:
                raw_status = "validated" if item.get("done") else "new"

            raw_task = item.get("task", "")
            raw_details = item.get("details", "")

            item_project = project_name
            if "Général" in item_project and not mpf:
                fb = extract_project_name(
                    [],
                    memory_title="",
                    journal="",
                    task_text=raw_task,
                    details_text=raw_details,
                )
                if "Général" not in fb:
                    item_project = fb

            entry = {
                "id": str(t_id).strip(),
                "task": clean_semantic_text(raw_task),
                "details": clean_semantic_text(raw_details),
                "journal": journal_text,
                "project": item_project,
                "timestamp": timestamp,
                "status": raw_status,
                "memory_title": memory_title,
                "memory_id": memory_id,
            }

            tasks_current[t_id] = entry
            if raw_status in history_by_status:
                history_by_status[raw_status].append(entry)

    return tasks_current, history_by_status


def format_live_document(status: str, items: list[dict[str, Any]]) -> str:
    """Formats the text of a Live Google Doc."""
    status_labels = {
        "new": "Tâches Ouvertes & Roadmap Récente (Live_New)",
        "await_validation": "Décisions & Livrables en Attente de Validation (Live_Await_Validation)",
        "validated": "Tâches Validées Récemment (Live_Validated)",
    }
    label = status_labels.get(status, f"Flux Live {status}")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# AIVC — {label}",
        f"Dernière synchronisation : {now_str}",
        f"Nombre d'éléments : {len(items)}",
        "---",
        "",
    ]

    if not items:
        lines.append("Aucun élément actif pour ce statut.")
        return "\n".join(lines)

    for i, it in enumerate(items, 1):
        lines.append(f"## {i}. [{it['id']}] {it['task']}")
        raw_proj = it.get("project", "")
        clean_proj = re.sub(r"^(?:Projet\s*:\s*)+", "", raw_proj).strip()
        lines.append(f"- **Projet** : {clean_proj if clean_proj else raw_proj}")
        lines.append(f"- **Date** : {it['timestamp']}")
        lines.append(f"- **Statut** : {it['status']}")

        if it.get("details"):
            details_label = "Décision & Détails" if status == "validated" else "Explications & Détails"
            lines.append(f"- **{details_label}** :\n{it['details']}")

        if status == "await_validation" and it.get("journal"):
            lines.append(f"- **Récit de Session (Journal)** :\n{it['journal']}")

        lines.append("")

    return "\n".join(lines)


def format_archive_document(
    status: str,
    items: list[dict[str, Any]],
    batch_index: int = 1,
    total_batches: int | None = None,
    **kwargs: Any,
) -> str:
    """Formats the text of an Archive Google Doc."""
    if "batch_num" in kwargs:
        batch_index = kwargs["batch_num"]
    status_labels = {
        "new": "Archives des Tâches Créées (Archives_New)",
        "await_validation": "Archives des Livrables en Attente (Archives_Await_Validation)",
        "validated": "Archives des Tâches Validées (Archives_Validated)",
    }
    label = status_labels.get(status, f"Archives {status}")

    lines = [
        f"# AIVC — {label} — Tranche d'archive {batch_index:03d} ({len(items)} éléments)",
        f"Nombre d'éléments : {len(items)}",
        "---",
        "",
    ]

    for i, it in enumerate(items, 1):
        lines.append(f"## {i}. [{it['id']}] {it['task']}")
        raw_proj = it.get("project", "")
        clean_proj = re.sub(r"^(?:Projet\s*:\s*)+", "", raw_proj).strip()
        lines.append(f"- **Projet** : {clean_proj if clean_proj else raw_proj}")
        lines.append(f"- **Date** : {it['timestamp']}")
        lines.append(f"- **Statut d'archive** : {it['status']}")

        if it.get("details"):
            details_label = "Décision & Détails" if status == "validated" else "Détails"
            lines.append(f"- **{details_label}** :\n{it['details']}")

        if it.get("journal"):
            lines.append(f"- **Journal associé** :\n{it['journal']}")

        lines.append("")

    return "\n".join(lines)


def is_test_environment() -> bool:
    """Detects if code is executing under pytest or unittest."""
    import os
    import sys
    if os.environ.get("PYTEST_CURRENT_TEST") or ("pytest" in sys.modules):
        return True
    if os.environ.get("UNITTEST_CURRENT_TEST") or os.environ.get("AIVC_TEST_MODE"):
        return True
    # unittest est importé transitivement par googleapiclient en production :
    # ne le considérer en test que si unittest est le script principal (__main__)
    main_mod = sys.modules.get("__main__")
    if main_mod and getattr(main_mod, "__file__", None):
        main_file = str(main_mod.__file__).lower()
        if "test" in os.path.basename(main_file) or "pytest" in main_file or "unittest" in main_file:
            return True
    return False


def is_canonical_storage(storage_root: Path | str | None) -> bool:
    """Verifies that storage_root points strictly to canonical ~/.aivc/storage.

    Explicitly rejects any path containing tmp, temp, test or diverging from Path.home() / .aivc / storage.
    """
    if not storage_root:
        return False
    try:
        from aivc.config import get_home_dir
        home_path = get_home_dir()
    except Exception:
        home_path = Path.home()

    canonical_target = (home_path / ".aivc" / "storage").resolve()
    try:
        resolved = Path(storage_root).resolve()
        path_str_lower = str(resolved).lower().replace("\\", "/")
        for suspect in ["/tmp", "\\tmp", "/temp", "\\temp", "test_storage", "/test", "\\test"]:
            if suspect in path_str_lower:
                return False
        return resolved == canonical_target
    except Exception:
        return False


class SemanticDriveExporter:
    """Synchronizes semantic state of AIVC tasks to Google Drive native Docs."""

    def __init__(self, drive_sync_manager: Any = None, storage_root: Path | None = None) -> None:
        from aivc.config import get_storage_root
        from aivc.sync.drive import NativeDriveSyncManager

        self.storage_root = storage_root or get_storage_root(allow_fallback=True)
        self.manager = drive_sync_manager or NativeDriveSyncManager(self.storage_root)
        self._doc_id_cache: dict[str, str] = {}
        self._content_hashes: dict[str, str] = {}
        self._cache_file = self.storage_root / "semantic_export_cache.json"
        self._load_export_cache()

    def _load_export_cache(self) -> None:
        """Loads cached Google Doc IDs and SHA-256 hashes from disk."""
        if self._cache_file.is_file():
            try:
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._doc_id_cache.update(data.get("doc_ids", {}))
                    self._content_hashes.update(data.get("content_hashes", {}))
            except Exception as e:
                logger.debug("Failed to load semantic export cache: %s", e)

    def _save_export_cache(self) -> None:
        """Persists Google Doc IDs and SHA-256 hashes safely to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self._cache_file.with_suffix(".tmp")
            payload = {
                "doc_ids": self._doc_id_cache,
                "content_hashes": self._content_hashes,
            }
            tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_file.replace(self._cache_file)
        except Exception as e:
            logger.debug("Failed to save semantic export cache: %s", e)

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        """Finds or creates a subfolder under parent_id, cleaning up duplicate folders if any."""
        service = self.manager._get_service()
        query = f"name = '{name}' and '{parent_id}' in parents and mimeType = '{MIME_FOLDER}' and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])
        if files:
            folder_id = files[0]["id"]
            if len(files) > 1:
                # Deduplicate: trash extraneous duplicate folders
                for dup in files[1:]:
                    try:
                        service.files().update(fileId=dup["id"], body={"trashed": True}).execute()
                        logger.warning("Trashed duplicate folder %s (id=%s)", name, dup["id"])
                    except Exception as e:
                        logger.warning("Failed to trash duplicate folder %s: %s", dup["id"], e)
            return folder_id

        metadata = {"name": name, "mimeType": MIME_FOLDER, "parents": [parent_id]}
        created = service.files().create(body=metadata, fields="id").execute()
        return created["id"]

    def _upsert_gdoc(self, title: str, parent_id: str, content: str) -> str:
        """Creates or updates a native Google Doc by complete content replacement, deduplicating if needed.

        Skips update if content SHA-256 hash matches the cached hash for this doc.
        """
        from googleapiclient.http import MediaInMemoryUpload
        from googleapiclient.errors import HttpError

        service = self.manager._get_service()
        cache_key = f"{parent_id}/{title}"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        file_id = self._doc_id_cache.get(cache_key)
        if not file_id:
            query = f"name = '{title}' and '{parent_id}' in parents and trashed = false"
            results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
            files = results.get("files", [])
            if files:
                file_id = files[0]["id"]
                self._doc_id_cache[cache_key] = file_id
                if len(files) > 1:
                    # Deduplicate: trash extraneous duplicate docs
                    for dup in files[1:]:
                        try:
                            service.files().update(fileId=dup["id"], body={"trashed": True}).execute()
                            logger.warning("Trashed duplicate doc %s (id=%s)", title, dup["id"])
                        except Exception as e:
                            logger.warning("Failed to trash duplicate doc %s: %s", dup["id"], e)

        # Skip update if content hash is unchanged
        if file_id and self._content_hashes.get(cache_key) == content_hash:
            logger.debug("SemanticDriveExporter: %s unchanged (hash %s), skipping update.", cache_key, content_hash[:8])
            return file_id

        media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain", resumable=True)

        if file_id:
            # Overwrite existing Google Doc via PATCH
            try:
                updated = service.files().update(fileId=file_id, media_body=media, fields="id").execute()
                self._doc_id_cache[cache_key] = updated["id"]
                self._content_hashes[cache_key] = content_hash
                self._save_export_cache()
                return updated["id"]
            except HttpError as e:
                if getattr(e, "resp", None) and getattr(e.resp, "status", None) == 404:
                    # Document was deleted remotely, re-create
                    file_id = None
                    self._doc_id_cache.pop(cache_key, None)
                    self._content_hashes.pop(cache_key, None)
                else:
                    raise

        # Create new native Google Doc
        body = {
            "name": title,
            "mimeType": MIME_GDOC,
            "parents": [parent_id],
        }
        created = service.files().create(body=body, media_body=media, fields="id").execute()
        self._doc_id_cache[cache_key] = created["id"]
        self._content_hashes[cache_key] = content_hash
        self._save_export_cache()
        return created["id"]

    def export_all(self, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
        """Main export method: generates Live docs and paginated archives on Google Drive.

        Returns:
            dict summarizing synced files and counts.
        """
        import os
        from aivc.config import get_storage_root

        # Strict Guard: prevent exporting to Google Drive if pytest/unittest is running OR if storage_root != canonical
        is_in_test = is_test_environment()
        is_canonical = is_canonical_storage(self.storage_root)

        if is_in_test or not is_canonical:
            # Allow mock testing if manager is an explicit Mock/MagicMock
            from unittest.mock import Mock, MagicMock
            if isinstance(self.manager, (Mock, MagicMock)):
                pass
            else:
                logger.warning(
                    "SemanticDriveExporter: Blocked export to Google Drive (is_test=%s, is_canonical=%s, storage=%s)",
                    is_in_test,
                    is_canonical,
                    self.storage_root,
                )
                return {
                    "status": "blocked",
                    "reason": "pytest_active" if is_in_test else "non_canonical_storage",
                    "synced_live": 0,
                    "synced_archives": 0,
                }

        if not getattr(self.manager, "enabled", False):
            logger.info("SemanticDriveExporter: Drive sync is disabled.")
            return {"status": "disabled", "synced_live": 0, "synced_archives": 0}

        try:
            root_id = self.manager._get_root_folder_id()
        except Exception as e:
            logger.warning("SemanticDriveExporter: Cannot access AIVC_Sync root: %s", e)
            return {"status": "error", "error": str(e)}

        commits_dir = self.storage_root / "commits"
        tasks_current, history_by_status = collect_tasks_from_commits(commits_dir)

        # 1. Export 3 Live Docs under AIVC_Sync
        live_files_synced = 0
        live_specs = [
            ("new", "Live_New"),
            ("await_validation", "Live_Await_Validation"),
            ("validated", "Live_Validated"),
        ]

        matching_by_status: dict[str, list[dict[str, Any]]] = {}
        for status_key, doc_name in live_specs:
            # Current tasks filtered by status, sorted reverse chronologically (most recent first)
            matching = [t for t in tasks_current.values() if t["status"] == status_key]
            matching.sort(key=lambda x: x["timestamp"], reverse=True)
            matching_by_status[status_key] = matching
            top_items = matching[:batch_size]

            text_content = format_live_document(status_key, top_items)
            try:
                self._upsert_gdoc(doc_name, root_id, text_content)
                live_files_synced += 1
            except Exception as e:
                logger.warning("Failed to sync live doc %s: %s", doc_name, e)

        # 2. Export 3 Archive Subfolders with paginated docs
        archive_specs = [
            ("new", "Archives_New", "Archive_New"),
            ("await_validation", "Archives_Await_Validation", "Archive_Await_Validation"),
            ("validated", "Archives_Validated", "Archive_Validated"),
        ]

        archives_synced = 0
        for status_key, folder_name, file_prefix in archive_specs:
            matching = matching_by_status.get(status_key, [])
            overflow = matching[batch_size:]
            overflow_chronological = list(reversed(overflow))
            batches = [
                overflow_chronological[i : i + batch_size]
                for i in range(0, len(overflow_chronological), batch_size)
            ]

            if not batches:
                continue

            try:
                folder_id = self._get_or_create_folder(folder_name, root_id)
            except Exception as e:
                logger.warning("Failed to get/create archive folder %s: %s", folder_name, e)
                continue

            total_batches = len(batches)
            for b_idx, batch_items in enumerate(batches, 1):
                doc_title = f"{file_prefix}_{b_idx:03d}"
                text_content = format_archive_document(status_key, batch_items, b_idx, total_batches)
                try:
                    self._upsert_gdoc(doc_title, folder_id, text_content)
                    archives_synced += 1
                except Exception as e:
                    logger.warning("Failed to sync archive doc %s: %s", doc_title, e)

        logger.info(
            "SemanticDriveExporter: Synchronized %d live docs and %d archive docs.",
            live_files_synced,
            archives_synced,
        )
        return {
            "status": "success",
            "synced_live": live_files_synced,
            "synced_archives": archives_synced,
            "live_counts": {
                "new": len([t for t in tasks_current.values() if t["status"] == "new"]),
                "await_validation": len([t for t in tasks_current.values() if t["status"] == "await_validation"]),
                "validated": len([t for t in tasks_current.values() if t["status"] == "validated"]),
            },
        }


def trigger_semantic_export_async(storage_root: Path | None = None) -> None:
    """Fires semantic export in a detached background daemon thread."""
    import threading

    # Strict Guard: do not trigger async export during pytest or unittest
    if is_test_environment():
        logger.debug("Skipping async semantic export: running inside test environment.")
        return

    # Strict Guard: must point strictly to canonical storage root
    if not is_canonical_storage(storage_root):
        if storage_root is None:
            try:
                from aivc.config import get_home_dir
                canonical = (get_home_dir() / ".aivc" / "storage").resolve()
            except Exception:
                canonical = (Path.home() / ".aivc" / "storage").resolve()
            if not is_canonical_storage(canonical):
                logger.debug("Skipping async semantic export: canonical storage is invalid.")
                return
            target_root = canonical
        else:
            logger.debug(
                "Skipping async semantic export: target root %s is not canonical.",
                storage_root,
            )
            return
    else:
        target_root = Path(storage_root).resolve()

    def _worker():
        try:
            exporter = SemanticDriveExporter(storage_root=target_root)
            exporter.export_all()
        except Exception as e:
            logger.debug("Async semantic export skipped/failed: %s", e)

    t = threading.Thread(target=_worker, name="AIVC-SemanticExporter", daemon=True)
    t.start()
