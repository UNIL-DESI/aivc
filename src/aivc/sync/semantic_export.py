"""
SemanticDriveExporter: Exports AIVC task states and memories to Google Drive as native Google Docs.

Maintains 3 Live Google Docs and 3 paginated Archive subfolders:
- Live_New, Live_Await_Validation, Live_Validated (25-30 most recent items)
- Archives_New, Archives_Await_Validation, Archives_Validated (Archive_<Status>_001, etc.)
Ensures 100% semantic text filtering: strips raw filesystem paths for audio readability.
"""

from __future__ import annotations

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
_FILE_PATH_REGEX = re.compile(
    r"(?:file:///?(?:[a-zA-Z]:)?[\\/]|(?:[a-zA-Z]:[\\/]|/home/[^/\s]+|~[\\/]))[^\s\"\'\`\<\>\:\;\,\?\!\)\(\[\]]+",
    re.IGNORECASE,
)
_BRAIN_PATH_REGEX = re.compile(r"\.?(?:gemini[\\/]antigravity[\\/]brain|brain[\\/])[^\s\"\'\`\<\>]+", re.IGNORECASE)


def clean_semantic_text(text: str) -> str:
    """Strips file paths, local URLs, and formats Markdown for clear voice synthesis.

    - Replaces markdown links [label](path/url) with 'label'.
    - Strips raw absolute file paths (Windows, POSIX, brain paths).
    - Cleans up duplicate blank lines.
    """
    if not text:
        return ""

    # 1. Replace markdown links [label](url_or_path) with just label
    cleaned = _LINK_REGEX.sub(r"\1", text)

    # 2. Strip raw file paths
    cleaned = _FILE_PATH_REGEX.sub("", cleaned)
    cleaned = _BRAIN_PATH_REGEX.sub("", cleaned)

    # 3. Clean up multiple empty lines and excessive spaces
    lines = [line.strip() for line in cleaned.splitlines()]
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


def extract_project_name(read_files: list[Any] | None, memory_title: str = "") -> str:
    """Converts read_files notes reference into a readable project title.

    Examples:
        'notes/Améliorer le système Antigravity Obsidian.md' -> 'Projet : Améliorer le système Antigravity Obsidian'
        '[Asharde] Session 12' in title -> 'Projet : Asharde'
    """
    if read_files:
        for rf in read_files:
            path_str = rf.get("path") if isinstance(rf, dict) else str(rf)
            p = path_str.replace("\\", "/")
            if "notes/" in p and p.endswith(".md"):
                name = p.split("notes/")[-1][:-3]
                return f"Projet : {name}"
            elif p.endswith(".md") and "voicenotes" in p.lower():
                name = p.split("/")[-1][:-3]
                return f"Projet : {name}"

    if memory_title:
        m = re.search(r"\[([^\]]+)\]", memory_title)
        if m:
            return f"Projet : {m.group(1)}"

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
        project_name = extract_project_name(m.get("read_files"), memory_title=m.get("title", ""))
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

            entry = {
                "id": str(t_id).strip(),
                "task": clean_semantic_text(raw_task),
                "details": clean_semantic_text(raw_details),
                "journal": journal_text,
                "project": project_name,
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
        lines.append(f"- **Projet** : {it['project']}")
        lines.append(f"- **Date** : {it['timestamp']}")
        lines.append(f"- **Statut** : {it['status']}")

        if it.get("details"):
            lines.append(f"- **Explications & Détails** :\n{it['details']}")

        if status == "await_validation" and it.get("journal"):
            lines.append(f"- **Récit de Session (Journal)** :\n{it['journal']}")

        lines.append("")

    return "\n".join(lines)


def format_archive_document(status: str, items: list[dict[str, Any]], batch_num: int, total_batches: int) -> str:
    """Formats the text of an Archive Google Doc."""
    status_labels = {
        "new": "Archives des Tâches Créées (Archives_New)",
        "await_validation": "Archives des Livrables en Attente (Archives_Await_Validation)",
        "validated": "Archives des Tâches Validées (Archives_Validated)",
    }
    label = status_labels.get(status, f"Archives {status}")

    lines = [
        f"# AIVC — {label} — Partie {batch_num:03d} / {total_batches:03d}",
        f"Nombre d'éléments : {len(items)}",
        "---",
        "",
    ]

    for i, it in enumerate(items, 1):
        lines.append(f"## {i}. [{it['id']}] {it['task']}")
        lines.append(f"- **Projet** : {it['project']}")
        lines.append(f"- **Date** : {it['timestamp']}")
        lines.append(f"- **Statut d'archive** : {it['status']}")

        if it.get("details"):
            lines.append(f"- **Détails** :\n{it['details']}")

        if it.get("journal"):
            lines.append(f"- **Journal associé** :\n{it['journal']}")

        lines.append("")

    return "\n".join(lines)


class SemanticDriveExporter:
    """Synchronizes semantic state of AIVC tasks to Google Drive native Docs."""

    def __init__(self, drive_sync_manager: Any = None, storage_root: Path | None = None) -> None:
        from aivc.config import get_storage_root
        from aivc.sync.drive import NativeDriveSyncManager

        self.storage_root = storage_root or get_storage_root(allow_fallback=True)
        self.manager = drive_sync_manager or NativeDriveSyncManager(self.storage_root)
        self._doc_id_cache: dict[str, str] = {}

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        """Finds or creates a subfolder under parent_id."""
        service = self.manager._get_service()
        query = f"name = '{name}' and '{parent_id}' in parents and mimeType = '{MIME_FOLDER}' and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {"name": name, "mimeType": MIME_FOLDER, "parents": [parent_id]}
        created = service.files().create(body=metadata, fields="id").execute()
        return created["id"]

    def _upsert_gdoc(self, title: str, parent_id: str, content: str) -> str:
        """Creates or updates a native Google Doc by complete content replacement."""
        from googleapiclient.http import MediaInMemoryUpload

        service = self.manager._get_service()
        cache_key = f"{parent_id}/{title}"

        file_id = self._doc_id_cache.get(cache_key)
        if not file_id:
            query = f"name = '{title}' and '{parent_id}' in parents and trashed = false"
            results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
            files = results.get("files", [])
            if files:
                file_id = files[0]["id"]
                self._doc_id_cache[cache_key] = file_id

        media = MediaInMemoryUpload(content.encode("utf-8"), mimetype="text/plain", resumable=True)

        if file_id:
            # Overwrite existing Google Doc via PATCH
            updated = service.files().update(fileId=file_id, media_body=media, fields="id").execute()
            return updated["id"]
        else:
            # Create new native Google Doc
            body = {
                "name": title,
                "mimeType": MIME_GDOC,
                "parents": [parent_id],
            }
            created = service.files().create(body=body, media_body=media, fields="id").execute()
            self._doc_id_cache[cache_key] = created["id"]
            return created["id"]

    def export_all(self, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
        """Main export method: generates Live docs and paginated archives on Google Drive.

        Returns:
            dict summarizing synced files and counts.
        """
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

        for status_key, doc_name in live_specs:
            # Current tasks filtered by status, sorted reverse chronologically (most recent first)
            matching = [t for t in tasks_current.values() if t["status"] == status_key]
            matching.sort(key=lambda x: x["timestamp"], reverse=True)
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
            try:
                folder_id = self._get_or_create_folder(folder_name, root_id)
            except Exception as e:
                logger.warning("Failed to get/create archive folder %s: %s", folder_name, e)
                continue

            history_items = history_by_status.get(status_key, [])
            # Split into batches of batch_size
            total_items = len(history_items)
            batches = [
                history_items[i : i + batch_size]
                for i in range(0, max(total_items, 1), batch_size)
            ]
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

    def _worker():
        try:
            exporter = SemanticDriveExporter(storage_root=storage_root)
            exporter.export_all()
        except Exception as e:
            logger.debug("Async semantic export skipped/failed: %s", e)

    t = threading.Thread(target=_worker, name="AIVC-SemanticExporter", daemon=True)
    t.start()
