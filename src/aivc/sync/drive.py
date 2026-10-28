"""
NativeDriveSyncManager: Google Drive synchronization using the official API.

Replaces the legacy rclone-based sync with a native Python implementation
using google-api-python-client, google-auth-httplib2, and google-auth-oauthlib.
"""

import json
from pathlib import Path
from typing import Any

from aivc.config import (
    get_aivc_config,
    get_machine_id,
    get_token_path,
    to_portable_path,
    from_portable_path,
)

# Scopes: full read/write on user's own Drive files
_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_ROOT_FOLDER_NAME = "AIVC_Sync"


class NativeDriveSyncManager:
    """Manages push/pull of memory metadata and files to Google Drive natively."""

    def __init__(self, storage_root: Path):
        import os
        self.storage_root = storage_root
        self.config = get_aivc_config().get("sync", {})
        disable_sync = os.environ.get("AIVC_DISABLE_SYNC", "").lower() in ("1", "true", "yes")
        self.enabled = (not disable_sync) and self.config.get("enabled", False)
        # Blob sync is disabled by default in Phase 29+ to avoid security/storage leaks
        self.sync_blobs = self.config.get("sync_blobs", False)
        # File sync for edited_files (enabled by default when sync is enabled)
        self.sync_files = self.config.get("sync_files", True)
        self.machine_id = get_machine_id()

        # Lazy Google Drive service (using thread local to ensure thread safety during parallel downloads)
        import threading
        self._thread_local = threading.local()

        # Folder ID cache to avoid repeated API lookups
        self._folder_cache: dict[str, str] = {}
        state_file = self.storage_root / "sync_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                self._folder_cache.update(data.get("folders", {}))
            except Exception:
                pass

    def _save_sync_state(self) -> None:
        state_file = self.storage_root / "sync_state.json"
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps({"folders": self._folder_cache}, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _get_service(self):
        """Lazy-load and authenticate with Google Drive API (thread-local)."""
        if getattr(self._thread_local, "service", None) is not None:
            return self._thread_local.service

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError:
            raise RuntimeError(
                "Google Drive API libraries are not installed. "
                "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        creds = None
        token_path = get_token_path()

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                token_path.write_text(creds.to_json(), encoding="utf-8")
            else:
                raise RuntimeError(
                    "Google Drive authentication expired or missing. "
                    "Run 'aivc sync setup' to re-authenticate."
                )

        service = build("drive", "v3", credentials=creds)
        self._thread_local.service = service
        return service

    # ------------------------------------------------------------------
    # Folder management (cached)
    # ------------------------------------------------------------------

    def _find_or_create_folder(self, name: str, parent_id: str | None = None) -> str:
        """Find or create a folder in Google Drive, caching the result."""
        cache_key = f"{parent_id or 'root'}/{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        service = self._get_service()

        # Search for existing folder
        query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        files = results.get("files", [])

        if files:
            folder_id = files[0]["id"]
        else:
            # Create the folder
            metadata: dict[str, Any] = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                metadata["parents"] = [parent_id]
            folder = service.files().create(body=metadata, fields="id").execute()
            folder_id = folder["id"]

        self._folder_cache[cache_key] = folder_id
        self._save_sync_state()
        return folder_id

    def _get_root_folder_id(self) -> str:
        """Get (or create) the AIVC_Sync root folder."""
        return self._find_or_create_folder(_ROOT_FOLDER_NAME)

    def _get_files_folder_id(self) -> str:
        """Get (or create) the files/ folder under AIVC_Sync."""
        root_id = self._get_root_folder_id()
        return self._find_or_create_folder("files", root_id)

    @property
    def files_folder_id(self) -> str:
        """Folder ID of AIVC_Sync/files/ on Google Drive."""
        return self._get_files_folder_id()

    def _get_root_memories_folder_id(self) -> str:
        """Get (or create) the memories/ folder directly under AIVC_Sync."""
        root_id = self._get_root_folder_id()
        return self._find_or_create_folder("memories", root_id)

    def _get_machine_folder_id(self) -> str:
        """Get (or create) the machine-specific folder under AIVC_Sync."""
        root_id = self._get_root_folder_id()
        return self._find_or_create_folder(self.machine_id, root_id)

    def _get_memories_folder_id(self) -> str:
        """Get (or create) the commits/ folder under the machine folder."""
        machine_id = self._get_machine_folder_id()
        # We keep the folder name "commits" for backward compatibility on Drive
        return self._find_or_create_folder("commits", machine_id)

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def _upload_file(self, local_path: Path, folder_id: str, filename: str | None = None, skip_if_exists: bool = False) -> str:
        """Upload a file to a specific Drive folder. Returns file ID."""
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        target_name = filename or local_path.name

        # Check if file already exists in folder
        query = f"name = '{target_name}' and '{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
        existing = results.get("files", [])

        if existing and skip_if_exists:
            return existing[0]["id"]

        media = MediaFileUpload(str(local_path), resumable=True)

        if existing:
            # Update existing file
            file_id = existing[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            return file_id
        else:
            # Create new file
            metadata: dict[str, Any] = {"name": target_name, "parents": [folder_id]}
            result = service.files().create(body=metadata, media_body=media, fields="id").execute()
            return result["id"]

    def _download_file(self, file_id: str, local_path: Path) -> None:
        """Download a file from Drive to disk."""
        from googleapiclient.http import MediaIoBaseDownload

        service = self._get_service()
        request = service.files().get_media(fileId=file_id)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    # ------------------------------------------------------------------
    # Public API (same interface as old RcloneSyncManager)
    # ------------------------------------------------------------------

    def push_memory(self, memory_id: str) -> None:
        """Push a local memory JSON to Google Drive (both machine commits/ and common memories/)."""
        if not self.enabled:
            return

        local_path = self.storage_root / "commits" / f"{memory_id}.json"
        if not local_path.exists():
            return

        folder_id = self._get_memories_folder_id()
        self._upload_file(local_path, folder_id)

        root_memories_id = self._get_root_memories_folder_id()
        self._upload_file(local_path, root_memories_id)

    def push_latest_file(self, file_path: str | Path) -> bool:
        """Push the latest version of a file to Google Drive under AIVC_Sync/files/.

        Overwrites existing file on Drive in place to keep only the latest version.
        Excludes files > 25 MB and .git / node_modules directories.
        """
        if not self.enabled:
            return False

        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            return False

        try:
            local_path = from_portable_path(file_path)
            if not local_path.is_file():
                return False

            # If file exceeds 25 MB, ignore
            try:
                if local_path.stat().st_size > 25 * 1024 * 1024:
                    return False
            except OSError:
                return False

            portable = to_portable_path(local_path)
            rel_drive_path = portable[2:] if portable.startswith("~/") else portable.lstrip("/")

            parts = Path(rel_drive_path).parts
            if not parts:
                return False

            # Exclude .git and node_modules folders
            parts_lower = [p.lower() for p in parts]
            if ".git" in parts_lower or "node_modules" in parts_lower:
                return False

            parent_folders = parts[:-1]
            filename = parts[-1]

            service = self._get_service()
            current_parent_id = self.files_folder_id

            for folder_name in parent_folders:
                current_parent_id = self._find_or_create_folder(folder_name, current_parent_id)

            query = f"name = '{filename}' and '{current_parent_id}' in parents and trashed = false"
            results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
            existing = results.get("files", [])

            media = MediaFileUpload(str(local_path), resumable=True)

            if existing:
                file_id = existing[0]["id"]
                service.files().update(fileId=file_id, media_body=media).execute()
            else:
                metadata: dict[str, Any] = {"name": filename, "parents": [current_parent_id]}
                service.files().create(body=metadata, media_body=media, fields="id").execute()

            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Failed to push latest file %s to Drive: %s", file_path, e)
            return False


    def pull_latest_file(self, file_path: str, target_path: str | Path | None = None) -> Path:
        """Download latest version of file from AIVC_Sync/files/ on Google Drive to local disk."""
        if not self.enabled:
            raise RuntimeError("Google Drive sync is disabled.")

        service = self._get_service()
        portable = to_portable_path(file_path)
        rel_drive_path = portable[2:] if portable.startswith("~/") else portable.lstrip("/")
        parts = Path(rel_drive_path).parts
        if not parts:
            raise ValueError(f"Invalid file path: {file_path}")

        parent_folders = parts[:-1]
        filename = parts[-1]

        current_parent_id = self.files_folder_id
        for folder_name in parent_folders:
            query = f"name = '{folder_name}' and '{current_parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
            files = results.get("files", [])
            if not files:
                raise FileNotFoundError(f"Folder '{folder_name}' not found on Google Drive under AIVC_Sync/files/.")
            current_parent_id = files[0]["id"]

        query = f"name = '{filename}' and '{current_parent_id}' in parents and trashed = false"
        results = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
        files = results.get("files", [])
        if not files:
            raise FileNotFoundError(f"File '{filename}' not found on Google Drive under AIVC_Sync/files/{'/'.join(parent_folders)}.")

        file_id = files[0]["id"]
        destination = from_portable_path(target_path or file_path)
        self._download_file(file_id, destination)
        return destination

    def push_missing(self) -> dict:
        """Finds all local memories and pushes those missing from Google Drive.
        
        Returns:
            dict with 'memories_pushed' count.
        """
        if not self.enabled:
            return {"memories_pushed": 0}

        service = self._get_service()
        memories_folder_id = self._get_memories_folder_id()

        # 1. Get all local memories
        local_memories_dir = self.storage_root / "commits"
        if not local_memories_dir.exists():
            return {"memories_pushed": 0}
            
        local_memories = {f.name for f in local_memories_dir.iterdir() if f.suffix == ".json"}
        
        # 2. Get remote memories for this machine
        query = f"'{memories_folder_id}' in parents and trashed = false"

        remote_memories = set()
        page_token = None
        while True:
            results = service.files().list(
                q=query, spaces="drive", fields="nextPageToken, files(name)",
                pageSize=1000, pageToken=page_token
            ).execute()
            remote_memories.update(f["name"] for f in results.get("files", []))

            page_token = results.get("nextPageToken")
            if not page_token:
                break
        
        missing_memories = local_memories - remote_memories
        
        memories_pushed = 0
        
        for memory_file in missing_memories:
            memory_id = memory_file.replace(".json", "")
            self.push_memory(memory_id)
            memories_pushed += 1

            # Push associated edited files if they exist locally
            try:
                mem_path = local_memories_dir / memory_file
                if mem_path.is_file():
                    mem_data = json.loads(mem_path.read_text(encoding="utf-8"))
                    for c in mem_data.get("changes", []):
                        if isinstance(c, dict) and c.get("action") in ("edited", "added", "modified"):
                            p_str = c.get("path")
                            if p_str:
                                self.push_latest_file(p_str)
                    for p_str in mem_data.get("edited_files", []):
                        if p_str:
                            self.push_latest_file(p_str)
            except Exception:
                pass
            
        return {"memories_pushed": memories_pushed}

    def auto_restore_brain_files_from_data(self, mem_data: dict) -> list[str]:
        """Auto-restore missing brain files referenced in a memory dict from Google Drive."""
        from aivc.config import get_home_dir, extract_brain_subpath
        import re

        restored: list[str] = []
        home = get_home_dir()

        # Collect candidate subpaths under brain/
        candidate_subpaths: set[str] = set()

        # 1. From changes
        for c in mem_data.get("changes", []):
            p = c.get("path") if isinstance(c, dict) else str(c)
            sub = extract_brain_subpath(p)
            if sub:
                candidate_subpaths.add(sub)

        # 2. From edited_files and read_files
        for p in mem_data.get("edited_files", []) + mem_data.get("read_files", []):
            sub = extract_brain_subpath(p)
            if sub:
                candidate_subpaths.add(sub)

        # 3. From user_items (task and details)
        brain_regex = re.compile(
            r"(?:file:///?(?:[A-Za-z]:)?[\\/]|~/|[A-Za-z]:[\\/]|/)?(?:Users|home)?[^\"\'\`\s\)\>]*?\.?gemini[\\/]antigravity[\\/]brain[\\/]([^\s\"\'\`\)\>:]+)",
            re.IGNORECASE,
        )
        for item in mem_data.get("user_items", []) or mem_data.get("tasks", []):
            if isinstance(item, dict):
                text_to_scan = f"{item.get('task', '')} {item.get('details', '')}"
                for m in brain_regex.finditer(text_to_scan):
                    raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw:
                        candidate_subpaths.add(raw)

        # 4. From journal and note
        for text in (mem_data.get("journal", ""), mem_data.get("note", "")):
            if text:
                for m in brain_regex.finditer(text):
                    raw = m.group(1).rstrip("\\/.,").split(":")[0].replace("\\", "/").strip("/")
                    if raw:
                        candidate_subpaths.add(raw)

        # Download any missing files
        for sub in candidate_subpaths:
            local_path = home / ".gemini" / "antigravity" / "brain" / Path(sub)
            if local_path.is_file():
                continue
            portable = f"~/.gemini/antigravity/brain/{sub.replace('\\', '/')}"
            try:
                dest = self.pull_latest_file(portable, target_path=local_path)
                if dest and dest.is_file():
                    restored.append(str(local_path))
            except Exception:
                pass

        return restored

    def pull_memories_from_others(self) -> int:
        """Pull memories from other machines listed in config.

        Returns:
            Number of memories downloaded.
        """
        if not self.enabled:
            return 0

        service = self._get_service()
        root_id = self._get_root_folder_id()

        # List machine folders in AIVC_Sync
        query = f"'{root_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"

        machine_folders = []
        page_token = None
        while True:
            results = service.files().list(
                q=query, spaces="drive", fields="nextPageToken, files(id, name)",
                pageSize=1000, pageToken=page_token
            ).execute()
            machine_folders.extend(results.get("files", []))

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        local_memories_dir = self.storage_root / "commits"
        local_memories_dir.mkdir(parents=True, exist_ok=True)
        existing_memories = {f.name for f in local_memories_dir.iterdir() if f.suffix == ".json"}

        # Collect all missing files to download in parallel
        missing_files = []

        for mf in machine_folders:
            if mf["name"] == self.machine_id or mf["name"] == "blobs":
                continue

            # Find commits/ subfolder (where memories are stored)
            memories_query = (
                f"name = 'commits' and '{mf['id']}' in parents "
                f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            )
            memories_result = service.files().list(
                q=memories_query, spaces="drive", fields="files(id)"
            ).execute()
            memories_folders = memories_result.get("files", [])
            if not memories_folders:
                continue

            memories_folder_id = memories_folders[0]["id"]

            # List memory files
            files_query = f"'{memories_folder_id}' in parents and trashed = false"

            page_token = None
            while True:
                files_result = service.files().list(
                    q=files_query, spaces="drive", fields="nextPageToken, files(id, name)",
                    pageSize=1000, pageToken=page_token
                ).execute()

                for remote_file in files_result.get("files", []):
                    if remote_file["name"] in existing_memories:
                        continue
                    missing_files.append((remote_file["id"], local_memories_dir / remote_file["name"]))
                    existing_memories.add(remote_file["name"])

                page_token = files_result.get("nextPageToken")
                if not page_token:
                    break

        if not missing_files:
            return 0

        # Perform parallel download using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import sys

        pulled_count = 0
        max_workers = min(16, len(missing_files))

        print(f"Starting parallel download of {len(missing_files)} files using {max_workers} threads...", file=sys.stderr)
        sys.stderr.flush()

        failures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._download_file, file_id, dest_path): dest_path.name
                for file_id, dest_path in missing_files
            }

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    future.result()
                    pulled_count += 1
                    if pulled_count % 10 == 0 or pulled_count == len(missing_files):
                        print(f"Downloaded {pulled_count}/{len(missing_files)} files...", file=sys.stderr)
                        sys.stderr.flush()
                except Exception as e:
                    print(f"Error downloading {filename}: {e}", file=sys.stderr)
                    sys.stderr.flush()
                    failures.append((filename, e))

        # Auto-restore brain files for newly downloaded memories if missing locally
        for file_id, dest_path in missing_files:
            try:
                if dest_path.is_file():
                    mem_data = json.loads(dest_path.read_text(encoding="utf-8"))
                    self.auto_restore_brain_files_from_data(mem_data)
            except Exception:
                pass

        if failures:
            error_details = ", ".join(f"{name}: {err}" for name, err in failures)
            raise RuntimeError(f"Failed to download {len(failures)} memory files: {error_details}")

        return pulled_count


    # Blob sync (push_blob, fetch_blob) has been purged in Phase 30.

    def list_remote_machines(self) -> list[str]:
        """List machine IDs found on the remote Drive."""
        if not self.enabled:
            return []

        service = self._get_service()
        root_id = self._get_root_folder_id()

        query = f"'{root_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"

        folders = []
        page_token = None
        while True:
            results = service.files().list(
                q=query, spaces="drive", fields="nextPageToken, files(name)",
                pageSize=1000, pageToken=page_token
            ).execute()
            folders.extend(results.get("files", []))

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return [f["name"] for f in folders if f["name"] != "blobs" and f["name"] != self.machine_id]
