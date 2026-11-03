"""
AIVC Command Line Interface.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Set mission-critical environment variables before heavy ML imports
# This completely bypasses the 5-minute atexit/thread deadlock on Windows
# caused by ChromaDB PostHog telemetry failing on corporate firewalls.
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SemanticEngine is imported inside `main()` after we check AIVC_STORAGE_ROOT
# to keep CLI fast and error out early if the env var is missing.

# ---------------------------------------------------------------------------
# ANSI Formatting helpers
# ---------------------------------------------------------------------------
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
MAGENTA = "\033[0;35m"
RESET = "\033[0m"

# Unicode symbols fallback for Windows terminals lacking UTF-8 support
try:
    "✓".encode(sys.stdout.encoding or "utf-8")
    CHECK_MARK = "✓"
    CROSS_MARK = "✗"
except (UnicodeEncodeError, AttributeError):
    CHECK_MARK = "[OK]"
    CROSS_MARK = "[Error]"



def _format_bytes(n: int | None) -> str:
    if n is None:
        return "missing"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


def _get_engine() -> "SemanticEngine":
    """Instantiate the SemanticEngine from AIVC_STORAGE_ROOT."""
    from aivc.config import get_storage_root
    storage_root = get_storage_root(allow_fallback=True)

    from aivc.semantic.engine import SemanticEngine
    return SemanticEngine(storage_root)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------



def cmd_log(args: argparse.Namespace) -> None:
    engine = _get_engine()
    commits = engine.get_log(limit=args.limit)
    
    if not commits:
        print("No memories found in history.")
        return

    from aivc.config import get_machine_id
    local_id = get_machine_id()

    for c in commits:
        files = engine.get_memory_files(c.id)
        files_str = ", ".join(files) if files else "—"
        
        m_id = c.machine_id or "unknown"
        machine_tag = f" {MAGENTA}[{m_id}]{RESET}" if m_id != local_id else ""

        print(f"{YELLOW}memory {c.id}{machine_tag}{RESET}")
        print(f"{DIM}Date:{RESET}    {c.timestamp}")
        print(f"{DIM}Files:{RESET}   {files_str}")
        print(f"\n    {BOLD}{c.title}{RESET}\n")


def cmd_search(args: argparse.Namespace) -> None:
    engine = _get_engine()
    if args.glob:
        print(f"{DIM}Recalling memories for: '{args.query}' (filter: '{args.glob}')...{RESET}\n")
    else:
        print(f"{DIM}Recalling memories for: '{args.query}'...{RESET}\n")

    # Ensure ML models are loaded and orphaned commits are reindexed
    import sys
    if sys.platform == "win32" or (os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()):
        print(f"{YELLOW}Chargement des modèles ML (cette étape prend ~10-15s à froid sous Windows)...{RESET}", flush=True)

    engine.warmup()
    
    results = engine.search(args.query, top_n=args.top_n, filter_glob=args.glob)
    
    if not results:
        print("No matching memories found.")
        return

    from aivc.config import get_machine_id
    local_id = get_machine_id()

    for i, r in enumerate(results, 1):
        m_id = r.machine_id or "unknown"
        machine_tag = f" {MAGENTA}[{m_id}]{RESET}" if m_id != local_id else ""
        
        print(f"{CYAN}{BOLD}{i}. {r.title}{RESET}{machine_tag} {DIM}(score: {r.score:.3f}){RESET}")
        print(f"   {DIM}ID:{RESET}    {r.memory_id}")
        print(f"   {DIM}Date:{RESET}  {r.timestamp}")
        print(f"   {DIM}Files:{RESET} {', '.join(r.file_paths) if r.file_paths else '—'}")
        print(f"\n      {r.snippet}\n")


def cmd_migrate(args: argparse.Namespace) -> None:
    """Explicitly migrate JSON commits to SQLite index."""
    print(f"{DIM}Checking for JSON commits to migrate to SQLite index...{RESET}")
    engine = _get_engine()
    engine.migrate_index()
    print(f"{GREEN}Migration complete.{RESET}")


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the Web Dashboard server."""
    from aivc.web.dashboard import main as dashboard_main
    # Override sys.argv so dashboard's own argparse picks up the port
    sys.argv = ["aivc-web", "--port", str(args.port)]
    dashboard_main()


def cmd_config(args: argparse.Namespace) -> None:
    """Manage AIVC configuration."""
    from aivc.config import get_aivc_config, save_aivc_config
    config = get_aivc_config()

    if args.key and args.value:
        # Simple nested key support (e.g. sync.enabled)
        parts = args.key.split(".")
        target = config
        for p in parts[:-1]:
            if p not in target: target[p] = {}
            target = target[p]
        
        # Type conversion attempt
        val = args.value
        if val.lower() == "true": val = True
        elif val.lower() == "false": val = False
        elif val.isdigit(): val = int(val)
        
        target[parts[-1]] = val
        save_aivc_config(config)
        print(f"{GREEN}Config updated: {args.key} = {val}{RESET}")
    else:
        import json
        print(f"{YELLOW}{BOLD}Current AIVC Configuration:{RESET}")
        print(json.dumps(config, indent=2))


RED = "\033[0;31m"


def cmd_sync_setup(args: argparse.Namespace) -> None:
    """Guide the user through Google Drive sync setup (native OAuth)."""
    from aivc.config import get_aivc_config, save_aivc_config, get_credentials_path, get_token_path

    config = get_aivc_config()
    if "sync" not in config:
        config["sync"] = {}

    print(f"{CYAN}{BOLD}=== AIVC Cloud Sync Setup (Google Drive) ==={RESET}")
    print("This will configure AIVC to sync your memory across machines via Google Drive.\n")

    # Step 1: Check if already authenticated
    token_path = get_token_path()
    if token_path.exists():
        print(f"{GREEN}{CHECK_MARK} Existing authentication found at {token_path}{RESET}")
        reauth = input("Re-authenticate? (y/n, default: n): ").strip().lower()
        if reauth != "y":
            config["sync"]["enabled"] = True
            save_aivc_config(config)
            print(f"\n{GREEN}Cloud sync is ENABLED.{RESET}")
            return

    # Step 2: Guide user to get credentials
    print(f"{YELLOW}{BOLD}--- Step 1: Create Google Cloud Credentials ---{RESET}")
    print("""
To sync via Google Drive, you need a Google Cloud OAuth Client ID.
Follow these steps (takes ~2 minutes):

  1. Go to: https://console.cloud.google.com/apis/credentials
  2. Create a project (or select an existing one).
  3. Click "+ CREATE CREDENTIALS" → "OAuth client ID".
  4. If prompted, configure the consent screen:
     - User Type: "External" → Create
     - App name: "AIVC" → Save (skip optional fields)
     - Add yourself as a test user → Save
  5. Back in Credentials:
     - Application type: "Desktop app"
     - Name: "AIVC CLI"
     - Click "Create"
  6. Copy the Client ID and Client Secret shown.

--- Step 2: Enable the Google Drive API ---

  You MUST enable the API for your project before syncing:
  → https://console.cloud.google.com/apis/library/drive.googleapis.com
  Click "ENABLE" and wait 1 minute.
""")

    client_id = input(f"{BOLD}Paste your Client ID:{RESET} ").strip()
    if not client_id:
        print(f"{RED}Aborted: Client ID is required.{RESET}")
        return

    client_secret = input(f"{BOLD}Paste your Client Secret:{RESET} ").strip()
    if not client_secret:
        print(f"{RED}Aborted: Client Secret is required.{RESET}")
        return

    # Step 3: Run OAuth flow
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    scopes = ["https://www.googleapis.com/auth/drive.file"]

    if getattr(args, "headless", False):
        try:
            from google_auth_oauthlib.flow import Flow
        except ImportError:
            print(f"{RED}Error: Google auth libraries not installed.{RESET}")
            print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return

        flow = Flow.from_client_config(
            client_config,
            scopes=scopes,
            redirect_uri="http://localhost"
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            prompt="consent",
            include_granted_scopes="true"
        )

        print(f"\n{YELLOW}{BOLD}--- Headless Authentication ---{RESET}")
        print("Please visit the following URL in a browser on any machine to authenticate:")
        print(f"\n  {CYAN}{auth_url}{RESET}\n")
        print("After approving, you will be redirected to a page that fails to load like:")
        print("  http://localhost/?state=...&code=...")
        print("Copy the ENTIRE redirect URL from your browser's address bar and paste it below.")
        print("(Alternatively, you can copy and paste just the 'code' parameter value.)\n")

        user_input = input(f"{BOLD}Paste redirect URL or Authorization Code:{RESET} ").strip()
        if not user_input:
            print(f"{RED}Aborted: Authentication input is required.{RESET}")
            return

        from urllib.parse import urlparse, parse_qs
        code = user_input
        if "code=" in user_input or user_input.startswith(("http://", "https://")):
            query = parse_qs(urlparse(user_input).query)
            if "code" in query and query["code"]:
                code = query["code"][0]

        print(f"\n{DIM}Exchanging authorization code for access token...{RESET}")
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
        except Exception as e:
            print(f"{RED}Error: Failed to fetch token: {e}{RESET}")
            return
    else:
        print(f"\n{DIM}Opening your browser for Google authorization...{RESET}")
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            print(f"{RED}Error: Google auth libraries not installed.{RESET}")
            print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return

        flow = InstalledAppFlow.from_client_config(client_config, scopes)
        creds = flow.run_local_server(port=0)

    # Save credentials and token
    creds_path = get_credentials_path()
    creds_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    creds_path.write_text(json.dumps(client_config), encoding="utf-8")
    token_path.write_text(creds.to_json(), encoding="utf-8")

    # Update config
    config["sync"]["enabled"] = True
    save_aivc_config(config)

    print(f"\n{GREEN}{BOLD}{CHECK_MARK} Authentication successful!{RESET}")
    print(f"  Token saved to: {token_path}")
    print("  Config saved to: ~/.aivc/config.json")
    print(f"\n{YELLOW}Cloud sync is now ENABLED.{RESET}")
    print("AIVC will automatically sync memory metadata (notes) to your Google Drive.")
    print(f"{DIM}Note: File contents (blobs) are not synchronized as of Phase 29.{RESET}")


def cmd_sync_push(args: argparse.Namespace) -> None:
    """Force sync (push) of all missing local commits to Google Drive."""
    from aivc.config import get_aivc_config, get_token_path

    config = get_aivc_config()
    sync_cfg = config.get("sync", {})
    
    if not sync_cfg.get("enabled"):
        print(f"{RED}Cloud sync is currently disabled. Run 'aivc sync setup' first.{RESET}")
        return
        
    if not get_token_path().exists():
        print(f"{RED}Google Drive authentication missing. Run 'aivc sync setup' first.{RESET}")
        return

    print(f"{CYAN}{BOLD}AIVC Force Sync Push{RESET}")
    print(f"{DIM}Analyzing local memories and comparing with Google Drive...{RESET}")
    
    try:
        from aivc.sync.drive import NativeDriveSyncManager
        manager = NativeDriveSyncManager(Path.home() / ".aivc" / "storage")
        result = manager.push_missing()
        
        commits = result.get("memories_pushed", 0)
        
        if commits == 0:
            print(f"\n{GREEN}{CHECK_MARK} Everything is up-to-date! No missing local memories found.{RESET}")
        else:
            print(f"\n{GREEN}{CHECK_MARK} Sync Push complete!{RESET}")
            print(f"  Pushed {commits} missing memory/ies.")
            
    except Exception as e:
        print(f"{RED}Error during sync push: {e}{RESET}")


def cmd_sync_semantic(args: argparse.Namespace) -> None:
    """Synchronize semantic task states and archives to Google Drive as native Docs."""
    from aivc.config import get_storage_root
    from aivc.sync.semantic_export import SemanticDriveExporter

    storage_root = get_storage_root(allow_fallback=True)
    exporter = SemanticDriveExporter(storage_root=storage_root)

    print(f"{CYAN}{BOLD}AIVC Semantic Drive Export:{RESET}")
    print(f"  {DIM}Storage root:{RESET} {storage_root}")
    print(f"  {DIM}Connecting to Google Drive and scanning tasks...{RESET}")

    result = exporter.export_all(batch_size=getattr(args, "batch_size", 25))
    if result.get("status") == "success":
        print(f"\n{GREEN}{CHECK_MARK} Semantic Export complete!{RESET}")
        print(f"  Live docs updated:     {result.get('synced_live')}/3")
        print(f"  Archive docs updated:  {result.get('synced_archives')}")
        counts = result.get("live_counts", {})
        print(f"  Active tasks count:    new={counts.get('new')}, await_validation={counts.get('await_validation')}, validated={counts.get('validated')}")
    else:
        print(f"{RED}Error during semantic export: {result.get('error', 'Unknown error')}{RESET}")

def _load_dream_inbox_processed() -> dict:
    """Load processed documents registry locally and merge with Google Drive AIVC_Sync if available."""
    from aivc.config import get_aivc_dir
    local_path = get_aivc_dir() / "dream_inbox_processed.json"
    data = {"processed": {}}
    if local_path.exists():
        try:
            loaded = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "processed" in loaded:
                data = loaded
        except Exception as e:
            print(f"Warning: could not parse local dream_inbox_processed.json: {e}", file=sys.stderr)

    # Attempt to sync/merge from Google Drive AIVC_Sync (scope drive.file compatible)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from aivc.sync.drive import NativeDriveSyncManager
            from aivc.config import get_storage_root
            mgr = NativeDriveSyncManager(get_storage_root())
            if getattr(mgr, "enabled", False):
                service = mgr._get_service()
                root_id = mgr._get_root_folder_id()
                q = f"name = 'dream_inbox_processed.json' and '{root_id}' in parents and trashed = false"
                res = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
                files = res.get("files", [])
                if files:
                    content = service.files().get_media(fileId=files[0]["id"]).execute().decode("utf-8")
                    remote_data = json.loads(content)
                    if isinstance(remote_data, dict) and "processed" in remote_data:
                        for k, v in remote_data["processed"].items():
                            if k not in data["processed"]:
                                data["processed"][k] = v
        except Exception as e:
            print(f"Warning: could not sync dream_inbox_processed from Drive: {e}", file=sys.stderr)

    return data


def _save_dream_inbox_processed(data: dict) -> None:
    """Save processed documents registry locally and to Google Drive AIVC_Sync."""
    from aivc.config import get_aivc_dir
    local_path = get_aivc_dir() / "dream_inbox_processed.json"
    try:
        local_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to save local dream inbox registry: {e}", file=sys.stderr)

    # Attempt to sync to Google Drive AIVC_Sync (scope drive.file compatible)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from aivc.sync.drive import NativeDriveSyncManager
            from aivc.config import get_storage_root
            from googleapiclient.http import MediaInMemoryUpload
            mgr = NativeDriveSyncManager(get_storage_root())
            if getattr(mgr, "enabled", False):
                service = mgr._get_service()
                root_id = mgr._get_root_folder_id()
                q = f"name = 'dream_inbox_processed.json' and '{root_id}' in parents and trashed = false"
                res = service.files().list(q=q, spaces="drive", fields="files(id, name)").execute()
                files = res.get("files", [])
                body_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
                media = MediaInMemoryUpload(body_bytes, mimetype="application/json")
                if files:
                    service.files().update(fileId=files[0]["id"], media_body=media).execute()
                else:
                    meta = {"name": "dream_inbox_processed.json", "parents": [root_id]}
                    service.files().create(body=meta, media_body=media).execute()
        except Exception as e:
            print(f"Warning: could not save dream_inbox_processed to Drive: {e}", file=sys.stderr)


_DREAM_INBOX_FOLDER_IDS_CACHE: list[str] | None = None


def _get_dream_inbox_folder_ids(use_cache: bool = True) -> list[str]:
    """Retrieve all Google Drive folder IDs named 'Dream_Inbox' at root."""
    global _DREAM_INBOX_FOLDER_IDS_CACHE
    if use_cache and _DREAM_INBOX_FOLDER_IDS_CACHE:
        return _DREAM_INBOX_FOLDER_IDS_CACHE

    from aivc.config import get_aivc_dir
    cache_file = get_aivc_dir() / "dream_inbox_folders.json"
    if use_cache and cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cache_data, dict) and "folder_ids" in cache_data:
                cached_ids = cache_data["folder_ids"]
                cached_time = cache_data.get("timestamp", 0)
                import time
                if time.time() - cached_time < 3600 and cached_ids:
                    _DREAM_INBOX_FOLDER_IDS_CACHE = cached_ids
                    return cached_ids
        except Exception as e:
            print(f"Warning: Failed to read Dream_Inbox folder cache: {e}", file=sys.stderr)

    import subprocess
    folder_ids = []
    try:
        out = subprocess.check_output(
            ["rclone", "lsf", "--dirs-only", "--format", "ip", "gdrive:"],
            stderr=subprocess.DEVNULL,
            encoding="utf-8"
        )
        for line in out.splitlines():
            line = line.strip()
            if not line or ";" not in line:
                continue
            parts = line.split(";", 1)
            f_id, f_path = parts[0].strip(), parts[1].strip()
            if f_path.rstrip("/") == "Dream_Inbox":
                folder_ids.append(f_id)
    except Exception as e:
        print(f"Warning: Failed to list Dream_Inbox folder IDs via rclone lsf: {e}", file=sys.stderr)

    if not folder_ids:
        folder_ids = ["1ZLsrkwRBhH2Xz82eooQC3ahsePFjm28h", "1JsNgV7uD1c981dNQpYEprKdjVgitLDo8"]

    _DREAM_INBOX_FOLDER_IDS_CACHE = folder_ids
    try:
        import time
        cache_file.write_text(json.dumps({"folder_ids": folder_ids, "timestamp": time.time()}), encoding="utf-8")
    except Exception as e:
        print(f"Warning: Failed to write Dream_Inbox folder cache: {e}", file=sys.stderr)

    return folder_ids


def _list_all_dream_inbox_files(folder_ids: list[str] | None = None) -> list[dict]:
    """Scan all Dream_Inbox root folders and return deduplicated list of files with FolderID attached."""
    import subprocess
    import json
    if folder_ids is None:
        folder_ids = _get_dream_inbox_folder_ids()

    seen_ids = set()
    all_files = []
    for fid in folder_ids:
        try:
            out = subprocess.check_output(
                ["rclone", "lsjson", "gdrive:", "--drive-root-folder-id", fid],
                stderr=subprocess.DEVNULL,
                encoding="utf-8"
            )
            files = json.loads(out)
            for f in files:
                f_id = f.get("ID")
                if f_id and f_id in seen_ids:
                    continue
                if f_id:
                    seen_ids.add(f_id)
                f["FolderID"] = fid
                all_files.append(f)
        except Exception as e:
            print(f"Warning: Failed to list files in folder {fid}: {e}", file=sys.stderr)
    return all_files


def _get_rclone_config_path() -> Path:
    """Resolve path to rclone.conf."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            p = Path(appdata) / "rclone" / "rclone.conf"
            if p.exists():
                return p
    default_p = Path.home() / ".config" / "rclone" / "rclone.conf"
    if default_p.exists():
        return default_p
    try:
        import subprocess
        out = subprocess.check_output(["rclone", "config", "file"], stderr=subprocess.DEVNULL, encoding="utf-8")
        for line in out.splitlines():
            line = line.strip()
            if line and not line.startswith("Configuration"):
                p = Path(line)
                if p.exists():
                    return p
    except Exception as e:
        print(f"Warning: could not resolve rclone config via CLI ({e}); using APPDATA default", file=sys.stderr)
    return Path(os.environ.get("APPDATA", "")) / "rclone" / "rclone.conf"


def _read_token_from_rclone_conf(conf_path: Path) -> tuple[str, str] | None:
    """Return (access_token, expiry_iso) from [gdrive] or [gdrive_rw] in rclone.conf."""
    import configparser
    cp = configparser.RawConfigParser()
    try:
        cp.read(conf_path, encoding="utf-8")
    except Exception:
        return None
    section = None
    if cp.has_section("gdrive") and cp.has_option("gdrive", "token"):
        section = "gdrive"
    elif cp.has_section("gdrive_rw") and cp.has_option("gdrive_rw", "token"):
        section = "gdrive_rw"
    if not section:
        return None
    raw_tok = cp.get(section, "token")
    try:
        tok_data = json.loads(raw_tok)
        return tok_data.get("access_token"), tok_data.get("expiry")
    except Exception:
        return None


def _get_rclone_drive_access_token() -> str:
    """Retrieve active Google Drive OAuth access_token from rclone, refreshing if needed."""
    import subprocess
    import datetime
    conf_path = _get_rclone_config_path()
    token_info = _read_token_from_rclone_conf(conf_path) if conf_path.exists() else None

    needs_refresh = True
    if token_info and token_info[0] and token_info[1]:
        try:
            exp_str = token_info[1]
            exp_dt = datetime.datetime.fromisoformat(exp_str)
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=datetime.timezone.utc)
            diff_sec = (exp_dt - now_dt).total_seconds()
            if diff_sec > 120:
                needs_refresh = False
        except Exception:
            needs_refresh = True

    if needs_refresh:
        try:
            res = subprocess.run(["rclone", "about", "gdrive:"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            if res.returncode != 0:
                print(f"Warning: rclone about gdrive: returned code {res.returncode}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to refresh token via rclone about gdrive: {e}", file=sys.stderr)
        token_info = _read_token_from_rclone_conf(conf_path) if conf_path.exists() else None

    if token_info and token_info[0]:
        return token_info[0]
    raise RuntimeError("Failed to obtain Google Drive access token from rclone configuration.")


def _drive_get_start_page_token(conn: http.client.HTTPSConnection, access_token: str) -> str:
    """Fetch initial startPageToken for Drive changes.list with transient retry."""
    import time
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Connection": "keep-alive"
    }
    for attempt in range(3):
        conn.request("GET", "/drive/v3/changes/startPageToken?supportsAllDrives=true", headers=headers)
        resp = conn.getresponse()
        if resp.status == 401:
            raise PermissionError("HTTP 401 Unauthorized: access token expired")
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status == 200:
            data = json.loads(body)
            return data["startPageToken"]
        if resp.status in (403, 429) and ("RATE_LIMIT_EXCEEDED" in body or "rateLimitExceeded" in body or resp.status == 429) and attempt < 2:
            time.sleep(2 * (attempt + 1))
            continue
        raise RuntimeError(f"Failed to get startPageToken (HTTP {resp.status}): {body}")
    raise RuntimeError("Failed to get startPageToken after retries.")


def _drive_poll_changes(conn: http.client.HTTPSConnection, access_token: str, page_token: str) -> tuple[str, list[dict]]:
    """Poll Drive changes.list from page_token and return (new_start_page_token, changes)."""
    import urllib.parse
    all_changes = []
    current_token = page_token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Connection": "keep-alive"
    }
    while True:
        params = {
            "pageToken": current_token,
            "includeRemoved": "false",
            "spaces": "drive",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "fields": "newStartPageToken,nextPageToken,changes(fileId,removed,file(id,name,parents,trashed,mimeType,modifiedTime))"
        }
        path = f"/drive/v3/changes?{urllib.parse.urlencode(params)}"
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        if resp.status == 401:
            raise PermissionError("HTTP 401 Unauthorized: access token expired")
        if resp.status != 200:
            body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to poll changes (HTTP {resp.status}): {body}")
        data = json.loads(resp.read().decode("utf-8"))
        changes = data.get("changes", [])
        all_changes.extend(changes)
        if "nextPageToken" in data and data["nextPageToken"]:
            current_token = data["nextPageToken"]
        elif "newStartPageToken" in data and data["newStartPageToken"]:
            return data["newStartPageToken"], all_changes
        else:
            return current_token, all_changes


def _extract_dream_inbox_doc_text(folder_id: str, file_name: str, file_id: str = "", mime_type: str = "") -> str:
    """Download and extract full text from a document in Dream_Inbox.

    Attempts direct REST API export/download via Google Drive v3 if file_id is provided,
    avoiding rclone subprocess overhead and handling native Google Docs seamlessly.
    Falls back to rclone copy if REST fails or file_id is not provided.
    """
    import tempfile
    import zipfile
    import xml.etree.ElementTree as ET
    import subprocess
    import io

    # 1. Direct REST API attempt via Google Drive v3 if file_id is provided
    if file_id:
        try:
            access_token = _get_rclone_drive_access_token()
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Connection": "close",
            }

            # Try export as text/plain (native Google Docs)
            conn = http.client.HTTPSConnection("www.googleapis.com", timeout=20)
            conn.request("GET", f"/drive/v3/files/{file_id}/export?mimeType=text%2Fplain", headers=headers)
            resp = conn.getresponse()
            if resp.status == 200:
                text = resp.read().decode("utf-8", errors="replace")
                conn.close()
                return text
            resp.read()  # Consume response body
            conn.close()

            # If export failed (e.g. HTTP 400/403 fileNotExportable for binary files), try alt=media
            conn = http.client.HTTPSConnection("www.googleapis.com", timeout=20)
            conn.request("GET", f"/drive/v3/files/{file_id}?alt=media", headers=headers)
            resp = conn.getresponse()
            if resp.status == 200:
                raw_bytes = resp.read()
                conn.close()
                if zipfile.is_zipfile(io.BytesIO(raw_bytes)):
                    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                        xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paras = []
                    for p in tree.iterfind(".//w:p", ns):
                        t_parts = [t.text for t in p.iterfind(".//w:t", ns) if t.text]
                        if t_parts:
                            paras.append("".join(t_parts))
                    return "\n".join(paras)
                else:
                    return raw_bytes.decode("utf-8", errors="replace")
            resp.read()
            conn.close()
        except Exception as e:
            print(f"{DIM}Note: Direct REST fetch for {file_name} ({file_id}) failed ({e}); falling back to rclone...{RESET}", file=sys.stderr)

    # 2. Fallback to rclone copy
    rclone_name = file_name
    if mime_type == "application/vnd.google-apps.document" and not rclone_name.endswith(".docx"):
        rclone_name = f"{rclone_name}.docx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = ["rclone", "copy", "--drive-root-folder-id", folder_id, f"gdrive:{rclone_name}", str(tmp_path)]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        target_file = tmp_path / rclone_name
        if not target_file.exists():
            candidates = list(tmp_path.iterdir())
            if candidates:
                target_file = candidates[0]
            else:
                raise FileNotFoundError(f"File {file_name} not found after rclone copy from folder {folder_id}")

        if zipfile.is_zipfile(target_file):
            with zipfile.ZipFile(target_file) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paras = []
            for p in tree.iterfind(".//w:p", ns):
                t_parts = [t.text for t in p.iterfind(".//w:t", ns) if t.text]
                if t_parts:
                    paras.append("".join(t_parts))
            return "\n".join(paras)
        else:
            return target_file.read_text(encoding="utf-8", errors="replace")


def _find_agy_executable() -> Path:
    """Resolve path to agy executable."""
    import shutil
    agy = shutil.which("agy")
    if agy:
        return Path(agy)
    candidates = [
        Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "agy" / "bin" / "agy.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Antigravity" / "bin" / "agy.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("agy.exe")


def _save_dream_inbox_local_copy(doc_name: str, doc_id: str, doc_text: str) -> Path:
    """Save full text copy of detected Dream_Inbox document to local archive immediately."""
    import re
    archive_dir = Path(r"C:\Users\Public\aivc\dream_inbox_archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    clean_stem = doc_name
    for ch in '<>:"/\\|?*':
        clean_stem = clean_stem.replace(ch, "_")
    clean_stem = re.sub(r"\.(docx|txt|md)$", "", clean_stem, flags=re.IGNORECASE).strip()
    if not clean_stem:
        clean_stem = f"doc_{doc_id}"
    local_path = archive_dir / f"{clean_stem}.md"
    local_path.write_text(doc_text, encoding="utf-8")
    return local_path.resolve()


def _trash_drive_file_by_id(doc_id: str | None) -> bool:
    """Trash a file in Google Drive by doc_id via direct REST PATCH.

    Returns True if successfully moved to Google Drive Trash (HTTP 200/204),
    or False with explicit diagnostic if scopes are insufficient or file is uneditable.
    """
    if not doc_id:
        return False

    try:
        token = _get_rclone_drive_access_token()
        conn = http.client.HTTPSConnection("www.googleapis.com", timeout=20)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        conn.request("PATCH", f"/drive/v3/files/{doc_id}?supportsAllDrives=true", body=b'{"trashed": true}', headers=headers)
        resp = conn.getresponse()
        status = resp.status
        resp_body = resp.read().decode("utf-8", errors="replace")
        conn.close()
        if status in (200, 204):
            return True
        try:
            err_data = json.loads(resp_body)
            err_msg = err_data.get("error", {}).get("message", resp_body[:160])
        except Exception:
            err_msg = resp_body[:160]
        print(f"  {YELLOW}Note API Drive: Impossible de mettre le fichier {doc_id} à la corbeille distante (HTTP {status}: {err_msg}).{RESET}", file=sys.stderr)
    except Exception as e:
        print(f"  {YELLOW}Note API Drive: Échec de la requête REST PATCH pour {doc_id} ({e}).{RESET}", file=sys.stderr)

    return False


def _record_dream_inbox_processed_doc(
    doc_id: str | None = None,
    doc_name: str | None = None,
    method: str = "manual_trash",
    local_path: str | None = None,
    extra_key: str | None = None,
) -> None:
    """Record a document as processed in both local dream_inbox_processed.json and Drive AIVC_Sync."""
    import datetime
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    reg = _load_dream_inbox_processed()
    primary_key = doc_id or doc_name or extra_key
    if not primary_key:
        return
    entry = {
        "trashed_at": now_iso,
        "method": method,
        "name": doc_name or extra_key or primary_key,
    }
    if doc_id:
        entry["doc_id"] = doc_id
    if local_path:
        entry["local_path"] = local_path

    processed = reg.setdefault("processed", {})
    processed[primary_key] = entry
    if doc_id:
        processed[doc_id] = entry
    if doc_name:
        processed[doc_name] = entry
        stem = Path(doc_name).stem
        if stem:
            processed[stem] = entry
            processed[f"{stem}.docx"] = entry
            processed[f"{stem}.md"] = entry
    if extra_key:
        processed[extra_key] = entry
    _save_dream_inbox_processed(reg)



def _watch_dream_inbox(args: argparse.Namespace) -> None:
    """Differential event-driven watcher for Google Drive Dream_Inbox via changes.list."""
    import subprocess
    import time
    import datetime

    interval = getattr(args, "interval", 15)
    once = getattr(args, "once", False)
    dry_run = getattr(args, "dry_run", False)
    workspace = getattr(args, "workspace", None) or str(Path.home() / "Documents" / "VoiceNotes")
    model = getattr(args, "model", "gemini-4-argon-high") or "gemini-4-argon-high"

    _watch_code_mtime = Path(__file__).stat().st_mtime

    print(f"{CYAN}{BOLD}=== AIVC Dream_Inbox Event Watcher ==={RESET}")
    print(f"  {DIM}Workspace:{RESET} {workspace}")
    print(f"  {DIM}Model:{RESET}     {model}")
    print(f"  {DIM}Interval:{RESET}  {interval}s (mode: {'once' if once else 'continuous'}{', dry-run' if dry_run else ''})")

    folder_ids = _get_dream_inbox_folder_ids()
    folder_ids_set = set(folder_ids)
    print(f"  {DIM}Dossiers Dream_Inbox détectés ({len(folder_ids)}) :{RESET} {', '.join(folder_ids)}")

    access_token = _get_rclone_drive_access_token()
    conn = http.client.HTTPSConnection("www.googleapis.com", timeout=30)
    try:
        page_token = _drive_get_start_page_token(conn, access_token)
        print(f"  {GREEN}{CHECK_MARK} Connexion API Google Drive établie. startPageToken initial: {page_token}{RESET}\n")
    except Exception as e:
        conn.close()
        raise RuntimeError(f"Échec d'initialisation startPageToken: {e}")

    agy_path = _find_agy_executable()

    def _process_doc(doc_name: str, doc_id: str, fid: str, is_demo: bool = False, mime_type: str = "") -> bool:
        print(f"{YELLOW}{BOLD}>>> Nouveau document détecté : {doc_name} (ID: {doc_id}, Folder: {fid}){' [DRY-RUN DEMO SUR DERNIER DOC]' if is_demo else ''}{RESET}")
        try:
            doc_text = _extract_dream_inbox_doc_text(fid, doc_name, file_id=doc_id, mime_type=mime_type)
        except Exception as ex:
            print(f"{RED}Erreur lors de l'extraction du document {doc_name}: {ex}{RESET}")
            return False

        lines = doc_text.splitlines()
        preview = "\n".join(lines[:10]) + ("\n..." if len(lines) > 10 else "")
        print(f"{DIM}Extrait du document ({len(doc_text)} caractères, {len(lines)} lignes) :{RESET}")
        print(f"{DIM}--------------------------------------------------{RESET}")
        print(preview)
        print(f"{DIM}--------------------------------------------------{RESET}")

        local_archive_path = _save_dream_inbox_local_copy(doc_name, doc_id, doc_text)
        print(f"  {GREEN}{CHECK_MARK} Copie locale sauvegardée immédiatement : {local_archive_path} ({len(doc_text)} caractères){RESET}")

        prompt = (
            f"Tu es réveillé en mode événementiel par `aivc dream-inbox watch` suite à l'arrivée du document Google Drive `{doc_name}` (ID: `{doc_id}`) dans `Dream_Inbox`.\n"
            f"Une copie locale intégrale du document a été sauvegardée sur le disque à l'emplacement suivant :\n"
            f"Fichier local : `{local_archive_path}`\n"
            f"Le fichier distant sur Google Drive a déjà été supprimé dès le début de la tâche pour éviter toute double exécution.\n\n"
            f"Instructions d'exécution immédiate (mode ciblé multi-projets, sans durée forcée de 45 min) :\n"
            f"1. Lis directement le fichier local `{local_archive_path}`. Identifie tous les projets ciblés par ce document, mets à jour sobrement les notes maîtresses Obsidian concernées (avec les citations exactes de la section 'Verbatim Exact d'Henri') et ajuste les scores de priorité via `project_memory_cli.py set-score` si demandé.\n"
            f"2. Lance en parallèle un sous-agent `superviseur-projet` (`Model: 'inherit'`) distinct sur chaque projet concerné.\n"
            f"3. RÈGLE STRICTE DE NON-RÉÉCRITURE : Ne réécris JAMAIS et ne reformule JAMAIS le propos d'Henri lorsque tu invoques un sous-agent `superviseur-projet`. Dans l'artefact de mission que tu passes à chaque `superviseur-projet`, donne-lui directement le lien/chemin vers le fichier téléchargé localement (`{local_archive_path}`) et demande-lui de lire directement ce fichier et de traiter uniquement les tâches qui concernent son projet dans ce fichier.\n"
            f"4. Enregistre chaque livrable, réponse ou clôture de tâche via `mcp-cli call aivc task_commit` sur la note maîtresse de chaque projet concerné afin que les Google Docs `Live_*` dans `AIVC_Sync/` soient mis à jour pour Gemini.\n"
        )

        agy_cmd = [str(agy_path), "--mode", "auto", "--model", model, "-p", prompt]
        if dry_run or is_demo:
            print(f"{CYAN}{BOLD}[DRY-RUN] Commande agy.exe préparée :{RESET}")
            print(f"  Executable:  {agy_path}")
            print(f"  Working Dir: {workspace}")
            print(f"  Model:       {model}")
            print(f"  Prompt size: {len(prompt)} caractères")
            print(f"  Local Copy:  {local_archive_path}")
            print(f"  Mode:        --mode auto --model {model} -p <prompt>")
            return True

        _record_dream_inbox_processed_doc(
            doc_id=doc_id,
            doc_name=doc_name,
            method="watch_event_start",
            local_path=str(local_archive_path),
        )

        trashed_ok = _trash_drive_file_by_id(doc_id)
        if trashed_ok:
            print(f"  {GREEN}{CHECK_MARK} Fichier distant supprimé de Google Drive dès le début de la tâche (ID: {doc_id}).{RESET}")
        else:
            print(f"  {YELLOW}Mise à la corbeille distante non confirmée pour {doc_id} : document verrouillé dans dream_inbox_processed.json.{RESET}")

        print(f"{CYAN}Exécution d'agy.exe pour traiter le document...{RESET}")
        res = subprocess.run(agy_cmd, cwd=workspace)
        if res.returncode == 0:
            print(f"{GREEN}{CHECK_MARK} Traitement agy réussi (returncode 0). Archive locale conservée dans {local_archive_path}.{RESET}")
            return True
        else:
            print(f"{RED}Erreur: agy.exe a échoué avec le code retour {res.returncode}. Archive locale conservée dans {local_archive_path}.{RESET}")
            return False

    reg = _load_dream_inbox_processed()
    processed_map = reg.get("processed", {})
    all_existing_files = _list_all_dream_inbox_files(folder_ids)

    unprocessed = []
    for f in all_existing_files:
        f_name = f.get("Name", "")
        f_id = f.get("ID", "")
        f_stem = Path(f_name).stem if f_name else ""
        if f_id not in processed_map and f_name not in processed_map and f_stem not in processed_map:
            unprocessed.append(f)

    if unprocessed:
        print(f"{YELLOW}Scan initial : {len(unprocessed)} document(s) non traité(s) détecté(s).{RESET}")
        for f in unprocessed:
            _process_doc(f.get("Name"), f.get("ID"), f.get("FolderID", folder_ids[0]), mime_type=f.get("MimeType") or f.get("mimeType", ""))
    else:
        print(f"{GREEN}{CHECK_MARK} Scan initial : aucun document non traité en attente.{RESET}")
        if dry_run and once and all_existing_files:
            sorted_files = sorted(all_existing_files, key=lambda x: x.get("ModTime", ""), reverse=True)
            demo_f = sorted_files[0]
            _process_doc(demo_f.get("Name"), demo_f.get("ID"), demo_f.get("FolderID", folder_ids[0]), is_demo=True, mime_type=demo_f.get("MimeType") or demo_f.get("mimeType", ""))

    if once:
        print(f"\n{GREEN}{CHECK_MARK} Cycle unique terminé (--once).{RESET}")
        conn.close()
        return

    print(f"\n{CYAN}Démarrage de la boucle de surveillance (intervalle: {interval}s)... Pressez Ctrl+C pour arrêter.{RESET}")
    err_count = 0
    try:
        while True:
            time.sleep(interval)
            current_mtime = Path(__file__).stat().st_mtime
            if current_mtime != _watch_code_mtime:
                print(f"\n{YELLOW}{BOLD}Modification du code source détectée ({Path(__file__).name}). Rechargement à chaud du processus...{RESET}")
                conn.close()
                os.execv(sys.executable, [sys.executable, "-m", "aivc.cli"] + sys.argv[1:])

            try:
                try:
                    new_token, changes = _drive_poll_changes(conn, access_token, page_token)
                except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError, PermissionError):
                    try:
                        conn.close()
                    except Exception as e:
                        print(f"Warning: Exception closing connection on reset: {e}", file=sys.stderr)
                    access_token = _get_rclone_drive_access_token()
                    conn = http.client.HTTPSConnection("www.googleapis.com", timeout=30)
                    new_token, changes = _drive_poll_changes(conn, access_token, page_token)

                page_token = new_token
                err_count = 0

                for ch in changes:
                    f_info = ch.get("file", {})
                    if not f_info or f_info.get("trashed"):
                        continue
                    f_name = f_info.get("name", "")
                    f_id = ch.get("fileId") or f_info.get("id")
                    f_mime = f_info.get("mimeType", "")
                    f_parents = f_info.get("parents", [])

                    is_inbox = any(p in folder_ids_set for p in f_parents)
                    if is_inbox:
                        reg = _load_dream_inbox_processed()
                        p_map = reg.get("processed", {})
                        f_stem = Path(f_name).stem if f_name else ""
                        if f_id not in p_map and f_name not in p_map and f_stem not in p_map:
                            target_fid = next((p for p in f_parents if p in folder_ids_set), folder_ids[0])
                            _process_doc(f_name, f_id, target_fid, mime_type=f_mime)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                err_count += 1
                backoff = min(interval * (2 ** err_count), 120)
                print(f"{RED}Erreur lors de l'interrogation Drive ({e}). Attente {backoff}s...{RESET}", file=sys.stderr)
                time.sleep(backoff)
    finally:
        try:
            conn.close()
        except Exception as e:
            print(f"Warning: Exception closing connection on exit: {e}", file=sys.stderr)


def cmd_dream_inbox(args: argparse.Namespace) -> None:
    """Inspect and manage Google Drive Dream_Inbox files for the /dream skill."""
    import subprocess
    import json
    import tempfile
    import zipfile
    import xml.etree.ElementTree as ET
    import sys
    import datetime

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation, OSError):
            pass

    action = args.inbox_action

    if action == "watch":
        _watch_dream_inbox(args)
        return

    folder_ids = _get_dream_inbox_folder_ids()

    def _resolve_dream_inbox_target(target_str: str) -> tuple[str | None, str | None, str | None]:
        files = _list_all_dream_inbox_files(folder_ids)
        for f in files:
            if f.get("Name") == target_str or f.get("ID") == target_str:
                return f.get("ID"), f.get("Name"), f.get("FolderID")
        if len(target_str) > 20 and "." not in target_str:
            return target_str, None, None
        return None, target_str, None

    if action == "list":
        print(f"{CYAN}{BOLD}Google Drive Dream_Inbox Listing:{RESET}")
        try:
            files = _list_all_dream_inbox_files(folder_ids)
            registry = _load_dream_inbox_processed()
            processed_map = registry.get("processed", {})
            include_all = getattr(args, "all", False)

            filtered_files = []
            for f in files:
                f_name = f.get("Name", "")
                f_id = f.get("ID", "")
                if not include_all and (f_id in processed_map or f_name in processed_map):
                    continue
                filtered_files.append(f)

            if not filtered_files:
                if files and not include_all:
                    print(f"  {GREEN}{CHECK_MARK} Dream_Inbox ne contient aucun nouveau document non traité ({len(files)} document(s) masqué(s) car déjà traité(s), utilisez '--all' pour afficher).{RESET}")
                else:
                    print(f"  {GREEN}{CHECK_MARK} Dream_Inbox est vide (0 fichier).{RESET}")
                return

            print(f"  {len(filtered_files)} fichier(s) trouvé(s) dans Dream_Inbox :")
            for i, f in enumerate(filtered_files, 1):
                f_name = f.get("Name")
                f_id = f.get("ID")
                f_fid = f.get("FolderID", "")[:8]
                f_mod = f.get("ModTime", "")[:19]
                status_tag = ""
                if f_id in processed_map or f_name in processed_map:
                    status_tag = f" {YELLOW}[déjà traité]{RESET}"
                print(f"  {i}. {BOLD}{f_name}{RESET} (ID: {f_id}, Dossier: {f_fid}..., Date: {f_mod}){status_tag}")
        except Exception as e:
            print(f"{RED}Error listing Dream_Inbox: {e}{RESET}")

    elif action == "read":
        target = args.target
        if not target:
            print(f"{RED}Error: please specify a document name or ID to read.{RESET}")
            return
        print(f"{DIM}Reading '{target}' from Dream_Inbox...{RESET}\n")

        doc_id, doc_name, doc_fid = _resolve_dream_inbox_target(target)
        target_name = doc_name or (f"{target}.docx" if len(target) > 20 and "." not in target else target)
        fid = doc_fid or (folder_ids[0] if folder_ids else "1ZLsrkwRBhH2Xz82eooQC3ahsePFjm28h")

        try:
            content = _extract_dream_inbox_doc_text(fid, target_name, file_id=doc_id)
            print(content)
        except Exception as e:
            # Fallback to direct download attempt via doc_id if extraction failed
            downloaded = False
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_target = Path(tmp_dir) / target_name
                if doc_id:
                    try:
                        from aivc.config import get_storage_root
                        from aivc.sync.drive import NativeDriveSyncManager
                        mgr = NativeDriveSyncManager(get_storage_root())
                        if getattr(mgr, "enabled", False):
                            service = mgr._get_service()
                            data = service.files().get_media(fileId=doc_id).execute()
                            local_target.write_bytes(data)
                            downloaded = True
                    except Exception as e:
                        print(f"  {YELLOW}Note: Native drive download failed for {doc_id}: {e}{RESET}", file=sys.stderr)
                if downloaded:
                    if zipfile.is_zipfile(local_target):
                        with zipfile.ZipFile(local_target) as z:
                            xml_content = z.read("word/document.xml")
                        tree = ET.fromstring(xml_content)
                        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                        paras = []
                        for p in tree.iterfind(".//w:p", ns):
                            t_parts = [t.text for t in p.iterfind(".//w:t", ns) if t.text]
                            if t_parts:
                                paras.append("".join(t_parts))
                        print("\n".join(paras))
                    else:
                        print(local_target.read_text(encoding="utf-8", errors="replace"))
                else:
                    print(f"{RED}Error reading file from Dream_Inbox: could not download '{target}': {e}{RESET}")

    elif action == "trash":
        target = args.target
        if not target:
            print(f"{RED}Error: please specify a document name or ID to trash.{RESET}")
            return
        print(f"{DIM}Attempting to trash '{target}' from Dream_Inbox...{RESET}")

        doc_id, doc_name, doc_fid = _resolve_dream_inbox_target(target)
        fid = doc_fid or (folder_ids[0] if folder_ids else "1ZLsrkwRBhH2Xz82eooQC3ahsePFjm28h")

        if not doc_id and len(target) > 20:
            doc_id = target

        trashed = _trash_drive_file_by_id(doc_id)
        _record_dream_inbox_processed_doc(
            doc_id=doc_id,
            doc_name=doc_name,
            method="drive_api" if trashed else "local_registry",
            extra_key=target,
        )

        if trashed:
            print(f"{GREEN}{CHECK_MARK} File '{target}' successfully moved to Google Drive Trash and recorded in processed registry.{RESET}")
        else:
            print(f"{GREEN}{CHECK_MARK} Document '{target}' (ID: {doc_id or 'n/a'}) enregistré dans le registre persistant comme traité.")
            print(f"Il est désormais masqué de 'dream-inbox list' (utiliser 'dream-inbox list --all' pour l'afficher).{RESET}")
            print(f"{YELLOW}Note : Les tokens locaux actuels sont restreints en lecture seule (scope 'drive.readonly') ou par application ('drive.file').{RESET}")
            print(f"Pour purger ce document manuellement sur Google Drive si vous le souhaitez :")
            print(f"  {CYAN}https://drive.google.com/drive/folders/{fid}{RESET}")


def cmd_backfill_main_project(args: argparse.Namespace) -> None:
    """Deterministically backfills missing main_project_file in AIVC commit JSONs."""
    from aivc.project_sync import backfill_missing_main_project_files
    print(f"{CYAN}{BOLD}Starting deterministic main_project_file backfill...{RESET}")
    res = backfill_missing_main_project_files(dry_run=getattr(args, "dry_run", False))
    print(f"{GREEN}{CHECK_MARK} Backfill complete!{RESET}")
    print(f"  Scanned commits:       {res.get('scanned')}")
    print(f"  Already had main file: {res.get('already_had_mpf')}")
    print(f"  Updated commits:       {res.get('updated')}")


def cmd_sync_status(args: argparse.Namespace) -> None:
    """Show cloud sync status and remote machines."""
    from aivc.config import get_aivc_config, get_machine_id, get_token_path

    config = get_aivc_config()
    sync_cfg = config.get("sync", {})
    m_id = get_machine_id()

    print(f"{CYAN}{BOLD}AIVC Sync Status:{RESET}")
    print(f"  {BOLD}Local Machine ID:{RESET}  {m_id}")
    print(f"  {BOLD}Sync Enabled:{RESET}      {GREEN if sync_cfg.get('enabled') else YELLOW}{sync_cfg.get('enabled', False)}{RESET}")
    print(f"  {BOLD}Auth Token:{RESET}        {f'{CHECK_MARK} present' if get_token_path().exists() else f'{CROSS_MARK} missing (run aivc sync setup)'}")

    if sync_cfg.get("enabled"):
        try:
            from aivc.sync.drive import NativeDriveSyncManager
            manager = NativeDriveSyncManager(Path.home() / ".aivc" / "storage")
            print(f"\n{DIM}Checking remote machines on Google Drive...{RESET}")
            others = manager.list_remote_machines()
            print(f"  {BOLD}Remote machines seen:{RESET} {', '.join(others) if others else 'none yet'}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")


def cmd_daemon(args: argparse.Namespace) -> None:
    """Manage the persistent background AIVC daemon."""
    port = args.port or int(os.environ.get("AIVC_DAEMON_PORT", "18765"))
    action = args.daemon_action

    from aivc.proxy import ensure_daemon_running
    import urllib.request
    import json
    import time

    def get_status() -> dict | None:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
        return None

    def post_shutdown() -> bool:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", data=b"{}", method="POST")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    if action == "status":
        st = get_status()
        if st:
            print(f"{GREEN}{CHECK_MARK} AIVC daemon is running on port {st.get('port', port)} (PID: {st.get('pid', 'unknown')}){RESET}")
        else:
            print(f"{YELLOW}AIVC daemon is not running on port {port}.{RESET}")

    elif action == "start":
        st = get_status()
        if st:
            print(f"{GREEN}{CHECK_MARK} AIVC daemon is already running on port {port} (PID: {st.get('pid')}){RESET}")
            return
        print(f"{DIM}Starting AIVC daemon on port {port}...{RESET}")
        ensure_daemon_running(port)
        st = get_status()
        if st:
            print(f"{GREEN}{CHECK_MARK} AIVC daemon started successfully on port {port} (PID: {st.get('pid')}){RESET}")
        else:
            print(f"{RED}Failed to verify AIVC daemon status.{RESET}")

    elif action == "stop":
        st = get_status()
        if not st:
            print(f"{YELLOW}AIVC daemon is not running on port {port}.{RESET}")
            return
        print(f"{DIM}Stopping AIVC daemon on port {port} (PID: {st.get('pid')})...{RESET}")
        post_shutdown()
        stopped = False
        for _ in range(30):
            time.sleep(0.1)
            if not get_status():
                stopped = True
                break
        if stopped:
            print(f"{GREEN}{CHECK_MARK} AIVC daemon stopped.{RESET}")
        else:
            print(f"{YELLOW}Daemon shutdown signal sent.{RESET}")

    elif action == "restart":
        if get_status():
            print(f"{DIM}Stopping running daemon...{RESET}")
            post_shutdown()
            for _ in range(30):
                time.sleep(0.1)
                if not get_status():
                    break
        print(f"{DIM}Starting AIVC daemon on port {port}...{RESET}")
        ensure_daemon_running(port)
        st = get_status()
        if st:
            print(f"{GREEN}{CHECK_MARK} AIVC daemon restarted on port {port} (PID: {st.get('pid')}){RESET}")
        else:
            print(f"{RED}Failed to restart AIVC daemon.{RESET}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AIVC — AI Version Control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)


    # migrate
    subparsers.add_parser(
        "migrate",
        help="Explicitly migrate JSON commits to SQLite index"
    )



    # memories (log)
    parser_log = subparsers.add_parser(
        "memories", 
        aliases=["log"],
        help="Show recent memory history"
    )
    parser_log.add_argument(
        "-n", "--limit", type=int, default=10, 
        help="Number of memories to show (default: 10)"
    )

    # recall (search)
    parser_search = subparsers.add_parser(
        "recall", 
        aliases=["search"],
        help="Semantic search over past memories"
    )
    parser_search.add_argument(
        "query", type=str, 
        help="Natural language query"
    )
    parser_search.add_argument(
        "-n", "--top-n", type=int, default=5, 
        help="Number of results to return (default: 5)"
    )
    parser_search.add_argument(
        "-g", "--glob", type=str, default="", 
        help="Optional glob pattern to restrict recall to matching files"
    )

    # web
    parser_web = subparsers.add_parser(
        "web",
        help="Launch the interactive Web Dashboard"
    )
    parser_web.add_argument(
        "-p", "--port", type=int, default=8765,
        help="Port to serve the dashboard on (default: 8765)"
    )

    # daemon
    parser_daemon = subparsers.add_parser(
        "daemon",
        help="Manage the persistent background AIVC daemon"
    )
    parser_daemon.add_argument(
        "daemon_action",
        choices=["start", "stop", "status", "restart"],
        help="Daemon action (start, stop, status, restart)"
    )
    parser_daemon.add_argument(
        "-p", "--port", type=int, default=None,
        help="Port for the daemon (default: 18765 or AIVC_DAEMON_PORT)"
    )

    # config
    parser_config = subparsers.add_parser(
        "config",
        help="Manage AIVC configuration"
    )
    parser_config.add_argument("key", type=str, nargs="?", help="Config key (path.to.key)")
    parser_config.add_argument("value", type=str, nargs="?", help="New value")

    # sync
    parser_sync = subparsers.add_parser(
        "sync",
        help="Manage cloud synchronization"
    )
    sync_sub = parser_sync.add_subparsers(dest="sync_command", required=True)
    parser_setup = sync_sub.add_parser("setup", help="Interactive Google Drive sync setup")
    parser_setup.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no browser, paste redirection URL)"
    )
    sync_sub.add_parser("status", help="Check cloud sync status")
    sync_sub.add_parser("push", help="Force push all missing local memories to Drive")
    parser_semantic = sync_sub.add_parser("semantic", help="Export semantic tasks and archives to Google Drive Docs")
    parser_semantic.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of tasks per document / archive batch (default: 25, max: 30)"
    )

    # sync-semantic alias
    parser_sync_semantic = subparsers.add_parser(
        "sync-semantic",
        help="Alias for 'aivc sync semantic'"
    )
    parser_sync_semantic.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of tasks per document / archive batch (default: 25, max: 30)"
    )

    # dream-inbox
    parser_inbox = subparsers.add_parser(
        "dream-inbox",
        help="Inspect and manage Google Drive Dream_Inbox for the /dream skill"
    )
    parser_inbox.add_argument(
        "inbox_action",
        choices=["list", "read", "trash", "watch"],
        help="Action to perform (list, read, trash, watch)"
    )
    parser_inbox.add_argument(
        "target",
        type=str,
        nargs="?",
        default="",
        help="Target document name or ID (for read or trash)"
    )
    parser_inbox.add_argument(
        "--all",
        action="store_true",
        help="List all files including already processed/trashed documents"
    )
    parser_inbox.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Polling interval in seconds for watch mode (default: 15)"
    )
    parser_inbox.add_argument(
        "--once",
        action="store_true",
        help="Run only one scan/detection cycle and exit (useful for CI/test)"
    )
    parser_inbox.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate detection and command generation without launching agy or modifying registry"
    )
    parser_inbox.add_argument(
        "--workspace",
        type=str,
        default=str(Path.home() / "Documents" / "VoiceNotes"),
        help="Target workspace directory for agy (default: ~/Documents/VoiceNotes)"
    )
    parser_inbox.add_argument(
        "--model",
        type=str,
        default="gemini-4-argon-high",
        help="Model passed to agy.exe in watch mode (default: gemini-4-argon-high)"
    )

    # backfill-main-project
    parser_backfill = subparsers.add_parser(
        "backfill-main-project",
        help="Deterministically backfill missing main_project_file in commit JSONs"
    )
    parser_backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate backfill without writing to commit JSONs"
    )

    args = parser.parse_args()

    if args.command == "migrate":
        cmd_migrate(args)

    elif args.command in ("memories", "log"):
        cmd_log(args)
    elif args.command in ("recall", "search"):
        cmd_search(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "sync-semantic":
        cmd_sync_semantic(args)
    elif args.command == "dream-inbox":
        cmd_dream_inbox(args)
    elif args.command == "backfill-main-project":
        cmd_backfill_main_project(args)
    elif args.command == "sync":
        if args.sync_command == "setup":
            cmd_sync_setup(args)
        elif args.sync_command == "status":
            cmd_sync_status(args)
        elif args.sync_command == "push":
            cmd_sync_push(args)
        elif args.sync_command == "semantic":
            cmd_sync_semantic(args)


if __name__ == "__main__":
    main()
