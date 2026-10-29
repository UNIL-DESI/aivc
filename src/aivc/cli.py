"""
AIVC Command Line Interface.
"""

from __future__ import annotations

import argparse
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
        if "code=" in user_input or user_input.startswith("http://") or user_input.startswith("https://"):
            try:
                parsed = urlparse(user_input)
                query = parse_qs(parsed.query)
                if "code" in query:
                    code = query["code"][0]
            except Exception:
                pass

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
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass


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
        except Exception:
            pass

    action = args.inbox_action

    if action == "list":
        print(f"{CYAN}{BOLD}Google Drive Dream_Inbox Listing:{RESET}")
        try:
            out = subprocess.check_output(
                ["rclone", "lsjson", "gdrive:Dream_Inbox"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8"
            )
            files = json.loads(out)
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
                f_mod = f.get("ModTime", "")[:19]
                status_tag = ""
                if f_id in processed_map or f_name in processed_map:
                    status_tag = f" {YELLOW}[déjà traité]{RESET}"
                print(f"  {i}. {BOLD}{f_name}{RESET} (ID: {f_id}, Date: {f_mod}){status_tag}")
        except Exception as e:
            print(f"{RED}Error listing Dream_Inbox: {e}{RESET}")

    elif action == "read":
        target = args.target
        if not target:
            print(f"{RED}Error: please specify a document name or ID to read.{RESET}")
            return
        print(f"{DIM}Reading '{target}' from Dream_Inbox...{RESET}\n")
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_target = Path(tmp_dir) / "doc.docx"
            try:
                remote_path = f"Dream_Inbox/{target}"
                subprocess.check_call(
                    ["rclone", "copyto", f"gdrive:{remote_path}", str(local_target)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
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
            except Exception as e:
                print(f"{RED}Error reading file from Dream_Inbox: {e}{RESET}")

    elif action == "trash":
        target = args.target
        if not target:
            print(f"{RED}Error: please specify a document name or ID to trash.{RESET}")
            return
        print(f"{DIM}Attempting to trash '{target}' from Dream_Inbox...{RESET}")

        doc_id = None
        doc_name = None
        try:
            out = subprocess.check_output(
                ["rclone", "lsjson", "gdrive:Dream_Inbox"],
                stderr=subprocess.DEVNULL,
                encoding="utf-8"
            )
            files = json.loads(out)
            for f in files:
                if f.get("Name") == target or f.get("ID") == target:
                    doc_id = f.get("ID")
                    doc_name = f.get("Name")
                    break
        except Exception:
            pass

        if not doc_id and len(target) > 20:
            doc_id = target

        trashed = False
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from aivc.config import get_token_path
            t_path = get_token_path()
            if t_path.exists():
                creds = Credentials.from_authorized_user_file(str(t_path))
                svc = build("drive", "v3", credentials=creds)
                svc.files().update(fileId=doc_id, body={"trashed": True}).execute()
                trashed = True
        except Exception:
            pass

        if not trashed:
            try:
                from aivc.config import get_home_dir
                import urllib.request
                import urllib.parse
                data_path = get_home_dir() / "Documents" / "VoiceNotes" / ".obsidian" / "plugins" / "gemini-sync" / "data.json"
                if data_path.exists():
                    data = json.loads(data_path.read_text(encoding="utf-8"))
                    params = {
                        "client_id": data["clientId"],
                        "client_secret": data["clientSecret"],
                        "refresh_token": data["refreshToken"],
                        "grant_type": "refresh_token"
                    }
                    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=urllib.parse.urlencode(params).encode("utf-8"), method="POST")
                    with urllib.request.urlopen(req) as resp:
                        tok_resp = json.loads(resp.read().decode("utf-8"))
                    svc = build("drive", "v3", credentials=Credentials(token=tok_resp["access_token"]))
                    svc.files().update(fileId=doc_id, body={"trashed": True}).execute()
                    trashed = True
            except Exception:
                pass

        if not trashed and doc_id:
            try:
                gdrive_script = Path.home() / ".gemini" / "config" / "plugins" / "gdrive" / "skills" / "gdrive" / "scripts" / "gdrive.cmd"
                if gdrive_script.exists():
                    res = subprocess.run([str(gdrive_script), "mutate", "trash", doc_id], capture_output=True, text=True)
                    if res.returncode == 0:
                        trashed = True
            except Exception:
                pass

        if not trashed:
            try:
                fname = doc_name or (target if "." in target else None)
                if fname:
                    res_rclone = subprocess.run(
                        ["rclone", "deletefile", f"gdrive:Dream_Inbox/{fname}"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if res_rclone.returncode == 0:
                        trashed = True
            except Exception:
                pass

        # Persist to registry (local + Google Drive AIVC_Sync)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        reg = _load_dream_inbox_processed()
        primary_key = doc_id or target
        entry = {
            "trashed_at": now_iso,
            "method": "drive_api" if trashed else "local_registry",
            "name": doc_name or target,
        }
        if doc_id:
            entry["doc_id"] = doc_id
        reg.setdefault("processed", {})[primary_key] = entry
        if doc_name and doc_name != primary_key:
            reg["processed"][doc_name] = entry
        if target and target != primary_key and target != doc_name:
            reg["processed"][target] = entry
        _save_dream_inbox_processed(reg)

        if trashed:
            print(f"{GREEN}{CHECK_MARK} File '{target}' successfully moved to Google Drive Trash and recorded in processed registry.{RESET}")
        else:
            print(f"{GREEN}{CHECK_MARK} Document '{target}' (ID: {doc_id or 'n/a'}) enregistré dans le registre persistant comme traité.")
            print(f"Il est désormais masqué de 'dream-inbox list' (utiliser 'dream-inbox list --all' pour l'afficher).{RESET}")
            print(f"{YELLOW}Note : Les tokens locaux actuels sont restreints en écriture (scope 'drive.readonly' ou 'drive.file').{RESET}")
            print(f"Pour purger ce document manuellement sur Google Drive si vous le souhaitez :")
            print(f"  {CYAN}https://drive.google.com/drive/folders/1JsNgV7uD1c981dNQpYEprKdjVgitLDo8{RESET}")


def cmd_backfill_main_project(args: argparse.Namespace) -> None:
    """Deterministically backfills missing main_project_file in AIVC commit JSONs."""
    from aivc.advance_sync import backfill_missing_main_project_files
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
        choices=["list", "read", "trash"],
        help="Action to perform (list, read, trash)"
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
