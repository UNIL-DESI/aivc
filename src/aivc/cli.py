"""
AIVC Command Line Interface.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

def cmd_status(args: argparse.Namespace) -> None:
    engine = _get_engine()
    statuses = engine.get_status()

    if not statuses:
        print("No files are currently tracked by AIVC.")
        return

    header = f"{CYAN}{'File Path':<60} {'Current':>10} {'History':>10}{RESET}"
    print(header)
    print("-" * 82)
    
    for s in statuses:
        curr = _format_bytes(s.current_size)
        hist = _format_bytes(s.history_size)
        print(f"{s.path:<60} {curr:>10} {hist:>10}")
        
    print("-" * 82)
    print(f"Total tracked: {len(statuses)} file(s)")

    # Watched directories (Phase 17)
    watched = engine.get_watched_dirs()
    if watched:
        print(f"\n{YELLOW}{BOLD}Watched Directories:{RESET}")
        for path, cfg in watched.items():
            ignores_str = f" (ignoring: {', '.join(cfg['ignores'])})" if cfg['ignores'] else ""
            print(f"  {CYAN}* {path}{RESET}{ignores_str}")


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


def cmd_search_files(args: argparse.Namespace) -> None:
    """Lexical search (BM25) over current tracked file contents."""
    engine = _get_engine()
    print(f"{DIM}Searching file contents for: '{args.query}' (BM25)...{RESET}\n")
    results = engine.search_files_bm25(args.query, top_n=args.top_n)
    
    if not results:
        print("No matching files found.")
        return

    for i, r in enumerate(results, 1):
        print(f"{CYAN}{BOLD}{i}. {r['path']}{RESET} {DIM}(score: {r['score']:.3f}){RESET}")
        print(f"   {r['snippet']}\n")

def cmd_track(args: argparse.Namespace) -> None:
    """Track a file, directory, or glob pattern."""
    engine = _get_engine()
    result = engine.track(args.path, ignores=args.ignore)
    newly_tracked = result["newly_tracked"]
    hidden_skipped = result["hidden_skipped"]

    if not newly_tracked:
        msg = f"{YELLOW}No new files to track{RESET} (already tracked or no match)."
        if hidden_skipped > 0:
            msg += f" {DIM}({hidden_skipped} hidden files ignored){RESET}"
        print(msg)
        return

    print(f"{GREEN}{BOLD}Tracked {len(newly_tracked)} new file(s):{RESET}")
    for f in newly_tracked:
        print(f"  {CYAN}+{RESET} {f}")
    
    if hidden_skipped > 0:
        print(f"{DIM}Note: {hidden_skipped} hidden files/folders were ignored.{RESET}")


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
    print(f"\n{DIM}Opening your browser for Google authorization...{RESET}")

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(f"{RED}Error: Google auth libraries not installed.{RESET}")
        print("Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return

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

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AIVC — AI Version Control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser(
        "status", 
        help="List tracked files with a navigable folder tree"
    )

    # migrate
    subparsers.add_parser(
        "migrate",
        help="Explicitly migrate JSON commits to SQLite index"
    )

    # track
    parser_track = subparsers.add_parser(
        "track",
        help="Track a file, directory, or glob pattern"
    )
    parser_track.add_argument(
        "path", type=str,
        help="File path, directory, or glob pattern to track"
    )
    parser_track.add_argument(
        "--ignore", type=str, nargs="+", default=[],
        help="Optional glob patterns to ignore (only when tracking a directory)"
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

    # search-files (BM25)
    parser_search_files = subparsers.add_parser(
        "search-files", 
        help="Lexical search (BM25) in current tracked files"
    )
    parser_search_files.add_argument(
        "query", type=str, 
        help="Keywords or exact terms to find"
    )
    parser_search_files.add_argument(
        "-n", "--top-n", type=int, default=5, 
        help="Number of results to return (default: 5)"
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
    sync_sub.add_parser("setup", help="Interactive Google Drive sync setup")
    sync_sub.add_parser("status", help="Check cloud sync status")
    sync_sub.add_parser("push", help="Force push all missing local memories to Drive")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "track":
        cmd_track(args)
    elif args.command in ("memories", "log"):
        cmd_log(args)
    elif args.command in ("recall", "search"):
        cmd_search(args)
    elif args.command == "search-files":
        cmd_search_files(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "sync":
        if args.sync_command == "setup":
            cmd_sync_setup(args)
        elif args.sync_command == "status":
            cmd_sync_status(args)
        elif args.sync_command == "push":
            cmd_sync_push(args)


if __name__ == "__main__":
    main()
