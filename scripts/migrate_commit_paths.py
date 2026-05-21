#!/usr/bin/env python3
"""
One-time migration script: convert relative paths in old commit JSON files
to absolute paths.

Reads the project root from workspace.json (longest common prefix of tracked
absolute paths) and prepends it to any relative path found in FileChange
entries.  Also handles stale WSL paths (/mnt/c/...) by re-rooting them.

Usage:
    python scripts/migrate_commit_paths.py                  # run migration
    python scripts/migrate_commit_paths.py --dry-run        # preview only
    python scripts/migrate_commit_paths.py --storage-root /path/to/.aivc/storage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath


def detect_project_root(workspace_path: Path) -> Path:
    """Infer the project root from absolute paths already in workspace.json.

    Strategy: take the longest common directory prefix of every tracked path.
    Falls back to CWD if workspace.json contains no absolute paths (should
    never happen post-Phase 5).

    Raises:
        RuntimeError: if no absolute paths are found in tracked files.
    """
    data = json.loads(workspace_path.read_text(encoding="utf-8"))
    tracked: dict[str, object] = data.get("tracked_files", {})

    absolute_paths = [p for p in tracked if Path(p).is_absolute() or p.startswith('/') or p.startswith('\\')]
    if not absolute_paths:
        raise RuntimeError(
            "No absolute paths in workspace.json — cannot infer project root. "
            "Fix workspace.json manually first."
        )

    # os.path.commonpath gives the longest shared prefix directory
    root = Path(os.path.commonpath(absolute_paths))

    return root


def _reroute_wsl_path(abs_path: str, project_root: Path) -> str | None:
    """If *abs_path* is a WSL-mount path (/mnt/c/...), map it back to the
    project root using its relative suffix.

    Returns the corrected absolute path, or None if the path is not a WSL path.
    """
    WSL_PREFIXES = ("/mnt/c/", "/mnt/d/", "/mnt/e/")
    for prefix in WSL_PREFIXES:
        if abs_path.startswith(prefix):
            # Extract the relative tail: try to find where the project-specific
            # part starts (e.g. "src/aivc/..." or "README.md")
            # We strip the WSL prefix and walk until we find a segment that
            # exists relative to project_root.
            tail = abs_path[len(prefix):]
            parts = PurePosixPath(tail).parts
            for i in range(len(parts)):
                candidate = Path(project_root, *parts[i:])
                if candidate.exists():
                    return str(candidate)
            # Fallback: just use the filename
            return str(project_root / PurePosixPath(tail).name)
    return None


def migrate_commit(
    commit_path: Path,
    project_root: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Migrate a single commit JSON file.  Returns the number of paths fixed."""
    data = json.loads(commit_path.read_text(encoding="utf-8"))
    changes: list[dict] = data.get("changes", [])

    fixed = 0
    for change in changes:
        original = change["path"]
        p = Path(original)

        if not (p.is_absolute() or original.startswith('/') or original.startswith('\\')):
            # Relative path → prepend project root
            change["path"] = str(project_root / original)
            fixed += 1
        else:
            # Already absolute — check for stale WSL paths
            rerouted = _reroute_wsl_path(original, project_root)
            if rerouted is not None and rerouted != original:
                change["path"] = rerouted
                fixed += 1

    if fixed > 0 and not dry_run:
        commit_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return fixed


def run_migration(storage_root: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Run the full migration over all commits.  Returns {commit_file: n_fixed}."""
    workspace_path = storage_root / "workspace.json"
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"workspace.json not found at {workspace_path}.  "
            "Is --storage-root pointing to the right directory?"
        )

    project_root = detect_project_root(workspace_path)
    commits_dir = storage_root / "commits"

    if not commits_dir.exists():
        raise FileNotFoundError(f"commits/ directory not found at {commits_dir}.")

    results: dict[str, int] = {}
    for commit_file in sorted(commits_dir.glob("*.json")):
        n = migrate_commit(commit_file, project_root, dry_run=dry_run)
        results[commit_file.name] = n

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate relative paths in AIVC commit files to absolute paths."
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Path to .aivc/storage/ (defaults to $AIVC_STORAGE_ROOT)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk.",
    )
    args = parser.parse_args()

    storage_root = args.storage_root
    if storage_root is None:
        env = os.environ.get("AIVC_STORAGE_ROOT")
        if not env:
            sys.exit(
                "ERROR: --storage-root not provided and AIVC_STORAGE_ROOT not set."
            )
        storage_root = Path(env)

    label = "[DRY RUN] " if args.dry_run else ""
    print(f"{label}Migrating commit paths in {storage_root}/commits/ ...")

    results = run_migration(storage_root, dry_run=args.dry_run)

    total = sum(results.values())
    for name, n in results.items():
        status = f"{n} path(s) fixed" if n > 0 else "already clean"
        print(f"  {name}: {status}")

    print(f"\n{label}Total: {total} path(s) migrated across {len(results)} commit(s).")


if __name__ == "__main__":
    main()
