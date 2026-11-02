"""
Backward compatibility wrapper for aivc.project_sync.
Re-exports all symbols from project_sync to avoid breaking legacy imports.
"""

from __future__ import annotations

from aivc.project_sync import (
    set_custom_sync_handler,
    get_workspace_dir,
    get_vault_dir,
    find_project_sync_cli_path,
    find_project_memory_cli_path,
    _load_project_memory_module,
    extract_project_notes,
    extract_obsidian_project_notes,
    is_project_related_memory,
    is_obsidian_related_memory,
    normalize_project_tldr_title,
    normalize_advance_tldr_title,
    format_french_task_date,
    is_corrupt_test_artifact,
    convert_wikilinks_to_markdown_links,
    restore_artifacts_for_memory,
    restore_brain_files_for_memory,
    rewrite_artifact_paths_to_temp,
    rewrite_brain_paths_to_temp_artifacts,
    sync_project_views,
    sync_advance_tldr,
    _mirror_advance_tldr_files,
    backfill_missing_main_project_files,
)

__all__ = [
    "set_custom_sync_handler",
    "get_workspace_dir",
    "get_vault_dir",
    "find_project_sync_cli_path",
    "find_project_memory_cli_path",
    "_load_project_memory_module",
    "extract_project_notes",
    "extract_obsidian_project_notes",
    "is_project_related_memory",
    "is_obsidian_related_memory",
    "normalize_project_tldr_title",
    "normalize_advance_tldr_title",
    "format_french_task_date",
    "is_corrupt_test_artifact",
    "convert_wikilinks_to_markdown_links",
    "restore_artifacts_for_memory",
    "restore_brain_files_for_memory",
    "rewrite_artifact_paths_to_temp",
    "rewrite_brain_paths_to_temp_artifacts",
    "sync_project_views",
    "sync_advance_tldr",
    "_mirror_advance_tldr_files",
    "backfill_missing_main_project_files",
]
