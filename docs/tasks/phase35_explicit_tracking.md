# Phase 35: Explicit File Tracking & Strict Path Validation

## Context & Objectives
To streamline file tracking within AIVC and eliminate background resource overhead:
- Automatic directory monitoring via watchdog observers has been completely removed.
- Bipartite memory-to-file links are now established explicitly using `read_files` and `edited_files` lists.
- Path inputs are strictly validated to prevent folders from being tracked, raising errors immediately.

## Detailed Description
1. **Tool and CLI Cleanup**:
   - The MCP `track` tool and CLI subcommand `aivc track` have been removed.
   - All continuous watchdog handlers and setup parameters (`watched_dirs` in `workspace.json`) have been cleaned up and deprecated.
2. **Signature Refactoring**:
   - `Workspace.create_memory` and `SemanticEngine.create_memory` accept `read_files: list[str]` and `edited_files: list[str]`.
   - The legacy `consulted_files` parameter is mapped to `read_files` for backwards compatibility.
3. **Strict Validation**:
   - Paths must be resolved to absolute paths.
   - Folders are rejected with a `ValueError`.
   - Untracked non-existent files are rejected with a `ValueError` (unless they existed in previous commits, permitting deletions).
4. **Auto-Tracking & Target Diffs**:
   - Discovered files that exist are automatically tracked on-the-fly.
   - Differences are computed strictly against the subset of files defined in `edited_files` rather than the entire workspace state.

## Verification
- Unit and integration tests modified to replace `.track()` and mock out dependencies correctly.
- 173/173 tests passing successfully via pytest.
