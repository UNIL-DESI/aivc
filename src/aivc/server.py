"""
AIVC MCP Server — Long-term memory for LLM agents.

Exposes the SemanticEngine as a set of MCP tools via FastMCP (stdio transport).
The server is started by: ``python -m aivc.server``

Environment variables:
    AIVC_STORAGE_ROOT  (required) — absolute path to the AIVC data directory.
"""

from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Set mission-critical environment variables before heavy ML imports
# This completely bypasses the 5-minute atexit/thread deadlock on Windows
# caused by ChromaDB PostHog telemetry failing on corporate firewalls.
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import Counter
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# System prompt — injected into every LLM context using this server
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
# AIVC — AI Version Control (Long-Term Memory)

You have access to a persistent, versioned memory system called AIVC.
AIVC is your long-term memory. Use it actively — it is the only way to preserve
context beyond a single conversation.

## Core Concept

AIVC stores **memories**: a short title + a detailed Markdown note you write yourself.
Every memory also automatically snapshots any tracked files that were modified.
Memories are indexed semantically, so you can retrieve them by meaning later.

## CRITICAL RULE — REMEMBER OFTEN

**You MUST create a memory (call `remember`) after EVERY significant step.**

A memory is required after:
- Completing a sub-task or an entire task.
- Creating or modifying any artifact (file, script, document, test, config…).
- Discovering a key finding or making an architectural decision.
- Finishing any phase of a plan, even if work is still ongoing.
- Any identifiable "checkpoint" in your reasoning.

The memory note must be **detailed**. Do not write one-liners.
Document your reasoning, the decisions made, the problems encountered,
and the solutions found. Think of it as a handover memo to your future self.

### Read and Edited Files

When you create a memory, you can specify `read_files` and `edited_files`.
`read_files` are files you have read and that were **truly useful** to you to
accomplish your task, but that you did not modify. `edited_files` are files
that were modified or created for the task.

## Recall Funnel

To retrieve memory, follow this two-step funnel:

1. **`recall`** — for semantic search by meaning (idea, topic, solution).
   → Returns memory titles/dates/IDs + snippets. NEVER the full note.
2. **`get_recent_memories`** — for recalling recent history chronologically.
3. **`consult_memory`** — to read the full note of a specific memory.
   → Call this AFTER identifying a relevant memory.

4. **`search_files`** — for keyword or regex search in the CURRENT state of files.

## Remote Memories & Sync Policy

AIVC synchronizes ONLY memory metadata (titles, notes) between machines. 
**File contents (blobs) are NOT synchronized.** 
If you see a memory marked as `[Remote: machine-id]`, the historical version 
of files associated with it might not be available for `read_past_file_content`.

## Tool Reference

| Tool | Purpose |
|------|---------|
| `remember` | Save a memory checkpoint. Call this VERY often. |
| `recall` | Semantic search over all past memory notes. |
| `get_recent_memories` | Recent memory log (paginable). |
| `consult_memory` | Read a specific memory note in full. |
| `get_file_history_metadata` | Get the AIVC history of a specific file. |
| `read_past_file_content` | Read the content of a file as it was at a specific past memory. |
| `get_status` | List tracked files with a navigable folder tree. |
| `search_files` | Lexical search (Keywords or Regex) over current tracked file contents. |

"""

# ---------------------------------------------------------------------------
# Bootstrap — engine initialisation
# ---------------------------------------------------------------------------

from aivc.config import get_storage_root

_storage_root = get_storage_root()

# SemanticEngine is imported here (triggering a fast eager init of Workspace +
# SQLite graph; the heavy ML components remain lazy until first use).
from aivc.semantic.engine import SemanticEngine  # noqa: E402
from aivc.sync.background import BackgroundSyncer
from aivc.config import get_machine_id
import threading

_engine: SemanticEngine | None = None
_local_machine_id: str | None = None
_lock = threading.Lock()

def _get_engine() -> SemanticEngine:
    """Lazy-load the SemanticEngine on the first tool call.
    This prevents heavy ML dependencies from being loaded at import time,
    which is crucial for fast CLI feedback and test suite stability.
    """
    global _engine, _local_machine_id
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = SemanticEngine(_storage_root)
                _local_machine_id = get_machine_id()
    return _engine

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(name="aivc", instructions=_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Helper formatting functions
# ---------------------------------------------------------------------------


def _render_file_tree(paths: list[str], path_extras: dict[str, str] = None, indent_prefix: str = "  ") -> str:
    """Render a list of absolute paths as a hierarchical tree."""
    if not paths:
        return "—"

    if len(paths) == 1:
        common_root = os.path.dirname(paths[0])
    else:
        try:
            # Safely find a common directory prefix
            abs_paths = [os.path.abspath(p) for p in paths]
            common_root = os.path.commonpath([os.path.dirname(p) for p in abs_paths])
        except ValueError:
            common_root = ""

    tree: dict = {}

    for path in paths:
        if common_root:
            try:
                rel_path = os.path.relpath(path, common_root)
            except ValueError:
                rel_path = path
        else:
            rel_path = path

        if rel_path == ".":
            continue

        parts = rel_path.split(os.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = path

    lines = []
    if common_root:
        # Avoid double slash if common_root is already root (e.g. "/")
        root_disp = common_root if common_root.endswith(os.sep) else common_root + os.sep
        lines.append(f"{indent_prefix}{root_disp}")

    def _traverse(node, prefix=""):
        items = sorted(node.items(), key=lambda x: (not isinstance(x[1], dict), x[0].lower()))

        for i, (name, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "

            if isinstance(value, dict):
                lines.append(f"{indent_prefix}{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                _traverse(value, prefix + extension)
            else:
                extra = (path_extras or {}).get(value, "")
                lines.append(f"{indent_prefix}{prefix}{connector}{name}{extra}")

    _traverse(tree)

    return "\n".join(lines) if len(lines) > 0 else f"{indent_prefix}—"


def _format_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _format_changes_compressed(changes, machine_id=None) -> str:
    """Render tracked file changes as a clear hierarchical tree."""
    if not changes:
        return "  (no tracked files changed)"
    
    paths = []
    extras = {}
    
    for c in changes:
        paths.append(c.path)
        extra_parts = [f"[{c.action}]"]
        if c.action != "consulted":
            extra_parts.append(f"({c.format_impact()})")
        
        if machine_id and machine_id != _local_machine_id:
            local_match = _get_engine().find_local_equivalent(c.path, c.blob_hash)
            if local_match:
                extra_parts.append(f"(probablement `{local_match}` localement)")
                
        extras[c.path] = " " + " ".join(extra_parts)

    return _render_file_tree(paths, extras, indent_prefix="  ")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def remember(
    title: str,
    note: str,
    read_files: list[str] = [],
    edited_files: list[str] = []
) -> str:
    """Persist a memory checkpoint in AIVC.

    Call this tool after EVERY meaningful step: task completion, artefact creation,
    architectural decision, key discovery, or any checkpoint in your work.
    The note should be a rich, detailed Markdown document — your future self will
    read it to recall this moment. All tracked files that have changed since the last
    memory are automatically associated with this memory.

    Args:
        title: Short, descriptive title (e.g. "Implemented user auth module").
        note: Detailed Markdown note documenting what was done, why, how, and any
              important context. The more detail, the better the future recall.
        read_files: Optional list of files that were consulted and
                    provided CRUCIAL context for this task, but not modified.
                    Files not yet tracked will be auto-tracked if they exist on disk.
                    Directories or untracked non-existent files will raise strict validation errors.
        edited_files: Optional list of file paths that were modified/created for this task.
                      Files not yet tracked will be auto-tracked if they exist on disk.
                      Directories or untracked non-existent files will raise strict validation errors.

    Returns:
        Confirmation with the memory ID and the list of files that were snapshotted.

    Raises:
        ValueError: If any paths in read_files or edited_files are directories or
                    untracked non-existent files.
        RuntimeError: If no tracked file has changed and no files were read/edited.
    """
    import asyncio
    from pathlib import Path

    engine = _get_engine()

    # Run the heavy vector encoding and DB insertion in a background thread
    task = asyncio.create_task(
        asyncio.to_thread(
            engine.create_memory,
            title,
            note,
            read_files=read_files,
            edited_files=edited_files
        )
    )

    def _done_callback(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception as exc:
            logger.error("Error in background memory creation: %s", exc)

    task.add_done_callback(_done_callback)

    return (
        f"✅ Memory creation scheduled in background.\n"
        f"Title     : {title}\n"
        f"Processing: vector encoding and database updates are running in the background."
    )


@mcp.tool()
async def recall(query: str, top_n: int = 5, filter_glob: str = "", only_local: bool = False) -> str:
    """Recall past memories by semantic meaning.

    Uses a Bi-Encoder + Cross-Encoder pipeline to retrieve the most relevant
    memories for a natural-language query. Returns only memory metadata (ID,
    title, date, score) — never the full note content — to avoid context bloat.
    Also surfaces the files most frequently associated with the top results.

    Call `consult_memory(memory_id)` on a specific result to read its full note.

    Args:
        query: Free-text search query. Write it as a question or a short description.
        top_n: Number of results to return (default 5, max 20).
        filter_glob: Optional glob pattern (e.g. "src/*.py") to restrict search to memories
                     that touched matching files.
        only_local: If True, only search memories created on this machine.
    """
    import asyncio
    top_n = min(top_n, 20)
    
    # Force import/load of the heavy CrossEncoder on the main thread to prevent Windows DLL deadlocks.
    # This runs only on the very first recall call and takes a few seconds, but is completely safe.
    import os
    disable_cross = os.environ.get("AIVC_DISABLE_CROSS_ENCODER", "False").lower() == "true"
    engine = _get_engine()
    if not disable_cross:
        try:
            searcher = engine._searcher
            if searcher is not None:
                _ = searcher._cross_encoder
        except Exception as e:
            import sys
            print(f"[aivc] Failed to eagerly load CrossEncoder on main thread: {e}", file=sys.stderr)

    # Check if indexing is in progress
    indexing_queue_size = engine.get_index_queue_size()
    warning_header = ""
    if indexing_queue_size > 0:
        warning_header = f"⚠️  Note: {indexing_queue_size} recent memory(ies) are still being indexed and may be missing from search results.\n\n"

    # Run the heavy semantic search query in a background thread to keep the event loop responsive
    results = await asyncio.to_thread(engine.search, query, top_n=top_n, filter_glob=filter_glob)

    if only_local:
        results = [r for r in results if getattr(r, 'machine_id', _local_machine_id) == _local_machine_id]

    if not results:
        return warning_header + "No matching memories found."

    # Build memory list
    memory_lines = []
    for i, r in enumerate(results, 1):
        m_id = getattr(r, 'machine_id', "")
        remote_tag = f" [Remote: {m_id}]" if m_id and m_id != _local_machine_id else ""
        
        memory_lines.append(
            f"{i}. [{r.timestamp[:10]}] {r.title}{remote_tag}\n"
            f"   ID    : {r.memory_id}\n"
            f"   Score : {r.score:.3f}\n"
            f"   > {r.snippet}"
        )

    # Aggregate file paths across top results (most frequently mentioned)
    file_counter: Counter[str] = Counter()
    for r in results:
        file_counter.update(r.file_paths)

    paths = []
    extras = {}
    for fp, count in file_counter.most_common(10):
        paths.append(fp)
        extra_parts = [f"(in {count}/{len(results)} results)"]

        # If results are remote, try to find local hints
        is_remote = any(getattr(r, 'machine_id', "") != _local_machine_id for r in results)
        if is_remote:
            local_match = _get_engine().find_local_equivalent(fp)
            if local_match:
                extra_parts.append(f"(probablement `{local_match}` localement)")

        extras[fp] = " " + " ".join(extra_parts)

    output = warning_header + "## Matching Memories\n\n"
    output += "\n".join(memory_lines)

    if paths:
        output += "\n\n## Most Relevant Files\n"
        output += _render_file_tree(paths, extras, indent_prefix="  ")
    else:
        output += "\n\n(No file associations found for these memories.)"

    output += "\n\n💡 Use `consult_memory(memory_id)` to read a full note."
    return output


@mcp.tool()
def search_files(
    query: str, 
    top_n: int = 5, 
    is_regex: bool = False,
    case_sensitive: bool = False
) -> str:
    """Search for keywords or regex patterns inside the content of tracked files.

    This tool performs a fast, parallel scan of all currently tracked files on disk.
    For keyword searches (default), it uses an 'AND' logic: it finds files where ALL
    provided words are present, regardless of their order or location.

    Args:
        query: Search terms (e.g. "auth error") or a regex pattern.
        top_n: Number of results to return (default 5).
        is_regex: If True, treats query as a regular expression.
        case_sensitive: If True, search is case sensitive (default False).
    """
    results = _get_engine().search_files(
        query, 
        top_n=top_n, 
        is_regex=is_regex, 
        case_sensitive=case_sensitive
    )

    if not results:
        type_str = "regex" if is_regex else "keyword"
        return f"No matches found for {type_str} query: '{query}'"

    lines = [f"## Search results for: `{query}`\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. `{r['path']}` (score: {r['score']:.1f})\n"
            f"   > {r['snippet']}"
        )

    return "\n".join(lines)


@mcp.tool()
def consult_memory(memory_id: str) -> str:
    """Read the full content of a specific memory.

    Returns the complete Markdown note written when the memory was created,
    along with a summary of the files that were changed (path, action, size impact).

    Args:
        memory_id: The UUID of the memory to read (obtained from `recall`
                   or `get_recent_memories`).

    Returns:
        The full Markdown note and the list of file changes.

    Raises:
        KeyError: If the memory_id does not exist.
    """
    memory = _get_engine().get_memory(memory_id)

    # Context (Prev/Next)
    prev_str = ""
    if memory.parent_id:
        try:
            parent = _get_engine().get_memory(memory.parent_id)
            prev_str = f"⬆️ **Prev** : {parent.title} (ID: {parent.id})\n"
        except KeyError:
            prev_str = f"⬆️ **Prev** : (metadata not found) (ID: {memory.parent_id})\n"

    next_str = ""
    try:
        child = _get_engine().find_child_memory(memory_id)
        if child:
            next_str = f"⬇️ **Next** : {child.title} (ID: {child.id})\n"
    except Exception:
        pass

    context_block = ""
    if prev_str or next_str:
        context_block = f"{prev_str}{next_str}\n"

    changes_summary_str = _format_changes_compressed(memory.changes, memory.machine_id)

    machine_line = ""
    remote_warning = ""
    if memory.machine_id and memory.machine_id != _local_machine_id:
        machine_line = f"**Machine**   : {memory.machine_id} (Distant)\n"
        remote_warning = "> [!WARNING]\n> This memory was created on a remote machine. Historical file contents may not be available.\n\n"

    return (
        f"# Memory: {memory.title}\n\n"
        f"{remote_warning}"
        f"**ID**        : {memory.id}\n"
        f"**Timestamp** : {memory.timestamp}\n"
        f"**Parent**    : {memory.parent_id or 'none (initial memory)'}\n"
        f"{machine_line}\n"
        f"{context_block}"
        f"## Files Recorded\n{changes_summary_str}\n\n"
        f"## Note\n\n{memory.note}"
    )


@mcp.tool()
def get_recent_memories(limit: int = 10, offset: int = 0, only_local: bool = False) -> str:
    """Display the recent memory history.

    Use this tool at the start of a session or when you need to recall what
    was done recently without having a specific search query.
    Results are in reverse chronological order (newest first).
    Use `offset` and `limit` to paginate (e.g. offset=10 to see memories 11-20).

    Args:
        limit:  Number of memories to show (default 10, max 50).
        offset: Number of memories to skip from the most recent (default 0).
        only_local: If True, only show memories created on this machine.
    """
    limit = min(limit, 50)

    # get_log fetches `offset + limit` memories and then slices.
    all_recent = _get_engine().get_log(limit=offset + limit)
    
    if only_local:
        all_recent = [m for m in all_recent if m.machine_id == _local_machine_id]

    page = all_recent[offset : offset + limit]

    if not page:
        return "No memories found in this range."

    lines = [f"Showing memories {offset + 1}–{offset + len(page)} (newest first)\n"]

    file_counter: Counter[str] = Counter()
    all_remote_paths = set()

    for i, memory in enumerate(page, offset + 1):
        m_tag = f" [Remote: {memory.machine_id}]" if memory.machine_id and memory.machine_id != _local_machine_id else ""
        lines.append(
            f"{i:>3}. [{memory.timestamp[:10]}] {memory.title}{m_tag}\n"
            f"      ID    : {memory.id}"
        )

        # Collect files for aggregation
        try:
            m_files = _get_engine().get_memory_files(memory.id)
            file_counter.update(m_files)
            if memory.machine_id and memory.machine_id != _local_machine_id:
                all_remote_paths.update(m_files)
        except KeyError:
            pass

    # Heatmap of modified files
    if file_counter:
        # Show top 10 or proportional to limit (but at least 10)
        num_files = max(10, limit // 2) if limit > 20 else 10
        top_files = file_counter.most_common(num_files)

        paths = []
        extras = {}
        for fp, count in top_files:
            paths.append(fp)
            extra_parts = [f"({count}x)"]

            # Maintenance of hints (probablement ...)
            if fp in all_remote_paths:
                local_match = _get_engine().find_local_equivalent(fp)
                if local_match:
                    extra_parts.append(f"(probablement `{local_match}` localement)")

            extras[fp] = " " + " ".join(extra_parts)

        lines.append("\n## Recent Activity Heatmap")
        lines.append(_render_file_tree(paths, extras, indent_prefix="  "))

    lines.append("\n💡 Use `consult_memory(memory_id)` to read a full memory note.")
    return "\n".join(lines)


@mcp.tool()
def get_file_history_metadata(file_path: str) -> str:
    """Retrieve the chronological list of all memories (commits) that modified or consulted a specific file. Useful to understand WHEN a file was changed and WHY, but DOES NOT return the actual file content.

    Args:
        file_path: The path of the file to look up (as tracked by AIVC).

    Returns:
        A list of commits that touched this file (ID, Date, Title).

    Raises:
        KeyError: If the file is not in the AIVC co-occurrence graph.
    """
    memory_ids = _get_engine().get_file_memories(file_path)

    if not memory_ids:
        return f"No memories found for file: {file_path}"

    lines = [f"## AIVC History for: `{file_path}`\n"]
    lines.append(f"{len(memory_ids)} memory(ies) have touched this file:\n")

    for mid in memory_ids:
        try:
            memory = _get_engine().get_memory(mid)
            lines.append(
                f"  - [{memory.timestamp[:10]}] {memory.title}\n"
                f"    ID: {memory.id}"
            )
        except KeyError:
            lines.append(f"  - [unknown date] Memory {mid} (metadata not found)")

    lines.append(
        "\n💡 Use `consult_memory(memory_id)` to read the full note of a specific memory."
        "\n💡 Use `read_past_file_content(file_path, memory_id)` to read the file content at that memory."
    )
    return "\n".join(lines)


@mcp.tool()
def read_past_file_content(file_path: str, memory_id: str) -> str:
    """Retrieve the actual text content of a file exactly as it was at the time of a specific past memory. Use this to restore old code or compare previous implementations. Note: requires both the file path and the memory_id obtained from get_file_history_metadata.

    Args:
        file_path: The path of the file to read.
        memory_id: The UUID of the memory at which to read the file.
    """
    try:
        raw: bytes = _get_engine().read_file_at_memory(file_path, memory_id)
        return raw.decode("utf-8")
    except (KeyError, FileNotFoundError):
        # Find which memory exactly has this blob to provide context
        target_memory = None
        mid = memory_id
        while mid:
            try:
                m = _get_engine().get_memory(mid)
                for change in m.changes:
                    if change.path == file_path and change.blob_hash:
                        target_memory = m
                        break
                if target_memory: break
                mid = m.parent_id
            except KeyError:
                break
            
        if target_memory and target_memory.machine_id and target_memory.machine_id != _local_machine_id:
            return (
                f"⚠️ ERROR: Content of `{file_path}` is NOT available locally.\n\n"
                f"This file version was recorded on a remote machine: `{target_memory.machine_id}`.\n"
                "AIVC Phase 29+ does not synchronize file contents (blobs) across machines for security and performance.\n"
                "Please synchronize your files manually (e.g., via `git pull`) to access this content."
            )
        
        return f"⚠️ ERROR: File `{file_path}` or its content at memory `{memory_id}` could not be found locally."


@mcp.tool()
def get_status(path: str = "") -> str:
    """List tracked files with storage usage in a navigable folder tree.

    Displays a tree of depth 1 starting from the given path (or root if empty).
    Shows the number of files and total size for each subfolder.

    Args:
        path: Optional subdirectory path to explore (e.g. "src/").
    """
    import os
    import getpass
    from aivc.config import get_storage_root
    storage_root = get_storage_root(allow_fallback=True)
    diag_info = f"[AIVC Diag] Storage Root: {storage_root} | User: {getpass.getuser()} | Env Root: {os.environ.get('AIVC_STORAGE_ROOT')}\n"

    # Use get_tracked_paths (fast) + metadata (fast, from memory)
    tracked_paths = _get_engine().get_tracked_paths()
    metadata = _get_engine().get_tracked_files_metadata()
    
    if not tracked_paths:
        return diag_info + "No files are currently tracked by AIVC."

    # Determine virtual root for display
    if path:
        root_path = str(Path(path).resolve())
    else:
        # Find common root to avoid showing /home/lopilo/... hierarchy
        try:
            root_path = os.path.commonpath(tracked_paths)
        except ValueError:
            root_path = ""

    # {name: {"files": int, "size": int, "is_dir": bool}}
    tree: dict[str, dict] = {}
    total_files = 0
    total_size = 0

    for abs_path in tracked_paths:
        if root_path and not abs_path.startswith(root_path):
            continue

        total_files += 1
        # Retrieve size from in-memory metadata (zero O/S overhead)
        file_meta = metadata.get(abs_path, {})
        raw_size = file_meta.get("size", 0) if isinstance(file_meta, dict) else 0
        size = int(raw_size) if raw_size is not None else 0
        total_size += size

        # Relative path from our virtual root
        try:
            rel_to_root = os.path.relpath(abs_path, root_path) if root_path else abs_path
        except ValueError:
            rel_to_root = abs_path
            
        if rel_to_root == ".":
            continue

        # Determine the first component
        parts = rel_to_root.split(os.sep)
        if not parts or not parts[0]:
            continue
        
        name = parts[0]
        # It's a directory if it has more components
        is_dir = len(parts) > 1
        
        if name not in tree:
            tree[name] = {"files": 0, "size": 0, "is_dir": is_dir}
        
        tree[name]["files"] += 1
        tree[name]["size"] += size

    if not tree and path:
        return f"No tracked files found under path: `{path}`"

    # Sort: directories first, then files
    sorted_items = sorted(tree.items(), key=lambda x: (not x[1]["is_dir"], x[0].lower()))

    lines = []
    header_path = path if path else "Root"
    lines.append(f"📁 {header_path} ({total_files} tracked files, {_format_bytes(total_size)})")
    lines.append("-" * 60)

    for name, info in sorted_items:
        prefix = "├── " if name != sorted_items[-1][0] else "└── "
        if info["is_dir"]:
            lines.append(f"{prefix}{name}/ ({info['files']} files, {_format_bytes(info['size'])})")
        else:
            lines.append(f"{prefix}{name} ({_format_bytes(info['size'])})")

    lines.append("-" * 60)
    lines.append("\n💡 TIP: Use `get_status(path='dir/name')` to explore subdirectories.")
    lines.append("💡 NOTE: Hidden files/folders (starting with '.') are NEVER tracked automatically.")
    
    return "\n".join(lines)




# No background watchers active

# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading
    import sys
    import os
    from aivc.sync.background import BackgroundSyncer
    
    # Under Windows, completely disable the heavy CrossEncoder by default.
    # This prevents PyTorch thread collisions and DLL Loader Lock deadlocks,
    # reduces RAM usage by 1.5GB, and drops first query latency from 10s to 0.1s.
    if sys.platform == "win32":
        os.environ["AIVC_DISABLE_CROSS_ENCODER"] = "True"
    
    # Eagerly load the lightweight Indexer (ChromaDB + FastEmbed) on the main thread.
    # This takes ~2 seconds and completely prevents Windows multi-thread import / ONNX deadlocks,
    # while remaining well within the IDE's strict 5-second connection timeout.
    try:
        _ = _get_engine()._indexer._collection
    except Exception as e:
        print(f"[aivc] Failed to eagerly load Indexer on main thread: {e}", file=sys.stderr)

    # Note: We completely removed the background thread warmup here to prevent Windows native GIL / ONNX DLL deadlock.

    def _on_sync_pull():
        try:
            _get_engine().migrate_index()
            # Safely set warmed_up to False to trigger a synchronous warmup on the next user query
            _get_engine()._warmed_up = False
        except Exception as e:
            import sys
            print(f"Error during sync post-processing: {e}", file=sys.stderr)

    _syncer = BackgroundSyncer(_storage_root, on_pull_callback=_on_sync_pull)
    
    # Start background tasks
    _syncer.start()
    
    # Run MCP server
    mcp.run(transport="stdio")
