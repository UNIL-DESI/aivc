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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic_settings.*")
warnings.filterwarnings("ignore", message=".*lifespan.*")
warnings.filterwarnings("ignore", message=".*file_cache is only supported.*")

import logging
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
logging.getLogger("googleapiclient").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
logging.getLogger("aivc").setLevel(logging.WARNING)

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# Set mission-critical environment variables before heavy ML imports
# This completely bypasses the 5-minute atexit/thread deadlock on Windows
# caused by ChromaDB PostHog telemetry failing on corporate firewalls.
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# System prompt — injected into every LLM context using this server
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """AIVC est le système de mémoire versionnée persistante pour agents LLM.
Il enregistre des souvenirs structurés (titre, journal, tâches utilisateur, fichiers modifiés et consultés).
La mémoire est indexée sémantiquement pour permettre la recherche par le sens via recall.
Les outils permettent de consulter les souvenirs passés, l'historique des fichiers et de restaurer des fichiers."""

# ---------------------------------------------------------------------------
# Bootstrap — engine initialisation
# ---------------------------------------------------------------------------

from aivc.config import get_storage_root

_storage_root = get_storage_root()

# SemanticEngine is imported here (triggering a fast eager init of Workspace +
# SQLite graph; the heavy ML components remain lazy until first use).
from aivc.semantic.engine import SemanticEngine  # noqa: E402
from aivc.sync.background import BackgroundSyncer
from aivc.config import get_machine_id, to_portable_path
import threading

_engine: SemanticEngine | None = None
_local_machine_id: str | None = None
_syncer: BackgroundSyncer | None = None
_lock = threading.Lock()


def _on_sync_pull():
    try:
        _get_engine().migrate_index()
        _get_engine().warmup_async()
    except Exception as e:
        import sys
        print(f"Error during sync post-processing: {e}", file=sys.stderr)


def _get_syncer() -> BackgroundSyncer:
    global _syncer
    if _syncer is None:
        with _lock:
            if _syncer is None:
                _syncer = BackgroundSyncer(_storage_root, on_pull_callback=_on_sync_pull)
                if _syncer.manager.enabled:
                    _syncer.start()
    return _syncer


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
        syncer = _get_syncer()
        if syncer.manager.enabled:
            syncer.trigger_sync()
    return _engine

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(name="aivc", instructions=_SYSTEM_PROMPT, log_level="ERROR")

# Prevent duplicate handlers on root logger configured by dependencies
for _h in list(logging.root.handlers):
    _h.setLevel(logging.WARNING)
logging.root.setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helper formatting functions
# ---------------------------------------------------------------------------


def _format_cold_start_payload(context_hint: str = "") -> str:
    """Standard assertive payload returned when no memories match or store is empty.
    Forces calling LLMs to immediately break out of memory exploration loops
    and fallback to raw file tools (view_file, grep_search, list_dir).
    """
    hint = f" ({context_hint})" if context_hint else ""
    return (
        f"Aucun souvenir trouvé{hint} (mémoire vide / première exploration sur ce composant). "
        "BASCULE IMMÉDIATE : Ne consultez pas davantage la mémoire, explorez directement le code "
        "via grep_search / view_file / list_dir. Appelez remember dès que vous progressez pour peupler la mémoire !"
    )


def _render_file_tree(paths: list[str], path_extras: dict[str, str] = None, indent_prefix: str = "  ") -> str:
    """Render a list of absolute paths as a hierarchical tree."""
    if not paths:
        return "—"

    if len(paths) == 1:
        common_root = os.path.dirname(os.path.abspath(os.path.expanduser(paths[0])))
    else:
        try:
            # Safely find a common directory prefix
            abs_paths = [os.path.abspath(os.path.expanduser(p)) for p in paths]
            common_root = os.path.commonpath([os.path.dirname(p) for p in abs_paths])
        except ValueError:
            common_root = ""

    tree: dict = {}

    for path in paths:
        norm_p = os.path.abspath(os.path.expanduser(path))
        if common_root:
            try:
                rel_path = os.path.relpath(norm_p, common_root)
            except ValueError:
                rel_path = norm_p
        else:
            rel_path = norm_p

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

    tree_str = "\n".join(lines) if len(lines) > 0 else f"{indent_prefix}\u2014"
    return f"```text\n{tree_str}\n```"


def _format_changes_compressed(changes, machine_id=None, memory_id=None, parent_id=None) -> str:
    """Render tracked file changes as a clear hierarchical tree."""
    if not changes:
        return "  (no tracked files recorded)"

    paths = []
    extras = {}

    for c in changes:
        paths.append(c.path)
        extras[c.path] = f" [{c.action}]"

    return _render_file_tree(paths, extras, indent_prefix="  ")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def remember(
    title: str,
    journal: str,
    main_project_file: str,
    tasks: list[dict[str, Any]] | None = None,
    user_items: list[dict[str, Any]] | None = None,
    edited_files: list[str] | None = None,
    read_files: list[str] | None = None,
    urls: list[str] | None = None,
    roadmap_items: list[dict[str, Any]] | None = None,
) -> str:
    """Enregistre un souvenir structuré avec journal, tâches (états: new, await_validation, validated) et fichiers.

    Args:
        title: Titre résumant le souvenir.
        journal: Récit détaillé de session.
        main_project_file: Chemin du fichier principal identifiant le projet ou sujet de ce souvenir (ex. la note maîtresse du projet dans un coffre Obsidian ou le README.md d'un dépôt de code). Doit exister sur le disque. Ajouté automatiquement aux fichiers consultés (read_files) s'il n'est ni dans read_files ni dans edited_files, et utilisé comme cible unique de rattachement des tâches (user_items).
        tasks: Liste unifiée des tâches.
        user_items: Actions utilisateur.
        edited_files: Fichiers édités ou créés.
        read_files: Fichiers consultés.
        urls: URLs consultées.
        roadmap_items: Tâches de roadmap.
    """
    import asyncio
    from pathlib import Path

    actual_read_files = read_files or []
    actual_edited_files = edited_files or []
    actual_urls = urls or []
    actual_user_items = user_items or []
    actual_tasks = tasks or []

    engine = _get_engine()

    engine_kwargs = {
        "title": title,
        "journal": journal,
        "main_project_file": main_project_file,
        "user_items": actual_user_items,
        "read_files": actual_read_files,
        "edited_files": actual_edited_files,
        "urls": actual_urls,
    }
    if tasks is not None:
        engine_kwargs["tasks"] = tasks
    if roadmap_items is not None:
        engine_kwargs["roadmap_items"] = roadmap_items

    memory = await asyncio.to_thread(
        engine.remember,
        **engine_kwargs
    )

    try:
        syncer = _get_syncer()
        if syncer and syncer.manager.enabled:
            syncer.trigger_sync()
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            from aivc.sync.semantic_export import trigger_semantic_export_async
            trigger_semantic_export_async(_storage_root)
    except Exception as e:
        import sys
        print(f"Warning: Failed to trigger instant sync push: {e}", file=sys.stderr)

    if memory.user_items:
        lines = []
        for it in memory.user_items:
            task_id = it.get("id") or "task"
            status = it.get("task_status") or ("validated" if it.get("done") else "new")
            lines.append(f"[{task_id}] {status}")
        return "\n".join(lines)

    short_cid = memory.id[:8] if memory.id else "ok"
    return f"OK ({short_cid})"


@mcp.tool()
async def recall(query: str, top_n: int = 5, filter_glob: str = "", only_local: bool = False) -> str:
    """Recherche sémantiquement les souvenirs passés par pertinence."""
    import asyncio
    top_n = min(top_n, 20)

    import os
    import sys
    default_disable = "True" if sys.platform == "win32" else "False"
    disable_cross = os.environ.get("AIVC_DISABLE_CROSS_ENCODER", default_disable).lower() == "true"
    engine = _get_engine()
    if not disable_cross:
        try:
            searcher = engine._searcher
            if searcher is not None:
                _ = searcher._cross_encoder
        except Exception as e:
            print(f"[aivc] Failed to eagerly load CrossEncoder on main thread: {e}", file=sys.stderr)

    indexing_queue_size = engine.get_index_queue_size()
    warning_header = ""
    if indexing_queue_size > 0:
        warning_header = f"⚠️  Note: {indexing_queue_size} recent memory(ies) are still being indexed and may be missing from search results.\n\n"

    results = await asyncio.to_thread(engine.search, query, top_n=top_n, filter_glob=filter_glob)

    if not results:
        return warning_header + _format_cold_start_payload()

    memory_lines = []
    for i, r in enumerate(results, 1):
        m_id = getattr(r, 'machine_id', "")
        remote_tag = f" [Remote: {m_id}]" if m_id and m_id != _local_machine_id else ""
        
        indented_snippet = "\n".join(f"   > {line}" for line in r.snippet.splitlines())
        memory_lines.append(
            f"{i}. [{r.timestamp[:10]}] {r.title}{remote_tag} (ID: {r.memory_id})\n"
            f"{indented_snippet}"
        )

    file_counter: Counter[str] = Counter()
    for r in results:
        file_counter.update(r.file_paths)

    paths = []
    extras = {}
    for fp, count in file_counter.most_common(10):
        paths.append(fp)
        extra_parts = [f"(in {count}/{len(results)} results)"]
        extras[fp] = " " + " ".join(extra_parts)

    output = warning_header + "## Matching Memories\n\n"
    output += "\n".join(memory_lines)

    if paths:
        output += "\n\n## Most Relevant Files\n"
        output += _render_file_tree(paths, extras, indent_prefix="  ")

    output += "\n\n💡 Use `consult_memory(memory_id)` to read a full note."
    return output


@mcp.tool()
def consult_memory(memory_id: str) -> str:
    """Affiche les détails, le journal, les tâches utilisateur et les fichiers d'un souvenir."""
    engine = _get_engine()
    memory = engine.get_memory(memory_id)
    try:
        engine.auto_restore_brain_files(memory)
    except Exception:
        pass

    prev_str = ""
    if memory.parent_id:
        try:
            parent = _get_engine().get_memory(memory.parent_id)
            prev_str = f"- ⬆️ **Prev** : {parent.title} (ID: {parent.id})\n\n"
        except KeyError:
            prev_str = f"- ⬆️ **Prev** : (metadata not found) (ID: {memory.parent_id})\n\n"

    next_str = ""
    try:
        child = _get_engine().find_child_memory(memory_id)
        if child:
            next_str = f"- ⬇️ **Next** : {child.title} (ID: {child.id})\n\n"
    except Exception:
        pass

    context_block = f"{prev_str}{next_str}" if (prev_str or next_str) else ""

    items_block = ""
    if getattr(memory, "user_items", None):
        item_lines = []
        for it in memory.user_items:
            status = "[x]" if it.get("done") else "[ ]"
            it_id = it.get("id", "")
            task = it.get("task", "")
            details = it.get("details", "")
            line = f"- {status} `[{it_id}]` {task}"
            if details:
                line += f"\n    *Détails :* {details}"
            item_lines.append(line)
        items_block = "## Actions Utilisateur (user_items)\n\n" + "\n".join(item_lines) + "\n\n"

    changes_sections = []
    modified_changes = [c for c in memory.changes if c.action in ("edited", "modified", "added", "deleted")]
    if modified_changes:
        modified_tree = _format_changes_compressed(
            modified_changes, memory.machine_id, memory.id, memory.parent_id
        )
        changes_sections.append(f"### Fichiers modifiés\n{modified_tree}")
    else:
        changes_sections.append("### Fichiers modifiés\n  (aucun)")

    consulted_changes = [c for c in memory.changes if c.action == "consulted"]
    if consulted_changes:
        consulted_tree = _format_changes_compressed(
            consulted_changes, memory.machine_id, memory.id, memory.parent_id
        )
        changes_sections.append(f"### Fichiers consultés\n{consulted_tree}")
    else:
        changes_sections.append("### Fichiers consultés\n  (aucun)")

    changes_summary_str = "\n\n".join(changes_sections)

    machine_line = f"**Machine**   : {memory.machine_id}\n" if memory.machine_id else ""

    urls_section = ""
    memory_urls = getattr(memory, "urls", []) or []
    if memory_urls:
        urls_list_str = "\n".join(f"- {url}" for url in memory_urls)
        urls_section = f"## URLs / Links Consulted\n\n{urls_list_str}\n\n"

    journal_text = getattr(memory, "journal", "") or memory.note

    return (
        f"# Memory: {memory.title}\n\n"
        f"**ID**        : {memory.id}\n"
        f"**Timestamp** : {memory.timestamp}\n"
        f"**Parent**    : {memory.parent_id or 'none (initial memory)'}\n"
        f"{machine_line}"
        f"{context_block}"
        f"{items_block}"
        f"## Journal\n\n{journal_text}\n\n"
        f"## Files Recorded\n\n{changes_summary_str}\n\n"
        f"{urls_section}"
    )


@mcp.tool()
def get_file_history(file_path: str) -> str:
    """Affiche l'historique chronologique des souvenirs liés à un fichier et leurs tâches ouvertes."""
    engine = _get_engine()
    memories = engine._workspace.get_file_history(file_path)

    if not memories:
        return _format_cold_start_payload(f"pour `{file_path}`")

    lines = [f"## AIVC History for: `{file_path}`\n"]
    lines.append(f"{len(memories)} memory(ies) have touched this file:\n")

    from aivc.config import to_portable_path
    target_portable = to_portable_path(file_path).lower()

    for mem in memories:
        lines.append(f"- [{mem.timestamp[:10]}] **{mem.title}** (ID: `{mem.id}`)")
        mpf = getattr(mem, "main_project_file", None)
        mpf_portable = to_portable_path(mpf).lower() if mpf else ""
        if not mpf_portable or mpf_portable == target_portable:
            open_items = [
                it for it in getattr(mem, "user_items", [])
                if not it.get("done", False) and it.get("task_status") != "validated"
            ]
            if open_items:
                for it in open_items:
                    it_id = it.get("id", "")
                    task = it.get("task", "")
                    lines.append(f"    - [ ] `[{it_id}]` {task}")

    lines.append(
        "\n💡 Use `consult_memory(memory_id)` to read the full note of a specific memory."
        "\n💡 Use `restore_file(file_path)` to restore the latest complete version from Google Drive."
    )
    return "\n".join(lines)


def complete_user_items(item_ids: list[str]) -> str:
    """Marque une ou plusieurs tâches utilisateur comme terminées (deprecated - utiliser remember avec task_status='validated')."""
    if not item_ids:
        return "Aucun identifiant de tâche spécifié."

    engine = _get_engine()
    done_ids = engine.complete_user_items(item_ids)

    if not done_ids:
        return f"Aucune tâche trouvée ou modifiée pour les IDs : {item_ids}"

    return f"✅ Tâche(s) marquée(s) comme terminée(s) : {', '.join(done_ids)}"


@mcp.tool()
def restore_file(file_path: str, target_path: str | None = None) -> str:
    """Télécharge la dernière version complète d'un fichier depuis Google Drive."""
    engine = _get_engine()
    try:
        dest = engine.pull_latest_file(file_path, target_path)
        return f"✅ Fichier restauré avec succès depuis Google Drive vers : `{dest}`"
    except Exception as e:
        return f"❌ Échec de la restauration de `{file_path}` : {e}"






# No background watchers active

# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    
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

    # Ensure background syncer is started
    _get_syncer()
    
    # Run MCP server
    mcp.run(transport="stdio")
