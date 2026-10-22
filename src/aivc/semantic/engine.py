"""
SemanticEngine: orchestrates Workspace + Indexer + CooccurrenceGraph + Searcher.

This is the single public facade for Phase 2.  It wraps the Phase 1 Workspace
(keeping it pure stdlib) and adds semantic indexing transparently on every
``create_commit`` call.

Option B architecture: Workspace is *owned* by SemanticEngine; the core module
is never aware of ChromaDB or SentenceTransformers.

Lazy Loading: The Indexer (ChromaDB + SentenceTransformer) and Searcher
(CrossEncoder) are only instantiated on first use, keeping ``__init__`` fast
for CLI-style invocations that only need Workspace or Graph features.
"""

import logging
import threading
import queue
import time
import atexit
import os
import re
from pathlib import Path
from typing import Any

from aivc.core.memory import Memory
from aivc.core.workspace import FileStatus, Workspace
from aivc.semantic.graph import CooccurrenceGraph
from aivc.config import get_machine_id, to_portable_path, from_portable_path
from aivc.sync.drive import NativeDriveSyncManager

logger = logging.getLogger(__name__)


class SemanticEngine:
    """High-level facade combining versioning + semantic search.

    Creates and owns:
    - A :class:`~aivc.core.workspace.Workspace` (Phase 1 core) — loaded eagerly
    - A :class:`~aivc.semantic.graph.CooccurrenceGraph` — loaded eagerly (SQLite, fast)
    - An :class:`~aivc.semantic.indexer.Indexer` (ChromaDB) — **lazy**
    - A :class:`~aivc.semantic.searcher.Searcher` (Cross-Encoder) — **lazy**

    All data is stored under ``storage_root``.
    """

    def __init__(self, storage_root: Path) -> None:
        """Initialise the lightweight sub-systems.

        Heavy ML dependencies (sentence-transformers, chromadb) are NOT loaded
        here.  They are loaded lazily on first ``create_commit`` or ``search``
        call.

        Args:
            storage_root: Single root directory for all AIVC data
                          (blobs, commits, ChromaDB, graph SQLite).
        """
        self._storage_root = storage_root
        self._workspace = Workspace(storage_root)
        self._graph = CooccurrenceGraph(storage_root)
        # Lazy — will be initialised on first access.
        self.__indexer = None
        self.__searcher = None
        self._local_hints_index = None
        
        # Prevent PyTorch/Chroma multithreading import gridlocks
        self._ml_lock = threading.RLock()
        self._warmed_up = False

        # Async Indexing & Sync
        self._index_queue = queue.Queue()
        self._sync_queue = queue.Queue()
        self._sync_manager = NativeDriveSyncManager(storage_root)
        
        self._indexing_thread = threading.Thread(target=self._indexing_worker_loop, daemon=True)
        self._sync_thread = threading.Thread(target=self._sync_worker_loop, daemon=True)
        
        self._indexing_thread.start()
        self._sync_thread.start()

        # Ensure graceful shutdown
        atexit.register(self.shutdown)

        # Register workspace reload callback
        self._workspace.register_reload_callback(self._on_workspace_reload)

    def _indexing_worker_loop(self):
        """Background worker thread to index memories from the queue."""
        while True:
            memory = self._index_queue.get()
            if memory is None: # Shutdown signal
                break
            try:
                if not self._warmed_up:
                    self.warmup()

                # Semantic indexing (triggers lazy load of indexer if needed)
                indexer = self._indexer
                if indexer is not None:
                    indexer.index_memory(memory)
            except Exception as e:
                # If we catch a late 'atexit' error here despite the property check
                if "atexit" in str(e):
                    pass
                else:
                    import sys
                    print(f"Error in async indexing for memory {memory.id}: {e}", file=sys.stderr)
            finally:
                self._index_queue.task_done()

    def _sync_worker_loop(self):
        """Background worker thread to push memories and latest files to cloud."""
        while True:
            memory = self._sync_queue.get()
            if memory is None:  # Shutdown signal
                break
            try:
                # Cloud Sync push (Memory JSON)
                self._sync_manager.push_memory(memory.id)
                # Push latest versions of changed/consulted files
                for c in getattr(memory, "changes", []):
                    if c.action in ("edited", "consulted", "added", "modified"):
                        self._sync_manager.push_latest_file(c.path)
            except Exception as e:
                import sys
                print(f"Error in async sync for memory {memory.id}: {e}", file=sys.stderr)
            finally:
                self._sync_queue.task_done()

    def shutdown(self, timeout: float = 5.0):
        """Signal workers to stop and wait for them to finish."""
        # 1. Signal indexing and sync independently
        self._index_queue.put(None)
        self._sync_queue.put(None)

        # 2. Wait for indexing and sync threads to finish
        self._indexing_thread.join(timeout=timeout/2)
        self._sync_thread.join(timeout=timeout/2)

    def get_index_queue_size(self) -> int:
        """Return the number of unfinished tasks (waiting + processing)."""
        return self._index_queue.unfinished_tasks + self._sync_queue.unfinished_tasks

    def wait_until_indexed(self, timeout: float = 30.0) -> bool:
        """Wait until all background tasks (indexing/sync) are complete.
        Returns True if finished, False if timed out.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get_index_queue_size() == 0:
                return True
            time.sleep(0.1)
        return False

    def _on_workspace_reload(self) -> None:
        """Callback from workspace when its state is reloaded from disk."""
        self._local_hints_index = None

    def warmup(self) -> None:
        """Eagerly load heavy ML components and reindex orphaned commits.

        This method:
        1. Forces lazy-loading of SentenceTransformer, CrossEncoder, and ChromaDB.
        2. Detects commits that exist on disk (JSON) but are missing from the
           ChromaDB vector index (e.g. because async indexing crashed on a
           previous run) and indexes them.

        Safe to call from a background thread; Python's import lock prevents
        race conditions with concurrent tool calls.
        """
        with self._ml_lock:
            if self._warmed_up:
                return

            import sys

            # Step 1: Force lazy evaluation of the heavy ML components
            _ = self._indexer._collection
            # _ = self._searcher._cross_encoder  # Commented out to prevent PyTorch background thread import deadlocks under Windows

            # Step 2: Reindex orphaned memories (on-disk JSON but not in ChromaDB)
            # Instead of self._workspace.get_log() which relies on linear chain from local HEAD (None on new machines),
            # we physically scan the commits directory to find all available memories.
            all_memories = []
            commits_dir = self._workspace._commits_dir
            if commits_dir.exists():
                for json_file in commits_dir.glob("*.json"):
                    try:
                        memory_id = json_file.stem
                        memory = self._workspace._load_memory(memory_id)
                        all_memories.append(memory)
                    except Exception as e:
                        print(
                            f"[aivc] Failed to load physical memory {json_file.name}: {e}",
                            file=sys.stderr,
                        )

            # Step 3: Automatically initialize head_commit_id if None and memories exist
            if self._workspace._state.get("head_commit_id") is None and all_memories:
                sorted_memories = sorted(all_memories, key=lambda m: m.timestamp, reverse=True)
                latest_memory = sorted_memories[0]
                self._workspace._state["head_commit_id"] = latest_memory.id
                self._workspace._save_state()
                print(
                    f"[aivc] Automatically initialized head_commit_id to latest physical commit: {latest_memory.id}",
                    file=sys.stderr,
                )

            # Step 4: Index missing physical commits in the co-occurrence graph
            try:
                existing_graph_ids = self._graph.get_indexed_commit_ids()
            except Exception as e:
                print(f"[aivc] Failed to read indexed commit IDs from graph: {e}", file=sys.stderr)
                existing_graph_ids = set()

            missing_graph = [m for m in all_memories if m.id not in existing_graph_ids]
            if missing_graph:
                print(
                    f"[aivc] Indexing {len(missing_graph)} physical memory(ies) in co-occurrence graph...",
                    file=sys.stderr,
                )
                for memory in missing_graph:
                    try:
                        self._graph.add_memory(memory)
                    except Exception as e:
                        print(
                            f"[aivc] Failed to add memory {memory.id} to co-occurrence graph: {e}",
                            file=sys.stderr,
                        )

            # Fetch all indexed IDs in a single, lightning-fast query
            try:
                indexed_ids = set(self._indexer._collection.get(include=[])["ids"])
            except Exception:
                indexed_ids = set()

            missing = [m for m in all_memories if m.id not in indexed_ids]

            if missing:
                print(
                    f"[aivc] Reindexing {len(missing)} orphaned memory(ies)...",
                    file=sys.stderr,
                )
                for memory in missing:
                    try:
                        self._indexer.index_memory(memory)
                    except Exception as e:
                        print(
                            f"[aivc] Failed to reindex {memory.id}: {e}",
                            file=sys.stderr,
                        )
                print(
                    f"[aivc] Warmup complete. Index now has {self._indexer._collection.count()} memory(ies).",
                    file=sys.stderr,
                )
            
            self._warmed_up = True

    def warmup_async(self) -> None:
        """Trigger background vector indexing of missing memories in a daemon thread."""
        self._warmed_up = False
        thread = threading.Thread(target=self.warmup, daemon=True, name="AIVC-SyncWarmup")
        thread.start()



    # ------------------------------------------------------------------
    # Lazy properties for heavy ML components
    # ------------------------------------------------------------------

    @property
    def _indexer(self):
        """Lazy-loaded Indexer (ChromaDB + SentenceTransformer bi-encoder)."""
        if self.__indexer is None:
            with self._ml_lock:
                if self.__indexer is None:
                    try:
                        from aivc.semantic.indexer import Indexer
                        self.__indexer = Indexer(self._storage_root)
                    except RuntimeError as e:
                        if "atexit" in str(e):
                            # We are likely shutting down; don't crash the background thread
                            return None
                        raise
        return self.__indexer

    @property
    def _searcher(self):
        """Lazy-loaded Searcher (Cross-Encoder reranking pipeline)."""
        if self.__searcher is None:
            with self._ml_lock:
                if self.__searcher is None:
                    try:
                        from aivc.semantic.searcher import Searcher
                        self.__searcher = Searcher(self._indexer)
                    except RuntimeError as e:
                        if "atexit" in str(e):
                            return None
                        raise
        return self.__searcher

    # ------------------------------------------------------------------
    # Commit lifecycle
    # ------------------------------------------------------------------

    def remember(
        self,
        title: str,
        journal: str = "",
        user_items: list[dict[str, Any]] | None = None,
        edited_files: list[str] | None = None,
        read_files: list[str] | None = None,
        urls: list[str] | None = None,
        note: str | None = None,
        consulted_files: list[str] | None = None,
    ) -> Memory:
        """Create a versioning memory and index it semantically (asynchronously)."""
        machine_id = get_machine_id()
        memory = self._workspace.remember(
            title=title,
            journal=journal,
            user_items=user_items,
            edited_files=edited_files,
            read_files=read_files,
            urls=urls,
            note=note,
            machine_id=machine_id,
            consulted_files=consulted_files,
        )

        # Step 2: graph update (SQLite, always fast)
        self._graph.add_memory(memory)

        # Step 3: async semantic indexing & cloud sync
        self._index_queue.put(memory)
        try:
            if self._sync_manager.enabled:
                self._sync_queue.put(memory)
        except Exception as e:
            logger.warning("Failed to queue memory %s for sync: %s", memory.id, e)

        # Step 4: Invalidate hints cache
        self._local_hints_index = None

        return memory

    def create_memory(
        self,
        title: str,
        note: str = "",
        read_files: list[str] | None = None,
        edited_files: list[str] | None = None,
        consulted_files: list[str] | None = None,
        urls: list[str] | None = None,
        user_items: list[dict[str, Any]] | None = None,
        journal: str = "",
    ) -> Memory:
        """Create a versioning memory (alias for remember)."""
        return self.remember(
            title=title,
            journal=journal or note,
            user_items=user_items,
            edited_files=edited_files,
            read_files=read_files,
            urls=urls,
            note=note,
            consulted_files=consulted_files,
        )

    def complete_user_items(self, item_ids: list[str]) -> list[str]:
        """Mark task items as completed across all memories and sync to Drive."""
        done_ids = self._workspace.complete_user_items(item_ids)
        if done_ids and self._sync_manager.enabled:
            import json
            try:
                for json_file in self._workspace._commits_dir.glob("*.json"):
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    for item in data.get("user_items", []):
                        if item.get("id") in done_ids:
                            self._sync_manager.push_memory(json_file.stem)
                            break
            except Exception as e:
                logger.warning("Failed to sync completed user_items: %s", e)
        return done_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 50,
        top_n: int = 5,
        filter_glob: str = "",
    ) -> list:
        """Semantic search over memory notes.

        Args:
            query: Free-text search query.
            top_k: Bi-Encoder recall breadth (capped at 50).
            top_n: Cross-Encoder reranked results to return.
            filter_glob: Optional glob pattern. If provided, restricts results
                         to memories that touch at least one matching file.

        Returns:
            A list of :class:`~aivc.semantic.searcher.SearchResult` sorted by
            relevance (descending).
        """
        if not self._warmed_up:
            self.warmup()

        try:
            if filter_glob:
                memory_ids = self._graph.get_memories_by_glob(filter_glob)
                if not memory_ids:
                    return []
                return self._searcher.search(query, top_k=top_k, top_n=top_n, filter_ids=memory_ids)
            
            return self._searcher.search(query, top_k=top_k, top_n=top_n)
        except Exception as e:
            err_str = str(e)
            if "Nothing found on disk" in err_str or "segment" in err_str.lower() or "hnsw" in err_str.lower():
                import sys
                print("[aivc] HNSW index missing or corrupt, rebuilding...", file=sys.stderr)
                try:
                    all_memories = []
                    commits_dir = self._workspace._commits_dir
                    if commits_dir.exists():
                        for json_file in commits_dir.glob("*.json"):
                            try:
                                memory = self._workspace._load_memory(json_file.stem)
                                all_memories.append(memory)
                            except Exception:
                                pass
                    self._indexer.reindex_all(all_memories)
                    
                    # Retry search
                    if filter_glob:
                        memory_ids = self._graph.get_memories_by_glob(filter_glob)
                        if not memory_ids:
                            return []
                        return self._searcher.search(query, top_k=top_k, top_n=top_n, filter_ids=memory_ids)
                    return self._searcher.search(query, top_k=top_k, top_n=top_n)
                except Exception as rebuild_err:
                    print(f"[aivc] Rebuild failed: {rebuild_err}", file=sys.stderr)
                    raise
            raise



    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_related_files(
        self, file_path: str, top_n: int = 10
    ) -> list[tuple[str, int]]:
        """Return files frequently committed alongside *file_path*.

        Args:
            file_path: Relative path of the reference file.
            top_n: Maximum number of co-occurring files to return.

        Returns:
            List of ``(file_path, cooccurrence_count)`` sorted descending.
        """
        return self._graph.get_related_files(to_portable_path(file_path), top_n=top_n)

    def get_memory_files(self, memory_id: str) -> list[str]:
        """Return file paths that were changed in a given memory."""
        return self._graph.get_memory_files(memory_id)

    def get_file_memories(self, file_path: str) -> list[str]:
        """Return memory IDs that have ever touched *file_path*."""
        return self._graph.get_file_memories(to_portable_path(file_path))

    def graph_vis_data(self) -> dict:
        """Return the graph in visualisation format (nodes + edges)."""
        return self._graph.to_vis_data()

    def get_file_node_data(self, connected_files: set[str] | None = None) -> list[dict]:
        """Return enriched data for file nodes (dashboard vis)."""
        return self._graph.get_file_node_data(connected_files=connected_files)

    def get_file_cooccurrences(self) -> list[dict]:
        """Return weighted file-to-file co-occurrence edges (dashboard vis)."""
        return self._graph.get_file_cooccurrences()

    # ------------------------------------------------------------------
    # Workspace pass-throughs
    # ------------------------------------------------------------------

    def get_tracked_files_metadata(self) -> dict[str, dict]:
        """Return the current metadata (mtime, size) for all tracked files from state."""
        return self._workspace.get_tracked_files_metadata()

    def get_tracked_paths(self) -> list[str]:
        """Return exactly the tracked paths from workspace (fast, no disk I/O)."""
        return self._workspace.get_tracked_paths()


    def get_log(
        self,
        limit: int = 20,
        offset: int = 0,
        include_remote: bool = True,
        only_local: bool = False,
    ) -> list[Memory]:
        """Return up to *limit* memories in reverse chronological order."""
        return self._workspace.get_log(
            limit=limit,
            offset=offset,
            include_remote=include_remote,
            only_local=only_local,
        )

    def get_file_history(self, file_path: str) -> list[dict]:
        """Return memories that touched *file_path* with metadata.

        Returns:
            List of ``{"memory_id": str, "title": str, "timestamp": str}``
            sorted by timestamp descending.
        """
        return self._graph.get_file_memories_with_metadata(file_path)

    def pull_latest_file(self, file_path: str, target_path: str | Path | None = None) -> Path:
        """Download latest version of file from Google Drive."""
        return self._sync_manager.pull_latest_file(file_path, target_path)

    def push_latest_file(self, file_path: str | Path) -> bool:
        """Push latest version of file to Google Drive."""
        return self._sync_manager.push_latest_file(file_path)

    def get_memory(self, memory_id: str) -> Memory:
        """Load a single memory by ID."""
        return self._workspace.get_memory(memory_id)

    def find_child_memory(self, memory_id: str) -> Memory | None:
        """Find the memory that has *memory_id* as its parent."""
        return self._workspace.find_child_memory(memory_id)

    def read_file_at_memory(self, file_path: str, memory_id: str) -> bytes:
        """Read a tracked file as it was at a specific memory."""
        return self._workspace.read_file_at_memory(file_path, memory_id)
    
    def migrate_index(self) -> None:
        """Explicitly migrate JSON memories to SQLite index."""
        self._workspace.migrate_index()

    def _get_local_hints_index(self) -> dict[str, list[str]]:
        """Lazy-build an inverted index {basename: [paths]} for fast lookups."""
        if self._local_hints_index is None:
            index = {}
            # Use get_tracked_paths for O(1) path retrieval (no FileStatus overhead)
            for path_str in self._workspace.get_tracked_paths():
                basename = Path(path_str).name
                if basename not in index:
                    index[basename] = []
                index[basename].append(path_str)
            self._local_hints_index = index
        return self._local_hints_index

    def find_local_equivalent(self, remote_path: str, remote_blob_hash: str | None = None) -> str | None:
        """Try to find a local tracked file that matches a remote path.
        
        Optimised O(1) basename lookup. Handles both Unix and Windows remote path styles.
        """
        # Cross-platform basename extraction
        remote_normalized = remote_path.replace("\\", "/")
        parts = [p for p in remote_normalized.split("/") if p]
        
        if not parts:
            return None
            
        remote_basename = parts[-1]
        remote_parent_name = parts[-2] if len(parts) > 1 else ""
        
        # O(1) lookup in inverted index
        candidates = self._get_local_hints_index().get(remote_basename, [])
        if not candidates:
            return None
        
        final_candidates = []
        for local_path in candidates:
            lp = Path(local_path)
            # 1. Check strong match via blob hash if available
            if remote_blob_hash:
                # _index is CoreIndex (SQLite) - this is still an I/O hit but 
                # only for files already matching the basename.
                local_blobs = self._workspace._index.get_blob_hashes_for_file(local_path)
                if remote_blob_hash in local_blobs:
                    return local_path
            
            # 2. Parent folder match (depth 1)
            local_parent_name = lp.parent.name if lp.parent else ""
            if local_parent_name == remote_parent_name:
                final_candidates.append(local_path)
        
        return final_candidates[0] if final_candidates else None

