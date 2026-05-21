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
from aivc.config import get_machine_id
from aivc.sync.drive import NativeDriveSyncManager


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
                # 1. Semantic indexing (triggers lazy load of indexer if needed)
                indexer = self._indexer
                if indexer is not None:
                    indexer.index_memory(memory)
                
                # 2. Forward to sync queue if enabled
                if self._sync_manager.enabled:
                    self._sync_queue.put(memory)
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
        """Background worker thread to push memories/blobs to cloud."""
        while True:
            memory = self._sync_queue.get()
            if memory is None: # Shutdown signal
                break
            try:
                # Cloud Sync push (Memory JSON only)
                self._sync_manager.push_memory(memory.id)
            except Exception as e:
                import sys
                print(f"Error in async sync for memory {memory.id}: {e}", file=sys.stderr)
            finally:
                self._sync_queue.task_done()

    def shutdown(self, timeout: float = 5.0):
        """Signal workers to stop and wait for them to finish."""
        # 1. Signal indexing
        self._index_queue.put(None)
        # 2. Wait for indexing to finish (so it pushes remaining to sync)
        self._indexing_thread.join(timeout=timeout/2)
        
        # 3. Signal sync
        self._sync_queue.put(None)
        # 4. Wait for sync
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
        import sys

        # Step 1: Force lazy evaluation of the heavy ML components
        _ = self._indexer._collection
        _ = self._searcher._cross_encoder

        # Step 2: Reindex orphaned memories (on-disk JSON but not in ChromaDB)
        all_memories = self._workspace.get_log(limit=999999)
        indexed_count = self._indexer._collection.count()

        if len(all_memories) > indexed_count:
            missing = []
            for memory in all_memories:
                if not self._indexer.is_indexed(memory.id):
                    missing.append(memory)

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

    def create_memory(
        self,
        title: str,
        note: str,
        consulted_files: list[str] | None = None
    ) -> Memory:
        """Create a versioning memory and index it semantically (asynchronously).

        1. Delegates to :meth:`Workspace.create_memory` (which detects diffs,
           stores blobs, and persists the memory JSON).
        2. Updates the co-occurrence graph.
        3. Pushes the memory to the background worker queue for semantic indexing.

        Args:
            title: Short memory title.
            note: Detailed Markdown note (the 'memory').
            consulted_files: Optional list of file paths consulted.

        Returns:
            The newly created :class:`~aivc.core.memory.Memory`.

        Raises:
            RuntimeError: if no changes detected and no files consulted.
        """
        # Step 1: core versioning (may raise RuntimeError if no changes)
        # Inject machine_id
        machine_id = get_machine_id()
        memory = self._workspace.create_memory(
            title, note, consulted_files=consulted_files, machine_id=machine_id
        )

        # Step 2: graph update (SQLite, always fast)
        self._graph.add_memory(memory)

        # Step 3: async semantic indexing
        self._index_queue.put(memory)

        # Step 4: Invalidate hints cache (new files might have been auto-tracked)
        self._local_hints_index = None

        return memory

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
        if filter_glob:
            memory_ids = self._graph.get_memories_by_glob(filter_glob)
            if not memory_ids:
                return []
            return self._searcher.search(query, top_k=top_k, top_n=top_n, filter_ids=memory_ids)
        
        return self._searcher.search(query, top_k=top_k, top_n=top_n)

    # Extensions known to be binary or useless for code/text search
    _SKIP_EXTS: set[str] = {
        '.pyc', '.pyo', '.map', '.npy', '.npz', '.pkl', '.pickle', '.bin',
        '.exe', '.dll', '.so', '.o', '.a', '.woff', '.woff2', '.ttf', '.eot',
        '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.webp',
        '.mp3', '.mp4', '.wav', '.zip', '.gz', '.tar', '.rar', '.7z', '.pdf',
        '.pth', '.pt', '.onnx', '.safetensors', '.parquet', '.feather',
        '.h5', '.hdf5', '.db', '.sqlite', '.sqlite3', '.tfevents',
    }
    _MAX_FILE_SIZE: int = 512_000  # 512 KB

    def _get_searchable_paths(self) -> list[str]:
        """Return tracked paths filtered to only text-like, reasonably sized files.
        
        Skips binary extensions and files > 512KB to avoid wasting I/O
        on enormous CSVs, numpy arrays, source-maps, etc.
        """
        tracked_paths = self._workspace.get_tracked_paths()
        metadata = self._workspace.get_tracked_files_metadata()
        filtered = []
        for p in tracked_paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in self._SKIP_EXTS:
                continue
            m = metadata.get(p, {})
            sz = m.get("size", 0) if isinstance(m, dict) else 0
            if sz and sz > self._MAX_FILE_SIZE:
                continue
            filtered.append(p)
        return filtered

    def _grep_search(
        self,
        paths: list[str],
        terms: list[str],
        *,
        is_regex: bool = False,
        case_sensitive: bool = False,
    ) -> list[str]:
        """Use GNU grep subprocess to find files matching ALL terms (AND logic).
        
        Falls back to a pure Python implementation if grep/xargs is not available
        (e.g., on Windows).
        """
        import subprocess
        import tempfile
        import shutil

        # Fallback pure Python matcher
        def _python_fallback_search(candidate_paths: list[str], search_terms: list[str]) -> list[str]:
            import re
            matched = []
            for path_str in candidate_paths:
                try:
                    p = Path(path_str)
                    if not p.is_file():
                        continue
                    
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    all_match = True
                    for term in search_terms:
                        if is_regex:
                            flags = re.IGNORECASE if not case_sensitive else 0
                            if not re.search(term, content, flags):
                                all_match = False
                                break
                        else:
                            t = term if case_sensitive else term.lower()
                            c = content if case_sensitive else content.lower()
                            if t not in c:
                                all_match = False
                                break
                    if all_match:
                        matched.append(path_str)
                except Exception:
                    continue
            return matched

        current_paths = paths
        grep_flags = ["-l"]  # list matching file names only
        if not case_sensitive:
            grep_flags.append("-i")
        if not is_regex:
            grep_flags.append("-F")  # fixed-string (faster)

        # Quick check if xargs and grep are available in PATH
        if not shutil.which("xargs") or not shutil.which("grep"):
            return _python_fallback_search(paths, terms)

        for term in terms:
            if not current_paths:
                return []

            # Write current candidate paths to a temp file (one per line)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            )
            try:
                for p in current_paths:
                    tmp.write(p + "\n")
                tmp.close()
                # Use xargs -a to read from the temp file directly, avoiding cat | xargs.
                # Use shell=False and pass arguments as a list to prevent command injection.
                cmd = ["xargs", "-a", tmp.name, "-d", "\n", "grep"] + grep_flags + ["--", term]
                result = subprocess.run(
                    cmd,
                    shell=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=30,
                )
                current_paths = [
                    l for l in result.stdout.strip().split("\n") if l
                ]
            except (FileNotFoundError, OSError, subprocess.SubprocessError):
                # If command fails due to environment mismatch (e.g. missing xargs/grep on Windows)
                return _python_fallback_search(current_paths, [term])
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass

        return current_paths

    def search_files(
        self,
        query: str,
        top_n: int = 5,
        is_regex: bool = False,
        case_sensitive: bool = False,
    ) -> list[dict]:
        """Perform a fast lexical search (Keyword or Regex) on current tracked file contents.

        Uses GNU grep as a subprocess for I/O-optimal matching, with
        intelligent pre-filtering to skip binary/large files.

        For keyword searches (default), uses AND logic: finds files where ALL
        provided words are present, regardless of order or location.

        Args:
            query: Search terms (e.g. "auth error") or a regex pattern.
            top_n: Number of results to return.
            is_regex: Whether to treat query as a regular expression.
            case_sensitive: Whether the search should be case sensitive.

        Returns:
            A list of dicts: {"path": str, "score": float, "snippet": str}.
        """
        searchable = self._get_searchable_paths()
        if not searchable:
            return []

        # Split query into terms for AND matching (keyword mode)
        # For regex mode, treat the whole query as a single pattern
        terms = [query] if is_regex else query.split()
        if not terms:
            return []

        # Phase 1: grep to find candidate files (fast, C-level I/O)
        matching_paths = self._grep_search(
            searchable, terms, is_regex=is_regex, case_sensitive=case_sensitive
        )

        if not matching_paths:
            return []

        # Phase 2: score + snippet only for matching files (small set)
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(query, flags)
        else:
            search_terms = (
                [t.lower() for t in terms]
                if not case_sensitive
                else list(terms)
            )

        results: list[dict] = []
        for path in matching_paths:
            try:
                content = Path(path).read_text(
                    encoding="utf-8", errors="ignore"
                )
                if is_regex:
                    match = pattern.search(content)
                    if not match:
                        continue
                    score = len(pattern.findall(content))
                    first_pos = match.start()
                else:
                    haystack = (
                        content if case_sensitive else content.lower()
                    )
                    score = sum(haystack.count(t) for t in search_terms)
                    first_pos = haystack.find(search_terms[0])

                # Build snippet around first match
                snip_start = max(0, first_pos - 100)
                snip_end = min(len(content), snip_start + 200)
                snippet = (
                    content[snip_start:snip_end]
                    .replace("\n", " ")
                    .strip()
                )
                if snip_start > 0:
                    snippet = "..." + snippet
                if snip_end < len(content):
                    snippet = snippet + "..."

                results.append(
                    {"path": path, "score": float(score), "snippet": snippet}
                )
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    def search_files_bm25(self, query: str, top_n: int = 5) -> list[dict]:
        """Legacy wrapper for compatibility."""
        return self.search_files(query, top_n=top_n)

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
        return self._graph.get_related_files(file_path, top_n=top_n)

    def get_memory_files(self, memory_id: str) -> list[str]:
        """Return file paths that were changed in a given memory."""
        return self._graph.get_memory_files(memory_id)

    def get_file_memories(self, file_path: str) -> list[str]:
        """Return memory IDs that have ever touched *file_path*."""
        return self._graph.get_file_memories(file_path)

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

    def track(self, path: str, ignores: list[str] | None = None) -> dict[str, Any]:
        """Track a file, directory, or glob. See :meth:`Workspace.track`."""
        self._local_hints_index = None
        return self._workspace.track(path, ignores)

    def get_watched_dirs(self) -> dict[str, dict[str, Any]]:
        """Return exactly the watched directories state from workspace."""
        return self._workspace.get_watched_dirs()

    def get_tracked_files_metadata(self) -> dict[str, dict]:
        """Return the current metadata (mtime, size) for all tracked files from state."""
        return self._workspace.get_tracked_files_metadata()

    def get_tracked_paths(self) -> list[str]:
        """Return exactly the tracked paths from workspace (fast, no disk I/O)."""
        return self._workspace.get_tracked_paths()

    def untrack(self, path_or_glob: str) -> None:
        """Remove a file/dir from tracking. See :meth:`Workspace.untrack`."""
        self._local_hints_index = None
        self._workspace.untrack(path_or_glob)

    def get_status(self) -> list[FileStatus]:
        """Return status of all tracked files. See :meth:`Workspace.get_status`."""
        return self._workspace.get_status()

    def get_log(self, limit: int = 20, offset: int = 0) -> list[Memory]:
        """Return up to *limit* memories in reverse chronological order."""
        return self._workspace.get_log(limit, offset)

    def get_file_history(self, file_path: str) -> list[dict]:
        """Return memories that touched *file_path* with metadata.

        Returns:
            List of ``{"memory_id": str, "title": str, "timestamp": str}``
            sorted by timestamp descending.
        """
        return self._graph.get_file_memories_with_metadata(file_path)

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

