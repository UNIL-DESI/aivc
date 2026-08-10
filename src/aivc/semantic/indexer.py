"""
Indexer: vectorises memory notes and persists them in a local ChromaDB collection.

The text indexed per memory is: ``f"{memory.title}\\n\\n{memory.note}"`` so
that semantic search captures both the short title and the detailed Markdown note.

ChromaDB is initialised with a custom SentenceTransformer embedding function so
that the same model is used consistently at index time and search time (Bi-Encoder
stage) — ensuring correct distance comparisons with the Cross-Encoder stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding

if TYPE_CHECKING:
    from aivc.core.memory import Memory

from aivc.config import BI_ENCODER_MODEL

_COLLECTION_NAME = "aivc_memories"

# Module-level singleton to prevent ChromaDB's build_from_config from
# loading the model a second time (~5s wasted on Windows NTFS).
_shared_model: TextEmbedding | None = None


def _get_shared_model() -> TextEmbedding:
    """Return the shared TextEmbedding model (lazy singleton)."""
    global _shared_model
    if _shared_model is None:
        # Use FastEmbed's TextEmbedding for faster embeddings and no torch dependency
        _shared_model = TextEmbedding(BI_ENCODER_MODEL)
    return _shared_model


class _FastEmbedEF(EmbeddingFunction):
    """ChromaDB-compatible embedding function backed by FastEmbed."""

    def __init__(self, model: TextEmbedding) -> None:
        self._model = model

    @staticmethod
    def name() -> str:
        # Spoof the original name to avoid ChromaDB schema validation errors
        # on existing collections during the FastEmbed migration.
        return "aivc_sentence_transformer"

    def get_config(self) -> dict:
        return {"name": self.name()}

    @staticmethod
    def build_from_config(config: dict) -> "_FastEmbedEF":
        # Reuse the shared singleton — do NOT create a new model instance
        return _FastEmbedEF(_get_shared_model())

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return [e.tolist() for e in self._model.embed(list(input))]


class Indexer:
    """Manages the ChromaDB collection for AIVC memory notes.

    Disk layout under ``storage_root``:
        chromadb/      — ChromaDB persistent database directory
    """

    _CHROMA_DIR = "chromadb"

    def __init__(self, storage_root: Path) -> None:
        """Initialise the ChromaDB client and load the bi-encoder model.

        Args:
            storage_root: Root directory shared with BlobStore / Workspace.
                          ChromaDB data will live in ``storage_root/chromadb/``.

        Raises:
            RuntimeError: if ChromaDB or the bi-encoder model cannot be loaded.
        """
        self._chroma_dir = storage_root / self._CHROMA_DIR
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        self._model = _get_shared_model()
        self._ef = _FastEmbedEF(self._model)

        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_text(memory: "Memory") -> str:
        """Build the text to be vectorised for a given memory."""
        urls_block = (
            ("\n\nURLs:\n" + "\n".join(memory.urls))
            if getattr(memory, "urls", None)
            else ""
        )
        return f"{memory.title}\n\n{memory.note}{urls_block}"

    @staticmethod
    def _memory_metadata(memory: "Memory") -> dict:
        """Build the metadata dict stored alongside the vector."""
        file_paths = [c.path for c in memory.changes if c.action != "deleted"]
        urls = getattr(memory, "urls", [])
        return {
            "memory_id": memory.id,
            "title": memory.title,
            "timestamp": memory.timestamp,
            "machine_id": memory.machine_id,
            # ChromaDB metadata values must be str/int/float/bool.
            # Store file paths as a comma-joined string.
            "file_paths": "\n".join(file_paths),
            "urls": "\n".join(urls),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_memory(self, memory: "Memory") -> None:
        """Upsert a memory into the vector index.

        Idempotent: calling with the same memory twice is safe.

        Args:
            memory: The memory to index.

        Raises:
            ValueError: if the memory has no id or note.
        """
        if not memory.id:
            raise ValueError("Cannot index a memory with an empty id.")
        if not memory.note.strip():
            raise ValueError(f"Cannot index memory {memory.id!r}: note is empty.")

        text = self._memory_text(memory)
        metadata = self._memory_metadata(memory)

        self._collection.upsert(
            ids=[memory.id],
            documents=[text],
            metadatas=[metadata],
        )

    def remove_memory(self, memory_id: str) -> None:
        """Remove a memory from the index.

        Args:
            memory_id: UUID of the memory to remove.

        Raises:
            KeyError: if the memory is not found in the index.
        """
        if not self.is_indexed(memory_id):
            raise KeyError(f"Memory {memory_id!r} is not in the index.")
        self._collection.delete(ids=[memory_id])

    def is_indexed(self, memory_id: str) -> bool:
        """Return True if the memory is already in the index."""
        result = self._collection.get(ids=[memory_id])
        return len(result["ids"]) > 0

    def reindex_all(self, memories: list["Memory"]) -> None:
        """Clear the collection and re-index all provided memories.

        Useful after a migration or if the index becomes stale.

        Args:
            memories: All memories to (re-)index.
        """
        # Wipe the collection by deleting all existing IDs.
        existing = self._collection.get()
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

        for memory in memories:
            self.index_memory(memory)

    def query(
        self,
        query_text: str,
        top_k: int,
        filter_ids: list[str] | None = None,
    ) -> list[dict]:
        """Run a bi-encoder query and return the top-k raw results.

        Args:
            query_text: The search query.
            top_k: Number of results to retrieve.
            filter_ids: Optional list of memory IDs to restrict the search to.

        Returns:
            A list of dicts with keys: ``commit_id``, ``title``,
            ``timestamp``, ``file_paths``, ``document`` (the indexed text).

        Raises:
            ValueError: if the collection is empty.
        """
        count = self._collection.count()
        if count == 0:
            raise ValueError("The index is empty — no memories have been indexed yet.")

        if filter_ids is not None:
            if not filter_ids:
                return []
            effective_k = min(top_k, count, len(filter_ids))
        else:
            effective_k = min(top_k, count)

        kwargs = {
            "query_texts": [query_text],
            "n_results": effective_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_ids is not None:
            kwargs["where"] = {"memory_id": {"$in": filter_ids}}

        results = self._collection.query(**kwargs)

        hits = []
        if not results["documents"] or not results["documents"][0]:
            return hits

        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            file_paths_str = meta.get("file_paths", "")
            urls_str = meta.get("urls", "")
            hits.append(
                {
                    "memory_id": meta["memory_id"],
                    "title": meta["title"],
                    "timestamp": meta["timestamp"],
                    "machine_id": meta.get("machine_id", ""),
                    "file_paths": [p for p in file_paths_str.split("\n") if p],
                    "urls": [u for u in urls_str.split("\n") if u],
                    "document": doc,
                }
            )
        return hits

