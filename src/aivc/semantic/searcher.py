"""
Searcher: Bi-Encoder → Cross-Encoder reranking pipeline.

Stage 1 (Bi-Encoder): ChromaDB retrieves the top-K semantically close
memories using the SentenceTransformer embeddings already stored by the Indexer.

Stage 2 (Cross-Encoder): the top-K candidates are re-ranked by the Cross-Encoder
for more precise relevance scoring.  Only top-K candidates are passed to the
Cross-Encoder (default cap: 50) to keep latency acceptable.

The CrossEncoder model is loaded lazily on first ``search()`` call, so that
importing this module or creating a Searcher instance is fast — critical for
CLI-style usage or when only Workspace features are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aivc.semantic.indexer import Indexer

from aivc.config import CROSS_ENCODER_MODEL

_MAX_CROSS_ENCODER_INPUTS = 50  # hard ceiling to keep latency under control


@dataclass
class SearchResult:
    """A single result from the semantic search pipeline."""

    memory_id: str
    """UUID of the matching memory."""

    title: str
    """Short memory title."""

    timestamp: str
    """ISO 8601 UTC creation timestamp."""

    score: float
    """Cross-Encoder relevance score (higher is better)."""

    snippet: str
    """First ~200 characters of the memory note (for preview)."""

    file_paths: list[str]
    """Files that were changed in this memory (excluding deleted files)."""

    machine_id: str = ""
    """ID of the machine where the memory was created."""


class Searcher:
    """Two-stage semantic search: Bi-Encoder (fast recall) → Cross-Encoder (precise rank).

    Args:
        indexer: A fully initialised :class:`Indexer` instance.
    """

    def __init__(self, indexer: "Indexer") -> None:
        self._indexer = indexer
        self.__cross_encoder = None  # lazy-loaded

    @property
    def _cross_encoder(self):
        """Lazy-loaded CrossEncoder model."""
        if self.__cross_encoder is None:
            import sys
            from sentence_transformers import CrossEncoder
            print("\n\033[2m[aivc] Initialising/Downloading CrossEncoder model (this may take a moment)...\033[0m", file=sys.stderr)
            self.__cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self.__cross_encoder

    def search(
        self,
        query: str,
        top_k: int = 50,
        top_n: int = 5,
        filter_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for memories semantically similar to *query*.

        Args:
            query: Free-text search query.
            top_k: Number of candidates retrieved by the Bi-Encoder stage.
                   Capped internally at ``_MAX_CROSS_ENCODER_INPUTS`` (50).
            top_n: Number of results returned after Cross-Encoder reranking.
            filter_ids: Optional list of memory IDs to restrict the search to.

        Returns:
            A list of at most *top_n* :class:`SearchResult` objects sorted
            descending by Cross-Encoder score.

        Raises:
            ValueError: if the index is empty or ``top_n`` > ``top_k``.
        """
        if top_n > top_k:
            raise ValueError(
                f"top_n ({top_n}) must be ≤ top_k ({top_k})."
            )

        effective_k = min(top_k, _MAX_CROSS_ENCODER_INPUTS)

        # ----------------------------------------------------------------
        # Stage 1: Bi-Encoder retrieval via ChromaDB
        # ----------------------------------------------------------------
        candidates = self._indexer.query(query, top_k=effective_k, filter_ids=filter_ids)

        if not candidates:
            return []

        import os
        disable_cross = os.environ.get("AIVC_DISABLE_CROSS_ENCODER", "False").lower() == "true"

        import re
        word_pattern = re.compile(r'\w+')
        query_words = set(word_pattern.findall(query.lower()))

        def extract_snippet(note_part: str) -> str:
            snippet = note_part[:200]
            if note_part and query_words:
                note_part_lower = note_part.lower()

                # Find all words that match the query and their positions in the text
                words_in_note = [
                    (m.group(), m.start(), m.end())
                    for m in word_pattern.finditer(note_part_lower)
                    if m.group() in query_words
                ]

                # Fast path: if no query words in the text, return early
                if not words_in_note:
                    return snippet

                best_score = -1
                best_idx = 0
                max_len = 200
                step = 50

                # Determine maximum index dynamically based on text length
                max_idx = max(1, len(note_part) - max_len + step)

                for idx in range(0, max_idx, step):
                    end_idx = idx + max_len
                    window_words = set()

                    # Check overlapping words
                    for word, start, end in words_in_note:
                        if end > idx and start < end_idx:
                            window_words.add(word)
                        elif start >= end_idx:
                            break # Optimization since word_matches is sorted by start idx

                    window_score = len(window_words)
                    if window_score > best_score:
                        best_score = window_score
                        best_idx = idx

                if best_score > 0:
                    start = best_idx
                    end = start + max_len
                    snippet = note_part[start:end].strip()
                    if start > 0:
                        snippet = "…" + snippet
                    if end < len(note_part):
                        snippet = snippet + "…"
            return snippet

        if disable_cross:
            # Bypass Stage 2 (Cross-Encoder reranking) and use Bi-Encoder results directly
            results = []
            # ChromaDB candidates are already sorted by distance (cosine similarity)
            for i, hit in enumerate(candidates[:top_n]):
                doc: str = hit["document"]
                lines = doc.split("\n", 2)
                note_part = lines[2] if len(lines) >= 3 else doc
                note_part = note_part.strip()
                
                snippet = extract_snippet(note_part)

                score = 1.0 - (i * 0.05)  # Assign a mock descending score
                results.append(
                    SearchResult(
                        memory_id=hit["memory_id"],
                        title=hit["title"],
                        timestamp=hit["timestamp"],
                        score=score,
                        snippet=snippet,
                        file_paths=hit["file_paths"],
                        machine_id=hit.get("machine_id", ""),
                    )
                )
            return results

        # ----------------------------------------------------------------
        # Stage 2: Cross-Encoder reranking
        # ----------------------------------------------------------------
        # Build (query, document) pairs for the cross-encoder.
        pairs = [(query, c["document"]) for c in candidates]
        scores: list[float] = self._cross_encoder.predict(pairs).tolist()

        # Sort candidates by descending cross-encoder score.
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, hit in ranked[:top_n]:
            doc: str = hit["document"]
            lines = doc.split("\n", 2)
            note_part = lines[2] if len(lines) >= 3 else doc
            note_part = note_part.strip()
            
            snippet = extract_snippet(note_part)

            results.append(
                SearchResult(
                    memory_id=hit["memory_id"],
                    title=hit["title"],
                    timestamp=hit["timestamp"],
                    score=score,
                    snippet=snippet,
                    file_paths=hit["file_paths"],
                    machine_id=hit.get("machine_id", ""),
                )
            )

        return results

