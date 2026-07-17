# Architecture Index

| Note Title | Short Description | Last modified | Tag |
|------------------|-------------------|----------------|-----|
| [Phase 1 — Core Versioning Engine](../src/aivc/core/blob_store.py) | SHA-256 BlobStore + Refcount/GC | 2026-03-18 | `Up to date` |
| [Phase 1 — Commit Structure](../src/aivc/core/commit.py) | Commit & FileChange Dataclasses with size impact | 2026-03-18 | `Up to date` |
| [Phase 1 — Diff Engine](../src/aivc/core/diff.py) | added/modified/deleted detection | 2026-03-18 | `Up to date` |
| [Phase 1 — Workspace Orchestrator](../src/aivc/core/workspace.py) | Track glob/dir, GC, status, log | 2026-03-18 | `Up to date` |
| [Phase 2 — ChromaDB Indexer](../src/aivc/semantic/indexer.py) | Commit notes vectorization + ChromaDB upsert | 2026-03-18 | `Up to date` |
| [Phase 2 — Bi/Cross-Encoder Searcher](../src/aivc/semantic/searcher.py) | Bi-Encoder → Cross-Encoder Pipeline, SearchResult | 2026-03-18 | `Up to date` |
| [Phase 2 — CooccurrenceGraph](../src/aivc/semantic/graph.py) | Bipartite files↔commits graph, visualization export | 2026-03-18 | `Up to date` |
| [Phase 2 — SemanticEngine](../src/aivc/semantic/engine.py) | Facade orchestrating Workspace + Indexer + Graph + Searcher | 2026-03-18 | `Up to date` |
| [Phase 17 — Watchdog Daemon](../src/aivc/server.py) | Real-time surveillance + Startup Sync | 2026-07-17 | `Legacy` |
| [Phase 17 — Watched Dirs Management](../src/aivc/core/workspace.py) | `watched_dirs` state management and hidden file filtering | 2026-03-19 | `Up to date` |
