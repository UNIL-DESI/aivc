# Tasks Index (Roadmap)

| Note Title | Short Description | Last modified | Tag |
|------------------|-------------------|----------------|-----|
| [Phase 1: Versioning Engine](tasks/phase1_versioning_engine.md) | SHA-256 Blobs, Commits, Diff, Workspace | 2026-03-18 | `Up to date` |
| [Phase 2: Semantic Engine](tasks/phase2_semantic_graph.md) | ChromaDB, Commit indexing, Aggregation, Graph | 2026-03-18 | `Up to date` |
| [Phase 3: MCP Interface](tasks/phase3_mcp_interface.md) | MCP Server, Exposed Tools, Prompts | 2026-03-18 | `Up to date` |
| [Phase 4: CLI & Web Dashboard](tasks/phase4_cli_and_dashboard.md) | Human CLI Tools, interactive UI Graph, Visual semantic search | 2026-03-18 | `Up to date` |
| [Phase 5: MVP Stabilization](tasks/phase5_stabilization.md) | Absolute paths, port autodiscovery, Cytoscape vendoring | 2026-03-18 | `Up to date` |
| [Phase 6: Absolute Consolidation & CLI](tasks/phase6_absolute_paths_fix.md) | Absolute history migration, `aivc track` addition | 2026-03-18 | `Up to date` |
| [Phase 7: Scoped Semantic Search](tasks/phase7_scoped_search.md) | Glob filtering in `search_memory` | 2026-03-18 | `Up to date` |
| [Phase 8: GEMINI.md Injection](tasks/phase8_gemini_injection.md) | Agent best practices injected via `install.sh` | 2026-03-18 | `Up to date` |
| [Phase 9: CLI Exposure](tasks/phase9_cli_exposure.md) | `~/.local/bin/aivc` Symlink | 2026-03-18 | `Up to date` |
| [Phase 10: Search Result Snippets](tasks/phase10_search_snippets.md) | Contextual snippets in `search_memory` | 2026-03-19 | `Up to date` |
| [Phase 11: Track MCP Tool](tasks/phase11_track_mcp_tool.md) | `track` tool exposure in the MCP server | 2026-03-19 | `Up to date` |
| [Phase 12: Consulted Files](tasks/phase12_consulted_files.md) | `consulted` action in commits, graph enrichment | 2026-03-19 | `Up to date` |
| [Phase 13: BM25 File Search](tasks/phase13_bm25_file_search.md) | BM25 lexical search on tracked files content | 2026-03-19 | `Up to date` |
| [Phase 14: Commit Context](tasks/phase14_commit_context.md) | Prev/Next Title and ID in `consult_commit` return | 2026-03-19 | `Up to date` |
| [Phase 15: I/O Performance Refactoring (CoreIndex)](tasks/phase15_io_refactoring.md) | Standalone SQLite CoreIndex in `core/`, `_all_commits()` elimination | 2026-03-19 | `Up to date` |
| [Phase 16: BM25 Reliability & CLI UX](tasks/phase16_fiabilisation_bm25_ux_cli.md) | SQLite BM25 cache, CLI Storage Fallback | 2026-03-19 | `Up to date` |
| [Phase 17: Watched Directories (JIT Watcher)](tasks/phase17_watched_directories.md) | JIT auto-discovery and auto-tracking on observed directories | 2026-03-19 | `Up to date` |
| [Phase 18: Internationalization](tasks/internationalization_and_english_docs.md) | Full translation of README, technical documentation | 2026-03-24 | `Up to date` |
| [Phase 19: Web Dashboard UX](tasks/phase19_web_dashboard_ux.md) | Sidebar Git Log, File history, cose layout fix | 2026-03-24 | `Up to date` |
| [Phase 20: Cloud Sync](tasks/phase20_google_drive_sync.md) | Google Drive Sync architecture for multi-machine commits | 2026-03-24 | `Up to date` |
| [Phase 21: Async & Sync Consolidation](tasks/phase21_async_sync_consolidation.md) | Async and cloud sync architectural debt consolidation | 2026-03-24 | `Up to date` |
| [Phase 22: Local Hints Optimization](tasks/phase22_local_hints_optimization.md) | O(1) basename index for local path hints | 2026-03-24 | `Up to date` |
| [Phase 23 — State Sync](tasks/phase23_state_sync_reload.md) | Reload JIT du workspace via mtime | 2026-03-24 | `Up to date` |
| [Phase 25 — Native Google Drive](tasks/phase25_native_google_drive.md) | Native Google Drive API replacing rclone | 2026-03-24 | `Up to date` |
| [Phase 26 — Auto Sync Push](tasks/phase26_auto_sync_push.md) | Automatic push of missing commits at startup | 2026-03-24 | `Up to date` |
| [Phase 27 — Remote Hints](tasks/phase27_remote_hints.md) | Multi-Machine Distinction & Path Mapping | 2026-03-24 | `Up to date` |
| [Phase 28 — I/O Optimization](tasks/phase28_io_optimizations.md) | Batching & Fast Diff for local indexing | 2026-03-24 | `Up to date` |
| [Phase 29 — Memory Refactor](tasks/phase29.md) | Remember tool, Blob sync removal, Tree Status | 2026-03-30 | `Up to date` |
| [Phase 30 — Debt Cleanup](tasks/phase30_debt_cleanup.md) | Unification du backend (Commit -> Memory) et purge drive.py | 2026-03-30 | `Up to date` |
| [Phase 32 — Windows Portability & Perf](tasks/phase32_windows_portability.md) | Native Windows fixes, file lock resolutions, clean tests | 2026-05-21 | `Up to date` |
| [Phase 33 — Windows Bulk Warmup & Physical Sync](tasks/phase33_bulk_warmup.md) | Physical commits scanning and O(1) bulk ID check in warmup | 2026-05-21 | `Up to date` |
| [Phase 34 — Windows Threading Deadlock & Cross-Encoder Bypass](tasks/phase34_windows_deadlocks.md) | Prevent Windows multi-thread deadlocks and add CrossEncoder bypass | 2026-05-22 | `Up to date` |

