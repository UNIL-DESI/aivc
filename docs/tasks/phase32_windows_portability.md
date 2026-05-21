# Phase 32: Windows Portability and Performance Fixes

## 1. Context & Discussion (Narrative)
Following the migration from WSL/Linux to a native Windows development environment, several critical portability and performance issues were discovered in AIVC. The most severe issue was that AIVC commands would freeze or take an infinite amount of time to execute. This was primarily caused by high CPU and I/O overhead from watchdog's `PollingObserver` trying to traverse deep virtual environments like `.venv`.

Other issues included SQLite file locking (`PermissionError: [WinError 32]`) preventing clean test run completions, test environment failures because of `Path.home()` raising a `RuntimeError` due to cleared environment variables in tests, path normalisation issues between Unix/Windows mocks, and POSIX absolute path assumptions in path migrations.

To ensure AIVC is fully functional, robust, and highly performant natively on Windows, these issues have been resolved, and a complete suite of non-ML tests was verified.

## 2. Files Concerned
- `src/aivc/server.py` — Native file observer implementation.
- `src/aivc/semantic/engine.py` — Pure Python fast search fallback when `grep`/`xargs` is missing.
- `src/aivc/semantic/graph.py` — SQLite database connection life cycle management.
- `src/tests/test_config.py` — Safe environment patching protecting `Path.home()` prerequisites.
- `src/tests/test_server_watcher.py` — Watcher path normalisation.
- `scripts/migrate_commit_paths.py` — Support for POSIX, WSL, and Windows absolute paths.
- `test_perf.py` & `test_perf_v2.py` — Performance test resource cleanup.

## 3. Objectives (Definition of Done)
* **Millisecond-level responses** natively on Windows.
* **No high CPU or IO spikes** from file system observers (watchdog).
* **Zero sqlite file leaks or locks** during test suite execution (fixing all WinError 32 occurrences).
* **100% of non-ML unit/integration tests passing** under native Windows environment.
* **AIVC successfully configured** inside AntiGravity MCP settings for real-time memory logging.
