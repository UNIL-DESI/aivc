# Phase 33: Windows Bulk Warmup & Physical Sync

## 1. Context & Discussion (Narrative)
When setting up AIVC on a brand-new Windows machine, the local `workspace.json` starts with `head_commit_id = None`. Consequently, the linear commit lineage is empty on the new machine. During initial setup, the Google Drive synchronizer pulls remote memories from other devices into the physical `commits/` directory, but the semantic indexing engine (`SemanticEngine.warmup()`) failed to index them because it relied on `self._workspace.get_log()`, which traverses from the empty local `HEAD`.

Furthermore, performing individual exists checks in ChromaDB (`indexer.is_indexed()`) for hundreds of commits at startup was causing slow startup times and thread gridlocks under the global `_ml_lock`.

This phase introduces physical commit directory scanning during warmup, combined with O(1) bulk ID check validation, to ensure newly synced memories are instantly and robustly indexed semantically on new machines.

## 2. Affected Files
- `src/aivc/semantic/engine.py` (Modified)
- `README.md` (Modified)
- `docs/index_tasks.md` (Modified)

## 3. Objectives (Definition of Done)
* **Robust Physical Scanning**: Warmup must physically scan the `commits/` directory for `*.json` files instead of relying on `get_log()`.
* **Graceful Corrupted Files Handling**: Corrupted or empty commit JSON files must be bypassed with clear stderr warning logs without crashing the indexing sequence.
* **O(1) Bulk Verification**: Rather than calling `is_indexed()` in a loop, the engine must fetch all indexed IDs in a single bulk query to keep startup overhead under `< 1s` (excluding native torch import time).
* **Flawless Windows Execution**: Indexing of hundreds of pulled memories must succeed and be fully queryable via local semantic search command line interface and MCP server.
