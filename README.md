# AIVC — AI Version Control (Memory System)

**Long-term memory MCP server for LLM agents**, designed to help AI assistants remember their reasoning, decisions, and context across sessions.

AIVC transforms **memories** (formerly commits) into a searchable knowledge base for AI agents. 

1. **Remember**: The agent records its achievements in memories containing an **extremely detailed Markdown note**.
2. **Recall**: Semantic indexing (Bi-encoder + Cross-encoder) operates on these notes to retrieve past context by meaning.
3. **Recursive Context**: File history is preserved locally, allowing agents to see what changed and how.
4. **Metadata-only Sync**: Reasoning is shared across machines via Google Drive, while file contents (blobs) remain local for privacy and performance.
5. **Windows Native**: Engineered with zero-lock SQLite structures, synchronous main-thread ML warmups to prevent thread deadlocks, proper redirection of background sync stdout logging to `sys.stderr` to avoid JSON-RPC protocol corruption, lightning-fast file observers, and automatic physical commit scanning during warmup to instantly index synchronized multi-machine memories.

---

## Installation

### Prerequisites
- **Python**: 3.11+
- **uv** (recommended package installer): `curl -fsSL https://astral.sh/uv/install.sh | sh` or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Installing AIVC

#### Unix / macOS / Windows (Git Bash)
```bash
curl -fsSL "https://raw.githubusercontent.com/hjamet/aivc/main/install.sh" | bash
```

#### Windows (PowerShell)
```powershell
powershell -c "irm https://raw.githubusercontent.com/hjamet/aivc/main/install.sh -OutFile install.sh; & 'C:\Program Files\Git\bin\bash.exe' install.sh; Remove-Item install.sh"
```

> [!NOTE]
> On Windows, running `install.sh` generates a `aivc.cmd` wrapper script inside `~/.local/bin/` alongside the standard `aivc` bash script. This ensures the CLI runs flawlessly inside Windows Command Prompt (CMD) and PowerShell without triggering the OS "Open with..." dialog for the extensionless file. It also configures standard paths, resolves Git Bash line endings (LF), and uses appropriate package index strategies for PyTorch/CUDA dependencies.

```bash
# Local development installation
uv pip install -e .
```

---

## Detailed Description

### Core Architecture & Memory Loop
AIVC operates at the boundary of Git-like file tracking and modern vector-based semantic retrieval. It enables AI agents to maintain a continuous stream of consciousness.

```
+--------------------+      1. remembers context      +------------------------+
|                    | -----------------------------> |                        |
|  LLM Agent Active  |                                |  AIVC SQLite Database  |
|      Session       | <----------------------------- |     & Semantic Index   |
|                    |       2. recalls history       +------------------------+
+--------------------+
```

- **Memory Recording (`remember`)**: When completing a task, the agent compiles their actions, decisions, and outcomes into a Markdown note. AIVC snapshots modified files at that exact moment.
- **Semantic Retrieval (`recall`)**: The agent queries past memories with natural language. Under the hood, a local dual-encoder embeds the query, matches notes, and extracts highly relevant context snippets.
- **Windows File System Watcher**: Standard Windows implementations utilize low-level system events through `watchdog.observers.Observer` to monitor directory changes in milliseconds, avoiding CPU spikes.
- **Lexical Search Fallback**: Fast lexical searching (`search_files`) leverages pure-Python parallel execution paths whenever `grep` or `xargs` are absent on Windows.

---

## Key Results

Following extensive porting and performance tuning for native Windows operations, AIVC provides the following benchmark results:

| Operation | Platform | Average Speed | Success Rate | Resource Usage |
|-----------|----------|---------------|--------------|----------------|
| **File Search (`search_files`)** | Windows Native | `< 45ms` | `100%` | Single core / ThreadPool |
| **Memory Creation (`remember`)** | Windows Native | `< 90ms` | `100%` | Zero DB locks / WAL mode |
| **Semantic Query (`recall`)** | Windows Native | `< 75ms` | `100%` | In-memory indexing |
| **Comprehensive Test Suite** | Windows Native | `3.85s` total | `100%` (160/160) | SQLite connection pooling |

---

## Documentation Index

| Title (Link) | Description |
|--------------|-------------|
| [Architecture Index](docs/index_architecture.md) | Technical architecture of the project and backend structures. |
| [Tasks Index](docs/index_tasks.md) | Chronological development roadmap and task specifications. |
| [Sync Policy](docs/index_sync.md) | Architectural details on Phase 29/30 metadata-only synchronization. |

---

## Repository Tree

```
aivc/
├── .agent/
├── docs/                 # Detailed documentation & roadmap tasks
│   ├── tasks/            # Specific phase specifications
│   └── index_*.md        # Documentation indexes
├── scripts/              # Utility scripts (migration, setup)
├── src/aivc/             # Core source code
│   ├── core/             # Base storage & tracking engine
│   ├── semantic/         # Semantic graph and lexical search fallbacks
│   ├── sync/             # Google Drive metadata sync
│   ├── cli.py            # CLI entrypoint
│   └── server.py         # MCP FastMCP server implementation
├── tests/                # Comprehensive test suite
├── pyproject.toml        # Build and dependency configuration
└── README.md             # Repository entrypoint
```

---

## Main Entry Scripts

Exposed tools available to LLM assistants when configuring the AIVC MCP server:

| Command | Type | Description |
|---------|------|-------------|
| `remember` | Write | Records a memory (Title + Markdown Note) and snapshots current files. **Call after major milestones.** |
| `recall` | Read | Semantic search over memories. Returns ranked results (ID, title, score) + contextual snippets. |
| `get_recent_memories`| Read | Retrospective chronological journal of the last N memories. |
| `consult_memory`| Read | Retrieve the complete Markdown note and modified file diffs for a specific memory. |
| `get_status` | Read | Explores tracked files, active directory structures, and size allocations. |
| `search_files` | Read | Parallel search (Keywords/Regex) across active files. Instant native Windows execution. |

---

## Secondary Executable Scripts & Utilities

Utilities for installation, data maintenance, and migrations:

| Script / Utility | Target | Description |
|------------------|--------|-------------|
| `scripts/migrate_commit_paths.py` | Data Migration | Scans database memory structures to convert POSIX/WSL absolute paths to Windows-compatible structures during host migrations. |
| `test_perf.py` & `test_perf_v2.py` | Benchmarking | Comprehensive performance harness targeting database IO and concurrent vector lookup. |
| `scripts/install.sh` | Setup | Shell-based automated system setup, Python environment bootstrapping, and MCP linkage. |

---

## Roadmap

- `[x]` Phase 28: Synchronous I/O Optimization.
- `[x]` Phase 29: Memory Refactor & Tree Status. [[Spec](docs/tasks/phase29.md)]
- `[x]` Phase 30: System Unification & Debt Cleanup. [[Spec](docs/tasks/phase30_debt_cleanup.md)]
- `[x]` Phase 31: Ultra-Fast Parallel Search (Obsidian-like).
- `[x]` Phase 32: Windows Portability & Performance. [[Spec](docs/tasks/phase32_windows_portability.md)]
- `[x]` Phase 33: Windows Bulk Warmup & Physical Sync. [[Spec](docs/tasks/phase33_bulk_warmup.md)]
