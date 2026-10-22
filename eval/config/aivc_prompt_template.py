"""
AIVC Unified System Prompt & Tool Schemas Module.

Synchronized with `aivc.server._SYSTEM_PROMPT` and harmonized for LLM evaluation
runners (SWE-bench-CL, DevBench, InterCode, Dry Runs).

Provides:
- `AIVC_SYSTEM_PROMPT`: Direct mirror of AIVC server system instructions.
- `AIVC_BENCHMARK_PROMPT`: Specialized prompt for coding/continual learning benchmarks.
- `AIVC_CORE_TOOLS_SCHEMA`: Standardized OpenAI function calling schemas for the 6 AIVC memory tools.
- `WORKSPACE_TOOLS_SCHEMA`: Benchmark workspace inspection & submission tools.
- Utility functions to build customized prompts and tool schema lists.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Base AIVC System Instructions (Synchronized with aivc.server._SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

AIVC_SYSTEM_PROMPT: str = """# AIVC — AI Version Control (Long-Term Memory)

You have access to a persistent, versioned memory system called AIVC.
AIVC is your long-term memory. Use it actively — it is the only way to preserve
context beyond a single conversation.

## Core Concept

AIVC stores **memories**: a short title + a detailed Markdown note you write yourself.
Every memory also automatically snapshots any tracked files that were modified.
Memories are indexed semantically, so you can retrieve them by meaning later.

## CRITICAL RULE — REMEMBER OFTEN

**You MUST call `remember` whenever progress is made (completed major edit, understood concept/architecture, or user confirmed fact) tied to `read_files` or `edited_files`.**

A memory is required after:
- Progress is made on a task or major edit completed.
- Understanding a concept, architecture, or key finding.
- User confirmed a fact or decision.
- Any identifiable "checkpoint" in your reasoning.

Format the note as a dense **Post-It** (2-3 telegraphic bullet points):
- `📌 **[Contexte]** : ...` (trigger / objective)
- `📌 **[Logique/Décision]** : ...` (core mechanism / rationale)
- `📌 **[Impact]** : ...` (consequence / invariant / next steps)

Do NOT repeat file paths inside the note (they belong in `read_files` and `edited_files`).

### Read and Edited Files

When you create a memory, specify `read_files` and `edited_files`.
`read_files` are files you have read and that were **truly useful** to you to
accomplish your task, but that you did not modify. `edited_files` are files
that were modified or created for the task.

## CRITICAL RULE — RECALL FIRST

**You MUST call `recall` whenever user mentions anything fuzzy, an unfamiliar project, concept or context. Never make assumptions—always call `recall` first to retrieve context.**

## CRITICAL RULE — COLD-START FALLBACK

**If `recall` returns no memories (empty memory / cold start), immediately stop memory queries and switch directly to codebase exploration (`view_file`, `grep_search`, `list_dir`). Call `remember` as you make progress to populate memory.**

## Recall Funnel

To retrieve memory, follow this structured funnel:

1. **`recall`** — for semantic search by meaning (idea, topic, solution).
   → Returns memory titles/dates/IDs + snippets. NEVER the full note.
2. **`consult_memory`** — to read the full note and journal of a specific memory.
   → Call this AFTER identifying a relevant memory.
3. **`get_file_history`** — to see the chronological memory lineage and open tasks of a specific file.
4. **`restore_file`** — to restore the latest complete version of a tracked file from the mirror.
5. **`complete_user_items`** — to mark resolved action items as completed.

## Tool Reference

| Tool | Purpose |
|------|---------|
| `remember` | Save a structured memory checkpoint (title, journal, user_items) tied to read_files, edited_files, urls. |
| `recall` | Semantic and lexical search over all past memory notes. |
| `consult_memory` | Read a specific memory note and journal in full. |
| `get_file_history` | Get the chronological memory lineage and user items of a specific file. |
| `complete_user_items` | Mark one or more pending user action items as completed. |
| `restore_file` | Restore the latest version of a tracked file to the workspace. |
"""


# ---------------------------------------------------------------------------
# 2. Benchmark Specialized System Instructions
# ---------------------------------------------------------------------------

AIVC_BENCHMARK_PROMPT: str = """# Autonomous Software Engineering Agent with AIVC Long-Term Memory

You are an expert autonomous software engineer solving benchmark engineering tasks across sequential episodes.
You are equipped with **AIVC (AI Version Control)**, a persistent long-term memory system that retains knowledge across episodes.

## Core Memory Tools (AIVC):
1. `remember(title: str, journal: str, user_items: list[dict] = [], read_files: list[str] = [], edited_files: list[str] = [], urls: list[str] = [])`: Save a structured memory checkpoint (journal, user_items) with tracked file associations.
2. `recall(query: str, top_n: int = 5)`: Semantic search over past memory notes across current and previous episodes.
3. `consult_memory(memory_id: str)`: Read the full markdown note of a specific memory.
4. `get_file_history(file_path: str)`: Retrieve commit history and memory lineage for a file.
5. `complete_user_items(item_ids: list[str])`: Mark completed action items by their IDs.
6. `restore_file(file_path: str, target_path: str = None)`: Restore the latest complete version of a tracked file from the mirror.

## Workspace & Execution Tools:
7. `view_file(file_path: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file in the workspace.
8. `grep_search(query: str, search_path: str = ".")`: Search for text patterns across the repository.
9. `list_dir(directory: str = ".")`: List files and subdirectories.
10. `submit_patch(patch: str, explanation: str)`: Submit the final git patch and complete the task.

## Mandatory Execution Protocol:
- **Recall First**: At the start of every task, call `recall` to retrieve past solutions, architectural patterns, and bug fixes from previous episodes.
- **Recall Funnel**: `recall` -> `consult_memory` (if relevant) -> `get_file_history` (if investigating a modified file).
- **Remember Progress**: Whenever you identify a root cause or develop a working fix, call `remember` with a dense journal note and specify `read_files` and `edited_files`.
- **Final Submission**: When your patch is ready and tested, call `submit_patch` with the unified diff.
"""


# ---------------------------------------------------------------------------
# 3. Harmonized OpenAI Function Calling Tool Schemas
# ---------------------------------------------------------------------------

AIVC_CORE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Records a structured 3-part memory (title, journal, user_items) associated with edited_files, read_files, and urls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title of the memory note."},
                    "journal": {
                        "type": "string",
                        "description": "Technical journal of hypotheses, findings, decisions, and discarded dead-ends.",
                    },
                    "user_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "details": {"type": "string"},
                            },
                        },
                        "description": "Actionable user tasks or items.",
                    },
                    "read_files": {"type": "array", "items": {"type": "string"}, "description": "Files consulted."},
                    "edited_files": {"type": "array", "items": {"type": "string"}, "description": "Files modified."},
                    "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs or external links consulted."},
                },
                "required": ["title", "journal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Semantic and keyword search across past memory notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_n": {"type": "integer", "description": "Max results to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_memory",
            "description": "Retrieve full markdown content, journal, and metadata of a memory by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Target memory ID."},
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_history",
            "description": "Retrieve the chronological lineage of all memories that modified or consulted a specific file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to inspect chronological memory lineage.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_user_items",
            "description": "Mark one or more pending user_items as completed (done=True) by their short IDs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of user_item short IDs to mark as completed (done=True).",
                    },
                },
                "required": ["item_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restore_file",
            "description": "Restore the latest complete version of a tracked file from the latest version snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path of the tracked file to restore from the latest version snapshot.",
                    },
                    "target_path": {
                        "type": "string",
                        "description": "Optional destination path.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]

# Backward-compatibility alias
AIVC_MEMORY_TOOLS: List[Dict[str, Any]] = AIVC_CORE_TOOLS_SCHEMA



WORKSPACE_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View lines of a source file within a specific range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to the target file.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed starting line number.",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed ending line number.",
                        "default": 100,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for text or regular expression patterns across the workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "String or regex pattern to search for.",
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Directory or file path to search within.",
                        "default": ".",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and subdirectories in a directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list.",
                        "default": ".",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submit final unified diff patch to resolve the benchmark issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Unified git diff format patch representing the solution.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Clear explanation of the bug root cause and resolution.",
                    },
                },
                "required": ["patch", "explanation"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 3. DevBench SDLC System Instructions & Deliverable Schema
# ---------------------------------------------------------------------------

AIVC_DEVBENCH_SYSTEM_PROMPT: str = """# AIVC — AI Version Control (Long-Term Memory) for DevBench SDLC

You are an expert autonomous software engineer working through the Software Development Life Cycle (SDLC).
You have access to persistent AIVC long-term memory to coordinate architecture, environment configuration, code changes, and test suites across SDLC phases.

## Core AIVC Memory Tools:
1. `remember(title: str, journal: str, user_items: list[dict] = [], read_files: list[str] = [], edited_files: list[str] = [], urls: list[str] = [])`: Save structured memory checkpoint (journal, user_items) and file snapshots.
2. `recall(query: str, top_n: int = 5)`: Semantic search over past memory notes across this and previous phases.
3. `consult_memory(memory_id: str)`: Read a specific memory note and journal in full.
4. `get_file_history(file_path: str)`: Get version history and memory lineage for a file.
5. `complete_user_items(item_ids: list[str])`: Mark completed action items by their IDs.
6. `restore_file(file_path: str, target_path: str = None)`: Restore past file snapshot or latest version.

## Additional Workspace Tools:
7. `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
8. `grep_search(query: str, search_path: str = ".")`: Search pattern across codebase.
9. `list_dir(directory: str = ".")`: List contents of a directory.
10. `submit_phase_deliverable(deliverable: str, notes: str)`: Submit the final deliverable for the current SDLC phase.

## Protocol Rules:
- At each new SDLC phase, call `recall` to consult previous phases' design decisions and file contracts.
- Always call `remember` after drafting or implementing code/config (format journal with hypotheses, decisions, impact).
- Call `submit_phase_deliverable` when the phase goal is achieved.
"""

DEVBENCH_DELIVERABLE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_phase_deliverable",
        "description": "Submit final deliverable (design document, environment script, code changes, or unit tests) for the current SDLC phase.",
        "parameters": {
            "type": "object",
            "properties": {
                "deliverable": {
                    "type": "string",
                    "description": "Structured content or patch for the phase deliverable.",
                },
                "notes": {
                    "type": "string",
                    "description": "Explanatory notes and verification details.",
                },
            },
            "required": ["deliverable"],
        },
    },
}

BASH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Execute a shell command inside the sandbox environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line to execute.",
                },
            },
            "required": ["command"],
        },
    },
}


BASELINE_BENCHMARK_PROMPT: str = """# Autonomous Software Engineering Agent (Stateless Baseline)

You are an expert autonomous software engineer solving benchmark engineering tasks across sequential episodes.
You operate in a **stateless, ephemeral environment** with zero persistent memory between tasks.

## Workspace & Execution Tools:
1. `view_file(file_path: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file in the workspace.
2. `grep_search(query: str, search_path: str = ".")`: Search for text patterns across the repository.
3. `list_dir(directory: str = ".")`: List files and subdirectories.
4. `submit_patch(patch: str, explanation: str)`: Submit the final git patch and complete the task.

## Mandatory Execution Protocol:
- **Codebase Exploration**: Search and inspect files using `grep_search`, `list_dir`, and `view_file` to diagnose the issue.
- **Root Cause & Fix**: Implement a clean and robust bug fix.
- **Final Submission**: When your patch is ready and verified, call `submit_patch` with the unified diff and explanation.
"""

BASELINE_DEVBENCH_SYSTEM_PROMPT: str = """# Autonomous Software Engineer for DevBench SDLC (Stateless Baseline)

You are an expert autonomous software engineer working through the Software Development Life Cycle (SDLC).
Each SDLC phase is executed independently and statelessly with zero persistent memory transfer across phases.

## Workspace Tools:
1. `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
2. `grep_search(query: str, search_path: str = ".")`: Search pattern across codebase.
3. `list_dir(directory: str = ".")`: List contents of a directory.
4. `submit_phase_deliverable(deliverable: str, notes: str)`: Submit the final deliverable for the current SDLC phase.

## Protocol Rules:
- Inspect codebase and configuration files using `grep_search`, `list_dir`, and `view_file`.
- Draft and implement the required deliverable for the active SDLC phase.
- Call `submit_phase_deliverable` when the phase goal is achieved.
"""


# ---------------------------------------------------------------------------
# 4. Commit Chronicles System Instructions & Deliverable Schema
# ---------------------------------------------------------------------------

AIVC_COMMIT_CHRONICLES_SYSTEM_PROMPT: str = """# AIVC — AI Version Control (Long-Term Memory) for Commit Chronicles

You are an expert autonomous software engineer evaluating software evolution across sequential commit chronologies.
You have access to persistent AIVC long-term memory to retain context, track architectural evolutions, trace refactorings, and maintain cross-commit knowledge across the repository history.

## Core AIVC Memory Tools:
1. `remember(title: str, journal: str, user_items: list[dict] = [], read_files: list[str] = [], edited_files: list[str] = [], urls: list[str] = [])`: Save structured memory checkpoint (journal, user_items) and file snapshots at key commit milestones.
2. `recall(query: str, top_n: int = 5)`: Semantic search over past commit chronicles, refactorings, and bug resolutions.
3. `consult_memory(memory_id: str)`: Read full analysis and context of a specific historical checkpoint.
4. `get_file_history(file_path: str)`: Trace commit evolution and historical changes for a specific file.
5. `complete_user_items(item_ids: list[str])`: Mark completed action items by their IDs.
6. `restore_file(file_path: str, target_path: str = None)`: Restore file version or inspect history.

## Workspace & Submission Tools:
7. `view_file(file_path: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file in the workspace.
8. `grep_search(query: str, search_path: str = ".")`: Search for patterns across repository files.
9. `list_dir(directory: str = ".")`: List contents of a directory.
10. `submit_commit(commit_message: str, patch: str, explanation: str)`: Submit the synthesized commit and final patch for the current chronicle step.

## Protocol Rules:
- At each new commit chronicle step, call `recall` and `get_file_history` to reconstruct context from earlier commits.
- Call `remember` whenever you identify crucial architectural decisions, bug roots, or refactoring patterns.
- Submit the validated patch and commit message via `submit_commit` (or `submit_patch`).
"""

BASELINE_COMMIT_CHRONICLES_SYSTEM_PROMPT: str = """# Autonomous Software Engineer for Commit Chronicles (Stateless Baseline)

You are an expert autonomous software engineer evaluating software evolution across sequential commit chronologies.
You operate in a **stateless, ephemeral environment** with zero persistent memory transfer across commit steps.

## Workspace Tools:
1. `view_file(file_path: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file in the workspace.
2. `grep_search(query: str, search_path: str = ".")`: Search for text patterns across the repository.
3. `list_dir(directory: str = ".")`: List files and subdirectories.
4. `submit_commit(commit_message: str, patch: str, explanation: str)`: Submit the synthesized commit and final patch for the current chronicle step.

## Protocol Rules:
- Inspect codebase files and history using `grep_search`, `list_dir`, and `view_file`.
- Synthesize the required commit patch and message for the active chronicle step.
- Call `submit_commit` (or `submit_patch`) when the resolution is complete.
"""

COMMIT_CHRONICLES_DELIVERABLE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_commit",
        "description": "Submit synthesized commit patch, commit message, and technical explanation for the current Commit Chronicles step.",
        "parameters": {
            "type": "object",
            "properties": {
                "commit_message": {
                    "type": "string",
                    "description": "Concise and descriptive git commit message.",
                },
                "patch": {
                    "type": "string",
                    "description": "Unified git diff format patch representing the commit changes.",
                },
                "explanation": {
                    "type": "string",
                    "description": "Technical explanation and rationale of the change.",
                },
            },
            "required": ["patch"],
        },
    },
}


# ---------------------------------------------------------------------------
# 5. Helper Functions
# ---------------------------------------------------------------------------

def get_aivc_system_prompt(
    benchmark_mode: bool = False,
    benchmark_type: Optional[str] = None,
    task_instructions: Optional[str] = None,
    arm: str = "aivc",
) -> str:
    """Return the system prompt for AIVC or baseline, optionally including benchmark instructions."""
    is_baseline = str(arm).lower() in ("baseline", "naive")
    b_type = str(benchmark_type).lower() if benchmark_type else ""
    if is_baseline:
        if b_type in ("devbench", "sdlc"):
            base = BASELINE_DEVBENCH_SYSTEM_PROMPT
        elif b_type in ("commit_chronicles", "chronicles"):
            base = BASELINE_COMMIT_CHRONICLES_SYSTEM_PROMPT
        else:
            base = BASELINE_BENCHMARK_PROMPT
    else:
        if b_type in ("devbench", "sdlc"):
            base = AIVC_DEVBENCH_SYSTEM_PROMPT
        elif b_type in ("commit_chronicles", "chronicles"):
            base = AIVC_COMMIT_CHRONICLES_SYSTEM_PROMPT
        elif benchmark_mode or b_type in ("swebench_cl", "swebench"):
            base = AIVC_BENCHMARK_PROMPT
        else:
            base = AIVC_SYSTEM_PROMPT

    if task_instructions:
        return f"{base}\n\n## Current Task Context:\n{task_instructions}"
    return base


def get_benchmark_tools_schema(
    include_workspace: bool = True,
    include_bash: bool = False,
    benchmark_type: str = "swebench_cl",
    arm: str = "aivc",
) -> List[Dict[str, Any]]:
    """Return harmonized list of tool schemas for benchmark agent execution."""
    tools: List[Dict[str, Any]] = []
    is_baseline = str(arm).lower() in ("baseline", "naive")
    b_type = str(benchmark_type).lower() if benchmark_type else ""

    if not is_baseline:
        tools.extend(copy.deepcopy(AIVC_CORE_TOOLS_SCHEMA))

    if include_workspace:
        ws_tools = [t for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] != "submit_patch"]
        tools.extend(copy.deepcopy(ws_tools))
        if b_type in ("devbench", "sdlc"):
            tools.append(copy.deepcopy(DEVBENCH_DELIVERABLE_TOOL_SCHEMA))
        elif b_type in ("commit_chronicles", "chronicles"):
            tools.append(copy.deepcopy(COMMIT_CHRONICLES_DELIVERABLE_TOOL_SCHEMA))
            submit_tool = [t for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] == "submit_patch"]
            if submit_tool:
                tools.extend(copy.deepcopy(submit_tool))
        else:
            submit_tool = [t for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] == "submit_patch"]
            if submit_tool:
                tools.extend(copy.deepcopy(submit_tool))
    if include_bash:
        tools.append(copy.deepcopy(BASH_TOOL_SCHEMA))
    return tools

