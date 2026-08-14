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

The memory note must be **detailed**. Do not write one-liners.
Document your reasoning, the decisions made, the problems encountered,
and the solutions found. Think of it as a handover memo to your future self.

### Read and Edited Files

When you create a memory, specify `read_files` and `edited_files`.
`read_files` are files you have read and that were **truly useful** to you to
accomplish your task, but that you did not modify. `edited_files` are files
that were modified or created for the task.

## CRITICAL RULE — RECALL FIRST

**You MUST call `recall` whenever user mentions anything fuzzy, an unfamiliar project, concept or context. Never make assumptions—always call `recall` first to retrieve context.**

## Recall Funnel

To retrieve memory, follow this structured funnel:

1. **`recall`** — for semantic search by meaning (idea, topic, solution).
   → Returns memory titles/dates/IDs + snippets. NEVER the full note.
2. **`get_recent_memories`** — for recalling recent history chronologically.
3. **`consult_memory`** — to read the full note of a specific memory.
   → Call this AFTER identifying a relevant memory.
4. **`get_file_history_metadata`** — to see the chronological commit/memory history of a specific file.
5. **`read_past_file_content`** — to inspect the actual historical content or diff of a file at a past memory.

## Remote Memories & Sync Policy

AIVC synchronizes ONLY memory metadata (titles, notes) between machines. 
File contents (blobs) are recorded locally. If a memory was created on another machine,
historical file content snapshots may not be locally available for `read_past_file_content`.

## Tool Reference

| Tool | Purpose |
|------|---------|
| `remember` | Save a memory checkpoint. Must be called whenever progress is made tied to read_files or edited_files. |
| `recall` | Semantic search over all past memory notes. Must be called whenever user mentions anything fuzzy or unfamiliar. |
| `get_recent_memories` | Recent memory log (paginable). |
| `consult_memory` | Read a specific memory note in full. |
| `get_file_history_metadata` | Get the AIVC history of a specific file. |
| `read_past_file_content` | Read the content of a file as it was at a specific past memory. |
"""


# ---------------------------------------------------------------------------
# 2. Benchmark Specialized System Instructions
# ---------------------------------------------------------------------------

AIVC_BENCHMARK_PROMPT: str = """# Autonomous Software Engineering Agent with AIVC Long-Term Memory

You are an expert autonomous software engineer solving benchmark engineering tasks across sequential episodes.
You are equipped with **AIVC (AI Version Control)**, a persistent long-term memory system that retains knowledge across episodes.

## Core Memory Tools (AIVC):
1. `remember(title: str, note: str, read_files: list[str] = [], edited_files: list[str] = [])`: Save a detailed memory checkpoint with tracked file associations.
2. `recall(query: str, top_n: int = 5)`: Semantic search over past memory notes across current and previous episodes.
3. `get_recent_memories(limit: int = 10, offset: int = 0)`: Inspect recent memory history chronologically.
4. `consult_memory(memory_id: str)`: Read the full markdown note of a specific memory.
5. `get_file_history_metadata(file_path: str)`: Retrieve commit history for a file.
6. `read_past_file_content(file_path: str, memory_id: str, diff_against: str = "current")`: Retrieve past file version or diff.

## Workspace & Execution Tools:
7. `view_file(file_path: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file in the workspace.
8. `grep_search(query: str, search_path: str = ".")`: Search for text patterns across the repository.
9. `list_dir(directory: str = ".")`: List files and subdirectories.
10. `submit_patch(patch: str, explanation: str)`: Submit the final git patch and complete the task.

## Mandatory Execution Protocol:
- **Recall First**: At the start of every task, call `recall` to retrieve past solutions, architectural patterns, and bug fixes from previous episodes.
- **Recall Funnel**: `recall` -> `consult_memory` (if relevant) -> `get_file_history_metadata` (if investigating a modified file).
- **Remember Progress**: Whenever you identify a root cause or develop a working fix, call `remember` with a detailed technical note and specify `read_files` and `edited_files`.
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
            "description": (
                "Save a detailed memory checkpoint. Must be called whenever progress is made "
                "(completed edit, understood concept/architecture, or found a solution) tied to "
                "read_files or edited_files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, descriptive title of the memory checkpoint.",
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "Detailed Markdown note documenting reasoning, findings, problems, "
                            "and solutions. Think of it as a handover memo to your future self."
                        ),
                    },
                    "read_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files read that provided essential context.",
                        "default": [],
                    },
                    "edited_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files created or modified for this task.",
                        "default": [],
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of URLs or web links consulted.",
                        "default": [],
                    },
                },
                "required": ["title", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Recall past memories by semantic meaning. Must be called whenever user mentions "
                "anything fuzzy, an unfamiliar project, concept or context. Uses semantic search "
                "and returns memory titles, IDs, dates, and snippets (never full notes)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text semantic search query describing the problem, concept, or topic.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20).",
                        "default": 5,
                    },
                    "filter_glob": {
                        "type": "string",
                        "description": "Optional glob pattern (e.g. 'src/*.py') to restrict search to touched files.",
                        "default": "",
                    },
                    "only_local": {
                        "type": "boolean",
                        "description": "If True, only search memories created on this machine.",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_memories",
            "description": (
                "Display recent memory history in reverse chronological order (newest first). "
                "Useful at the start of a session or to inspect chronological activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of memories to return (default 10, max 50).",
                        "default": 10,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of memories to skip from the most recent (default 0).",
                        "default": 0,
                    },
                    "only_local": {
                        "type": "boolean",
                        "description": "If True, only show memories created on this machine.",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_memory",
            "description": (
                "Read the complete content of a specific memory note. Call this after identifying "
                "a relevant memory ID via `recall` or `get_recent_memories`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the memory note to read.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_history_metadata",
            "description": (
                "Retrieve the chronological list of all memories that modified or consulted a specific file. "
                "Useful to understand when a file was changed and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative or absolute path of the file to inspect.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_past_file_content",
            "description": (
                "Retrieve the actual text content or unified diff of a file at a specific past memory snapshot. "
                "Requires both the file path and the memory ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read.",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the past memory snapshot.",
                    },
                    "diff_against": {
                        "type": "string",
                        "enum": ["current", "parent", "none"],
                        "description": (
                            "Comparison mode: 'current' (diff vs local disk), "
                            "'parent' (diff vs parent memory), 'none' (raw content)."
                        ),
                        "default": "current",
                    },
                },
                "required": ["file_path", "memory_id"],
            },
        },
    },
]


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
1. `remember(title: str, note: str, read_files: list, edited_files: list)`: Save memory note and file snapshots.
2. `recall(query: str, limit: int = 5)`: Semantic search over past memory notes across this and previous phases.
3. `get_recent_memories(limit: int = 10, offset: int = 0)`: Get recent memory logs chronologically.
4. `consult_memory(memory_id: str)`: Read a specific memory note in full.
5. `get_file_history_metadata(filepath: str)`: Get version history metadata for a file.
6. `read_past_file_content(filepath: str, memory_id: str)`: Read past file snapshot.

## Additional Workspace Tools:
7. `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
8. `grep_search(query: str, search_path: str = ".")`: Search pattern across codebase.
9. `list_dir(directory: str = ".")`: List contents of a directory.
10. `submit_phase_deliverable(deliverable: str, notes: str)`: Submit the final deliverable for the current SDLC phase.

## Protocol Rules:
- At each new SDLC phase, call `recall` to consult previous phases' design decisions and file contracts.
- Always call `remember` after drafting or implementing code/config.
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


# ---------------------------------------------------------------------------
# 4. Helper Functions
# ---------------------------------------------------------------------------

def get_aivc_system_prompt(
    benchmark_mode: bool = False,
    benchmark_type: Optional[str] = None,
    task_instructions: Optional[str] = None,
) -> str:
    """Return the system prompt for AIVC, optionally including benchmark instructions."""
    if benchmark_type in ("devbench", "sdlc"):
        base = AIVC_DEVBENCH_SYSTEM_PROMPT
    elif benchmark_mode or benchmark_type in ("swebench_cl", "swebench", "agentic_rag", "rag"):
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
) -> List[Dict[str, Any]]:
    """Return harmonized list of tool schemas for benchmark agent execution."""
    tools: List[Dict[str, Any]] = copy.deepcopy(AIVC_CORE_TOOLS_SCHEMA)
    if include_workspace:
        ws_tools = [t for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] != "submit_patch"]
        tools.extend(copy.deepcopy(ws_tools))
        if benchmark_type in ("devbench", "sdlc"):
            tools.append(copy.deepcopy(DEVBENCH_DELIVERABLE_TOOL_SCHEMA))
        else:
            submit_tool = [t for t in WORKSPACE_TOOLS_SCHEMA if t["function"]["name"] == "submit_patch"]
            if submit_tool:
                tools.extend(copy.deepcopy(submit_tool))
    if include_bash:
        tools.append(copy.deepcopy(BASH_TOOL_SCHEMA))
    return tools
