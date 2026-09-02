"""
Prompt Templates and Tool Schemas for AIVC Agentic RAG Continual Learning Benchmark.

Defines:
1. AIVC System Prompt (Mode B --arm aivc): Persistent long-term memory, cross-query accumulation.
2. Naive Baseline System Prompt (Mode A --arm naive): Ephemeral stateless execution.
3. OpenAI/OpenRouter compatible tool schema definitions for both arms.
4. Prompt formatters for multi-hop code retrieval & reasoning queries.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

AIVC_AGENTIC_RAG_SYSTEM_PROMPT = r"""# AIVC — AI Version Control (Continual Learning & Long-Term Memory)

You are an expert autonomous software architect evaluating codebases in an **Online Continual Learning** setting.
You are equipped with **AIVC Long-Term Memory**, enabling persistent knowledge accumulation, cross-query recall, and codebase dependency graph versioning.

## Multi-Hop Code Retrieval & Reasoning Mission:
You will receive a sequential stream of technical queries ($q_1, q_2, \dots, q_T$) investigating complex multi-file relationships, symbol definitions, architecture invariants, and call graphs.

## Tool Arsenal:
### 1. AIVC Persistent Memory Actions:
- `recall(query: str, top_n: int = 5)`: Perform semantic and keyword search across all memory notes accumulated from current and past queries.
- `consult_memory(memory_id: str)`: Retrieve full contents and file associations of a specific memory note.
- `get_recent_memories(limit: int = 10, offset: int = 0)`: View recent memory logs in reverse chronological order.
- `get_file_history_metadata(file_path: str)`: Check tracked version history and notes associated with a given file.
- `read_past_file_content(file_path: str, memory_id: str, diff_against: str = "current")`: Retrieve file snapshot content or diff associated with a past memory checkpoint.
- `remember(title: str, note: str, read_files: list = [], edited_files: list = [])`: Save key discoveries, architectural insights, and mapped file dependencies into long-term memory.

### 2. Codebase Exploration Actions:
- `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
- `grep_search(query: str, search_path: str = ".")`: Search for patterns, symbols, or identifiers across the codebase.
- `list_dir(directory: str = ".")`: List files and subdirectories in a directory.
- `find_symbol(symbol_name: str, symbol_type: str = "any")`: Locate definition and usage sites of a class, function, or variable.

### 3. Submission Action:
- `submit_answer(answer: str, relevant_files: list, explanation: str)`: Submit your final answer to the query with the ranked list of relevant files and explanation.

## Protocol Rules:
1. **Recall First**: At the start of every query, immediately call `recall` to verify if relevant architectural facts or file relationships were discovered in previous queries.
2. **Minimize Redundant Exploration**: Leverage memory records to directly pinpoint files instead of performing full codebase scans.
3. **Remember Findings**: Before submitting, call `remember` with a concise note summarizing key findings and relevant file dependencies so future queries can reuse this knowledge.
4. **Conclude with Submission**: Call `submit_answer` when you have sufficient evidence to answer the query accurately.
"""


NAIVE_AGENTIC_RAG_SYSTEM_PROMPT = """# Autonomous Code Reasoning Agent (Stateless Baseline)

You are an expert autonomous software engineer solving technical queries over large codebases.
You operate in a **stateless, ephemeral environment** with zero persistent memory between tasks.

## Tool Arsenal:
### 1. Codebase Exploration Actions:
- `view_file(filepath: str, start_line: int = 1, end_line: int = 100)`: Read lines from a file.
- `grep_search(query: str, search_path: str = ".")`: Search for patterns, symbols, or identifiers across the codebase.
- `list_dir(directory: str = ".")`: List files and subdirectories in a directory.
- `find_symbol(symbol_name: str, symbol_type: str = "any")`: Locate definition and usage sites of a class, function, or variable.

### 2. Submission Action:
- `submit_answer(answer: str, relevant_files: list, explanation: str)`: Submit your final answer to the query with the ranked list of relevant files and explanation.

## Protocol Rules:
1. Search and inspect files using `grep_search`, `list_dir`, and `view_file`.
2. Conclude by calling `submit_answer` once you have identified the answer and the target relevant files.
"""


# ---------------------------------------------------------------------------
# Tool Schema Definitions (OpenAI / OpenRouter Compatible)
# ---------------------------------------------------------------------------

COMMON_EXPLORATION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View contents of a file within a specified line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to target file"},
                    "start_line": {"type": "integer", "default": 1, "description": "1-indexed starting line number"},
                    "end_line": {"type": "integer", "default": 100, "description": "1-indexed ending line number"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for text or regex pattern across codebase files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text pattern or regex to search for"},
                    "search_path": {"type": "string", "default": ".", "description": "Directory or file to search within"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories within a given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": ".", "description": "Directory path to list"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Locate symbol definitions (functions, classes, variables) across the codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_name": {"type": "string", "description": "Name of symbol to find"},
                    "symbol_type": {"type": "string", "enum": ["function", "class", "variable", "any"], "default": "any"},
                },
                "required": ["symbol_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final answer for the multi-hop query with ranked relevant files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "Concise technical answer addressing the query"},
                    "relevant_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ranked list of relevant file paths that contain the ground-truth evidence",
                    },
                    "explanation": {"type": "string", "description": "Brief explanation of how the answer was derived"},
                },
                "required": ["answer", "relevant_files"],
            },
        },
    },
]

AIVC_MEMORY_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a structured memory note with tracked read and edited file associations into persistent AIVC memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Concise descriptive title of the memory note"},
                    "note": {"type": "string", "description": "Detailed markdown note capturing architectural facts, discovered relationships, and code insights"},
                    "read_files": {"type": "array", "items": {"type": "string"}, "description": "List of files consulted during investigation", "default": []},
                    "edited_files": {"type": "array", "items": {"type": "string"}, "description": "List of files modified or referenced as core dependencies", "default": []},
                },
                "required": ["title", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Perform semantic and keyword search across past memory notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or technical query"},
                    "top_n": {"type": "integer", "default": 5, "description": "Maximum number of memory items to return (default 5, max 20)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_memories",
            "description": "Retrieve recent memories in reverse chronological order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10, "description": "Number of recent memories"},
                    "offset": {"type": "integer", "default": 0, "description": "Offset from most recent"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_memory",
            "description": "Retrieve full markdown content and metadata of a memory by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Target memory ID (e.g. 'mem-0001')"},
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_history_metadata",
            "description": "Get AIVC version history and associated memory notes for a tracked file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative or absolute path of file"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_past_file_content",
            "description": "Read file snapshot content associated with a specific memory checkpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Relative or absolute path of file"},
                    "memory_id": {"type": "string", "description": "Memory snapshot ID"},
                    "diff_against": {
                        "type": "string",
                        "enum": ["current", "parent", "none"],
                        "description": "Comparison mode: 'current' (diff vs local disk), 'parent' (diff vs parent memory), 'none' (raw content).",
                        "default": "current",
                    },
                },
                "required": ["file_path", "memory_id"],
            },
        },
    },
]

# Combined tools schema
AIVC_RAG_TOOLS_SCHEMA: List[Dict[str, Any]] = AIVC_MEMORY_TOOLS + COMMON_EXPLORATION_TOOLS
NAIVE_RAG_TOOLS_SCHEMA: List[Dict[str, Any]] = COMMON_EXPLORATION_TOOLS


# ---------------------------------------------------------------------------
# Prompt Formatters
# ---------------------------------------------------------------------------

def format_agentic_rag_prompt(
    query_item: Dict[str, Any],
    arm: str = "aivc",
    episode_index: int = 1,
    total_episodes: int = 1,
) -> str:
    """Format the user prompt for a multi-hop code retrieval / reasoning query."""
    query_id = query_item.get("query_id", f"Q-{episode_index:03d}")
    repo = query_item.get("repo", "unknown_repo")
    query_text = query_item.get("query", "")
    context_hint = query_item.get("context_hint", "")
    hops = query_item.get("hops", 2)

    prompt_lines = [
        f"### Continual Learning Episode {episode_index}/{total_episodes}",
        f"**Repository**: `{repo}`",
        f"**Query ID**: `{query_id}`",
        f"**Query Complexity**: {hops}-hop Codebase Reasoning",
        "",
        "**Technical Query**:",
        query_text,
    ]

    if context_hint:
        prompt_lines.extend(["", f"**Initial Entry Point / Context Hint**:\n{context_hint}"])

    if arm == "aivc":
        prompt_lines.extend([
            "",
            "> **AIVC Instruction**: Check past memories via `recall` first. "
            "Record new discoveries via `remember` and finalize by calling `submit_answer`.",
        ])
    else:
        prompt_lines.extend([
            "",
            "> **Stateless Instruction**: Explore the repository using grep and file viewing tools, "
            "then call `submit_answer` with the identified files and answer.",
        ])

    return "\n".join(prompt_lines)
