#!/usr/bin/env bash
# install_dev.sh — AIVC development installer
# Install local repo in an isolated local .venv, with ALL dependencies
# and registers it as the active MCP server.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[aivc-dev]${NC} $*"; }
success() { echo -e "${GREEN}[aivc-dev]${NC} $*"; }
die()     { echo -e "${RED}[aivc-dev] ERROR:${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

command -v uv >/dev/null 2>&1 || die "'uv' is not installed."

VENV_DIR="${SCRIPT_DIR}/.venv"
info "Creating local virtual environment at ${VENV_DIR} ..."
uv venv "${VENV_DIR}" --python python3

info "Installing aivc[all] into the local venv ..."
uv pip install --python "${VENV_DIR}/bin/python" -e ".[all]"

MCP_CONFIG="${HOME}/.gemini/config/mcp_config.json"
info "Configuring MCP server entry in ${MCP_CONFIG} ..."

"${VENV_DIR}/bin/python" - <<PYEOF
import json
import pathlib
import sys

# Use pathlib to get native home (handles ~ correctly on both OS)
home = pathlib.Path.home()
config_path = home / ".gemini" / "config" / "mcp_config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

if config_path.exists():
    try:
        raw = config_path.read_text(encoding="utf-8").strip()
        if not raw:
            config = {}
        else:
            config = json.loads(raw)
    except Exception as exc:
        sys.exit(f"[aivc-dev] ERROR: {config_path} contains invalid JSON: {exc}")
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

config["mcpServers"]["aivc"] = {
    "command": "${VENV_DIR}/bin/python",
    "args": ["-m", "aivc.server"],
    "env": {
        "AIVC_STORAGE_ROOT": str(home / ".aivc" / "storage")
    },
}

config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(f"[aivc-dev] MCP entry written to {config_path}")
PYEOF

# ---------------------------------------------------------------------------
# Inject AIVC best practices into ~/.gemini/GEMINI.md
# ---------------------------------------------------------------------------

GEMINI_MD="${HOME}/.gemini/GEMINI.md"
MARKER_START="<!-- AIVC:START -->"
MARKER_END="<!-- AIVC:END -->"

AIVC_BLOCK="${MARKER_START}
# AIVC — AI Version Control (Long-Term Memory)

You have access to a persistent, versioned memory system called **AIVC**.
AIVC is your **long-term memory**. Use it actively — it is the only way to
preserve context beyond a single conversation.

## CRITICAL: Memory Discipline

### 1. Remember After EVERY Modification
- Call \`remember\` (the MCP tool, NOT git) after **every meaningful step**.
- A memory is required after: completing a sub-task, creating/modifying any file,
  making a decision, discovering something, encountering an error, or any checkpoint.
- **If you did something worth remembering, commit it NOW.**

### 2. Start Every Session with Context Recovery
Before doing ANY work, you MUST reconstruct your working context:
1. Call \`get_recent_memories\` to see what was done recently.
2. Perform **at least 5 \`recall\` calls** with varied queries to explore
   relevant past memories (e.g. the current topic, related files, similar problems).
3. Use \`consult_memory\` on the most relevant results to read the full details.
4. Use \`consult_file\` to understand the history of files you will modify.

### 3. Explore Before You Act
- **Never attempt a modification that was already done in the past.** Search your
  memory first to check.
- Understand the links between files and their shared history.
- Explore broadly — your memory contains solutions, patterns, and lessons learned.

### 4. Write Extremely Detailed Memory Notes
Your memory notes are your future self's lifeline. They must include:
- **What was done** and why
- **Errors encountered** and how they were resolved
- **Decisions made** and the reasoning behind them
- **Observations and surprises** discovered during the work
- **Recommendations for the future** — what should be done next, what to watch out for
- **Links to related memories or files** when relevant

> A one-liner memory note is a **failure**. Write as if briefing a colleague
> who has zero context but needs to continue your work tomorrow.
${MARKER_END}"

info "Injecting AIVC best practices into ${GEMINI_MD} ..."
mkdir -p "$(dirname "${GEMINI_MD}")"

if [[ -f "${GEMINI_MD}" ]] && grep -qF "${MARKER_START}" "${GEMINI_MD}"; then
    # Update existing block (replace content between markers).
    "${VENV_DIR}/bin/python" - <<PYEOF2
import pathlib, re

gemini = pathlib.Path("${GEMINI_MD}")
content = gemini.read_text(encoding="utf-8")

block = '''${AIVC_BLOCK}'''

pattern = re.compile(
    r"${MARKER_START}.*?${MARKER_END}",
    re.DOTALL,
)
new_content = pattern.sub(block, content)
gemini.write_text(new_content, encoding="utf-8")
print(f"[aivc-dev] Updated AIVC block in {gemini}")
PYEOF2
else
    # Append block to end of file (or create file).
    echo "" >> "${GEMINI_MD}"
    echo "${AIVC_BLOCK}" >> "${GEMINI_MD}"
    info "Appended AIVC block to ${GEMINI_MD}"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

success "AIVC Dev environment installed in ${VENV_DIR}!"
echo ""
echo "  Venv         : ${VENV_DIR}"

# Symlink CLI to ~/.local/bin
USER_BIN_DIR="${HOME}/.local/bin"
mkdir -p "${USER_BIN_DIR}"
ln -sf "${VENV_DIR}/bin/aivc" "${USER_BIN_DIR}/aivc"

echo "  CLI Command  : aivc (symlinked in ${USER_BIN_DIR})"
echo "  MCP config   : ${MCP_CONFIG}"

# Run migration
info "Running CoreIndex migration..."
AIVC_STORAGE_ROOT="${HOME}/.aivc/storage" "${VENV_DIR}/bin/aivc" migrate || info "Migration skipped (or not needed)."

info "Pre-downloading FastEmbed model..."
AIVC_STORAGE_ROOT="${HOME}/.aivc/storage" "${VENV_DIR}/bin/python" -c "
from pathlib import Path
import os
storage_root = Path(os.environ['AIVC_STORAGE_ROOT'])
storage_root.mkdir(parents=True, exist_ok=True)
cache_dir = storage_root / 'fastembed_cache'
cache_dir.mkdir(parents=True, exist_ok=True)
from aivc.semantic.indexer import _get_shared_model
print(f'Downloading model to {cache_dir}...')
_get_shared_model(cache_dir)
print('Model ready!')
" || info "FastEmbed model download failed, will retry later."

echo "  Agent rules  : ${GEMINI_MD}"
echo ""
echo "Restart Gemini Antigravity to pick up the changes."
