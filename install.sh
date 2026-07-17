#!/usr/bin/env bash
# install.sh — AIVC installer
#
# Usage:
#   bash install.sh                  # install from the current directory
#   curl -fsSL <raw_url> | bash      # install from GitHub (repo cloned to ~/.aivc)
#
# What it does:
#   1. Checks that 'uv' is available.
#   2. Resolves the AIVC source directory (current dir or ~/.aivc/).
#   3. Creates an isolated venv via 'uv venv'.
#   4. Installs the package with semantic dependencies via 'uv pip install'.
#   5. Registers AIVC in ~/.gemini/antigravity/mcp_config.json using Python's
#      json module (no jq, no sed — crash-safe).
#
# NO FALLBACK: any error aborts the entire script (set -euo pipefail).

set -euo pipefail

# Ensure local bin is in PATH (especially for uv on Windows)
export PATH="${HOME}/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Colour

info()    { echo -e "${BLUE}[aivc]${NC} $*"; }
success() { echo -e "${GREEN}[aivc]${NC} $*"; }
die()     { echo -e "${RED}[aivc] ERROR:${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------

command -v uv >/dev/null 2>&1 || die "'uv' is not installed. Install it with: curl -fsSL https://astral.sh/uv/install.sh | sh"
command -v python3 >/dev/null 2>&1 || die "python3 is required but not found."

# ---------------------------------------------------------------------------
# 2. Resolve source directory
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-install.sh}")" 2>/dev/null && pwd || true)"
AIVC_HOME="${HOME}/.aivc"

# If the script is being piped (BASH_SOURCE is empty) or run outside the repo,
# we clone/update the repo into ~/.aivc/repo.
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]] && grep -q 'name = "aivc"' "${SCRIPT_DIR}/pyproject.toml" 2>/dev/null; then
    SOURCE_DIR="${SCRIPT_DIR}"
    info "Using local source directory: ${SOURCE_DIR}"
else
    REPO_DIR="${AIVC_HOME}/repo"
    info "Source directory not found locally — cloning into ${REPO_DIR} ..."
    mkdir -p "${AIVC_HOME}"
    if [[ -d "${REPO_DIR}/.git" ]]; then
        git -C "${REPO_DIR}" pull --ff-only
    else
        REPO_URL="${AIVC_REPO_URL:-https://github.com/hjamet/aivc.git}"
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi
    SOURCE_DIR="${REPO_DIR}"
fi

# ---------------------------------------------------------------------------
# 3. Create isolated venv
# ---------------------------------------------------------------------------

VENV_DIR="${AIVC_HOME}/venv"
info "Creating virtual environment at ${VENV_DIR} ..."

# Robustly clear existing venv to avoid "os error 1" on Windows/MSYS2
if [[ -d "${VENV_DIR}" ]]; then
    rm -rf "${VENV_DIR}"
fi

uv venv "${VENV_DIR}" --python python3

# Detect venv python path (Windows uses Scripts/, Unix uses bin/)
if [[ -f "${VENV_DIR}/Scripts/python.exe" ]]; then
    VENV_PYTHON="${VENV_DIR}/Scripts/python.exe"
    VENV_BIN_DIR="${VENV_DIR}/Scripts"
    WINDOWS=1
else
    VENV_PYTHON="${VENV_DIR}/bin/python"
    VENV_BIN_DIR="${VENV_DIR}/bin"
    WINDOWS=0
fi

# 4. Install the package with semantic dependencies
# ---------------------------------------------------------------------------

info "Installing aivc[all] into the venv (this may take a moment) ..."
pushd "${SOURCE_DIR}" >/dev/null

EXTRA_ARGS=""
if [[ "$WINDOWS" == "1" ]]; then
    info "Windows detected — adding CUDA 12.1 index for GPU acceleration..."
    EXTRA_ARGS="--extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match"
fi

uv pip install $EXTRA_ARGS --python "${VENV_PYTHON}" -e ".[all]"
popd >/dev/null

# ---------------------------------------------------------------------------
# 5. Inject AIVC into ~/.gemini/config/mcp_config.json
# ---------------------------------------------------------------------------

MCP_CONFIG="${HOME}/.gemini/config/mcp_config.json"
info "Configuring MCP server entry in ${MCP_CONFIG} ..."

# Use Python's json module to safely read, update, and write the config.
# We use the venv python itself to resolve native Windows paths if needed.
"${VENV_PYTHON}" - <<PYEOF
import json
import pathlib
import sys
import os

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
    except Exception:
        # If anything goes wrong (invalid JSON, empty, etc.), start fresh.
        config = {}
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

aivc_env = {
    "AIVC_STORAGE_ROOT": str(home / ".aivc" / "storage"),
    "HOME": str(home),
    "USERPROFILE": str(home)
}
if sys.platform == "win32":
    aivc_env.update({
        "SystemRoot": "C:\\Windows",
        "PATH": "C:\\Windows\\system32;C:\\Windows"
    })

config["mcpServers"]["aivc"] = {
    "command": sys.executable,
    "args": ["-m", "aivc.server"],
    "env": aivc_env,
}

config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
print(f"[aivc] MCP entry written to {config_path}")

# --- Generate aivc config.json ---
aivc_config_path = home / ".aivc" / "config.json"
import socket
if not aivc_config_path.exists():
    aivc_config = {
        "machine_id": socket.gethostname(),
        "sync": {
            "enabled": False,
            "remote_name": "aivc_remote",
            "sync_blobs": False,
            "remote_machines": []
        }
    }
    aivc_config_path.write_text(json.dumps(aivc_config, indent=4), encoding="utf-8")
    print(f"[aivc] Default config created at {aivc_config_path}")

PYEOF

# ---------------------------------------------------------------------------
# 6. Inject AIVC best practices into ~/.gemini/GEMINI.md
# ---------------------------------------------------------------------------

GEMINI_MD="${HOME}/.gemini/GEMINI.md"
MARKER_START="<!-- AIVC:START -->"
MARKER_END="<!-- AIVC:END -->"

AIVC_BLOCK="${MARKER_START}
# AIVC — AI Version Control (Long-Term Memory)

> [!IMPORTANT]
> **USE MCP TOOLS ONLY — NEVER RUN CLI SHELL COMMANDS:**
> As an AI assistant, you MUST interact with AIVC **exclusively** through its registered MCP tools (\`remember\`, \`recall\`, \`get_recent_memories\`, \`consult_memory\`, \`get_file_history_metadata\`, \`read_past_file_content\`, \`get_status\`, etc.).
> **NEVER execute \`aivc\` CLI shell commands in the terminal (e.g. \`aivc sync\`, \`aivc status\`, \`aivc recall\`)** under any circumstances. Running the CLI in the terminal spawns separate process environments, misses the current session context, and is strictly reserved for the human user.

## Rules

1. **Remember often.** Call \`remember\` after every meaningful step (sub-task done, file created/modified, decision made, error resolved, checkpoint reached). Notes must be **detailed**: what, why, errors, decisions, observations, next steps. A one-liner is a failure.
2. **Start sessions with context recovery.** Before any work: \`get_recent_memories\` → \`recall\` (≥1 query) → \`consult_memory\` on relevant hits → \`get_file_history_metadata\` on files you'll modify.
3. **Explore before you act.** Search memory first — never redo past work. Your memory contains solutions, patterns, and lessons.
4. **Mention files you work on.** Always pass the files you consulted in \`read_files\` and the files you modified in \`edited_files\` when calling \`remember\`. This is how AIVC tracks file associations — there is no separate tracking tool.
5. **Write for your future self.** Memory notes are handover memos — include reasoning, context, and recommendations as if briefing a colleague with zero context.
${MARKER_END}"

info "Injecting AIVC best practices into ${GEMINI_MD} ..."
mkdir -p "$(dirname "${GEMINI_MD}")"

if [[ -f "${GEMINI_MD}" ]] && grep -qF "${MARKER_START}" "${GEMINI_MD}"; then
    # Update existing block (replace content between markers).
    "${VENV_PYTHON}" - <<PYEOF2
import pathlib, re

# Handle Windows home correctly via pathlib
gemini = pathlib.Path.home() / ".gemini" / "GEMINI.md"
content = gemini.read_text(encoding="utf-8")

block = r'''${AIVC_BLOCK}'''

pattern = re.compile(
    r"${MARKER_START}.*?${MARKER_END}",
    re.DOTALL,
)
new_content = pattern.sub(block, content)
gemini.write_text(new_content, encoding="utf-8")
print(f"[aivc] Updated AIVC block in {gemini}")
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

success "AIVC installed successfully!"
echo ""
echo "  Storage root : ${AIVC_HOME}/storage"
echo "  Venv         : ${VENV_DIR}"

# 7. Expose CLI wrapper
USER_BIN_DIR="${HOME}/.local/bin"
mkdir -p "${USER_BIN_DIR}"
AIVC_WRAPPER="${USER_BIN_DIR}/aivc"

info "Creating CLI wrapper at ${AIVC_WRAPPER} ..."
cat <<EOF > "${AIVC_WRAPPER}"
#!/usr/bin/env bash
# AIVC CLI Wrapper
exec "${VENV_PYTHON}" -m aivc.cli "\$@"
EOF
chmod +x "${AIVC_WRAPPER}"

if [[ "$WINDOWS" == "1" ]]; then
    AIVC_CMD_WRAPPER="${USER_BIN_DIR}/aivc.cmd"
    info "Windows detected — creating CMD wrapper at ${AIVC_CMD_WRAPPER} ..."
    NATIVE_PYTHON=$("${VENV_PYTHON}" -c "import sys, os; print(os.path.normpath(sys.executable))" | tr -d '\r')
    cat <<EOF > "${AIVC_CMD_WRAPPER}"
@echo off
"${NATIVE_PYTHON}" -m aivc.cli %*
EOF
    echo "  CLI Command  : aivc (wrappers in ${USER_BIN_DIR}: aivc, aivc.cmd)"
else
    echo "  CLI Command  : aivc (wrapper in ${USER_BIN_DIR})"
fi
echo "  MCP config   : ${MCP_CONFIG}"

# 8. Run migration
info "Running CoreIndex migration..."
AIVC_STORAGE_ROOT="${AIVC_HOME}/storage" "${VENV_PYTHON}" -m aivc.cli migrate || info "Migration skipped (or not needed)."

echo "  Agent rules  : ${GEMINI_MD}"
echo ""
echo "Restart Gemini Antigravity to pick up the new MCP server."

echo "  Agent rules  : ${GEMINI_MD}"
echo ""
echo "Restart Gemini Antigravity to pick up the new MCP server."
