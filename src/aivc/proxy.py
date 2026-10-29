"""
AIVC MCP Stdio Proxy.

Ultra-lightweight stdio-to-HTTP JSON-RPC bridge using standard library only.
Starts the persistent AIVC daemon out of the calling process / Job Object if not already running,
and routes MCP stdio requests with < 80ms latency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import urllib.error
import urllib.request
from pathlib import Path


def _check_health(port: int, timeout: float = 0.15) -> bool:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _is_port_occupied(port: int) -> bool:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except Exception:
        return False


def _get_daemon_json_path() -> Path:
    custom_state = os.environ.get("AIVC_DAEMON_STATE_FILE")
    if custom_state:
        return Path(custom_state)
    storage_root = os.environ.get("AIVC_STORAGE_ROOT")
    if storage_root:
        return Path(storage_root).parent / "daemon.json"
    return Path.home() / ".aivc" / "daemon.json"


def _cleanup_zombie(port: int) -> None:
    daemon_path = _get_daemon_json_path()
    stale_pid = None
    if daemon_path.exists():
        try:
            state = json.loads(daemon_path.read_text(encoding="utf-8"))
            stale_pid = state.get("pid")
        except Exception:
            pass
        try:
            daemon_path.unlink()
        except Exception:
            pass

    if stale_pid:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(stale_pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            else:
                os.kill(stale_pid, 9)
        except Exception:
            pass

    if sys.platform == "win32":
        ps_kill = (
            f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
            "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
        )
        try:
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_kill], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except Exception:
            pass
    time.sleep(0.15)


def ensure_daemon_running(port: int, timeout: float = 15.0) -> None:
    """Ensure the AIVC daemon is running on the given port, launching it if necessary."""
    if _check_health(port, timeout=0.2):
        return

    # If port is occupied or daemon.json exists but /health fails, clean up stale zombie process
    if _is_port_occupied(port) or _get_daemon_json_path().exists():
        _cleanup_zombie(port)

    if sys.platform == "win32":
        venv_pythonw = Path(sys.executable).with_name("pythonw.exe")
        if not venv_pythonw.exists():
            venv_pythonw = Path(sys.executable).resolve().with_name("pythonw.exe")
        if not venv_pythonw.exists():
            venv_pythonw = Path(sys.executable).resolve()
        exe = str(venv_pythonw)
    else:
        exe = str(Path(sys.executable).resolve())

    cmd_line = f'"{exe}" -m aivc.daemon'
    cwd_str = str(Path.home())

    if sys.platform == "win32":
        # 1. Attempt launch via WMI / CIM with SW_HIDE (ShowWindow=0) to detach outside calling Job Object
        ps_cmd = (
            "$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
            "$res = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            f"-Arguments @{{CommandLine='{cmd_line}'; CurrentDirectory='{cwd_str}'; ProcessStartupInformation=$si}}; "
            "if ($res.ReturnValue -ne 0) { exit $res.ReturnValue }"
        )
        wmi_cmd = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            ps_cmd,
        ]
        wmi_launched = False
        try:
            res = subprocess.run(wmi_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            if res.returncode == 0:
                wmi_launched = True
        except Exception:
            wmi_launched = False

        # 2. Fallback: detached Popen only if WMI failed
        if not wmi_launched:
            # DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200) | CREATE_NO_WINDOW (0x08000000)
            flags = 0x00000008 | 0x00000200 | 0x08000000
            try:
                subprocess.Popen(
                    [exe, "-m", "aivc.daemon"],
                    creationflags=flags,
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
            except Exception:
                pass
    else:
        # Non-Windows daemon launch
        try:
            subprocess.Popen(
                [exe, "-m", "aivc.daemon"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            pass

    # Wait for daemon to become healthy
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _check_health(port, timeout=0.15):
            return
        time.sleep(0.05)

    raise RuntimeError(f"AIVC daemon failed to start on port {port} within {timeout}s")


def main() -> None:
    if sys.platform == "win32":
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    port = int(os.environ.get("AIVC_DAEMON_PORT", "18765"))
    ensure_daemon_running(port)

    url = f"http://127.0.0.1:{port}/mcp"

    for line in sys.stdin:
        raw_line = line.strip()
        if not raw_line:
            continue

        req = urllib.request.Request(
            url,
            data=raw_line.encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-AIVC-Cwd": os.getcwd(),
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    body = resp.read().decode("utf-8").strip()
                    if body:
                        sys.stdout.write(body + "\n")
                        sys.stdout.flush()
        except urllib.error.HTTPError as e:
            if e.code == 204:
                continue
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                msg_in = json.loads(raw_line)
                if "id" in msg_in:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": msg_in["id"],
                        "error": {"code": -32603, "message": f"HTTP {e.code}: {err_body}"},
                    }
                    sys.stdout.write(json.dumps(err_resp) + "\n")
                    sys.stdout.flush()
            except Exception:
                pass
        except Exception as e:
            try:
                msg_in = json.loads(raw_line)
                if "id" in msg_in:
                    err_resp = {
                        "jsonrpc": "2.0",
                        "id": msg_in["id"],
                        "error": {"code": -32603, "message": str(e)},
                    }
                    sys.stdout.write(json.dumps(err_resp) + "\n")
                    sys.stdout.flush()
            except Exception:
                pass


if __name__ == "__main__":
    main()
