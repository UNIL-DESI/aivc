"""
AIVC Persistent Background Daemon.

Runs an in-memory HTTP server keeping SemanticEngine, ChromaDB, FastEmbed ONNX
and the BackgroundSyncer pre-warmed on a single main thread to eliminate cold-start
latency and prevent subprocess thread deadlocks on Windows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Mission-critical Windows environment variables before heavy imports
if sys.platform == "win32":
    os.environ["AIVC_DISABLE_CROSS_ENCODER"] = "True"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("aivc.daemon")


def get_aivc_home() -> Path:
    storage_root = os.environ.get("AIVC_STORAGE_ROOT")
    if storage_root:
        return Path(storage_root).parent
    try:
        from aivc.config import get_home_dir
        return get_home_dir() / ".aivc"
    except Exception:
        return Path.home() / ".aivc"


def _get_daemon_json_path() -> Path:
    custom_state = os.environ.get("AIVC_DAEMON_STATE_FILE")
    if custom_state:
        return Path(custom_state)
    return get_aivc_home() / "daemon.json"


def _is_relative_path(v: str) -> bool:
    if not isinstance(v, str) or not v.strip():
        return False
    v_strip = v.strip()
    if v_strip.startswith(("~", "/", "\\")):
        return False
    if len(v_strip) >= 2 and v_strip[1] == ":":
        return False
    return True


def _resolve_relative_args(arguments: dict[str, Any], client_cwd: str | None) -> dict[str, Any]:
    if not client_cwd or not isinstance(client_cwd, str):
        return arguments
    cwd_path = Path(client_cwd)
    if not cwd_path.is_dir():
        return arguments

    resolved = dict(arguments)
    for k in ("file_path", "target_path"):
        if k in resolved and isinstance(resolved[k], str) and resolved[k].strip():
            val = resolved[k]
            if _is_relative_path(val):
                candidate = cwd_path / val
                if k == "target_path":
                    if candidate.exists() or candidate.parent.exists():
                        resolved[k] = str(candidate.resolve())
                else:
                    if candidate.exists():
                        resolved[k] = str(candidate.resolve())

    for k in ("read_files", "edited_files"):
        if k in resolved and isinstance(resolved[k], list):
            new_list = []
            for item in resolved[k]:
                if isinstance(item, str) and _is_relative_path(item):
                    candidate = cwd_path / item
                    if candidate.exists():
                        new_list.append(str(candidate.resolve()))
                    else:
                        new_list.append(item)
                else:
                    new_list.append(item)
            resolved[k] = new_list

    return resolved


class DaemonHTTPRequestHandler(BaseHTTPRequestHandler):
    server: "DaemonHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard logging to keep daemon quiet
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def _send_json_response(self, status_code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        self.server.touch()
        if self.path == "/health":
            self._send_json_response(200, {
                "status": "ok",
                "pid": os.getpid(),
                "port": self.server.server_port,
            })
        else:
            self._send_json_response(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        self.server.touch()
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""

        if self.path == "/health":
            self._send_json_response(200, {
                "status": "ok",
                "pid": os.getpid(),
                "port": self.server.server_port,
            })
            return

        if self.path == "/shutdown":
            self.server.running = False
            self._send_json_response(200, {"status": "shutting_down"})
            return

        if self.path == "/mcp":
            if not post_data:
                self._send_json_response(400, {"error": "Empty request body"})
                return

            raw_body = post_data.decode("utf-8")
            try:
                msg = json.loads(raw_body)
            except Exception as e:
                self._send_json_response(400, {"error": f"Invalid JSON: {e}"})
                return

            # Notifications (no 'id' field in JSON-RPC)
            if "id" not in msg:
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                return

            req_id = msg["id"]
            method = msg.get("method", "")

            try:
                if method == "initialize":
                    from aivc import server as aivc_server
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                            "capabilities": {"tools": {"listChanged": False}},
                            "serverInfo": {"name": "aivc", "version": "0.1.0"},
                            "instructions": getattr(aivc_server, "_SYSTEM_PROMPT", ""),
                        },
                    }
                    self._send_json_response(200, resp)

                elif method == "ping":
                    self._send_json_response(200, {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {},
                    })

                elif method == "tools/list":
                    from aivc import server as aivc_server
                    tools = asyncio.run(aivc_server.mcp.list_tools())
                    tools_list = [t.model_dump(exclude_none=True) for t in tools]
                    self._send_json_response(200, {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"tools": tools_list},
                    })

                elif method == "tools/call":
                    from aivc import server as aivc_server
                    params = msg.get("params", {})
                    name = params.get("name", "")
                    arguments = params.get("arguments", {})
                    client_cwd = self.headers.get("X-AIVC-Cwd")
                    arguments = _resolve_relative_args(arguments, client_cwd)

                    try:
                        res = asyncio.run(aivc_server.mcp.call_tool(name, arguments))
                        # FastMCP returns a tuple (content_blocks, extra_context)
                        if isinstance(res, tuple):
                            content_blocks = res[0]
                        elif isinstance(res, list):
                            content_blocks = res
                        else:
                            content_blocks = [res]

                        content_list = []
                        for c in content_blocks:
                            if hasattr(c, "model_dump"):
                                content_list.append(c.model_dump(exclude_none=True))
                            elif isinstance(c, dict):
                                content_list.append(c)
                            else:
                                content_list.append({"type": "text", "text": str(c)})

                        self._send_json_response(200, {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": content_list,
                                "isError": False,
                            },
                        })
                    except Exception as err:
                        self._send_json_response(200, {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [{"type": "text", "text": str(err)}],
                                "isError": True,
                            },
                        })

                else:
                    self._send_json_response(200, {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Method {method!r} not found"},
                    })

            except Exception as e:
                self._send_json_response(200, {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Internal daemon error: {e}"},
                })
        else:
            self._send_json_response(404, {"error": "Not Found"})


class DaemonHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if sys.platform == "win32":
            try:
                import socket
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                pass
        super().server_bind()

    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type[BaseHTTPRequestHandler]):
        super().__init__(server_address, RequestHandlerClass)
        self.running = True
        self.last_activity = time.time()

    def touch(self) -> None:
        self.last_activity = time.time()

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc_type, exc_val, _ = sys.exc_info()
        if exc_type is not None and issubclass(exc_type, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def run_daemon(port: int | None = None, idle_timeout: int | None = None) -> None:
    """Start and run the persistent AIVC daemon."""
    # Redirect stdout and stderr to daemon.log for pythonw windowless execution
    aivc_home = get_aivc_home()
    aivc_home.mkdir(parents=True, exist_ok=True)
    log_file = (aivc_home / "daemon.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    if port is None:
        port = int(os.environ.get("AIVC_DAEMON_PORT", "18765"))
    if idle_timeout is None:
        idle_timeout = int(os.environ.get("AIVC_DAEMON_IDLE_TIMEOUT", "7200"))

    # Mission-critical Windows pre-configurations
    if sys.platform == "win32":
        os.environ["AIVC_DISABLE_CROSS_ENCODER"] = "True"

    # Start HTTP server immediately so /health responds within < 1s
    try:
        server = DaemonHTTPServer(("127.0.0.1", port), DaemonHTTPRequestHandler)
    except OSError as e:
        # Check if another daemon instance is already healthy on this port
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                if resp.status == 200:
                    print(f"[aivc.daemon] Daemon already running and healthy on port {port}. Exiting.", flush=True)
                    return
        except Exception:
            pass
        print(f"[aivc.daemon] Failed to bind port {port}: {e}", file=sys.stderr, flush=True)
        raise

    server.timeout = 1.0  # Periodic wakeup for idle timeout check and shutdown flag

    # Record daemon state
    daemon_path = _get_daemon_json_path()
    daemon_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "pid": os.getpid(),
        "port": port,
        "started_at": time.time(),
    }
    daemon_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"[aivc.daemon] Daemon listening on http://127.0.0.1:{port} (PID: {os.getpid()})", flush=True)

    # Pre-warm server and engine components in a non-blocking background thread
    def _warmup_worker() -> None:
        from aivc import server as aivc_server
        try:
            engine = aivc_server._get_engine()
            _ = engine._indexer._collection
            print("[aivc.daemon] Semantic indexer pre-warmed successfully.", flush=True)
        except Exception as e:
            print(f"[aivc.daemon] Warning during indexer pre-warming: {e}", file=sys.stderr, flush=True)

        try:
            syncer = aivc_server._get_syncer()
            if syncer and syncer.manager.enabled:
                syncer.start()
                print("[aivc.daemon] Background syncer started successfully.", flush=True)
        except Exception as e:
            print(f"[aivc.daemon] Warning during syncer start: {e}", file=sys.stderr, flush=True)

    warmup_thread = threading.Thread(target=_warmup_worker, daemon=True, name="AivcDaemonWarmup")
    warmup_thread.start()

    try:
        while server.running:
            server.handle_request()
            if time.time() - server.last_activity > idle_timeout:
                print(f"[aivc.daemon] Idle timeout ({idle_timeout}s) exceeded. Shutting down.", flush=True)
                break
    finally:
        server.server_close()
        try:
            if daemon_path.exists():
                daemon_path.unlink()
        except Exception:
            pass
        print("[aivc.daemon] Daemon shut down cleanly.", flush=True)


if __name__ == "__main__":
    run_daemon()
