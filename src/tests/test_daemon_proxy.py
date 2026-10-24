"""
Tests for AIVC Daemon and Proxy.

Validates the HTTP Daemon endpoints (health, initialize, tools/list, tools/call, shutdown)
and the stdio proxy communication layer.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import tempfile

_TEST_TMP_DIR = tempfile.TemporaryDirectory()
os.environ["AIVC_STORAGE_ROOT"] = str(Path(_TEST_TMP_DIR.name) / "storage")
os.environ["AIVC_DISABLE_CROSS_ENCODER"] = "True"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestDaemonAndProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _get_free_port()
        from aivc.daemon import DaemonHTTPServer, DaemonHTTPRequestHandler

        cls.server = DaemonHTTPServer(("127.0.0.1", cls.port), DaemonHTTPRequestHandler)
        cls.server.timeout = 1.0

        def _serve():
            while cls.server.running:
                cls.server.handle_request()

        cls.thread = threading.Thread(target=_serve, daemon=True)
        cls.thread.start()

        # Wait for server to respond to health check
        deadline = time.time() + 5.0
        started = False
        while time.time() < deadline:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{cls.port}/health")
                with urllib.request.urlopen(req, timeout=0.2) as resp:
                    if resp.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(0.05)
        if not started:
            raise RuntimeError(f"Test daemon server failed to start on port {cls.port}")

    @classmethod
    def tearDownClass(cls):
        cls.server.running = False
        try:
            cls.server.server_close()
        except Exception:
            pass
        try:
            _TEST_TMP_DIR.cleanup()
        except Exception:
            pass

    def _post_mcp(self, payload: dict) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mcp",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}

    def test_01_health_endpoint(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("port"), self.port)
            self.assertEqual(data.get("pid"), os.getpid())

    def test_02_initialize(self):
        status, data = self._post_mcp({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        })
        self.assertEqual(status, 200)
        self.assertEqual(data.get("jsonrpc"), "2.0")
        self.assertEqual(data.get("id"), 1)
        res = data.get("result", {})
        self.assertEqual(res.get("serverInfo", {}).get("name"), "aivc")
        self.assertEqual(res.get("serverInfo", {}).get("version"), "0.1.0")
        self.assertIn("instructions", res)

    def test_03_tools_list_has_streamlined_6_tools(self):
        status, data = self._post_mcp({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        self.assertEqual(status, 200)
        tools = data.get("result", {}).get("tools", [])
        tool_names = {t["name"] for t in tools}
        expected_tools = {
            "remember",
            "recall",
            "consult_memory",
            "get_file_history",
            "complete_user_items",
            "restore_file",
        }
        self.assertEqual(tool_names, expected_tools)

    def test_04_tools_call_get_file_history(self):
        status, data = self._post_mcp({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_file_history",
                "arguments": {"file_path": "nonexistent_file_test_path.py"},
            },
        })
        self.assertEqual(status, 200)
        res = data.get("result", {})
        self.assertFalse(res.get("isError"))
        content = res.get("content", [])
        self.assertGreater(len(content), 0)
        text = content[0].get("text", "")
        self.assertIn("nonexistent_file_test_path.py", text)

    def test_05_tools_call_recall(self):
        status, data = self._post_mcp({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "recall",
                "arguments": {"query": "test checkpoint"},
            },
        })
        self.assertEqual(status, 200)
        res = data.get("result", {})
        self.assertFalse(res.get("isError"))
        content = res.get("content", [])
        self.assertGreater(len(content), 0)

    def test_06_notification_returns_204(self):
        data = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mcp",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 204)

    def test_07_shutdown_endpoint(self):
        # We start a dedicated temporary server to test shutdown
        temp_port = _get_free_port()
        from aivc.daemon import DaemonHTTPServer, DaemonHTTPRequestHandler
        srv = DaemonHTTPServer(("127.0.0.1", temp_port), DaemonHTTPRequestHandler)
        t = threading.Thread(target=lambda: srv.handle_request(), daemon=True)
        t.start()
        time.sleep(0.05)

        req = urllib.request.Request(
            f"http://127.0.0.1:{temp_port}/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json", "Connection": "close"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "shutting_down")
        self.assertFalse(srv.running)
        srv.server_close()

    def test_08_handle_error_silences_connection_reset(self):
        from aivc.daemon import DaemonHTTPServer, DaemonHTTPRequestHandler
        srv = DaemonHTTPServer(("127.0.0.1", 0), DaemonHTTPRequestHandler)
        try:
            try:
                raise ConnectionResetError("[WinError 10054] An existing connection was forcibly closed")
            except ConnectionResetError:
                # Should return cleanly without reraising or printing traceback
                srv.handle_error(None, ("127.0.0.1", 12345))

            try:
                raise BrokenPipeError("Broken pipe")
            except BrokenPipeError:
                srv.handle_error(None, ("127.0.0.1", 12345))

            try:
                raise ConnectionAbortedError("Connection aborted")
            except ConnectionAbortedError:
                srv.handle_error(None, ("127.0.0.1", 12345))
        finally:
            srv.server_close()

    def test_09_resolve_relative_args(self):
        from aivc.daemon import _resolve_relative_args
        with tempfile.TemporaryDirectory() as client_dir:
            client_path = Path(client_dir)
            test_file = client_path / "sub" / "file.txt"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("hello", encoding="utf-8")

            args = {
                "file_path": "sub/file.txt",
                "target_path": "sub/output.txt",
                "read_files": ["sub/file.txt", "nonexistent.txt"],
                "edited_files": ["sub/file.txt"],
            }
            resolved = _resolve_relative_args(args, client_dir)
            self.assertEqual(resolved["file_path"], str(test_file.resolve()))
            self.assertEqual(resolved["target_path"], str((client_path / "sub" / "output.txt").resolve()))
            self.assertEqual(resolved["read_files"], [str(test_file.resolve()), "nonexistent.txt"])
            self.assertEqual(resolved["edited_files"], [str(test_file.resolve())])

    def test_10_workspace_relative_and_basename_history_and_remember(self):
        from aivc.core.workspace import Workspace
        with tempfile.TemporaryDirectory() as storage_tmp:
            storage_path = Path(storage_tmp)
            ws = Workspace(storage_path)
            try:
                fake_tracked = "~/Documents/VoiceNotes/AIVC.md"
                ws._state["tracked_files"][fake_tracked] = {"hash": None, "mtime": None, "size": None}
                ws._save_state()

                # Create an initial memory touching this file
                mem = ws.remember(
                    title="Initial note for AIVC",
                    journal="Journal note",
                    read_files=[fake_tracked],
                )

                # 1. get_file_history with relative basename "AIVC.md"
                hist1 = ws.get_file_history("AIVC.md")
                self.assertEqual(len(hist1), 1)
                self.assertEqual(hist1[0].id, mem.id)

                # 2. get_file_history with alternative relative prefix "notes/AIVC.md"
                hist2 = ws.get_file_history("notes/AIVC.md")
                self.assertEqual(len(hist2), 1)
                self.assertEqual(hist2[0].id, mem.id)

                # 3. remember with relative paths resolving to fake_tracked
                mem2 = ws.remember(
                    title="Second note via relative names",
                    journal="Consulting AIVC.md",
                    read_files=["AIVC.md", "notes/AIVC.md"],
                )
                # Should have deduplicated to single change with fake_tracked
                self.assertEqual(len(mem2.changes), 1)
                self.assertEqual(mem2.changes[0].path, fake_tracked)

                # File history should now contain 2 memories
                hist3 = ws.get_file_history("AIVC.md")
                self.assertEqual(len(hist3), 2)
            finally:
                ws._index.close()


if __name__ == "__main__":
    unittest.main()
