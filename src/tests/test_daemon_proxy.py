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

# Configure test environment
os.environ["AIVC_STORAGE_ROOT"] = os.environ.get("AIVC_STORAGE_ROOT", str(Path.home() / ".aivc" / "storage"))
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

        req = urllib.request.Request(
            f"http://127.0.0.1:{temp_port}/shutdown",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "shutting_down")
        self.assertFalse(srv.running)
        srv.server_close()


if __name__ == "__main__":
    unittest.main()
