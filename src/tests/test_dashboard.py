"""
Tests for the Dashboard API.
"""

from unittest.mock import MagicMock, patch
from aivc.web.dashboard import DashboardHandler, DashboardServer, is_client_disconnect_exception


def test_dashboard_api_graph():
    engine = MagicMock()
    engine.get_file_node_data.return_value = [{"id": "a.py"}]
    engine.get_file_cooccurrences.return_value = [{"source": "a.py", "target": "b.py", "weight": 2}]
    
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_graph()
    assert "nodes" in res
    assert "edges" in res
    assert res["nodes"][0]["id"] == "a.py"
    assert res["edges"][0]["weight"] == 2


def test_dashboard_api_search():
    engine = MagicMock()
    mock_result = MagicMock()
    mock_result.memory_id = "c1"
    mock_result.title = "test title"
    mock_result.timestamp = "2024-01-01"
    mock_result.score = 0.95
    mock_result.snippet = "snippet text"
    mock_result.file_paths = ["a.py"]
    
    engine.search.return_value = [mock_result]
    
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_search("query")
    assert len(res) == 1
    assert res[0]["memory_id"] == "c1"
    assert res[0]["title"] == "test title"
    assert res[0]["score"] == 0.95


def test_dashboard_api_search_empty():
    engine = MagicMock()
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_search("")
    assert res == []
    engine.search.assert_not_called()


def test_dashboard_api_head():
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.path = "/api/graph"
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    
    handler.do_HEAD()
    
    handler.send_response.assert_called_once_with(200)
    handler.end_headers.assert_called_once()

def test_dashboard_api_log():
    engine = MagicMock()
    mock_memory = MagicMock()
    mock_memory.id = "c1"
    mock_memory.title = "test log"
    mock_memory.timestamp = "2024-01-01"
    mock_memory.changes = ["a.py", "b.py"]
    
    engine.get_log.return_value = [mock_memory]
    
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_log(offset=5, limit=2)
    engine.get_log.assert_called_once_with(limit=2, offset=5)
    
    assert len(res) == 1
    assert res[0]["id"] == "c1"
    assert res[0]["file_count"] == 2


def test_dashboard_api_file_history():
    engine = MagicMock()
    engine.get_file_history.return_value = [{"memory_id": "c1", "title": "test", "timestamp": "2024-01-01"}]
    
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_file_history("a.py")
    engine.get_file_history.assert_called_once_with("a.py")
    assert res[0]["memory_id"] == "c1"


def test_dashboard_api_file_history_error():
    engine = MagicMock()
    engine.get_file_history.side_effect = KeyError("Not found")
    
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    
    res = handler._api_file_history("unknown.py")
    assert "error" in res


def test_dashboard_api_blob():
    engine = MagicMock()
    engine._workspace._blob_store.retrieve.side_effect = lambda h: b"\x89PNG\r\n\x1a\nfake" if h == "valid_png" else raise_key_error(h)

    def raise_key_error(h):
        raise KeyError(h)

    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine
    handler.send_json = MagicMock()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.wfile = MagicMock()

    # Valid PNG Blob
    handler._serve_blob("valid_png")
    handler.send_response.assert_called_with(200)

    # Unknown Blob -> 404
    handler._serve_blob("unknown")
    handler.send_json.assert_called_with({"error": "Blob unknown not found"}, status=404)


def test_dashboard_api_diff():
    engine = MagicMock()
    mock_mem = MagicMock()
    mock_mem.id = "m1"
    mock_mem.parent_id = "m0"

    mock_change = MagicMock()
    mock_change.path = "app.py"
    mock_change.action = "modified"
    mock_change.blob_hash = "h2"
    mock_mem.changes = [mock_change]

    engine.get_memory.return_value = mock_mem
    engine._workspace._blob_store.retrieve.return_value = b"line1\nline2_modified\n"
    engine.read_file_at_memory.return_value = b"line1\nline2\n"

    handler = DashboardHandler.__new__(DashboardHandler)
    handler.engine = engine

    res = handler._get_file_diff_and_stats("m1", "app.py")
    assert res["memory_id"] == "m1"
    assert res["path"] == "app.py"
    assert res["action"] == "modified"
    assert res["lines_added"] == 1
    assert res["lines_removed"] == 1
    assert "+line2_modified" in res["diff"]


def test_is_client_disconnect_exception():
    assert is_client_disconnect_exception(ConnectionAbortedError(10053, "WinError 10053")) is True
    assert is_client_disconnect_exception(ConnectionResetError()) is True
    assert is_client_disconnect_exception(BrokenPipeError()) is True
    assert is_client_disconnect_exception(OSError(10053, "Connection aborted")) is True
    assert is_client_disconnect_exception(ValueError("Unrelated error")) is False
    assert is_client_disconnect_exception(None) is False


def test_dashboard_server_handle_error_suppresses_disconnect():
    server = MagicMock(spec=DashboardServer)
    server.handle_error = DashboardServer.handle_error.__get__(server, DashboardServer)

    # Disconnect error should be caught and suppressed (no exception raised, returns None)
    err = ConnectionAbortedError(10053, "WinError 10053")
    with patch("sys.exc_info", return_value=(ConnectionAbortedError, err, None)):
        server.handle_error(None, ("127.0.0.1", 12345))

    # Generic error should call super().handle_error
    err_val = ValueError("Unexpected error")
    with patch("sys.exc_info", return_value=(ValueError, err_val, None)):
        with patch("http.server.HTTPServer.handle_error") as mock_super_handle_error:
            server.handle_error(None, ("127.0.0.1", 12345))
            mock_super_handle_error.assert_called_once_with(None, ("127.0.0.1", 12345))


def test_dashboard_handler_disconnect_handling():
    handler = DashboardHandler.__new__(DashboardHandler)
    handler.close_connection = False

    # Test send_json disconnect suppression
    handler.send_response = MagicMock(side_effect=ConnectionAbortedError(10053, "WinError 10053"))
    handler.send_json({"data": 1})  # should not raise

    # Test copyfile disconnect suppression
    with patch("http.server.SimpleHTTPRequestHandler.copyfile", side_effect=ConnectionAbortedError(10053, "WinError 10053")):
        handler.copyfile(MagicMock(), MagicMock())  # should not raise

    # Test do_GET disconnect setting close_connection
    with patch("http.server.SimpleHTTPRequestHandler.do_GET", side_effect=ConnectionAbortedError(10053, "WinError 10053")):
        handler.path = "/index.html"
        handler.do_GET()
        assert handler.close_connection is True

    # Test handle_one_request disconnect setting close_connection
    handler.close_connection = False
    with patch("http.server.SimpleHTTPRequestHandler.handle_one_request", side_effect=ConnectionResetError()):
        handler.handle_one_request()
        assert handler.close_connection is True


