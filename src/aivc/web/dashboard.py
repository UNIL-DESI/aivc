"""
AIVC Web Dashboard mini-server (Phase 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from aivc.semantic.engine import SemanticEngine


def is_client_disconnect_exception(exc: Exception | None) -> bool:
    """Check if an exception is caused by a client socket disconnect/abort."""
    if exc is None:
        return False
    if isinstance(exc, (ConnectionError, BrokenPipeError)):
        return True
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        errno = getattr(exc, "errno", None)
        if winerror in (10053, 10054, 10038, 10061) or errno in (32, 104, 10053, 10054, 10038):
            return True
        err_msg = str(exc).lower()
        if any(msg in err_msg for msg in ("10053", "10054", "broken pipe", "connection aborted", "connection reset")):
            return True
    return False


class DashboardServer(HTTPServer):
    """Custom HTTPServer that silently handles client disconnect errors."""

    def handle_error(self, request, client_address):
        exctype, value, tb = sys.exc_info()
        if value and is_client_disconnect_exception(value):
            return
        if exctype and issubclass(exctype, (ConnectionError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, engine=None, **kwargs):
        self.engine = engine
        # Serve static files from src/aivc/web/static
        static_dir = Path(__file__).parent / "static"
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                self.close_connection = True
            else:
                raise

    def copyfile(self, infile, outfile):
        try:
            super().copyfile(infile, outfile)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                pass
            else:
                raise

    def do_HEAD(self):
        try:
            parsed = urlparse(self.path)
            
            if (
                parsed.path in ("/api/graph", "/api/search", "/api/log", "/api/diff")
                or parsed.path.startswith("/api/blob/")
                or parsed.path.startswith("/api/memory/")
            ):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                return
                
            super().do_HEAD()
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                self.close_connection = True
            else:
                raise

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            
            if parsed.path == "/api/graph":
                self.send_json(self._api_graph())
                return
                
            if parsed.path == "/api/search":
                qs = parse_qs(parsed.query)
                query = qs.get("q", [""])[0]
                self.send_json(self._api_search(query))
                return

            if parsed.path == "/api/log":
                qs = parse_qs(parsed.query)
                offset = int(qs.get("offset", ["0"])[0])
                limit = int(qs.get("limit", ["10"])[0])
                self.send_json(self._api_log(offset=offset, limit=limit))
                return

            if parsed.path == "/api/diff":
                qs = parse_qs(parsed.query)
                memory_id = qs.get("memory_id", [""])[0]
                file_path = qs.get("path", [""])[0]
                self.send_json(self._get_file_diff_and_stats(memory_id, file_path))
                return

            if parsed.path.startswith("/api/blob/"):
                blob_hash = parsed.path[len("/api/blob/"):]
                self._serve_blob(blob_hash)
                return

            if parsed.path.startswith("/api/memory/"):
                memory_id = parsed.path[len("/api/memory/"):]
                self.send_json(self._api_memory(memory_id))
                return

            if parsed.path.startswith("/api/file-history/"):
                import urllib.parse
                # The path might be url-encoded (e.g. spaces, slashes)
                file_path = urllib.parse.unquote(parsed.path[len("/api/file-history/"):])
                self.send_json(self._api_file_history(file_path))
                return
                
            # Default behavior: serve static files
            super().do_GET()
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                self.close_connection = True
            else:
                raise

    def send_json(self, data: dict | list, status: int = 200):
        content = json.dumps(data).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                pass
            else:
                raise

    def _serve_blob(self, blob_hash: str):
        """Serve raw binary or document blob with automatic MIME detection."""
        try:
            data = self.engine._workspace._blob_store.retrieve(blob_hash)
        except KeyError:
            self.send_json({"error": f"Blob {blob_hash} not found"}, status=404)
            return

        mime = "application/octet-stream"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
            mime = "image/webp"
        elif data.startswith(b"%PDF"):
            mime = "application/pdf"
        elif b"<svg" in data[:1024].lower():
            mime = "image/svg+xml"
        elif b"\x00" not in data[:4096]:
            mime = "text/plain; charset=utf-8"

        try:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            if is_client_disconnect_exception(e):
                pass
            else:
                raise

    def _compute_change_line_stats(self, memory, change) -> tuple[int, int]:
        """Compute lines added and removed for a specific FileChange."""
        if isinstance(change, str):
            return 0, 0
        if isinstance(change, dict):
            action = change.get("action", "")
            path = change.get("path", "")
            blob_hash = change.get("blob_hash")
        else:
            action = getattr(change, "action", "")
            path = getattr(change, "path", "")
            blob_hash = getattr(change, "blob_hash", None)

        if action == "consulted":
            return 0, 0

        old_text = ""
        new_text = ""
        is_binary = False

        if action != "deleted" and blob_hash:
            try:
                raw_new = self.engine._workspace._blob_store.retrieve(blob_hash)
                if b"\x00" in raw_new[:2048]:
                    is_binary = True
                else:
                    new_text = raw_new.decode("utf-8", errors="ignore")
            except Exception:
                is_binary = True

        parent_id = getattr(memory, "parent_id", None)
        if action != "added" and parent_id:
            try:
                raw_old = self.engine.read_file_at_memory(path, parent_id)
                if b"\x00" in raw_old[:2048]:
                    is_binary = True
                else:
                    old_text = raw_old.decode("utf-8", errors="ignore")
            except Exception:
                pass

        lines_added = 0
        lines_removed = 0
        if not is_binary and (old_text or new_text):
            import difflib
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
                if line.startswith("+") and not line.startswith("+++"):
                    lines_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    lines_removed += 1

        return lines_added, lines_removed

    def _get_file_diff_and_stats(self, memory_id: str, file_path: str) -> dict:
        """Return unified diff structure for a given memory and file path."""
        try:
            memory = self.engine.get_memory(memory_id)
        except (KeyError, FileNotFoundError, AttributeError):
            return {"error": f"Memory {memory_id} not found"}

        change = None
        changes = getattr(memory, "changes", []) or []
        for c in changes:
            c_path = c if isinstance(c, str) else (c.get("path") if isinstance(c, dict) else getattr(c, "path", ""))
            if c_path == file_path or Path(c_path).name == Path(file_path).name:
                change = c
                break

        if not change:
            return {"error": f"File {file_path} not found in memory {memory_id}"}

        if isinstance(change, str):
            action = "modified"
            path = change
            blob_hash = None
        elif isinstance(change, dict):
            action = change.get("action", "modified")
            path = change.get("path", file_path)
            blob_hash = change.get("blob_hash")
        else:
            action = getattr(change, "action", "modified")
            path = getattr(change, "path", file_path)
            blob_hash = getattr(change, "blob_hash", None)

        if action == "consulted":
            content = None
            is_binary = False
            if blob_hash:
                try:
                    raw = self.engine._workspace._blob_store.retrieve(blob_hash)
                    if b"\x00" in raw[:4096]:
                        is_binary = True
                    else:
                        content = raw.decode("utf-8", errors="ignore")
                except Exception:
                    pass

            if content is None and not is_binary:
                try:
                    p = Path(path)
                    if p.exists() and p.is_file():
                        raw = p.read_bytes()
                        if b"\x00" in raw[:4096]:
                            is_binary = True
                        else:
                            content = raw.decode("utf-8", errors="ignore")
                except (FileNotFoundError, OSError, PermissionError):
                    content = None

            if is_binary:
                diff_text = "[Fichier binaire ou média]"
            elif content is not None:
                diff_text = content
            else:
                diff_text = "[Fichier non disponible]"

            return {
                "memory_id": memory_id,
                "path": path,
                "action": action,
                "blob_hash": blob_hash,
                "is_binary": is_binary,
                "lines_added": 0,
                "lines_removed": 0,
                "diff": diff_text,
            }

        old_text = ""
        new_text = ""
        is_binary = False

        if action != "deleted" and blob_hash:
            try:
                raw_new = self.engine._workspace._blob_store.retrieve(blob_hash)
                if b"\x00" in raw_new[:4096]:
                    is_binary = True
                else:
                    new_text = raw_new.decode("utf-8", errors="ignore")
            except Exception:
                is_binary = True

        parent_id = getattr(memory, "parent_id", None)
        if action != "added" and parent_id:
            try:
                raw_old = self.engine.read_file_at_memory(path, parent_id)
                if b"\x00" in raw_old[:4096]:
                    is_binary = True
                else:
                    old_text = raw_old.decode("utf-8", errors="ignore")
            except Exception:
                pass

        lines_added = 0
        lines_removed = 0
        diff_lines = []

        if not is_binary:
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            import difflib
            diff_generator = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
            diff_lines = list(diff_generator)
            for line in diff_lines:
                if line.startswith("+") and not line.startswith("+++"):
                    lines_added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    lines_removed += 1

        return {
            "memory_id": memory_id,
            "path": path,
            "action": action,
            "blob_hash": blob_hash,
            "is_binary": is_binary,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "diff": "\n".join(diff_lines) if not is_binary else "[Fichier binaire ou média]",
        }

    def _api_graph(self):
        """Return file nodes and co-occurrence edges."""
        edges = self.engine.get_file_cooccurrences()
        # Only send nodes that participate in at least one edge
        connected_files = set()
        for e in edges:
            connected_files.add(e["source"])
            connected_files.add(e["target"])
        nodes = self.engine.get_file_node_data(connected_files=connected_files)
        return {"nodes": nodes, "edges": edges}

    def _api_search(self, query: str):
        """Return search results enriched with line stats and change details."""
        if not query:
            return []
            
        results = self.engine.search(query, top_n=20)
        out = []
        for r in results:
            try:
                mem = self.engine.get_memory(r.memory_id)
                total_added = 0
                total_removed = 0
                changes_summary = []
                has_consulted = False
                has_modified = False
                changes = getattr(mem, "changes", []) or []
                for ch in changes:
                    added, removed = self._compute_change_line_stats(mem, ch)
                    total_added += added
                    total_removed += removed
                    if isinstance(ch, str):
                        action = "modified"
                        path = ch
                        blob_hash = None
                    elif isinstance(ch, dict):
                        action = ch.get("action", "modified")
                        path = ch.get("path", "")
                        blob_hash = ch.get("blob_hash")
                    else:
                        action = getattr(ch, "action", "modified")
                        path = getattr(ch, "path", "")
                        blob_hash = getattr(ch, "blob_hash", None)

                    if action == "consulted":
                        has_consulted = True
                    else:
                        has_modified = True
                    changes_summary.append({
                        "path": path,
                        "action": action,
                        "blob_hash": blob_hash,
                        "lines_added": added,
                        "lines_removed": removed,
                    })
            except Exception:
                mem = None
                total_added, total_removed = 0, 0
                changes_summary = []
                has_consulted, has_modified = False, True

            out.append({
                "memory_id": r.memory_id,
                "title": r.title,
                "timestamp": r.timestamp,
                "score": r.score,
                "snippet": r.snippet,
                "file_paths": r.file_paths,
                "note": mem.note if mem else "",
                "urls": getattr(mem, "urls", []) if mem else [],
                "parent_id": getattr(mem, "parent_id", None) if mem else None,
                "lines_added": total_added,
                "lines_removed": total_removed,
                "has_consulted": has_consulted,
                "has_modified": has_modified,
                "changes": changes_summary,
            })
        return out

    def _api_memory(self, memory_id: str):
        """Return full memory details enriched with blob hashes and line stats."""
        try:
            memory = self.engine.get_memory(memory_id)
            if not memory:
                return {"error": f"Memory {memory_id} not found"}
        except (KeyError, FileNotFoundError, AttributeError):
            return {"error": f"Memory {memory_id} not found"}

        changes_data = []
        total_added = 0
        total_removed = 0

        changes = getattr(memory, "changes", []) or []
        for c in changes:
            added, removed = self._compute_change_line_stats(memory, c)
            total_added += added
            total_removed += removed
            if isinstance(c, str):
                action = "modified"
                path = c
                blob_hash = None
                size_before = 0
                size_after = 0
            elif isinstance(c, dict):
                action = c.get("action", "modified")
                path = c.get("path", "")
                blob_hash = c.get("blob_hash")
                size_before = c.get("size_before", 0)
                size_after = c.get("size_after", 0)
            else:
                action = getattr(c, "action", "modified")
                path = getattr(c, "path", "")
                blob_hash = getattr(c, "blob_hash", None)
                size_before = getattr(c, "bytes_removed", 0)
                size_after = getattr(c, "bytes_added", 0)

            changes_data.append({
                "path": path,
                "action": action,
                "blob_hash": blob_hash,
                "size_before": size_before,
                "size_after": size_after,
                "lines_added": added,
                "lines_removed": removed,
            })

        return {
            "id": getattr(memory, "id", memory_id),
            "title": getattr(memory, "title", ""),
            "timestamp": getattr(memory, "timestamp", ""),
            "note": getattr(memory, "note", ""),
            "urls": getattr(memory, "urls", []),
            "parent_id": getattr(memory, "parent_id", None),
            "total_lines_added": total_added,
            "total_lines_removed": total_removed,
            "changes": changes_data,
        }

    def _api_log(self, offset: int = 0, limit: int = 10):
        """Return paginated memory log enriched with blob hashes and line stats."""
        memories = self.engine.get_log(limit=limit, offset=offset)
        out = []
        for c in memories:
            changes_summary = []
            total_added = 0
            total_removed = 0
            has_consulted = False
            has_modified = False

            changes = getattr(c, "changes", []) or []
            for ch in changes:
                added, removed = self._compute_change_line_stats(c, ch)
                total_added += added
                total_removed += removed
                if isinstance(ch, str):
                    action = "modified"
                    path = ch
                    blob_hash = None
                elif isinstance(ch, dict):
                    action = ch.get("action", "modified")
                    path = ch.get("path", "")
                    blob_hash = ch.get("blob_hash")
                else:
                    action = getattr(ch, "action", "modified")
                    path = getattr(ch, "path", "")
                    blob_hash = getattr(ch, "blob_hash", None)

                if action == "consulted":
                    has_consulted = True
                else:
                    has_modified = True
                changes_summary.append({
                    "path": path,
                    "action": action,
                    "blob_hash": blob_hash,
                    "lines_added": added,
                    "lines_removed": removed,
                })

            out.append({
                "id": getattr(c, "id", ""),
                "title": getattr(c, "title", ""),
                "timestamp": getattr(c, "timestamp", ""),
                "note": getattr(c, "note", ""),
                "urls": getattr(c, "urls", []),
                "parent_id": getattr(c, "parent_id", None),
                "file_count": len(changes),
                "lines_added": total_added,
                "lines_removed": total_removed,
                "has_consulted": has_consulted,
                "has_modified": has_modified,
                "changes": changes_summary,
            })
        return out

    def _api_file_history(self, file_path: str):
        """Return commit history for a specific file."""
        try:
            history = self.engine.get_file_history(file_path)
            return history
        except KeyError:
            return {"error": f"File {file_path} not found in history."}


def main():
    parser = argparse.ArgumentParser(description="AIVC Web Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default 8765)")
    args = parser.parse_args()

    from aivc.config import get_storage_root
    storage_root = get_storage_root(allow_fallback=True)

    print(f"Loading AIVC SemanticEngine from {storage_root} ...")
    engine = SemanticEngine(storage_root)

    def handler_factory(*args, **kwargs):
        return DashboardHandler(*args, engine=engine, **kwargs)

    port = args.port
    server = None
    for p in range(args.port, args.port + 20):
        try:
            server = DashboardServer(("0.0.0.0", p), handler_factory)
            port = p
            break
        except OSError as e:
            if getattr(e, "errno", 0) in (98, 10048) or "already in use" in str(e):
                continue
            raise

    if not server:
        print(f"[aivc] FATAL: Could not find an open port in range {args.port}-{args.port+19}", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Dashboard running at http://0.0.0.0:{port}/ (accessible via http://localhost:{port}/ or local IP)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
        server.server_close()


if __name__ == "__main__":
    main()

