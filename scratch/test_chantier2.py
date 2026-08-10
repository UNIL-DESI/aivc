import tempfile
from pathlib import Path
import unittest
import asyncio

from aivc.core.memory import Memory, FileChange
from aivc.semantic.engine import SemanticEngine
from aivc.server import remember, consult_memory, _get_engine
import aivc.server
from aivc.web.dashboard import DashboardHandler


class TestChantier2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temp_dir.name)
        # Patch server engine
        aivc.server._storage_root = self.storage_root
        aivc.server._engine = SemanticEngine(self.storage_root)
        self.engine = aivc.server._engine

    def tearDown(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.shutdown()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_remember_and_consult_memory_with_urls(self):
        # Create a test file
        test_file = self.storage_root / "test_doc.txt"
        test_file.write_text("Hello World\n", encoding="utf-8")

        # Call remember tool
        res = asyncio.run(
            remember(
                title="Test Memory Title",
                note="Detailed test note.",
                read_files=[str(test_file)],
                urls=["https://example.com/spec", "https://docs.aivc.io"],
            )
        )
        self.assertIn("Memory successfully created", res)

        # Retrieve created memory ID
        recent = self.engine.get_log(limit=1)
        self.assertEqual(len(recent), 1)
        mem = recent[0]
        self.assertEqual(mem.urls, ["https://example.com/spec", "https://docs.aivc.io"])

        # Call consult_memory
        consult_res = consult_memory(mem.id)
        self.assertIn("## URLs / Links Consulted", consult_res)
        self.assertIn("- https://example.com/spec", consult_res)
        self.assertIn("- https://docs.aivc.io", consult_res)

    def test_dashboard_urls_and_consulted_diff(self):
        test_file = self.storage_root / "consulted_doc.txt"
        test_file.write_text("Consulted content line 1\nLine 2\n", encoding="utf-8")

        res = asyncio.run(
            remember(
                title="Consulted Test Memory",
                note="Testing consulted file diff.",
                read_files=[str(test_file)],
                urls=["https://aivc.dev"],
            )
        )
        mem = self.engine.get_log(limit=1)[0]

        handler = DashboardHandler.__new__(DashboardHandler)
        handler.engine = self.engine

        # Check _api_memory
        api_mem = handler._api_memory(mem.id)
        self.assertEqual(api_mem["urls"], ["https://aivc.dev"])

        # Check _api_log
        api_log = handler._api_log(limit=1)
        self.assertEqual(api_log[0]["urls"], ["https://aivc.dev"])

        # Check _get_file_diff_and_stats for consulted file
        diff_info = handler._get_file_diff_and_stats(mem.id, str(test_file))
        self.assertEqual(diff_info["action"], "consulted")
        self.assertEqual(diff_info["lines_added"], 0)
        self.assertEqual(diff_info["lines_removed"], 0)
        self.assertIn("Consulted content line 1", diff_info["diff"])
        self.assertNotIn("--- a/", diff_info["diff"])
        self.assertNotIn("+++ b/", diff_info["diff"])

        # Check FileNotFoundError fallback
        missing_diff = handler._get_file_diff_and_stats(mem.id, "non_existent_file.txt")
        self.assertIn("error", missing_diff)


if __name__ == "__main__":
    unittest.main()
