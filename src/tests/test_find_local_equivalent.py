"""
Tests for SemanticEngine.find_local_equivalent:
- Prevents cross-UUID fallback (e.g. uuid1/scratch/sources_executant.md matching uuid2)
- Prevents fallback on generic parent directory names (scratch, artifacts, brain, etc.)
- Enforces strict blob-hash / path match.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.semantic.engine import SemanticEngine


def test_find_local_equivalent_prevents_cross_uuid_fallback(tmp_path: Path):
    storage_root = tmp_path / "storage"
    engine = SemanticEngine(storage_root)

    uuid1 = "11111111-1111-1111-1111-111111111111"
    uuid2 = "22222222-2222-2222-2222-222222222222"

    local_path2 = f"~/.gemini/antigravity/brain/{uuid2}/scratch/sources_executant.md"
    remote_path1 = f"~/.gemini/antigravity/brain/{uuid1}/scratch/sources_executant.md"

    # Mock tracked paths to contain uuid2
    engine._workspace.get_tracked_paths = MagicMock(return_value=[local_path2])
    engine._local_hints_index = None

    # Should NOT fall back across UUIDs even though both basename and parent name ("scratch") match
    res = engine.find_local_equivalent(remote_path1)
    assert res is None

    # Even if remote_blob_hash is provided, cross-UUID candidates must be skipped
    mock_index = MagicMock()
    mock_index.get_blob_hashes_for_file.return_value = {"blob123"}
    engine._workspace._index = mock_index

    res_with_blob = engine.find_local_equivalent(remote_path1, remote_blob_hash="blob123")
    assert res_with_blob is None

    # Matching same UUID should succeed with blob hash
    local_path1 = f"~/.gemini/antigravity/brain/{uuid1}/scratch/sources_executant.md"
    engine._workspace.get_tracked_paths = MagicMock(return_value=[local_path1, local_path2])
    engine._local_hints_index = None

    res_same_uuid = engine.find_local_equivalent(remote_path1, remote_blob_hash="blob123")
    assert res_same_uuid == local_path1


def test_find_local_equivalent_prevents_generic_parent_fallback(tmp_path: Path):
    storage_root = tmp_path / "storage"
    engine = SemanticEngine(storage_root)

    # Generic parent names like "scratch", "artifacts", "notes"
    local_path = "project_a/notes/summary.md"
    remote_path = "project_b/notes/summary.md"

    engine._workspace.get_tracked_paths = MagicMock(return_value=[local_path])
    engine._local_hints_index = None

    # Because "notes" is a generic parent name, parent matching must not be accepted
    assert engine.find_local_equivalent(remote_path) is None

    # Non-generic parent name should match
    local_specific = "modules/billing_engine/config.py"
    remote_specific = "other/billing_engine/config.py"
    engine._workspace.get_tracked_paths = MagicMock(return_value=[local_specific])
    engine._local_hints_index = None

    assert engine.find_local_equivalent(remote_specific) == local_specific
