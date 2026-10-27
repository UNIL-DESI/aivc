"""
Tests for brain files auto-restoration and path normalization.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from aivc.config import to_portable_path, from_portable_path, get_home_dir
from aivc.core.workspace import Workspace
from aivc.core.memory import Memory, FileChange


def test_path_normalization_missing_slash_gemini():
    """Verify paths with missing slash before .gemini are properly normalized."""
    bad_windows = r"C:\Users\Jamet.gemini\antigravity\brain\123\file.md"
    bad_posix = "C:/Users/Jamet.gemini/antigravity/brain/123/file.md"
    bad_unix = "/Users/Jamet.gemini/antigravity/brain/123/file.md"

    p1 = to_portable_path(bad_windows)
    assert p1 == "~/.gemini/antigravity/brain/123/file.md"

    p2 = to_portable_path(bad_posix)
    assert p2 == "~/.gemini/antigravity/brain/123/file.md"

    p3 = to_portable_path(bad_unix)
    assert p3 == "~/.gemini/antigravity/brain/123/file.md"

    local = from_portable_path(bad_windows)
    home = get_home_dir()
    expected = home / ".gemini" / "antigravity" / "brain" / "123" / "file.md"
    assert local == expected


def test_auto_restore_brain_files_from_blob_store(tmp_path: Path, monkeypatch):
    """Verify auto-restoration restores missing brain files from blob_store."""
    # Mock home to tmp_path
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    # Store a blob in ws._blob_store
    content = b"# Virtual Draft Title\nThis is content."
    blob_hash = ws._blob_store.store(content)

    brain_file_path = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-1" / "draft.md"
    assert not brain_file_path.exists()

    # Memory referencing this file
    fc = FileChange(
        path=str(brain_file_path),
        action="edited",
        blob_hash=blob_hash,
        bytes_added=len(content),
        bytes_removed=0,
    )
    mem = Memory(
        id="mem-test-1",
        timestamp="2026-10-27T10:00:00Z",
        title="Test Memory",
        note="Note",
        parent_id=None,
        changes=[fc],
    )

    restored = ws.auto_restore_brain_files(mem)
    assert str(brain_file_path) in restored
    assert brain_file_path.exists()
    assert brain_file_path.read_bytes() == content


def test_auto_restore_skips_existing_files(tmp_path: Path, monkeypatch):
    """Verify existing brain files are not overwritten."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    old_content = b"Existing local content"
    brain_file_path = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-2" / "existing.md"
    brain_file_path.parent.mkdir(parents=True, exist_ok=True)
    brain_file_path.write_bytes(old_content)

    new_content = b"New blob content"
    blob_hash = ws._blob_store.store(new_content)

    fc = FileChange(
        path=str(brain_file_path),
        action="edited",
        blob_hash=blob_hash,
        bytes_added=len(new_content),
        bytes_removed=0,
    )
    mem = Memory(
        id="mem-test-2",
        timestamp="2026-10-27T10:00:00Z",
        title="Test Memory 2",
        note="Note",
        parent_id=None,
        changes=[fc],
    )

    restored = ws.auto_restore_brain_files(mem)
    assert restored == []
    assert brain_file_path.read_bytes() == old_content


def test_auto_restore_skips_non_brain_files(tmp_path: Path, monkeypatch):
    """Verify files outside of brain/ are not restored by auto_restore_brain_files."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    code_file = tmp_path / "Documents" / "code" / "project" / "main.py"
    blob_hash = ws._blob_store.store(b"print('hello')")

    fc = FileChange(
        path=str(code_file),
        action="edited",
        blob_hash=blob_hash,
        bytes_added=14,
        bytes_removed=0,
    )
    mem = Memory(
        id="mem-test-3",
        timestamp="2026-10-27T10:00:00Z",
        title="Test Memory 3",
        note="Note",
        parent_id=None,
        changes=[fc],
    )

    restored = ws.auto_restore_brain_files(mem)
    assert restored == []
    assert not code_file.exists()


def test_auto_restore_fallback_to_index_blob(tmp_path: Path, monkeypatch):
    """Verify fallback to blob found in other commits when current change has blob_hash=None."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    content = b"# Document Content From Prior Commit"
    blob_hash = ws._blob_store.store(content)

    brain_file_path = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-fallback" / "guide.md"

    # Prior commit that registered the blob
    prior_mem = Memory(
        id="prior-commit-id",
        timestamp="2026-10-20T10:00:00Z",
        title="Prior Commit",
        note="Note",
        parent_id=None,
        changes=[
            FileChange(
                path=str(brain_file_path),
                action="added",
                blob_hash=blob_hash,
                bytes_added=len(content),
                bytes_removed=0,
            )
        ],
    )
    ws._index.add_memory(prior_mem)

    # Current memory with blob_hash=None
    current_mem = Memory(
        id="current-commit-id",
        timestamp="2026-10-27T10:00:00Z",
        title="Current Commit",
        note="Note",
        parent_id="prior-commit-id",
        changes=[
            FileChange(
                path=str(brain_file_path),
                action="edited",
                blob_hash=None,
                bytes_added=0,
                bytes_removed=0,
            )
        ],
    )

    assert not brain_file_path.exists()
    restored = ws.auto_restore_brain_files(current_mem)
    assert str(brain_file_path) in restored
    assert brain_file_path.exists()
    assert brain_file_path.read_bytes() == content


def test_path_normalization_tilde_concatenation():
    """Verify paths with ~ or \\~\\ segments are resolved to home."""
    home = get_home_dir()
    bad_concat = r"C:\Users\hjamet\Documents\VoiceNotes\~\.gemini\antigravity\brain\conv-1\test.md"
    portable = to_portable_path(bad_concat)
    assert portable == "~/.gemini/antigravity/brain/conv-1/test.md"

    resolved = from_portable_path(bad_concat)
    assert resolved == home / ".gemini" / "antigravity" / "brain" / "conv-1" / "test.md"

    # Also test with ~\.gemini
    win_tilde = r"~\.gemini\antigravity\brain\conv-1\test.md"
    portable_win = to_portable_path(win_tilde)
    assert portable_win == "~/.gemini/antigravity/brain/conv-1/test.md"
    assert from_portable_path(win_tilde) == home / ".gemini" / "antigravity" / "brain" / "conv-1" / "test.md"


def test_auto_restore_negative_cache(tmp_path: Path, monkeypatch):
    """Verify negative cache prevents repeated Google Drive calls for missing files."""
    import json
    import time
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    brain_file_path = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-neg" / "missing.md"
    fc = FileChange(
        path=str(brain_file_path),
        action="edited",
        blob_hash=None,
        bytes_added=0,
        bytes_removed=0,
    )
    mem = Memory(
        id="mem-neg-test",
        timestamp="2026-10-27T10:00:00Z",
        title="Negative Cache Test",
        note="Note",
        parent_id=None,
        changes=[fc],
    )

    mock_sync = MagicMock()
    mock_sync.enabled = True
    mock_sync.pull_latest_file.return_value = None  # File not found on Drive

    # 1. First attempt: calls pull_latest_file and populates negative cache
    restored1 = ws.auto_restore_brain_files(mem, sync_manager=mock_sync)
    assert restored1 == []
    assert mock_sync.pull_latest_file.call_count == 1

    cache_file = storage_root / "missing_blobs_cache.json"
    assert cache_file.exists()
    cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
    portable_key = "~/.gemini/antigravity/brain/conv-neg/missing.md"
    assert portable_key in cache_data
    assert isinstance(cache_data[portable_key], (int, float))

    # 2. Second attempt: uses negative cache and does NOT call pull_latest_file again
    mock_sync.pull_latest_file.reset_mock()
    restored2 = ws.auto_restore_brain_files(mem, sync_manager=mock_sync)
    assert restored2 == []
    assert mock_sync.pull_latest_file.call_count == 0

