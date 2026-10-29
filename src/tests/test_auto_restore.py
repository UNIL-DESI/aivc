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


def test_aivc_artifacts_portability_and_mirror(tmp_path: Path, monkeypatch):
    """Verify aivc/artifacts paths are recognized as portable and mirrored to temp."""
    import tempfile
    from aivc.config import extract_brain_subpath, to_portable_path

    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    monkeypatch.setenv("AIVC_TEMP_DIR", str(tmp_path / "custom_temp"))

    # 1. extract_brain_subpath
    uri = "file:///C:/Users/someone/AppData/Local/Temp/aivc/artifacts/conv-123/scratch/plan.md"
    assert extract_brain_subpath(uri) == "conv-123/scratch/plan.md"

    # 2. to_portable_path
    native = r"C:\Users\someone\AppData\Local\Temp\aivc\artifacts\conv-123\scratch\plan.md"
    portable = to_portable_path(native)
    assert portable == "~/.gemini/antigravity/brain/conv-123/scratch/plan.md"

    # 3. auto_restore mirrors to temp
    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)
    content = b"# Artifact Test"
    blob_hash = ws._blob_store.store(content)

    mem = Memory(
        id="mem-art-1",
        timestamp="2026-10-27T10:00:00Z",
        title="Artifact Commit",
        note="Note with file:///tmp/aivc/artifacts/conv-art/doc.md",
        parent_id=None,
        changes=[
            FileChange(
                path=native,
                action="added",
                blob_hash=blob_hash,
                bytes_added=len(content),
                bytes_removed=0,
            )
        ],
    )
    ws.auto_restore_brain_files(mem)
    expected_temp = tmp_path / "custom_temp" / "artifacts" / "conv-123" / "scratch" / "plan.md"
    assert expected_temp.is_file()
    assert expected_temp.read_bytes() == content


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


def test_auto_restore_brain_files_from_user_items_and_links(tmp_path: Path, monkeypatch):
    """Verify brain files linked in user_items details or journal (even foreign user paths) are restored."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    expected_local = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-other" / "plan.md"
    assert not expected_local.exists()

    foreign_link = "file:///C:/Users/other_user/.gemini/antigravity/brain/conv-other/plan.md"
    mem = Memory(
        id="mem-link-test",
        timestamp="2026-10-28T10:00:00Z",
        title="Deliverable with brain link",
        note="Some note",
        parent_id=None,
        user_items=[
            {
                "id": "deliv-1",
                "task": "Review plan",
                "details": f"See artifact [Plan]({foreign_link})",
                "task_status": "await_validation",
            }
        ],
    )

    mock_sync = MagicMock()
    mock_sync.enabled = True

    def fake_pull(file_path, target_path=None):
        target = Path(target_path) if target_path else from_portable_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# Restored Plan Content")
        return target

    mock_sync.pull_latest_file.side_effect = fake_pull

    restored = ws.auto_restore_brain_files(mem, sync_manager=mock_sync)
    assert str(expected_local) in restored
    assert expected_local.is_file()
    assert expected_local.read_bytes() == b"# Restored Plan Content"


def test_push_latest_file_exclusions(tmp_path: Path, monkeypatch):
    """Verify push_latest_file ignores .git, node_modules, >25MB files, and handles absent Drive gracefully."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    from aivc.sync.drive import NativeDriveSyncManager

    mgr = NativeDriveSyncManager(tmp_path / "storage")
    mgr.enabled = True

    # 1. File inside .git folder
    git_file = tmp_path / ".git" / "config"
    git_file.parent.mkdir(parents=True, exist_ok=True)
    git_file.write_bytes(b"git config")
    assert mgr.push_latest_file(git_file) is False

    # 2. File inside node_modules
    nm_file = tmp_path / "node_modules" / "pkg" / "index.js"
    nm_file.parent.mkdir(parents=True, exist_ok=True)
    nm_file.write_bytes(b"exports = {}")
    assert mgr.push_latest_file(nm_file) is False

    # 3. File > 25 MB
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"0" * 1024)
    # Mock stat to simulate > 25MB
    mock_stat = MagicMock()
    mock_stat.st_size = 26 * 1024 * 1024
    monkeypatch.setattr(Path, "stat", lambda self: mock_stat if self == large_file else Path.stat(self))
    assert mgr.push_latest_file(large_file) is False

    # 4. Absent / unauthenticated Drive does not crash and returns False
    normal_file = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-x" / "test.md"
    normal_file.parent.mkdir(parents=True, exist_ok=True)
    normal_file.write_bytes(b"content")
    # Will fail authentication gracefully
    res = mgr.push_latest_file(normal_file)
    assert res is False


def test_advance_sync_auto_restore_brain_files(tmp_path: Path, monkeypatch):
    """Verify sync_advance_tldr and restore_brain_files_for_memory restore brain files on new PC."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)
    from aivc.advance_sync import restore_brain_files_for_memory, sync_advance_tldr

    brain_file = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-adv" / "report.md"
    assert not brain_file.exists()

    foreign_path = r"C:\Users\hjamet\.gemini\antigravity\brain\conv-adv\report.md"
    mem = Memory(
        id="mem-adv-1",
        timestamp="2026-10-28T10:00:00Z",
        title="[Advance] Test Project",
        note="Note with report link",
        parent_id=None,
        changes=[FileChange(path=foreign_path, action="edited")],
        user_items=[{"id": "t-1", "task": "Check report", "details": foreign_path, "task_status": "await_validation"}],
    )


    mock_sync = MagicMock()
    mock_sync.enabled = True

    def fake_pull(file_path, target_path=None):
        target = Path(target_path) if target_path else from_portable_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# Restored Report Content")
        return target

    mock_sync.pull_latest_file.side_effect = fake_pull

    restored = restore_brain_files_for_memory(mem, sync_manager=mock_sync)
    assert str(brain_file) in restored
    assert brain_file.is_file()
    assert brain_file.read_bytes() == b"# Restored Report Content"


def test_auto_restore_brain_files_background_drive(tmp_path: Path, monkeypatch):
    """Verify background_drive=True returns immediately while Drive pulls complete asynchronously."""
    import time
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    expected_local = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-bg" / "slow_file.md"
    assert not expected_local.exists()

    mem = Memory(
        id="mem-bg-test",
        timestamp="2026-10-29T12:00:00Z",
        title="Async Background Test",
        note="Note",
        parent_id=None,
        changes=[FileChange(path=str(expected_local), action="edited")],
    )

    mock_sync = MagicMock()
    mock_sync.enabled = True

    def slow_pull(file_path, target_path=None):
        time.sleep(0.05)
        target = Path(target_path) if target_path else from_portable_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"# Restored Async Content")
        return target

    mock_sync.pull_latest_file.side_effect = slow_pull

    # Call with background_drive=True -> must return immediately without blocking
    start_t = time.time()
    restored = ws.auto_restore_brain_files(mem, sync_manager=mock_sync, background_drive=True)
    duration = time.time() - start_t

    # Immediate return: synchronous list is empty, duration is minimal
    assert restored == []
    assert duration < 0.04
    assert hasattr(ws, "_last_restore_thread")
    assert ws._last_restore_thread.is_alive()

    # Wait for background thread to complete
    ws._last_restore_thread.join(timeout=2.0)
    assert not ws._last_restore_thread.is_alive()

    # File is now restored on disk
    assert expected_local.is_file()
    assert expected_local.read_bytes() == b"# Restored Async Content"


def test_auto_restore_unquotes_urls_and_skips_directories(tmp_path: Path, monkeypatch):
    """Verify auto_restore handles %20 URL decoding and skips existing directories / extensionless references."""
    monkeypatch.setattr("aivc.config.get_home_dir", lambda: tmp_path)

    storage_root = tmp_path / ".aivc" / "storage"
    ws = Workspace(storage_root)

    # 1. An existing directory on disk
    dir_path = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-unquote" / "subfolder"
    dir_path.mkdir(parents=True, exist_ok=True)

    # 2. File with encoded spaces in URL
    file_with_spaces = tmp_path / ".gemini" / "antigravity" / "brain" / "conv-unquote" / "Doc With Spaces.md"
    content = b"# Spaces Content"
    blob_hash = ws._blob_store.store(content)

    mem = Memory(
        id="mem-unquote-test",
        timestamp="2026-10-30T00:00:00Z",
        title="Unquote test",
        note="Refers to file:///C:/Users/test/.gemini/antigravity/brain/conv-unquote/Doc%20With%20Spaces.md and subfolder/",
        parent_id=None,
        changes=[
            FileChange(
                path=str(file_with_spaces),
                action="added",
                blob_hash=blob_hash,
                bytes_added=len(content),
                bytes_removed=0,
            )
        ],
        user_items=[
            {
                "id": "item-1",
                "task": "Task with dir",
                "details": "See file:///C:/Users/test/.gemini/antigravity/brain/conv-unquote/subfolder and non_existent_folder",
                "task_status": "await_validation",
            }
        ],
    )

    restored = ws.auto_restore_brain_files(mem)
    assert str(file_with_spaces) in restored
    assert file_with_spaces.is_file()
    assert file_with_spaces.read_bytes() == content
    # Ensure directory is still a directory and not overwritten with blob
    assert dir_path.is_dir()



