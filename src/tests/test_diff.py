"""Unit tests for the diff engine."""

import pytest
from pathlib import Path
from aivc.core.blob_store import BlobStore
from aivc.core.diff import compute_diff


@pytest.fixture
def store(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "store")


def test_new_file_action_is_added(tmp_path: Path, store: BlobStore) -> None:
    f = tmp_path / "new_file.py"
    f.write_bytes(b"print('hello')")

    changes = compute_diff({str(f): None}, store)

    assert len(changes) == 1
    assert changes[0].action == "added"
    assert changes[0].blob_hash is not None
    assert changes[0].bytes_added == len(b"print('hello')")
    assert changes[0].bytes_removed == 0


def test_modified_file_action_is_modified(tmp_path: Path, store: BlobStore) -> None:
    f = tmp_path / "file.py"
    original_content = b"version 1"
    f.write_bytes(original_content)
    old_hash = store.store(original_content)

    f.write_bytes(b"version 2")
    changes = compute_diff({str(f): old_hash}, store)

    assert len(changes) == 1
    assert changes[0].action == "modified"
    assert changes[0].bytes_removed == len(original_content)
    assert changes[0].bytes_added == len(b"version 2")


def test_deleted_file_action_is_deleted(tmp_path: Path, store: BlobStore) -> None:
    data = b"was here"
    old_hash = store.store(data)
    ghost_path = tmp_path / "ghost.py"
    # File does NOT exist on disk.

    changes = compute_diff({str(ghost_path): old_hash}, store)

    assert len(changes) == 1
    assert changes[0].action == "deleted"
    assert changes[0].blob_hash is None
    assert changes[0].bytes_removed == len(data)
    assert changes[0].bytes_added == 0


def test_unchanged_file_produces_no_change(tmp_path: Path, store: BlobStore) -> None:
    content = b"stable content"
    f = tmp_path / "stable.py"
    f.write_bytes(content)
    known_hash = store.store(content)

    changes = compute_diff({str(f): known_hash}, store)

    assert changes == []


def test_directory_path_crashes(tmp_path: Path, store: BlobStore) -> None:
    d = tmp_path / "subdir"
    d.mkdir()

    with pytest.raises(IsADirectoryError):
        compute_diff({str(d): None}, store)


def test_multiple_files_mixed_changes(tmp_path: Path, store: BlobStore) -> None:
    # File A: new
    fa = tmp_path / "a.py"
    fa.write_bytes(b"new file")
    # File B: unchanged
    fb = tmp_path / "b.py"
    content_b = b"unchanged"
    fb.write_bytes(content_b)
    hash_b = store.store(content_b)
    # File C: deleted (does not exist)
    content_c = b"deleted"
    hash_c = store.store(content_c)

    tracked = {
        str(fa): None,
        str(fb): hash_b,
        str(tmp_path / "c.py"): hash_c,
    }
    changes = compute_diff(tracked, store)
    actions = {c.path: c.action for c in changes}

    assert actions[str(fa)] == "added"
    assert actions[str(tmp_path / "c.py")] == "deleted"
    assert str(fb) not in actions  # unchanged


def test_deleted_file_with_no_hash_is_ignored(tmp_path: Path, store: BlobStore) -> None:
    ghost_path = tmp_path / "ghost.py"
    # File does NOT exist on disk and was never committed (hash is None)

    changes = compute_diff({str(ghost_path): None}, store)

    assert changes == []

