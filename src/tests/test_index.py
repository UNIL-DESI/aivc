"""Unit tests for CoreIndex SQLite."""

import pytest
from pathlib import Path
from aivc.core.index import CoreIndex
from aivc.core.memory import Memory, FileChange

@pytest.fixture
def index(tmp_path: Path) -> CoreIndex:
    idx = CoreIndex(tmp_path / "index_storage")
    yield idx
    idx.close()

def test_add_memory_inserts_metadata(index: CoreIndex) -> None:
    fc = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    memory = Memory.create(title="Init", note="Note", parent_id=None, changes=[fc])
    
    index.add_memory(memory)
    
    # Verify metadata via find_child (testing lookup)
    # Since it's the first memory, it has no parent. Let's check with a second memory.
    fc2 = FileChange(path="b.py", action="added", blob_hash="h2", bytes_added=20, bytes_removed=0)
    memory2 = Memory.create(title="Next", note="Note 2", parent_id=memory.id, changes=[fc2])
    index.add_memory(memory2)
    
    child = index.find_child(memory.id)
    assert child is not None
    assert child[0] == memory2.id
    assert child[1] == "Next"

def test_add_memory_is_idempotent(index: CoreIndex) -> None:
    fc = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    memory = Memory.create(title="Init", note="Note", parent_id=None, changes=[fc])
    
    index.add_memory(memory)
    index.add_memory(memory)  # Should not raise
    
    hashes = index.get_blob_hashes_for_file("a.py")
    assert hashes == {"h1"}

def test_get_blob_hashes_for_file(index: CoreIndex) -> None:
    fc1 = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    m1 = Memory.create(title="m1", note="n1", parent_id=None, changes=[fc1])
    index.add_memory(m1)
    
    fc2 = FileChange(path="a.py", action="modified", blob_hash="h2", bytes_added=15, bytes_removed=10)
    m2 = Memory.create(title="m2", note="n2", parent_id=m1.id, changes=[fc2])
    index.add_memory(m2)
    
    hashes = index.get_blob_hashes_for_file("a.py")
    assert hashes == {"h1", "h2"}

def test_remove_file_changes(index: CoreIndex) -> None:
    fc = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    memory = Memory.create(title="Init", note="Note", parent_id=None, changes=[fc])
    index.add_memory(memory)
    
    assert index.get_blob_hashes_for_file("a.py") == {"h1"}
    index.remove_file_changes("a.py")
    assert index.get_blob_hashes_for_file("a.py") == set()

def test_get_memories_touching_file(index: CoreIndex) -> None:
    fc = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    memory = Memory.create(title="Init", note="Note", parent_id=None, changes=[fc])
    index.add_memory(memory)
    
    memories = index.get_memories_touching_file("a.py")
    assert memories == [memory.id]

def test_migrate_from_json(tmp_path: Path, index: CoreIndex) -> None:
    from aivc.core.memory import memory_to_dict
    import json
    
    memories_dir = tmp_path / "commits" # Folder is still named 'commits' on disk
    memories_dir.mkdir()
    
    fc = FileChange(path="a.py", action="added", blob_hash="h1", bytes_added=10, bytes_removed=0)
    memory = Memory.create(title="JSON Memory", note="Note", parent_id=None, changes=[fc])
    
    (memories_dir / f"{memory.id}.json").write_text(json.dumps(memory_to_dict(memory)))
    
    count = index.migrate_from_json(memories_dir)
    assert count == 1
    
    assert index.get_memories_touching_file("a.py") == [memory.id]
    
    # Second run should skip
    count2 = index.migrate_from_json(memories_dir)
    assert count2 == 0


def test_get_blob_hashes_for_file_filtering(index: CoreIndex) -> None:
    fc1 = FileChange(path="src/aivc/file.py", action="added", blob_hash="hash_a", bytes_added=10, bytes_removed=0)
    fc2 = FileChange(path="other/not_file.py", action="added", blob_hash="hash_b", bytes_added=10, bytes_removed=0)
    fc3 = FileChange(path="prefix_file.py", action="added", blob_hash="hash_c", bytes_added=10, bytes_removed=0)
    m = Memory.create(title="Multi", note="Note", parent_id=None, changes=[fc1, fc2, fc3])
    index.add_memory(m)

    assert index.get_blob_hashes_for_file("src/aivc/file.py") == {"hash_a"}
    assert index.get_blob_hashes_for_file("prefix_file.py") == {"hash_c"}
    assert index.get_blob_hashes_for_file("file.py") == set()
    assert index.get_blob_hashes_for_file("") == set()

