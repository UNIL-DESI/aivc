import os
import pytest
from pathlib import Path
from aivc.config import get_storage_root
from aivc.core.memory import Memory, FileChange, memory_to_dict, memory_from_dict
from aivc.core.workspace import Workspace
from aivc.semantic.engine import SemanticEngine

def test_config_storage_root(tmp_path):
    env_var = "AIVC_STORAGE_ROOT"
    old_val = os.environ.get(env_var)
    
    try:
        # Test 1: Env var set
        os.environ[env_var] = str(tmp_path)
        assert get_storage_root() == tmp_path
        
        # Test 2: Env var missing, no fallback
        del os.environ[env_var]
        with pytest.raises(SystemExit):
            get_storage_root(allow_fallback=False)
            
        # Test 3: Env var missing, with fallback
        # (Fallbacks to ~/.aivc/storage by default)
        root = get_storage_root(allow_fallback=True)
        assert root == Path.home() / ".aivc" / "storage"
        
    finally:
        if old_val:
            os.environ[env_var] = old_val
        elif env_var in os.environ:
            del os.environ[env_var]
            
def test_consulted_file_change_validation():
    # Valid consulted change
    fc = FileChange(path="foo.txt", action="consulted", blob_hash=None, bytes_added=0, bytes_removed=0)
    assert fc.action == "consulted"
    
    # Invalid: consulted with blob_hash
    with pytest.raises(ValueError, match="must have blob_hash=None"):
        FileChange(path="foo.txt", action="consulted", blob_hash="abc", bytes_added=0, bytes_removed=0)

def test_memory_serialization_with_consulted():
    fc = FileChange(path="foo.txt", action="consulted", blob_hash=None, bytes_added=0, bytes_removed=0)
    m = Memory.create(title="Test", note="Note", parent_id=None, changes=[fc])
    
    data = memory_to_dict(m)
    assert data["changes"][0]["action"] == "consulted"
    
    m2 = memory_from_dict(data)
    assert m2.changes[0].action == "consulted"
    assert m2.changes[0].path == "foo.txt"
    # Actually Memory.create doesn't resolve paths, Workspace does.

def test_workspace_create_memory_only_consulted(tmp_path):
    ws = Workspace(tmp_path)
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")
    
    ws.track(str(file_a))
    # First memory to have a clean state
    ws.create_memory("Initial", "Note")
    
    # Now create a memory with ONLY consulted file
    # We need to make sure compute_diff returns empty
    memory = ws.create_memory("Consulted Only", "Note", consulted_files=[str(file_a)])
    
    assert len(memory.changes) == 1
    assert memory.changes[0].action == "consulted"
    assert memory.changes[0].path == str(file_a.resolve())

def test_workspace_create_memory_mixed(tmp_path):
    ws = Workspace(tmp_path)
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello")
    file_b.write_text("world")
    
    ws.track(str(file_a))
    ws.track(str(file_b))
    
    # Memory b once so it's not "added" anymore
    ws.create_memory("Setup", "Note")
    
    # Now modify a and consult b
    file_a.write_text("modified")
    memory = ws.create_memory("Mixed", "Note", consulted_files=[str(file_b)])
    
    actions = {c.path: c.action for c in memory.changes}
    assert actions[str(file_a.resolve())] == "modified"
    assert actions[str(file_b.resolve())] == "consulted"
    assert len(memory.changes) == 2

def test_workspace_create_memory_autotrack_consulted(tmp_path):
    """Consulted files that exist on disk but aren't tracked should be auto-tracked."""
    ws = Workspace(tmp_path)
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")
    
    # file_a is NOT tracked — create_memory should auto-track it
    memory = ws.create_memory("AutoTrack", "Note", consulted_files=[str(file_a)])
    assert len(memory.changes) == 1
    assert memory.changes[0].action == "consulted"
    assert memory.changes[0].path == str(file_a.resolve())


def test_workspace_create_memory_nonexistent_consulted(tmp_path):
    """Consulted files that don't exist on disk and are untracked should raise ValueError."""
    ws = Workspace(tmp_path)
    fake_path = tmp_path / "does_not_exist.txt"
    
    with pytest.raises(ValueError, match="Validation error in read_files"):
        ws.create_memory("Skip", "Note", consulted_files=[str(fake_path)])

def test_workspace_create_memory_strict_validation(tmp_path):
    ws = Workspace(tmp_path)
    # 1. Test directory raises ValueError
    dir_path = tmp_path / "subdir"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        ws.create_memory("Title", "Note", read_files=[str(dir_path)])
    with pytest.raises(ValueError, match="is a directory"):
        ws.create_memory("Title", "Note", edited_files=[str(dir_path)])

    # 2. Test untracked non-existent file raises ValueError
    fake_file = tmp_path / "does_not_exist.txt"
    with pytest.raises(ValueError, match="is an untracked non-existent file"):
        ws.create_memory("Title", "Note", read_files=[str(fake_file)])
    with pytest.raises(ValueError, match="is an untracked non-existent file"):
        ws.create_memory("Title", "Note", edited_files=[str(fake_file)])

    # 3. Test tracked non-existent file (deleted file) does NOT raise ValueError
    real_file = tmp_path / "real.txt"
    real_file.write_text("hello")
    ws.track(str(real_file))
    ws.create_memory("Initial", "Note")
    real_file.unlink() # Delete it
    # Since it is tracked, it should not raise ValueError
    memory = ws.create_memory("Deleted", "Note")
    assert any(c.path == str(real_file.resolve()) and c.action == "deleted" for c in memory.changes)

def test_semantic_engine_graph_updates(tmp_path):
    engine = SemanticEngine(tmp_path)
    file_a = tmp_path / "a.txt"
    file_a.write_text("hello")
    
    engine.track(str(file_a))
    # Memory with only consulted file
    memory = engine.create_memory("Consulted Only", "Note", consulted_files=[str(file_a)])
    
    # Check graph
    files = engine.get_memory_files(memory.id)
    assert str(file_a.resolve()) in files
    
    memories = engine.get_file_memories(str(file_a.resolve()))
    assert memory.id in memories
