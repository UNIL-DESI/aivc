import time
import json
from aivc.core.workspace import Workspace
from aivc.semantic.engine import SemanticEngine

def test_workspace_jit_reload(tmp_path):
    storage_root = tmp_path / "aivc"
    storage_root.mkdir()
    
    # 1. Initialize workspace
    ws = Workspace(storage_root)
    
    # Create a real file to track so len > 0
    tfile = tmp_path / "exists.txt"
    tfile.write_text("data")
    ws.create_memory("Initial", "Note", edited_files=[str(tfile)])
    
    original_paths = ws.get_tracked_paths()
    assert len(original_paths) > 0
    
    # 2. Simulate external modification (CLI)
    workspace_json = storage_root / "workspace.json"
    state = json.loads(workspace_json.read_text())
    
    # Add a fake file to tracking directly on disk
    fake_file = "/tmp/fake_file.txt"
    state["tracked_files"][fake_file] = None
    
    # Wait to ensure mtime changes
    time.sleep(0.1) 
    
    workspace_json.write_text(json.dumps(state))
    
    # 3. Check JIT reload
    new_paths = ws.get_tracked_paths()
    assert fake_file in new_paths
    assert len(new_paths) == len(original_paths) + 1

def test_semantic_engine_cache_invalidation_on_reload(tmp_path):
    storage_root = tmp_path / "aivc"
    storage_root.mkdir()
    
    # Create a real file to track
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    
    engine = SemanticEngine(storage_root)
    engine.create_memory("Initial", "Note", edited_files=[str(test_file)])
    
    # Trigger lazy build of hints index
    hints = engine._get_local_hints_index()
    assert "test.txt" in hints
    
    # Simulate external modification
    workspace_json = storage_root / "workspace.json"
    state = json.loads(workspace_json.read_text())
    
    fake_file_path = str(tmp_path / "fake.txt")
    state["tracked_files"][fake_file_path] = None
    
    # Wait to ensure mtime changes
    time.sleep(0.1)
    workspace_json.write_text(json.dumps(state))
    
    # Accessing workspace through engine should trigger reload and callback
    # get_status calls _reload_state_if_needed in Workspace
    engine.get_status()
    
    # The cache should have been invalidated by the callback
    assert engine._local_hints_index is None
    
    # Re-trigger lazy build
    new_hints = engine._get_local_hints_index()
    assert "fake.txt" in new_hints