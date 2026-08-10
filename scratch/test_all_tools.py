import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add src/ directory to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from aivc.semantic.engine import SemanticEngine
from aivc.core.workspace import Workspace

def run_test():
    print("=== AIVC End-to-End Tools Integration Test ===")
    
    # 1. Create a clean temporary directory to act as a workspace/storage
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    print(f"Created temporary directory: {temp_path}")
    
    workspace_dir = temp_path / "workspace"
    storage_dir = temp_path / "storage"
    workspace_dir.mkdir()
    storage_dir.mkdir()
    
    # Set the storage root env var so that the engine/config targets it
    os.environ["AIVC_STORAGE_ROOT"] = str(storage_dir)
    
    try:
        # 2. Initialize AIVC SemanticEngine and get Workspace
        print("Initializing AIVC SemanticEngine...")
        engine = SemanticEngine(storage_dir)
        workspace = engine._workspace
        
        # 3. Create a test file with initial content
        test_file_path = workspace_dir / "test_file.txt"
        initial_content = "Hello AIVC integration test!\nInitial content here."
        test_file_path.write_text(initial_content, encoding="utf-8", newline="")
        print(f"Created test file at: {test_file_path}")
        
        # 4. Call workspace.create_memory with edited_files (simulating remember)
        print("Creating first memory (Initial)...")
        m1 = workspace.create_memory(
            title="Initial Memory",
            note="## Initial\nCreating the file with some initial content.",
            edited_files=[str(test_file_path)]
        )
        print(f"First memory created successfully. ID: {m1.id}")
        
        # Verify it gets auto-tracked
        tracked_paths = workspace.get_tracked_paths()
        abs_test_file_path = str(test_file_path.resolve())
        print(f"Tracked paths: {tracked_paths}")
        assert abs_test_file_path in tracked_paths, "File should be automatically tracked!"
        print("Auto-tracking verification passed!")
        
        # Record blob hash from m1
        assert len(m1.changes) == 1
        blob_hash_1 = m1.changes[0].blob_hash
        assert blob_hash_1 is not None
        print(f"Blob 1 Hash: {blob_hash_1}")
        
        # 5. Modify test_file.txt with new content and call create_memory again
        new_content = "Hello AIVC integration test!\nNew content has replaced the initial content."
        test_file_path.write_text(new_content, encoding="utf-8", newline="")
        print("Modifying test file content...")
        
        print("Creating second memory (Modify)...")
        m2 = workspace.create_memory(
            title="Second Memory",
            note="## Modify\nUpdating the file content.",
            edited_files=[str(test_file_path)]
        )
        print(f"Second memory created successfully. ID: {m2.id}")
        assert len(m2.changes) == 1
        blob_hash_2 = m2.changes[0].blob_hash
        assert blob_hash_2 is not None
        print(f"Blob 2 Hash: {blob_hash_2}")
        
        # Manually register memories to co-occurrence graph database to back engine.get_file_memories
        # (This avoids loading heavy ML modules / ChromaDB / sentence-transformers)
        print("Registering memories in co-occurrence graph...")
        engine._graph.add_memory(m1)
        engine._graph.add_memory(m2)
        
        # 6. Call engine.get_file_memories and verify it lists memories
        print("Checking engine.get_file_memories...")
        memories = engine.get_file_memories(abs_test_file_path)
        print(f"Memories found for file: {memories}")
        assert len(memories) == 2, f"Expected 2 memories, got {len(memories)}"
        assert m1.id in memories, "Memory 1 ID missing"
        assert m2.id in memories, "Memory 2 ID missing"
        print("get_file_memories verification passed!")
        
        # 7. Call workspace.read_file_at_memory to read content at first memory
        print("Checking workspace.read_file_at_memory at first memory...")
        content_m1 = workspace.read_file_at_memory(abs_test_file_path, m1.id)
        retrieved_str = content_m1.decode("utf-8").replace("\r\n", "\n")
        print(f"Retrieved content at first memory: {retrieved_str!r}")
        assert retrieved_str == initial_content, "Retrieved content does not match initial content!"
        print("read_file_at_memory verification passed!")
        
        # 8. Delete test_file.txt and call workspace.create_memory with it in edited_files
        print("Deleting test file from disk...")
        test_file_path.unlink()
        
        print("Creating third memory (Delete)...")
        m3 = workspace.create_memory(
            title="Delete Memory",
            note="## Delete\nDeleted the test file.",
            edited_files=[str(test_file_path)]
        )
        print(f"Third memory created successfully. ID: {m3.id}")
        
        # 9. Verify the file is no longer in tracked_files and that its old blobs are garbage collected
        tracked_paths = workspace.get_tracked_paths()
        print(f"Tracked paths after deletion: {tracked_paths}")
        assert abs_test_file_path not in tracked_paths, "File should no longer be tracked!"
        
        blob_path_1 = storage_dir / "blobs" / blob_hash_1
        blob_path_2 = storage_dir / "blobs" / blob_hash_2
        assert not blob_path_1.exists(), f"Blob 1 {blob_hash_1} still exists on disk!"
        assert not blob_path_2.exists(), f"Blob 2 {blob_hash_2} still exists on disk!"
        print("Verified blob files are deleted from disk!")
        
        refcounts = workspace._blob_store._load_refcounts()
        print(f"Remaining refcounts: {refcounts}")
        assert blob_hash_1 not in refcounts, "Blob 1 hash still in refcounts!"
        assert blob_hash_2 not in refcounts, "Blob 2 hash still in refcounts!"
        print("Verified blob hashes are removed from refcounts!")
        
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        
    finally:
        # Shutdown engine and close SQLite databases cleanly
        print("Cleaning up database connections and worker threads...")
        engine.shutdown()
        workspace._index.close()
        engine._graph.close()
        
        # Clean up temporary directory
        try:
            temp_dir.cleanup()
            print("Temporary directory cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

if __name__ == "__main__":
    run_test()
