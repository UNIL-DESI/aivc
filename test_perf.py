import time
import uuid
import tempfile
from pathlib import Path
from aivc.semantic.graph import CooccurrenceGraph
from aivc.core.memory import Memory, FileChange

def run_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "storage"
        graph = CooccurrenceGraph(storage)

        # Create 1 large memory with 10000 file modifications
        num_files = 10000
        file_paths = [f"file_{i}.txt" for i in range(num_files)]

        # also add these files to another memory so they have remaining edges
        changes2 = [
            FileChange(
                path=fp, action="modified", blob_hash="deadbeef",
                bytes_added=10, bytes_removed=5,
            )
            for fp in file_paths
        ]
        m2 = Memory(
            id=str(uuid.uuid4()),
            timestamp="2026-01-01T00:00:00+00:00",
            title="Another Memory",
            note="note",
            parent_id=None,
            changes=changes2,
        )
        graph.add_memory(m2)

        changes = [
            FileChange(
                path=fp, action="modified", blob_hash="deadbeef",
                bytes_added=10, bytes_removed=5,
            )
            for fp in file_paths
        ]

        memory_id = str(uuid.uuid4())
        m = Memory(
            id=memory_id,
            timestamp="2026-01-01T00:00:00+00:00",
            title="Large Memory",
            note="note",
            parent_id=None,
            changes=changes,
        )

        print(f"Adding memory to graph (N={num_files})...")
        graph.add_memory(m)

        print("Removing memory from graph...")
        start_time = time.perf_counter()
        graph.remove_memory(memory_id)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Removed memory with {num_files} files in {duration:.4f} seconds.")
        graph.close()

if __name__ == "__main__":
    run_benchmark()
