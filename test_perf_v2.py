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

        # Case 1: Files become orphans
        num_files = 10000
        file_paths = [f"file_{i}.txt" for i in range(num_files)]

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

        print("Removing memory from graph (all become orphans)...")
        start_time = time.perf_counter()
        graph.remove_memory(memory_id)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Removed memory with {num_files} orphans in {duration:.4f} seconds.")

        # Case 2: Files do NOT become orphans (already have other edges)
        graph = CooccurrenceGraph(storage) # Reset or just keep going

        m2_id = str(uuid.uuid4())
        m2 = Memory(
            id=m2_id,
            timestamp="2026-01-01T00:00:00+00:00",
            title="Keep Aliver",
            note="note",
            parent_id=None,
            changes=changes,
        )
        graph.add_memory(m2)

        memory_id3 = str(uuid.uuid4())
        m3 = Memory(
            id=memory_id3,
            timestamp="2026-01-01T00:00:00+00:00",
            title="Large Memory 2",
            note="note",
            parent_id=None,
            changes=changes,
        )
        graph.add_memory(m3)

        print("Removing memory from graph (none become orphans)...")
        start_time = time.perf_counter()
        graph.remove_memory(memory_id3)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Removed memory with {num_files} non-orphans in {duration:.4f} seconds.")

if __name__ == "__main__":
    run_benchmark()
