import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))

print("[*] Initializing SemanticEngine...")
from aivc.semantic.engine import SemanticEngine

storage_root = Path.home() / ".aivc" / "storage"
engine = SemanticEngine(storage_root)

print("[*] Executing SemanticEngine warmup to index all physical commits...")
t0 = time.time()
try:
    engine.warmup()
    print(f"[*] Warmup finished successfully in {time.time() - t0:.2f}s!")
except Exception as e:
    print(f"[ERROR] Warmup failed: {e}")
    sys.exit(1)

print("\n[*] Initial collection count:")
try:
    count = engine._indexer._collection.count()
    print(f"  -> Chroma collection count: {count}")
except Exception as e:
    print(f"[ERROR] Could not get Chroma collection count: {e}")
    sys.exit(1)

print("\n[*] Executing search query 'google drive'...")
t0 = time.time()
try:
    results = engine.search("google drive", top_n=3)
    print(f"[*] Search completed in {time.time() - t0:.2f}s!")
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Title: {r.title} (Score: {r.score:.4f})")
        print(f"      ID: {r.memory_id}")
        print(f"      Snippet: {r.snippet[:120]}...")
except Exception as e:
    print(f"[ERROR] Search failed: {e}")
