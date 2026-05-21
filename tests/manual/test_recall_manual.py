import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))

print("[1] Importing core elements...")
from aivc.semantic.indexer import Indexer
from aivc.semantic.searcher import Searcher

print("[2] Initializing Indexer...")
storage_root = Path.home() / ".aivc" / "storage"
indexer = Indexer(storage_root)

print("[3] Indexer details:")
try:
    count = indexer._collection.count()
    print(f"  -> Chroma collection count: {count}")
except Exception as e:
    print(f"  -> Error getting Chroma count: {e}")
    sys.exit(1)

print("[4] Initializing Searcher (lazy-loaded CrossEncoder)...")
searcher = Searcher(indexer)

print("[5] Executing search query 'google drive'...")
try:
    results = searcher.search("google drive", top_n=3)
    print("\nSearch completed successfully!")
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] Title: {r.title} (Score: {r.score:.4f})")
        print(f"      ID: {r.memory_id}")
        print(f"      Snippet: {r.snippet[:100]}...")
except Exception as e:
    print(f"\nError during search: {e}")
