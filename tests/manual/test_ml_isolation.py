import time
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))

print("[*] Starting isolation diagnostic for ML components...")

print("\n1. Testing fastembed.TextEmbedding loading...")
t0 = time.time()
try:
    from fastembed import TextEmbedding
    print(f"   -> Import fastembed took: {time.time() - t0:.2f}s")
    
    t1 = time.time()
    # fastembed downloads/loads the model
    from aivc.config import BI_ENCODER_MODEL
    print(f"   -> Model identifier: {BI_ENCODER_MODEL}")
    model = TextEmbedding(BI_ENCODER_MODEL)
    print(f"   -> Loading TextEmbedding took: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    embeddings = list(model.embed(["Hello world", "AIVC semantic test"]))
    print(f"   -> Embedding generation took: {time.time() - t2:.2f}s")
    print(f"   -> Generated {len(embeddings)} embeddings, dim: {len(embeddings[0])}")
except Exception as e:
    print(f"   [ERROR] FastEmbed failed: {e}")

print("\n2. Testing sentence_transformers.CrossEncoder loading...")
t0 = time.time()
try:
    from sentence_transformers import CrossEncoder
    print(f"   -> Import sentence_transformers took: {time.time() - t0:.2f}s")
    
    t1 = time.time()
    from aivc.config import CROSS_ENCODER_MODEL
    print(f"   -> Model identifier: {CROSS_ENCODER_MODEL}")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print(f"   -> Loading CrossEncoder took: {time.time() - t1:.2f}s")
    
    t2 = time.time()
    scores = cross_encoder.predict([("Is this a test?", "This is indeed a test."), ("Is this a test?", "Bananas are yellow.")])
    print(f"   -> CrossEncoder prediction took: {time.time() - t2:.2f}s")
    print(f"   -> Scores: {scores}")
except Exception as e:
    print(f"   [ERROR] CrossEncoder failed: {e}")

print("\n3. Testing ChromaDB client loading...")
t0 = time.time()
try:
    import chromadb
    print(f"   -> Import chromadb took: {time.time() - t0:.2f}s")
    
    storage_root = Path.home() / ".aivc" / "storage"
    chroma_dir = storage_root / "chromadb"
    print(f"   -> Chroma directory: {chroma_dir}")
    
    t1 = time.time()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    print(f"   -> PersistentClient init took: {time.time() - t1:.2f}s")
except Exception as e:
    print(f"   [ERROR] ChromaDB failed: {e}")

print("\n[*] Diagnostic finished.")
