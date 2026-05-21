import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path("src").resolve()))

print("[*] Starting eager warmup on main thread...")
t0 = time.time()

import os
os.environ["AIVC_STORAGE_ROOT"] = str(Path.home() / ".aivc" / "storage")

from aivc.semantic.engine import SemanticEngine
engine = SemanticEngine(Path(os.environ["AIVC_STORAGE_ROOT"]))

t1 = time.time()
engine.warmup()
print(f"[*] engine.warmup() completed in {time.time() - t1:.4f}s")
print(f"[*] Total eager warmup took: {time.time() - t0:.4f}s")
