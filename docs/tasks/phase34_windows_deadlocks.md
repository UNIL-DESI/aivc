# Phase 34: Windows Threading Deadlock & Cross-Encoder Bypass

## 1. Context & Discussion (Narrative)
> *AIVC experienced complete, indefinite lock-ups under Windows when running semantic `recall` or `remember` inside agent environments. This was caused by multi-threading locks under Windows (specifically PyTorch DLL Loader Lock and ONNX Thread Pool conflicts with Python's GIL).*

During local operations, a background warmup thread was automatically spawned on startup or during Google Drive sync callbacks to pre-load deep learning model files. While this worked fine under macOS or WSL, under native Windows environments, launching background threads that perform complex DLL loading (such as ONNX Runtime or PyTorch dependencies via `sentence-transformers`) while the main thread communicates over stdio triggers an environment lock-up. 

To solve this permanently, we:
- Completely disabled the async background warmup thread.
- Made the lightweight Indexer initialization synchronous on the main thread (safely under the 5-second IDE timeout).
- Converted `recall` into an asynchronous tool (`async def recall`) so FastMCP does not spawn uncontrolled background threads.
- Added a bypass mode (`AIVC_DISABLE_CROSS_ENCODER = "True"`) under Windows to completely skip PyTorch / Cross-Encoder reranking, bringing first-query latency from ~10 seconds down to 0.1 seconds, eliminating PyTorch import overhead, and bypassing native C++ GIL/ONNX deadlocks entirely.

## 2. Affected Files
- `src/aivc/semantic/searcher.py`
- `src/aivc/server.py`

## 3. Objectives (Definition of Done)
* **Zero lock-ups or deadlocks** during client initialization and query execution under native Windows environments.
* **Instant semantic query responses** using the optimized Bi-Encoder (ChromaDB + FastEmbed) pipeline when the Cross-Encoder is disabled.
* **Synchronous main-thread loading** of the lightweight Indexer collection, ensuring no background model imports trigger DLL Loader Locks.
* **Safe async wrapper** for the `recall` tool, preventing event loop blocking.
