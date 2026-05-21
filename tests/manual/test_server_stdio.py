import sys
import os
import json
import time
import subprocess
import threading
from queue import Queue, Empty
from pathlib import Path

# Enforce UTF-8 for our script's output to handle Windows console encoding issues
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def test_mcp_server():
    print("[*] Starting AIVC MCP server subprocess...")
    t0 = time.time()
    
    # Resolve the storage root directory
    storage_root = str(Path.home() / ".aivc" / "storage")
    print(f"[*] Resolved storage root: {storage_root}")
    
    # Prepare environment with AIVC_STORAGE_ROOT set
    env = os.environ.copy()
    env["AIVC_STORAGE_ROOT"] = storage_root
    
    # Run the server with UTF-8 encoding explicitly specified
    process = subprocess.Popen(
        [sys.executable, "-m", "aivc.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env
    )
    print(f"[*] Subprocess started in {time.time() - t0:.4f}s. PID: {process.pid}")

    # Queues for stdout and stderr lines
    stdout_queue = Queue()
    stderr_queue = Queue()

    # Thread to read stdout
    def read_stdout():
        for line in iter(process.stdout.readline, ''):
            stdout_queue.put(line)
        process.stdout.close()

    # Thread to read stderr
    def read_stderr():
        for line in iter(process.stderr.readline, ''):
            stderr_queue.put(line)
        process.stderr.close()

    t_stdout = threading.Thread(target=read_stdout, daemon=True)
    t_stderr = threading.Thread(target=read_stderr, daemon=True)
    t_stdout.start()
    t_stderr.start()

    # Helper to send a JSON-RPC message
    def send_msg(msg):
        payload = json.dumps(msg) + "\n"
        print(f"--> Sending: {payload.strip()}")
        process.stdin.write(payload)
        process.stdin.flush()

    # Helper to read a JSON-RPC message
    def read_msg(timeout=20.0):
        t_start = time.time()
        while time.time() - t_start < timeout:
            if process.poll() is not None:
                print(f"[!] Process terminated with code {process.returncode}")
                # Empty stderr queue and print it
                err_lines = []
                while not stderr_queue.empty():
                    err_lines.append(stderr_queue.get())
                print(f"[stderr]:\n" + "".join(err_lines))
                return None
            
            try:
                # Get a line from stdout with a short timeout
                line = stdout_queue.get(timeout=0.1)
                print(f"<-- Received ({time.time() - t_start:.2f}s): {line.strip()}")
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    print(f"[!] Decode error for line: {line}")
            except Empty:
                # Also print any stderr lines that arrive
                while not stderr_queue.empty():
                    print(f"[stderr] {stderr_queue.get().strip()}")
                
        print(f"[!] Timeout waiting for message after {timeout}s")
        # Empty stderr queue
        err_lines = []
        while not stderr_queue.empty():
            err_lines.append(stderr_queue.get())
        if err_lines:
            print(f"[stderr]:\n" + "".join(err_lines))
        return None

    # Step 1: Send Initialize Request
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
            "protocolVersion": "2024-11-05"
        }
    }
    
    t_init = time.time()
    send_msg(init_req)
    
    init_res = read_msg(timeout=25.0)
    if not init_res:
        print("[!] Initialize failed or timed out.")
        process.kill()
        return

    print(f"[*] Initialize completed in {time.time() - t_init:.4f}s")

    # Step 2: Send initialized notification (required by MCP spec)
    init_notif = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    send_msg(init_notif)

    # Step 3: Call get_status tool
    call_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_status",
            "arguments": {}
        }
    }
    t_call = time.time()
    send_msg(call_req)
    call_res = read_msg(timeout=25.0)
    print(f"[*] tools/call 'get_status' completed in {time.time() - t_call:.4f}s")

    # Step 4: Call recall tool
    call_req_2 = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {"query": "google drive"}
        }
    }
    t_call_2 = time.time()
    send_msg(call_req_2)
    call_res_2 = read_msg(timeout=40.0)
    print(f"[*] tools/call 'recall' completed in {time.time() - t_call_2:.4f}s")

    # Clean up
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()

if __name__ == "__main__":
    test_mcp_server()
