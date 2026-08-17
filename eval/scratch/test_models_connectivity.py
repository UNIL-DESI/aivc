import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from eval.config.config_loader import load_env_file, load_models_registry
from eval.inference_client import InferenceClient

load_env_file()

print("Testing Model Connectivity...")

models_registry = load_models_registry()
print(f"Loaded {len(models_registry)} models in registry.")

models_to_test = ["qwen/qwen3.7-flash", "openai/gpt-oss-20b"]

from eval.config import get_benchmark_tools_schema, get_aivc_system_prompt

tools_schema = get_benchmark_tools_schema(include_workspace=True, benchmark_type="swebench_cl")

for model_id in models_to_test:
    spec = models_registry.get(model_id)
    print(f"\n=======================================================")
    print(f"Testing Model: {model_id} (Provider: {spec.provider if spec else 'unknown'})")
    print(f"=======================================================")
    client = InferenceClient(default_model=model_id)
    
    messages = [
        {"role": "system", "content": get_aivc_system_prompt(benchmark_type="swebench_cl")},
        {"role": "user", "content": "You are starting work on issue SWE-101. First, search long-term memory using the recall tool to see if there is relevant past context for django auth."}
    ]
    
    try:
        resp = client.complete(messages=messages, tools=tools_schema, max_tokens=500)
        choice = resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        print("Message dict keys:", list(msg.keys()))
        print("Tool calls in msg:", msg.get("tool_calls"))
        print("Content in msg:", repr(msg.get("content")))
    except Exception as e:
        print(f"ERROR testing {model_id}: {e}")




