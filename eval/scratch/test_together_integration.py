"""
Test & Validation Script for Together AI Integration in AIVC.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = REPO_ROOT / "eval"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from eval.config.config_loader import load_env_file, load_models_registry, resolve_config, ModelSpec
from eval.inference_client import (
    InferenceClient,
    TOGETHER_BASE_URL,
    OPENROUTER_BASE_URL,
    TOGETHER_BATCH_ENDPOINT,
    TOGETHER_FILES_ENDPOINT,
)

def run_tests():
    print("=" * 70)
    print(" RUNNING TOGETHER AI INTEGRATION & MULTI-PROVIDER VALIDATION")
    print("=" * 70)

    # 1. Test .env loading
    env_vars = load_env_file()
    assert "TOGETHER_API_KEY" in env_vars, "TOGETHER_API_KEY missing from .env!"
    assert env_vars["TOGETHER_API_KEY"].startswith("tgp_v1_"), "Invalid TOGETHER_API_KEY format!"
    assert os.getenv("TOGETHER_API_KEY") == env_vars["TOGETHER_API_KEY"], "TOGETHER_API_KEY not in os.environ!"
    print("✅ 1. Environment Variables & .env: OK")

    # 2. Test Model Registry
    registry = load_models_registry()
    assert "meta-models/Muse-Glimmer-30B" in registry, "Muse-Glimmer-30B missing from registry!"
    spec: ModelSpec = registry["meta-models/Muse-Glimmer-30B"]
    assert spec.provider == "together", f"Expected provider 'together', got '{spec.provider}'"
    assert spec.prompt_price_per_1m == 0.35, f"Expected 0.35, got {spec.prompt_price_per_1m}"
    assert spec.completion_price_per_1m == 1.50, f"Expected 1.50, got {spec.completion_price_per_1m}"
    assert spec.batch_prompt_price_per_1m == 0.175, f"Expected 0.175, got {spec.batch_prompt_price_per_1m}"
    assert spec.batch_completion_price_per_1m == 0.75, f"Expected 0.75, got {spec.batch_completion_price_per_1m}"
    assert spec.context_window == 131072, f"Expected 131072, got {spec.context_window}"
    print("✅ 2. Models Registry & Muse-Glimmer-30B Specs: OK")

    # 3. Test Cost Computation (Standard vs Batch)
    # N=15 token averages: 4,940,248 prompt, 65,796 completion
    p_15 = 4940248
    c_15 = 65796
    cost_std_15 = spec.compute_cost(p_15, c_15, is_batch=False)
    cost_batch_15 = spec.compute_cost(p_15, c_15, is_batch=True)
    expected_std_15 = (p_15 / 1e6) * 0.35 + (c_15 / 1e6) * 1.50
    expected_batch_15 = (p_15 / 1e6) * 0.175 + (c_15 / 1e6) * 0.75
    assert abs(cost_std_15 - expected_std_15) < 1e-6, f"Standard cost mismatch: {cost_std_15} vs {expected_std_15}"
    assert abs(cost_batch_15 - expected_batch_15) < 1e-6, f"Batch cost mismatch: {cost_batch_15} vs {expected_batch_15}"
    assert abs(cost_batch_15 - (cost_std_15 / 2.0)) < 1e-6, "Batch cost is not exactly 50% of standard cost!"
    print(f"✅ 3. Cost Calculation (N=15): Standard=${cost_std_15:.4f}, Batch=${cost_batch_15:.4f} (-50%): OK")

    # N=273 token averages (extrapolated from 15 episodes):
    p_273 = int(273 * (4940248 / 15))
    c_273 = int(273 * (65796 / 15))
    cost_std_273 = spec.compute_cost(p_273, c_273, is_batch=False)
    cost_batch_273 = spec.compute_cost(p_273, c_273, is_batch=True)
    print(f"✅ 3b. Cost Calculation (N=273): Standard=${cost_std_273:.4f}, Batch=${cost_batch_273:.4f} (-50%): OK")

    # 4. Test Config Resolver
    cfg = resolve_config(profile="production", model="meta-models/Muse-Glimmer-30B")
    assert cfg.model == "meta-models/Muse-Glimmer-30B"
    assert cfg.model_spec.provider == "together"
    assert cfg.limit == 273
    print("✅ 4. Config Resolver for production profile: OK")

    # 5. Test InferenceClient Multi-Provider Routing
    client_together = InferenceClient(default_model="meta-models/Muse-Glimmer-30B")
    assert client_together.provider == "together", f"Expected provider 'together', got {client_together.provider}"
    assert client_together.base_url == TOGETHER_BASE_URL, f"Expected {TOGETHER_BASE_URL}, got {client_together.base_url}"
    assert client_together.api_key == env_vars["TOGETHER_API_KEY"], "Together API key not correctly bound to Together client!"

    client_openrouter = InferenceClient(default_model="qwen/qwen3.7-flash")
    assert client_openrouter.provider == "openrouter", f"Expected provider 'openrouter', got {client_openrouter.provider}"
    assert client_openrouter.base_url == OPENROUTER_BASE_URL, f"Expected {OPENROUTER_BASE_URL}, got {client_openrouter.base_url}"
    assert client_openrouter.api_key == env_vars["OPENROUTER_API_KEY"], "OpenRouter API key not correctly bound to OpenRouter client!"
    print("✅ 5. InferenceClient Provider Auto-Routing: OK")

    # 6. Test Together Batch Request Formatting & JSONL Generation
    sample_messages = [
        {"role": "system", "content": "You are an expert SWE agent."},
        {"role": "user", "content": "Fix issue #42 in repository django/django."},
    ]
    batch_req1 = InferenceClient.create_batch_request_item(
        custom_id="req-swe-001",
        messages=sample_messages,
        model="meta-models/Muse-Glimmer-30B",
        max_tokens=4096,
        temperature=0.2,
    )
    batch_req2 = InferenceClient.create_batch_request_item(
        custom_id="req-swe-002",
        messages=sample_messages,
        model="meta-models/Muse-Glimmer-30B",
        max_tokens=4096,
        temperature=0.2,
    )
    jsonl_output = InferenceClient.prepare_batch_jsonl([batch_req1, batch_req2])
    assert "req-swe-001" in jsonl_output
    assert "req-swe-002" in jsonl_output
    lines = [l for l in jsonl_output.strip().split("\n") if l]
    assert len(lines) == 2
    print("✅ 6. Together AI Batch API JSONL Formatter: OK")

    print("=" * 70)
    print(" ALL TESTS PASSED SUCCESSFULLY! 🎉")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
