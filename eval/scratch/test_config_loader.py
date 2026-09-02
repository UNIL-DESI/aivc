import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.config.config_loader import (
    resolve_config,
    load_profile_yaml,
    load_params_yaml,
    EvalProfileConfig,
    load_benchmark_config,
)

def run_tests():
    print("=== 1. Testing load_params_yaml() ===")
    params = load_params_yaml()
    print("params.yaml:", params)
    assert params["profile"] == "dry_run", f"Expected profile 'dry_run', got {params.get('profile')}"
    assert params["eval"]["limit"] == 30, f"Expected eval.limit 30, got {params.get('eval', {}).get('limit')}"

    print("\n=== 2. Testing load_profile_yaml('dry_run') ===")
    dry_run_yaml = load_profile_yaml("dry_run")
    print("dry_run.yaml:", dry_run_yaml)
    assert "dry_run" not in dry_run_yaml, "Redundant 'dry_run' boolean key should not be present in dry_run.yaml"
    assert dry_run_yaml["limits"] == {"agentic_rag": 30, "swebench_cl": 30, "devbench": 30}

    print("\n=== 3. Testing resolve_config(profile='dry_run') ===")
    cfg = resolve_config(profile="dry_run")
    print("Resolved config dict:", cfg.to_dict())
    assert cfg.profile == "dry_run"
    assert cfg.dry_run is True
    assert cfg.limit == 30
    assert cfg.limits == {"agentic_rag": 30, "swebench_cl": 30, "devbench": 30}
    assert cfg.get_benchmark_limit("agentic_rag") == 30
    assert cfg.get_benchmark_limit("swebench_cl") == 30
    assert cfg.get_benchmark_limit("devbench") == 30
    assert cfg.get_benchmark_limit("unknown_bench") == 30
    assert cfg.models == ["qwen/qwen3.7-flash", "openai/gpt-oss-20b"]

    print("\n=== 4. Testing resolve_config(profile='production') ===")
    prod_cfg = resolve_config(profile="production")
    print("Resolved prod limit:", prod_cfg.limit, "dry_run:", prod_cfg.dry_run)
    assert prod_cfg.profile == "production"
    assert prod_cfg.dry_run is False
    assert prod_cfg.limit == 273
    assert prod_cfg.limits == {"agentic_rag": 273, "swebench_cl": 273, "devbench": 273}

    print("\n=== 5. Testing CLI Override (--limit 5) ===")
    override_cfg = resolve_config(profile="dry_run", limit=5)
    assert override_cfg.limit == 5
    print("Override limit:", override_cfg.limit)

    print("\n=== 6. Testing eval.config.__init__ package imports ===")
    from eval.config import resolve_config as pkg_resolve_config, EvalProfileConfig as PkgEvalConfig
    pkg_cfg = pkg_resolve_config()
    assert pkg_cfg.profile == "dry_run"
    assert pkg_cfg.limit == 30
    print("Package resolve_config() profile:", pkg_cfg.profile, "limit:", pkg_cfg.limit)

    print("\n[SUCCESS] ALL CONFIG LOADER & PROFILE TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
