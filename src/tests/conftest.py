"""
conftest.py — Shared pytest fixtures and markers for AIVC.

The ``requires_ml`` marker is used to skip tests that need sentence-transformers,
chromadb, and a working numpy/scipy stack. These tests are intended to run inside
the isolated venv created by ``install.sh``.

Run all tests:        python -m pytest src/tests/
Run only ML tests:    python -m pytest src/tests/ -m requires_ml
Skip ML tests:        python -m pytest src/tests/ -m "not requires_ml"
"""

import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_aivc_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate AIVC_TEMP_DIR for 100% of tests to prevent writing to real AppData/Local/Temp/aivc."""
    test_temp = tmp_path / "aivc_test_temp"
    test_temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AIVC_TEMP_DIR", str(test_temp))
    return test_temp


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-ml",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.requires_ml",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip `requires_ml` tests unless --run-ml is passed OR the deps are importable."""
    run_ml = config.getoption("--run-ml")

    # Try to import the ML stack to see if it works in the current environment.
    ml_available = _check_ml_available()

    if run_ml and not ml_available:
        pytest.exit(
            "ERROR: --run-ml was passed but sentence-transformers/chromadb "
            "cannot be imported in this environment. "
            "Run inside the venv created by install.sh.",
            returncode=1,
        )

    if not run_ml and not ml_available:
        skip_reason = pytest.mark.skip(
            reason=(
                "ML dependencies not available in this environment. "
                "Run inside the venv created by install.sh, "
                "or pass --run-ml to force (which will fail with a clear error)."
            )
        )
        for item in items:
            if item.get_closest_marker("requires_ml"):
                item.add_marker(skip_reason)


def _check_ml_available() -> bool:
    """Return True if fastembed and chromadb can be imported."""
    try:
        import fastembed  # noqa: F401
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False
