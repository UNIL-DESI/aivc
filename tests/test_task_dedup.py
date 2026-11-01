"""
Re-export tests for task deduplication so pytest finds them from repo root as well.
"""
from src.tests.test_task_dedup import *  # noqa: F401, F403
