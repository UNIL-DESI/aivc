"""
Re-export tests for AIVC Daemon and Proxy so pytest finds them from repo root as well.
"""
from src.tests.test_daemon_proxy import *  # noqa: F401, F403
