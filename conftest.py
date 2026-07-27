"""
Root pytest configuration.

Exists so that:
  * the project root is importable from any test without a sys.path hack
    (belt-and-braces alongside pythonpath in pyproject.toml);
  * shared fixtures live in exactly one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from utils.llm_client import LLMClient  # noqa: E402


@pytest.fixture
def mock_llm() -> LLMClient:
    """An LLMClient pinned to mock mode, regardless of what is in .env."""
    return LLMClient(force_mock=True)


@pytest.fixture
def audit_tmp_dir(tmp_path, monkeypatch):
    """
    Redirect the audit log into pytest's tmp_path so tests never write into
    the real logs/audit/ directory.
    """
    from config.settings import settings
    monkeypatch.setattr(settings.orchestrator, "audit_log_dir", tmp_path)
    return tmp_path
