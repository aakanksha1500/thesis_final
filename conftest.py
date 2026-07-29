"""
Root pytest configuration.

Exists so that:
  * the project root is importable from any test without a sys.path hack
    (belt-and-braces alongside pythonpath in pyproject.toml);
  * shared fixtures live in exactly one place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from utils.llm_client import LLMClient  # noqa: E402


def pytest_configure(config):
    """
    Say loudly when the live-API unlock is active.

    EVAL_LIVE_API is a plain environment variable, which means it persists: put
    it in .env, or export it once in a shell, and EVERY subsequent pytest run
    silently makes paid, rate-limited API calls. That is precisely the property
    patch 07 exists to prevent, re-enabled by accident.

    The symptom is not an error. It is a suite that used to take 40 seconds
    taking twenty minutes, because Groq returns 429 and LLMClient's backoff
    sleeps up to 120s per attempt. A banner is cheap; discovering this from a
    stack trace is not.
    """
    if os.getenv("EVAL_LIVE_API") == "1":
        config.stash  # noqa: B018 - touch, keeps linters quiet about the import
        bar = "!" * 74
        print(f"\n{bar}")
        print(" EVAL_LIVE_API=1 — @pytest.mark.evaluation tests WILL call the "
              "live API.")
        print(" That is correct inside scripts/regenerate_evidence.py and wrong "
              "anywhere else.")
        print(" If you did not mean this:  unset EVAL_LIVE_API   "
              "(and remove it from .env)")
        print(bar)

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


# ── API isolation ──────────────────────────────────────────────────────────
_PROVIDER_KEY_VARS = (
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_live_api_in_tests(request, monkeypatch):
    """
    Force every test into LLM mock mode, EXCEPT those marked @pytest.mark.real.

    Opt out for a genuinely real-mode test with @pytest.mark.real.
    """
    if "real" in request.keywords:
        return

    if "evaluation" in request.keywords and os.getenv("EVAL_LIVE_API") == "1":
        return

    for var in _PROVIDER_KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    try:
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    except ImportError:
        pass
