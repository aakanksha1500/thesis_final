"""
Root pytest configuration.

Exists so that:
  * the project root is importable from any test without a sys.path hack
    (belt-and-braces alongside pythonpath in pyproject.toml);
  * shared fixtures live in exactly one place.
"""
from __future__ import annotations

import pyarrow.dataset  # noqa: E402, F401
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

@pytest.fixture(autouse=True)
def _approval_store_isolation(tmp_path, monkeypatch):
    """
    ALWAYS isolate the Day 8 approval-gate database — unlike audit_tmp_dir
    above (opt-in; fine there, since AuditLog is per-SESSION, so a test
    that forgets it just leaves harmless stale data in its own inspectable
    -only file). orchestrator/approvals.py's ApprovalStore is deliberately
    ONE SHARED file across every session — a test that doesn't isolate it
    doesn't leave harmless stale data, it pollutes the actual reviewer
    queue a real deployment's GET /approvals serves from.

    That risk is broad, not confined to Day 8's own test file: mock-mode
    synthesis text (this whole suite runs in mock mode by default — see
    _no_live_api_in_tests below) never grounds against real RAG context,
    so the hallucination detector's fallback heuristic correctly, and
    near-universally, flags it — meaning ANY test that runs InvestmentAgent
    through a real process_turn() call trips the gate's hallucination_
    flagged trigger, whether or not that test file has ever heard of Day
    8. Found by actually running run_demo.py after the full suite, not
    assumed — the "0 failed" pytest result never would have caught this,
    since nothing in those tests' own assertions checks approvals.db.
    """
    from config.settings import settings
    from orchestrator.approvals import ApprovalStore, set_approval_store

    db_path = tmp_path / "approvals.db"
    monkeypatch.setattr(settings.approval, "db_path", db_path, raising=False)
    set_approval_store(ApprovalStore(db_path=db_path))
    yield
    set_approval_store(None)

@pytest.fixture(autouse=True)
def _customer_memory_isolation(tmp_path, monkeypatch):
    """
    Same reasoning as _approval_store_isolation immediately above, for
    Day 9's data/customer_memory.py — also one shared SQLite file across
    every session (see that module's docstring), also written on every
    process_turn() call that has a customer_id set (Orchestrator.
    _persist_customer_memory()), so the identical class of silent
    cross-test pollution applies. Fixed proactively this time, before a
    second "0 failed, real file polluted anyway" discovery — see Day 8's
    equivalent finding above for why that would otherwise happen again.
    """
    from config.settings import settings
    from data.customer_memory import CustomerMemoryStore, set_customer_memory_store

    db_path = tmp_path / "customer_memory.db"
    monkeypatch.setattr(settings.conversational, "memory_db_path", db_path, raising=False)
    set_customer_memory_store(CustomerMemoryStore(db_path=db_path))
    yield
    set_customer_memory_store(None)


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
