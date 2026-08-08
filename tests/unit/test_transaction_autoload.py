"""
Regression tests for Orchestrator._load_customer_transactions().

WHAT THIS GUARDS AGAINST
    "where does my money go?" for a real customer_id (DEMO_GMSC_318),
    asked through the API/UI with no --persona involved, produced a
    confused, hallucinated response ("I've routed the request to the
    BudgetAgent... I only received a response from the ConversationalAgent")
    because context never had a "transactions" key at all — not even an
    empty one. BudgetAgent was never even named in the plan. The intent-
    classifier fix (test_intent_classifier_prompt.py) was necessary but
    not sufficient: even routed correctly, BudgetAgent still had nothing
    to work with. This is the other half — see orchestrator.py's
    _load_customer_transactions() docstring for the full diagnosis.

RUNNING
    python -m pytest tests/unit/test_transaction_autoload.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from config.settings import settings
from data.transaction_store import TransactionStore


def _orch(session_id, customer_id=None, transaction_store=None):
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    return Orchestrator(
        LLMClient(force_mock=True), session_id=session_id,
        customer_id=customer_id, transaction_store=transaction_store,
    )


def _store_with(records: dict, tmp_path: Path) -> TransactionStore:
    path = tmp_path / "txns.json"
    path.write_text(json.dumps({"customers": records}))
    return TransactionStore(path=path)


class TestLoadCustomerTransactions:

    def test_no_customer_id_never_touches_transactions(self, audit_tmp_dir):
        orch = _orch("txn-1")
        assert "transactions" not in orch._session_state

    def test_unknown_customer_id_gets_empty_list_not_missing_key(
        self, audit_tmp_dir, tmp_path,
    ):
        """The distinction that matters: BudgetAgent has a real, graceful
        path for [] ('checked, nothing there'), and none at all for a
        missing key ('never checked') — see agents/data_sufficiency.py."""
        store = _store_with({}, tmp_path)
        orch = _orch("txn-2", customer_id="NOBODY_HOME", transaction_store=store)
        assert orch._session_state["transactions"] == []

    def test_known_customer_id_loads_their_real_transactions(
        self, audit_tmp_dir, tmp_path,
    ):
        store = _store_with(
            {"CUST_A": {"scenario": "test", "transactions": [
                {"date": "2025-06-01", "category": "food", "amount": -50.0},
            ]}},
            tmp_path,
        )
        orch = _orch("txn-3", customer_id="CUST_A", transaction_store=store)
        assert len(orch._session_state["transactions"]) == 1
        assert orch._session_state["transactions"][0]["category"] == "food"

    def test_disabled_config_skips_loading_entirely(
        self, audit_tmp_dir, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(settings.budget, "auto_load_transactions", False)
        store = _store_with(
            {"CUST_A": {"scenario": "test", "transactions": [
                {"date": "2025-06-01", "category": "food", "amount": -50.0},
            ]}},
            tmp_path,
        )
        orch = _orch("txn-4", customer_id="CUST_A", transaction_store=store)
        assert "transactions" not in orch._session_state

    def test_an_intentionally_empty_injected_store_is_not_silently_replaced(
        self, audit_tmp_dir, tmp_path,
    ):
        """
        Regression guard: TransactionStore defines __len__ but not
        __bool__, so `transaction_store or TransactionStore()` treats an
        empty-but-real store as falsy and silently swaps in a brand-new
        default one reading the actual data/processed/transactions.json
        — found because this exact empty store, for customer_id=
        "DEMO_GMSC_318", started returning that file's real (non-empty)
        data the moment scripts/add_demo_customer_transactions.py gave
        that ID a real entry, even though the test explicitly injected
        an empty store. Fixed with `is not None`, not `or` — this test
        pins the injected store as the one actually used, unconditionally.
        """
        store = _store_with({}, tmp_path)  # deliberately empty: len(store) == 0
        assert len(store) == 0
        orch = _orch("txn-5", customer_id="DEMO_GMSC_318", transaction_store=store)
        assert orch.transaction_store is store
        assert orch._session_state["transactions"] == []


class TestBudgetAgentActuallyReachableEndToEnd:
    """
    The real regression: BudgetAgent must appear in agents_invoked and
    reach its own data_sufficiency logic for ANY customer_id, not just
    ones run_demo.py's --persona system happens to cover.
    """

    def test_budget_agent_runs_and_reports_insufficient_for_empty_history(
        self, audit_tmp_dir, tmp_path,
    ):
        from orchestrator.orchestrator import RoutingDecision

        store = _store_with({}, tmp_path)
        orch = _orch("txn-e2e-1", customer_id="DEMO_GMSC_318", transaction_store=store)
        orch._classify_intent = lambda msg: (
            RoutingDecision.BUDGET, "forced:budget_analysis", 1.0,
        )
        result = orch.process_turn("where does my money go?")

        assert "BudgetAgent" in result.agents_invoked
        bud = next(r for r in result.agent_results if r.agent_name == "BudgetAgent")
        assert bud.payload.get("status") == "insufficient_history"
        assert "questions" in result.final_response.lower()

    def test_budget_agent_completes_for_a_customer_with_real_history(
        self, audit_tmp_dir, tmp_path,
    ):
        from orchestrator.orchestrator import RoutingDecision

        store = _store_with(
            {"REAL_CUST": {"scenario": "test", "transactions": [
                {"date": f"2025-{m:02d}-15", "category": "food", "amount": -120.0}
                for m in range(1, 13)
            ]}},
            tmp_path,
        )
        orch = _orch("txn-e2e-2", customer_id="REAL_CUST", transaction_store=store)
        orch._session_state["user_features"] = {"income": 55000}
        orch._classify_intent = lambda msg: (RoutingDecision.BUDGET, "forced", 1.0)
        result = orch.process_turn("where does my money go?")

        bud = next(r for r in result.agent_results if r.agent_name == "BudgetAgent")
        assert bud.payload.get("status") == "complete"
        assert bud.payload["data_sufficiency"]["coverage_tier"] == "high"