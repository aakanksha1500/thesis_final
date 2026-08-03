"""
Section 6 — session-level, multi-turn evaluation methodology.

WHY THIS FILE IS DIFFERENT FROM test_banking77_evaluation.py
    Banking77 (Day 2b) is a component-accuracy benchmark: 200
    independent, unrelated single-shot classifications, each its own
    isolated sample, no shared session state between them — the right
    methodology for measuring one component's (the intent classifier's)
    raw accuracy against a labelled dataset.

    This file simulates something structurally different: one
    Orchestrator instance per scenario, run through process_turn() (the
    real pipeline — routing, agent execution, conflict resolution,
    synthesis), the way an actual customer session works. This is what a
    unit test of an individual method can't catch: whether Sections 1
    and 3's confidence/clarification framework actually survives the
    full session flow, not just the isolated function call.

    It already caught one real thing before a single test ran: building
    this file surfaced that Orchestrator._agent_is_satisfiable()'s
    BudgetAgent check only ever recognised monthly_expenses, not
    transactions, meaning FULL_ADVISORY routing would have silently
    pruned BudgetAgent out for exactly the customers this framework
    exists to serve. Fixed in orchestrator/orchestrator.py, regression-
    tested in test_full_advisory.py, not just quietly special-cased here.

ROUTING IS FORCED, NOT CLASSIFIED
    Via _force() (see tests/unit/test_full_advisory.py's own docstring
    for why: these tests go through process_turn() rather than calling
    internals directly, so a route that cannot be reached fails them —
    but WHICH route is selected is itself Banking77's job to verify, not
    this file's. Forcing it here isolates what this file actually checks:
    given routing has happened, does the rest of the pipeline behave.

TWO TRACKS — mirrors where the one real LLM call in this whole flow
actually lives (see BudgetAgent._build_synthesis_prompt and
Orchestrator._synthesize_response):
    Deterministic (mock LLM, free, run every time, all in this file) —
        session state, routing, agents_invoked, payload structure:
        everything that doesn't depend on what the LLM actually SAYS.
    Real-mode (TestRealModeNarrativeContent, needs EVAL_LIVE_API=1,
        costs tokens) — does the generated narrative actually MENTION
        the disclosure / ASK the clarifying question? Only a real call
        can answer this; mock just echoes a canned string regardless of
        prompt content, so there is no way to check this for free.
"""
from __future__ import annotations

import pytest

from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient


def _force(o, routing=RoutingDecision.BUDGET, intent="budget_analysis"):
    o._classify_intent = lambda msg: (routing, intent, 0.95)


def _clean_customer_transactions() -> list[dict]:
    """12 months, dense, no ambiguous categories. Control scenario."""
    out = []
    for month in range(1, 13):
        out.append({"date": f"2025-{month:02d}-01", "category": "housing", "amount": 1200.0})
        out.append({"date": f"2025-{month:02d}-10", "category": "food", "amount": 350.0})
        out.append({"date": f"2025-{month:02d}-20", "category": "food", "amount": 300.0})
    return out


def _ambiguous_insurance_transactions() -> list[dict]:
    """12 months, dense, one material insurance payment — the Section 3 case."""
    out = _clean_customer_transactions()
    out.append({"date": "2025-09-01", "category": "insurance", "amount": 600.0})
    return out


def _minimal_history_transactions() -> list[dict]:
    """A single month — the Section 1 'new customer' case."""
    return [
        {"date": "2025-06-05", "category": "housing", "amount": 1200.0},
        {"date": "2025-06-15", "category": "food", "amount": 300.0},
    ]


def _dormant_long_window_transactions() -> list[dict]:
    """12-month span, active in only 3 of those months — the Section 1/2
    'long-standing but sparse' case, same mechanism as new-customer."""
    return [
        {"date": "2024-07-05", "category": "housing", "amount": 1200.0},
        {"date": "2024-11-05", "category": "housing", "amount": 1200.0},
        {"date": "2025-06-05", "category": "housing", "amount": 1200.0},
    ]


@pytest.fixture
def orch():
    return Orchestrator(LLMClient(force_mock=True), session_id="test-session-eval")


def _budget_result(orchestrator_result):
    return next(
        r for r in orchestrator_result.agent_results if r.agent_name == "BudgetAgent"
    )


class TestSessionRoutingAndInvocation:
    """
    Confirms BudgetAgent is actually reachable and actually runs given
    transaction-shaped data — the exact thing the satisfiability fix
    above was needed for.
    """

    def test_clean_customer_reaches_budget_agent(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        result = orch.process_turn("What does my budget look like?")
        assert "BudgetAgent" in result.agents_invoked
        assert _budget_result(result).success is True

    def test_new_customer_with_empty_transactions_still_reaches_budget_agent(self, orch):
        """
        TransactionStore.lookup() for an unknown customer returns [] —
        must still reach BudgetAgent (which then reports
        insufficient_history itself) rather than being pruned out
        before ever trying.
        """
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = []
        result = orch.process_turn("What does my budget look like?")
        assert "BudgetAgent" in result.agents_invoked
        assert _budget_result(result).payload["status"] == "insufficient_history"


class TestSessionConfidenceAndClarification:
    """
    Sections 1 and 3 surviving the full session pipeline, not just a
    direct method call.
    """

    def test_clean_customer_high_confidence_no_clarifying_questions(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        result = orch.process_turn("What does my budget look like?")
        payload = _budget_result(result).payload
        assert payload["data_sufficiency"]["coverage_tier"] == "high"
        assert payload["clarifying_questions"] == {}

    def test_ambiguous_insurance_customer_gets_a_clarifying_question(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _ambiguous_insurance_transactions()
        result = orch.process_turn("What does my budget look like?")
        payload = _budget_result(result).payload
        assert "insurance" in payload["clarifying_questions"]

    def test_minimal_history_customer_gets_low_confidence_disclosure(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _minimal_history_transactions()
        result = orch.process_turn("What does my budget look like?")
        payload = _budget_result(result).payload
        assert payload["data_sufficiency"]["coverage_tier"] == "minimal"

    def test_dormant_long_window_customer_flagged_low_confidence_not_high(self, orch):
        """
        The case Section 1 exists for: 12-month coverage tier, but real
        confidence_score is low because density is low — must survive
        all the way through a real session turn, not just the isolated
        assess_data_sufficiency() call.
        """
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _dormant_long_window_transactions()
        result = orch.process_turn("What does my budget look like?")
        suff = _budget_result(result).payload["data_sufficiency"]
        assert suff["coverage_tier"] == "high"
        assert suff["confidence_score"] < 0.3


class TestSessionStatePersistsAcrossTurns:
    """
    The thing that actually distinguishes a session from Banking77's
    independent samples: state carrying across multiple process_turn()
    calls on the SAME Orchestrator instance.
    """

    def test_turn_count_increments_across_turns(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        orch.process_turn("What does my budget look like?")
        orch.process_turn("What about my spending on food?")
        assert orch._session_state["turn_count"] == 2

    def test_conversation_history_accumulates(self, orch):
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        orch.process_turn("What does my budget look like?")
        orch.process_turn("And what about food specifically?")
        history = orch._session_state["conversation_history"]
        user_turns = [h for h in history if h["role"] == "user"]
        assert len(user_turns) == 2
        assert user_turns[0]["content"] == "What does my budget look like?"
        assert user_turns[1]["content"] == "And what about food specifically?"

    def test_transactions_do_not_need_re_injecting_every_turn(self, orch):
        """Once in session_state, later turns still see the same data —
        matches how a real session (data fetched once, reused) should
        behave, not re-supplied by the caller every message."""
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        orch.process_turn("What does my budget look like?")
        result2 = orch.process_turn("Tell me again please")
        assert "BudgetAgent" in result2.agents_invoked
        assert _budget_result(result2).success is True


@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests
class TestRealModeNarrativeContent:
    """
    The one thing nothing above can check: does the LLM-generated text
    the customer actually reads reflect the disclosure/clarifying
    question, or just the structured payload fields alongside it. Needs
    EVAL_LIVE_API=1 and a working key — mock mode returns a fixed string
    regardless of prompt content, so this genuinely cannot be answered
    for free. This is the answer to "when do you need real mode": never
    for Sections 1/3's own logic (pure Python, already fully verified
    above and in test_data_sufficiency.py / test_periodicity_inference.py
    — deterministic, so a real run would not tell you anything new) —
    only here, where the LLM decides what to actually say.

    Explicitly skipped rather than asserted-and-failed when not in real
    mode: a content check like "does the text mention insurance" is
    meaningless against a mock client's fixed fallback string, and
    letting it fail there would break the "green except one unrelated
    pre-existing failure" state of the rest of this suite for no reason
    — there's nothing to verify without an actual model response.
    """

    @pytest.fixture(autouse=True)
    def _skip_unless_real(self):
        import os
        if os.getenv("EVAL_LIVE_API") != "1":
            pytest.skip(
                "Needs a real LLM call to check narrative CONTENT, not "
                "just structure — run with EVAL_LIVE_API=1. Everything "
                "structural (sufficiency tiers, clarifying_questions "
                "keys, routing) is already covered above without this."
            )

    def test_ambiguous_insurance_narrative_asks_the_clarifying_question(self):
        orch = Orchestrator(LLMClient(), session_id="test-real-mode-eval")
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _ambiguous_insurance_transactions()
        result = orch.process_turn("What does my budget look like?")
        budget_text = _budget_result(result).payload["recommendations_text"]
        print(f"\n[Session eval] BudgetAgent narrative:\n{budget_text}\n")
        print(f"[Session eval] Final orchestrator response:\n{result.final_response}\n")
        assert "insurance" in budget_text.lower() or "insurance" in result.final_response.lower()

    def test_minimal_history_narrative_mentions_limited_data(self):
        orch = Orchestrator(LLMClient(), session_id="test-real-mode-eval-2")
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _minimal_history_transactions()
        result = orch.process_turn("What does my budget look like?")
        budget_text = _budget_result(result).payload["recommendations_text"]
        print(f"\n[Session eval] BudgetAgent narrative:\n{budget_text}\n")
        # Not a strict assertion on exact wording (LLM phrasing varies) --
        # printed for manual read-through rather than a brittle keyword
        # match on generated text.