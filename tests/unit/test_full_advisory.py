"""
WHY THESE TESTS LOOK THE WAY THEY DO
    The old FULL_ADVISORY test called _get_agent_sequence() directly. That is
    exactly why R7 went unnoticed for so long: the route had a sequence, and
    asserting on the sequence made it look covered while nothing could ever
    select it. These tests therefore go through _classify_intent and
    process_turn, so a route that cannot be reached fails them.

RUNNING
    python -m pytest tests/unit/test_full_advisory.py -v
"""
from __future__ import annotations

import pytest

from agents.base_agent import AgentResult
from agents.conversational_agent import INTENT_BUCKETS
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient

COMPLETE_FEATURES = {
    "age": 34, "income": 55000, "employment_status": "employed",
    "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
    "loss_tolerance": 4, "financial_knowledge_score": 3,
}

EXPENSES = {
    "housing": 1400, "food": 520, "transport": 260,
    "utilities": 180, "entertainment": 220, "other": 300,
}


@pytest.fixture
def orch():
    return Orchestrator(LLMClient(force_mock=True), session_id="test-full-advisory")


def _force(o, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory"):
    o._classify_intent = lambda msg: (routing, intent, 0.95)


# R7 — reachability

class TestFullAdvisoryIsReachable:

    def test_every_routing_decision_is_reachable_from_some_intent(self, orch):
        """
        The regression guard for R7 itself. If someone adds a RoutingDecision
        without an intent, this fails rather than logging at DEBUG where
        nobody reads it.
        """
        # INTENT_TO_ROUTING is local to _classify_intent, so exercise it the
        # way production does: classify, then check the mapping covered it.
        reachable = set()
        for bucket in INTENT_BUCKETS:
            orch._agents["ConversationalAgent"].classify_only = (
                lambda msg, b=bucket: (b, 1.0)
            )
            routing, _, _ = orch._classify_intent("anything")
            reachable.add(routing)
        assert set(RoutingDecision) == reachable, (
            f"unreachable routes: {sorted(r.value for r in set(RoutingDecision) - reachable)}"
        )

    def test_full_advisory_bucket_exists(self):
        assert "full_advisory" in INTENT_BUCKETS

    def test_full_advisory_intent_routes_to_full_advisory(self, orch):
        orch._agents["ConversationalAgent"].classify_only = (
            lambda msg: ("full_advisory", 0.9)
        )
        routing, intent, _ = orch._classify_intent(
            "I know nothing about finance, tell me what to do"
        )
        assert routing is RoutingDecision.FULL_ADVISORY
        assert intent == "full_advisory"



# R7 — the readiness gate

class TestReadinessGate:

    def test_new_user_gets_elicitation_not_four_failed_agents(self, orch):
        """
        The scenario that motivated mapping the intent at all: someone who
        knows nothing arrives and asks for everything. Before the gate this
        ran four agents, three of which could not possibly succeed.
        """
        _force(orch)
        result = orch.process_turn("I know nothing about finance, help me")

        assert result.agents_invoked == [], (
            "no agent should run when nothing about the user is known"
        )
        assert all(not r.success for r in result.agent_results) or not result.agent_results
        # The reply must name what is needed, not apologise vaguely.
        assert "I need" in result.final_response
        assert orch._session_state["awaiting_full_advisory_inputs"]

    def test_elicitation_costs_no_tokens(self, orch):
        _force(orch)
        result = orch.process_turn("where do I start?")
        assert sum(r.tokens_used for r in result.agent_results) == 0

    def test_elicitation_carries_the_required_disclaimer(self, orch):
        """A message that describes advice must still be compliant."""
        _force(orch)
        result = orch.process_turn("give me full advice")
        blocked = [v for v in result.constraint_violations
                   if v.get("severity") == "hard_block"]
        assert not blocked

    def test_partial_data_runs_what_it_can(self, orch):
        """Features but no expenses → BudgetAgent is pruned, the rest run."""
        _force(orch)
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        result = orch.process_turn("review my finances")

        assert "RiskProfilingAgent" in result.agents_invoked
        assert "InvestmentAgent" in result.agents_invoked
        assert "BudgetAgent" not in result.agents_invoked

    def test_complete_data_runs_the_whole_sequence(self, orch):
        _force(orch)
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_expenses"] = dict(EXPENSES)
        orch._session_state["monthly_income"] = COMPLETE_FEATURES["income"] / 12
        result = orch.process_turn("review my finances")

        assert result.agents_invoked == [
            "RiskProfilingAgent", "InvestmentAgent",
            "BudgetAgent", "ExplainabilityAgent",
        ]

    def test_gate_does_not_touch_other_routes(self, orch):
        """
        Scoped to FULL_ADVISORY on purpose — INVESTMENT's agents_invoked is
        baked into committed RQ2/RQ4 results and must not shift as a side
        effect of this fix.
        """
        _force(orch, RoutingDecision.INVESTMENT, "investment_advice")
        result = orch.process_turn("should I invest?")
        assert result.agents_invoked == [
            "RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent",
        ], "INVESTMENT must still invoke all three even when inputs are missing"

    def test_explainability_alone_is_not_enough_to_proceed(self, orch):
        seq, unmet = orch._prune_unsatisfiable(
            ["RiskProfilingAgent", "ExplainabilityAgent"], {}
        )
        assert seq == [], "ExplainabilityAgent has nothing to explain on its own"
        assert unmet



# R15b — a safety downgrade must persist

class TestDowngradeSurvivesTheTurn:

    def test_low_confidence_downgrade_reaches_session_state(self, orch):
        """
        Before R15 the resolver mutated payloads in place, so the downgrade
        reached session state by aliasing. Fixing R15 removed the accident and
        left the downgrade lasting exactly one turn.
        """
        _force(orch, RoutingDecision.INVESTMENT, "investment_advice")
        orch._agents["RiskProfilingAgent"].run = lambda ctx: AgentResult(
            agent_name="RiskProfilingAgent", success=True,
            payload={
                "status": "complete", "risk_class": "aggressive",
                "confidence": 0.30, "feature_importance": {}, "rationale": "x",
            },
        )
        result = orch.process_turn("invest my savings")

        assert "LOW_CONFIDENCE_AGGRESSIVE" in [c["type"] for c in result.conflicts]

        cached = orch._session_state["risk_profile"]
        assert cached["risk_class"] == "moderate", (
            "the downgrade must persist — the next turn reuses this profile"
        )
        assert cached["risk_class_original"] == "aggressive", (
            "the pre-resolution value must remain recoverable for the audit trail"
        )
    def test_resolver_does_not_mutate_the_caller(self, orch):
        """R15 proper: resolve() returns new objects."""
        from orchestrator.conflict_resolver import ConflictResolver

        risk = AgentResult(
            agent_name="RiskProfilingAgent", success=True,
            payload={"status": "complete", "risk_class": "conservative",
                     "confidence": 0.9, "feature_importance": {}, "rationale": "x"},
        )
        inv = AgentResult(
            agent_name="InvestmentAgent", success=True,
            payload={"status": "complete", "risk_class": "conservative",
                     "synthesis": "s", "deliverable": True,
                     "shortlist": [
                         {"name": "Aggressive Equity Fund", "category": "equity_fund"},
                         {"name": "State Savings", "category": "savings_account"},
                     ]},
        )
        out, conflicts = ConflictResolver().resolve([risk, inv])

        assert len(inv.payload["shortlist"]) == 2, "caller's payload was mutated"
        resolved_inv = next(r for r in out if r.agent_name == "InvestmentAgent")
        assert len(resolved_inv.payload["shortlist"]) == 1
        assert resolved_inv is not inv