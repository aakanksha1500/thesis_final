"""
Agent handoff contract tests (QW13).

WHY THIS FILE EXISTS
    344 tests passed while the investment pipeline was silently broken end to
    end, because InvestmentAgent read context["risk_class"] while the
    Orchestrator wrote context["risk_agent_payload"]. Every existing test set
    the key it wanted directly, so nothing exercised the CONTRACT BETWEEN
    agents — only each agent in isolation.

    These tests assert that what a producing agent writes is what the
    consuming agent reads. They are the class of test that catches an entire
    category of silent-degradation bug.

RUNNING
    python -m pytest tests/integration/test_agent_contracts.py -v
"""
from __future__ import annotations

import pytest

from agents.base_agent import AgentResult
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient

COMPLETE_FEATURES = {
    "age": 34, "income": 55000, "employment_status": "employed",
    "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
    "loss_tolerance": 4, "financial_knowledge_score": 3,
}


@pytest.fixture
def orch(tmp_path, monkeypatch) -> Orchestrator:
    from config.settings import settings
    monkeypatch.setattr(settings.orchestrator, "audit_log_dir", tmp_path)
    o = Orchestrator(LLMClient(force_mock=True), session_id="contract-test")
    o._session_state["user_features"] = dict(COMPLETE_FEATURES)
    return o


def _result(results: list[AgentResult], name: str) -> AgentResult:
    match = next((r for r in results if r.agent_name == name), None)
    assert match is not None, f"{name} did not run"
    return match


class TestRiskToInvestmentHandoff:
    """The handoff that was broken. These fail without the QW3 fix."""

    def test_investment_agent_completes_through_the_orchestrator(self, orch):
        context = orch._build_context("I want to invest for retirement")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "InvestmentAgent"], context
        )
        inv = _result(results, "InvestmentAgent")
        assert inv.payload.get("status") == "complete", (
            f"InvestmentAgent degraded to {inv.payload.get('status')!r}: "
            f"{inv.payload.get('message') or inv.error}. The Orchestrator is "
            f"not supplying the context key InvestmentAgent.run() reads."
        )

    def test_shortlist_is_actually_produced(self, orch):
        context = orch._build_context("Recommend an investment product")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "InvestmentAgent"], context
        )
        shortlist = _result(results, "InvestmentAgent").payload.get("shortlist", [])
        assert shortlist, "no products recommended — the pipeline silently degraded"
        assert len(shortlist) <= 3

    def test_risk_class_agrees_across_both_agents(self, orch):
        context = orch._build_context("Invest for me")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "InvestmentAgent"], context
        )
        risk = _result(results, "RiskProfilingAgent").payload["risk_class"]
        inv = _result(results, "InvestmentAgent").payload["risk_class"]
        assert risk == inv


class TestContextKeysProducedAndConsumed:
    """Assert the producer writes exactly what the consumer reads."""

    def test_orchestrator_publishes_every_key_investment_agent_reads(self, orch):
        context = orch._build_context("invest")
        context["_turn_id"] = "t-contract"
        orch._run_agent_sequence(["RiskProfilingAgent", "InvestmentAgent"], context)
        for key in ("risk_class", "risk_agent_payload", "user_features"):
            assert key in context, f"InvestmentAgent reads {key!r}, nobody wrote it"

    def test_orchestrator_publishes_every_key_explainability_reads(self, orch):
        context = orch._build_context("why?")
        context["_turn_id"] = "t-contract"
        orch._run_agent_sequence(
            ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"], context
        )
        for key in ("risk_agent_payload", "investment_agent_payload",
                    "proxy_fields", "proxy_metadata"):
            assert key in context, f"ExplainabilityAgent reads {key!r}, nobody wrote it"

    def test_risk_payload_contains_the_fields_downstream_agents_need(self, orch):
        context = orch._build_context("assess my risk")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(["RiskProfilingAgent"], context)
        payload = _result(results, "RiskProfilingAgent").payload
        for key in ("risk_class", "confidence", "feature_importance", "rationale"):
            assert key in payload, f"downstream agents read {key!r}; it is absent"

    def test_investment_payload_contains_the_fields_downstream_agents_need(self, orch):
        context = orch._build_context("invest")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "InvestmentAgent"], context
        )
        payload = _result(results, "InvestmentAgent").payload
        for key in ("shortlist", "synthesis", "hallucination_flagged"):
            assert key in payload, f"ExplainabilityAgent reads {key!r}; it is absent"


class TestExplainabilityReceivesRealInputs:

    def test_explainability_sees_a_real_risk_class_not_its_default(self, orch):
        context = orch._build_context("explain")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "ExplainabilityAgent"], context
        )
        risk = _result(results, "RiskProfilingAgent").payload["risk_class"]
        expl = _result(results, "ExplainabilityAgent").payload
        assert expl["risk_class"] == risk
        assert expl["shap_narrative"], (
            "Layer A produced nothing — feature_importance did not reach it"
        )

    def test_shap_layer_receives_attributions(self, orch):
        context = orch._build_context("explain")
        context["_turn_id"] = "t-contract"
        results, _ = orch._run_agent_sequence(
            ["RiskProfilingAgent", "ExplainabilityAgent"], context
        )
        assert "shap" in _result(results, "ExplainabilityAgent").payload["layers_applied"]


class TestRoutingSequencesRespectDependencies:

    @pytest.mark.parametrize("routing", list(RoutingDecision))
    def test_investment_never_precedes_risk(self, orch, routing):
        seq = orch._get_agent_sequence(routing)
        if "InvestmentAgent" in seq:
            assert "RiskProfilingAgent" in seq, f"{routing.value} has no risk step"
            assert seq.index("RiskProfilingAgent") < seq.index("InvestmentAgent")

    @pytest.mark.parametrize("routing", list(RoutingDecision))
    def test_explainability_is_always_last(self, orch, routing):
        seq = orch._get_agent_sequence(routing)
        if "ExplainabilityAgent" in seq:
            assert seq[-1] == "ExplainabilityAgent"

    @pytest.mark.parametrize("routing", list(RoutingDecision))
    def test_every_agent_in_every_sequence_is_registered(self, orch, routing):
        for name in orch._get_agent_sequence(routing):
            assert name in orch._agents, f"{name} is routed to but not registered"

    @pytest.mark.parametrize("routing", list(RoutingDecision))
    def test_every_routed_agent_has_a_recovery_strategy(self, orch, routing):
        for name in orch._get_agent_sequence(routing):
            recovery = orch._failure_handler.attempt_recovery(name, Exception("x"), {})
            assert recovery["strategy"] != "none", (
                f"{name} is routed to but has no FailureHandler strategy"
            )


class TestFullTurnDoesNotSilentlyDegrade:

    def test_no_agent_fails_on_a_complete_profile(self, orch):
        result = orch.process_turn("Hello")
        failed = [r.agent_name for r in result.agent_results if not r.success]
        assert not failed, f"agents failed on a complete profile: {failed}"

    def test_turn_count_increments_exactly_once_per_turn(self, orch):
        orch.process_turn("Hello")
        orch.process_turn("Tell me more")
        assert orch._session_state["turn_count"] == 2
