"""
Agent collaboration tests (Day 7 / G6 §6.6).

WHAT THESE TESTS ARE FOR
    Day 6 established prevention: don't invoke an agent whose precondition
    is already known to be unmet. Day 7 is repair: an agent got invoked
    anyway (the static path, or a need too specific for CAPABILITIES.
    requires' coarse declaration) and, instead of just failing, names
    exactly what it's missing (payload={"status": "needs_input", "needs":
    [...]}) so the orchestrator can go get it and retry. _execute_agent is
    stubbed throughout most of this file, same principle test_dynamic_
    routing.py uses — what's under test is _satisfy_needs()'s own logic,
    not any real agent's behaviour.

WHY THE END-TO-END SCENARIO IS ExplainabilityAgent, NOT InvestmentAgent
    BUILD_PLAN.md's own worked example is "ask 'should I invest?' with no
    risk profile; InvestmentAgent returns needs: ['risk_class']". Day 6
    already structurally prevents that exact trigger: PlanValidator and
    _execute_plan's dynamic gate share the identical satisfiability check,
    so a plan naming InvestmentAgent without risk_class being producible
    is rejected before execution ever reaches it (see test_dynamic_routing
    .py::TestOrchestratorResultSkipped's docstring for the full argument).
    ExplainabilityAgent is the genuine trigger in this build: its
    CAPABILITIES.requires is empty by design (X1 — it explains whatever
    ran, or nothing), so nothing gates it, and it used to silently default
    to risk_class="moderate" when it had nothing to explain rather than
    ask for a real one. See explainability_agent.py's needs_input branch.

RUNNING
    python -m pytest tests/unit/test_agent_collaboration.py -v
"""
from __future__ import annotations

import pytest

from agents.base_agent import AgentResult


def _orch(session_id: str):
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    return Orchestrator(LLMClient(force_mock=True), session_id=session_id)


# ── _satisfy_needs(): the three bounds, in isolation ───────────────────────

class TestSatisfyNeedsBounds:

    def test_not_needs_input_status_is_a_no_op(self, audit_tmp_dir):
        orch = _orch("collab-1")
        result = AgentResult(agent_name="InvestmentAgent", success=True,
                              payload={"status": "complete"})
        final, produced, events = orch._satisfy_needs(result, {})
        assert final is result
        assert produced == []
        assert events == []

    def test_disabled_setting_is_a_no_op(self, audit_tmp_dir, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings.collaboration, "enabled", False)
        orch = _orch("collab-2")
        result = AgentResult(agent_name="ExplainabilityAgent", success=True,
                              payload={"status": "needs_input", "needs": ["risk_class"]})
        final, produced, events = orch._satisfy_needs(result, {})
        assert final is result
        assert produced == []
        assert events == []

    def test_unproducible_need_is_a_hard_stop_not_a_retry(self, audit_tmp_dir):
        """A need nothing in CAPABILITIES produces must return the
        ORIGINAL needs_input result unchanged — not attempt a retry that
        would just ask again."""
        orch = _orch("collab-3")
        result = AgentResult(agent_name="ExplainabilityAgent", success=True,
                              payload={"status": "needs_input",
                                       "needs": ["something_nobody_produces"]})
        final, produced, events = orch._satisfy_needs(result, {})
        assert final is result                       # unchanged, not retried
        assert final.payload["status"] == "needs_input"
        assert produced == []
        assert len(events) == 1 and events[0]["resolved"] is False
        assert "not producible" in events[0]["reason"]

    def test_depth_limit_stops_without_attempting_anything(self, audit_tmp_dir, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings.collaboration, "max_depth", 0)
        orch = _orch("collab-4")
        result = AgentResult(agent_name="ExplainabilityAgent", success=True,
                              payload={"status": "needs_input", "needs": ["risk_class"]})
        final, produced, events = orch._satisfy_needs(result, {"user_features": {"age": 40}})
        assert final is result
        assert produced == []
        assert len(events) == 1 and "max_depth" in events[0]["reason"]

    def test_requester_cannot_be_asked_to_produce_for_itself(self, audit_tmp_dir):
        """If (hypothetically) an agent asked for something only it
        itself produces, the visited-set exclusion must stop it from
        being selected as its own producer."""
        orch = _orch("collab-5")
        # RiskProfilingAgent is the only producer of "risk_class" — asking
        # for it as the REQUESTER must not select itself.
        result = AgentResult(agent_name="RiskProfilingAgent", success=True,
                              payload={"status": "needs_input", "needs": ["risk_class"]})
        final, produced, events = orch._satisfy_needs(result, {})
        assert produced == []
        assert events[0]["resolved"] is False
        assert "not producible" in events[0]["reason"]


# ── _satisfy_needs(): a real, resolvable need ──────────────────────────────

class TestSatisfyNeedsResolution:

    def test_producer_runs_and_requester_is_retried(self, audit_tmp_dir):
        orch = _orch("collab-6")
        context = {"user_features": {
            "age": 34, "income": 55000, "employment_status": "employed",
            "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
            "loss_tolerance": 4, "financial_knowledge_score": 3,
        }}
        needy = AgentResult(agent_name="ExplainabilityAgent", success=True,
                             payload={"status": "needs_input", "needs": ["risk_class"]})

        final, produced, events = orch._satisfy_needs(needy, context)

        assert [r.agent_name for r in produced] == ["RiskProfilingAgent"]
        assert produced[0].payload.get("status") == "complete"
        assert context["risk_class"] == produced[0].payload["risk_class"]
        assert context["risk_agent_payload"] is produced[0].payload

        # retried: same agent, real invocation, no longer needs_input
        assert final.agent_name == "ExplainabilityAgent"
        assert final.payload.get("status") != "needs_input"

        resolutions = [e for e in events if not e.get("retried")]
        assert len(resolutions) == 1
        assert resolutions[0]["resolved"] is True
        assert resolutions[0]["producer"] == "RiskProfilingAgent"
        assert any(e.get("retried") for e in events)

    def test_producer_that_cannot_itself_complete_leaves_need_unresolved(
        self, audit_tmp_dir,
    ):
        """RiskProfilingAgent runs (it's a valid producer) but reports its
        own status="incomplete" because user_features is partial — no
        risk_class lands in context, so the need stays unresolved and the
        original needs_input result is returned unchanged, not a retry
        that would just fail too."""
        orch = _orch("collab-7")
        context = {"user_features": {"age": 34}}   # nowhere near complete
        needy = AgentResult(agent_name="ExplainabilityAgent", success=True,
                             payload={"status": "needs_input", "needs": ["risk_class"]})

        final, produced, events = orch._satisfy_needs(needy, context)

        assert [r.agent_name for r in produced] == ["RiskProfilingAgent"]
        assert produced[0].payload.get("status") == "incomplete"
        assert "risk_class" not in context
        assert final is needy   # unchanged — not retried
        resolutions = [e for e in events if not e.get("retried")]
        assert resolutions[0]["resolved"] is False


# ── CollaborationStats ──────────────────────────────────────────────────────

class TestCollaborationStats:

    def test_resolution_rate_excludes_retry_bookkeeping_entries(self):
        from orchestrator.orchestrator import CollaborationStats
        stats = CollaborationStats()
        stats.record([
            {"requester": "ExplainabilityAgent", "needs": ["risk_class"],
             "resolved": True, "producer": "RiskProfilingAgent"},
            {"requester": "ExplainabilityAgent", "needs": ["risk_class"],
             "retried": True},
        ])
        assert stats.attempts == 1
        assert stats.resolved == 1
        assert stats.resolution_rate == 1.0

    def test_empty_stats_do_not_divide_by_zero(self):
        from orchestrator.orchestrator import CollaborationStats
        stats = CollaborationStats()
        assert stats.resolution_rate == 0.0
        assert stats.as_dict()["attempts"] == 0


# ── Integration: end to end through process_turn() ─────────────────────────

class TestExplainabilityCollaborationIntegration:
    """
    The real, reachable demo scenario (see module docstring for why this
    is ExplainabilityAgent and not InvestmentAgent): a customer with a
    complete profile on file, asking a bare explanation question with
    nothing computed yet this session.
    """

    def test_full_advisory_route_never_needs_it_but_explanation_alone_does(
        self, audit_tmp_dir,
    ):
        from orchestrator.orchestrator import RoutingDecision

        orch = _orch("collab-int-1")
        orch._classify_intent = lambda msg: (
            RoutingDecision.EXPLANATION_REQUEST, "forced:explanation_request", 1.0,
        )
        orch._session_state["user_features"] = {
            "age": 34, "income": 55000, "employment_status": "employed",
            "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
            "loss_tolerance": 4, "financial_knowledge_score": 3,
        }
        result = orch.process_turn("Can you explain my risk level?")

        assert "RiskProfilingAgent" in result.agents_invoked
        assert "ExplainabilityAgent" in result.agents_invoked
        assert result.agents_invoked.index("RiskProfilingAgent") < \
               result.agents_invoked.index("ExplainabilityAgent")
        assert len(result.collaboration_events) >= 1
        assert any(
            e.get("resolved") and e.get("producer") == "RiskProfilingAgent"
            for e in result.collaboration_events if not e.get("retried")
        )
        assert orch.collaboration_stats["attempts"] >= 1
        assert orch.collaboration_stats["resolved"] >= 1

    def test_no_features_at_all_falls_back_gracefully_no_collaboration(
        self, audit_tmp_dir,
    ):
        """Nothing to explain AND nothing to ask for either — must not
        attempt collaboration it already knows would fail (see
        explainability_agent.py's `context.get("user_features")` gate)."""
        from orchestrator.orchestrator import RoutingDecision

        orch = _orch("collab-int-2")
        orch._classify_intent = lambda msg: (
            RoutingDecision.EXPLANATION_REQUEST, "forced:explanation_request", 1.0,
        )
        result = orch.process_turn("Can you explain my risk level?")

        assert result.agents_invoked == ["ExplainabilityAgent"]
        assert result.collaboration_events == []
        assert result.skipped == []