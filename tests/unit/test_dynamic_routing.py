"""
Dynamic tool routing tests (Day 6 / G6).

WHAT THESE TESTS ARE FOR
    Day 4-5 established that a PLAN can be validated before execution.
    Day 6 is about what happens between "validated" and "done": a plan
    accepted against the context at planning time can still be wrong by
    the time execution reaches a given step (an earlier agent in the same
    plan can fail, or a "requires" key nobody thought to check for this
    route before now turns out to be missing). _execute_plan() is the
    thing that re-checks, and _publish() is what replaces the class of
    bug (R2) that hand-written `if agent_name == "InvestmentAgent":`
    injection blocks produced — see orchestrator.py's docstrings on both.

    _execute_agent is stubbed throughout most of this file. What's under
    test is the skip-or-run decision and the declarative context handoff,
    not any real agent's behaviour (that belongs to each agent's own test
    file) — same principle test_planner.py uses for the LLM.

TWO REGRESSIONS THIS FILE GUARDS, FOUND WHILE BUILDING DAY 6 ITSELF
    Generalising dependency-checking beyond FULL_ADVISORY (which is all
    _prune_unsatisfiable ever covered) exposed two declarations that were
    stricter than what the agents they describe actually need, because
    nothing had ever gated the BUDGET/RISK_PROFILING/INVESTMENT routes on
    them before:
      1. available_context_keys() treated an EMPTY transactions list as
         "no expense data" — wrong, it's BudgetAgent's own signal to fall
         to the self-report questionnaire.
      2. CAPABILITIES["BudgetAgent"].requires included "monthly_income",
         which BudgetAgent asks for itself as the questionnaire's first
         question and does not need pre-supplied.
    Both are regression-guarded here, not just fixed silently.

RUNNING
    python -m pytest tests/unit/test_dynamic_routing.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.base_agent import AgentResult
from agents.payloads import CAPABILITIES, validate_capability_graph
from orchestrator.planner import available_context_keys


def _orch(session_id: str):
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    return Orchestrator(LLMClient(force_mock=True), session_id=session_id)


# ── _publish() ────────────────────────────────────────────────────────────

class TestPublish:
    """
    _publish() (Day 6): the declarative producer -> context mapping that
    replaced the hand-written injection blocks. Two things it does, both
    tested here: flattens CAPABILITIES[agent].produces keys out of the
    payload, and — for the three agents something downstream reads the
    FULL payload of (see _AGENT_PAYLOAD_CONTEXT_KEY) — also publishes that
    under a fixed key.
    """

    def test_flattens_declared_produces_keys(self):
        orch = _orch("publish-1")
        result = AgentResult(
            agent_name="RiskProfilingAgent", success=True,
            payload={
                "status": "complete", "risk_class": "moderate",
                "confidence": 0.8, "feature_importance": {"age": 0.1},
                "rationale": "not a produces key",
            },
        )
        published = orch._publish("RiskProfilingAgent", result)
        assert published["risk_class"] == "moderate"
        assert published["confidence"] == 0.8
        assert published["feature_importance"] == {"age": 0.1}

    def test_includes_full_payload_under_conventional_key(self):
        orch = _orch("publish-2")
        payload = {
            "status": "complete", "risk_class": "moderate",
            "confidence": 0.8, "feature_importance": {},
        }
        result = AgentResult(agent_name="RiskProfilingAgent", success=True, payload=payload)
        published = orch._publish("RiskProfilingAgent", result)
        assert published["risk_agent_payload"] is payload

    def test_investment_and_budget_also_get_their_payload_key(self):
        orch = _orch("publish-3")
        inv = orch._publish("InvestmentAgent", AgentResult(
            agent_name="InvestmentAgent", success=True,
            payload={"status": "complete", "shortlist": [], "synthesis": "ok"},
        ))
        assert "investment_agent_payload" in inv

        bud = orch._publish("BudgetAgent", AgentResult(
            agent_name="BudgetAgent", success=True,
            payload={"status": "complete", "disposable_income": 100.0,
                     "savings_rate_pct": 10.0},
        ))
        assert "budget_agent_payload" in bud

    def test_conversational_agent_gets_no_payload_key(self):
        """Nothing downstream reads a 'conversational_agent_payload' —
        only the three producers ExplainabilityAgent/InvestmentAgent
        actually read the full payload of."""
        orch = _orch("publish-4")
        published = orch._publish("ConversationalAgent", AgentResult(
            agent_name="ConversationalAgent", success=True,
            payload={"response": "hi", "intent": "general_query"},
        ))
        assert not any(k.endswith("_agent_payload") for k in published)

    def test_returns_empty_dict_on_failed_result(self):
        orch = _orch("publish-5")
        result = AgentResult(agent_name="RiskProfilingAgent", success=False, error="boom")
        assert orch._publish("RiskProfilingAgent", result) == {}

    def test_returns_empty_dict_for_agent_not_in_capabilities(self):
        """Defensive: an unknown agent_name shouldn't crash the executor,
        just publish nothing beyond what a plain payload dict has no
        declared meaning for."""
        orch = _orch("publish-6")
        result = AgentResult(agent_name="SomeFutureAgent", success=True,
                              payload={"status": "complete"})
        assert orch._publish("SomeFutureAgent", result) == {}


# ── _execute_plan() ─────────────────────────────────────────────────────

class TestExecutePlan:
    """_execute_plan(): skip-or-run per step, re-checked at run time."""

    def test_skips_agent_whose_requires_are_unmet(self, audit_tmp_dir):
        orch = _orch("exec-1")
        calls: list[str] = []

        def fake_execute(agent_name, context):
            calls.append(agent_name)
            return AgentResult(agent_name=agent_name, success=True, payload={}), False

        orch._execute_agent = fake_execute

        # InvestmentAgent requires risk_class + user_features; empty context has neither.
        results, recovered, skipped = orch._execute_plan(["InvestmentAgent"], {})

        assert calls == []          # never actually invoked
        assert results == []
        assert len(skipped) == 1
        assert skipped[0]["agent"] == "InvestmentAgent"
        assert "risk_class" in skipped[0]["reason"]
        assert "user_features" in skipped[0]["reason"]

    def test_runs_and_publishes_when_requires_are_met(self, audit_tmp_dir):
        orch = _orch("exec-2")

        def fake_execute(agent_name, context):
            payload = {
                "status": "complete", "risk_class": "moderate",
                "confidence": 0.8, "feature_importance": {},
            }
            return AgentResult(agent_name=agent_name, success=True, payload=payload), False

        orch._execute_agent = fake_execute

        context = {"user_features": {"age": 40}}
        results, recovered, skipped = orch._execute_plan(["RiskProfilingAgent"], context)

        assert skipped == []
        assert len(results) == 1
        assert context["risk_class"] == "moderate"                        # flattened
        assert context["risk_agent_payload"]["risk_class"] == "moderate"  # full payload

    def test_second_step_sees_first_steps_published_keys(self, audit_tmp_dir):
        """The scenario _publish() exists for: RiskProfilingAgent produces
        risk_class in step 1; InvestmentAgent's requires is satisfied by
        that alone in step 2, with nothing pre-seeded in context."""
        orch = _orch("exec-3")

        def fake_execute(agent_name, context):
            if agent_name == "RiskProfilingAgent":
                payload = {"status": "complete", "risk_class": "moderate",
                          "confidence": 0.8, "feature_importance": {}}
            else:
                payload = {"status": "complete", "risk_class": context.get("risk_class"),
                          "shortlist": [], "synthesis": "ok", "deliverable": True}
            return AgentResult(agent_name=agent_name, success=True, payload=payload), False

        orch._execute_agent = fake_execute

        context = {"user_features": {"age": 40}}
        results, recovered, skipped = orch._execute_plan(
            ["RiskProfilingAgent", "InvestmentAgent"], context,
        )

        assert skipped == []
        assert [r.agent_name for r in results] == ["RiskProfilingAgent", "InvestmentAgent"]

    def test_failed_result_publishes_nothing(self, audit_tmp_dir):
        """Distinguishes SKIP (never ran) from RAN-BUT-FAILED (ran,
        published nothing) — both are absence of a key downstream, but
        only one is a wasted call worth reporting differently."""
        orch = _orch("exec-4")

        def fake_execute(agent_name, context):
            return AgentResult(agent_name=agent_name, success=False, error="boom"), False

        orch._execute_agent = fake_execute

        context = {"user_features": {"age": 40}}   # satisfies requires -> runs, not skipped
        results, recovered, skipped = orch._execute_plan(["RiskProfilingAgent"], context)

        assert skipped == []
        assert len(results) == 1 and not results[0].success
        assert "risk_class" not in context

    def test_run_agent_sequence_shim_matches_pre_day6_signature(self, audit_tmp_dir):
        """Back-compat: the old name/signature keeps working for any
        caller that only needs (results, recovered_agent_names)."""
        orch = _orch("exec-5")

        def fake_execute(agent_name, context):
            payload = {"response": "hi", "intent": "general_query",
                      "confidence": 1.0, "escalation_needed": False}
            return AgentResult(agent_name=agent_name, success=True, payload=payload), False

        orch._execute_agent = fake_execute

        results, recovered = orch._run_agent_sequence(["ConversationalAgent"], {})
        assert len(results) == 1
        assert recovered == []


# ── Regression: transactions-empty-list (available_context_keys) ──────────

class TestAvailableContextKeysTransactionsRegression:
    """
    Found while building Day 6: an empty transactions list is a REAL,
    present signal ("checked, nothing there — try the self-report
    questionnaire"), not the absence of one. See agents/budget_agent.py's
    "transactions" in context check, which this must agree with.
    """

    def test_empty_transactions_list_still_counts_as_monthly_expenses(self):
        assert "monthly_expenses" in available_context_keys({"transactions": []})

    def test_missing_transactions_key_does_not_count(self):
        assert "monthly_expenses" not in available_context_keys({})

    def test_direct_monthly_expenses_still_works_without_transactions(self):
        keys = available_context_keys({"monthly_expenses": {"food": 100.0}})
        assert "monthly_expenses" in keys


# ── Regression: BudgetAgent.requires included monthly_income ──────────────

class TestBudgetCapabilityDeclaration:
    """
    Found while building Day 6: BudgetAgent doesn't need monthly_income
    pre-supplied — it asks for it itself, as the questionnaire's first
    mandatory question. Requiring it upfront made every route except
    FULL_ADVISORY (which was never gated on this before) silently skip
    BudgetAgent for a brand-new customer the instant Day 6 started
    gating them too.
    """

    def test_monthly_income_is_not_required(self):
        assert "monthly_income" not in CAPABILITIES["BudgetAgent"].requires

    def test_monthly_expenses_is_still_required(self):
        assert "monthly_expenses" in CAPABILITIES["BudgetAgent"].requires

    def test_capability_graph_is_still_internally_satisfiable(self):
        """validate_capability_graph()'s own static check must still pass
        after narrowing this — every requires key must still be covered
        by some capability's produces or ORCHESTRATOR_SUPPLIED_KEYS."""
        assert validate_capability_graph() == []


# ── Integration: OrchestratorResult.skipped, end to end ───────────────────

class TestOrchestratorResultSkipped:
    """
    dynamic=True only applies when the planner's own proposal was accepted
    (see _execute_plan()'s docstring), so these tests stub the planner's
    LLM call to return a valid, accepted plan — mirroring test_orchestrator
    .py's TestPlannerWiring._orch_with_plan.

    WHY THE SCENARIO BELOW IS "PARTIAL FEATURES", NOT "NO FEATURES AT ALL"
        PlanValidator and _execute_plan share the identical
        available_context_keys() check. A plan naming RiskProfilingAgent
        when user_features is completely empty is REJECTED at planning
        time — it never reaches execution to be skipped there. So
        RiskProfilingAgent itself is never the one _execute_plan skips in
        practice (a totally new customer reaches the risk-elicitation
        loop through Case 2a's "incomplete" branch instead — see
        test_static_fallback_path_never_skips_investment_or_risk below,
        and TestExecutePlan above for _execute_plan's skip logic tested
        directly, bypassing the planner).

        The scenario _execute_plan's runtime re-check actually exists for
        is the NEXT step in a plan: the validator's simulated walk assumes
        a step produces everything in its declared `produces` if its own
        `requires` looked satisfied — it has no way to know the agent will
        itself refuse. So a customer with SOME but not all required
        features gets a plan ACCEPTED (user_features is non-empty, clears
        the planning-time check), RiskProfilingAgent genuinely runs and
        returns status="incomplete" (correctly — it still won't guess),
        risk_class never lands in context, and InvestmentAgent — whose
        precondition looked fine at planning time — gets skipped rather
        than run and refuse for itself. That's the wasted call Day 6
        removes.
    """

    @staticmethod
    def _orch_with_plan(plan_json: str, session_id: str):
        from orchestrator.orchestrator import Orchestrator
        from utils.llm_client import LLMClient
        from config.prompts import PLANNER_SYSTEM

        orch = Orchestrator(LLMClient(force_mock=True), session_id=session_id)

        def fake_chat(system, messages, temperature=None):
            resp = MagicMock()
            resp.content = (
                plan_json if system == PLANNER_SYSTEM else "[MOCK RESPONSE] synthesis"
            )
            resp.tokens_used = 7
            return resp

        orch._planner.llm.chat = fake_chat
        return orch

    def test_investment_skipped_when_risk_profiling_turns_out_incomplete(
        self, audit_tmp_dir,
    ):
        orch = self._orch_with_plan(
            '{"plan": ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"]}',
            "skip-int-1",
        )
        # Enough for the plan to be ACCEPTED (user_features is non-empty),
        # not enough for RiskProfilingAgent to actually produce risk_class.
        orch._session_state["user_features"] = {"age": 34, "income": 55000}
        result = orch.process_turn("Should I invest?")

        assert result.plan.accepted, (
            f"expected the plan to be accepted; rejections={result.plan.rejections}"
        )
        risk_result = next(
            r for r in result.agent_results if r.agent_name == "RiskProfilingAgent"
        )
        assert risk_result.payload.get("status") == "incomplete"
        assert any(
            s["agent"] == "InvestmentAgent" and "risk_class" in s["reason"]
            for s in result.skipped
        ), result.skipped

    def test_static_fallback_path_never_skips_investment_or_risk(self, audit_tmp_dir):
        """
        The other half of the same guarantee test_full_advisory.py::
        TestReadinessGate::test_gate_does_not_touch_other_routes makes:
        when the planner gives nothing usable (mock LLM, here), the turn
        must run exactly what the static table would have — no skips —
        because INVESTMENT's agents_invoked is baked into committed
        RQ2/RQ4 results. This is also what a genuinely new customer
        (empty user_features) hits in the current build: RiskProfilingAgent
        runs via the static path and reports "incomplete" itself, which is
        what starts the risk-elicitation loop (Case 2a's other branch).
        """
        from orchestrator.orchestrator import RoutingDecision

        orch = _orch("skip-int-2")
        orch._classify_intent = lambda msg: (
            RoutingDecision.INVESTMENT, "forced:investment", 1.0,
        )
        result = orch.process_turn("should I invest?")

        assert not result.plan.accepted
        assert result.skipped == []
        assert result.agents_invoked == [
            "RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent",
        ]
        risk_result = next(
            r for r in result.agent_results if r.agent_name == "RiskProfilingAgent"
        )
        assert risk_result.payload.get("status") == "incomplete"
        conv = orch._agents["ConversationalAgent"]
        assert conv.questionnaire_active and conv.questionnaire_kind == "risk"

    def test_known_customer_produces_no_skips(self, audit_tmp_dir):
        """Sanity check in the other direction: a fully-featured customer
        should never appear in `skipped` at all, even on the dynamic path."""
        orch = self._orch_with_plan(
            '{"plan": ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"]}',
            "skip-int-3",
        )
        orch._session_state["user_features"] = {
            "age": 34, "income": 55000, "employment_status": "employed",
            "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
            "loss_tolerance": 4, "financial_knowledge_score": 3,
        }
        result = orch.process_turn("Should I invest?")
        assert result.plan.accepted
        assert result.skipped == []
        assert result.agents_invoked == [
            "RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent",
        ]