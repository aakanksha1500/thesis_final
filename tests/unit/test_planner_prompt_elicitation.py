"""Regression tests for the planner prompt rule that keeps risk elicitation reachable.

THE BUG THIS GUARDS AGAINST
    A new customer asking "should I invest my money?" got a generic reply and was
    never asked the risk questions. The planner proposed ConversationalAgent
    alone, because the old rule 3 read as "an agent whose requirements are not in
    the context cannot be planned", and a new customer has empty user_features.

    RiskProfilingAgent therefore appeared neither in the results nor in the
    skipped list, and Orchestrator._advance_questionnaire's risk branch keys off
    exactly those two things. Nothing started the questionnaire.

    Rule 7 tells the planner that missing user information is not a reason to
    drop a specialist. The plan it now proposes is still rejected by
    PlanValidator (the requirement genuinely is unmet), which falls back to the
    static table, which contains RiskProfilingAgent, which gets skipped at
    execution, which is what the questionnaire branch watches for.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.payloads import STATIC_SEQUENCES
from config.prompts import PLANNER_SYSTEM
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from orchestrator.planner import Plan, PlanValidator, available_context_keys
from utils.llm_client import LLMClient

NEW_CUSTOMER_CONTEXT: dict = {"user_features": {}, "risk_profile": None}


@pytest.fixture
def orchestrator(audit_tmp_dir):
    """A brand-new-customer session: no customer_id, nothing on file."""
    return Orchestrator(LLMClient(force_mock=True), session_id="new-customer")


class TestPlannerPromptStatesTheRule:

    def test_rule_7_is_present(self):
        assert "Missing information about the user is NOT a reason" in PLANNER_SYSTEM

    def test_rule_7_names_every_specialist_that_can_be_dropped(self):
        """Naming only RiskProfilingAgent fixed the investment route and left
        the budget route broken in exactly the same way. The rule has to cover
        the class of bug, not the one report of it.
        """
        tail = PLANNER_SYSTEM.split("7.", 1)[1]
        assert "RiskProfilingAgent" in tail
        assert "BudgetAgent" in tail

    def test_rule_3_no_longer_reads_as_a_permission_check(self):
        """Rule 3 constrains ORDER, not whether an agent may be planned at all.

        The old wording ("must already be in the context") is what the model was
        obeying when it dropped the specialist, so it must not come back.
        """
        assert "must already be in the context" not in PLANNER_SYSTEM
        assert "Order the plan so that" in PLANNER_SYSTEM

    def test_conversational_only_is_still_scoped_to_rule_6(self):
        assert "never as a stand-in for the specialist" in PLANNER_SYSTEM


class TestNewCustomerStillCannotSatisfyRiskProfiling:
    """The validator's behaviour is unchanged — only the prompt moved."""

    def test_empty_user_features_is_not_an_available_context_key(self):
        assert "user_features" not in available_context_keys(NEW_CUSTOMER_CONTEXT)

    def test_a_plan_with_risk_profiling_is_rejected_for_a_new_customer(self):
        problems = PlanValidator().validate(
            ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"],
            available_context_keys(NEW_CUSTOMER_CONTEXT),
        )
        assert problems, "a new customer cannot satisfy RiskProfilingAgent's requires"
        assert {p.code.value for p in problems} == {"unsatisfied_requires"}
        assert any("RiskProfilingAgent" in p.detail for p in problems)

    def test_the_static_fallback_still_contains_risk_profiling(self):
        """This is what makes the rejection recoverable rather than a dead end."""
        assert "RiskProfilingAgent" in STATIC_SEQUENCES["investment"]


class TestQuestionnaireIsReachedEndToEnd:
    """The behavioural assertion — the reason the prompt rule exists at all."""

    def _run(self, plan_steps, source, orch):
        with patch.object(
            Orchestrator, "_classify_intent",
            return_value=(RoutingDecision.INVESTMENT, "investment_advice", 0.95),
        ), patch.object(
            Orchestrator, "_plan_turn",
            return_value=Plan(steps=tuple(plan_steps), source=source, intent="investment"),
        ):
            return orch.process_turn("should I invest my money?")

    def test_conversational_only_plan_never_starts_the_questionnaire(self, orchestrator):
        """The old behaviour, pinned so a prompt regression fails loudly here."""
        self._run(["ConversationalAgent"], "planner", orchestrator)
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is False

    def test_static_fallback_plan_does_start_the_questionnaire(self, orchestrator):
        result = self._run(
            list(STATIC_SEQUENCES["investment"]), "static_fallback", orchestrator,
        )
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is True
        assert conv.questionnaire_kind == "risk"
        assert "?" in result.final_response

    def test_accepted_planner_plan_with_risk_also_starts_the_questionnaire(
        self, orchestrator
    ):
        """Belt and braces: if a future validator change lets this plan through,
        the skip-at-execution path must still reach the questionnaire."""
        result = self._run(list(STATIC_SEQUENCES["investment"]), "planner", orchestrator)
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is True
        assert "?" in result.final_response


class TestBudgetQuestionnaireIsReachedEndToEnd:
    """The same bug on the budget route, which the risk-only fix did not cover.

    A brand-new session has no "transactions" key, so BudgetAgent never reaches
    its sufficiency assessment and returns status="incomplete" rather than
    "insufficient_history". Matching only the latter — and not checking
    `skipped` at all — left this customer with a generic non-answer on every
    path.
    """

    def _run(self, plan_steps, source, orch):
        with patch.object(
            Orchestrator, "_classify_intent",
            return_value=(RoutingDecision.BUDGET, "budget_analysis", 0.95),
        ), patch.object(
            Orchestrator, "_plan_turn",
            return_value=Plan(steps=tuple(plan_steps), source=source, intent="budget"),
        ):
            return orch.process_turn("where does my money go?")

    def test_budget_agent_skipped_by_dynamic_routing_still_asks(self, orchestrator):
        result = self._run(list(STATIC_SEQUENCES["budget"]), "planner", orchestrator)
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is True
        assert conv.questionnaire_kind == "budget"
        assert "?" in result.final_response

    def test_budget_agent_returning_incomplete_still_asks(self, orchestrator):
        """The static path: BudgetAgent runs, hits its missing-income guard."""
        result = self._run(list(STATIC_SEQUENCES["budget"]), "static_fallback", orchestrator)
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is True
        assert conv.questionnaire_kind == "budget"
        assert "?" in result.final_response

    def test_the_customer_never_sees_the_developer_diagnostic(self, orchestrator):
        """BudgetAgent's "incomplete" message names context keys and is written
        for whoever wired the call. It must not become the preamble.
        """
        result = self._run(list(STATIC_SEQUENCES["budget"]), "static_fallback", orchestrator)
        assert "monthly_income" not in result.final_response
        assert "user_features" not in result.final_response
        assert "context" not in result.final_response

    def test_conversational_only_plan_still_cannot_reach_it(self, orchestrator):
        """Pinned so the prompt rule stays the only thing standing between a
        budget question and a generic answer — if this ever starts passing,
        the orchestrator gained a second trigger and this test should be
        rewritten deliberately, not silently.
        """
        self._run(["ConversationalAgent"], "planner", orchestrator)
        conv = orchestrator._agents["ConversationalAgent"]
        assert conv.questionnaire_active is False
