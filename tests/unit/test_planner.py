"""
Planner + PlanValidator tests (Day 4-5 / G6).

WHAT THESE TESTS ARE FOR
    The claim this component makes is not "the LLM plans well". It is "an
    invalid plan cannot execute". That claim is only worth making if the
    validator is exhaustively tested against the ways a plan can be invalid,
    and if the fallback is proven to always produce something runnable —
    because a planner that rejects correctly and then falls back to garbage
    has moved the failure, not removed it.

    So the LLM is stubbed everywhere below. What is under test is the
    deterministic half: the validator, the JSON extraction, and the accept /
    reject / fall-back decision.

RUNNING
    python -m pytest tests/unit/test_planner.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.payloads import CAPABILITIES, STATIC_SEQUENCES
from orchestrator.planner import (
    Planner,
    PlanValidator,
    RejectionReason,
    available_context_keys,
    extract_plan_json,
)
from utils.llm_client import LLMClient


@pytest.fixture(autouse=True)
def _planner_on(monkeypatch):
    """
    Pin PLANNER_ENABLED for this module regardless of the ambient environment.

    PLANNER_ENABLED is a plain environment variable, so a shell that exported
    it (or a .env carrying the static-arm setting from an evaluation run)
    would otherwise turn these tests into no-ops that still pass — the planner
    would short-circuit to the static table and every "falls back" assertion
    would hold for entirely the wrong reason. The one test that cares about
    the disabled path sets it explicitly.
    """
    from config.settings import settings
    monkeypatch.setattr(settings.planner, "enabled", True)


# A context in which a full investment plan is satisfiable.
FULL_CONTEXT = {
    "user_features": {"age": 35, "income": 55000, "existing_debt": 5000,
                      "dependents": 1},
    "monthly_income": 4500,
    "monthly_expenses": 2800,
}

# A brand-new user: nothing known.
EMPTY_CONTEXT: dict = {
    "user_features": {},
    "risk_profile": None,
    "conversation_history": [],
    "turn_count": 0,
}


def make_planner(responses: list[str]) -> Planner:
    """
    A Planner whose LLM returns a fixed sequence of raw strings.

    Deliberately stubs .chat rather than the whole client, so the Planner's
    own call signature (system=, messages=, temperature=) stays under test —
    a planner that silently stopped passing temperature=0.0 would otherwise
    go unnoticed until the agreement-rate metric started wobbling.
    """
    client = LLMClient(force_mock=True)
    responses_iter = iter(responses)
    seen: list[dict] = []

    def fake_chat(system, messages, temperature=None):
        seen.append({"system": system, "messages": messages,
                     "temperature": temperature})
        resp = MagicMock()
        resp.content = next(responses_iter, "")
        resp.tokens_used = 42
        return resp

    client.chat = fake_chat
    planner = Planner(client)
    planner._calls = seen          # test-visible record of what was sent
    return planner


def plan_json(steps: list[str]) -> str:
    return json.dumps({"plan": steps, "reason": "test"})


# ── the validator, in isolation ────────────────────────────────────────────

class TestPlanValidatorAcceptsValidPlans:

    def test_valid_investment_plan_accepted(self):
        v = PlanValidator()
        problems = v.validate(
            ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"],
            available_context_keys(FULL_CONTEXT),
        )
        assert problems == [], [str(p) for p in problems]

    def test_prerequisite_satisfied_by_an_earlier_step_not_by_context(self):
        """The whole reason validate() walks the plan in order: risk_class is
        absent from context but produced by step 1."""
        v = PlanValidator()
        available = available_context_keys(FULL_CONTEXT)
        assert "risk_class" not in available
        assert v.validate(["RiskProfilingAgent", "InvestmentAgent"], available) == []

    def test_conversational_only_plan_needs_nothing(self):
        v = PlanValidator()
        assert v.validate(["ConversationalAgent"],
                          available_context_keys(EMPTY_CONTEXT)) == []

    def test_every_static_sequence_is_structurally_valid(self):
        """The fallback must never itself be malformed — if it were, a
        rejection would swap a bad plan for another bad plan."""
        v = PlanValidator()
        for intent, steps in STATIC_SEQUENCES.items():
            problems = v.validate_structure(list(steps))
            assert problems == [], f"{intent}: {[str(p) for p in problems]}"


class TestPlanValidatorRejects:

    def _codes(self, problems):
        return {p.code for p in problems}

    def test_missing_prerequisite_rejected(self):
        v = PlanValidator()
        problems = v.validate(["InvestmentAgent"],
                              available_context_keys(FULL_CONTEXT))
        assert RejectionReason.UNSATISFIED_REQUIRES in self._codes(problems)

    def test_wrong_order_rejected_even_though_the_set_is_complete(self):
        """Investment before Risk. Both agents are present, so a set-based
        check would pass this; ordering is the whole point."""
        v = PlanValidator()
        problems = v.validate(["InvestmentAgent", "RiskProfilingAgent"],
                              available_context_keys(FULL_CONTEXT))
        assert RejectionReason.UNSATISFIED_REQUIRES in self._codes(problems)

    def test_unknown_agent_rejected(self):
        v = PlanValidator()
        problems = v.validate(["TaxOptimisationAgent"],
                              available_context_keys(FULL_CONTEXT))
        assert RejectionReason.UNKNOWN_AGENT in self._codes(problems)

    def test_duplicate_step_rejected(self):
        v = PlanValidator()
        problems = v.validate(
            ["RiskProfilingAgent", "RiskProfilingAgent", "ExplainabilityAgent"],
            available_context_keys(FULL_CONTEXT),
        )
        assert RejectionReason.DUPLICATE_STEP in self._codes(problems)

    def test_plan_longer_than_max_steps_rejected(self):
        v = PlanValidator(max_plan_steps=2)
        problems = v.validate_structure(
            ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"]
        )
        assert RejectionReason.TOO_LONG in self._codes(problems)

    def test_empty_plan_rejected(self):
        v = PlanValidator()
        problems = v.validate([], available_context_keys(FULL_CONTEXT))
        assert RejectionReason.EMPTY_PLAN in self._codes(problems)

    def test_explainability_not_last_rejected(self):
        v = PlanValidator()
        problems = v.validate(
            ["ExplainabilityAgent", "RiskProfilingAgent"],
            available_context_keys(FULL_CONTEXT),
        )
        assert RejectionReason.EXPLAINABILITY_NOT_LAST in self._codes(problems)

    def test_non_list_rejected(self):
        v = PlanValidator()
        problems = v.validate_structure("RiskProfilingAgent")  # type: ignore[arg-type]
        assert RejectionReason.NOT_A_LIST in self._codes(problems)

    def test_unknown_agent_short_circuits_the_walk(self):
        """A plan naming an agent with no capability entry cannot be walked —
        the walk would KeyError. It must report the unknown agent and stop,
        not crash."""
        v = PlanValidator()
        problems = v.validate(["NopeAgent", "InvestmentAgent"],
                              available_context_keys(EMPTY_CONTEXT))
        assert self._codes(problems) == {RejectionReason.UNKNOWN_AGENT}

    def test_rejection_carries_actionable_detail(self):
        v = PlanValidator()
        problems = v.validate(["InvestmentAgent"], available_context_keys(FULL_CONTEXT))
        assert "risk_class" in problems[0].detail
        assert problems[0].as_dict()["code"] == "unsatisfied_requires"


class TestValidatorIsPure:

    def test_same_input_same_verdict(self):
        v = PlanValidator()
        steps = ["InvestmentAgent", "RiskProfilingAgent"]
        available = available_context_keys(FULL_CONTEXT)
        first = [str(p) for p in v.validate(list(steps), set(available))]
        second = [str(p) for p in v.validate(list(steps), set(available))]
        assert first == second

    def test_validate_does_not_mutate_its_arguments(self):
        v = PlanValidator()
        steps = ["RiskProfilingAgent", "InvestmentAgent"]
        available = available_context_keys(FULL_CONTEXT)
        before_steps, before_keys = list(steps), set(available)
        v.validate(steps, available)
        assert steps == before_steps
        assert available == before_keys


# ── context key resolution ─────────────────────────────────────────────────

class TestAvailableContextKeys:

    def test_empty_dict_value_counts_as_absent(self):
        """Seeded-but-empty is the default state of a new session. If it read
        as present, every precondition would look satisfied on turn one."""
        assert "user_features" not in available_context_keys({"user_features": {}})

    def test_none_counts_as_absent(self):
        assert "risk_class" not in available_context_keys({"risk_profile": None})

    def test_risk_class_resolved_from_cached_risk_profile(self):
        keys = available_context_keys({"risk_profile": {"risk_class": "balanced"}})
        assert "risk_class" in keys

    def test_monthly_income_resolved_from_user_features_income(self):
        keys = available_context_keys({"user_features": {"income": 55000}})
        assert "monthly_income" in keys

    def test_transactions_satisfy_monthly_expenses(self):
        keys = available_context_keys({"transactions": [{"amount": 12.0}]})
        assert "monthly_expenses" in keys


# ── JSON extraction ────────────────────────────────────────────────────────

class TestExtractPlanJson:

    def test_plain_json_object(self):
        assert extract_plan_json(plan_json(["BudgetAgent"])) == ["BudgetAgent"]

    def test_bare_list(self):
        assert extract_plan_json('["BudgetAgent"]') == ["BudgetAgent"]

    def test_markdown_fenced(self):
        raw = "```json\n" + plan_json(["BudgetAgent"]) + "\n```"
        assert extract_plan_json(raw) == ["BudgetAgent"]

    def test_json_with_preamble_prose(self):
        raw = "Sure! Here is the plan:\n" + plan_json(["BudgetAgent"])
        assert extract_plan_json(raw) == ["BudgetAgent"]

    def test_whitespace_is_stripped_from_agent_names(self):
        assert extract_plan_json('{"plan": [" BudgetAgent "]}') == ["BudgetAgent"]

    def test_prose_only_is_none(self):
        assert extract_plan_json("I think you should run the budget agent.") is None

    def test_empty_string_is_none(self):
        assert extract_plan_json("") is None

    def test_mock_response_is_none(self):
        """Mock mode returns a labelled string, not JSON. It must read as
        malformed rather than as an accidental plan."""
        assert extract_plan_json("[MOCK RESPONSE] Agent role: '...'") is None

    def test_wrong_shape_is_none(self):
        assert extract_plan_json('{"plan": "BudgetAgent"}') is None
        assert extract_plan_json('{"plan": [1, 2, 3]}') is None


# ── the planner end to end, LLM stubbed ────────────────────────────────────

class TestPlannerAcceptsValidProposals:

    def test_valid_plan_accepted_and_used(self):
        steps = ["RiskProfilingAgent", "InvestmentAgent", "ExplainabilityAgent"]
        p = make_planner([plan_json(steps)])
        plan = p.plan("should I invest?", FULL_CONTEXT, "investment")
        assert plan.accepted
        assert plan.source == "planner"
        assert list(plan.steps) == steps
        assert plan.rejections == ()

    def test_planner_may_choose_a_shorter_plan_than_the_table(self):
        """The reason G6 exists: the table always runs Explainability on the
        investment route; the planner is allowed not to."""
        p = make_planner([plan_json(["RiskProfilingAgent", "InvestmentAgent"])])
        plan = p.plan("just the products please", FULL_CONTEXT, "investment")
        assert plan.accepted
        assert list(plan.steps) != STATIC_SEQUENCES["investment"]

    def test_tokens_and_duration_recorded(self):
        p = make_planner([plan_json(["ConversationalAgent"])])
        plan = p.plan("hello", EMPTY_CONTEXT, "conversational_only")
        assert plan.llm_tokens == 42
        assert plan.duration_ms >= 0.0

    def test_system_prompt_is_the_one_from_config_prompts(self):
        """
        Every other agent's system prompt lives in config/prompts.py, and
        evaluation/results_io.py stamps PROMPT_VERSION from that module into
        the _meta block of every results file. A planner prompt defined
        anywhere else would make that stamp a lie: the results would claim to
        describe a prompt set that does not include the prompt that chose the
        agents.
        """
        from config.prompts import PLANNER_SYSTEM

        p = make_planner([plan_json(["ConversationalAgent"])])
        p.plan("hello", EMPTY_CONTEXT, "conversational_only")
        assert p._calls[0]["system"] == PLANNER_SYSTEM

    def test_planner_module_does_not_define_its_own_system_prompt(self):
        """Guards the import, not just the value — a module-level constant
        re-added to planner.py would satisfy the test above while quietly
        reintroducing the second home for prompts."""
        import orchestrator.planner as planner_module
        from config.prompts import PLANNER_SYSTEM

        assert planner_module.PLANNER_SYSTEM is PLANNER_SYSTEM

    def test_prompt_rules_correspond_to_validator_checks(self):
        """
        The prompt asserts six rules. Each is only true because the validator
        enforces it — a rule stated to the model and checked nowhere is a
        constraint the system merely hopes for. This pins the two that are
        easiest to drop from the validator while leaving the prompt intact.
        """
        v = PlanValidator()
        available = available_context_keys(FULL_CONTEXT)
        # rule 2 — never repeat an agent
        assert v.validate(["BudgetAgent", "BudgetAgent"], available)
        # rule 4 — ExplainabilityAgent last
        assert v.validate(["ExplainabilityAgent", "BudgetAgent"], available)

    def test_planner_calls_llm_at_temperature_zero(self):
        p = make_planner([plan_json(["ConversationalAgent"])])
        p.plan("hello", EMPTY_CONTEXT, "conversational_only")
        assert p._calls[0]["temperature"] == 0.0

    def test_prompt_lists_every_capability_by_name(self):
        p = make_planner([plan_json(["ConversationalAgent"])])
        p.plan("hello", EMPTY_CONTEXT, "conversational_only")
        prompt = p._calls[0]["messages"][0]["content"]
        for name in CAPABILITIES:
            assert name in prompt

    def test_prompt_does_not_leak_session_bookkeeping(self):
        """conversation_history / turn_count are not routing signals; putting
        them in the prompt spends tokens and invites the model to reason
        about the wrong thing."""
        p = make_planner([plan_json(["ConversationalAgent"])])
        p.plan("hello", {**EMPTY_CONTEXT, "turn_count": 7}, "conversational_only")
        prompt = p._calls[0]["messages"][0]["content"]
        assert "turn_count" not in prompt
        assert "conversation_history" not in prompt


class TestPlannerFallsBack:

    def test_missing_prerequisite_falls_back_to_static(self):
        p = make_planner([plan_json(["InvestmentAgent"])])
        plan = p.plan("invest?", FULL_CONTEXT, "investment")
        assert not plan.accepted
        assert plan.source == "static_fallback"
        assert list(plan.steps) == STATIC_SEQUENCES["investment"]
        assert RejectionReason.UNSATISFIED_REQUIRES.value in plan.rejection_codes

    def test_unknown_agent_falls_back(self):
        p = make_planner([plan_json(["CryptoAgent"])])
        plan = p.plan("buy bitcoin?", FULL_CONTEXT, "investment")
        assert plan.source == "static_fallback"
        assert RejectionReason.UNKNOWN_AGENT.value in plan.rejection_codes

    def test_duplicate_falls_back(self):
        p = make_planner([plan_json(["BudgetAgent", "BudgetAgent"])])
        plan = p.plan("budget?", FULL_CONTEXT, "budget")
        assert plan.source == "static_fallback"
        assert RejectionReason.DUPLICATE_STEP.value in plan.rejection_codes

    def test_malformed_json_falls_back(self):
        p = make_planner(["I recommend running the risk agent first."])
        plan = p.plan("invest?", FULL_CONTEXT, "investment")
        assert plan.source == "static_fallback"
        assert plan.rejection_codes == (RejectionReason.MALFORMED_JSON.value,)

    def test_llm_exception_falls_back_and_does_not_raise(self):
        client = LLMClient(force_mock=True)

        def boom(system, messages, temperature=None):
            raise RuntimeError("429 rate limit")

        client.chat = boom
        plan = Planner(client).plan("invest?", FULL_CONTEXT, "investment")
        assert plan.source == "static_fallback"
        assert plan.rejection_codes == (RejectionReason.LLM_ERROR.value,)
        assert list(plan.steps) == STATIC_SEQUENCES["investment"]

    def test_empty_plan_falls_back(self):
        p = make_planner([plan_json([])])
        plan = p.plan("hello", EMPTY_CONTEXT, "conversational_only")
        assert plan.source == "static_fallback"
        assert RejectionReason.EMPTY_PLAN.value in plan.rejection_codes

    def test_unknown_intent_falls_back_to_the_configured_default(self):
        p = make_planner(["not json"])
        plan = p.plan("???", EMPTY_CONTEXT, "an_intent_that_does_not_exist")
        assert list(plan.steps) == STATIC_SEQUENCES["conversational_only"]

    def test_rejected_proposal_is_preserved_for_the_metrics(self):
        """The rejected plan IS the RQ4 measurement. Storing only the fallback
        would make 'what did the planner want?' unanswerable."""
        p = make_planner([plan_json(["InvestmentAgent"])])
        plan = p.plan("invest?", FULL_CONTEXT, "investment")
        assert list(plan.proposed) == ["InvestmentAgent"]
        assert list(plan.steps) != list(plan.proposed)


class TestFallbackIsAlwaysRunnable:

    @pytest.mark.parametrize("intent", sorted(STATIC_SEQUENCES))
    def test_fallback_for_every_intent_is_structurally_valid(self, intent):
        p = make_planner(["not json"])
        plan = p.plan("anything", EMPTY_CONTEXT, intent)
        assert plan.source == "static_fallback"
        assert PlanValidator().validate_structure(list(plan.steps)) == []

    @pytest.mark.parametrize("intent", sorted(STATIC_SEQUENCES))
    def test_fallback_never_empty(self, intent):
        p = make_planner(["not json"])
        assert p.plan("anything", EMPTY_CONTEXT, intent).steps

    def test_fallback_equals_the_static_table_exactly(self):
        """The planner-vs-static comparison is only meaningful if the two arms
        differ solely in WHO chose. Any divergence here would make the
        'agreement rate' a comparison of two different tables."""
        p = make_planner(["not json"] * len(STATIC_SEQUENCES))
        for intent, expected in STATIC_SEQUENCES.items():
            assert list(p.plan("x", EMPTY_CONTEXT, intent).steps) == list(expected)


class TestPlannerStats:

    def test_counts_accepted_and_fallback(self):
        p = make_planner([
            plan_json(["ConversationalAgent"]),          # accepted
            "not json",                                  # malformed
            plan_json(["InvestmentAgent"]),              # unsatisfied
        ])
        p.plan("hi", EMPTY_CONTEXT, "conversational_only")
        p.plan("hi", EMPTY_CONTEXT, "conversational_only")
        p.plan("invest?", FULL_CONTEXT, "investment")

        stats = p.stats.as_dict()
        assert stats["attempts"] == 3
        assert stats["accepted"] == 1
        assert stats["fallbacks"] == 2
        assert stats["validity_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert stats["rejection_counts"]["malformed_json"] == 1
        assert stats["rejection_counts"]["unsatisfied_requires"] == 1

    def test_validity_rate_is_zero_not_an_error_before_any_attempt(self):
        assert make_planner([]).stats.validity_rate == 0.0

    def test_static_plan_helper_makes_no_llm_call(self):
        p = make_planner([])
        plan = p.static_plan("investment")
        assert list(plan.steps) == STATIC_SEQUENCES["investment"]
        assert p._calls == []
        assert plan.llm_tokens == 0


class TestPlannerDisabled:

    def test_disabled_planner_makes_no_call_and_returns_the_table(self, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings.planner, "enabled", False)
        p = make_planner([plan_json(["ConversationalAgent"])])
        plan = p.plan("invest?", FULL_CONTEXT, "investment")
        assert p._calls == []
        assert list(plan.steps) == STATIC_SEQUENCES["investment"]
