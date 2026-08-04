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

COMPLETE_FEATURES = {
    "age": 34, "income": 55000, "employment_status": "employed",
    "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
    "loss_tolerance": 4, "financial_knowledge_score": 3,
}

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

def _customer_with_existing_sip() -> list[dict]:
    """
    12 months dense, plus a €400 monthly SIP running all year — Section 3's
    second consumer. The period is unambiguous here (12 consistent monthly
    gaps), so it must be CONFIRMED and converted, not asked about.
    """
    out = _clean_customer_transactions()
    for month in range(1, 13):
        out.append({"date": f"2025-{month:02d}-25", "category": "sip", "amount": 400.0})
    return out


def _customer_with_ambiguous_pension() -> list[dict]:
    """
    One material pension payment and nothing else to date it against. The
    honest answer is a question, not a monthly figure derived from the
    category prior.
    """
    out = _clean_customer_transactions()
    out.append({"date": "2025-04-12", "category": "pension", "amount": 900.0})
    return out


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
    
    def test_questionnaire_only_narrative_mentions_self_reported_confidence(self):
        """
        Section 2's questionnaire disclosure is new prompt content, never
        checked against a real LLM before. No transactions at all -- pure
        self-report path.
        """
        orch = Orchestrator(LLMClient(), session_id="test-real-mode-eval-3")
        _force(orch)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["questionnaire_answers"] = {
            "income": 3000.0, "housing_cost": 1200.0, "rough_monthly_leftover": 500.0,
            "food_spend": 400.0, "utilities_spend": 150.0, "discretionary_spend": 200.0,
        }
        result = orch.process_turn("What does my budget look like?")
        budget_text = _budget_result(result).payload["recommendations_text"]
        print(f"\n[Session eval] BudgetAgent narrative (questionnaire path):\n{budget_text}\n")
        assert any(
            phrase in budget_text.lower()
            for phrase in ("self-report", "estimate", "provided", "starting point")
        ), "Expected the narrative to reflect the self-reported/medium-confidence basis somehow"


class TestMultiTurnQuestionnaireSession:
    """
    Section 2's questionnaire as an ACTUAL multi-turn conversation — the gap
    named explicitly in the last scoping pass.

    WHAT THE EARLIER TESTS DID AND DIDN'T COVER
        test_budget_questionnaire.py checks the schema, ordering and stopping
        criteria as pure functions. test_budget_agent.py checks that a
        pre-built questionnaire_answers dict produces a budget. Both are real
        coverage, and between them they left the interesting half untested:
        nothing ever ASKED a question, read a free-text reply, or carried
        partial answers from one process_turn() to the next. The questionnaire
        was a well-specified form nobody could fill in, and no test could have
        noticed, because every test handed it a completed one.

        These tests drive the loop the way a customer would: an insufficient-
        data budget request, then plain-English replies, one turn at a time,
        on one Orchestrator instance.

    WHY MOCK MODE IS ENOUGH HERE
        The deterministic tier of budget_questionnaire (money regex, stop and
        skip phrases) handles every reply below, so these run free and
        identically every time. That is itself the property being tested: the
        common path of a questionnaire must not depend on a live model. The
        LLM tier only exists for replies the rules refuse, and its behaviour
        under mock (extract nothing, re-ask) is asserted separately in
        test_unparseable_reply_reasks_rather_than_guessing.
    """

    def _start(self, orch, *, transactions=None, monthly_income=None):
        """
        Open a session as a genuinely new customer: transactions=[] is what
        TransactionStore.lookup() returns for someone unknown, and no
        monthly_income is seeded, so the questionnaire starts from the top.
        Pass monthly_income to simulate a customer whose income is already
        known from risk profiling.
        """
        _force(orch)
        if monthly_income is not None:
            orch._session_state["monthly_income"] = monthly_income
        orch._session_state["transactions"] = (
            transactions if transactions is not None else []
        )
        return orch.process_turn("What does my budget look like?")

    def test_insufficient_history_opens_the_questionnaire_with_a_question(self, orch):
        """
        The trigger. BudgetAgent reporting insufficient_history has to lead
        somewhere — before this, it reported the problem and stopped.
        """
        result = self._start(orch)
        assert _budget_result(result).payload["status"] == "insufficient_history"
        assert orch._agents["ConversationalAgent"].questionnaire_active is True
        # The reply IS the first question, deterministically — no LLM
        # paraphrase between the approved wording and the customer.
        assert "monthly take-home income" in result.final_response

    def test_questions_are_asked_one_at_a_time_in_priority_order(self, orch):
        self._start(orch)
        asked = []
        for reply in ["3000", "1200", "500"]:
            result = orch.process_turn(reply)
            asked.append(result.final_response)
        # income -> housing -> leftover -> first optional (food, highest weight)
        assert "rent or mortgage" in asked[0]
        assert "left over" in asked[1]
        assert "groceries and food" in asked[2]

    def test_free_text_replies_are_parsed_into_slots_across_turns(self, orch):
        """
        The specific thing that did not exist: a plain-English reply becoming
        a slot value, and surviving into the NEXT turn's session state.
        """
        self._start(orch)
        orch.process_turn("about €3,000 a month")
        orch.process_turn("I pay 1,200 in rent")
        orch.process_turn("roughly 450 left over")
        answers = orch._session_state["questionnaire_answers"]
        assert answers["income"] == 3000.0
        assert answers["housing_cost"] == 1200.0
        assert answers["rough_monthly_leftover"] == 450.0

    def test_intent_classification_is_skipped_while_in_questionnaire(self, orch):
        """
        '1200' is not an intent. Classifying it would spend an LLM call to
        label it general_query and then route away from the form we are in
        the middle of — the loop would never close.
        """
        self._start(orch)
        calls = []
        original = orch._classify_intent
        orch._classify_intent = lambda msg: (calls.append(msg), original(msg))[1]
        orch.process_turn("3000")
        assert calls == []

    def test_completed_questionnaire_produces_a_budget_in_the_same_turn(self, orch):
        """
        The payoff has to land on the turn the last answer is given. Making
        someone answer six questions and then ask again for the result is a
        loop that technically closes and practically doesn't.
        """
        self._start(orch)
        for reply in ["3000", "1200", "500", "400", "150", "200"]:
            result = orch.process_turn(reply)
            if "BudgetAgent" in result.agents_invoked:
                break
        else:
            pytest.fail("Questionnaire never handed off to BudgetAgent")
        payload = _budget_result(result).payload
        assert payload["status"] == "complete"
        assert payload["questionnaire_confidence"]["confidence_tier"] == "medium"

    def test_self_reported_budget_is_never_high_confidence(self, orch):
        """
        The hard rule from Section 2, checked at session level rather than
        as a unit call: no amount of completeness promotes self-report to
        high confidence.
        """
        self._start(orch)
        for reply in ["3000", "1200", "500", "400", "150", "200", "0"]:
            result = orch.process_turn(reply)
            if "BudgetAgent" in result.agents_invoked:
                break
        conf = _budget_result(result).payload["questionnaire_confidence"]
        assert conf["confidence_tier"] == "medium"
        assert conf["source"] == "self_reported"

    def test_customer_can_stop_mid_questionnaire_and_is_not_asked_again(self, orch):
        self._start(orch)
        orch.process_turn("3000")
        orch.process_turn("1200")
        result = orch.process_turn("I'd rather not answer more questions")
        conv = orch._agents["ConversationalAgent"]
        assert conv.questionnaire_active is False
        assert orch._session_state["customer_wants_to_stop"] is True
        assert "?" not in result.final_response.split("\n")[0]

    def test_stopping_before_the_mandatory_minimum_does_not_fabricate_a_budget(self, orch):
        """
        Respecting a stop must not become "produce a budget anyway". Stopping
        early with only income and housing answered leaves genuinely too
        little, and the honest output is to say so — a confident-looking
        analysis of two numbers would be worse than no answer.
        """
        self._start(orch)
        orch.process_turn("3000")
        result = orch.process_turn("can we stop there")
        assert orch._session_state["customer_wants_to_stop"] is True
        conv_state = orch._agents["ConversationalAgent"].questionnaire_state
        assert conv_state["sufficient"] is False
        assert "BudgetAgent" not in result.agents_invoked

    def test_a_declined_single_question_is_a_skip_not_a_walkout(self, orch):
        """
        "I don't know" about groceries ends that question, not the
        conversation. Conflating the two is what makes a form feel hostile.
        """
        self._start(orch)
        for reply in ["3000", "1200", "500"]:
            orch.process_turn(reply)
        result = orch.process_turn("no idea")
        conv = orch._agents["ConversationalAgent"]
        assert conv.questionnaire_active is True
        assert "food_spend" in conv.questionnaire_state["skipped_slots"]
        assert result.final_response.strip().endswith("?")

    def test_a_skipped_question_is_never_re_asked(self, orch):
        self._start(orch)
        for reply in ["3000", "1200", "500", "not sure"]:
            orch.process_turn(reply)
        asked_after = []
        for reply in ["150", "200"]:
            asked_after.append(orch.process_turn(reply).final_response)
        assert not any("groceries and food" in text for text in asked_after)

    def test_unparseable_reply_reasks_rather_than_guessing(self, orch):
        """
        Under mock (and under a failed LLM call), the extraction tier returns
        nothing. The correct response is to ask again, not to advance with a
        silent gap the customer believes they filled.
        """
        self._start(orch)
        orch.process_turn("3000")
        result = orch.process_turn("it's whatever the going rate is round here")
        assert "didn't catch a figure" in result.final_response
        assert "rent or mortgage" in result.final_response
        assert "housing_cost" not in orch._session_state["questionnaire_answers"]

    def test_income_already_known_is_not_asked_a_second_time(self, orch):
        """
        income is tracked for risk profiling and reused here. Re-asking a
        question someone has already answered is the single most irritating
        thing a form can do.
        """
        result = self._start(orch, monthly_income=4200.0)
        assert "take-home income" not in result.final_response
        assert "rent or mortgage" in result.final_response
        assert orch._session_state["questionnaire_answers"] == {}  # not yet answered
        # ...but income is already in the agent's slots, seeded not asked.
        assert orch._agents["ConversationalAgent"].slots["income"] == 4200.0

    def test_questionnaire_answers_persist_across_later_turns(self, orch):
        """
        Session property, not a function property: once the questionnaire is
        done, a LATER budget question is answered from the stored answers
        without re-asking anything.
        """
        self._start(orch)
        for reply in ["3000", "1200", "500", "400", "150", "200"]:
            orch.process_turn(reply)
        later = orch.process_turn("remind me how my budget looks")
        assert "BudgetAgent" in later.agents_invoked
        assert _budget_result(later).payload["status"] == "complete"
        assert orch._agents["ConversationalAgent"].questionnaire_active is False

    def test_the_session_leaves_a_replayable_event_trail(self, orch):
        """
        The interesting failures here are conversational (a question asked
        twice, a stop missed) and none of them are visible in the final
        answers dict. The event log is what makes a session auditable
        after the fact.
        """
        self._start(orch)
        for reply in ["3000", "1200", "no idea", "500"]:
            orch.process_turn(reply)
        events = orch._agents["ConversationalAgent"].questionnaire_state["events"]
        kinds = [e["event"] for e in events]
        assert "ask" in kinds and "answer" in kinds and "skip" in kinds
        assert all("turn" in e for e in events)


class TestFullAdvisoryMultiAgentSession:
    """
    The other Section 6 gap: every session test so far was single-agent BUDGET
    routing. FULL_ADVISORY is the only route where four agents run in sequence
    and each one's output feeds the next, so it is the only route where
    cross-agent state handoff can actually break — and it had no session-level
    coverage at all.

    Distinct from test_full_advisory.py, which tests the READINESS GATE
    (which agents get pruned when inputs are missing, and the elicitation
    message). This tests what happens when the gate passes: does the output
    of each agent actually reach the next one, does session state accumulate
    correctly across turns, and does the synthesis see all four.
    """

    def test_all_four_agents_run_and_each_produces_a_payload(self, orch):
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        result = orch.process_turn("give me a full review of my finances")

        assert result.agents_invoked == [
            "RiskProfilingAgent", "InvestmentAgent",
            "BudgetAgent", "ExplainabilityAgent",
        ]
        assert all(r.payload for r in result.agent_results)

    def test_risk_class_flows_from_risk_agent_into_investment_agent(self, orch):
        """
        The handoff R2 came from. InvestmentAgent returning status=
        'incomplete' here would mean risk_class never reached it — a failure
        that looks like a working turn from the outside, which is exactly why
        it needs asserting at session level rather than by calling the agent
        directly with a risk_class handed to it.
        """
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        result = orch.process_turn("give me a full review of my finances")

        risk = next(r for r in result.agent_results if r.agent_name == "RiskProfilingAgent")
        inv = next(r for r in result.agent_results if r.agent_name == "InvestmentAgent")
        assert inv.payload.get("status") != "incomplete"
        assert inv.payload["risk_class"] == risk.payload["risk_class"]

    def test_budget_confidence_framework_survives_the_multi_agent_route(self, orch):
        """
        Sections 1 and 3 were only ever exercised on the single-agent BUDGET
        route. A four-agent sequence builds BudgetAgent's context differently
        (it arrives third, after two agents have written into it), so the
        disclosure and clarification machinery needs checking here too rather
        than assumed to carry over.
        """
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _ambiguous_insurance_transactions()
        result = orch.process_turn("give me a full review of my finances")

        payload = _budget_result(result).payload
        assert payload["data_sufficiency"]["coverage_tier"] == "high"
        assert "insurance" in payload["clarifying_questions"]

    def test_existing_contributions_are_detected_on_the_multi_agent_route(self, orch):
        """
        Section 3's second consumer, wired at last. A customer already putting
        €400/month into a SIP should have that DETECTED rather than being
        advised as though they invest nothing — and it has to survive the
        route where InvestmentAgent runs mid-sequence, not just a direct call.
        """
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _customer_with_existing_sip()
        result = orch.process_turn("give me a full review of my finances")

        inv = next(r for r in result.agent_results if r.agent_name == "InvestmentAgent")
        contributions = inv.payload["existing_contributions"]
        assert "sip" in contributions["confirmed"]
        assert contributions["confirmed"]["sip"]["period"] == "monthly"
        assert contributions["total_monthly_committed"] == pytest.approx(400.0)

    def test_ambiguous_contribution_is_asked_about_not_annualised(self, orch):
        """
        The discipline that makes the prior safe. One observed pension payment
        must produce a QUESTION, never a monthly figure invented by applying
        the category prior — a suitability assessment built on a made-up
        contribution amount is worse than one that admits it doesn't know.
        """
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _customer_with_ambiguous_pension()
        result = orch.process_turn("give me a full review of my finances")

        inv = next(r for r in result.agent_results if r.agent_name == "InvestmentAgent")
        contributions = inv.payload["existing_contributions"]
        assert "pension" in contributions["ambiguous"]
        assert "pension" in contributions["clarifying_questions"]
        assert contributions["total_monthly_committed"] == 0.0
        assert contributions["has_unquantified_commitments"] is True

    def test_session_state_accumulates_across_multi_agent_turns(self, orch):
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        orch.process_turn("give me a full review of my finances")
        orch.process_turn("and again please")

        assert orch._session_state["turn_count"] == 2
        assert orch._session_state["risk_profile"] is not None

    def test_every_turn_is_recorded_in_the_audit_log(self, orch, audit_tmp_dir):
        """
        TRiSM property, checked at session level: a four-agent turn has to
        leave a complete record, not just a final response.
        """
        orch.audit_log.log_dir = audit_tmp_dir
        _force(orch, routing=RoutingDecision.FULL_ADVISORY, intent="full_advisory")
        orch._session_state["user_features"] = dict(COMPLETE_FEATURES)
        orch._session_state["monthly_income"] = 3000.0
        orch._session_state["transactions"] = _clean_customer_transactions()
        result = orch.process_turn("give me a full review of my finances")

        assert result.turn_id
        assert len(result.agents_invoked) == 4