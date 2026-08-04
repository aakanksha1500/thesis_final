"""
Tests for agents/budget_questionnaire.py — Section 2's insufficient-data
questionnaire framework. Pure logic, no LLM, no I/O.
"""
from __future__ import annotations

import pytest

from agents.budget_questionnaire import (
    MAX_QUESTIONS_PER_SITTING,
    QUESTIONNAIRE_SCHEMA,
    SELF_REPORT_CONFIDENCE_CEILING,
    blend_expenses,
    build_monthly_expenses_from_slots,
    is_sufficient,
    next_question,
    self_report_confidence,
)


class TestQuestionOrdering:

    def test_mandatory_questions_come_first(self):
        first = next_question({})
        assert first.slot_name == "income"

    def test_mandatory_questions_asked_in_order(self):
        q1 = next_question({})
        assert q1.slot_name == "income"
        q2 = next_question({"income": 3000})
        assert q2.slot_name == "housing_cost"
        q3 = next_question({"income": 3000, "housing_cost": 1200})
        assert q3.slot_name == "rough_monthly_leftover"

    def test_optional_questions_ordered_by_weight_descending(self):
        all_mandatory = {"income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500}
        q = next_question(all_mandatory)
        assert q.slot_name == "food_spend"  # highest optional weight (0.14)


class TestStoppingCriteria:

    def test_not_sufficient_until_mandatory_answered(self):
        assert is_sufficient({"income": 3000}) is False
        assert is_sufficient({"income": 3000, "housing_cost": 1200}) is False

    def test_sufficient_once_mandatory_and_enough_optional_coverage(self):
        """food(0.14) + utilities(0.07) + discretionary(0.06) = 0.27,
        which is >= 70% of total optional weight (0.37 * 0.7 = 0.259) —
        should stop here, without needing debt_repayments or
        large_recurring_items."""
        answered = {
            "income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500,
            "food_spend": 400, "utilities_spend": 150, "discretionary_spend": 200,
        }
        assert is_sufficient(answered) is True

    def test_not_sufficient_with_only_two_optional_answered(self):
        """food + utilities alone = 0.21, short of the 0.259 threshold."""
        answered = {
            "income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500,
            "food_spend": 400, "utilities_spend": 150,
        }
        assert is_sufficient(answered) is False

    def test_customer_wants_to_stop_overrides_coverage(self):
        """Only mandatory answered, but the customer said stop -- respect it."""
        answered = {"income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500}
        assert is_sufficient(answered, customer_wants_to_stop=True) is True

    def test_customer_wants_to_stop_does_not_override_missing_mandatory(self):
        """Can't be "sufficient" without the mandatory minimum, no matter
        what the customer says -- there's no budget at all without income."""
        assert is_sufficient({"income": 3000}, customer_wants_to_stop=True) is False

    def test_hard_cap_stops_it_even_below_coverage_threshold(self):
        """All mandatory + 4 of 5 optional (everything except food, the
        highest-weight one) = 7 questions, coverage = 0.23/0.37 = 0.62,
        below the 70% threshold -- but the hard cap should still stop it."""
        answered = {
            "income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500,
            "utilities_spend": 150, "discretionary_spend": 200,
            "debt_repayments": 100, "large_recurring_items": 1200,
        }
        assert sum(1 for v in answered.values() if v is not None) == MAX_QUESTIONS_PER_SITTING
        assert is_sufficient(answered) is True

    def test_next_question_returns_none_once_sufficient(self):
        answered = {
            "income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500,
            "food_spend": 400, "utilities_spend": 150, "discretionary_spend": 200,
        }
        assert next_question(answered) is None

    def test_next_question_returns_none_when_everything_answered(self):
        all_answered = {q.slot_name: 100 for q in QUESTIONNAIRE_SCHEMA}
        assert next_question(all_answered) is None


class TestBuildMonthlyExpensesFromSlots:

    def test_maps_simple_categories_directly(self):
        slots = {"income": 3000, "housing_cost": 1200, "food_spend": 400}
        expenses = build_monthly_expenses_from_slots(slots)
        assert expenses["housing"] == 1200.0
        assert expenses["food"] == 400.0

    def test_income_and_leftover_excluded_not_expense_categories(self):
        slots = {"income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500}
        expenses = build_monthly_expenses_from_slots(slots)
        assert "income" not in expenses
        assert "rough_monthly_leftover" not in expenses

    def test_large_recurring_items_divided_by_twelve(self):
        """Collected as an annual figure (that's what the question asks
        for) -- must be converted to a monthly-equivalent, not used raw."""
        slots = {"large_recurring_items": 1200.0}
        expenses = build_monthly_expenses_from_slots(slots)
        assert expenses["other_recurring"] == 100.0

    def test_unanswered_categories_absent_not_zero(self):
        slots = {"housing_cost": 1200}
        expenses = build_monthly_expenses_from_slots(slots)
        assert "food" not in expenses
        assert "utilities" not in expenses


class TestSelfReportConfidence:

    def test_capped_at_medium_even_when_fully_answered(self):
        """Every question answered -- confidence must still be 'medium',
        never 'high', regardless of completeness."""
        all_answered = {q.slot_name: 100 for q in QUESTIONNAIRE_SCHEMA}
        result = self_report_confidence(all_answered)
        assert result["confidence_tier"] == SELF_REPORT_CONFIDENCE_CEILING
        assert result["confidence_tier"] != "high"

    def test_capped_at_medium_with_minimal_answers_too(self):
        minimal = {"income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500}
        result = self_report_confidence(minimal)
        assert result["confidence_tier"] == "medium"

    def test_source_is_labelled_self_reported(self):
        result = self_report_confidence({"income": 3000})
        assert result["source"] == "self_reported"

    def test_questions_answered_count_is_accurate(self):
        answered = {"income": 3000, "housing_cost": 1200}
        result = self_report_confidence(answered)
        assert result["questions_answered"] == 2


class TestBlendExpenses:

    def test_verified_transaction_data_overrides_self_report(self):
        self_reported = {"housing": 1200.0, "food": 400.0}
        transaction_derived = {"housing": 1150.0}
        blended = blend_expenses(
            self_reported, transaction_derived,
            transaction_verified_categories={"housing"},
        )
        assert blended["housing"] == 1150.0  # transaction data wins
        assert blended["food"] == 400.0      # self-report survives, untouched

    def test_unverified_transaction_data_does_not_override(self):
        """A category present in transaction_derived but NOT in the
        verified set (e.g. a single ambiguous occurrence) must not
        silently replace a self-reported figure."""
        self_reported = {"insurance": 50.0}
        transaction_derived = {"insurance": 5.0}  # a single stray transaction, unverified
        blended = blend_expenses(
            self_reported, transaction_derived,
            transaction_verified_categories=set(),
        )
        assert blended["insurance"] == 50.0  # self-report retained

    def test_self_reported_only_category_survives_with_no_transaction_data(self):
        self_reported = {"housing": 1200.0, "discretionary": 200.0}
        blended = blend_expenses(self_reported, {}, set())
        assert blended["discretionary"] == 200.0

# Section 2 — the deterministic free-text tier
#
# These cover the layer that decides whether an LLM call is needed at all. It
# matters that they are cheap and reproducible: this is the path serving the
# customers the system knows least about, and a questionnaire whose common
# case depends on a live model is one that fails exactly when the provider
# does.

from agents.budget_questionnaire import (  # noqa: E402
    detect_stop_intent,
    is_plausible,
    is_skip_answer,
    is_zero_answer,
    parse_money_answer,
)


class TestMoneyParsing:

    @pytest.mark.parametrize("text,expected", [
        ("1200", 1200.0),
        ("€1,200", 1200.0),
        ("about 1200 a month", 1200.0),
        ("£950.50", 950.50),
        ("1.2k", 1200.0),
        ("2 grand", 2000.0),
        ("I pay 1,450 in rent", 1450.0),
    ])
    def test_common_shapes_parse_without_an_llm(self, text, expected):
        value, basis = parse_money_answer(text)
        assert value == pytest.approx(expected)
        assert basis in {"single_figure", "range_midpoint"}

    def test_a_range_becomes_its_midpoint(self):
        value, basis = parse_money_answer("somewhere between 400 and 600")
        assert value == pytest.approx(500.0)
        assert basis == "range_midpoint"

    def test_none_is_a_real_answer_of_zero_not_a_skip(self):
        """
        A customer with no debt and a customer who won't discuss their debt
        are in different positions; their budgets should not look identical.
        """
        value, basis = parse_money_answer("none")
        assert value == 0.0
        assert basis == "explicit_zero"
        assert is_zero_answer("I don't have any") is True

    def test_a_decline_is_a_skip_with_no_value(self):
        value, basis = parse_money_answer("no idea")
        assert value is None
        assert basis == "skip"
        assert is_skip_answer("not sure") is True

    def test_multiple_unrelated_numbers_are_refused_rather_than_guessed(self):
        """
        Picking one of two unrelated numbers would be a coin flip written
        into someone's budget. Refusing escalates it, which is correct.
        """
        value, basis = parse_money_answer("I moved in in 2019 and pay 1200")
        assert value is None
        assert basis == "unparsed"

    def test_a_reply_with_no_figure_at_all_is_unparsed_not_zero(self):
        value, basis = parse_money_answer("it's whatever the going rate is")
        assert value is None
        assert basis == "unparsed"


class TestStopIntentDetection:

    @pytest.mark.parametrize("text", [
        "I'd rather not answer more questions",
        "stop asking",
        "that's enough",
        "no more questions",
        "can we stop",
        "let's continue this later",
        "not right now",
    ])
    def test_recognised_stop_phrasings_need_no_llm(self, text):
        assert detect_stop_intent(text) is True

    @pytest.mark.parametrize("text", [
        "1200",
        "about 450 a month",
        "none",
        "no idea",
    ])
    def test_a_usable_answer_is_definitionally_not_a_walkout(self, text):
        """
        Returning False here (rather than None) is what keeps the common
        path free — a message containing an answer never needs a model to
        confirm the customer is still engaging.
        """
        assert detect_stop_intent(text) is False

    @pytest.mark.parametrize("text", [
        "hmm",
        "this is a lot",
        "why do you need all this",
    ])
    def test_genuinely_ambiguous_replies_return_none_for_escalation(self, text):
        """
        None, not False. Silently reading "this is a lot" as consent to
        continue is precisely the behaviour the stop rule exists to prevent
        — the caller escalates these to one LLM call instead.
        """
        assert detect_stop_intent(text) is None

    def test_a_skip_phrase_is_not_a_stop_phrase(self):
        """The distinction that keeps the form from feeling hostile."""
        assert detect_stop_intent("no idea") is False
        assert is_skip_answer("no idea") is True


class TestPlausibilityBounds:
    """
    These bound what an LLM is allowed to propose, not what a customer is
    allowed to have. A monthly rent of 120,000 means the model misread
    something; accepting it produces a confidently wrong budget, whereas
    rejecting it costs one repeated question.
    """

    def test_an_order_of_magnitude_error_is_rejected(self):
        assert is_plausible("housing_cost", 1200.0) is True
        assert is_plausible("housing_cost", 120_000.0) is False

    def test_leftover_may_legitimately_be_negative(self):
        """
        Negative disposable income is a real and important situation, not
        a parse error — bounding it at zero would erase the customers who
        most need the analysis.
        """
        assert is_plausible("rough_monthly_leftover", -400.0) is True

    def test_large_recurring_items_is_bounded_as_an_annual_figure(self):
        """The question asks for a yearly total, so the bound is yearly."""
        assert is_plausible("large_recurring_items", 24_000.0) is True


class TestPerSittingCapAndSkips:

    def test_per_sitting_count_is_separate_from_slots_filled(self):
        """
        They diverge whenever slots were seeded from an earlier conversation
        (income reused from risk profiling) or skipped. The cap is about not
        exhausting the customer, so it must count what they were actually
        asked.
        """
        answered = {
            "income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500,
            "food_spend": 400,
        }
        # Four slots filled but only two questions actually put to them:
        # nowhere near the cap, and coverage not yet met -> keep going.
        assert next_question(answered, questions_asked_this_sitting=2) is not None
        # Same answers, but they have already been asked seven times.
        assert next_question(answered, questions_asked_this_sitting=7) is None

    def test_a_skipped_slot_is_never_re_asked(self):
        answered = {"income": 3000, "housing_cost": 1200, "rough_monthly_leftover": 500}
        following = next_question(answered, skipped_slots={"food_spend"})
        assert following is not None
        assert following.slot_name != "food_spend"

    def test_a_skipped_mandatory_question_cannot_trap_the_loop(self):
        """
        Without this, declining a mandatory question would re-ask it forever
        and the questionnaire could never end. It moves on to the optionals
        instead — note it does NOT become "sufficient": a skipped mandatory
        item is resolved, not answered, and there is still no housing figure
        to build a budget from.
        """
        skipped = {"housing_cost", "rough_monthly_leftover"}
        following = next_question({"income": 3000}, skipped_slots=skipped)
        assert following is not None
        assert following.slot_name not in skipped
        assert following.mandatory is False

    def test_stop_ends_the_asking_without_manufacturing_sufficiency(self):
        """
        The deliberate split: customer_wants_to_stop governs whether we keep
        ASKING; it has no bearing on whether we have enough to say anything.
        Collapsing the two would either badger someone who asked to stop, or
        hand them a budget built on one answer.
        """
        barely_started = {"income": 3000}
        assert next_question(barely_started, customer_wants_to_stop=True) is None
        assert is_sufficient(barely_started, customer_wants_to_stop=True) is False