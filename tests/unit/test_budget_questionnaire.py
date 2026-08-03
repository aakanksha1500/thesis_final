"""
Tests for agents/budget_questionnaire.py — Section 2's insufficient-data
questionnaire framework. Pure logic, no LLM, no I/O.
"""
from __future__ import annotations

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