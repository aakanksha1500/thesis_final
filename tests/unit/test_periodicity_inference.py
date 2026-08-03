"""
Tests for agents/periodicity_inference.py — Section 3's ambiguity-
resolution framework. Pure logic, no LLM, no I/O.
"""
from __future__ import annotations

import pytest

from agents.periodicity_inference import (
    AMBIGUOUS_CATEGORIES,
    CATEGORY_PERIODICITY_PRIORS,
    PeriodicityResult,
    build_clarifying_question,
    infer_periodicity,
)

INCOME = 4000.0  # monthly, used across tests for materiality checks


def _txn(date: str, amount: float) -> dict:
    return {"date": date, "amount": amount}


class TestEmptyAndSingleOccurrence:

    def test_no_transactions_returns_zero_occurrences_no_clarification(self):
        result = infer_periodicity("insurance", [], INCOME)
        assert result.occurrence_count == 0
        assert result.needs_clarification is False
        assert result.inferred_period is None

    def test_single_occurrence_ambiguous_and_material_needs_clarification(self):
        # €400 on a €4000 income = 10% -- comfortably material
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        assert result.occurrence_count == 1
        assert result.is_material is True
        assert result.needs_clarification is True
        assert result.inferred_period is None
        assert result.confidence == 0.0

    def test_single_occurrence_ambiguous_but_immaterial_no_clarification(self):
        # €10 on a €4000 income = 0.25% -- not worth interrupting for
        result = infer_periodicity("subscription", [_txn("2025-06-01", 10.0)], INCOME)
        assert result.is_material is False
        assert result.needs_clarification is False

    def test_single_occurrence_non_ambiguous_category_never_asks(self):
        """Housing is obviously monthly by category — not in AMBIGUOUS_CATEGORIES,
        so even a large single occurrence shouldn't trigger a question."""
        assert "housing" not in AMBIGUOUS_CATEGORIES
        result = infer_periodicity("housing", [_txn("2025-06-01", 2000.0)], INCOME)
        assert result.needs_clarification is False

    def test_materiality_threshold_is_configurable(self):
        # €150 / €4000 = 3.75% -- below default 5% threshold
        result_default = infer_periodicity("insurance", [_txn("2025-06-01", 150.0)], INCOME)
        assert result_default.is_material is False

        result_lower_threshold = infer_periodicity(
            "insurance", [_txn("2025-06-01", 150.0)], INCOME,
            materiality_threshold_pct=3.0,
        )
        assert result_lower_threshold.is_material is True


class TestMultipleOccurrencesConsistentGaps:

    def test_two_occurrences_monthly_gap_confidently_inferred(self):
        result = infer_periodicity(
            "subscription",
            [_txn("2025-01-15", 15.0), _txn("2025-02-14", 15.0)],
            INCOME,
        )
        assert result.inferred_period == "monthly"
        assert result.confidence == 0.9
        assert result.needs_clarification is False

    def test_three_occurrences_consistent_quarterly_gap(self):
        result = infer_periodicity(
            "insurance",
            [_txn("2025-01-01", 400.0), _txn("2025-04-01", 400.0), _txn("2025-07-02", 400.0)],
            INCOME,
        )
        assert result.inferred_period == "quarterly"
        assert result.needs_clarification is False

    def test_two_occurrences_annual_gap_confidently_inferred(self):
        result = infer_periodicity(
            "insurance",
            [_txn("2024-06-01", 2000.0), _txn("2025-06-05", 2000.0)],
            INCOME,
        )
        assert result.inferred_period == "annual"
        assert result.needs_clarification is False

    def test_consistent_gap_still_confident_even_when_immaterial(self):
        """A confidently-inferred period doesn't need to ask regardless
        of materiality -- materiality only gates the AMBIGUOUS cases."""
        result = infer_periodicity(
            "subscription",
            [_txn("2025-01-15", 5.0), _txn("2025-02-14", 5.0)],
            INCOME,
        )
        assert result.needs_clarification is False
        assert result.inferred_period == "monthly"


class TestMultipleOccurrencesInconsistentGaps:

    def test_inconsistent_gaps_material_needs_clarification(self):
        result = infer_periodicity(
            "insurance",
            [_txn("2025-01-01", 500.0), _txn("2025-03-01", 500.0), _txn("2025-11-01", 500.0)],
            INCOME,
        )
        assert result.inferred_period is None
        assert result.is_material is True
        assert result.needs_clarification is True

    def test_inconsistent_gaps_immaterial_no_clarification(self):
        result = infer_periodicity(
            "subscription",
            [_txn("2025-01-01", 8.0), _txn("2025-03-01", 8.0), _txn("2025-11-01", 8.0)],
            INCOME,
        )
        assert result.needs_clarification is False

    def test_unrecognisable_gap_treated_as_inconsistent(self):
        """A 50-day gap doesn't match any KNOWN_PERIODS band -- must not
        be silently rounded to the nearest one."""
        result = infer_periodicity(
            "insurance",
            [_txn("2025-01-01", 500.0), _txn("2025-02-20", 500.0)],
            INCOME,
        )
        assert result.inferred_period is None
        assert result.needs_clarification is True  # material, ambiguous


class TestClarifyingQuestion:

    def test_none_when_no_clarification_needed(self):
        result = infer_periodicity(
            "subscription",
            [_txn("2025-01-15", 15.0), _txn("2025-02-14", 15.0)],
            INCOME,
        )
        assert build_clarifying_question(result) is None

    def test_includes_category_and_options_when_needed(self):
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        question = build_clarifying_question(result)
        assert question is not None
        assert "insurance" in question
        assert "monthly" in question and "quarterly" in question and "annual" in question

    def test_includes_suggested_default_when_a_prior_exists(self):
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        question = build_clarifying_question(result)
        assert CATEGORY_PERIODICITY_PRIORS["insurance"].replace("_", " ") in question

    def test_handles_ambiguous_category_with_no_prior(self):
        """estimated_tax is in AMBIGUOUS_CATEGORIES but deliberately has
        no entry in CATEGORY_PERIODICITY_PRIORS -- must not crash or
        silently fabricate a default."""
        assert "estimated_tax" in AMBIGUOUS_CATEGORIES
        assert "estimated_tax" not in CATEGORY_PERIODICITY_PRIORS
        result = infer_periodicity("estimated_tax", [_txn("2025-06-01", 500.0)], INCOME)
        question = build_clarifying_question(result)
        assert question is not None
        assert "estimated tax" in question


class TestResultSerialization:

    def test_to_dict_round_trips_all_fields(self):
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        d = result.to_dict()
        assert d["category"] == "insurance"
        assert d["occurrence_count"] == 1
        assert d["needs_clarification"] is True
        assert isinstance(d["confidence"], float)