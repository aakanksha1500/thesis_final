"""
Tests for agents/periodicity_inference.py — Section 3's ambiguity-
resolution framework. Pure logic, no LLM, no I/O.
"""
from __future__ import annotations

import pytest

from agents.periodicity_inference import (
    AMBIGUOUS_CATEGORIES,
    CATEGORY_PERIODICITY_PRIORS,
    PRIORS,
    PeriodicityResult,
    build_clarifying_question,
    infer_periodicity,
    load_priors,
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

class TestPriorsAreAdvisoryOnly:
    """
    The invariant that makes shipping an illustrative prior defensible:
    a prior may influence PHRASING and RANKING, and nothing else.

    Referenced by name from agents/periodicity_inference.py's header and
    from data/periodicity_priors.json's _meta.invariant — if this class is
    ever deleted, those two claims become unenforced assertions in a
    comment, which is the exact failure mode the provenance work exists
    to fix.
    """

    def test_a_prior_never_suppresses_a_clarifying_question(self):
        """
        insurance has a 'quarterly' prior. A single material occurrence must
        still ask — using the prior to skip the question would be the silent
        assumption the whole component exists to prevent.
        """
        assert PRIORS["insurance"].modal_period == "quarterly"
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        assert result.needs_clarification is True
        assert result.inferred_period is None

    def test_a_prior_is_never_returned_as_an_inferred_period(self):
        """
        suggested_default and inferred_period are different fields with
        different authority. Collapsing them would let a phrasing hint be
        read downstream as an observation.
        """
        result = infer_periodicity("pension", [_txn("2025-03-01", 900.0)], INCOME)
        assert result.suggested_default == "monthly"
        assert result.inferred_period is None
        assert result.confidence == 0.0

    def test_observed_evidence_overrides_a_contradicting_prior(self):
        """
        school_fees' prior is 'annual'. Three consistent monthly gaps say
        monthly, and observation must win — a prior that could override
        evidence would be worse than no prior at all.
        """
        assert PRIORS["school_fees"].modal_period == "annual"
        txns = [_txn("2025-01-10", 500.0), _txn("2025-02-10", 500.0),
                _txn("2025-03-10", 500.0), _txn("2025-04-10", 500.0)]
        result = infer_periodicity("school_fees", txns, INCOME)
        assert result.inferred_period == "monthly"
        assert result.needs_clarification is False

    def test_every_shipped_prior_declares_itself_not_evidence_based(self):
        """
        Nothing in this project is calibrated against a real population, and
        the code must say so rather than leaving it to a comment. If someone
        later fits real priors, this test failing is the correct prompt to
        update the thesis's limitations section too.
        """
        assert PRIORS, "priors failed to load at all"
        for category, prior in PRIORS.items():
            assert prior.is_evidence_based is False, (
                f"{category} claims to be evidence-based — if that's now true, "
                f"say where the evidence came from and update the write-up"
            )
            assert prior.source == "illustrative_default"

    def test_provenance_travels_with_the_result(self):
        """
        An auditor reading a stored recommendation six months later must be
        able to tell that 'most customers pay this quarterly' rested on an
        illustrative default. That requires the provenance to be IN the
        payload, not only in the source file.
        """
        result = infer_periodicity("insurance", [_txn("2025-06-01", 400.0)], INCOME)
        provenance = result.to_dict()["prior_provenance"]
        assert provenance["source"] == "illustrative_default"
        assert provenance["is_evidence_based"] is False
        assert provenance["last_reviewed"]

    def test_a_null_modal_period_asks_without_inventing_a_default(self):
        """
        estimated_tax deliberately has modal_period null. The correct
        behaviour is a question with no 'most customers' clause — not a
        made-up default chosen to make the sentence read better.
        """
        result = infer_periodicity("estimated_tax", [_txn("2025-06-01", 500.0)], INCOME)
        assert result.suggested_default is None
        question = build_clarifying_question(result)
        assert question is not None
        assert "most customers" not in question


class TestPriorsLoading:

    def test_missing_priors_file_degrades_to_builtin_defaults(self, tmp_path):
        """
        A bad config file must not fail a customer's budget analysis. The
        fallback values are identical to the shipped ones on purpose, so a
        broken override changes provenance labelling, not behaviour.
        """
        loaded = load_priors(tmp_path / "does_not_exist.json")
        assert loaded["insurance"].modal_period == "quarterly"
        assert loaded["insurance"].source == "illustrative_default"

    def test_priors_can_be_overridden_without_a_code_change(self, tmp_path):
        """
        The point of externalising the file: an institution with real data
        swaps in fitted priors and marks them calibrated, and the system
        reports them as evidence-based from then on.
        """
        import json  # noqa: PLC0415
        path = tmp_path / "custom_priors.json"
        path.write_text(json.dumps({
            "priors": {
                "insurance": {
                    "ambiguous": True, "modal_period": "annual",
                    "prior_strength": "calibrated", "source": "internal_txn_study_2026",
                    "source_detail": "Fitted on 1.2M policy payments.",
                    "last_reviewed": "2026-01-15",
                }
            }
        }), encoding="utf-8")
        loaded = load_priors(path)
        assert loaded["insurance"].modal_period == "annual"
        assert loaded["insurance"].is_evidence_based is True

    def test_ambiguous_categories_are_derived_from_the_same_file(self):
        """
        'Which categories are ambiguous' and 'what do we assume about them'
        are one declaration, so they cannot drift apart.
        """
        assert AMBIGUOUS_CATEGORIES == frozenset(
            c for c, p in PRIORS.items() if p.ambiguous
        )