"""
Tests for agents/data_sufficiency.py — Section 1's coverage + density
confidence framework. Pure logic, no LLM, no I/O.
"""
from __future__ import annotations

from agents.data_sufficiency import (
    TIER_DISCLOSURE,
    assess_data_sufficiency,
)

INCOME = 4000.0


def _txn(date: str, category: str, amount: float) -> dict:
    return {"date": date, "category": category, "amount": amount}


def _monthly_series(category: str, start_year: int, start_month: int,
                     n_months: int, amount: float = 100.0) -> list[dict]:
    """n_months of one transaction per month in `category`, starting at
    (start_year, start_month)."""
    out = []
    y, m = start_year, start_month
    for _ in range(n_months):
        out.append(_txn(f"{y}-{m:02d}-05", category, amount))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


class TestEmptyAndCoverageTiers:

    def test_empty_transactions_is_insufficient(self):
        result = assess_data_sufficiency([])
        assert result.coverage_tier == "insufficient"
        assert result.confidence_score == 0.0
        assert result.available_months == 0

    def test_single_month_is_minimal_tier(self):
        txns = [_txn("2025-06-05", "housing", 1000.0), _txn("2025-06-20", "food", 100.0)]
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "minimal"
        assert result.available_months == 1
        assert result.coverage_score == 0.2

    def test_two_months_is_low_tier(self):
        txns = _monthly_series("housing", 2025, 5, 2)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "low"

    def test_four_months_is_medium_tier(self):
        txns = _monthly_series("housing", 2025, 3, 4)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "medium"

    def test_seven_months_is_medium_high_tier(self):
        txns = _monthly_series("housing", 2024, 12, 7)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "medium_high"

    def test_twelve_months_is_high_tier(self):
        txns = _monthly_series("housing", 2024, 7, 12)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "high"
        assert result.coverage_score == 1.0

    def test_boundary_just_under_a_tier_uses_the_lower_one(self):
        # 6 months should be "medium" (needs 7 for medium_high)
        txns = _monthly_series("housing", 2025, 1, 6)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "medium"

    def test_more_than_twelve_months_still_high_not_a_new_tier(self):
        txns = _monthly_series("housing", 2023, 1, 24)
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "high"
        assert result.coverage_score == 1.0


class TestDensity:

    def test_fully_dense_history_has_density_one(self):
        txns = _monthly_series("housing", 2024, 7, 12)  # one txn every month
        result = assess_data_sufficiency(txns)
        assert result.density_score == 1.0
        assert result.confidence_score == result.coverage_score  # coverage * 1.0

    def test_dormant_account_long_window_low_density_flagged(self):
        """
        The Section 2 motivating case, handled by the SAME mechanism as
        Section 1's coverage tiers rather than a separate system: 12
        months of window, but the account only shows activity in 3 of
        those months (salary paid elsewhere, rarely used).
        """
        txns = (
            [_txn("2024-07-05", "housing", 900.0)]
            + [_txn("2024-11-05", "housing", 900.0)]
            + [_txn("2025-06-05", "housing", 900.0)]
        )
        result = assess_data_sufficiency(txns)
        assert result.coverage_tier == "high"  # 12-month SPAN exists
        assert result.active_months == 3
        assert result.density_score < 0.3  # but density is poor
        assert result.confidence_score < 0.3  # so overall confidence must be low too

    def test_active_months_never_exceeds_available_months_in_density_calc(self):
        """Sanity: density_score is capped at 1.0 even if the counting
        logic were ever off by one somewhere."""
        txns = _monthly_series("housing", 2025, 1, 3)
        result = assess_data_sufficiency(txns)
        assert result.density_score <= 1.0


class TestCategoryVerification:

    def test_non_ambiguous_category_verified_after_single_observation(self):
        txns = [_txn("2025-06-05", "housing", 1000.0)]
        result = assess_data_sufficiency(txns)
        assert result.category_sufficiency["housing"].verified is True

    def test_ambiguous_category_single_occurrence_material_unverified(self):
        txns = [_txn("2025-06-01", "insurance", 500.0)]  # 12.5% of income -- material
        result = assess_data_sufficiency(txns, monthly_income=INCOME)
        assert result.category_sufficiency["insurance"].verified is False

    def test_ambiguous_category_without_income_is_unverified(self):
        """No income given -> can't assess materiality/periodicity ->
        conservatively unverified, never silently assumed fine."""
        txns = [_txn("2025-06-01", "insurance", 500.0)]
        result = assess_data_sufficiency(txns)  # no monthly_income
        assert result.category_sufficiency["insurance"].verified is False
        assert "no income" in result.category_sufficiency["insurance"].reason.lower()

    def test_ambiguous_category_confidently_inferred_period_is_verified(self):
        """Reuses periodicity_inference directly: a consistently-monthly
        'subscription' should come out verified, not blanket-flagged
        just for being in AMBIGUOUS_CATEGORIES."""
        txns = _monthly_series("subscription", 2025, 1, 3, amount=15.0)
        result = assess_data_sufficiency(txns, monthly_income=INCOME)
        assert result.category_sufficiency["subscription"].verified is True

    def test_category_never_observed_not_present_in_results(self):
        txns = [_txn("2025-06-05", "housing", 1000.0)]
        result = assess_data_sufficiency(txns, monthly_income=INCOME)
        assert "insurance" not in result.category_sufficiency


class TestDisclosureText:

    def test_disclosure_matches_tier(self):
        txns = [_txn("2025-06-05", "housing", 1000.0)]
        result = assess_data_sufficiency(txns)
        assert result.disclosure == TIER_DISCLOSURE["minimal"]

    def test_disclosure_interpolates_month_count(self):
        txns = _monthly_series("housing", 2025, 1, 4)
        result = assess_data_sufficiency(txns)
        assert "4 months" in result.disclosure

    def test_high_tier_disclosure_has_no_month_count_placeholder_issue(self):
        txns = _monthly_series("housing", 2024, 7, 12)
        result = assess_data_sufficiency(txns)
        assert result.disclosure == TIER_DISCLOSURE["high"]  # no {months} in this template


class TestAsOfParameter:

    def test_as_of_overrides_latest_transaction_date(self):
        txns = [_txn("2025-01-05", "housing", 1000.0)]
        result = assess_data_sufficiency(txns, as_of="2025-06-30")
        # Jan through June = 6 months span, even though only 1 txn exists
        assert result.available_months == 6
        assert result.active_months == 1


class TestSerialization:

    def test_to_dict_round_trips(self):
        txns = _monthly_series("housing", 2024, 7, 12)
        result = assess_data_sufficiency(txns, monthly_income=INCOME)
        d = result.to_dict()
        assert d["coverage_tier"] == "high"
        assert isinstance(d["confidence_score"], float)
        assert "housing" in d["category_sufficiency"]
        assert d["category_sufficiency"]["housing"]["verified"] is True