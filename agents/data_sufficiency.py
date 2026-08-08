"""
Scores how much we actually know about a customer, based on data coverage and density.

TWO DIMENSIONS
    coverage  — how many months of history exist at all. Determines
                what's even theoretically observable: you cannot see an
                annual cost in 3 months of data, full stop, no matter
                how good the data is.
    density   — what fraction of the available months actually contain
                activity. Determines whether what you CAN see is
                trustworthy: a dormant account with 12 months of window
                but activity in 3 of them is not a reliable 12-month
                customer for these purposes, even though coverage alone
                would say HIGH.

    confidence_score = coverage_score * density_score
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agents.periodicity_inference import AMBIGUOUS_CATEGORIES, infer_periodicity

# (min_available_months, tier_name, coverage_score) — the LAST entry
# whose min_available_months is met wins (a staircase, not a range
# lookup), so this list must stay sorted ascending by min_available_months.
_COVERAGE_TIERS: list[tuple[int, str, float]] = [
    (0, "insufficient", 0.0),
    (1, "minimal", 0.2),
    (2, "low", 0.4),
    (4, "medium", 0.6),
    (7, "medium_high", 0.8),
    (12, "high", 1.0),
]

TIER_DISCLOSURE: dict[str, str] = {
    "insufficient": (
        "There isn't enough transaction history yet to generate a "
        "reliable budget."
    ),
    "minimal": (
        "This is based on a single month of history — only your overall "
        "income versus spending is shown here; category-level detail "
        "and irregular costs like insurance aren't captured yet."
    ),
    "low": (
        "This is based on {months} months of history — frequent "
        "spending like groceries should be fairly accurate; less "
        "frequent costs are not yet verified."
    ),
    "medium": (
        "This is based on {months} months of history — most regular "
        "monthly costs are well-established; annual or irregular costs "
        "may not be fully captured yet."
    ),
    "medium_high": (
        "This is based on {months} months of history — nearly all your "
        "regular spending is well-established; a small number of "
        "infrequent costs are still being verified."
    ),
    "high": (
        "This is based on a full year of history, covering annual costs "
        "like insurance."
    ),
}


def _coverage_tier(available_months: int) -> tuple[str, float]:
    tier, score = _COVERAGE_TIERS[0][1], _COVERAGE_TIERS[0][2]
    for min_months, name, coverage_score in _COVERAGE_TIERS:
        if available_months >= min_months:
            tier, score = name, coverage_score
        else:
            break
    return tier, score


@dataclass
class CategorySufficiency:
    category: str
    months_observed: int
    verified: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "months_observed": self.months_observed,
            "verified": self.verified,
            "reason": self.reason,
        }


@dataclass
class DataSufficiencyResult:
    available_months: int
    active_months: int
    coverage_tier: str
    coverage_score: float
    density_score: float
    confidence_score: float
    category_sufficiency: dict[str, CategorySufficiency]
    disclosure: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_months": self.available_months,
            "active_months": self.active_months,
            "coverage_tier": self.coverage_tier,
            "coverage_score": round(self.coverage_score, 3),
            "density_score": round(self.density_score, 3),
            "confidence_score": round(self.confidence_score, 3),
            "category_sufficiency": {
                k: v.to_dict() for k, v in self.category_sufficiency.items()
            },
            "disclosure": self.disclosure,
        }


def assess_data_sufficiency(
    transactions: list[dict[str, Any]],
    monthly_income: float | None = None,
    as_of: str | None = None,
) -> DataSufficiencyResult:
    """
    Scores how much of a budget we can stand behind, from coverage (months of history) times
    density (how many of those months have acitivity).
    """
    if not transactions:
        return DataSufficiencyResult(
            available_months=0, active_months=0,
            coverage_tier="insufficient", coverage_score=0.0,
            density_score=0.0, confidence_score=0.0,
            category_sufficiency={},
            disclosure=TIER_DISCLOSURE["insufficient"],
        )

    dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in transactions]
    end_date = datetime.strptime(as_of, "%Y-%m-%d") if as_of else max(dates)
    start_date = min(dates)

    available_months = (
        (end_date.year - start_date.year) * 12
        + (end_date.month - start_date.month) + 1
    )

    active_month_keys = {(d.year, d.month) for d in dates}
    active_months = len(active_month_keys)
    density_score = min(1.0, active_months / max(available_months, 1))

    coverage_tier, coverage_score = _coverage_tier(available_months)
    confidence_score = coverage_score * density_score

    by_category: dict[str, list[dict[str, Any]]] = {}
    for t in transactions:
        by_category.setdefault(t["category"], []).append(t)

    category_sufficiency: dict[str, CategorySufficiency] = {}
    for category, txns in by_category.items():
        months_observed = len({
            (datetime.strptime(t["date"], "%Y-%m-%d").year,
             datetime.strptime(t["date"], "%Y-%m-%d").month)
            for t in txns
        })
        if category in AMBIGUOUS_CATEGORIES:
            if monthly_income:
                result = infer_periodicity(category, txns, monthly_income)
                verified = (
                    not result.needs_clarification
                    and result.inferred_period is not None
                )
                reason = result.reason
            else:
                verified = False
                reason = (
                    "Ambiguous category and no income given to assess "
                    "materiality or periodicity."
                )
        else:
            verified = months_observed >= 1
            reason = (
                "Non-ambiguous category, observed at least once."
                if verified else
                "Not observed in the available history."
            )
        category_sufficiency[category] = CategorySufficiency(
            category=category, months_observed=months_observed,
            verified=verified, reason=reason,
        )

    disclosure = TIER_DISCLOSURE[coverage_tier].format(months=available_months)

    return DataSufficiencyResult(
        available_months=available_months,
        active_months=active_months,
        coverage_tier=coverage_tier,
        coverage_score=coverage_score,
        density_score=density_score,
        confidence_score=confidence_score,
        category_sufficiency=category_sufficiency,
        disclosure=disclosure,
    )