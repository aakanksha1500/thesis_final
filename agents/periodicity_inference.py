"""
Recurring-expense periodicity inference — Section 3 of the production
readiness review (data/../production_readiness_review.md if you still
have it; the short version: a single observed insurance payment could be
monthly, quarterly, half-yearly, annual, or one-time, and assuming any
one of those without evidence is a silent guess dressed up as an
analysis).

Deliberately pure Python, no LLM call. The reasoning here needs to be
reproducible and auditable — the same transaction history should always
produce the same answer. Where an LLM belongs is downstream of this:
phrasing the clarifying question naturally and interpreting the
customer's free-text reply, never deciding the ambiguity itself.

WHERE THIS FITS
    BudgetAgent calls infer_periodicity() per category before treating a
    category's observed amount as a reliable monthly figure. Designed to
    be reusable by InvestmentAgent too, for SIP/contribution detection —
    same shape of problem (a single contribution seen once, periodicity
    unknown), so no reason to duplicate the logic there.

THE THREE-STEP LOGIC
    1. Occurrence count + gap consistency: 2+ occurrences with a
       consistent gap between them -> confidently infer the period,
       no need to ask. Inconsistent gaps, or a single occurrence,
       -> genuinely ambiguous.
    2. Materiality: is the amount large enough (relative to income)
       that getting the period wrong would meaningfully distort a
       budget? If not, don't interrupt the conversation over it.
    3. Category prior (weak signal only): used ONLY to phrase a better
       default in the clarifying question ("most customers pay this
       quarterly") -- never to skip asking, never to silently assume.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

PRIORS_PATH = Path(settings.budget.periodicity_priors_path)


@dataclass(frozen=True)
class CategoryPrior:
    """
    One category's typical-periodicity prior, with the provenance needed to
    judge how much weight it deserves. See data/periodicity_priors.json's
    _meta.field_definitions for what each field means.
    """
    category: str
    ambiguous: bool
    modal_period: str | None      # None = no defensible default; ask without one
    prior_strength: str           # "weak" (illustrative) | "calibrated" (fitted)
    source: str
    source_detail: str = ""
    last_reviewed: str = ""
    notes: str = ""

    @property
    def is_evidence_based(self) -> bool:
        """
        False for every prior currently shipped. Exists so the distinction
        between an illustrative default and a fitted one is a value a caller
        can branch on and an auditor can read, not a comment in a source file.
        """
        return self.prior_strength == "calibrated" and self.source != "illustrative_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "modal_period": self.modal_period,
            "prior_strength": self.prior_strength,
            "source": self.source,
            "source_detail": self.source_detail,
            "last_reviewed": self.last_reviewed,
            "is_evidence_based": self.is_evidence_based,
        }


# Fallback used when the JSON is missing or malformed. Deliberately identical
# in VALUES to the shipped file — a bad override should degrade to the
# documented defaults, not to different behaviour, and certainly not to a
# crash inside a customer session.
_BUILTIN_PRIORS: dict[str, CategoryPrior] = {
    category: CategoryPrior(
        category=category, ambiguous=True, modal_period=modal_period,
        prior_strength="weak", source="illustrative_default",
        source_detail="Built-in fallback: data/periodicity_priors.json was unreadable.",
    )
    for category, modal_period in {
        "insurance": "quarterly",
        "subscription": "monthly",
        "school_fees": "annual",
        "sip": "monthly",
        "pension": "monthly",
        "membership": "annual",
        "maintenance": "monthly",
        "property_tax": "annual",
        "estimated_tax": None,
    }.items()
}


def load_priors(path: Path | str | None = None) -> dict[str, CategoryPrior]:
    """
    Read the priors file. Never raises: a missing or malformed file logs a
    warning and returns _BUILTIN_PRIORS, because failing a customer's budget
    analysis over a phrasing hint would be a wildly disproportionate response
    to a bad config file.
    """
    prior_path = Path(path) if path is not None else PRIORS_PATH
    try:
        raw = json.loads(prior_path.read_text(encoding="utf-8"))
        entries = raw["priors"]
    except Exception as exc:
        logger.warning(
            f"[periodicity] Could not load priors from {prior_path} ({exc}) "
            f"— falling back to built-in illustrative defaults. Phrasing only; "
            f"no inference behaviour changes."
        )
        return dict(_BUILTIN_PRIORS)

    loaded: dict[str, CategoryPrior] = {}
    for category, entry in entries.items():
        try:
            loaded[category] = CategoryPrior(
                category=category,
                ambiguous=bool(entry.get("ambiguous", True)),
                modal_period=entry.get("modal_period"),
                prior_strength=str(entry.get("prior_strength", "weak")),
                source=str(entry.get("source", "unknown")),
                source_detail=str(entry.get("source_detail", "")),
                last_reviewed=str(entry.get("last_reviewed", "")),
                notes=str(entry.get("notes", "")),
            )
        except Exception as exc:
            logger.warning(f"[periodicity] Skipping malformed prior {category!r}: {exc}")
    return loaded or dict(_BUILTIN_PRIORS)


PRIORS: dict[str, CategoryPrior] = load_priors()

# Backwards-compatible flat view: category -> modal period string. Kept
# because existing callers and tests read it, and because a plain mapping is
# the right shape for anything that only needs the value. Anything that needs
# to know how much the value is worth should read PRIORS instead.

CATEGORY_PERIODICITY_PRIORS: dict[str, str] = {
    category: prior.modal_period
    for category, prior in PRIORS.items()
    if prior.modal_period is not None
}

# Recognised period labels and the day-gap band that counts as "this
# period" between two consecutive occurrences. Bands are tolerant
# (a "monthly" bill doesn't land on exactly the same day every time)
# but don't overlap, so a gap is never ambiguously classified as two
# different periods at once.
KNOWN_PERIODS: dict[str, tuple[int, int]] = {
    "weekly": (5, 9),
    "monthly": (25, 35),
    "quarterly": (80, 100),
    "half_yearly": (170, 190),
    "annual": (350, 380),
}

# How many of each period fall in a month — the conversion from "€600 every
# quarter" to "€200/month". Only ever applied to a CONFIDENTLY INFERRED
# period (needs_clarification False, inferred_period not None). Applying it
# to a suggested_default would be precisely the silent assumption this
# module exists to prevent: it would take an illustrative phrasing hint and
# turn it into a number in someone's budget.
PERIOD_TO_MONTHLY_DIVISOR: dict[str, float] = {
    "weekly": 52 / 12,
    "monthly": 1.0,
    "quarterly": 1 / 3,
    "half_yearly": 1 / 6,
    "annual": 1 / 12,
}

# Categories where a single occurrence is genuinely ambiguous enough to
# be worth asking about (given materiality) -- generalised from the
# insurance case. Categories NOT in this set (housing, food, most
# discretionary spend) don't need this treatment: their periodicity is
# either obvious from the category itself (housing is monthly) or not
# consequential enough to matter (a one-off discretionary purchase).
#
# Derived from the same priors file rather than declared separately, so
# "which categories are ambiguous" and "what do we assume about them" can
# never drift apart — they are one declaration, reviewed together.
AMBIGUOUS_CATEGORIES: frozenset[str] = frozenset(
    category for category, prior in PRIORS.items() if prior.ambiguous
)

@dataclass
class PeriodicityResult:
    category: str
    occurrence_count: int
    inferred_period: str | None       # None if not confidently inferred
    confidence: float                 # 0.0-1.0
    is_material: bool
    needs_clarification: bool
    suggested_default: str | None     # weak-prior phrasing hint, not an assumption
    reason: str                       # human-readable, for logging/audit
    prior_provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "occurrence_count": self.occurrence_count,
            "inferred_period": self.inferred_period,
            "confidence": round(self.confidence, 3),
            "is_material": self.is_material,
            "needs_clarification": self.needs_clarification,
            "suggested_default": self.suggested_default,
            "reason": self.reason,
            "prior_provenance": self.prior_provenance,
        }


def _classify_gap(days: float) -> str | None:
    for label, (lo, hi) in KNOWN_PERIODS.items():
        if lo <= days <= hi:
            return label
    return None


def infer_periodicity(
    category: str,
    transactions: list[dict[str, Any]],
    monthly_income: float,
    materiality_threshold_pct: float = 5.0,
) -> PeriodicityResult:
    """
    transactions: THIS category's transactions only (pre-filtered by the
        caller), each a dict with "date" ("YYYY-MM-DD") and "amount".
        Order doesn't matter, sorted internally.
    monthly_income: used for the materiality check.
    materiality_threshold_pct: a category counts as material if its
        average per-occurrence amount is at least this percentage of
        monthly income. Default 5% -- roughly "big enough that guessing
        wrong meaningfully distorts a budget", not a cited figure.
    """
    if not transactions:
        return PeriodicityResult(
            category=category, occurrence_count=0, inferred_period=None,
            confidence=0.0, is_material=False, needs_clarification=False,
            suggested_default=None,
            reason="No transactions in this category.",
        )

    dates = sorted(datetime.strptime(t["date"], "%Y-%m-%d") for t in transactions)
    amounts = [t["amount"] for t in transactions]
    avg_amount = sum(amounts) / len(amounts)
    is_material = (
        monthly_income > 0
        and (avg_amount / monthly_income) * 100 >= materiality_threshold_pct
    )
    prior_entry = PRIORS.get(category)
    prior = prior_entry.modal_period if prior_entry else None
    provenance = prior_entry.to_dict() if prior_entry else None

    if len(dates) == 1:
        if category not in AMBIGUOUS_CATEGORIES or not is_material:
            return PeriodicityResult(
                category=category, occurrence_count=1, inferred_period=None,
                confidence=0.0, is_material=is_material,
                needs_clarification=False, suggested_default=prior, prior_provenance=provenance,
                reason=(
                    "Single occurrence, but immaterial or not a "
                    "known-ambiguous category — not worth interrupting "
                    "the conversation over."
                ),
            )
        return PeriodicityResult(
            category=category, occurrence_count=1, inferred_period=None,
            confidence=0.0, is_material=True, needs_clarification=True,
            suggested_default=prior, prior_provenance=provenance,
            reason=(
                "Single occurrence of a material, typically-ambiguous "
                "category — periodicity genuinely unknown from one "
                "data point alone."
            ),
        )

    # 2+ occurrences: do consecutive gaps agree on one period?
    gaps_days = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    gap_labels = [_classify_gap(g) for g in gaps_days]

    if gap_labels and all(
        label is not None and label == gap_labels[0] for label in gap_labels
    ):
        return PeriodicityResult(
            category=category, occurrence_count=len(dates),
            inferred_period=gap_labels[0], confidence=0.9,
            is_material=is_material, needs_clarification=False,
            suggested_default=None, prior_provenance=provenance,
            reason=(
                f"{len(dates)} occurrences with a consistent "
                f"~{gap_labels[0]} gap ({gaps_days} days) — confidently "
                f"inferred, no need to ask."
            ),
        )

    if not is_material:
        return PeriodicityResult(
            category=category, occurrence_count=len(dates),
            inferred_period=None, confidence=0.2, is_material=False,
            needs_clarification=False, suggested_default=prior,  prior_provenance=provenance,
            reason=(
                f"{len(dates)} occurrences but inconsistent gaps "
                f"({gaps_days} days) — inconclusive, but immaterial so "
                f"not worth asking about."
            ),
        )
    return PeriodicityResult(
        category=category, occurrence_count=len(dates),
        inferred_period=None, confidence=0.2, is_material=True,
        needs_clarification=True, suggested_default=prior,  prior_provenance=provenance,
        reason=(
            f"{len(dates)} occurrences but inconsistent gaps "
            f"({gaps_days} days) — can't confidently infer a single "
            f"period, and the amount is material enough to matter."
        ),
    )


def build_clarifying_question(result: PeriodicityResult) -> str | None:
    """
    Deterministic template, not an LLM call — the question text itself
    doesn't need creativity, and a fixed template is easier to audit and
    test than a freshly-generated one each time. If BudgetAgent's
    narrative layer wants to soften the phrasing for tone, it can rewrite
    around this, but the FACTS in the question (category, candidate
    periods, suggested default) come from here, not from the LLM.
    """
    if not result.needs_clarification:
        return None
    category_label = result.category.replace("_", " ")
    options = "monthly, quarterly, half-yearly, or annual"
    if result.suggested_default:
        return (
            f"I see a {category_label} payment, but I can't tell how "
            f"often it recurs from the data alone. Is it {options}? "
            f"(most customers' {category_label} payments are "
            f"{result.suggested_default.replace('_', ' ')})"
        )
    return (
        f"I see a {category_label} payment, but I can't tell how often "
        f"it recurs from the data alone. Is it {options}?"
    )