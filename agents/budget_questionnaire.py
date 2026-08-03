"""
Insufficient-data questionnaire — Section 2 of the production readiness
review.

WHEN THIS APPLIES
    BudgetAgent.run() returns status="insufficient_history" when
    agents.data_sufficiency.assess_data_sufficiency() finds less than a
    month of real transaction history (or none at all — a brand new
    customer, or TransactionStore.lookup() returning [] for an unknown
    one). This module is what the conversation should do NEXT in that
    case instead of just reporting "insufficient" and stopping: ask a
    short, prioritised set of questions and build a budget from
    self-reported answers, honestly capped at a lower confidence
    ceiling than transaction-derived data ever gets.

REUSES THE EXISTING ELICITATION MECHANISM, DOESN'T DUPLICATE IT
    ConversationalAgent already tracks slots across turns
    (settings.conversational.tracked_slots, update_slots(),
    get_missing_slots()) for risk-profiling questions. The budget slot
    names below are meant to be added to that same tracked_slots list —
    "income" is already tracked for risk profiling and is reused
    directly here, not asked twice.

QUESTION SET AND ORDERING
    Mandatory (asked first, always, in this order):
      income               — reused from risk profiling if already known
      housing_cost         — largest, most stable category, highest
                              leverage for a first estimate
      rough_monthly_leftover — crude self-report sanity check against
                              income minus known costs
    Optional (asked in priority order — by typical share of household
    spend, HBS-anchored where a direct category exists):
      food_spend, utilities_spend, discretionary_spend,
      debt_repayments, large_recurring_items (insurance/subscriptions/
      school fees/etc. combined — the same categories Section 3 treats
      as periodicity-ambiguous when they show up in real transactions;
      self-reported here, so periodicity has to be asked directly
      rather than inferred from recurrence, since there's no
      transaction history to infer it from).

STOPPING CRITERIA
    All mandatory answered, AND EITHER:
      - cumulative weight of answered OPTIONAL categories reaches
        OPTIONAL_COVERAGE_STOP_THRESHOLD of total optional weight
        (prioritising food/utilities/discretionary by weight naturally
        gets there in ~3 optional questions, not all 5 — see
        QUESTIONNAIRE_SCHEMA's ordering), or
      - a hard cap of MAX_QUESTIONS_PER_SITTING is reached, or
      - the customer indicates they want to stop — an EXTERNAL signal,
        not something this module decides on its own; see
        is_sufficient()'s customer_wants_to_stop parameter. Respecting
        that over completeness matters more than a full picture: a
        half-finished budget with clear caveats beats an abandoned
        conversation.

CONFIDENCE CEILING
    Self-reported data is capped at "medium" confidence regardless of
    completeness, never "high" — people round, misremember, and
    systematically underestimate discretionary spend. This is a hard
    rule (SELF_REPORT_CONFIDENCE_CEILING), not a per-response judgment
    call, so it can't quietly drift upward over time.

COMBINING WITH TRANSACTION DATA
    Once real transaction data exists for a category with sufficient
    density, prefer it over the self-reported figure for THAT category
    specifically — blend_expenses() does this category by category, not
    all-or-nothing. Don't discard questionnaire answers the moment ANY
    real data shows up; phase them out per-category as coverage improves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import settings

MAX_QUESTIONS_PER_SITTING = 7
OPTIONAL_COVERAGE_STOP_THRESHOLD = 0.70
SELF_REPORT_CONFIDENCE_CEILING = "medium"


@dataclass
class QuestionnaireQuestion:
    slot_name: str
    question_text: str
    mandatory: bool
    weight: float          # rough share of typical household spend this represents
    expense_category: str | None  # maps to a BudgetAgent monthly_expenses key, if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "question_text": self.question_text,
            "mandatory": self.mandatory,
            "weight": self.weight,
            "expense_category": self.expense_category,
        }


_HBS = settings.budget.ireland_hbs_benchmarks

# Order matters: within each mandatory/optional group, this is the ASK
# order. Optional questions are ordered by weight descending, so the
# 70%-of-optional-weight stopping criterion is reached in the fewest
# questions rather than by chance of dict ordering.
QUESTIONNAIRE_SCHEMA: list[QuestionnaireQuestion] = [
    QuestionnaireQuestion(
        "income", "What's your monthly take-home income?",
        mandatory=True, weight=0.0, expense_category=None,
    ),
    QuestionnaireQuestion(
        "housing_cost", "What's your monthly rent or mortgage payment?",
        mandatory=True, weight=_HBS["housing"], expense_category="housing",
    ),
    QuestionnaireQuestion(
        "rough_monthly_leftover",
        "After your regular bills, roughly how much is usually left over each month?",
        mandatory=True, weight=0.0, expense_category=None,
    ),
    QuestionnaireQuestion(
        "food_spend", "Roughly how much do you spend on groceries and food per month?",
        mandatory=False, weight=_HBS["food"], expense_category="food",
    ),
    QuestionnaireQuestion(
        "utilities_spend", "Roughly how much do your utility bills (electricity, gas, water) come to per month?",
        mandatory=False, weight=_HBS["utilities"], expense_category="utilities",
    ),
    QuestionnaireQuestion(
        "discretionary_spend",
        "Roughly how much do you spend on non-essentials — eating out, entertainment, hobbies — per month? "
        "(This one's easy to underestimate, so a rough figure is fine.)",
        mandatory=False, weight=_HBS["entertainment"], expense_category="entertainment",
    ),
    QuestionnaireQuestion(
        "debt_repayments", "Do you have any regular loan or credit repayments? If so, roughly how much per month?",
        mandatory=False, weight=0.05, expense_category="debt_repayments",
    ),
    QuestionnaireQuestion(
        "large_recurring_items",
        "Do you have any large recurring costs — insurance, subscriptions, school fees — you pay periodically "
        "rather than monthly? If so, roughly how much in total per YEAR?",
        mandatory=False, weight=0.05, expense_category="other_recurring",
    ),
]

_MANDATORY = [q for q in QUESTIONNAIRE_SCHEMA if q.mandatory]
_OPTIONAL = sorted(
    (q for q in QUESTIONNAIRE_SCHEMA if not q.mandatory),
    key=lambda q: q.weight, reverse=True,
)
_TOTAL_OPTIONAL_WEIGHT = sum(q.weight for q in _OPTIONAL)


def _optional_coverage(collected_slots: dict[str, Any]) -> float:
    if _TOTAL_OPTIONAL_WEIGHT == 0:
        return 1.0
    answered_weight = sum(
        q.weight for q in _OPTIONAL
        if collected_slots.get(q.slot_name) is not None
    )
    return answered_weight / _TOTAL_OPTIONAL_WEIGHT


def is_sufficient(
    collected_slots: dict[str, Any],
    customer_wants_to_stop: bool = False,
) -> bool:
    """
    True once the questionnaire has enough to build a (self-report,
    medium-confidence-ceiling) budget and should stop asking.
    """
    mandatory_done = all(
        collected_slots.get(q.slot_name) is not None for q in _MANDATORY
    )
    if not mandatory_done:
        return False
    if customer_wants_to_stop:
        return True
    total_answered = sum(
        1 for q in QUESTIONNAIRE_SCHEMA if collected_slots.get(q.slot_name) is not None
    )
    if total_answered >= MAX_QUESTIONS_PER_SITTING:
        return True
    return _optional_coverage(collected_slots) >= OPTIONAL_COVERAGE_STOP_THRESHOLD


def next_question(
    collected_slots: dict[str, Any],
    customer_wants_to_stop: bool = False,
) -> QuestionnaireQuestion | None:
    """
    The next question to ask, or None if the questionnaire should stop
    (either because is_sufficient() is True, or because every question
    has already been answered — nothing left to ask regardless).
    """
    if is_sufficient(collected_slots, customer_wants_to_stop=customer_wants_to_stop):
        return None
    for q in _MANDATORY:
        if collected_slots.get(q.slot_name) is None:
            return q
    for q in _OPTIONAL:
        if collected_slots.get(q.slot_name) is None:
            return q
    return None  # every question answered — nothing left, regardless of coverage math


def build_monthly_expenses_from_slots(
    collected_slots: dict[str, Any],
) -> dict[str, float]:
    """
    Convert questionnaire answers into the same monthly_expenses shape
    BudgetAgent already works with. large_recurring_items is collected
    as an ANNUAL total (that's what the question asks for) and divided
    by 12 here to produce a monthly-equivalent figure — a labelled
    assumption, not a periodicity inference (there's no transaction
    history to infer anything from; the customer was asked for an
    annual figure directly because a single self-reported "monthly"
    guess for something like insurance would be exactly the kind of
    unfounded assumption Section 3 exists to avoid).
    """
    expenses: dict[str, float] = {}
    for q in QUESTIONNAIRE_SCHEMA:
        if q.expense_category is None:
            continue
        value = collected_slots.get(q.slot_name)
        if value is None:
            continue
        if q.slot_name == "large_recurring_items":
            expenses[q.expense_category] = round(float(value) / 12, 2)
        else:
            expenses[q.expense_category] = round(float(value), 2)
    return expenses


def self_report_confidence(collected_slots: dict[str, Any]) -> dict[str, Any]:
    """
    Confidence summary for a self-report-only budget. Deliberately
    capped at SELF_REPORT_CONFIDENCE_CEILING ("medium") regardless of
    how complete the answers are — this is a hard rule, not computed
    from completeness, so it can never drift up to "high" just because
    every optional question happened to get answered.
    """
    total_answered = sum(
        1 for q in QUESTIONNAIRE_SCHEMA if collected_slots.get(q.slot_name) is not None
    )
    return {
        "source": "self_reported",
        "confidence_tier": SELF_REPORT_CONFIDENCE_CEILING,
        "questions_answered": total_answered,
        "questions_total": len(QUESTIONNAIRE_SCHEMA),
        "optional_coverage": round(_optional_coverage(collected_slots), 3),
        "disclosure": (
            "This budget is based on your own estimates, not transaction "
            "history — treat it as a starting point rather than a precise "
            "figure, especially for spending that's easy to underestimate "
            "like eating out or subscriptions."
        ),
    }


def blend_expenses(
    self_reported: dict[str, float],
    transaction_derived: dict[str, float],
    transaction_verified_categories: set[str],
) -> dict[str, float]:
    """
    Category by category, not all-or-nothing: once transaction data
    exists for a category AND it's verified (see
    agents.data_sufficiency's per-category verification — the caller is
    expected to pass only the categories that came back verified there),
    prefer it over the self-reported figure for that category
    specifically. Self-reported answers for categories transaction data
    doesn't (yet) cover survive untouched — don't discard a customer's
    own estimates the moment ANY real data shows up elsewhere.
    """
    blended = dict(self_reported)
    for category, value in transaction_derived.items():
        if category in transaction_verified_categories:
            blended[category] = value
    return blended