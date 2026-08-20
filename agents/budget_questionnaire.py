"""
Asks the user for spending figures when their transaction history is too thin to work from.

WHEN THIS APPLIES
    BudgetAgent.run() returns status="insufficient_history" when
    agents.data_sufficiency.assess_data_sufficiency() finds less than a
    month of real transaction history (or none at all — a brand new
    customer, or TransactionStore.lookup() returning [] for an unknown
    one).

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
      school fees/etc. 

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

import re

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
    answer_type: str = "money"   # "money" | "integer" | "scale_1_5" | "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "question_text": self.question_text,
            "mandatory": self.mandatory,
            "weight": self.weight,
            "expense_category": self.expense_category,
            "answer_type": self.answer_type,
        }


_HBS = settings.budget.ireland_hbs_benchmarks


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
         "transport_spend",
        "Roughly how much do you spend on transport per month — car costs, "
        "fuel, or public transport?",
        mandatory=False, weight=_HBS["transport"], expense_category="transport",
    ),
    QuestionnaireQuestion(
        "utilities_spend", "Roughly how much do your utility bills (electricity, gas, water) come to per month?",
        mandatory=False, weight=_HBS["utilities"], expense_category="utilities",
    ),
    QuestionnaireQuestion(
        "healthcare_spend",
        "Roughly how much do you spend on healthcare per month — GP visits, "
        "pharmacy, or health insurance?",
        mandatory=False, weight=_HBS["healthcare"], expense_category="healthcare",
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
    questions_asked_this_sitting: int | None = None,
    skipped_slots: set[str] | None = None,
) -> bool:
    """
    True once the questionnaire has enough to build a (self-report,
    medium-confidence-ceiling) budget and should stop asking.
    """
    skipped = skipped_slots or set()
    mandatory_done = all(
        collected_slots.get(q.slot_name) is not None or q.slot_name in skipped for q in _MANDATORY
    )
    if not mandatory_done:
        return False
    if customer_wants_to_stop:
        return True
    if questions_asked_this_sitting is not None:
        if questions_asked_this_sitting >= MAX_QUESTIONS_PER_SITTING:
            return True
    else:
        total_answered = sum(
            1 for q in QUESTIONNAIRE_SCHEMA
            if collected_slots.get(q.slot_name) is not None
        )
        if total_answered >= MAX_QUESTIONS_PER_SITTING:
            return True

    return _optional_coverage(collected_slots) >= OPTIONAL_COVERAGE_STOP_THRESHOLD


def next_question(
    collected_slots: dict[str, Any],
    customer_wants_to_stop: bool = False,
    questions_asked_this_sitting: int | None = None,
    skipped_slots: set[str] | None = None,
) -> QuestionnaireQuestion | None:
    """
    Returns the next question to ask, or None when enough has been answered, 
    the customer asked to stop, or the per-sitting limit is reached.
    """
    skipped = skipped_slots or set()
    if customer_wants_to_stop:
        return None
    if is_sufficient(
        collected_slots,
        customer_wants_to_stop=customer_wants_to_stop,
        questions_asked_this_sitting=questions_asked_this_sitting,
        skipped_slots=skipped,
    ):
        return None
    for q in _MANDATORY:
        if collected_slots.get(q.slot_name) is None and q.slot_name not in skipped:
            return q
    for q in _OPTIONAL:
        if collected_slots.get(q.slot_name) is None and q.slot_name not in skipped:
            return q
    return None  # every question answered — nothing left, regardless of coverage math


def build_monthly_expenses_from_slots(
    collected_slots: dict[str, Any],
) -> dict[str, float]:
    """
    Convert questionnaire answers into the same monthly_expenses shape
    BudgetAgent expects. Annual figures are divided by 12.
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
    Confidence summary for a self-report-only budget. Deliberately capped at
    SELF_REPORT_CONFIDENCE_CEILING ("medium") regardless of how complete the answers are — this
    is a hard rule, not computed from completeness, so it can never drift up to "high" just
    because every optional question happened to get answered
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

_STOP_PATTERNS: tuple[str, ...] = (
    r"\b(i'?d )?rather not (answer|say|share|continue)",
    r"\bstop (asking|there|here)\b",
    r"\bno more questions\b",
    r"\bthat'?s enough\b",
    r"\benough questions\b",
    r"\b(i'?m|im) done\b",
    r"\bskip (the rest|all of|these|them)\b",
    r"\bdon'?t want to (answer|continue|keep going)",
    r"\bcan we (stop|move on|be done)\b",
    r"\blet'?s (stop|move on|wrap up)\b",
    r"\bnot right now\b",
    r"\b(maybe )?(some other|another) time\b",
    r"\bcontinue (this )?later\b",
    r"\bfinish (this )?later\b",
    r"\bjust (use|go with) what (you|we) (have|'ve got)\b",
)

# Phrases that decline ONE question without ending the questionnaire. These
# are NOT stop signals, and conflating the two is the specific mistake that
# makes a form feel hostile: "I don't have any debt" is an answer (zero), and
# "I'd rather not say" about one item is a skip, not a walkout.
_SKIP_PATTERNS: tuple[str, ...] = (
    r"\b(i )?don'?t know\b",
    r"\bno idea\b",
    r"\bnot sure\b",
    r"\bskip (this|that|it)\b",
    r"\bpass\b",
    r"\bnext question\b",
    r"\bprefer not to say\b",
)

# "None"/"nothing" is a real, informative answer meaning zero — distinct from
# a skip, which leaves the category unknown. 
_ZERO_PATTERNS: tuple[str, ...] = (
    r"^\s*(none|nothing|nil|zero|n/?a)\s*[.!]?\s*$",
    r"\b(i )?(have|got) (no|none|nothing)\b",
    r"\b(don'?t|do not|doesn'?t|does not) (have|got|pay|spend)( any| anything)?\b",
    r"\bnot? (any|anything)\b",
    r"^\s*no\s*[.!]?\s*$",
)

# A range is two numbers with ONLY a connector between them. Checking the
# span between the two matches, rather than searching the whole message for
# a connector word.
_RANGE_CONNECTOR_RE = re.compile(r"^\s*(to|-|–|—|and|or)\s*$", re.IGNORECASE)

_MONEY_RE = re.compile(
    r"""
    (?:[€$£]\s*)?                  # optional leading currency symbol
    (?P<num>\d{1,3}(?:,\d{3})+     # 1,200 / 12,345
          | \d+(?:\.\d+)?          # 1200 / 1200.50 / 1.2
    )
    \s*(?P<suffix>k|grand)?        # 1.2k / 2 grand
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_stop_intent(text: str) -> bool | None:
    """
    Deterministic tier of the hybrid stop detector.

    Returns:
        True  — a recognised, unambiguous request to stop the questionnaire.
        False — the message contains a usable answer, so it is definitionally
                not a walkout; no LLM call needed.
        None  — genuinely unclear. The CALLER decides what to do with this;
                ConversationalAgent escalates to one LLM call. Returning None
                rather than False matters: silently treating "hmm, this is a
                lot" as consent to continue is exactly the behaviour the
                'respect the customer's wish to stop' rule exists to prevent.
    """
    if not text or not text.strip():
        return None
    lowered = text.strip().lower()

    if any(re.search(pattern, lowered) for pattern in _STOP_PATTERNS):
        return True
    # An answer present means they're still engaging — no need to ask a model.
    if _MONEY_RE.search(lowered) or is_skip_answer(lowered) or is_zero_answer(lowered):
        return False
    return None


def is_skip_answer(text: str) -> bool:
    """Declines this one question, without ending the questionnaire."""
    lowered = (text or "").strip().lower()
    return any(re.search(pattern, lowered) for pattern in _SKIP_PATTERNS)


def is_zero_answer(text: str) -> bool:
    """
    A real answer of zero, as opposed to a skip. Checked BEFORE the money
    regex by parse_money_answer(), because "none" contains no digits and
    would otherwise fall through to the LLM for no reason.
    """
    lowered = (text or "").strip().lower()
    return any(re.search(pattern, lowered) for pattern in _ZERO_PATTERNS)


def parse_money_answer(text: str) -> tuple[float | None, str]:
    """
    Deterministic tier of answer parsing.

    Returns (value, basis) where basis is one of:
        "explicit_zero"    — "none", "I don't have any"
        "single_figure"    — exactly one number found; unambiguous
        "range_midpoint"   — "1200 to 1400" / "between 1200 and 1400"
        "unparsed"         — value is None; caller may escalate to the LLM
        "skip"             — value is None; the customer declined this item
    """
    if not text or not text.strip():
        return None, "unparsed"
    lowered = text.strip().lower()

    if is_skip_answer(lowered):
        return None, "skip"
    if is_zero_answer(lowered):
        return 0.0, "explicit_zero"

    matches = list(_MONEY_RE.finditer(lowered))
    if not matches:
        return None, "unparsed"

    values: list[float] = []
    for match in matches:
        raw = match.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if (match.group("suffix") or "").lower() in {"k", "grand"}:
            value *= 1000
        values.append(value)

    if not values:
        return None, "unparsed"
    if len(values) == 1:
        return round(values[0], 2), "single_figure"

    # Two numbers with nothing but a connector between them is a range, and
    # its midpoint is a defensible reading. Anything else is rejected.
    if len(values) == 2:
        between = lowered[matches[0].end():matches[1].start()]
        if _RANGE_CONNECTOR_RE.match(between):
            return round((values[0] + values[1]) / 2, 2), "range_midpoint"
    return None, "unparsed"


# Plausibility bounds per slot, in euros. These are NOT validation of the
# customer ("your rent can't be that high") — they exist solely to bound what
# an LLM-extracted number is allowed to be before it's written into a budget.
# A model that returns 120000 for a monthly rent has misread something, and
# accepting it would produce a confidently wrong analysis; refusing it and
# re-asking costs one turn. Wide on purpose: the job is catching an order-of-
# magnitude error, not second-guessing anyone's circumstances.
SLOT_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "income":                 (0.0, 100_000.0),
    "housing_cost":           (0.0, 20_000.0),
    "rough_monthly_leftover": (-50_000.0, 100_000.0),   # can legitimately be negative
    "food_spend":             (0.0, 10_000.0),
    "utilities_spend":        (0.0, 5_000.0),
    "discretionary_spend":    (0.0, 20_000.0),
    "debt_repayments":        (0.0, 50_000.0),
    "large_recurring_items":  (0.0, 200_000.0),          # asked as an ANNUAL total
}


def is_plausible(slot_name: str, value: float) -> bool:
    low, high = SLOT_PLAUSIBLE_RANGE.get(slot_name, (-1e9, 1e9))
    return low <= value <= high

def text_vocab(slot_name: str) -> set[str] | None:
    """No text-type slots in the budget schema — always None, so
    ConversationalAgent._parse_text_answer() falls back to its pre-existing
    accept-any-non-empty-string behaviour for this kind. Exists only to
    keep _QUESTIONNAIRE_KINDS' two entries symmetric; see
    risk_questionnaire.text_vocab() for the kind that actually uses this."""
    return None

def blend_expenses(
    self_reported: dict[str, float],
    transaction_derived: dict[str, float],
    transaction_verified_categories: set[str],
) -> dict[str, float]:
    """
    Combine transaction figures with questionnaire answers, category by category. A verified
    transaction figure wins; otherwise the self-reported answer is used.
    """
    blended = dict(self_reported)
    for category, value in transaction_derived.items():
        if category in transaction_verified_categories:
            blended[category] = value
    return blended