"""
Asks the risk-profiling questions when the customer record is missing the answers.

ANSWER TYPES
    Every question here sets QuestionnaireQuestion.answer_type so
    ConversationalAgent's parser knows how to read the reply:
      integer    — age, dependents, investment_horizon (plain counts/years)
      money      — annual_income, existing_debt (reuses budget_questionnaire's
                   money parser and stop/skip detection, which are
                   generic text-pattern matchers, not budget-specific)
      scale_1_5  — loss_tolerance, financial_knowledge_score (clamped to
                   1-5, not rejected, since RiskProfilingAgent itself
                   treats these on a 1-5 scale centred on 3 — see
                   agents/risk_profiling_agent.py and
                   data/psychometric_proxy.py's derive_loss_tolerance_proxy())
      text       — employment_status (categorical, no numeric parse)
"""
from __future__ import annotations

from typing import Any

from agents.budget_questionnaire import QuestionnaireQuestion

# Order matters: easy, low-friction facts first; the two subjective
# 1-5 scale questions last, since they need the most framing and are
# the ones a customer is most likely to want explained.
QUESTIONNAIRE_SCHEMA: list[QuestionnaireQuestion] = [
    QuestionnaireQuestion(
        "age", "What's your age?",
        mandatory=True, weight=0.0, expense_category=None, answer_type="integer",
    ),
    QuestionnaireQuestion(
        "employment_status",
        "What's your current employment status — employed, self-employed, "
        "unemployed, retired, or student?",
        mandatory=True, weight=0.0, expense_category=None, answer_type="text",
    ),
    QuestionnaireQuestion(
        "annual_income", "What's your gross annual income, in euros?",
        mandatory=True, weight=0.0, expense_category=None, answer_type="money",
    ),
    QuestionnaireQuestion(
        "dependents",
        "How many dependants do you have — for example children or others "
        "who rely on you financially? Enter 0 if none.",
        mandatory=True, weight=0.0, expense_category=None, answer_type="integer",
    ),
    QuestionnaireQuestion(
        "existing_debt",
        "Roughly how much existing debt do you have in total, in euros? "
        "Enter 0 if none.",
        mandatory=True, weight=0.0, expense_category=None, answer_type="money",
    ),
    QuestionnaireQuestion(
        "investment_horizon",
        "Over how many years are you planning to invest or save, roughly?",
        mandatory=True, weight=0.0, expense_category=None, answer_type="integer",
    ),
    QuestionnaireQuestion(
        "loss_tolerance",
        "On a scale of 1 to 5, if your investments dropped sharply in a bad "
        "year, how would you react? 1 means you'd sell everything "
        "immediately, 3 means you'd stay the course, 5 means you'd see it "
        "as a buying opportunity.",
        mandatory=True, weight=0.0, expense_category=None, answer_type="scale_1_5",
    ),
    QuestionnaireQuestion(
        "financial_knowledge_score",
        "On a scale of 1 to 5, how would you rate your own knowledge of "
        "investing and financial products? 1 means beginner, 3 means "
        "comfortable with the basics, 5 means very experienced.",
        mandatory=True, weight=0.0, expense_category=None, answer_type="scale_1_5",
    ),
]

# Every question above is mandatory and none carry weight, so the
# optional-coverage math in budget_questionnaire.is_sufficient() has
# nothing to do here — this collapses to "every mandatory slot is
# present or skipped, or the customer asked to stop". Written directly
# rather than imported, because budget_questionnaire's functions read
# its OWN module-level schema, not a parameter — see the module
# docstring's note on why the two schemas stay separate modules.

SLOT_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "age":                       (18.0, 100.0),
    "annual_income":             (0.0, 2_000_000.0),
    "dependents":                (0.0, 15.0),
    "existing_debt":             (0.0, 2_000_000.0),
    "investment_horizon":        (0.0, 60.0),
    "loss_tolerance":            (1.0, 5.0),
    "financial_knowledge_score": (1.0, 5.0),
}


def is_plausible(slot_name: str, value: float) -> bool:
    low, high = SLOT_PLAUSIBLE_RANGE.get(slot_name, (-1e9, 1e9))
    return low <= value <= high

SLOT_TEXT_VOCAB: dict[str, set[str]] = {
    "employment_status": {
        "employed", "self-employed", "unemployed", "retired", "student",
    },
}


def text_vocab(slot_name: str) -> set[str] | None:
    """The closed vocabulary for a text-type slot, or None if the slot
    has no fixed vocabulary (falls back to accepting any non-empty text,
    the pre-existing behaviour)."""
    return SLOT_TEXT_VOCAB.get(slot_name)


def is_sufficient(
    collected_slots: dict[str, Any],
    customer_wants_to_stop: bool = False,
    questions_asked_this_sitting: int | None = None,
    skipped_slots: set[str] | None = None,
) -> bool:
    """
    True once every required field is present (or skipped) or the
    customer has asked to stop. questions_asked_this_sitting is accepted
    only to keep the same call signature ConversationalAgent uses for
    both kinds — there's no per-sitting cap here; eight questions is the
    whole form.
    """
    if customer_wants_to_stop:
        return True
    skipped = skipped_slots or set()
    return all(
        collected_slots.get(q.slot_name) is not None or q.slot_name in skipped
        for q in QUESTIONNAIRE_SCHEMA
    )


def next_question(
    collected_slots: dict[str, Any],
    customer_wants_to_stop: bool = False,
    questions_asked_this_sitting: int | None = None,
    skipped_slots: set[str] | None = None,
) -> QuestionnaireQuestion | None:
    """The next unanswered required question, or None if done/stopped."""
    skipped = skipped_slots or set()
    if customer_wants_to_stop:
        return None
    if is_sufficient(collected_slots, customer_wants_to_stop,
                      questions_asked_this_sitting, skipped):
        return None
    for q in QUESTIONNAIRE_SCHEMA:
        if collected_slots.get(q.slot_name) is None and q.slot_name not in skipped:
            return q
    return None


def build_user_features_from_slots(collected_slots: dict[str, Any]) -> dict[str, Any]:
    """
    Map elicited answers onto RiskConfig.required_features' exact keys.
    Only annual_income is renamed (-> "income"); every other slot name
    here already IS the RiskConfig feature name.
    """
    features: dict[str, Any] = {}
    for q in QUESTIONNAIRE_SCHEMA:
        value = collected_slots.get(q.slot_name)
        if value is None:
            continue
        key = "income" if q.slot_name == "annual_income" else q.slot_name
        features[key] = value
    return features