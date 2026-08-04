"""
Risk-profiling elicitation — the risk-side counterpart to
agents/budget_questionnaire.py, sharing ConversationalAgent's
questionnaire engine (start_questionnaire/_questionnaire_turn) rather
than duplicating it. See ConversationalAgent's "kind" parameter.

WHY THIS EXISTS
    RiskProfilingAgent.run() returns status="incomplete" when
    RiskConfig.required_features aren't all present
    (_check_missing_features) — deliberate, it will not guess. Until
    now nothing asked for what was missing in a way a customer could
    actually answer: Orchestrator._elicitation_response() writes good
    copy but only fires for RoutingDecision.FULL_ADVISORY pruned to
    nothing runnable, a route the RISK_PROFILING/INVESTMENT paths never
    go through; and RiskProfilingAgent's own "incomplete" message is
    written for whatever calls it programmatically, not for the
    customer. This module is what the conversation does next.

WHY THIS IS SIMPLER THAN budget_questionnaire.py
    BudgetAgent degrades gracefully — self-reported answers blend with
    whatever transaction data exists, capped at a lower confidence
    ceiling, and stop early is fine (a half-finished budget with
    caveats beats an abandoned conversation). RiskProfilingAgent's
    contract is binary: ALL required_features or nothing runs. So
    there's no optional-question weighting, no coverage threshold, no
    confidence ceiling to compute — every question below is mandatory,
    and is_sufficient() is just "all mandatory answered or skipped, or
    the customer asked to stop". If the customer stops early, elicitation
    ends the same way budget's does (respecting that over completeness),
    and RiskProfilingAgent simply reports "incomplete" again next time,
    now customer-facing (see risk_profiling_agent.py's message text).

THE "annual_income" / "income" RENAME — READ BEFORE CHANGING SLOT NAMES
    budget_questionnaire.py's "income" slot is monthly take-home
    ("What's your monthly take-home income?"), reused directly by
    BudgetAgent (_advance_questionnaire treats questionnaire_answers
    ["income"] as a monthly figure). RiskConfig.required_features'
    "income" is annual gross (run_demo.py's DEMO_FEATURES, demo_
    customers.json, and RiskProfilingAgent's own scoring all treat it
    that way). Same word, different unit — reusing the slot name
    "income" here would silently overwrite one with the other the first
    time both questionnaires touch the same session. This module asks
    for and stores "annual_income" instead, and only renames it to
    "income" at the very end, in build_user_features_from_slots(),
    when handing the finished answers to RiskProfilingAgent. Do not
    rename this back to "income" without re-checking that boundary.

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
    here already IS the RiskConfig feature name. See the module
    docstring's note on why annual_income can't just be called "income".
    """
    features: dict[str, Any] = {}
    for q in QUESTIONNAIRE_SCHEMA:
        value = collected_slots.get(q.slot_name)
        if value is None:
            continue
        key = "income" if q.slot_name == "annual_income" else q.slot_name
        features[key] = value
    return features