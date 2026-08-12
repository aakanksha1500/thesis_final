"""
The only aagent the user talks to directly. Classifies intent, fills slots, and hands off 
to specialist agents.

Evaluation:
    - Banking77 intent accuracy (Phase 2 eval)
    - Slot fill rate across multi-turn fixtures.
    Both scores written to results/phase2_conversaional_baseline.json
    after running: pytest tests/unit/test_conversational_agent.py -v
"""

from __future__ import annotations
 
import json
import re
import time
from typing import Any
 
import agents.risk_questionnaire as risk_questionnaire
from agents.base_agent import AgentResult, BaseAgent
from agents.budget_questionnaire import (
    MAX_QUESTIONS_PER_SITTING,
    QuestionnaireQuestion,
    detect_stop_intent,
    is_skip_answer,
    parse_money_answer,
)
from agents.budget_questionnaire import QUESTIONNAIRE_SCHEMA as _BUDGET_SCHEMA
from agents.budget_questionnaire import is_plausible as _budget_is_plausible
from agents.budget_questionnaire import is_sufficient as _budget_is_sufficient
from agents.budget_questionnaire import next_question as _budget_next_question
from config.prompts import CONVERSATIONAL_SYSTEM, INTENT_CLASSIFIER_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger
 
logger = get_logger(__name__)


# Registry so start_questionnaire()/_questionnaire_turn() are schema-agnostic:
# add a new kind here (schema, next_question, is_sufficient, is_plausible)
_QUESTIONNAIRE_KINDS: dict[str, dict[str, Any]] = {
    "budget": {
        "schema": _BUDGET_SCHEMA,
        "next_question": _budget_next_question,
        "is_sufficient": _budget_is_sufficient,
        "is_plausible": _budget_is_plausible,
    },
    "risk": {
        "schema": risk_questionnaire.QUESTIONNAIRE_SCHEMA,
        "next_question": risk_questionnaire.next_question,
        "is_sufficient": risk_questionnaire.is_sufficient,
        "is_plausible": risk_questionnaire.is_plausible,
    },
}

INTENT_BUCKETS: dict[str, list[str]] = {
    "general_query": [
        "balance", "spending_limit", "exchange_rate", "card_about_to_expire",
        "beneficiary_not_allowed",
    ],
    "risk_profiling": [
        "risk_assessment", "financial_goals", "investment_horizon",
    ],
    "investment_advice": [
        "investment", "investment_returns", "savings", "savings_interest",
        "top_up_savings",
    ],
    "budget_analysis": [
        "cashflow", "budget", "disposable_income", "spending_breakdown",
    ],
    "product_suggestion": [
        "card_payment", "card_type", "card_delivery", "order_checque",
    ],
    "explanation_request": [
        "why", "explain", "how_did_you", "reasoning",
    ],
    # the bucket that makes RoutingDecision.FULL_ADVISORY reachable.
    # FULL_ADVISORY had a definition and a sequence but no intent bucket to reach it.
    "full_advisory": [
        "complete_financial_review", "where_do_i_start", "holistic_plan",
        "new_to_finance", "overall_situation",
    ],
    "out_of_scope": [
        "legal_advice", "medical", "complaint_other",
    ],
}

# Aliases so settings.conversational.escalation_intents names map to bucket keys.
INTENT_ALIASES: dict[str, str] = {
    "investment": "investment_advice",
    "budget": "budget_analysis",
    "savings": "investment_advice",
    "financial_advice": "investment_advice",
    "card": "product_suggestion",
    "loan": "product_suggestion",
    "transfer": "general_query",
    "risk_profiling": "risk_profiling",
}

_CLASS_TO_BUCKET: dict[str, str] = {
    cls: bucket
    for bucket, classes in INTENT_BUCKETS.items()
    for cls in classes
}

INTENT_BUCKET_GUIDE: dict[str, str] = {
    "general_query": (
        "generic banking questions not needing a specialist — balance, "
        "spending limits, exchange rates, card status, AND status/why "
        "questions about your own account or a transaction, even when "
        "phrased with 'why' or 'explain' — a pending payment, a charge "
        "you don't recognise, a declined transaction. The 'why' is about "
        "the transaction/account, not about anything this assistant said. "
        'e.g. "what\'s my balance", "is my card about to expire", '
        '"what\'s the exchange rate", "why is my transfer still '
        'pending", "why was I charged for that", "how come my payment '
        'was declined"'
    ),
    "risk_profiling": (
        "wants their own risk tolerance or investor profile assessed. "
        'e.g. "what\'s my risk profile", "am I a cautious investor", '
        '"assess my risk tolerance"'
    ),
    "investment_advice": (
        "wants investment or savings recommendations, or asks what "
        'return they could get. e.g. "should I invest my savings", '
        '"how can I grow my money", "is now a good time to invest"'
    ),
    "budget_analysis": (
        "wants to understand THEIR OWN spending, cash flow, or budget — "
        "including financial difficulty, arrears, or debt they're "
        "struggling with. "
        'e.g. "where does my money go", "what am I spending on", '
        '"show me my budget", "am I overspending", "how much do I '
        'have left each month", "I\'m behind on payments, what are my '
        'rights", "I can\'t afford my bills", "what happens if I miss a '
        'payment"'
    ),
    "product_suggestion": (
        "wants to REQUEST or ORDER a specific banking product that's NEW "
        "to them — a card, cheque book, account type. NOT a status, "
        "rule, or problem question about a card/account they ALREADY "
        "have (that's general_query even though it mentions a card — "
        '"is my card about to expire" or "why was my card payment '
        'declined" are general_query, not this). '
        'e.g. "I need a new card", "order me a cheque book", '
        '"what card types do you offer", "can I get a virtual card"'
    ),
    "explanation_request": (
        "wants to understand WHY a previous answer or recommendation "
        "from THIS ASSISTANT, in THIS conversation, was given — never a "
        "general 'why is this happening' question about an account or "
        "transaction, even when it uses the words 'why' or 'explain' "
        "(that's general_query — see its own examples). This bucket "
        "needs a prior recommendation to be pointing back at. "
        'e.g. "why did you recommend that", "explain that", '
        '"how did you work that out", "why do you think I\'m moderate '
        'risk"'
    ),
    "full_advisory": (
        "wants a complete financial review with no specific question — "
        'scope, not topic. e.g. "I don\'t know where to start", "give me '
        'a full picture of my finances", "I\'m new to all this"'
    ),
    "out_of_scope": (
        "not a banking or financial matter — legal, medical, or "
        'unrelated. e.g. "can you give me legal advice", "what '
        'medication should I take"'
    ),
}
assert set(INTENT_BUCKET_GUIDE) == set(INTENT_BUCKETS), (
    "INTENT_BUCKET_GUIDE must describe exactly the buckets INTENT_BUCKETS "
    "declares — an added/removed bucket here without updating the other "
    "would silently leave the live classifier prompt incomplete."
)

class ConversationalAgent(BaseAgent):
    """
    User-facing conversational interface.

    Owns its own slots and history for the session, classifies each message's intent, and 
    signals the Orchestrator when a specialist is needed.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client,
        name="ConversationalAgent")
        self._slots: dict[str, Any] = {}
        self._history: list[dict] = []
        self._history_is_external: bool = False
        self._turn_count: int = 0
        self.last_classification_error: str | None = None
        self.last_classification_error_status: int | None = None
        self._questionnaire_active: bool = False
        self._questionnaire_kind: str = "budget"  # or "risk" — see _QUESTIONNAIRE_KINDS
        self._pending_question: QuestionnaireQuestion | None = None
        self._questions_asked_this_sitting: int = 0
        self._skipped_slots: set[str] = set()
        self._customer_wants_to_stop: bool = False
        self._questionnaire_events: list[dict[str, Any]] = []

    @property
    def system_prompt(self) -> str:
        return CONVERSATIONAL_SYSTEM

    def update_slots(self, new_slots: dict[str, Any]) -> None:
        """
        Merges any newly-extracted slot values into session state, but
        only for slots that are in the tracked list — this stops the LLM
        from injecting arbitrary keys into long-lived session state.
        """

        for key, value in new_slots.items():
            if key in settings.conversational.tracked_slots and value is not None:
                self._slots[key] = value
                logger.debug(f"[ConversationalAgent] Slot updated: {key}={value!r}")

    def get_missing_slots(self, required: list[str]) -> list[str]:
        """
        Returns which of a specialist agent's required slots we don't have yet.
        """
        return [s for s in required if s not in self._slots]

    def _slot_context_string(self) -> str:
        """
        Formats current slots as plain text to inject into the next LLM prompt.
        """
        if not self._slots:
            return "No user information collected yet"
        return "\n".join(f"   {k}: {v}" for k,v in self._slots.items())

    # Intent Classification

    def _classify_intent(self, user_message: str) -> tuple[str, float]:
        """
        Asks the LLM to classify the message into one of 7 buckets as
        JSON. There's no separate classifier model — the LLM does double
        duty as both classifier and responder.

        Returns:
            (bucket_name, confidence_float)
        """
        bucket_list = "\n".join(
            f"  {bucket}: {description}"
            for bucket, description in INTENT_BUCKET_GUIDE.items()
        )
        prompt = (
            f"Classify this user message into exactly one intent bucket.\n\n"
            f"User message: \"{user_message}\"\n\n"
            f"Available buckets and example intents: \n{bucket_list}\n\n"
            f"Respond with ONLY valid JSON, no other text: \n"
            f"{{\"intent\": \"<bucket_name>\", \"confidence\": <float 0.0-1.0>}}"
        )
        self.last_classification_error = None
        self.last_classification_error_status = None
        try:
            raw, _ = self._call_llm(
                prompt, temperature=0.0, system_override=INTENT_CLASSIFIER_SYSTEM
            )
            # Extract JSON even if LLM adds surrounding text
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                parsed = json.loads(match.group())
                intent = parsed.get("intent", "general_query")
                confidence = float(parsed.get("confidence", 0.5))
                # Validate bucket name
                if intent not in INTENT_BUCKETS:
                    logger.warning(
                        f"[ConversationalAgent] Unknown bucket '{intent}' - "
                        f"falling back to general_query"
                    )
                    intent = "general_query"
                return intent, confidence
            self.last_classification_error = f"No JSON object found in response: {raw[:200]!r}"
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Intent classification failed: {exc}")
            self.last_classification_error = f"{type(exc).__name__}: {exc}"
            self.last_classification_error_status = (
                getattr(exc, "status_code", None)
                or getattr(getattr(exc, "response", None), "status_code", None)
            )

        return "general_query", 0.5


    def classify_only(self, user_message: str) -> tuple[str, float]:
        """
        Classify intent with ONE LLM call and no side effects.

        Use this when you need the routing label only. Use run() when you
        actually want a reply.
        """
        return self._classify_intent(user_message)

    # Slot extraction from LLM response

    def _extract_slots_from_response(self, llm_response: str) -> dict[str, Any]:
        """
        Second LLM call that tries to pull slot values (age, income, etc.)
        out of what the user just said. Returns {} on failure — a safe,
        silent no-op rather than raising.
        """

        prompt = (
            f"Extract any of these slot values from the text below.\n"
            f"Slots to look for: {settings.conversational.tracked_slots}\n\n"
            f"Text: \"{llm_response}\"\n\n"
            f"Respond with ONLY a JSON object of slot_name -> value for slots "
            f"that are clearly stated. Return {{}} if nothing is clearly stated. "
            f"No other text."
        )
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
            match = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    # QUESTIONNAIRE MODE — turn-by-turn loop

    def _kind_funcs(self, kind: str | None = None) -> dict[str, Any]:
        """
        Look up the active (or given) questionnaire kind's schema/functions
        from _QUESTIONNAIRE_KINDS. Defaults to "budget".
        """
        return _QUESTIONNAIRE_KINDS[kind or self._questionnaire_kind or "budget"]

    def start_questionnaire(
        self, seed_slots: dict[str, Any] | None = None, kind: str = "budget",
    ) -> QuestionnaireQuestion | None:
        """
        Enter questionnaire mode for `kind` ("budget" or "risk") and return
        the first question to ask, or None if nothing needs asking.
        Seeds from. slots already collected this session - e.g. age/income may already be 
        known from eariler small talk or the other questionnaire, and are reused rather than asked again,
        which is the single most irritating thing a form can do to someone who has already answered it. 
        """
        self._questionnaire_active = True
        self._questionnaire_kind = kind
        self._questions_asked_this_sitting = 0
        self._skipped_slots = set()
        self._customer_wants_to_stop = False
        if seed_slots:
            self.update_slots(seed_slots)

        funcs = self._kind_funcs(kind)
        question = funcs["next_question"](
            self.questionnaire_answers,
            questions_asked_this_sitting=0,
            skipped_slots=self._skipped_slots,
        )
        self._pending_question = question
        if question is None:
            self._questionnaire_active = False
            self._record_questionnaire_event("start", detail="nothing to ask")
        else:
            self._questions_asked_this_sitting += 1
            self._record_questionnaire_event("ask", slot=question.slot_name)
        return question

    @property
    def questionnaire_kind(self) -> str:
        """Which schema is currently active — read by the Orchestrator to
        decide which agent to re-run once questionnaire_state['sufficient']."""
        return self._questionnaire_kind

    @property
    def questionnaire_answers(self) -> dict[str, Any]:
        """
        The active questionnaire's view of session slots: only the slots
        ITS schema cares about, so e.g. a risk-profiling slot like `age`
        never leaks into BudgetAgent's answer set, and vice versa.
        """
        schema = self._kind_funcs()["schema"]
        return {
            q.slot_name: self._slots[q.slot_name]
            for q in schema
            if self._slots.get(q.slot_name) is not None
        }

    @property
    def questionnaire_state(self) -> dict[str, Any]:
        """Everything the Orchestrator/BudgetAgent/RiskProfilingAgent needs
        to continue or finish, for whichever kind is currently active."""
        answers = self.questionnaire_answers
        funcs = self._kind_funcs()
        return {
            "kind": self._questionnaire_kind,
            "active": self._questionnaire_active,
            "pending_question": (
                self._pending_question.to_dict() if self._pending_question else None
            ),
            "answers": answers,
            "skipped_slots": sorted(self._skipped_slots),
            "questions_asked_this_sitting": self._questions_asked_this_sitting,
            "customer_wants_to_stop": self._customer_wants_to_stop,
            "sufficient": funcs["is_sufficient"](
                answers,
                customer_wants_to_stop=self._customer_wants_to_stop,
                questions_asked_this_sitting=self._questions_asked_this_sitting,
                skipped_slots=self._skipped_slots,
            ),
            "events": list(self._questionnaire_events),
        }

    def _record_questionnaire_event(self, kind: str, **fields: Any) -> None:
        """
        Append-only trace of the questionnaire. The failures here are conversational such as a question
        asked twice or a stop signal missed.
        """
        self._questionnaire_events.append(
            {"turn": self._turn_count, "event": kind, **fields}
        )

    def _detect_stop_intent(self, user_message: str) -> tuple[bool, str]:
        """
        Decised whether the customer wants to stop the quesstionnaire. Rules runn first and 
        cover the common phrasings: only a genuienly ambiguous message costs an LLM call.
        Returns the decisin and which of the two produced it.
        """
        deterministic = detect_stop_intent(user_message)
        if deterministic is not None:
            return deterministic, "deterministic"

        topic = "risk-profiling" if self._questionnaire_kind == "risk" else "budgeting"
        prompt = (
            f"A customer is being asked a short series of {topic} questions. "
            f"They just replied:\n\n\"{user_message}\"\n\n"
            f"Are they asking to STOP answering questions (now or for the "
            f"moment), or are they still engaging with the conversation?\n"
            f"Respond with ONLY valid JSON, no other text:\n"
            f'{{"wants_to_stop": true|false}}'
        )
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
            match = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed.get("wants_to_stop"), bool):
                    return parsed["wants_to_stop"], "llm"
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Stop-intent LLM check failed: {exc}")
        return False, "llm_unavailable_default_continue"

    def _parse_questionnaire_answer(
        self, user_message: str, question: QuestionnaireQuestion
    ) -> tuple[Any, str]:
        """
        Dispatches to a type-specific parser based on answer_type (defaults to "money"
        - every existing budget question). Returns (value, basis), where value None means "not
        obstained" - either declined or unreadable.
        """
        answer_type = getattr(question, "answer_type", "money")
        if answer_type == "text":
            return self._parse_text_answer(user_message)
        if answer_type == "integer":
            return self._parse_numeric_answer(user_message, question, integer=True)
        if answer_type == "scale_1_5":
            return self._parse_scale_answer(user_message, question)
        return self._parse_numeric_answer(user_message, question, integer=False)

    def _is_plausible(self, slot_name: str, value: float) -> bool:
        return self._kind_funcs()["is_plausible"](slot_name, value)

    def _parse_text_answer(self, user_message: str) -> tuple[str | None, str]:
        """Deterministic only — categorical free text (e.g. employment
        status) has no numeric plausibility check to escalate to an LLM
        for; if the rules can't read it there's nothing more reliable to try."""
        if is_skip_answer(user_message):
            return None, "skip"
        text = (user_message or "").strip()
        if not text:
            return None, "unparsed"
        return text.lower(), "single_figure"

    def _parse_numeric_answer(
        self, user_message: str, question: QuestionnaireQuestion, integer: bool,
    ) -> tuple[float | None, str]:
        """
        Shared by "money" and "integer" — both are ultimately "find the
        number", differing only in rounding and which prompt wording asks
        for money vs. a plain count.
        """
        value, basis = parse_money_answer(user_message)
        if basis in {"explicit_zero", "single_figure", "range_midpoint"}:
            if value is not None:
                if not self._is_plausible(question.slot_name, value):
                    return None, "rejected_implausible_deterministic"
                value = round(value) if integer else round(value, 2)
            return value, basis
        if basis == "skip":
            return None, "skip"

        noun = "a whole number" if integer else "an amount in euros"
        prompt = (
            f"Extract {noun} from a customer's reply.\n\n"
            f"Question asked: \"{question.question_text}\"\n"
            f"Customer replied: \"{user_message}\"\n\n"
            f"If the reply states or clearly implies a value, return it as a "
            f"plain number. If it does not, return null — do not guess, and do "
            f"not infer a value from anything other than what they said.\n"
            f"Respond with ONLY valid JSON, no other text:\n"
            f'{{"value": <number or null>}}'
        )
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
            match = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                amount = parsed.get("value")
                if amount is None:
                    return None, "llm_found_nothing"
                amount = float(amount)
                if not self._is_plausible(question.slot_name, amount):
                    logger.warning(
                        f"[ConversationalAgent] LLM proposed implausible "
                        f"{question.slot_name}={amount} — rejected, re-asking"
                    )
                    return None, "rejected_implausible_llm"
                return (round(amount) if integer else round(amount, 2)), "llm"
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Answer-extraction LLM call failed: {exc}")
        return None, "unparsed"

    def _parse_scale_answer(
        self, user_message: str, question: QuestionnaireQuestion,
    ) -> tuple[int | None, str]:
        """
        1-5 self-rating questions (loss_tolerance, financial_knowledge_score).
        Clamped to 1-5 rather than rejected when out of range
        """
        value, basis = parse_money_answer(user_message)
        if basis == "skip":
            return None, "skip"
        if basis in {"explicit_zero", "single_figure", "range_midpoint"} and value is not None:
            return max(1, min(5, round(value))), basis

        prompt = (
            f"Extract a single rating from 1 to 5 from a customer's reply.\n\n"
            f"Question asked: \"{question.question_text}\"\n"
            f"Customer replied: \"{user_message}\"\n\n"
            f"If the reply states or clearly implies a rating on this scale, "
            f"return it as an integer 1-5. If it does not, return null.\n"
            f"Respond with ONLY valid JSON, no other text:\n"
            f'{{"rating": <integer 1-5 or null>}}'
        )
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
            match = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                rating = parsed.get("rating")
                if rating is None:
                    return None, "llm_found_nothing"
                return max(1, min(5, round(float(rating)))), "llm"
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Answer-extraction LLM call failed: {exc}")
        return None, "unparsed"

    def _questionnaire_turn(self, user_message: str) -> dict[str, Any]:
        """
        One questionnaire turn: interpret the reply to the pending question,
        then either ask the next one or hand the finished answers on.
        """
        funcs = self._kind_funcs()
        question = self._pending_question
        if question is None:
            # Defensive: mode is active but nothing was pending. Re-derive
            # rather than dropping the customer into a dead conversation.
            question = funcs["next_question"](
                self.questionnaire_answers,
                customer_wants_to_stop=self._customer_wants_to_stop,
                questions_asked_this_sitting=self._questions_asked_this_sitting,
                skipped_slots=self._skipped_slots,
            )
            self._pending_question = question
            if question is None:
                self._questionnaire_active = False
                return self._questionnaire_payload(finished=True, reason="nothing_pending")

        value, parse_basis = self._parse_questionnaire_answer(user_message, question)
        if value is not None:
            self.update_slots({question.slot_name: value})
            self._record_questionnaire_event(
                "answer", slot=question.slot_name, value=value, basis=parse_basis
            )

        wants_stop, stop_basis = self._detect_stop_intent(user_message)
        if wants_stop:
            self._customer_wants_to_stop = True
            self._questionnaire_active = False
            self._pending_question = None
            self._record_questionnaire_event("stop", basis=stop_basis)
            logger.info(f"[ConversationalAgent] Questionnaire stopped by customer ({stop_basis})")
            return self._questionnaire_payload(finished=True, reason="customer_stopped")

        if value is None:
            if parse_basis == "skip":
                self._skipped_slots.add(question.slot_name)
                self._record_questionnaire_event("skip", slot=question.slot_name)
            else:
                already_retried = any(
                    e.get("event") == "reask" and e.get("slot") == question.slot_name
                    for e in self._questionnaire_events
                )
                if not already_retried:
                    self._record_questionnaire_event(
                        "reask", slot=question.slot_name, basis=parse_basis
                    )
                    return self._questionnaire_payload(
                        finished=False, reason="unparsed_reask",
                        question_text=(
                            f"Sorry, I didn't catch a figure there. "
                            if getattr(question, "answer_type", "money") != "text"
                            else "Sorry, I didn't quite catch that. "
                        ) + question.question_text,
                        )
                self._skipped_slots.add(question.slot_name)
                self._record_questionnaire_event(
                    "skip", slot=question.slot_name, basis="unparsed_twice"
                )

        following = funcs["next_question"](
            self.questionnaire_answers,
            customer_wants_to_stop=self._customer_wants_to_stop,
            questions_asked_this_sitting=self._questions_asked_this_sitting,
            skipped_slots=self._skipped_slots,
        )
        self._pending_question = following
        if following is None:
            self._questionnaire_active = False
            # MAX_QUESTIONS_PER_SITTING is a budget-specific ceiling (there
            # to stop the optional tail of 5 low-priority questions).
            if (self._questionnaire_kind == "budget"
                    and self._questions_asked_this_sitting >= MAX_QUESTIONS_PER_SITTING):
                reason = "per_sitting_cap"
            else:
                reason = "sufficient"
            self._record_questionnaire_event("complete", reason=reason)
            return self._questionnaire_payload(finished=True, reason=reason)

        self._questions_asked_this_sitting += 1
        self._record_questionnaire_event("ask", slot=following.slot_name)
        return self._questionnaire_payload(finished=False, reason="continuing")

    def _questionnaire_payload(
        self,
        finished: bool,
        reason: str,
        question_text: str | None = None,
    ) -> dict[str, Any]:
        """
        Builds the turn payload in questionnaire mode, in the same shape as a normal turn so the
        Orchestrator needs no special case.
        """
        state = self.questionnaire_state
        kind = self._questionnaire_kind
        # Two audiences use this label: (1) the Orchestrator, which needs it
        # verbatim to keep routing to the right agent while a form is
        # mid-flight (see process_turn's questionnaire short-circuit); (2)
        # INTENT_BUCKETS' own vocabulary, so a real classifier could in
        # principle produce the same string. "risk_profiling" is already a
        # bucket name; budget's mode reuses "budget_analysis" as it always has.
        intent_label = "risk_profiling" if kind == "risk" else "budget_analysis"

        if finished:
            if reason == "customer_stopped":
                response = (
                    "No problem — I'll stop there and work with what you've "
                    "given me. I'll be clear about what's estimated and what's "
                    "missing, and we can pick this up whenever you like."
                )
            elif reason == "per_sitting_cap":
                response = (
                    "That's enough to work with for now — I won't keep you with "
                    "more questions. We can fill in the rest another time if "
                    "you'd like a sharper picture."
                )
            elif kind == "risk":
                response = (
                    "Thanks — that's everything I need to assess your risk "
                    "profile. Give me a moment to work it out."
                )
            else:
                response = (
                    "Thanks — that's everything I need for a first budget. "
                    "Bear in mind it's built from your own estimates rather "
                    "than your transaction history, so treat it as a starting "
                    "point."
                )
        else:
            response = question_text or (
                self._pending_question.question_text if self._pending_question else ""
            )

        return {
            "intent": intent_label,
            "confidence": 1.0,       # deterministic mode, not a classification
            "escalation_needed": bool(finished),
            "response": response,
            "collected_slots": dict(self._slots),
            "turn_count": self._turn_count,
            "escalation_block": (
                {
                    "escalate": True,
                    "intent": intent_label,
                    "collected_slots": dict(self._slots),
                    "questionnaire_answers": state["answers"],
                    "customer_wants_to_stop": state["customer_wants_to_stop"],
                }
                if finished else None
            ),
            "mode": "questionnaire",
            "questionnaire": {**state, "finished": finished, "reason": reason},
        }

    # Escalation signal

    def _parse_response(self, raw: str) -> dict:
        """
        Conversational replies are free text, so this just wraps the raw string.
        """
        return {"response": raw}

    def _needs_escalation(self, intent: str, confidence: float) -> bool:
        """
        True when the intent is one a specialist handles and confidence clears the threshold.
        Low-confidence guesses stay in conversation.
        """
        resolved = INTENT_ALIASES.get(intent, intent)
        escalation_buckets = {
            INTENT_ALIASES.get(i, i)
            for i in settings.conversational.escalation_intents
        }
        return(
            resolved in escalation_buckets
            and confidence >= settings.conversational.intent_confidence_threshold
        )

    def _build_escalation_block(
            self, intent: str, user_message: str
    ) -> dict[str, Any]:
        """
        The structured signal a Orchestrator would read to decide
        which specialist agent to call next, and with what slots already
        collected.
        """
        return{
            "escalate": True,
            "intent": intent,
            "collected_slots": dict(self._slots),
            "user_message": user_message,
        }

    # Response generation

    def _generate_response(
            self, user_message: str, intent: str, needs_escalation: bool
    ) -> str:
        """
        Builds the final reply prompt from slot state + recent history,
        and — if escalating — instructs the LLM to append a JSON
        escalation block on its own line at the end of the reply.
        """
        slot_ctx = self._slot_context_string()
        history_ctx = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self._history[-4:]  # last 2 turns for context window efficiency
        )

        escalation_instruction = (
            "\n\nIMPORTANT: Since this requires a specialist agent, end your response "
            "with this exact JSON on its own line (fill in the values): \n"
            '{"escalate": true, "intent": "' + intent + '", "collected_slots": {}}'
            if needs_escalation else ""
        )

        prompt = (
            f"Known user information:\n{slot_ctx}\n\n"
            f"Recent conversation: \n{history_ctx}\n\n"
            f"User just said: \"{user_message}\"\n"
            f"Detected intent: {intent}\n"
            f"{escalation_instruction}\n\n"
            f"Respond naturally and helpfully."
        )
        raw, _ = self._call_llm(prompt)
        return raw.strip()

    # Main entry point

    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        One conversational turn: classify intent -> extract slots ->
        decide on escalation -> generate reply -> update history ->
        return everything in a payload dict. Never raises; a failed reply
        generation falls back to a generic apology message instead.
        """
        start_time = time.perf_counter()
        self._turn_count += 1

        user_message: str = context.get("user_message", "").strip()
        if not user_message:
            return self._make_result(
                payload={"error": "Empty user message"},
                error="Empty user message",
                duration_ms=0.0,
            )

        # Merge any externally tracked history into local history
        external_history = context.get("conversation_history")
        self._history_is_external = external_history is not None
        if self._history_is_external:
            self._history = list(external_history)

        if self._questionnaire_active:
            payload = self._questionnaire_turn(user_message)
            response_text = payload["response"]
            if not self._history_is_external:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": response_text})
            logger.info(
                f"[ConversationalAgent] turn={self._turn_count} questionnaire "
                f"reason={payload['questionnaire']['reason']} "
                f"asked={payload['questionnaire']['questions_asked_this_sitting']}"
            )
            return self._make_result(
                payload=payload,
                raw=response_text,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                routing_context={
                    "intent": "budget_analysis",
                    "confidence": 1.0,
                    "escalation_needed": payload["escalation_needed"],
                    "questionnaire_finished": payload["questionnaire"]["finished"],
                },
            )

        # Step 1 - Classify Intent
        intent, confidence = self._classify_intent(user_message)
        logger.info(
            f"[ConversationalAgent] turn={self._turn_count} "
            f"intent={intent} confidence={confidence:.2f}"
        )

        # Step 2 - Extract any slots revealed in the user's message
        new_slots = self._extract_slots_from_response(user_message)
        self.update_slots(new_slots)

        # Step 3 - Decide if esclation is warranted
        needs_escalation = self._needs_escalation(intent, confidence)

        # Step 4 - Generate natural-language response
        try:
            response_text = self._generate_response(user_message, intent, needs_escalation)
        except Exception as exc:
            logger.error(f"[ConversationalAgent] Response generation failed: {exc}")
            response_text = (
                "I'm sorry, I encountered an issue processing your request. "
                "Please try again."
            )

        # Step 5 - Update conversation history
        if not self._history_is_external:
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": response_text})
        # else: the Orchestrator owns the history. Appending here would
        # duplicate the user message (already present) and record a reply that
        # synthesis is about to overwrite.

        # Step 6 - Build payload
        escalation_block = (
            self._build_escalation_block(intent, user_message)
            if needs_escalation else None
        )

        payload = {
            "intent": intent,
            "confidence": round(confidence, 4),
            "escalation_needed": needs_escalation,
            "response": response_text,
            "collected_slots": dict(self._slots),
            "turn_count": self._turn_count,
            "escalation_block": escalation_block,
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        return self._make_result(
            payload=payload,
            raw=response_text,
            duration_ms=duration_ms,
            routing_context={
                "intent": intent,
                "confidence": confidence,
                "escalation_needed": needs_escalation,
            }
        )

    # Accessors - used by tests and the Orchestrator

    @property
    def slots(self) -> dict[str, Any]:
        """Read-only view of current slot state."""
        return dict(self._slots)

    @property
    def history(self) -> list[dict]:
        """Read-only view of conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_count


    @property
    def questionnaire_active(self) -> bool:
        """
        True while a questionnaire is mid-flight. The Orchestrator reads this
        to skip intent classification entirely for the turn — see
        Orchestrator.process_turn's questionnaire short-circuit.
        """
        return self._questionnaire_active