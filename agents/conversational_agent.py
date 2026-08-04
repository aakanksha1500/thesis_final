"""
Resposibilities:
    1. Intent classification against the banking77 taxanomy.
    2. Multi-turn slot tracking in the style of MultiWOZ - persistent state across conversation turns.
    3. Natural elicitation of missing slots before escalation.
    4. Safe escalation signal to the Orchestrator when the user's intent requires a specialist agent.

Literature grounding:
    - Gap 1 (monolithic bottleneck): this agent is the ONLY user-facing surface. It never produces financial
        content itself - It routes
    - Sharma et al. [5] identify the absence of this separation as the primary scalability bottleneck
        in existing financial AI systems.
    - Artusi et al. [10]: tone must be accessible for non-expert retail investors.
        This is enforced in the system prompt (config/prompts.py)
    - Takayanagi et al. [7]: users over-trust AI financial advice. The agent
        explicitly avoids implying authority it does not have.

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

from agents.base_agent import AgentResult, BaseAgent
from agents.budget_questionnaire import (
    MAX_QUESTIONS_PER_SITTING,
    QuestionnaireQuestion,
    detect_stop_intent,
    is_plausible,
    is_sufficient,
    next_question,
    parse_money_answer,
)
from config.prompts import CONVERSATIONAL_SYSTEM, INTENT_CLASSIFIER_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

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
    # R7 - the bucket that makes RoutingDecision.FULL_ADVISORY reachable.
    #
    # FULL_ADVISORY had a definition, an agent sequence and a test, but no
    # intent mapped to it, so the most comprehensive route in the system could
    # never be selected at runtime. The tests passed because they called
    # _get_agent_sequence() directly, which made it look covered.
    #
    # The target user is the one who arrives with no specific question:
    # "I know nothing about finance, tell me what to do." That request is not
    # investment_advice (no product question), not budget_analysis (no spending
    # question) and not general_query (it wants advice, not a fact).
    #
    # DISCRIMINATOR: scope, not topic. See the SCOPE RULE in _classify_intent's
    # prompt — without it this bucket becomes a magnet and steals traffic from
    # investment_advice and budget_analysis, which would quietly degrade RQ4
    # routing accuracy rather than improve coverage.
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

class ConversationalAgent(BaseAgent):
    """
    User-facing conversational interface.

    Owns its own session state (slots + history) rather than relying on
    the Orchestrator to track it — one instance per user session.

    State management:
        self._slots - dict of slot_name -> value, persists across turns
        self._history - list of {"role": str, "content": str} dicts
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
        duty as both classifier and responder. Falls back to
        ("general_query", 0.5) on any parse failure so a malformed LLM
        reply can never crash the turn.

        Returns:
            (bucket_name, confidence_float)
        """
        bucket_list = "\n".join(
            f"  {bucket}: {', '.join(examples[:3])}"
            for bucket, examples in INTENT_BUCKETS.items()
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
                        f"[ConversationalAgent] Unkown bucket '{intent}' - "
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

        QW9: the Orchestrator used to call run() purely to read the intent
        label, which cost three LLM calls (classify + extract slots +
        generate a reply) and then threw the generated reply away. It also
        incremented _turn_count, so a CONVERSATIONAL_ONLY turn counted twice.

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
    #
    # WHY THIS LIVES IN ConversationalAgent
    #     agents/budget_questionnaire.py already had the schema, the ordering,
    #     the stopping criteria and the confidence ceiling; what it never had
    #     was anything that actually ASKED. questionnaire_answers had to
    #     arrive at BudgetAgent already-structured, which meant the module was
    #     a well-specified form nobody could fill in.
    #
    #     The asking belongs here for the same reason slot tracking already
    #     does: this agent is the only user-facing surface in the system, it
    #     already owns persistent per-session slot state, and it already has
    #     the one LLM client that's allowed to interpret free text. Putting
    #     the loop in BudgetAgent would either give a specialist agent a
    #     second conversational surface (the exact monolith Gap 1 is about) or
    #     require it to return "ask this next" on every turn and trust the
    #     caller to do it — which is what the previous design did, and why the
    #     loop was never actually closed.
    #
    #     The division of labour is unchanged and deliberate: this agent
    #     decides WHAT TO ASK and PARSES what comes back; BudgetAgent decides
    #     what the answers MEAN. No budget arithmetic happens in here.
    #
    # WHERE THE LLM IS AND ISN'T USED
    #     Deterministic (budget_questionnaire's regex layer): which question
    #     is next, whether a reply contains a stop signal, whether it contains
    #     a skip, and the number itself in the common cases. Free, reproducible,
    #     unit-testable, and identical every run.
    #     LLM: only the residue — a reply the rules couldn't read at all
    #     ("about twelve hundred"), and an ambiguous non-answer that might or
    #     might not be someone asking to stop. Even then the model only
    #     PROPOSES: an extracted figure is range-checked before it's accepted,
    #     because a misread order of magnitude here goes straight into a
    #     budget, and mock/offline mode must degrade to "ask again" rather
    #     than to a wrong number.

    def start_questionnaire(
        self, seed_slots: dict[str, Any] | None = None
    ) -> QuestionnaireQuestion | None:
        """
        Enter questionnaire mode and return the first question to ask, or
        None if nothing needs asking.

        Seeds from slots already collected this session — income is tracked
        for risk profiling and is reused rather than asked a second time,
        which is the single most irritating thing a form can do to someone
        who has already answered it.
        """
        self._questionnaire_active = True
        self._questions_asked_this_sitting = 0
        self._skipped_slots = set()
        self._customer_wants_to_stop = False
        if seed_slots:
            self.update_slots(seed_slots)

        question = next_question(
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
    def questionnaire_answers(self) -> dict[str, Any]:
        """
        The questionnaire's view of session slots: only the slots the
        questionnaire schema actually cares about, so a risk-profiling slot
        like `age` never leaks into BudgetAgent's answer set.
        """
        from agents.budget_questionnaire import QUESTIONNAIRE_SCHEMA
        return {
            q.slot_name: self._slots[q.slot_name]
            for q in QUESTIONNAIRE_SCHEMA
            if self._slots.get(q.slot_name) is not None
        }

    @property
    def questionnaire_state(self) -> dict[str, Any]:
        """Everything the Orchestrator/BudgetAgent needs to continue or finish."""
        answers = self.questionnaire_answers
        return {
            "active": self._questionnaire_active,
            "pending_question": (
                self._pending_question.to_dict() if self._pending_question else None
            ),
            "answers": answers,
            "skipped_slots": sorted(self._skipped_slots),
            "questions_asked_this_sitting": self._questions_asked_this_sitting,
            "customer_wants_to_stop": self._customer_wants_to_stop,
            "sufficient": is_sufficient(
                answers,
                customer_wants_to_stop=self._customer_wants_to_stop,
                questions_asked_this_sitting=self._questions_asked_this_sitting,
                skipped_slots=self._skipped_slots,
            ),
            "events": list(self._questionnaire_events),
        }

    def _record_questionnaire_event(self, kind: str, **fields: Any) -> None:
        """
        Append-only trace of the questionnaire. Exists because the interesting
        failure modes here are conversational, not computational — a question
        asked twice, a stop signal missed, a figure accepted from the LLM that
        the rules had already refused — and none of them are visible in the
        final answers dict alone. This is what makes a session replayable.
        """
        self._questionnaire_events.append(
            {"turn": self._turn_count, "event": kind, **fields}
        )

    def _detect_stop_intent(self, user_message: str) -> tuple[bool, str]:
        """
        Hybrid stop detection. Returns (wants_to_stop, basis).

        Rules first — they're high-precision and cover the common phrasings.
        Only a genuinely ambiguous message (no recognised stop phrase, no
        number, no skip phrase) costs an LLM call, and if that call fails or
        is running against a mock client the answer is False: the fallback on
        an unreadable message is to keep the conversation going, because the
        recovery from that (they say "stop" again, more plainly) is cheap and
        obvious, whereas silently ending a questionnaire someone wanted is not
        recoverable at all — they just get a worse budget and no explanation.
        """
        deterministic = detect_stop_intent(user_message)
        if deterministic is not None:
            return deterministic, "deterministic"

        prompt = (
            f"A customer is being asked a short series of budgeting questions. "
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
    ) -> tuple[float | None, str]:
        """
        Hybrid answer parsing for one question. Returns (value, basis), where
        value None means "not obtained" — either declined or unreadable.

        The LLM tier is strictly a proposal: whatever number it returns is
        range-checked against budget_questionnaire.SLOT_PLAUSIBLE_RANGE before
        it's accepted. That check isn't validating the customer, it's
        validating the model — a monthly rent of 120,000 means something was
        misread, and the cost of accepting it (a confidently wrong budget) is
        far higher than the cost of rejecting it (one repeated question).
        """
        value, basis = parse_money_answer(user_message)
        if basis in {"explicit_zero", "single_figure", "range_midpoint"}:
            if value is not None and not is_plausible(question.slot_name, value):
                return None, "rejected_implausible_deterministic"
            return value, basis
        if basis == "skip":
            return None, "skip"

        prompt = (
            f"Extract a single euro amount from a customer's reply.\n\n"
            f"Question asked: \"{question.question_text}\"\n"
            f"Customer replied: \"{user_message}\"\n\n"
            f"If the reply states or clearly implies an amount, return it as a "
            f"plain number. If it does not, return null — do not guess, and do "
            f"not infer an amount from anything other than what they said.\n"
            f"Respond with ONLY valid JSON, no other text:\n"
            f'{{"amount": <number or null>}}'
        )
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
            match = re.search(r'\{[^}]*\}', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                amount = parsed.get("amount")
                if amount is None:
                    return None, "llm_found_nothing"
                amount = float(amount)
                if not is_plausible(question.slot_name, amount):
                    logger.warning(
                        f"[ConversationalAgent] LLM proposed implausible "
                        f"{question.slot_name}={amount} — rejected, re-asking"
                    )
                    return None, "rejected_implausible_llm"
                return round(amount, 2), "llm"
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Answer-extraction LLM call failed: {exc}")
        return None, "unparsed"

    def _questionnaire_turn(self, user_message: str) -> dict[str, Any]:
        """
        One questionnaire turn: interpret the reply to the pending question,
        then either ask the next one or hand the finished answers on.

        Precedence is deliberate and not arbitrary:
          1. STOP outranks everything, including a usable answer in the same
             message ("it's about 1200 but I'd rather not go through more") —
             the answer is still recorded, and then we stop.
          2. SKIP applies to this question only and never re-asks it.
          3. Otherwise, parse; an unreadable reply re-asks the SAME question
             once rather than silently advancing past it, because advancing
             would leave a gap the customer thinks they filled.
        """
        question = self._pending_question
        if question is None:
            # Defensive: mode is active but nothing was pending. Re-derive
            # rather than dropping the customer into a dead conversation.
            question = next_question(
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
                # Unreadable. Re-ask this question once; if it's still
                # unreadable next turn we treat it as a skip rather than
                # looping, because a customer repeating something the system
                # can't parse is a system problem, and making them do it a
                # third time is not going to fix it.
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
                            f"{question.question_text}"
                        ),
                    )
                self._skipped_slots.add(question.slot_name)
                self._record_questionnaire_event(
                    "skip", slot=question.slot_name, basis="unparsed_twice"
                )

        following = next_question(
            self.questionnaire_answers,
            customer_wants_to_stop=self._customer_wants_to_stop,
            questions_asked_this_sitting=self._questions_asked_this_sitting,
            skipped_slots=self._skipped_slots,
        )
        self._pending_question = following
        if following is None:
            self._questionnaire_active = False
            reason = (
                "per_sitting_cap"
                if self._questions_asked_this_sitting >= MAX_QUESTIONS_PER_SITTING
                else "sufficient"
            )
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
        The turn payload in questionnaire mode. Shaped to satisfy
        ConversationalAgent's existing payload contract (payloads.py) so this
        mode isn't a second, differently-shaped kind of turn the Orchestrator
        has to special-case.
        """
        state = self.questionnaire_state
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
            "intent": "budget_analysis",
            "confidence": 1.0,       # deterministic mode, not a classification
            "escalation_needed": bool(finished),
            "response": response,
            "collected_slots": dict(self._slots),
            "turn_count": self._turn_count,
            "escalation_block": (
                {
                    "escalate": True,
                    "intent": "budget_analysis",
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
        True only if the (alias-resolved) intent is in the escalation set
        AND confidence clears the configured threshold — low-confidence
        classifications stay in-conversation instead of routing to a
        specialist on a guess.
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