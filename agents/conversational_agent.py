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