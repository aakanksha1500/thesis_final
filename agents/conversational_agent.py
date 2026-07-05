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

from agents.base_agent import BaseAgent, AgentResult
from config.prompts import CONVERSATIONAL_SYSTEM
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
    
    State management:
        self._slots - dict of slot_name -> value, persists across turns
        self._history - list of {"role": str, "content": str} dicts
        
    Both are pre-session (one instance per user session). The Orchestrator
    passes context dicts in, but the agent owns its own slot state so that
    the Orchestrator never needs to track individual conversation fields.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="ConversationalAgent")
        self._slots: dict[str, Any] = {}
        self._history: list[dict] = []
        self._turn_count: int = 0
    
    @property
    def system_prompt(self) -> str:
        return CONVERSATIONAL_SYSTEM
    
    def update_slots(self, new_slots: dict[str, Any]) -> None:
        """
        Merge newly extracted slots into session state.
        Called after every turn where the LLM response contains slot data.
        Slots expire after settings.conversational.slot_TTL_turns turns of
        non-reference - preventing stale data from influencing later advice.
        """

        for key, value in new_slots.items():
            if key in settings.conversational.tracked_slots and value is not None:
                self._slots[key] = value
                logger.debug(f"[ConversationalAgent] Slot updated: {key}={value!r}")

    def get_missing_slots(self, required: list[str]) -> list[str]:
        """
        Return which required slots are not yet collected
        """
        return [s for s in required if s not in self._slots]
    
    def _slot_context_string(self) -> str:
        """
        Format current slot state for injection into LLM prompt.
        """
        if not self._slots:
            return "No user information collected yet"
        return "\n".join(f"   {k}: {v}" for k,v in self._slots.items())
    
    # Intent Classification

    def _classify_intent(self, user_message: str) -> tuple[str, float]:
        """
        Classify user message into one of the 7 routing buckets.
        
        Strategy: ask the LLM to output JSON with intent and confidence.
        This gives us a structured signal without a separate classifier model.
        
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
        try:
            raw, _ = self._call_llm(prompt, temperature=0.0)
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
        except Exception as exc:
            logger.warning(f"[ConversationalAgent] Intent classification failed: {exc}")

        return "general_query", 0.5
    
    # Slot extraction from LLM response

    def _extract_slots_from_response(self, llm_response: str) -> dict[str, Any]:
        """
        Attempt to extract slot values the user revealed in their message.
        Uses a lightweighted JSON extraction (not a. seperate NLU model)
        Returns empty dict if nothing found (safe default).
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
        ConversationalAgent responses are free-from natural language.
        Structured outputs (intent, slots) are parsed by dedicated methods.
        """
        return {"response": raw}
    
    def _needs_escalation(self, intent: str, confidence: float) -> bool:
        """
        Determine whether to escalate to a specialist agent.
        Handles both full bucket names and short aliases from settings.
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
        Structured escalation signal appended to responses.
        The Orchestrator reads this to decide routing.
        Committed here so that tests can assert its structure now
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
        Generate the natural-language response to the user.
        If escalation is needed, appends the JSON escalation block.
        """
        slot_ctx = self._slot_context_string()
        history_ctx = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self._history[-4:]  # last 2 turns for context widow efficiency
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
        Process one conversational turn.
        
        Context keys used:
        'user_message'      (str, required)
        'conversation_history' (list, optional - external history)
        
        Returns AgentResult with payload:
            intent, confidence, escalation_needed, response,
            collected_slots, escalation_block (if needed)
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
        external_history = context.get("conversation_history", [])
        if external_history and not self._history:
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
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": response_text})

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
