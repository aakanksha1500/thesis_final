"""
Phase 7 - HALO style hierarchical Orchestrator
ddresses RQ4: multi-agent vs monolithic coherence.

HALO three-layer hierarchy:
  Layer 1 — Goal decomposition: parse intent → routing decision
  Layer 2 — Agent selection:   route sub-tasks → specialist agents
  Layer 3 — Execution monitoring: conflict detection, constraint validation,
             failure recovery, synthesis

Additional design decisions implemented:
  O2 — LangChain-compatible abstraction (agent registry pattern)
  O3 — TRiSM audit log: every decision, call, conflict, violation logged
  O4 — Self-healing via FailureHandler (Wang et al. AgentFixer [12])

Session management:
  One Orchestrator instance per user session. It holds:
    _session_state: conversation history, collected slots, prior risk profile
    _agents:        instantiated agent registry (one instance per agent type)
    audit_log:      per-session JSONL audit writer

Routing decisions (5 buckets matching ConversationalAgent intent taxonomy):
  CONVERSATIONAL_ONLY   → ConversationalAgent
  RISK_PROFILING        → RiskProfilingAgent (+ Explainability)
  INVESTMENT            → Risk → Investment → Explainability
  BUDGET                → BudgetAgent
  FULL_ADVISORY         → Risk → Investment → Budget → Explainability
  EXPLANATION_REQUEST   → ExplainabilityAgent (uses cached prior outputs)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agents.base_agent import AgentResult
from agents.budget_agent import BudgetAgent
from agents.conversational_agent import ConversationalAgent
from agents.investment_agent import InvestmentAgent
from agents.payloads import CAPABILITIES, STATIC_SEQUENCES
from agents.risk_profiling_agent import RiskProfilingAgent
import agents.risk_questionnaire as risk_questionnaire
from orchestrator.approvals import ApprovalStore, PendingApproval, get_approval_store
from data.customer_memory import (
    CustomerMemoryStore,
    extract_preferences_from_slots,
    get_customer_memory_store,
)
from config.constraints import financial_constraints
from config.prompts import ORCHESTRATOR_SYSTEM, SUMMARISER_SYSTEM
from config.settings import settings
from data.customer_store import CustomerStore
from data.psychometric_proxy import derive_loss_tolerance_proxy
from explainability.explainability_agent import ExplainabilityAgent
from orchestrator.audit_log import AuditLog
from orchestrator.conflict_resolver import ConflictResolver
from orchestrator.failure_handler import FailureHandler
from orchestrator.planner import Plan, Planner, available_context_keys
from utils import trace
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class RoutingDecision(str, Enum):
    CONVERSATIONAL_ONLY  = "conversational_only"
    RISK_PROFILING       = "risk_profiling"
    INVESTMENT           = "investment"
    BUDGET               = "budget"
    FULL_ADVISORY        = "full_advisory"
    EXPLANATION_REQUEST  = "explanation_request"

@dataclass
class CollaborationStats:
    """
    Day 7 (G6 §6.6) session-level counters — read by the evaluation
    harness for the "collaboration events per turn, and resolution rate"
    RQ4 column. Mirrors orchestrator.planner.PlannerStats deliberately —
    same shape, same reasoning: a metric class next to the mechanism it
    measures, not folded into a general-purpose logger nobody queries.

    One (requester, need) pair is one attempt, regardless of how it
    resolves. A retry of the original agent, once every need in its
    request was satisfied, is bookkeeping about the OUTCOME of already-
    counted attempts, not a new attempt itself, so it's excluded from
    these counts — see record()'s skip of event["retried"].
    """
    attempts: int = 0
    resolved: int = 0
    unresolved: int = 0

    def record(self, events: list[dict]) -> None:
        for event in events:
            if event.get("retried"):
                continue
            self.attempts += 1
            if event.get("resolved"):
                self.resolved += 1
            else:
                self.unresolved += 1

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.attempts if self.attempts else 0.0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "resolved": self.resolved,
            "unresolved": self.unresolved,
            "resolution_rate": round(self.resolution_rate, 4),
        }

@dataclass
class OrchestratorResult:
    """
    Full result of one conversational tur through the HALO pipeline.
    Consumed by: API layer, evaluation, integration tests.
    """
    session_id: str
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    routing_decision: RoutingDecision = RoutingDecision.CONVERSATIONAL_ONLY
    agents_invoked: list[str] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)
    final_response: str = ""
    conflicts: list[dict] = field(default_factory=list)
    constraint_violations: list[dict] = field(default_factory=list)
    recovered_agents: list[str] = field(default_factory=list)
    # Day 6 (G6 dynamic routing): agents the plan named but that were never
    # invoked because their `requires` weren't met at execution time — e.g.
    # a plan step still shows up here even when _execute_plan() decided not
    # to run it. Each entry: {"agent": name, "reason": "missing [...]"}.
    # A skip is a trust artefact, not a failure — see _execute_plan()'s
    # docstring — so it's reported alongside recovered_agents rather than
    # folded into `success`/`error`.
    skipped: list[dict] = field(default_factory=list)
    # Day 7 (G6 §6.6): one entry per (requester, need) collaboration
    # attempt this turn — see _satisfy_needs()'s docstring for the shape.
    # Same "trust artefact, not a failure" reasoning as `skipped` above.
    collaboration_events: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    plan: Plan | None = None

    @property
    def plan_source(self) -> str:
        """'planner', 'static_fallback', or 'static' when G6 is switched off."""
        return self.plan.source if self.plan else "static"

class Orchestrator:
    """
    Central coordinator implementing the HALO three-layer hierarchy.
    One instance per user session.
    """

    def __init__(
            self,
            llm_client: LLMClient,
            session_id: str | None = None,
            customer_id: str | None = None,
            customer_context: dict[str, Any] | None = None,
            customer_store: CustomerStore | None = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.llm = llm_client
        self.specialist_llm = self._derive_client(
            llm_client, settings.llm.specialist_model, "specialist",
            provider=settings.llm.specialist_provider,
        )
        self.judge_llm = self._derive_client(
            llm_client, settings.llm.judge_model, "judge"
        )
        self.customer_store = customer_store or CustomerStore()

        # Per-session components
        self.audit_log = AuditLog(session_id=self.session_id)
        self._conflict_resolver = ConflictResolver()
        self._failure_handler = FailureHandler()
        self._planner = Planner(llm_client)
        self._collaboration_stats = CollaborationStats()
        self._approval_store: ApprovalStore = get_approval_store()
        self._memory_store: CustomerMemoryStore = get_customer_memory_store()

        # Agent registry — one instance per agent type, shared across turns
        self._agents: dict[str, Any] = {
            "ConversationalAgent": ConversationalAgent(llm_client),
            "RiskProfilingAgent":  RiskProfilingAgent(self.specialist_llm),
            "InvestmentAgent":     InvestmentAgent(self.specialist_llm),
            "BudgetAgent":         BudgetAgent(self.specialist_llm),
            "ExplainabilityAgent": ExplainabilityAgent(self.specialist_llm),
        }

        # Session state — persists across turns
        self._session_state: dict[str, Any] = {
            "conversation_history": [],
            "user_features": {},
            "risk_profile": None,
            "prior_investment_output": None,
            "turn_count": 0,
            "customer_id": None,
            "customer_known": False,
            "missing_customer_fields": [],
            "awaiting_full_advisory_inputs": [],
            "ground_truth_risk_class": None,
            "proxy_fields": [],
            "proxy_metadata": {},
            "customer_name": None,
            "questionnaire_answers": {},
            "customer_wants_to_stop": False,
            "questionnaire_completed": False,
            # Risk-elicitation loop (parallel to the budget questionnaire
            # state above, kept as separate keys so the two loops can't
            # clobber each other in the same session — see
            # _advance_questionnaire()'s generalisation notes).
            "risk_elicitation_answers": {},
            "risk_customer_wants_to_stop": False,
            "risk_elicitation_completed": False,
            "risk_elicitation_pending_sequence": [],
        }

        logger.info(f"[Orchestrator] Session {self.session_id} initialised")

        if customer_context is not None:
            self._load_customer(customer_id, customer_context=customer_context)
        elif customer_id:
            self._load_customer(customer_id)

        if customer_id:
            self._load_customer_memory(customer_id)


    @staticmethod
    def _derive_client(primary: LLMClient, model: str, role: str, provider: str | None = None) -> LLMClient:
        """
        Return a client for `role`, reusing `primary` where a second one would
        be pointless or harmful.

        Fixes R10: LLMConfig has defined specialist_model and judge_model since
        Phase 1, but LLMClient only ever read ORCHESTRATOR_MODEL and one client
        was shared by everything — so the four specialist agents were running
        the orchestrator's model to narrate figures already computed in Python.

        Two cases MUST reuse `primary` rather than build a new client:

          mock mode   Tests inject a mock client and some patch .chat on that
                      exact instance (see make_agent_with_responses). A second
                      client would bypass the fake and could reach a real API
                      from a unit test.
          same model  No reason to hold two clients for one model.
        """
        if primary.mode == "mock":
            return primary
        same_model = not model or model == primary.model
        same_provider = provider is None or provider == primary.mode
        if same_model and same_provider:
            return primary

        client = LLMClient(model=model, provider=provider)
        logger.info(
            f"[Orchestrator] {role} tier → provider={client.mode} model={model} "
            f"(mode={client.mode}); orchestrator tier → provider={primary.mode} "
            f"model={primary.model}"
        )
        return client

    # Existing-customer pipeline

    def _load_customer(
        self,
        customer_id: str | None,
        customer_context: dict[str, Any] | None = None,
    ) -> None:
        if customer_context is not None:
            required = settings.risk.required_features
            features = {k: v for k, v in customer_context.items() if k in required}
            missing = [k for k in required if k not in features]
            result = {
                "features": features,
                "missing_fields": missing,
                "ground_truth_risk_class": None,
                "source_dataset": "bank_api_push",
            }
            self._session_state["customer_name"] = customer_context.get("name")
        else:
            result = self.customer_store.lookup(customer_id) if customer_id else None

        if result is None:
            self._session_state["customer_id"] = customer_id
            self._session_state["customer_known"] = False
            logger.info(
                f"[Orchestrator] customer_id={customer_id!r} not found = "
                f"treating as new customer, elicitation will proceed normally."
            )
            self.audit_log.record_customer_load(
                customer_id=customer_id or "unknown",
                known=False,
                fields_loaded=0,
                missing_fields=[],
            )
            return

        features = dict(result["features"])
        missing = list(result["missing_fields"])

        proxy_fields: list[str] = []
        proxy_metadata: dict[str, Any] = {}
        if "loss_tolerance" in missing:
            proxy = derive_loss_tolerance_proxy(features)
            if proxy is not None:
                features["loss_tolerance"] = proxy["value"]
                missing.remove("loss_tolerance")
                proxy_fields.append("loss_tolerance")
                proxy_metadata["loss_tolerance"] = proxy
                logger.info(
                    f"[Orchestrator] Derived loss_tolerance proxy="
                    f"{proxy['value']} (confidence={proxy['confidence']}, "
                    f"basis={proxy['basis']}) for customer_id={customer_id!r}. "
                    f"Proceeding without user confirmation — disclosed instead "
                    f"via ExplainabilityAgent's estimated-input note "
                    f"(see explainability/explainability_agent.py)."
                )

        self._session_state["customer_id"] = customer_id
        self._session_state["customer_known"] = True
        self._session_state["user_features"] = features
        self._session_state["missing_customer_fields"] = missing
        self._session_state["ground_truth_risk_class"] = result["ground_truth_risk_class"]
        self._session_state["proxy_fields"] = proxy_fields
        self._session_state["proxy_metadata"] = proxy_metadata

        logger.info(
            f"[Orchestrator] Loaded customer_id={customer_id!r} via "
            f"{result['source_dataset']} — {len(features)} features "
            f"pre-filled ({proxy_fields} derived), missing={missing}"
        )
        self.audit_log.record_customer_load(
            customer_id=customer_id or "unknown",
            known=True,
            fields_loaded=len(features),
            missing_fields=missing,
        )

    def _load_customer_memory(self, customer_id: str) -> None:
        """
        Day 9 (G7) — restore what THIS SYSTEM remembers about a returning
        customer, on top of whatever _load_customer() just restored from
        CustomerStore. Two different things, deliberately: _load_customer
        restores the bank's own record (age, income, ...); this restores
        what HALO itself learned or computed in earlier sessions — see
        data/customer_memory.py's module docstring for why these aren't
        merged into one store.

        risk_profile restored here lands in the EXACT session_state key
        (_session_state["risk_profile"]) _execute_plan() already reads as
        a cross-turn fallback for ExplainabilityAgent/InvestmentAgent
        (see _cache_agent_output's docstring) and available_context_keys()
        already treats as a source of a satisfied "risk_class" (orchestrator
        /planner.py). No new plumbing needed for a returning customer's
        remembered risk profile to actually get used — it was already
        the mechanism Day 6/7 built for a DIFFERENT reason (staying
        useful across turns within one session); this just seeds it one
        step earlier, from BEFORE the session started at all.

        preferences restored here seed ConversationalAgent's slot
        tracking directly, so a previously-stated investment_goal or name
        is not re-asked — mirrors exactly how the risk/budget
        questionnaires already seed slots from context (seed_slots=...).
        """
        if not settings.conversational.memory_enabled:
            return
        memory = self._memory_store.get(customer_id)
        if memory is None:
            return

        if memory.risk_profile:
            self._session_state["risk_profile"] = memory.risk_profile

        if memory.preferences:
            self._agents["ConversationalAgent"].update_slots(memory.preferences)

        self._session_state["lifetime_turn_count"] = memory.turn_count
        self._session_state["last_seen"] = memory.last_seen

        logger.info(
            f"[Orchestrator] Restored memory for customer_id={customer_id!r} — "
            f"lifetime_turns={memory.turn_count} last_seen={memory.last_seen} "
            f"risk_profile={'present' if memory.risk_profile else 'none'} "
            f"preferences={sorted(memory.preferences)}"
        )

    def _persist_customer_memory(self) -> None:
        """
        Day 9 (G7) — called at the end of every turn (see process_turn()).
        A no-op for a session with no identified customer_id, matching
        update_customer_features()'s own guard — memory needs somewhere
        durable to be keyed to.
        """
        if not settings.conversational.memory_enabled:
            return
        customer_id = self._session_state.get("customer_id")
        if not customer_id:
            return

        conv_agent = self._agents["ConversationalAgent"]
        preferences = extract_preferences_from_slots(conv_agent.slots)

        self._memory_store.upsert(
            customer_id,
            risk_profile=self._session_state.get("risk_profile"),
            preferences=preferences or None,
            turn_increment=1,
        )

    def _estimate_history_tokens(self) -> int:
        """
        Cheap, deterministic estimate — ~4 characters per token, the
        standard rule-of-thumb for English text absent a real tokenizer.
        Good enough to trigger summarisation near the right point; this
        is a local-growth bound, not an attempt to match a specific
        model's tokenizer exactly (see settings.conversational's
        summarise_above_tokens docstring).
        """
        history = self._session_state["conversation_history"]
        total_chars = sum(len(m.get("content", "")) for m in history)
        return total_chars // 4

    def _llm_summarise(self, messages: list[dict]) -> str:
        """
        One LLM call, orchestrator-tier client (same reasoning as
        _synthesise_response: this is a system-level operation on the
        conversation as a whole, not a specialist task). Degrades to a
        short deterministic placeholder on failure rather than raising —
        summarisation must never be the reason a turn fails; the
        un-summarised history is still sitting right there if this call
        doesn't come back.
        """
        transcript = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages
        )
        try:
            response = self.llm.chat(
                system=SUMMARISER_SYSTEM,
                messages=[{"role": "user", "content": transcript}],
                temperature=0.0,
            )
            trace.emit("LLM", f"← {response.tokens_used} tok (summarisation)",
                       model=response.model, mode=self.llm.mode)
            return response.content.strip()
        except Exception as exc:
            logger.error(f"[Orchestrator] Summarisation failed: {exc}")
            return (
                f"[{len(messages)} earlier messages — summarisation "
                f"unavailable this turn, original content not recovered here]"
            )

    def _maybe_summarise(self) -> None:
        """
        Day 9 (G17). Called at the end of every turn (see process_turn()),
        after the turn's own exchange has already been appended — so a
        summarisation triggered by THIS turn still keeps this turn's own
        exchange verbatim (it's within the last summarise_keep_last_n
        entries), never summarising something that hasn't been replied
        to yet.
        """
        cfg = settings.conversational
        if not cfg.summarise_enabled:
            return
        history = self._session_state["conversation_history"]
        keep = cfg.summarise_keep_last_n
        if len(history) <= keep:
            return
        if self._estimate_history_tokens() < cfg.summarise_above_tokens:
            return

        to_summarise = history[:-keep]
        summary = self._llm_summarise(to_summarise)
        self._session_state["conversation_history"] = [
            {"role": "system", "content": f"[Earlier conversation] {summary}"}
        ] + history[-keep:]
        logger.info(
            f"[Orchestrator] Summarised {len(to_summarise)} earlier history "
            f"entries -> 1 summary + {keep} kept verbatim"
        )

    def update_customer_features(self, features: dict[str, Any]) -> None:
        """
        Merge updated/newly-elicited features into the current session and
        persist them back to the CustomerStore.

        Covers two cases with the same call:
          - Flow 3: known customer, changed circumstances
            (e.g. income changed since last visit)
          - Known customer with missing_customer_fields now supplied
            (e.g. loss_tolerance elicited on first contact)

        A brand-new customer_id not yet in the store is created on first
        save — this is also how a new customer graduates into the
        existing-customer pipeline for their next session.
        """
        self._session_state["user_features"].update(features)

        # A field the user later corrects (via Flow 3) is no longer an
        # unconfirmed estimate — drop it from both proxy trackers.
        for f in list(features.keys()):
            if f in self._session_state["proxy_fields"]:
                self._session_state["proxy_fields"].remove(f)
            self._session_state["proxy_metadata"].pop(f, None)

        customer_id = self._session_state.get("customer_id")
        if not customer_id:
            logger.warning(
                "[Orchestrator] update_customer_features called with no "
                "customer_id on session — features kept in-session only, "
                "not persisted to CustomerStore."
            )
            return

        self.customer_store.save(customer_id, features)

        # Recompute which required fields are still missing after the update
        still_missing = [
            f for f in self._session_state["missing_customer_fields"]
            if f not in features
        ]
        self._session_state["missing_customer_fields"] = still_missing
        self._session_state["customer_known"] = True

        logger.info(
            f"[Orchestrator] customer_id={customer_id!r} features updated: "
            f"{sorted(features.keys())} — still missing: {still_missing}"
        )


    # Layer 1 - Goal decomposition
    def _classify_intent(self, user_message: str) -> tuple[RoutingDecision, str, float]:
        """
        Classify user intent using the ConversationalAgent's Banking77 classifier.
        Returns (RoutingDecision, intent_string, confidence).

        Mapping from ConversationalAgent intent buckets to RoutingDecision:
          general_query       → CONVERSATIONAL_ONLY
          risk_profiling      → RISK_PROFILING
          investment_advice   → INVESTMENT
          budget_analysis     → BUDGET
          product_suggestion  → INVESTMENT (treated as investment query)
          explanation_request → EXPLANATION_REQUEST
          out_of_scope        → CONVERSATIONAL_ONLY
        """
        INTENT_TO_ROUTING = {
            "general_query":       RoutingDecision.CONVERSATIONAL_ONLY,
            "risk_profiling":      RoutingDecision.RISK_PROFILING,
            "investment_advice":   RoutingDecision.INVESTMENT,
            "budget_analysis":     RoutingDecision.BUDGET,
            "product_suggestion":  RoutingDecision.INVESTMENT,
            "explanation_request": RoutingDecision.EXPLANATION_REQUEST,
            "full_advisory":       RoutingDecision.FULL_ADVISORY,
            "out_of_scope":        RoutingDecision.CONVERSATIONAL_ONLY,
        }

        try:
            conv_agent = self._agents["ConversationalAgent"]
            intent, confidence = conv_agent.classify_only(user_message)
            confidence = float(confidence)

            # Below confidence threshold → treat as general query
            if confidence < settings.orchestrator.routing_confidence_threshold:
                logger.info(
                    f"[Orchestrator] Intent '{intent}' confidence {confidence:.2f} "
                    f"below threshold {settings.orchestrator.routing_confidence_threshold} "
                    f"→ CONVERSATIONAL_ONLY"
                )
                return RoutingDecision.CONVERSATIONAL_ONLY, intent, confidence

            _unreachable = set(RoutingDecision) - set(INTENT_TO_ROUTING.values())

            if _unreachable:
                logger.debug(
                    f"[Orchestrator] Routing decisions unreachable from any "
                    f"intent: {sorted(r.value for r in _unreachable)}. "
                    f"Either map an intent to them or remove them."
                )

            routing = INTENT_TO_ROUTING.get(intent, RoutingDecision.CONVERSATIONAL_ONLY)
            return routing, intent, confidence

        except Exception as exc:
            logger.error(f"[Orchestrator] Intent classification failed: {exc}")
            return RoutingDecision.CONVERSATIONAL_ONLY, "general_query", 0.0

    # Layer 2 - Agent selection and execution

    def _execute_agent(
        self, agent_name: str, context: dict
    ) -> tuple[AgentResult, bool]:
        """
        Execute one agent with retry and failure recovery (O4).
        Returns (result, was_recovered).
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            logger.error(f"[Orchestrator] Unknown agent: {agent_name}")
            return AgentResult(
                agent_name=agent_name,
                success=False,
                error=f"Agent '{agent_name}' not in registry",
            ), False

        last_error = None
        _max_retries = settings.orchestrator.max_agent_retries
        for attempt in range(_max_retries + 1):
            try:
                with trace.span("▶ AGENT", agent_name,
                                attempt=f"{attempt + 1}/{_max_retries + 1}"):
                    result = agent.run(context)
                trace.emit("  ↳ result",
                           f"status={result.payload.get('status', 'ok')}",
                           ok=result.success,
                           ms=round(result.duration_ms),
                           tokens=result.tokens_used)
                return result, False
            except Exception as exc:
                last_error = exc
                if attempt < settings.orchestrator.max_agent_retries:
                    logger.warning(
                        f"[Orchestrator] {agent_name} attempt {attempt+1} failed: {exc} "
                        f"— retrying"
                    )
                    self.audit_log.record_agent_call(
                        turn_id=context.get("_turn_id", "unknown"),
                        agent_name=agent_name,
                        success=False,
                        duration_ms=0.0,
                        tokens_used=0,
                        step_id="retry",
                        error=str(exc),
                    )

        # All retries exhausted — invoke failure handler (O4)
        recovery = self._failure_handler.attempt_recovery(
            agent_name, last_error, context
        )
        self.audit_log.record_agent_failure(
            turn_id=context.get("_turn_id", "unknown"),
            agent_name=agent_name,
            error=str(last_error),
            recovery_strategy=recovery["strategy"],
            recovery_success=recovery["success"],
        )
        trace.emit("↻ RECOVER", agent_name,
                   strategy=recovery["strategy"], success=recovery["success"])

        recovered_result = AgentResult(
            agent_name=agent_name,
            success=recovery["success"],
            payload=recovery["recovered_payload"],
            error=str(last_error),
        )
        return recovered_result, True

    def _build_context(self, user_message: str) -> dict[str, Any]:
        """Build the context dict passed to every agent."""
        return {
            "user_message": user_message,
            "session_id": self.session_id,
            **self._session_state,
        }

    # Which context key carries an agent's FULL payload downstream, for
    # consumers (today, only ExplainabilityAgent) that need more than the
    # flattened keys CAPABILITIES.produces declares. Not derivable from the
    # class name mechanically (RiskProfilingAgent -> "risk_agent_payload",
    # not "riskprofiling_agent_payload") — this is the exact set of keys
    # explainability/explainability_agent.py already reads via context.get().
    _AGENT_PAYLOAD_CONTEXT_KEY: dict[str, str] = {
        "RiskProfilingAgent": "risk_agent_payload",
        "InvestmentAgent": "investment_agent_payload",
        "BudgetAgent": "budget_agent_payload",
    }

    def _publish(self, agent_name: str, result: AgentResult) -> dict[str, Any]:
        """
        Day 6 (G6 dynamic routing) — declarative producer -> context mapping.

        Replaces the hand-written `if agent_name == "InvestmentAgent":
        context["risk_class"] = ...` blocks that used to live in this file.
        Those blocks were the exact class of bug R2 came from: the injection
        and the capability declaration were two separate, hand-maintained
        statements of the same fact, and they had already drifted once.
        Here there is one statement — agents/payloads.CAPABILITIES[agent
        _name].produces — and this function is the only place that reads it
        to decide what a successful run adds to context.

        WHY produces ALONE ISN'T QUITE ENOUGH, AND WHAT _AGENT_PAYLOAD_
        CONTEXT_KEY IS FOR
            CAPABILITIES.produces is deliberately abstract ("risk_class",
            not "the whole RiskProfilingAgent payload") because that
            abstraction is what the planner reasons over. But
            ExplainabilityAgent's job is literally to narrate what earlier
            agents did in full — SHAP attributions, the ranked shortlist,
            benchmark gaps — which cannot be flattened into a handful of
            top-level keys without losing the structure it explains. So a
            successful run also publishes its full payload under a fixed,
            documented key, alongside the flattened ones. Nothing here is
            new capability; it's the same two things _run_agent_sequence
            always injected, now driven by a declaration instead of an
            agent-name string match.
        """
        if not result.success:
            return {}
        published: dict[str, Any] = {}
        cap = CAPABILITIES.get(agent_name)
        if cap is not None:
            for key in cap.produces:
                value = result.payload.get(key)
                if value is not None:
                    published[key] = value
        payload_key = self._AGENT_PAYLOAD_CONTEXT_KEY.get(agent_name)
        if payload_key is not None:
            published[payload_key] = result.payload
        return published

    def _cache_agent_output(
        self, agent_name: str, result: AgentResult, context: dict,
    ) -> None:
        """
        Cross-TURN caching + hallucination audit logging. Split out from
        _execute_plan()'s loop only for readability — unchanged in
        substance from what _run_agent_sequence always did here. This is
        deliberately separate from _publish(): _publish() is what THIS
        turn's later steps see; this is what NEXT turn's ExplainabilityAgent
        falls back to when this turn doesn't re-run RiskProfilingAgent/
        InvestmentAgent at all (e.g. a bare "why?" follow-up).
        """
        if not result.success:
            return
        if agent_name == "RiskProfilingAgent":
            self._session_state["risk_profile"] = result.payload
        if agent_name == "InvestmentAgent":
            self._session_state["prior_investment_output"] = result.payload
            hreport = result.payload.get("hallucination_report")
            if hreport and settings.hallucination.log_flagged_to_audit:
                for claim_record in hreport.get("claims", []):
                    if claim_record.get("flagged"):
                        self.audit_log.record_hallucination_flag(
                            turn_id=context.get("_turn_id", "unknown"),
                            agent_name=agent_name,
                            claim=claim_record["claim"],
                            score=claim_record["score"],
                            threshold=settings.hallucination.hhem_threshold,
                            mode=hreport.get("mode", "fallback"),
                        )

    def _find_producer(self, need: str, exclude: frozenset[str]) -> str | None:
        """
        Which capability declares `need` in its produces, excluding names
        in `exclude` (the collaboration chain's visited set — an agent can
        never be asked to produce for itself, and a cycle can't loop back
        through an agent already in the chain). validate_capability_graph()
        (agents/payloads.py) guarantees at most one match exists; this
        just finds it.
        """
        for name, cap in CAPABILITIES.items():
            if name in exclude:
                continue
            if need in cap.produces:
                return name
        return None

    def _satisfy_needs(
        self,
        result: AgentResult,
        context: dict,
        depth: int = 0,
        visited: frozenset[str] | None = None,
    ) -> tuple[AgentResult, list[AgentResult], list[dict]]:
        """
        Day 7 (G6 §6.6) — bounded mid-plan collaboration.

        An agent that cannot proceed names what it's missing instead of
        just refusing (payload={"status": "needs_input", "needs": [...]}
        — see explainability/explainability_agent.py's one current
        emitter). This finds a capability that produces each need, runs
        it, and retries the ORIGINAL agent — once, with everything now in
        context — rather than leaving a "needs_input" result to propagate
        out as if it were a final answer.

        WHY THIS IS DIFFERENT FROM DAY 6's SKIP-CHECK
            Day 6 (_execute_plan's dynamic gate) is PREVENTION: don't call
            an agent whose precondition is already known to be unmet.
            Collaboration is REPAIR: an agent got called anyway (dynamic=
            False on the static path, or a need too specific for the
            coarse CAPABILITIES.requires declaration to have caught up
            front) and, rather than just failing, names exactly what's
            missing so the orchestrator can go get it. The two are not
            redundant — Day 6 stops wasted calls it can predict; this
            handles the ones it couldn't.

        THREE BOUNDS, ALL NECESSARY (settings.collaboration)
            max_depth            — a chain of retries this many producer-
                                    then-retry hops deep and no further.
                                    Prevents A needing B needing C ...
                                    unboundedly, even where no cycle
                                    exists.
            visited set           — prevents A -> B -> A: an agent already
                                    in this chain is never re-entered as
                                    a producer, so a genuine cycle in
                                    CAPABILITIES (which shouldn't exist —
                                    see validate_capability_graph()'s
                                    uniqueness check — but this is the
                                    runtime backstop, not the only one)
                                    cannot spin.
            producibility check   — a need with no producer at all (not
                                    "the producer failed", but "nothing
                                    in CAPABILITIES makes this") is a hard
                                    stop, returned as the ORIGINAL
                                    needs_input result unchanged. Retrying
                                    something structurally unproducible
                                    would just ask again forever.

        Returns (final_result, produced_results, events).
            final_result     — a fresh retry of the original agent if every
                                stated need was resolved, or the original
                                needs_input result unchanged otherwise —
                                never a half-updated result, so a caller
                                checking `.success`/`.payload["status"]`
                                never has to know collaboration happened.
            produced_results — the AgentResult for every producer actually
                                run while resolving this request, in run
                                order. The caller (_execute_plan) extends
                                its own `results` list with these BEFORE
                                appending final_result, so a producer that
                                ran as part of collaboration shows up in
                                agents_invoked exactly as if it had been a
                                normal plan step — hiding it there would
                                understate what the turn actually did and
                                cost, the same reasoning Day 6's `skipped`
                                exists for in the other direction.
            events            — one dict per (need, attempt), consumed by
                                CollaborationStats and reported on
                                OrchestratorResult.collaboration_events.
        """
        cfg = settings.collaboration
        events: list[dict] = []
        produced: list[AgentResult] = []

        if not cfg.enabled or result.payload.get("status") != "needs_input":
            return result, produced, events

        requester = result.agent_name
        needs = list(result.payload.get("needs", []))
        chain = (visited or frozenset()) | {requester}

        if depth >= cfg.max_depth:
            events.append({
                "requester": requester, "needs": needs, "resolved": False,
                "reason": f"max_depth={cfg.max_depth} reached",
            })
            logger.info(
                f"[Orchestrator] collaboration for {requester} stopped — "
                f"{events[-1]['reason']}"
            )
            trace.emit("🤝 COLLABORATE", f"{requester} — depth limit reached",
                       needs=needs)
            return result, produced, events

        all_resolved = True
        for need in needs:
            if context.get(need) not in (None, "", [], {}, ()):
                continue  # already available — nothing to fetch for this one

            producer = self._find_producer(need, exclude=chain)
            if producer is None:
                events.append({
                    "requester": requester, "needs": [need], "resolved": False,
                    "reason": f"{need!r} is not producible by anything reachable",
                })
                logger.info(
                    f"[Orchestrator] {requester} needs {need!r} — no producer "
                    f"found (chain so far: {sorted(chain)})"
                )
                all_resolved = False
                continue

            trace.emit("🤝 COLLABORATE", f"{requester} needs {need!r} → {producer}",
                       depth=depth)
            logger.info(
                f"[Orchestrator] {requester} needs {need!r} — running "
                f"{producer} (collaboration depth={depth})"
            )
            producer_result, _recovered = self._execute_agent(producer, context)

            # The producer might itself need something — resolve that one
            # level deeper before judging whether it produced what THIS
            # need was waiting on.
            if producer_result.payload.get("status") == "needs_input":
                producer_result, nested_produced, nested_events = self._satisfy_needs(
                    producer_result, context, depth=depth + 1, visited=chain,
                )
                produced.extend(nested_produced)
                events.extend(nested_events)
            produced.append(producer_result)

            published = self._publish(producer, producer_result)
            if published:
                context.update(published)
            self._cache_agent_output(producer, producer_result, context)

            resolved = context.get(need) not in (None, "", [], {}, ())
            events.append({
                "requester": requester, "needs": [need], "resolved": resolved,
                "producer": producer,
                "reason": "" if resolved else
                          f"{producer} ran but did not produce {need!r} this time",
            })
            if not resolved:
                all_resolved = False

        if not all_resolved:
            return result, produced, events

        retried, _recovered = self._execute_agent(requester, context)
        events.append({"requester": requester, "needs": needs, "retried": True})
        trace.emit("🤝 COLLABORATE",
                   f"{requester} retried — status={retried.payload.get('status', 'complete')}")
        return retried, produced, events

    def _execute_plan(
        self,
        agent_names: list[str],
        context: dict,
        dynamic: bool = True,
    ) -> tuple[list[AgentResult], list[str], list[dict], list[dict]]:
        """
        Day 6 (G6) — dynamic tool routing: the executor that satisfies
        capability dependencies at RUN TIME, rather than trusting that a
        plan valid at planning time is still valid by the time execution
        reaches a given step.

        WHY RE-CHECK AT ALL, GIVEN THE PLANNER/STATIC TABLE ALREADY DECIDED
            A plan validated against the capability graph can still fail at
            runtime: RiskProfilingAgent may itself return status=
            "incomplete" because a feature the CALLER didn't know was
            missing turns out to be missing, so risk_class never lands in
            context and InvestmentAgent's precondition silently stops
            holding one step later. Before Day 6 that meant InvestmentAgent
            ran anyway and produced its own "incomplete" refusal — a wasted
            LLM call whose only output was a second way of saying the same
            thing. Re-checking `requires` immediately before each step turns
            that into a SKIP, with a stated reason, and no call made.

        WHY `dynamic` EXISTS, AND DEFAULTS TO True
            "Dynamic routing" (Day 6) is specifically about EXECUTING WHAT
            THE PLANNER PROPOSED (Day 4-5) at run time instead of blindly
            trusting it — see the module-level framing in orchestrator/
            planner.py: the static table is deliberately kept as "the
            safety net and the baseline", unchanged, not a second thing
            Day 6 also reaches into. tests/unit/test_full_advisory.py::
            TestReadinessGate::test_gate_does_not_touch_other_routes says
            this in code: INVESTMENT's agents_invoked is baked into
            committed RQ2/RQ4 results and must not shift when the plan for
            that turn came from the static table, only when it came from
            an LLM proposal the validator actually accepted. So
            process_turn() passes dynamic=(turn_plan.accepted) — True only
            when plan.source == "planner" — and dynamic=False reproduces
            the exact pre-Day-6 behaviour: every named step runs, no
            skip-check, agents report their own "incomplete" as they
            always did. The declarative _publish() handoff below runs
            EITHER way — that part was never in tension with any test, it
            is the same values the old hand-written injection blocks set,
            just read from one declaration instead of an agent-name
            string match.

        WHY THIS REUSES orchestrator.planner.available_context_keys()
            RATHER THAN A SIMPLER `set(context)` CHECK
            A context dict always HAS a "user_features" key (seeded to {}
            at session start) — a bare key-presence check would treat an
            empty profile as satisfied on turn one, which is precisely the
            plans this gate exists to stop. available_context_keys() is
            also what PlanValidator used to accept this plan in the first
            place; reusing it here means the planning-time check and the
            execution-time check can never quietly disagree about what
            "available" means — they're the same function, called twice.

        Returns (results, recovered_agent_names, skipped, collaboration_
        events), where each skipped entry is {"agent": name, "reason":
        "missing [...]"} — always [] when dynamic=False. This supersedes
        _run_agent_sequence, which now delegates here with dynamic=False
        and drops the extra two lists, for full backward compatibility
        with any caller that doesn't know Day 6/7 exist.
        """
        results: list[AgentResult] = []
        recovered: list[str] = []
        skipped: list[dict] = []
        collaboration_events: list[dict] = []

        # Cross-turn fallback for ExplainabilityAgent (see
        # _cache_agent_output's docstring): if THIS turn re-runs
        # RiskProfilingAgent/InvestmentAgent, _publish() below overwrites
        # these with the fresh payload before Explainability's own turn in
        # the loop; if it doesn't, Explainability still sees last turn's.
        context.setdefault(
            "risk_agent_payload", self._session_state.get("risk_profile") or {}
        )
        context.setdefault(
            "investment_agent_payload",
            self._session_state.get("prior_investment_output") or {},
        )

        for agent_name in agent_names:
            if dynamic:
                cap = CAPABILITIES.get(agent_name)
                if cap is not None:
                    available = available_context_keys(context)
                    missing = sorted(cap.requires - available)
                    if missing:
                        skipped.append({"agent": agent_name, "reason": f"missing {missing}"})
                        trace.emit("⏭ SKIP", agent_name, missing=missing)
                        logger.info(
                            f"[Orchestrator] {agent_name} skipped at execution "
                            f"— missing {missing}"
                        )
                        continue

            result, was_recovered = self._execute_agent(agent_name, context)

            if result.success and result.payload.get("status") == "needs_input":
                result, produced, events = self._satisfy_needs(result, context)
                # Producers that ran as part of collaboration are real
                # agent invocations this turn — they belong in agents_
                # invoked/agent_results exactly like a normal plan step,
                # not hidden inside _satisfy_needs. See its docstring.
                results.extend(produced)
                collaboration_events.extend(events)

            results.append(result)
            if was_recovered:
                recovered.append(agent_name)

            published = self._publish(agent_name, result)
            if published:
                context.update(published)
                prev_agent = results[-2].agent_name if len(results) > 1 else "(session cache)"
                trace.emit("⇄ HANDOFF", f"{prev_agent} → {agent_name}",
                           keys=sorted(published))

            self._cache_agent_output(agent_name, result, context)

        if collaboration_events:
            self._collaboration_stats.record(collaboration_events)

        return results, recovered, skipped, collaboration_events

    def _run_agent_sequence(
        self,
        agent_names: list[str],
        context: dict,
    ) -> tuple[list[AgentResult], list[str]]:
        """
        Back-compat shim over _execute_plan() (Day 6/7) — same two-tuple
        signature this had before Day 6, AND dynamic=False, so any
        existing caller reproduces the exact pre-Day-6 behaviour: every
        named agent runs, none are skipped. Collaboration (Day 7) still
        runs regardless of `dynamic` — repair, unlike the skip-check, was
        never in tension with the static-path tests, since a needs_input
        result becomes a retried (or unchanged) AgentResult either way,
        not a change to WHICH agents get invoked from agent_names itself.
        """
        results, recovered, _skipped, _collab = self._execute_plan(
            agent_names, context, dynamic=False,
        )
        return results, recovered

    @property
    def collaboration_stats(self) -> dict:
        """Session-level collaboration counters — read by the evaluation
        harness (RQ4: collaboration events per turn, resolution rate)."""
        return self._collaboration_stats.as_dict()

    # Layer 3 - Execution monitoring
    def _check_constraints(
        self, response_text: str, agent_results: list[AgentResult], turn_id: str = "unknown",
    ) -> tuple[str, list[dict], bool]:
        """
        Apply financial rule constraints (E4 — Nguyen et al. [8]).
        Returns (response_text, violations, was_blocked).
        Hard blocks replace response with a safe fallback.
        """
        violations: list[dict] = []
        risk_class = None
        claimed_return = None

        # Extract risk class and top product return from results
        for r in agent_results:
            if r.agent_name == "RiskProfilingAgent" and r.success:
                risk_class = r.payload.get("risk_class")
            if r.agent_name == "InvestmentAgent" and r.success:
                shortlist = r.payload.get("shortlist", [])
                if shortlist:
                    claimed_return = shortlist[0].get("expected_return_pct")

        advisory = any(
            r.agent_name in ("InvestmentAgent", "BudgetAgent", "RiskProfilingAgent")
            and r.success
            for r in agent_results
        )

        deliverable, raw_violations = financial_constraints.validate_response(
            response_text=response_text,
            risk_class=risk_class,
            claimed_return=claimed_return,
            advisory=advisory,
        )

        for v in raw_violations:
            vdict = {
                "rule_id": v.rule_id,
                "severity": v.severity,
                "description": v.description,
            }
            violations.append(vdict)
            self.audit_log.record_constraint_violation(
                turn_id=turn_id,  # filled in by caller
                rule_id=v.rule_id,
                severity=v.severity,
                description=v.description,
                blocked=not deliverable,
            )

        if not deliverable:
            blocked_response = (
                "I'm unable to deliver this response as it does not meet "
                "regulatory safety requirements. "
                "This is not regulated financial advice. "
                "Consult a qualified advisor. "
                "Past performance is not indicative of future results."
            )
            return blocked_response, violations, True

        return response_text, violations, False

    def _check_approval_gate(
        self,
        agent_results: list[AgentResult],
        violations: list[dict],
        conflicts: list[dict],
    ) -> list[str]:
        """
        Day 8 — human approval gate. Returns the reasons this turn should
        be held for review, or [] if none apply. Every check below reuses
        a signal this system already computes elsewhere in the same
        turn — see settings.approval's docstring for exactly which, and
        why LOW_CONFIDENCE_AGGRESSIVE is read from `conflicts` rather than
        re-derived from risk_class (ConflictResolver has already
        downgraded risk_class to "moderate" for routing by the time this
        runs — the conflict record is what preserves that it happened).

        Config-gated per check (settings.approval.gate_on_*), per the
        build plan's own instruction that trigger conditions are config,
        not hard-coded — an ablation can switch off one trigger without
        touching this method.
        """
        cfg = settings.approval
        if not cfg.enabled:
            return []
        reasons: list[str] = []

        if cfg.gate_on_hard_block and any(
            v.get("severity") == "hard_block" for v in violations
        ):
            reasons.append("hard_block constraint violation")

        if cfg.gate_on_low_confidence_aggressive and any(
            c.get("type") == "LOW_CONFIDENCE_AGGRESSIVE" for c in conflicts
        ):
            reasons.append(
                "risk class was aggressive/moderately_aggressive with "
                "confidence below threshold"
            )

        inv_result = next(
            (r for r in agent_results if r.agent_name == "InvestmentAgent"), None
        )
        if inv_result is not None and inv_result.success:
            if cfg.gate_on_hallucination_flagged and inv_result.payload.get(
                "hallucination_flagged"
            ):
                reasons.append("hallucination detector flagged a claim in the synthesis")

            shortlist = inv_result.payload.get("shortlist") or []
            over_ceiling = [
                p for p in shortlist
                if isinstance(p.get("expected_return_pct"), (int, float))
                and p["expected_return_pct"] > cfg.max_expected_return_pct
            ]
            if over_ceiling:
                names = ", ".join(
                    p.get("name", p.get("product_id", "?")) for p in over_ceiling
                )
                reasons.append(
                    f"recommendation includes expected_return_pct above "
                    f"{cfg.max_expected_return_pct}% ({names})"
                )

        return reasons

    def collect_approved_response(self, turn_id: str) -> str | None:
        """
        Retrieve a gated turn's real response once a reviewer has approved
        it, and mark it delivered. Returns None if the turn is unknown,
        still pending, or was rejected — never raises, since "nothing to
        collect yet" is the normal outcome of asking too early, not an
        error (mirroring ProductDataClient.get_fact()'s reasoning for
        returning None rather than raising on the common case).

        This is the customer-facing half of the state machine's last
        transition (APPROVED -> DELIVERED); approving and delivering are
        deliberately separate events — see ApprovalStore.mark_delivered()'s
        docstring for why "approved" cannot itself mean "the customer has
        seen it".
        """
        pending = self._approval_store.get(turn_id)
        if pending is None or pending.status != "approved":
            return None
        delivered = self._approval_store.mark_delivered(turn_id)
        return delivered.draft_response

    def collect_all_approved_responses(self) -> list[str]:
        """
        Every turn from THIS session that a reviewer approved since it was
        last checked, delivered now. Meant to be called at the start of a
        turn in a live session (see run_demo.py's -i mode) — the natural
        place for "watch it deliver" to actually happen from the
        customer's side of an ongoing conversation, since the review
        itself typically happens in a separate process entirely.
        """
        approved = self._approval_store.list_approved(session_id=self.session_id)
        return [
            text for p in approved
            if (text := self.collect_approved_response(p.turn_id)) is not None
        ]

    def _synthesise_response(
        self,
        user_message: str,
        agent_results: list[AgentResult],
        routing_decision: RoutingDecision,
    ) -> str:
        """
        LLM synthesis turn — Orchestrator integrates all agent outputs.
        Uses ORCHESTRATOR_SYSTEM prompt (synthesis role only — O1 Layer 3).
        """
        # Build results summary for synthesis prompt
        results_summary_parts = []
        for r in agent_results:
            if r.success and r.payload:
                key_fields = {
                    k: v for k, v in r.payload.items()
                    if k in (
                        "risk_class", "confidence", "rationale",
                        "synthesis", "recommendations_text",
                        "full_explanation", "savings_rate_pct",
                        "disposable_income",
                    )
                }
                results_summary_parts.append(
                    f"[{r.agent_name}]\n{json.dumps(key_fields, indent=2)}"
                )
            elif r.payload.get("message"):
                status_fields = {
                    k: v for k, v in r.payload.items()
                    if k in ("status", "message")
                }
                results_summary_parts.append(
                    f"[{r.agent_name} — UNSUCCESSFUL]\n{json.dumps(status_fields, indent=2)}"
                )

        results_summary = "\n\n".join(results_summary_parts)
        slots = self._session_state.get("user_features", {})

        synthesis_prompt = (
            f"User message: '{user_message}'\n"
            f"Routing: {routing_decision.value}\n"
            f"Known user context: {json.dumps(slots)}\n\n"
            f"Agent outputs:\n{results_summary}\n\n"
            f"Synthesise a coherent, concise response (max 150 words) "
            f"for the retail investor. Include the CBI disclaimer if any "
            f"investment content is present. If any agent output above is "
            f"marked UNSUCCESSFUL, state its 'message' plainly and honestly "
            f"as part of your response — do not omit it and do not invent "
            f"a recommendation to fill the gap."
        )

        try:
            trace.emit("PROMPT", "orchestrator_synthesis",
                       chars=len(synthesis_prompt), temp=0.2)
            response = self.llm.chat(
                system=ORCHESTRATOR_SYSTEM,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
            )
            trace.emit("LLM", f"← {response.tokens_used} tok",
                       model=response.model, mode=self.llm.mode)
            return response.content.strip()
        except Exception as exc:
            logger.error(f"[Orchestrator] Synthesis failed: {exc}")
            # Graceful degradation — use best available agent output
            for r in reversed(agent_results):
                for field in ("full_explanation", "synthesis",
                              "recommendations_text", "rationale", "response"):
                    if r.success and r.payload.get(field):
                        return str(r.payload[field])
            return (
                "I was unable to process your request at this time. "
                "Please try again or consult a qualified financial advisor."
            )

    # Main entry point
    def process_turn(self, user_message: str) -> OrchestratorResult:
        """
        Process one conversational turn through the full HALO pipeline.

        Flow:
          Layer 1: classify intent → RoutingDecision
          Layer 2: execute agent sequence per routing plan
          Layer 3: resolve conflicts, check constraints, synthesise response
          Audit:   log every step to TRiSM audit log (O3)
        """
        turn_start = time.perf_counter()
        turn_id = str(uuid.uuid4())[:8]
        trace.set_session(self.session_id)
        trace.new_turn(turn_id)
        trace.turn_banner(turn_id, self.session_id,
                          self._session_state["turn_count"] + 1, user_message)
        self._session_state["turn_count"] += 1
        self._session_state["conversation_history"].append(
            {"role": "user", "content": user_message}
        )

        self.audit_log.record_turn_start(turn_id, user_message)
        trace.emit("ORCH", "orchestrator start",
                   customer=self._session_state.get("customer_id"),
                   known=self._session_state.get("customer_known"))

        # -- Layer 1: Classify intent --
        # QUESTIONNAIRE SHORT-CIRCUIT (Section 2)
        #     If ConversationalAgent is mid-questionnaire, this message is an
        #     answer to a question the system itself just asked, and there is
        #     nothing to classify. Running the classifier anyway would spend
        #     an LLM call to label "1200" as a general_query and then route
        #     away from the very form we're in the middle of — the loop would
        #     never close. Skipping it is both cheaper and the only routing
        #     that makes a multi-turn form work.
        conv_agent = self._agents["ConversationalAgent"]
        in_questionnaire = getattr(conv_agent, "questionnaire_active", False)
        if in_questionnaire:
            _kind = getattr(conv_agent, "questionnaire_kind", "budget")
            routing, intent, confidence = (
                RoutingDecision.CONVERSATIONAL_ONLY,
                "risk_profiling" if _kind == "risk" else "budget_analysis",
                1.0,
            )
            trace.emit("L1", f"{_kind} questionnaire mode — classification skipped")
        else:
            with trace.span("[L1]", "goal decomposition"):
                routing, intent, confidence = self._classify_intent(user_message)

        _threshold = settings.orchestrator.routing_confidence_threshold
        trace.emit("ROUTING", f"{intent} → {routing.value}",
                   confidence=round(confidence, 2),
                   reason=("above threshold" if confidence >= _threshold
                           else f"below {_threshold} → conversational"))
        self.audit_log.record_routing(
            turn_id=turn_id,
            routing_decision=routing.value,
            intent=intent,
            confidence=confidence,
            rationale=f"Banking77 intent classification → {intent} ({confidence:.2f})",
        )
        logger.info(
            f"[Orchestrator] turn={turn_id} routing={routing.value} "
            f"intent={intent} confidence={confidence:.2f}"
        )

        # -- Layer 2: Execute agent sequence --
        context = self._build_context(user_message)
        context["_turn_id"] = turn_id

        turn_plan = self._plan_turn(user_message, context, routing, intent, turn_id)
        agent_sequence = list(turn_plan.steps)

        elicitation: str | None = None
        if routing is RoutingDecision.FULL_ADVISORY:
            planned = agent_sequence
            agent_sequence, unmet = self._prune_unsatisfiable(agent_sequence, context)
            if agent_sequence != planned:
                trace.emit("PRUNE", f"{' → '.join(planned)}  ⇒  "
                                    f"{' → '.join(agent_sequence) or '(none runnable)'}",
                           unmet=len(unmet))
                logger.info(
                    f"[Orchestrator] FULL_ADVISORY pruned "
                    f"{[a for a in planned if a not in agent_sequence]} "
                    f"— unmet inputs: {unmet}"
                )
            if not agent_sequence:
                # Nothing can run. Ask for what is missing instead of spending
                # four LLM calls to produce three refusals.
                elicitation = self._elicitation_response(unmet)
                self._session_state["awaiting_full_advisory_inputs"] = unmet
        with trace.span("[L2]", f"plan: {' → '.join(agent_sequence) or 'elicitation'}"):
            # dynamic=True only when the PLANNER's own proposal was
            # accepted (turn_plan.accepted, i.e. source == "planner") — the
            # static table stays the unchanged safety net. See
            # _execute_plan()'s docstring for why this distinction exists.
            agent_results, recovered, skipped, collaboration_events = self._execute_plan(
                agent_sequence, context, dynamic=turn_plan.accepted,
            )
        if skipped:
            trace.emit("SKIPPED", "", agents=[s["agent"] for s in skipped])

        # -- Layer 2b: Section 2 questionnaire loop --
        questionnaire_response = self._advance_questionnaire(
            agent_results, context, turn_id, agent_sequence=agent_sequence, skipped=skipped,
        )
        if questionnaire_response is not None:
            elicitation = questionnaire_response
            agents_extra = [r.agent_name for r in agent_results]
            logger.info(
                f"[Orchestrator] turn={turn_id} questionnaire drives the "
                f"response (agents so far: {agents_extra})"
            )

        # -- Layer 3a: Conflict resolution --
        with trace.span("[L3]", "execution monitoring"):
            if settings.orchestrator.enable_conflict_resolution:
                agent_results, conflicts = self._conflict_resolver.resolve(agent_results)
            else:
                conflicts = []
            for conflict in conflicts:
                self.audit_log.record_conflict(
                    turn_id=turn_id,
                    conflict_type=conflict["type"],
                    description=conflict["description"],
                    resolution=conflict["resolution"],
                )
            if conflicts:
                for r in agent_results:
                    if not r.success:
                        continue
                    if r.agent_name == "RiskProfilingAgent":
                        self._session_state["risk_profile"] = r.payload
                    elif r.agent_name == "InvestmentAgent":
                        self._session_state["prior_investment_output"] = r.payload

            # -- Layer 3b: Synthesis --
            if elicitation is not None:
                # R7 - no agent produced anything to synthesise; the response
                # IS the request for missing inputs.
                raw_response = elicitation
            else:
                raw_response = self._synthesise_response(
                    user_message, agent_results, routing
                )
            # -- Layer 3c: Constraint validation --
            final_response, violations, was_blocked = self._check_constraints(
                raw_response, agent_results
            )

            trace.emit(
                "✓ VALIDATE",
                "final response",
                violations=len(violations),
                blocked=was_blocked,
            )

            # -- Layer 3d: Human approval gate (Day 8) --
            # Deliberately skipped when elicitation is not None — a
            # deterministic "what's your age?" / questionnaire prompt was
            # never synthesised from agent output, so there is nothing
            # here for a reviewer to judge, and gating it would just add
            # a pointless hold to an ordinary clarifying question.
            gate_reasons: list[str] = []
            if elicitation is None:
                gate_reasons = self._check_approval_gate(agent_results, violations, conflicts)
            if gate_reasons:
                pending = PendingApproval(
                    turn_id=turn_id,
                    session_id=self.session_id,
                    reasons=gate_reasons,
                    draft_response=final_response,
                    agent_results=[
                        {
                            "agent_name": r.agent_name,
                            "success": r.success,
                            "payload": r.payload,
                            "error": r.error,
                        }
                        for r in agent_results
                    ],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
                self._approval_store.create(pending)
                self.audit_log.record_approval_gate(turn_id=turn_id, reasons=gate_reasons)
                trace.emit("🔒 APPROVAL GATE", f"turn {turn_id} held", reasons=gate_reasons)
                logger.info(
                    f"[Orchestrator] turn={turn_id} held for approval — {gate_reasons}"
                )
                final_response = settings.approval.withheld_message

        # Update conversation history
        self._session_state["conversation_history"].append(
            {"role": "assistant", "content": final_response}
        )
        self._persist_customer_memory()
        self._maybe_summarise()
        total_ms = (time.perf_counter() - turn_start) * 1000
        agents_invoked = [r.agent_name for r in agent_results]

        self.audit_log.record_turn_end(
            turn_id=turn_id,
            response_length=len(final_response),
            agents_invoked=agents_invoked,
            total_duration_ms=total_ms,
            hard_blocks=sum(1 for v in violations if v.get("severity") == "hard_block"),
            conflicts_resolved=len(conflicts),
        )

        logger.info(
            f"[Orchestrator] turn={turn_id} complete "
            f"agents={agents_invoked} "
            f"conflicts={len(conflicts)} violations={len(violations)} "
            f"recovered={recovered} skipped={[s['agent'] for s in skipped]} "
            f"collaborations={len(collaboration_events)} "
            f"duration={total_ms:.0f}ms"
        )

        trace.emit("TURN END", "",
                   duration_ms=round(total_ms), agents=len(agents_invoked),
                   conflicts=len(conflicts), violations=len(violations),
                   recovered=len(recovered), skipped=len(skipped),
                   collaborations=len(collaboration_events))

        return OrchestratorResult(
            session_id=self.session_id,
            turn_id=turn_id,
            routing_decision=routing,
            agents_invoked=agents_invoked,
            agent_results=agent_results,
            final_response=final_response,
            conflicts=conflicts,
            constraint_violations=violations,
            recovered_agents=recovered,
            skipped=skipped,
            collaboration_events=collaboration_events,
            total_duration_ms=total_ms,
            success=True,
            plan=turn_plan,
        )

     # Layer 1b - G6 planning (Day 4-5)
    def _plan_turn(
        self,
        user_message: str,
        context: dict,
        routing: RoutingDecision,
        intent: str,
        turn_id: str,
    ) -> Plan:
        """
        Decide this turn's agent sequence, and record how the decision was made.

        THE QUESTIONNAIRE AND ELICITATION PATHS DO NOT GO THROUGH THE PLANNER
            Mid-questionnaire, the sequence is not a choice: the customer is
            answering a question the system asked, and the only correct
            response is ConversationalAgent continuing the form. Asking a
            model to plan that turn spends a call to arrive at the one answer
            that was never in doubt, and gives it the opportunity to route
            away from the form and strand it (the same reasoning that already
            skips intent classification here).

        THE STATIC TABLE IS KEYED BY ROUTING DECISION, NOT RAW INTENT
            `intent` is a Banking77 bucket; `routing` is the six-way decision
            the bucket maps to, and two intents can share one route
            (product_suggestion and investment_advice both mean INVESTMENT).
            Keying the fallback on routing.value guarantees the fallback is
            byte-identical to what _get_agent_sequence would have returned,
            which is the premise of the planner-vs-static comparison: the two
            arms must differ only in who chose.
        """
        conv_agent = self._agents["ConversationalAgent"]
        if getattr(conv_agent, "questionnaire_active", False):
            return Plan(steps=("ConversationalAgent",), source="static_fallback",
                        intent=routing.value)

        if not settings.planner.enabled:
            return Plan(steps=tuple(self._get_agent_sequence(routing)),
                        source="static", intent=routing.value)

        with trace.span("[L1b]", "planning"):
            plan = self._planner.plan(user_message, context, routing.value)

        trace.emit(
            "PLAN",
            f"{plan.source}: {' → '.join(plan.steps) or '(empty)'}",
            accepted=plan.accepted,
            rejections=len(plan.rejections),
            tokens=plan.llm_tokens,
        )
        self.audit_log.record_plan(turn_id=turn_id, plan=plan.as_dict())

        if not plan.accepted and plan.proposed:
            logger.info(
                f"[Orchestrator] turn={turn_id} planner proposed "
                f"{list(plan.proposed)} — rejected "
                f"({', '.join(plan.rejection_codes)}) — using static table"
            )
        return plan

    @property
    def planner_stats(self) -> dict:
        """Session-level planner counters — read by the evaluation harness."""
        return self._planner.stats.as_dict()

    # Section 2 — questionnaire loop closure
    # ----------------------------------------------------------------------
    def _advance_questionnaire(
        self,
        agent_results: list[AgentResult],
        context: dict,
        turn_id: str,
        agent_sequence: list[str] | None = None,
        skipped: list[dict] | None = None,
    ) -> str | None:
        """
        Open, advance, or close whichever deterministic elicitation loop is
        relevant this turn — budget's (agents/budget_questionnaire.py) or
        risk-profiling's (agents/risk_questionnaire.py). Both run through
        ConversationalAgent's one questionnaire engine
        (start_questionnaire/_questionnaire_turn), distinguished by "kind".

        Returns a deterministic response string when THIS turn's reply is a
        questionnaire question (or its wrap-up), or None to let the normal
        synthesis path produce the reply.

        WHY THE ORCHESTRATOR IS INVOLVED AT ALL, GIVEN ConversationalAgent
        OWNS THE LOOP
            ConversationalAgent decides what to ask and reads the answers.
            But only the Orchestrator can see BudgetAgent say
            "insufficient_history" or RiskProfilingAgent say "incomplete"
            (those are the triggers to start asking), only it can put the
            finished answers back into the context the specialist reads,
            and only it can re-run that specialist (and, for risk,
            whatever was queued behind it — InvestmentAgent, Explainability)
            in the same turn once they arrive. Those are routing
            responsibilities, not conversational ones.

        NO LLM CALL ANYWHERE IN HERE
            Same reasoning as _elicitation_response(): the questions are a
            fixed, reviewed set, and asking a model to rephrase them adds a
            hallucination surface (inventing a question, implying advice) to
            a message whose only job is to be exactly the question that was
            approved.

        RQ COMPARABILITY NOTE (risk elicitation is new; re-run before citing)
            This does NOT touch _get_agent_sequence / _prune_unsatisfiable /
            STATIC_SEQUENCES for RISK_PROFILING or INVESTMENT — the turn
            where RiskProfilingAgent first reports "incomplete" still
            invokes exactly the agents it always did (_prune_unsatisfiable
            stays scoped to FULL_ADVISORY only, per its own docstring).
            Only the response TEXT for that turn changes, and new turns get
            added after it. Those new turns have no committed RQ1/RQ2/RQ4
            baseline yet — re-run the eval harnesses deliberately before
            treating multi-turn risk elicitation as covered by them.
        """
        conv_agent = self._agents["ConversationalAgent"]
        state = self._session_state

        # --- Case 1: this turn WAS a questionnaire turn (either kind) -----
        conv_result = next(
            (r for r in agent_results if r.agent_name == "ConversationalAgent"), None
        )
        if conv_result is not None and conv_result.payload.get("mode") == "questionnaire":
            block = conv_result.payload["questionnaire"]

            if block["kind"] == "risk":
                return self._advance_risk_questionnaire(
                    block, context, turn_id, conv_result, agent_results,
                )

            # kind == "budget" — unchanged.
            state["questionnaire_answers"] = dict(block["answers"])
            state["customer_wants_to_stop"] = bool(block["customer_wants_to_stop"])
            self.audit_log.record_agent_call(
                turn_id=turn_id, agent_name="ConversationalAgent",
                success=True, duration_ms=conv_result.duration_ms,
                tokens_used=conv_result.tokens_used,
                step_id=f"questionnaire:{block['reason']}",
            )
            trace.emit("Q&A", f"questionnaire {block['reason']}",
                       answered=len(block["answers"]),
                       asked=block["questions_asked_this_sitting"],
                       finished=block["finished"])

            if not block["finished"]:
                return conv_result.payload["response"]

            state["questionnaire_completed"] = True
            if not block["sufficient"]:
                # Stopped early, below the minimum viable budget. Say so
                # plainly rather than producing an analysis of three
                # numbers and calling it a budget.
                return conv_result.payload["response"]

            # Enough to build a first budget — run BudgetAgent now, in this
            # same turn, so the customer sees the payoff for answering rather
            # than having to ask again.
            budget_context = {
                **context,
                "questionnaire_answers": state["questionnaire_answers"],
                "customer_wants_to_stop": state["customer_wants_to_stop"],
                "monthly_income": (
                    context.get("monthly_income")
                    or state["questionnaire_answers"].get("income")
                ),
            }
            budget_result, _ = self._execute_agent("BudgetAgent", budget_context)
            agent_results.append(budget_result)
            return None      # let synthesis narrate the budget it just built

        # --- Case 2a: RiskProfilingAgent reported incomplete, OR (Day 6)
        # was skipped before it even ran ----------------------------------
        # Checked before budget's trigger: risk precedes investment/budget
        # in every routing table this system has, so if both somehow fired
        # the same turn, resolving risk first is the correct order anyway.
        #
        # THE "OR SKIPPED" HALF IS NEW AS OF DAY 6
        #     Before _execute_plan() re-checked `requires` at run time,
        #     RiskProfilingAgent always actually ran and reported its own
        #     status="incomplete" — there was nothing else this branch
        #     needed to know about. Day 6 means a customer with a
        #     completely empty user_features (the true brand-new case) now
        #     gets RiskProfilingAgent SKIPPED rather than invoked, because
        #     its `requires={"user_features"}` isn't met at all. Without
        #     this half, that customer would fall through both Case 2a and
        #     Case 2b with nothing catching them. Treating a requires-skip
        #     of RiskProfilingAgent the same as an incomplete status means
        #     both "some fields known, a few missing" (agent ran,
        #     incomplete) and "nothing known at all" (agent skipped) reach
        #     the same elicitation loop — just with an empty seed instead
        #     of a partial one.
        risk_result = next(
            (r for r in agent_results if r.agent_name == "RiskProfilingAgent"), None
        )
        risk_skip = next(
            (s for s in (skipped or []) if s["agent"] == "RiskProfilingAgent"), None
        )
        if (
            (risk_result is not None and risk_result.payload.get("status") == "incomplete")
            or risk_skip is not None
        ) and (
            not state["risk_elicitation_completed"]
            and not getattr(conv_agent, "questionnaire_active", False)
        ):
            known = context.get("user_features") or {}
            seed = {}
            for q in risk_questionnaire.QUESTIONNAIRE_SCHEMA:
                feature_key = "income" if q.slot_name == "annual_income" else q.slot_name
                if known.get(feature_key) is not None:
                    seed[q.slot_name] = known[feature_key]

            first_question = conv_agent.start_questionnaire(seed_slots=seed, kind="risk")
            if first_question is None:
                return None
            # So Case 1 can resume exactly what this turn was trying to do
            # (RISK_PROFILING alone, or INVESTMENT's Risk->Investment->
            # Explain) once every field is in, not just RiskProfilingAgent
            # on its own.
            state["risk_elicitation_pending_sequence"] = list(agent_sequence or [
                "RiskProfilingAgent", "ExplainabilityAgent",
            ])
            trigger = "incomplete" if risk_result is not None else "skipped_no_features"
            trace.emit("Q&A", "risk elicitation started",
                       trigger=trigger, first=first_question.slot_name)
            logger.info(
                f"[Orchestrator] RiskProfilingAgent {trigger} — starting "
                f"risk elicitation at {first_question.slot_name!r}"
            )
            preamble = (
                risk_result.payload.get("message") if risk_result is not None else None
            ) or (
                "I don't have any information about your finances on file yet, "
                "so I can't assess your risk profile."
            )
            return (
                f"{preamble}\n\n"
                f"I can go through a few quick questions to work it out — "
                f"you can stop at any point.\n\n"
                f"{first_question.question_text}"
            )

        # --- Case 2b: BudgetAgent just reported insufficient data ---------
        budget_result = next(
            (r for r in agent_results if r.agent_name == "BudgetAgent"), None
        )
        if (
            budget_result is not None
            and budget_result.payload.get("status") == "insufficient_history"
            and not state["questionnaire_completed"]
            and not getattr(conv_agent, "questionnaire_active", False)
        ):
            first_question = conv_agent.start_questionnaire(
                seed_slots={
                    k: v for k, v in {
                        "income": context.get("monthly_income")
                                  or (context.get("user_features") or {}).get("income"),
                        **state["questionnaire_answers"],
                    }.items() if v is not None
                },
                kind="budget",
            )
            if first_question is None:
                return None
            trace.emit("Q&A", "questionnaire started",
                       trigger="insufficient_history", first=first_question.slot_name)
            logger.info(
                f"[Orchestrator] Insufficient transaction history — starting "
                f"Section 2 questionnaire at {first_question.slot_name!r}"
            )
            preamble = budget_result.payload.get("message") or (
                "There isn't enough transaction history yet to build a "
                "reliable budget."
            )
            return (
                f"{preamble}\n\n"
                f"I can still put together a first estimate if you answer a "
                f"few short questions — you can stop at any point.\n\n"
                f"{first_question.question_text}"
            )

        return None

    def _advance_risk_questionnaire(
        self,
        block: dict,
        context: dict,
        turn_id: str,
        conv_result: AgentResult,
        agent_results: list[AgentResult],
    ) -> str | None:
        """
        Case 1 for kind == "risk": interpret this turn's answer, either ask
        the next question or — once every required field is in — persist
        the answers and resume whatever agent sequence risk elicitation
        interrupted (RiskProfilingAgent alone, or Risk -> Investment ->
        Explainability), in this same turn, the same way budget's Case 1
        re-runs BudgetAgent. Split out from _advance_questionnaire() rather
        than inlined only because that resume step needs several lines
        `agent_results.extend(...)` mutates the SAME list process_turn()
        already uses for synthesis and agents_invoked — no side channel.
        """
        state = self._session_state
        state["risk_elicitation_answers"] = dict(block["answers"])
        state["risk_customer_wants_to_stop"] = bool(block["customer_wants_to_stop"])
        self.audit_log.record_agent_call(
            turn_id=turn_id, agent_name="ConversationalAgent",
            success=True, duration_ms=conv_result.duration_ms,
            tokens_used=conv_result.tokens_used,
            step_id=f"risk_elicitation:{block['reason']}",
        )
        trace.emit("Q&A", f"risk elicitation {block['reason']}",
                   answered=len(block["answers"]),
                   asked=block["questions_asked_this_sitting"],
                   finished=block["finished"])

        if not block["finished"]:
            return conv_result.payload["response"]

        state["risk_elicitation_completed"] = True
        if not block["sufficient"]:
            # Customer stopped before every required field was given.
            # RiskProfilingAgent will report "incomplete" again if invoked
            # — correctly; it still can't classify without them, and it
            # will not guess.
            return conv_result.payload["response"]

        new_features = risk_questionnaire.build_user_features_from_slots(
            state["risk_elicitation_answers"]
        )
        # Persists to CustomerStore AND mutates context["user_features"] in
        # place (same dict object _build_context() spread by reference —
        # see update_customer_features()), so _run_agent_sequence below
        # sees the completed profile with no extra plumbing.
        self.update_customer_features(new_features)

        resume_sequence = state.get("risk_elicitation_pending_sequence") or [
            "RiskProfilingAgent", "ExplainabilityAgent",
        ]
        resumed_results, _ = self._run_agent_sequence(resume_sequence, context)
        agent_results.extend(resumed_results)
        return None  # let synthesis narrate what the resumed agents produced

    def _agent_is_satisfiable(self, agent_name: str, context: dict) -> tuple[bool, str]:
        """Return (can_run, human-readable reason it cannot)."""
        if agent_name == "RiskProfilingAgent":
            features = context.get("user_features") or {}
            missing = [f for f in settings.risk.required_features if f not in features]
            if missing:
                return False, "your age, income, employment, dependants, debts, and how you feel about risk"
            return True, ""

        if agent_name == "InvestmentAgent":
            # Satisfiable if a risk class already exists, or if RiskProfiling
            # will produce one earlier in this same turn.
            if context.get("risk_class") or context.get("risk_profile"):
                return True, ""
            ok, _ = self._agent_is_satisfiable("RiskProfilingAgent", context)
            if not ok:
                return False, "a completed risk profile"
            return True, ""

        if agent_name == "BudgetAgent":
            has_income = bool(
                context.get("monthly_income")
                or (context.get("user_features") or {}).get("income")
            )
            has_expense_data = bool(
                context.get("monthly_expenses") or "transactions" in context
            )
            if not has_expense_data:
                return False, "your monthly spending by category (rent, food, transport, ...)"
            if not has_income:
                return False, "your monthly income"
            return True, ""

        return True, ""      # ConversationalAgent / ExplainabilityAgent

    def _prune_unsatisfiable(
        self, sequence: list[str], context: dict
    ) -> tuple[list[str], list[str]]:
        """
        Drop agents whose inputs are absent, and report what was missing. (R7)

        WHY THIS EXISTS AT ALL
            FULL_ADVISORY runs Risk → Investment → Budget → Explainability.
            The person most likely to ask for it ("I know nothing about
            finance, tell me what to do") is by definition a new user with no
            stored features and no spending data. Running the sequence anyway
            costs four LLM calls, produces three status="incomplete" results,
            and hands the synthesis step nothing to synthesise. The user gets
            a vague non-answer, and — worse for the research — the turn is
            recorded as four agents invoked, which inflates the coordination
            metrics with agents that never had a chance to contribute.
            Pruning first means the route does what it CAN do and says plainly
            what it needs for the rest.

        SCOPED TO FULL_ADVISORY ON PURPOSE
            The same gate would help INVESTMENT and RISK_PROFILING, but those
            routes' agents_invoked lists are already baked into RQ2 and RQ4.
            Changing them here would silently make new results
            non-comparable with the committed ones. Extend it deliberately,
            with a re-run, not as a side effect of this fix.
        """
        runnable, unmet = [], []
        for agent_name in sequence:
            ok, reason = self._agent_is_satisfiable(agent_name, context)
            if ok:
                runnable.append(agent_name)
            elif reason and reason not in unmet:
                unmet.append(reason)

        # ExplainabilityAgent explains other agents' output. On its own it has
        # nothing to explain, so it is not a "specialist survived" signal.
        if not [a for a in runnable if a != "ExplainabilityAgent"]:
            runnable = []

        return runnable, unmet

    @staticmethod
    def _elicitation_response(unmet: list[str]) -> str:
        """
        Deterministic. No LLM call. (R7)

        The whole point of this branch is that nothing is known about the user
        yet, so there is nothing for a model to reason over — and asking one to
        phrase a fixed list of required fields would add a hallucination
        surface (inventing a field, or implying advice) to a message whose only
        job is to be accurate about what the system needs.
        """
        if not unmet:
            return (
                "I can give you a full review of your finances. To start, tell "
                "me a little about your situation."
            )
        bullets = "\n".join(f"  • {item}" for item in unmet)
        return (
            "Happy to give you a full picture of your finances — that covers "
            "your risk profile, what to do with savings, and where your money "
            "goes each month.\n\n"
            "To do that properly rather than guess, I need:\n"
            f"{bullets}\n\n"
            "You can give me whatever you have and we'll start there. This is "
            "not regulated financial advice, and you should consult a qualified "
            "financial advisor before acting on it."
        )

    # def _get_agent_sequence(self, routing: RoutingDecision) -> list[str]:
    #     """
    #     Map routing decision to ordered agent execution sequence.
    #     ExplainabilityAgent always runs last (X1 — in-pipeline).
    #     """
    #     sequences = {
    #         RoutingDecision.CONVERSATIONAL_ONLY: [
    #             "ConversationalAgent",
    #         ],
    #         RoutingDecision.RISK_PROFILING: [
    #             "RiskProfilingAgent",
    #             "ExplainabilityAgent",
    #         ],
    #         RoutingDecision.INVESTMENT: [
    #             "RiskProfilingAgent",
    #             "InvestmentAgent",
    #             "ExplainabilityAgent",
    #         ],
    #         RoutingDecision.BUDGET: [
    #             "BudgetAgent",
    #             "ExplainabilityAgent",
    #         ],
    #         RoutingDecision.FULL_ADVISORY: [
    #             "RiskProfilingAgent",
    #             "InvestmentAgent",
    #             "BudgetAgent",
    #             "ExplainabilityAgent",
    #         ],
    #         RoutingDecision.EXPLANATION_REQUEST: [
    #             "ExplainabilityAgent",
    #         ],
    #     }
    #     return sequences.get(routing, ["ConversationalAgent"])

    @staticmethod
    def _get_agent_sequence(routing: RoutingDecision) -> list[str]:
        """
        Map routing decision to ordered agent execution sequence — the static
        table. ExplainabilityAgent always runs last (X1 — in-pipeline).

        THE TABLE ITSELF MOVED TO agents/payloads.STATIC_SEQUENCES (Day 4-5)
            It now has two consumers: this method, and Planner's fallback.
            Two copies would drift, and the moment they did, the
            planner-vs-static comparison would be measuring a difference
            between two tables rather than between LLM planning and
            hand-written routing. One declaration, read from both places.
        """
        return list(STATIC_SEQUENCES.get(routing.value, ["ConversationalAgent"]))