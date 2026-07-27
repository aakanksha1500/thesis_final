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
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.base_agent import AgentResult
from agents.budget_agent import BudgetAgent
from agents.conversational_agent import ConversationalAgent
from agents.investment_agent import InvestmentAgent
from agents.risk_profiling_agent import RiskProfilingAgent
from config.constraints import financial_constraints
from config.prompts import ORCHESTRATOR_SYSTEM
from config.settings import settings
from data.customer_store import CustomerStore
from data.psychometric_proxy import derive_loss_tolerance_proxy
from explainability.explainability_agent import ExplainabilityAgent
from orchestrator.audit_log import AuditLog
from orchestrator.conflict_resolver import ConflictResolver
from orchestrator.failure_handler import FailureHandler
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
    total_duration_ms: float = 0.0
    success: bool = True
    error: str | None = None

class Orchestrator:
    """
    Central cordinator implementing the HALO three-layer hierarchy.
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
        self.customer_store = customer_store or CustomerStore()

        # Per-session components
        self.audit_log = AuditLog(session_id=self.session_id)
        self._conflict_resolver = ConflictResolver()
        self._failure_handler = FailureHandler()

        # Agent registry — one instance per agent type, shared across turns
        self._agents: dict[str, Any] = {
            "ConversationalAgent": ConversationalAgent(llm_client),
            "RiskProfilingAgent":  RiskProfilingAgent(llm_client),
            "InvestmentAgent":     InvestmentAgent(llm_client),
            "BudgetAgent":         BudgetAgent(llm_client),
            "ExplainabilityAgent": ExplainabilityAgent(llm_client),
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
            "ground_truth_risk_class": None,
            "proxy_fields": [],
            "proxy_metadata": {},
            "customer_name": None,
        }

        logger.info(f"[Orchestrator] Session {self.session_id} initialised")

        if customer_context is not None:
            self._load_customer(customer_id, customer_context=customer_context)
        elif customer_id:
            self._load_customer(customer_id)

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
        for attempt in range(settings.orchestrator.max_agent_retries + 1):
            try:
                result = agent.run(context)
                self.audit_log.record_agent_call(
                    turn_id=context.get("_turn_id", "unknown"),
                    agent_name=agent_name,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    tokens_used=result.tokens_used,
                    step_id=result.step_id,
                    error=result.error,
                )
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

    def _run_agent_sequence(
        self,
        agent_names: list[str],
        context: dict,
    ) -> tuple[list[AgentResult], list[str]]:
        """
        Execute a sequence of agents, passing each result into the next
        agent's context. Returns (results, recovered_agent_names).
        """
        results: list[AgentResult] = []
        recovered: list[str] = []

        for agent_name in agent_names:
            # Inject prior results into context for downstream agents
            if agent_name == "InvestmentAgent":
                risk_result = next(
                    (r for r in results if r.agent_name == "RiskProfilingAgent"), None
                )
                if risk_result and risk_result.success:
                    context["risk_agent_payload"] = risk_result.payload
                    self._session_state["risk_profile"] = risk_result.payload

            if agent_name == "ExplainabilityAgent":
                risk_result = next(
                    (r for r in results if r.agent_name == "RiskProfilingAgent"), None
                )
                inv_result = next(
                    (r for r in results if r.agent_name == "InvestmentAgent"), None
                )
                context["risk_agent_payload"] = (
                    risk_result.payload if risk_result else
                    self._session_state.get("risk_profile") or {}
                )
                context["investment_agent_payload"] = (
                    inv_result.payload if inv_result else
                    self._session_state.get("prior_investment_output") or {}
                )

            result, was_recovered = self._execute_agent(agent_name, context)
            results.append(result)
            if was_recovered:
                recovered.append(agent_name)

            # Cache successful outputs for future turns
            if result.success:
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

        return results, recovered

    # Layer 3 - Execution monitoring
    def _check_constraints(
        self, response_text: str, agent_results: list[AgentResult]
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

        deliverable, raw_violations = financial_constraints.validate_response(
            response_text=response_text,
            risk_class=risk_class,
            claimed_return=claimed_return,
        )

        for v in raw_violations:
            vdict = {
                "rule_id": v.rule_id,
                "severity": v.severity,
                "description": v.description,
            }
            violations.append(vdict)
            self.audit_log.record_constraint_violation(
                turn_id="",  # filled in by caller
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

        results_summary = "\n\n".join(results_summary_parts)
        slots = self._session_state.get("user_features", {})

        synthesis_prompt = (
            f"User message: '{user_message}'\n"
            f"Routing: {routing_decision.value}\n"
            f"Known user context: {json.dumps(slots)}\n\n"
            f"Agent outputs:\n{results_summary}\n\n"
            f"Synthesise a coherent, concise response (max 150 words) "
            f"for the retail investor. Include the CBI disclaimer if any "
            f"investment content is present."
        )

        try:
            response = self.llm.chat(
                system=ORCHESTRATOR_SYSTEM,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
            )
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
        self._session_state["turn_count"] += 1
        self._session_state["conversation_history"].append(
            {"role": "user", "content": user_message}
        )

        self.audit_log.record_turn_start(turn_id, user_message)

        # -- Layer 1: Classify intent --
        routing, intent, confidence = self._classify_intent(user_message)
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

        agent_sequence = self._get_agent_sequence(routing)
        agent_results, recovered = self._run_agent_sequence(agent_sequence, context)

        # -- Layer 3a: Conflict resolution --
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

        # -- Layer 3b: Synthesis --
        raw_response = self._synthesise_response(
            user_message, agent_results, routing
        )

        # -- Layer 3c: Constraint validation --
        final_response, violations, was_blocked = self._check_constraints(
            raw_response, agent_results
        )

        # Update conversation history
        self._session_state["conversation_history"].append(
            {"role": "assistant", "content": final_response}
        )

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
            f"recovered={recovered} duration={total_ms:.0f}ms"
        )

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
            total_duration_ms=total_ms,
            success=True,
        )

    def _get_agent_sequence(self, routing: RoutingDecision) -> list[str]:
        """
        Map routing decision to ordered agent execution sequence.
        ExplainabilityAgent always runs last (X1 — in-pipeline).
        """
        sequences = {
            RoutingDecision.CONVERSATIONAL_ONLY: [
                "ConversationalAgent",
            ],
            RoutingDecision.RISK_PROFILING: [
                "RiskProfilingAgent",
                "ExplainabilityAgent",
            ],
            RoutingDecision.INVESTMENT: [
                "RiskProfilingAgent",
                "InvestmentAgent",
                "ExplainabilityAgent",
            ],
            RoutingDecision.BUDGET: [
                "BudgetAgent",
            ],
            RoutingDecision.FULL_ADVISORY: [
                "RiskProfilingAgent",
                "InvestmentAgent",
                "BudgetAgent",
                "ExplainabilityAgent",
            ],
            RoutingDecision.EXPLANATION_REQUEST: [
                "ExplainabilityAgent",
            ],
        }
        return sequences.get(routing, ["ConversationalAgent"])

