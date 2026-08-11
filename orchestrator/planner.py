"""

THE DESIGN DECISION THIS FILE EXISTS TO MAKE
    The planner PROPOSES; a pure-Python validator DECIDES.

    An LLM emitting an arbitrary execution plan is neither auditable nor
    testable, which is precisely the standing criticism of agentic systems in
    regulated settings. An unvalidated plan can invoke InvestmentAgent with no
    risk class (R2 again), name an agent that does not exist, or repeat a step
    until the token budget runs out. None of those fail loudly: the agent
    returns status="incomplete", the synthesis step papers over the gap, and
    the turn looks fine in the results file.

    So acceptance is a pure function of the capability graph in
    agents/payloads.CAPABILITIES, every rejection carries a machine-readable
    reason, and rejection falls back to the hand-written table in
    agents/payloads.STATIC_SEQUENCES.

WHY KEEPING THE STATIC TABLE IS THE POINT, NOT A HEDGE
    Deleting it would remove the only baseline this system can be measured
    against. Keeping it makes the static table simultaneously the safety net
    and the control arm, which buys an evaluation that is otherwise
    unavailable:

        planner vs static table, same scenarios
          · plan validity rate      — how often the LLM proposes something
                                      the validator accepts
          · agreement rate          — how often an accepted plan equals what
                                      the table would have chosen
          · rejection reason mix    — WHY it fails when it fails
          · CSS / TUE for each arm

    A high rejection rate is a finding about LLM planning in regulated
    routing, not a failure of the build.

TWO LEVELS OF VALIDITY, DELIBERATELY SEPARATED
    structural  — agents exist, no duplicates, within max_plan_steps,
                  ExplainabilityAgent last, non-empty. Depends on nothing but
                  the plan itself.
    contextual  — every step's `requires` is satisfied by the context plus the
                  `produces` of earlier steps.

    The LLM's proposal must pass both. The static fallback is only required to
    pass STRUCTURAL validity, because the table is written for an intent, not
    for a particular customer's context — a budget plan for a user who has not
    given expenses yet is a correct plan that cannot run today. 

RUNNING
    python -m pytest tests/unit/test_planner.py -v
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from agents.payloads import CAPABILITIES, STATIC_SEQUENCES, AgentCapability
from config.prompts import PLANNER_SYSTEM
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_SPECIALIST_AGENTS = frozenset({"BudgetAgent", "RiskProfilingAgent", "InvestmentAgent"})


class RejectionReason(str, Enum):
    """
    Machine-readable rejection codes.

    Free-text reasons are for humans reading a log; these are for the
    rejection-reason histogram in the planner-vs-static comparison. A metric
    built on substring-matching log lines stops working the first time
    someone rewords a message.
    """
    MALFORMED_JSON        = "malformed_json"
    LLM_ERROR             = "llm_error"
    EMPTY_PLAN            = "empty_plan"
    UNKNOWN_AGENT         = "unknown_agent"
    DUPLICATE_STEP        = "duplicate_step"
    TOO_LONG              = "too_long"
    EXPLAINABILITY_NOT_LAST = "explainability_not_last"
    UNSATISFIED_REQUIRES  = "unsatisfied_requires"
    NOT_A_LIST            = "not_a_list"


@dataclass(frozen=True)
class Rejection:
    """One reason a plan was not accepted, with enough detail to act on."""
    code: RejectionReason
    detail: str

    def __str__(self) -> str:            # what lands in the log line
        return f"{self.code.value}: {self.detail}"

    def as_dict(self) -> dict:           # what lands in the audit log / results
        return {"code": self.code.value, "detail": self.detail}


@dataclass(frozen=True)
class Plan:
    """
    The outcome of one planning attempt — accepted or not.

    `proposed` is kept even when the plan was rejected. The rejected proposal
    IS the measurement: "the planner wanted Investment → Risk → Explainability
    and was refused because risk_class was not yet produced"
    """
    steps: tuple[str, ...]
    source: str                                   # "planner" | "static_fallback"
    intent: str = ""
    proposed: tuple[str, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    llm_tokens: int = 0
    duration_ms: float = 0.0

    @property
    def accepted(self) -> bool:
        """True when the LLM's own proposal survived validation."""
        return self.source == "planner"

    @property
    def rejection_codes(self) -> tuple[str, ...]:
        return tuple(r.code.value for r in self.rejections)

    @property
    def specialists_present(self) -> tuple[str, ...]:
        """Which of BudgetAgent/RiskProfilingAgent/InvestmentAgent this plan runs."""
        return tuple(sorted(_SPECIALIST_AGENTS.intersection(self.steps)))

    @property
    def explainability_skipped(self) -> bool:
        """
        True when this plan runs a specialist agent (BudgetAgent/
        RiskProfilingAgent/InvestmentAgent — financial guidance the user
        may act on) without ExplainabilityAgent to calibrate, ground, or
        disclaim it.

        NOT a rejection — G6's whole point is that the planner is allowed
        to choose exactly this, to save a model call when it judges
        explanation unnecessary. This property exists so that choice is
        visible (trace, audit log, planner-vs-static metrics) instead of
        a silent gap the response text gives no sign of. See
        orchestrator.orchestrator's "SKIP-EXPLAIN" trace emission and
        AuditLog.record_plan, both driven by this.
        """
        specialists_present = self.specialists_present
        return bool(specialists_present) and "ExplainabilityAgent" not in self.steps

    def as_dict(self) -> dict:
        return {
            "steps": list(self.steps),
            "source": self.source,
            "intent": self.intent,
            "proposed": list(self.proposed),
            "accepted": self.accepted,
            "rejections": [r.as_dict() for r in self.rejections],
            "llm_tokens": self.llm_tokens,
            "duration_ms": round(self.duration_ms, 2),
            "explainability_skipped": self.explainability_skipped,
        }


# ── context inspection ─────────────────────────────────────────────────────

def available_context_keys(context: dict) -> set[str]:
    """
    Which capability-graph keys the context can currently satisfy.

    PRESENT-BUT-EMPTY IS ABSENT
        The orchestrator seeds `user_features: {}` and `risk_profile: None`
        into every session's state before anything has filled them. Testing
        `key in context` would therefore report every precondition satisfied
        on turn one, and the validator would accept exactly the plans it
        exists to reject. Truthiness is the right test here: an empty feature
        dict is not a set of features.

    TWO ALIASES ARE RESOLVED HERE, NOT IN THE GRAPH
        risk_class      may arrive as a bare key on the context, or inside the
                        cached `risk_profile` payload from an earlier turn.
        monthly_income  may arrive directly, or as user_features["income"]
                        loaded from the CustomerStore.
        Both mirror what Orchestrator._agent_is_satisfiable already accepts.
        Resolving them here keeps CAPABILITIES a clean declaration instead of
        making every consumer re-implement the same two special cases.
    """
    keys = {k for k, v in context.items() if v not in (None, "", [], {}, ())}

    risk_profile = context.get("risk_profile") or {}
    if isinstance(risk_profile, dict) and risk_profile.get("risk_class"):
        keys.add("risk_class")

    features = context.get("user_features") or {}
    if isinstance(features, dict) and features:
        keys.add("user_features")
        if features.get("income"):
            keys.add("monthly_income")

    if "transactions" in context or context.get("monthly_expenses"):
        keys.add("monthly_expenses")

    return keys


# ── the validator ──────────────────────────────────────────────────────────

class PlanValidator:
    """
    Pure Python. No LLM, no I/O, no clock. Same input, same verdict, always —
    which is what makes a plan rejection something you can put in a table.
    """

    def __init__(
        self,
        capabilities: dict[str, AgentCapability] | None = None,
        max_plan_steps: int | None = None,
        require_explainability_last: bool | None = None,
        allow_empty_plan: bool | None = None,
    ):
        cfg = settings.planner
        self.capabilities = capabilities if capabilities is not None else CAPABILITIES
        self.max_plan_steps = max_plan_steps or cfg.max_plan_steps
        self.require_explainability_last = (
            cfg.require_explainability_last
            if require_explainability_last is None
            else require_explainability_last
        )
        self.allow_empty_plan = (
            cfg.allow_empty_plan if allow_empty_plan is None else allow_empty_plan
        )

    # -- structural: depends only on the plan ------------------------------
    def validate_structure(self, steps: list[str]) -> list[Rejection]:
        problems: list[Rejection] = []

        if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
            return [Rejection(RejectionReason.NOT_A_LIST,
                              f"expected a list of agent names, got {type(steps).__name__}")]

        if not steps and not self.allow_empty_plan:
            problems.append(Rejection(RejectionReason.EMPTY_PLAN,
                                      "plan has no steps"))

        if len(steps) > self.max_plan_steps:
            problems.append(Rejection(
                RejectionReason.TOO_LONG,
                f"{len(steps)} steps exceeds max_plan_steps={self.max_plan_steps}",
            ))

        unknown = [s for s in steps if s not in self.capabilities]
        if unknown:
            problems.append(Rejection(
                RejectionReason.UNKNOWN_AGENT,
                f"not in CAPABILITIES: {sorted(set(unknown))}",
            ))

        duplicates = sorted({s for s in steps if steps.count(s) > 1})
        if duplicates:
            problems.append(Rejection(RejectionReason.DUPLICATE_STEP,
                                      f"repeated: {duplicates}"))

        if (
            self.require_explainability_last
            and "ExplainabilityAgent" in steps
            and steps[-1] != "ExplainabilityAgent"
        ):
            problems.append(Rejection(
                RejectionReason.EXPLAINABILITY_NOT_LAST,
                f"ExplainabilityAgent at position {steps.index('ExplainabilityAgent')} "
                f"of {len(steps)}; it explains what earlier agents produced (X1)",
            ))

        return problems

    # -- contextual: depends on the plan AND this turn's context -----------
    def validate(self, steps: list[str], available: set[str]) -> list[Rejection]:
        """
        Structural checks, then a simulated walk of the plan.

        THE WALK IS WHY THIS IS NOT JUST A SET INTERSECTION
            A step's preconditions may be met by the context OR by an earlier
            step in the same plan. ["RiskProfilingAgent", "InvestmentAgent"]
            is valid for a user with no risk class, because step 1 produces
            it; ["InvestmentAgent", "RiskProfilingAgent"] is not, and the
            difference is order. Checking the union of everything the plan
            produces would accept both.
        """
        problems = self.validate_structure(steps)
        if any(p.code in (RejectionReason.NOT_A_LIST,
                          RejectionReason.UNKNOWN_AGENT) for p in problems):
            # Cannot walk a plan naming agents that do not exist.
            return problems

        satisfied = set(available)
        for position, name in enumerate(steps):
            cap = self.capabilities[name]
            missing = cap.requires - satisfied
            if missing:
                problems.append(Rejection(
                    RejectionReason.UNSATISFIED_REQUIRES,
                    f"step {position + 1} {name} requires {sorted(missing)}, "
                    f"not present in context and not produced by any earlier step",
                ))
            satisfied |= cap.produces

        return problems


# the planner 



def _format_capabilities(capabilities: dict[str, AgentCapability]) -> str:
    lines = []
    for cap in capabilities.values():
        lines.append(
            f"- {cap.name} [{cap.cost_hint}]\n"
            f"    does:      {cap.description}\n"
            f"    requires:  {sorted(cap.requires) or 'nothing'}\n"
            f"    produces:  {sorted(cap.produces)}"
        )
    return "\n".join(lines)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_plan_json(raw: str) -> list[str] | None:
    """
    Pull the plan list out of a model response, or return None.

    TOLERANT OF WRAPPING, STRICT ABOUT CONTENT
        Models wrap JSON in ```json fences or a sentence of preamble often
        enough that refusing those would inflate the malformed_json rate with
        cases the system could trivially have handled — and that rate is a
        reported metric, so padding it with formatting noise would misstate
        the finding. What is NOT tolerated is a plan that is not a list of
        strings: that is a shape error, and repairing it would mean guessing
        at intent.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidates = [text]
    match = _JSON_OBJECT.search(text)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            plan = parsed.get("plan")
        elif isinstance(parsed, list):
            plan = parsed
        else:
            continue
        if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
            return [s.strip() for s in plan]
        return None            # right JSON, wrong shape — a real rejection
    return None


@dataclass
class PlannerStats:
    """Per-session counters, read by the evaluation harness."""
    attempts: int = 0
    accepted: int = 0
    fallbacks: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)

    def record(self, plan: Plan) -> None:
        self.attempts += 1
        if plan.accepted:
            self.accepted += 1
        else:
            self.fallbacks += 1
        for code in plan.rejection_codes:
            self.rejection_counts[code] = self.rejection_counts.get(code, 0) + 1

    @property
    def validity_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "fallbacks": self.fallbacks,
            "validity_rate": round(self.validity_rate, 4),
            "rejection_counts": dict(self.rejection_counts),
        }


class Planner:
    """
    Layer 1 goal decomposition, LLM-driven, deterministically validated.

    One instance per session, sharing the Orchestrator's LLM client so that
    planning tokens land in the same accounting as everything else.
    """

    def __init__(
        self,
        llm_client,
        capabilities: dict[str, AgentCapability] | None = None,
        static_sequences: dict[str, list[str]] | None = None,
        validator: PlanValidator | None = None,
    ):
        self.llm = llm_client
        self.capabilities = capabilities if capabilities is not None else CAPABILITIES
        self.static_sequences = (
            static_sequences if static_sequences is not None else STATIC_SEQUENCES
        )
        self.validator = validator or PlanValidator(capabilities=self.capabilities)
        self.stats = PlannerStats()

    # -- public API --------------------------------------------------------
    def plan(self, user_message: str, context: dict, intent: str) -> Plan:
        """
        Propose, validate, and either accept or fall back. Never raises.

        NEVER RAISES, ON PURPOSE
            A planner that throws takes the whole turn down. Every failure
            mode here — no client, timeout, malformed output, invalid plan —
            has the same answer: use the static table and record why. The
            system's behaviour degrades to exactly Phase 7, which is a
            known-good state, and the reason is in the Plan for the metrics
            to pick up.
        """
        started = time.perf_counter()

        if not settings.planner.enabled:
            return self._fallback(intent, (), (), started, tokens=0)

        proposed, tokens, llm_rejection = self._ask_llm(user_message, context, intent)
        if llm_rejection is not None:
            plan = self._fallback(intent, (), (llm_rejection,), started, tokens)
            self.stats.record(plan)
            return plan

        problems = self.validator.validate(
            list(proposed), available_context_keys(context)
        )
        if problems:
            logger.warning(
                f"[Planner] plan rejected {list(proposed)} — "
                + "; ".join(str(p) for p in problems)
                + " — falling back to static table"
            )
            plan = self._fallback(
                intent, tuple(proposed), tuple(problems), started, tokens
            )
        else:
            plan = Plan(
                steps=tuple(proposed),
                source="planner",
                intent=intent,
                proposed=tuple(proposed),
                rejections=(),
                llm_tokens=tokens,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            logger.info(f"[Planner] plan accepted: {' → '.join(plan.steps)}")

        self.stats.record(plan)
        return plan

    def static_plan(self, intent: str) -> Plan:
        """
        The control arm: what the hand-written table would do, no LLM call.
        Used by scripts/eval_planner_vs_static.py and by callers that have
        turned the planner off.
        """
        return self._fallback(intent, (), (), time.perf_counter(), tokens=0)

    # -- internals ---------------------------------------------------------
    def _ask_llm(
        self, user_message: str, context: dict, intent: str
    ) -> tuple[tuple[str, ...], int, Rejection | None]:
        prompt = self._build_prompt(user_message, context, intent)
        try:
            response = self.llm.chat(
                system=PLANNER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.planner.temperature,
            )
        except Exception as exc:
            logger.error(f"[Planner] LLM call failed: {exc}")
            return (), 0, Rejection(RejectionReason.LLM_ERROR, str(exc)[:200])

        tokens = getattr(response, "tokens_used", 0) or 0
        steps = extract_plan_json(getattr(response, "content", "") or "")
        if steps is None:
            preview = (getattr(response, "content", "") or "")[:120].replace("\n", " ")
            logger.warning(f"[Planner] unparsable plan response: {preview!r}")
            return (), tokens, Rejection(RejectionReason.MALFORMED_JSON,
                                         f"no JSON plan list in response: {preview!r}")
        return tuple(steps), tokens, None

    def _build_prompt(self, user_message: str, context: dict, intent: str) -> str:
        available = sorted(available_context_keys(context) & self._graph_keys())
        return (
            f"User message: {user_message!r}\n"
            f"Classified intent: {intent}\n"
            f"Context keys already available: {available or 'none'}\n\n"
            f"Available agents:\n{_format_capabilities(self.capabilities)}\n\n"
            f"Return the plan as JSON."
        )

    def _graph_keys(self) -> set[str]:
        """Only show the model keys that appear in the graph — the session
        state carries a dozen bookkeeping fields (turn_count, session_id,
        conversation_history) that are noise to a routing decision and would
        crowd the prompt."""
        keys: set[str] = set()
        for cap in self.capabilities.values():
            keys |= cap.requires | cap.produces
        return keys

    def _fallback(
        self,
        intent: str,
        proposed: tuple[str, ...],
        rejections: tuple[Rejection, ...],
        started: float,
        tokens: int,
    ) -> Plan:
        steps = self.static_sequences.get(
            intent, self.static_sequences[settings.planner.fallback_intent]
        )
        return Plan(
            steps=tuple(steps),
            source="static_fallback",
            intent=intent,
            proposed=proposed,
            rejections=rejections,
            llm_tokens=tokens,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
