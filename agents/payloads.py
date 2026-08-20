"""
Declares what each agent needs and produces, plus the fallback routing table.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

class PayloadContractError(TypeError):
    """Raised only when STRICT_PAYLOADS=true. Never raised in normal running."""

PAYLOAD_CONTRACTS: dict[str, dict[str, Any]] = {
    "RiskProfilingAgent": {
        "status": str,
        "risk_class": str,
        "confidence": (int, float),
        "feature_importance": dict,
        "rationale": str,
    },
    "InvestmentAgent": {
        "status": str,
        "risk_class": str,
        "shortlist": list,
        "synthesis": str,
        "deliverable": bool,
    },
    "BudgetAgent": {
        "status": str,
        "disposable_income": (int, float),
        "savings_rate_pct": (int, float),
        "recommendations_text": str,
    },
    "ConversationalAgent": {
        "intent": str,
        "confidence": (int, float),
        "response": str,
        "escalation_needed": bool,
    },
    "ExplainabilityAgent": {
        "layers_applied": list,
        "calibration_note": str,
        "confidence": (int, float),
        "full_explanation": str,
    },
}

PARTIAL_STATUSES = {"incomplete", "blocked", "error", "skipped", "needs_input"}

def strict_mode() -> bool:
    return os.getenv("STRICT_PAYLOADS", "true").lower() != "false"

def validate_payload(
    agent_name: str,
    payload: dict[str, Any],
    success: bool = True,
) -> list[str]:
    """
    Return a list of human-readable problems. Empty list means the payload
    satisfies its contract. Never raises — the caller decides what to do.
    """
    contract = PAYLOAD_CONTRACTS.get(agent_name)
    if contract is None:
        return []                       # unknown agent: nothing to enforce
    if not success:
        return []                       # a failed run has no output contract
    if payload.get("status") in PARTIAL_STATUSES:
        return []                       # deliberate refusal, see above

    problems: list[str] = []
    for key, expected in contract.items():
        if key not in payload:
            problems.append(f"missing key {key!r}")
            continue
        value = payload[key]
        if value is None:
            problems.append(f"{key!r} is None")
            continue
        if not isinstance(value, expected):
            names = (expected.__name__ if isinstance(expected, type)
                     else "/".join(t.__name__ for t in expected))
            problems.append(
                f"{key!r} is {type(value).__name__}, expected {names}"
            )
    return problems


def enforce(
    agent_name: str,
    payload: dict[str, Any],
    success: bool = True,
) -> None:
    """
    Validate and react. Called from BaseAgent._make_result — the one place
    every agent's output passes through.
    """
    problems = validate_payload(agent_name, payload, success)
    if not problems:
        return
    message = f"[{agent_name}] payload contract violation: " + "; ".join(problems)
    if strict_mode():
        raise PayloadContractError(message)
    logger.error(message + "  (STRICT_PAYLOADS=false — logged, not raised)")

STATIC_SEQUENCES: dict[str, list[str]] = {
    "conversational_only":  ["ConversationalAgent"],
    "risk_profiling":       ["RiskProfilingAgent", "ExplainabilityAgent"],
    "investment":           ["RiskProfilingAgent", "InvestmentAgent",
                             "ExplainabilityAgent"],
    "budget":               ["BudgetAgent", "ExplainabilityAgent"],
    "full_advisory":        ["RiskProfilingAgent", "InvestmentAgent",
                             "BudgetAgent", "ExplainabilityAgent"],
    "explanation_request":  ["ExplainabilityAgent"],
}


@dataclass(frozen=True)
class AgentCapability:
    """
    What an agent NEEDS and what it PRODUCES, expressed as context-dict keys.

    A planner can never propose a step whose `requires` nothing in the
    graph produces — see validate_capability_graph() below — because that
    check runs over this exact declaration.
    """
    name: str
    requires: frozenset[str]      # context keys that must be present
    produces: frozenset[str]      # context keys it adds
    description: str              # shown to the planner LLM
    cost_hint: str                # "cheap" | "llm" | "expensive"


CAPABILITIES: dict[str, AgentCapability] = {
    "RiskProfilingAgent": AgentCapability(
        name="RiskProfilingAgent",
        requires=frozenset({"user_features"}),
        produces=frozenset({"risk_class", "confidence", "feature_importance"}),
        description="Classifies a customer's risk tier from their financial "
                     "features. Required before any investment recommendation.",
        cost_hint="llm",
    ),
    "InvestmentAgent": AgentCapability(
        name="InvestmentAgent",
        requires=frozenset({"risk_class", "user_features"}),
        produces=frozenset({"shortlist", "synthesis"}),
        description="Recommends products suitable for a given risk class.",
        cost_hint="llm",
    ),
    "BudgetAgent": AgentCapability(
        name="BudgetAgent",
        requires=frozenset({"monthly_expenses"}),
        produces=frozenset({"disposable_income", "savings_rate_pct",
                             "benchmark_comparison"}),
        description="Analyses spending against Irish household benchmarks. "
                     "Will ask for income itself if not already known.",
        cost_hint="cheap",
    ),
    "ExplainabilityAgent": AgentCapability(
        name="ExplainabilityAgent",
        requires=frozenset(),
        produces=frozenset({"full_explanation", "calibration_note"}),
        description="Explains whatever prior agents produced. Runs last.",
        cost_hint="llm",
    ),
    "ConversationalAgent": AgentCapability(
        name="ConversationalAgent",
        requires=frozenset(),
        produces=frozenset({"response", "intent"}),
        description="Handles general conversation and clarification.",
        cost_hint="llm",
    ),
}

# Context keys that arrive from the user/customer record rather than from
# any agent's `produces` — user_features via conversation slot-filling or a
# CustomerStore lookup, monthly_income/monthly_expenses as direct turn
# input. validate_capability_graph() treats these as satisfied without
# requiring a producing capability.
ORCHESTRATOR_SUPPLIED_KEYS = frozenset({
    "user_features",
    "monthly_income",
    "monthly_expenses",
})


def validate_capability_graph() -> list[str]:
    """
    Static check on CAPABILITIES itself, independent of any particular run.
    """
    produced = {key for cap in CAPABILITIES.values() for key in cap.produces}
    satisfiable = produced | ORCHESTRATOR_SUPPLIED_KEYS
    problems: list[str] = []
    for cap in CAPABILITIES.values():
        unmet = cap.requires - satisfiable
        if unmet:
            problems.append(
                f"{cap.name}: requires {sorted(unmet)}, which nothing in "
                f"CAPABILITIES produces and which is not in "
                f"ORCHESTRATOR_SUPPLIED_KEYS"
            )
    producers: dict[str, list[str]] = {}
    for cap in CAPABILITIES.values():
        for key in cap.produces:
            producers.setdefault(key, []).append(cap.name)
    for key, names in producers.items():
        if len(names) > 1:
            problems.append(
                f"{key!r} is produced by more than one capability {sorted(names)} "
                f"— _satisfy_needs() cannot pick one unambiguously"
            )
    return problems
