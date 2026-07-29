from __future__ import annotations

import os
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

_PARTIAL_STATUSES = {"incomplete", "blocked", "error", "skipped"}

def strict_mode() -> bool:
    return os.getenv("STRICT_PAYLOADS", "false").lower() == "true"

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
    if payload.get("status") in _PARTIAL_STATUSES:
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
    logger.error(message + "  (set STRICT_PAYLOADS=true to make this fatal)")
