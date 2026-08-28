"""
Derives a RISK CAPACITY proxy for 'loss_tolerance'
from hard financial data, for use ONLY when the self-reported value is
NOT available — a fallback, never a substitute for direct self-report.

- Risk CAPACITY - objective, financial: how much loss can this person's
    balance sheet absorb without jeopardising their goals? Derivable from
    income, debt, dependencies, investment horizon.
- Risk TOLERANCE / WILLINGNESS - subjective, psychological: how much
    loss is this person comfortable with emotionally? NOT derivable from
    balance-sheet data — requires direct self-report.
- Risk KNOWLEDGE / literacy - factual understanding of financial
    concepts. Has no defensible correlate in credit-bureau-style
    demographic
"""

from __future__ import annotations

from typing import Any, TypedDict

class LossToleranceProxy(TypedDict):
    value: int
    confidence: str
    basis: list[str]

def derive_loss_tolerance_proxy(features: dict[str, Any]) -> LossToleranceProxy| None:
    horizon = features.get("investment_horizon")
    dependents = features.get("dependents")
    age = features.get("age")
    income = features.get("income")
    debt = features.get("existing_debt")

    if horizon is None or dependents is None or age is None:
        return None
    
    score = 3.0
    basis: list[str] = []

    if horizon >= 10:
        score += 1.0
        basis.append("long investment horizon (>=18y)")
    elif horizon <= 3:
        score -= 1.0
        basis.append("short investment horizon (<=3y)")

    if dependents == 0:
        score += 0.5
        basis.append("no dependents")
    elif dependents >= 2:
        score -= 0.5
        basis.append(">=2 dependents")

    if age < 35:
        score += 0.5
        basis.append("age < 35")
    elif age >= 60:
        score -= 0.5
        basis.append("age >= 60")

    confidence = "low"
    if income and debt is not None and income > 0:
        dti = debt / income
        if dti < 0.2:
            score += 0.5
            basis.append("low debt-to-income ratio (<0.2)")
        elif dti > 0.5:
            score -= 0.5
            basis.append("high debt-to-income ratio (>0.5)")
        confidence = "medium"

    clamped = max(1, min(5, round(score)))
    return {"value": clamped, "confidence": confidence, "basis": basis}