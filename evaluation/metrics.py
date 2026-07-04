"""
This file is grown phase-by-phase alongside each agent.
Only the metrics needed for the current agent are defined here.

Phase 2 (this commit): intent_accuracy, slot_fill_rate

All metric functions return EvalResult(metric_name, value, details).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass
class EvalResult:
    """
    Standardised result for every metric function.
    value - the primary scalar reported in the dissertation table.
    details - supporting breakdown committed to results/*.json.
    """
    metric_name: str
    value: float
    details: dict[str,Any] | None = None

    def to_dict(self) -> dict:
        return {
            "metric": self.metric_name,
            "value": round(self.value, 4),
            "details": self.details or {},
        }


# Phase 2 metrics - Conversational evaluation

def intent_accuracy(
        predictions: list[str],
        ground_truth: list[str],
) -> EvalResult:
    """
    Intent classification accuracy against banking77 [D5] ground truth.
    
    This is a flat accuracy metric - fraction of messages where the
    predicted routing bucket matches the labelled bucket exactly.
    
    Args:
        predictions: list of predicted bucket strings from ConversationalAgent
        ground_truth: list of labelled bucket strings from Banking77 fixtures
        
    Returns:
        EvalResult with value in [0, 1] and per-bucket breakdown in details.
    """
    if not predictions:
        return EvalResult("intent_accuracy", 0.0, {"error": "empty predictions list"})
    
    if len(predictions) != len(ground_truth):
        return EvalResult(
            "intent_accuracy", 0.0,
            {"error": f"length mismatch: {len(predictions)} vs {len(ground_truth)}"}
        )
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    accuracy = correct / len(predictions)

    # per-bucket breakdown for qualitative analysis
    bucket_stats: dict[str, dict] = {}
    for pred, gold in zip(predictions, ground_truth):
        if gold not in bucket_stats:
            bucket_stats[gold] = {"total": 0, "correct": 0}
        bucket_stats[gold]["total"] += 1
    
    per_bucket = {
        bucket: round(v["correct"] / v["total"], 3)
        for bucket, v in bucket_stats.items()
    }

    return EvalResult(
        metric_name="intent_accuracy",
        value=round(accuracy,4),
        details={
            "n": len(predictions),
            "correct": correct,
            "per_bucket_accuracy": per_bucket,
        }
    )

def slot_fill_rate(
        sessions: list[dict[str,Any]],
        required_slots: list[str],
) -> EvalResult:
    """
    Slot fill rate across multi-turn conversation sessions.

    Measures how often the ConversationalAgent successfully collects
    all required slots through natural elicitation before escalating.
    Based on MultiWOZ [D6] dialogue state tracking evaluation convention.

    A session is a dict:
      {
        "collected_slots": {"age": 35, "income": 60000, ...},
        "turns_taken": 3,
        "escalated": True
      }

    Slot fill rate = fraction of required slots collected, averaged
    across all sessions that reached escalation.

    Args:
        sessions:       list of session result dicts (from test fixtures)
        required_slots: slot names that must be collected before escalation

    Returns:
        EvalResult with value in [0, 1] and per-slot fill rates in details.
    """
    if not sessions:
        return EvalResult("slot_fill_rate", 0.0, {"error": "no sessions provided"})

    slot_fill_counts: dict[str, int] = {s: 0 for s in required_slots}
    escalated_sessions = [s for s in sessions if s.get("escalated", False)]
    n = len(escalated_sessions) if escalated_sessions else len(sessions)
    eval_sessions = escalated_sessions if escalated_sessions else sessions

    for session in eval_sessions:
        collected = session.get("collected_slots", {})
        for slot in required_slots:
            if slot in collected and collected[slot] is not None:
                slot_fill_counts[slot] += 1
    
    per_slot = {
        slot: round(count / n, 3)
        for slot, count in slot_fill_counts.items()
    }
    overall = sum(per_slot.values()) / len(required_slots) if required_slots else 0.0

    return EvalResult(
        metric_name="slot_fill_rate",
        value=round(overall, 4),
        details={
            "n_sessions": n,
            "escalated_sessions": len(escalated_sessions),
            "per_slot_fill_rate": per_slot,
            "required_slots": required_slots,
        },
    )