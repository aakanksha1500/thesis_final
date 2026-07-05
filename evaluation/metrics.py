"""
This file is grown phase-by-phase alongside each agent.
Only the metrics needed for the current agent are defined here.

Phase 2 (last commit): intent_accuracy, slot_fill_rate
Phase 3 (RQ1, this commit): risk_alignment_rate, f1_risk_classification, auc_roc

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
        if pred == gold:
            bucket_stats[gold]["correct"] += 1
    
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
        },
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

# Phase 3 - RiskProfilingAgent evaluation (RQ1)

def risk_alignment_rate(
        predictions: list[str],
        ground_truth: list[str],
) -> EvalResult:
    """
    Risk Alignment Rate (RAR) - primary RQ1 metric.
    
    Fraction of predictions within 1 tier of ground_truth.
    Used instead of exact match because risk profiling has inherent
    subjectivity - adjacent tier predictions are clinically acceptable
    (e.g. predicting 'moderate' when true label is 'moderately_conservative').
    
    Args:
        predictions: list of predicted risk class strings
        ground_truth: list of labelled risk class strings (from test fixtures)
    """
    TIERS = [
        "conservative", "moderately_conservative", "moderate",
        "moderately_aggressive", "aggressive",
    ]
    tier_idx = {t: i for i, t in enumerate(TIERS)}

    if not predictions:
        return EvalResult("risk_alignment_rate", 0.0, {"error": "empty predictions"})
    if len(predictions) != len(ground_truth):
        return EvalResult("risk_alignment_rate", 0.0,
                          {"error": "length mismatch"})
    
    aligned = sum(
        abs(tier_idx.get(p, 2) - tier_idx.get(g, 2)) <= 1
        for p, g in zip(predictions, ground_truth)
    )
    rar = aligned / len(predictions)

    # Per-tier breakdown for qualitative analysis
    tier_stats: dict = {}
    for p, g in zip(predictions, ground_truth):
        tier_stats.setdefault(g, {"total": 0, "aligned": 0})
        tier_stats[g]["total"] += 1
        if abs(tier_idx.get(p, 2) - tier_idx.get(g, 2)) <= 1:
            tier_stats[g]["aligned"] += 1
    
    per_tier = {
        tier: round(v["aligned"] / v["total"], 3)
        for tier, v in tier_stats.items()
    }

    return EvalResult(
        metric_name="risk_alignment_rate",
        value=round(rar, 4),
        details={
            "n": len(predictions),
            "aligned": aligned,
            "per_tier_alignment": per_tier,
        },
    )

def f1_risk_classification(
        predictions: list[str],
        ground_truth: list[str],
) -> EvalResult:
    """
    Macro-averaged F1 score for 5-class risk classification (RQ1).
    
    Uses macro averaging (equal weight per class) rather than weighted,
    because all five risk tiers are equally important to classify correctly - 
    we do not want the dominant class to inflate the score.
    """

    if not predictions or len(predictions) != len(ground_truth):
        return EvalResult("f1_risk_classification", 0.0,
                          {"error": "empty or mismatched inputs"})
    
    labels = list(dict.fromkeys(ground_truth))
    from collections import defaultdict
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)

    for p, g in zip(predictions, ground_truth):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    
    f1s = []
    per_class = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        f1s.append(f1)
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }
    
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return EvalResult(
        metric_name="f1_risk_classification",
        value=round(macro_f1, 4),
        details={"per_class": per_class},
    )

def hybrid_vs_rule_only_delta(
    hybrid_predictions: list[str],
    rule_only_predictions: list[str],
    ground_truth: list[str],
) -> EvalResult:
    """
    Delta metric comparing hybrid model to rule-only baseline (RQ1).

    Returns the RAR improvement of hybrid over rule-only.
    Positive value = hybrid is better. Zero or negative = ML adds no value.
    Written to results/phase3_risk_baseline.json alongside individual scores.
    """
    hybrid_rar = risk_alignment_rate(hybrid_predictions, ground_truth).value
    rule_rar = risk_alignment_rate(rule_only_predictions, ground_truth).value
    delta = hybrid_rar - rule_rar

    return EvalResult(
        metric_name="hybrid_vs_rule_only_delta",
        value=round(delta, 4),
        details={
            "hybrid_rar": hybrid_rar,
            "rule_only_rar": rule_rar,
            "improvement": f"{delta:+.4f}",
            "interpretation": (
                "Hybrid outperforms rule-only baseline"
                if delta > 0
                else "Rule-only baseline matches or exceeds hybrid"
            ),
        },
    )
