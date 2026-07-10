"""
This file is grown phase-by-phase alongside each agent.
Only the metrics needed for the current agent are defined here.

Phase 2: intent_accuracy, slot_fill_rate
Phase 3 (RQ1, last commit): risk_alignment_rate, f1_risk_classification, auc_roc
Phase 4 (RQ2 this commit): ndcg_at_k, precision_at_k

All metric functions return EvalResult(metric_name, value, details).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

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

def auc_roc(
        hybrid_scores: list[float],
        ground_truth: list[str],
        risk_classes: list[str] | None = None,
) -> EvalResult:
    """
    Macro-averaged one-vs-rest AUC-ROC for 5-tier risk classification (RQ1).
    Args:
        hybrid_scores: continuous hybrid scores in [0, 1], one per profile
                        (output of RiskProfilingAgent._hybrid_score()).
        ground_truth: labelled risk class strings, same order/length.
        risk_classes: ordered tier names low->high risk. Defaults to the 
                        five standard tiers.
    """
    TIERS = risk_classes or [
        "conservative", "moderately_conservative", "moderate",
        "moderately_aggressive", "aggressive",
    ]

    if not hybrid_scores or len(hybrid_scores) != len(ground_truth):
        return EvalResult("auc_roc", 0.0, {"error": "empty or mismatched inputs"})
    
    present_classes = sorted(set(ground_truth), key=TIERS.index)
    if len(present_classes) < 2:
        return EvalResult("auc_roc", 0.0, {
            "error": "need >= 2 distinct classes in ground_truth to compute AUC-ROC",
            "classes_present": present_classes,
        })
    
    counts = {c: ground_truth.count(c) for c in present_classes}
    if any(n < 2 for n in counts.values()):
        return EvalResult("auc_roc", 0.0, {
            "error": "each class needs >= 2 samples for stable AUC-ROC",
            "class_counts": counts,
        })
    
    n_tiers = len(TIERS)
    tier_centers = np.array([(i + 0.5) / n_tiers for i in range(n_tiers)])

    y_true = np.array([TIERS.index(g) for g in ground_truth])
    scores = np.clip(np.array(hybrid_scores, dtype=float), 0.0, 1.0)

    # Distance-based softmax: closer tier centers get higher pseudo-probability.
    # Temperature tuned so a score exactly on a tier boundary splits mass
    # mostly between the two adjacent tiers, not uniformly across all five.
    temperature = 1.0 / n_tiers
    dists = np.abs(scores[:, None] - tier_centers[None, :])
    logits = -dists / temperature
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Macro AUC is averaged only over classes that actually appear in this
    # ground_truth sample (with >= 2 examples). Passing sklearn a fixed
    # 5-class label set when a fixture only contains 3 tiers produces NaN
    # for the absent classes and poisons a built-in macro average — so
    # per-class AUC is computed directly and averaged manually instead.
    per_class_auc: dict = {}
    for i, tier in enumerate(TIERS):
        if tier not in present_classes:
            continue
        y_bin = (y_true == i).astype(int)
        if len(set(y_bin.tolist())) < 2:
            continue
        try:
            per_class_auc[tier] = round(float(roc_auc_score(y_bin, proba[:, i])), 4)
        except ValueError:
            continue

    if not per_class_auc:
        return EvalResult("auc_roc", 0.0, {
            "error": "no class had both positive and negative examples",
            "class_counts": counts,
        })

    macro_auc = sum(per_class_auc.values()) / len(per_class_auc)

    return EvalResult(
        metric_name="auc_roc",
        value=round(macro_auc, 4),
        details={
            "method": "one_vs_rest_macro",
            "n": len(hybrid_scores),
            "classes_evaluated": list(per_class_auc.keys()),
            "per_class_auc": per_class_auc,
            "class_counts": counts,
        },
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

def precision_at_k(
    ranked_ids: list[str],
    relevant_ids: set[str],
    k: int = 3,
) -> EvalResult:
    """
    Precision@k — primary RQ2 ranking-quality metric.

    Fraction of the top-k ranked product IDs that are members of the
    ground-truth relevant set for a given query. Unlike NDCG, this ignores
    the *order* within the top-k — it only asks whether the right products
    made the cut at all, which is the more forgiving first-pass check
    before the order-sensitive NDCG metric is applied.

    Used in: tests/unit/test_investment_agent.py
    Written to: results/rq2_investment_baseline.json

    Args:
        ranked_ids:   product_id strings in ranked order (best first),
                      already truncated to length k by the caller or not —
                      only the first k entries are considered either way.
        relevant_ids: set of product_id strings considered ground-truth
                      correct/suitable for this query.
        k: cutoff rank to evaluate at (default 3, matching top_k default).

    Returns:
        EvalResult with value in [0, 1].
    """
    if not ranked_ids:
        return EvalResult("precision_at_k", 0.0, {"error": "empty ranked_ids", "k": k})

    top_k = ranked_ids[:k]
    hits = sum(1 for pid in top_k if pid in relevant_ids)
    precision = hits / len(top_k) if top_k else 0.0

    return EvalResult(
        metric_name="precision_at_k",
        value=round(precision, 4),
        details={
            "k": k,
            "top_k_ids": top_k,
            "hits": hits,
            "relevant_ids": sorted(relevant_ids),
        },
    )

def ndcg_at_k(
    ranked_ids: list[str],
    relevance_scores: dict[str, float],
    k: int = 3,
) -> EvalResult:
    """
    Normalised Discounted Cumulative Gain @ k — secondary RQ2 metric.

    Unlike precision_at_k, NDCG is order-sensitive: placing the most
    relevant product first scores higher than placing it third, even if
    both rankings contain the same top-k set. This directly evaluates
    the _rank_products() scoring/sorting logic (hybrid layer 2), not
    just the _filter_by_risk_class() suitability gate (hybrid layer 1).

    DCG@k = sum_{i=1}^{k} relevance_i / log2(i + 1)
    IDCG@k = DCG@k computed on the ideal (sorted-descending) ordering
    NDCG@k = DCG@k / IDCG@k   (0.0 if IDCG@k is 0, i.e. no relevant items)

    Used in: tests/unit/test_investment_agent.py
    Written to: results/rq2_investment_baseline.json

    Args:
        ranked_ids:       product_id strings in ranked order (best first).
        relevance_scores: product_id -> graded relevance (e.g. 0/1/2, or
                          continuous). product_ids absent from this dict
                          are treated as relevance 0.
        k: cutoff rank to evaluate at.

    Returns:
        EvalResult with value in [0, 1].
    """
    import math

    if not ranked_ids:
        return EvalResult("ndcg_at_k", 0.0, {"error": "empty ranked_ids", "k": k})

    top_k = ranked_ids[:k]

    def dcg(ids: list[str]) -> float:
        return sum(
            relevance_scores.get(pid, 0.0) / math.log2(i + 2)
            for i, pid in enumerate(ids)
        )

    actual_dcg = dcg(top_k)

    ideal_order = sorted(
        relevance_scores.keys(), key=lambda pid: relevance_scores[pid], reverse=True
    )[:k]
    ideal_dcg = dcg(ideal_order)

    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return EvalResult(
        metric_name="ndcg_at_k",
        value=round(ndcg, 4),
        details={
            "k": k,
            "top_k_ids": top_k,
            "dcg": round(actual_dcg, 4),
            "idcg": round(ideal_dcg, 4),
            "ideal_order": ideal_order,
        },
    )
