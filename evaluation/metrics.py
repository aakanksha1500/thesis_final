"""
This file is grown phase-by-phase alongside each agent.
Only the metrics needed for the current agent are defined here.

Phase 2: intent_accuracy, slot_fill_rate
Phase 3 (RQ1): risk_alignment_rate, f1_risk_classification, auc_roc
Phase 4 (RQ2 last commit): ndcg_at_k, precision_at_k
Phase 6 (RQ3 this commit): transparency_perception_score, trust_calibration_index

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

# Phase 6 metrics — ExplainabilityAgent evaluation

def transparency_perception_score(
    survey_responses: list[dict],
) -> EvalResult:
    """
    Transparency Perception Score (TPS) — RQ3 primary metric.

    Aggregates simulated Likert-scale survey responses across three
    explanation quality dimensions. In the full dissertation study,
    these come from a pilot study (n≥20 participants). In Phase 6,
    they come from synthetic fixtures that represent the expected
    distribution of responses per ablation condition.

    Each survey_response dict:
      {
        "clarity":              int (1-5) — how clear was the explanation?
        "source_visible":       int (0/1) — could you see where info came from?
        "counterfactual_useful":int (1-5) — was the 'what if' useful?
        "trust_appropriate":    int (1-5) — did explanation help calibrate trust?
      }

    TPS = mean of normalised scores across all dimensions and responses.
    Range [0, 1]; higher = better perceived transparency.

    Written to: results/rq3_*.json (one per ablation condition)
    """
    if not survey_responses:
        return EvalResult(
            "transparency_perception_score", 0.0,
            {"error": "no survey responses provided"}
        )

    clarity_scores   = [r.get("clarity", 3) / 5.0 for r in survey_responses]
    source_scores    = [float(r.get("source_visible", 0)) for r in survey_responses]
    cf_scores        = [r.get("counterfactual_useful", 3) / 5.0 for r in survey_responses]
    trust_scores     = [r.get("trust_appropriate", 3) / 5.0 for r in survey_responses]

    import statistics
    dim_means = [
        statistics.mean(clarity_scores),
        statistics.mean(source_scores),
        statistics.mean(cf_scores),
        statistics.mean(trust_scores),
    ]
    tps = statistics.mean(dim_means)

    return EvalResult(
        metric_name="transparency_perception_score",
        value=round(tps, 4),
        details={
            "n_responses": len(survey_responses),
            "mean_clarity": round(dim_means[0], 3),
            "mean_source_visible": round(dim_means[1], 3),
            "mean_counterfactual_useful": round(dim_means[2], 3),
            "mean_trust_appropriate": round(dim_means[3], 3),
        },
    )


def trust_calibration_index(
    user_trust_scores: list[float],
    advice_quality_scores: list[float],
) -> EvalResult:
    """
    Trust Calibration Index (TCI) — RQ3 secondary metric.

    Takayanagi et al. [7] (2025): users express high trust in AI financial
    advice even when advice quality is objectively poor — trust-quality
    decoupling. TCI directly measures whether the ExplainabilityAgent's
    X3 design goal is achieved: does trust track quality?

    TCI = 1 - mean(|trust - quality| / max_scale)
    Range [0, 1]:
      1.0 = perfectly calibrated (trust always matches quality)
      0.0 = maximally miscalibrated (trust inversely correlates with quality)

    Args:
      user_trust_scores:    list of user-reported trust scores (1-5 scale)
      advice_quality_scores: list of objective quality scores (1-5 scale,
                             from Agent-as-Judge evaluator — Phase 7)

    The Takayanagi et al. finding: without XAI, TCI is low (users trust
    regardless of quality). With full 3-layer XAI, TCI increases because
    users can now assess quality through the explanation and calibrate
    accordingly. The ablation table should show TCI increasing across
    conditions A → B → C.

    Written to: results/rq3_*.json alongside TPS.
    """
    if not user_trust_scores:
        return EvalResult(
            "trust_calibration_index", 0.0,
            {"error": "empty trust scores"}
        )
    if len(user_trust_scores) != len(advice_quality_scores):
        return EvalResult(
            "trust_calibration_index", 0.0,
            {"error": "length mismatch between trust and quality scores"}
        )

    MAX_SCALE = 4.0   # range is 1-5, so max deviation is 4
    deviations = [
        abs(t - q) / MAX_SCALE
        for t, q in zip(user_trust_scores, advice_quality_scores)
    ]
    import statistics
    tci = 1.0 - statistics.mean(deviations)

    return EvalResult(
        metric_name="trust_calibration_index",
        value=round(tci, 4),
        details={
            "n": len(user_trust_scores),
            "mean_trust": round(statistics.mean(user_trust_scores), 3),
            "mean_quality": round(statistics.mean(advice_quality_scores), 3),
            "mean_deviation": round(statistics.mean(deviations), 3),
            "interpretation": (
                "Well-calibrated: trust tracks advice quality"
                if tci >= 0.7
                else "Poorly calibrated: trust does not track advice quality "
                     "(Takayanagi et al. trust-quality decoupling)"
            ),
        },
    )

# Phase 7 metrics - Orchestrator evaluation
def step_progress_rate(
        step_records: list[dict]
) -> EvalResult:
    """
    Step-level Progress Rate - AgentBoard (E1)
    
    Measures fraction of agent execution steps successfully completed
    across a sesson or scenario. Enables failure LOCALISATION within the 
    pipeline (which agent failed?) rather than only end-task success/failure.
    
    This is the key methodological contribution of E1:
    Esisting benchmarks only measured final task success. AgentBoard
    showed that stage-level measurement reveals failure patterns invisible
    at the taask level - e.g. an agent that partially completes and 
    produces plausible-looking output before failing.
    
    step_records: list of dicts from AgentResult.to_step_record():
        {"step_id": str, "agent": str, "completed": bool, "duration_ms": float}
    
    Written to: results/rq4_mas_coherence.json
    """
    if not step_records:
        return EvalResult(
            "step_progress_rate", 0.0,
            {"error": "no step records provided"}
        )

    total = len(step_records)
    completed = sum(1 for s in step_records if s.get("completed", False))
    rate = completed / total

    # Per-agent breakdown - identifies which agent has the lowest completion rate
    by_agent: dict[str, list[bool]] = {}
    for s in step_records:
        agent = s.get("agent", "unknown")
        by_agent.setdefault(agent, [])
        by_agent[agent].append(s.get("completed", False))

    per_agent_rate = {
        agent: round(sum(completions) / len(completions), 3)
        for agent, completions in by_agent.items()
    }

    # Identify bottleneck agent
    bottleneck = min(per_agent_rate, key=per_agent_rate.get) if per_agent_rate else None

    return EvalResult(
        metric_name="step_progress_rate",
        value=round(rate, 4),
        details={
            "total_steps": total,
            "completed_steps": completed,
            "per_agent_rate": per_agent_rate,
            "bottleneck_agent": bottleneck,
        }
    )

def component_synergy_score(
        audit_records: list[dict],
) -> EvalResult:
    """
    Component Synergy Score (CSS)
    
    Measures how effectively the agents collaborate as a system.
    Penalises conflicts and failures: rewards successful handoffs.
    
    CSS = (successful_calls - 0.5 * conflicts - failures) / total_calls
    Range [0, 1]: clamped to [0, 1].
    
    The 0.5 penalty for conflicts (vs 1.0 for failures) reflrcts the 
    ConflictResolver's ability to recover from conflicts gracefully - 
    they reduce quality but do not break the pipelne.
    
    audit_records: JSONL records from Auditing.read_all().
    Written to: results/rq4_mas_coherence.json
    """
    if not audit_records:
        return EvalResult(
            "component_synergy_score", 0.0,
            {"error": "no audit records provided"}
        )

    agent_calls = [
        r for r in audit_records
        if r.get("event_type") == "AGENT_CALL"
    ]
    failures = [
        r for r in audit_records
        if r.get("event_type") == "AGENT_FAILURE"
    ]
    conflicts = [
        r for r in audit_records
        if r.get("event_type") == "CONFLICT_DETECTED"
    ]

    total = len(agent_calls)
    if total == 0:
        return EvalResult(
            "component_synergy_score", 0.0,
            {"error": "no AGENT_CALL records in audit log"}
        )

    successful = sum(
        1 for r in agent_calls
        if r.get("payload", {}).get("success", False)
    )

    n_failures = len(failures)
    n_conflicts = len(conflicts)

    raw_css = (successful - 0.5 * n_conflicts - n_failures) / total
    css = max(0.0, min(1.0, raw_css))

    return EvalResult(
        metric_name="component_synergy_score",
        value=round(css, 4),
        details={
            "total_agent_calls": total,
            "successful_calls": successful,
            "conflicts": n_conflicts,
            "failures": n_failures,
            "raw_css": round(raw_css, 4),
        }
    )

def tool_utilisation_efficacy(
        audit_records: list[dict],
        routing_plans: list[list[str]] | None = None,
) -> EvalResult:
    """
    Tool Utilisation Efficacy
    
    Measures whether agents were invoked appropriately - not too many 
    (wasteful), not too few (incomplete). The ideal is that every agent
    invoked was necessary and every necessary agent was invoked.
    
    In Phase 7 without a ground-truth routing plan, TUE is computed as:
      TUE = 1 - (redundant_calls / total_calls)

    A call is redundant if the same agent was invoked more than once in
    the same turn without a failure between invocations.

    When routing_plans is provided (list of expected agent sequences per
    turn), TUE measures alignment with the plan:
      TUE = mean(matched_agents / union(planned, actual)) per turn

    Written to: results/rq4_mas_coherence.json
    """
    if not audit_records:
        return EvalResult(
            "tool_utilisation_efficacy", 0.0,
            {"error": "no audit records"}
        )

    # Group agent calls by turn_id
    turns: dict[str, list[str]] = {}
    for r in audit_records:
        if r.get("event_type") == "AGENT_CALL":
            turn_id = r.get("turn_id", "unknown")
            agent = r.get("payload", {}).get("agent", "unknown")
            turns.setdefault(turn_id, []).append(agent)

    if not turns:
        return EvalResult(
            "tool_utilisation_efficacy", 0.0,
            {"error": "no agent calls by turn found"}
        )

    # Compute redundancy per turn
    redundant_total = 0
    total_calls = 0
    for turn_id, agents in turns.items():
        total_calls += len(agents)
        seen = set()
        for agent in agents:
            if agent in seen:
                redundant_total += 1
            seen.add(agent)

    tue = 1.0 - (redundant_total / total_calls) if total_calls > 0 else 0.0

    return EvalResult(
        metric_name="tool_utilisation_efficacy",
        value=round(tue, 4),
        details={
            "total_calls": total_calls,
            "redundant_calls": redundant_total,
            "turns_analysed": len(turns),
        }
    )

def routing_accuracy(
        predicted_routings: list[str],
        expected_routings: list[str],
) -> EvalResult:
    """
    Routing accuracy - fraction of turns where the Orchestrator chose
    the correct agent sequence
    
    Directly measures the HALO layer 1 decomposition quality.
    A wrong routing (e.g. CONVERSATIONAL_ONLY when  INVESTMENT was needed)
    is the primary failure mode in muti-agent systems - the right answer
    cannot be produced if the wrong agents are called.
    
    Written to: results/rq4_mas_coherence.json alongside CSS and TUE
    """
    if not predicted_routings:
        return EvalResult("routing_accuracy", 0.0, {"error": "empty predictions"})
    if len(predicted_routings) != len(expected_routings):
        return EvalResult(
            "routing_accuracy", 0.0,
            {"error": "length mismatch"}
        )

    correct = sum(p == e for p, e in zip(predicted_routings, expected_routings))
    accuracy = correct / len(predicted_routings)

    per_routing: dict[str, dict] = {}
    for pred, exp in zip(predicted_routings, expected_routings):
        per_routing.setdefault(exp, {"total": 0, "correct": 0})
        per_routing[exp]["total"] += 1
        if pred == exp:
            per_routing[exp]["correct"] += 1

    per_routing_acc = {
        r: round(v["correct"] / v["total"], 3)
        for r, v in per_routing.items()
    }

    return EvalResult(
        metric_name="routing_accuracy",
        value=round(accuracy, 4),
        details={
            "n": len(predicted_routings),
            "correct": correct,
            "per_routing_accuracy": per_routing_acc,
        },
    )

# Phase 8 - RAG + hallucination detection evaluation

def _normalise_numeric_answer(raw: str) -> float | None:
    """
     Normalise a FinQA-style answer string to a float for comparison.
    Strips currency symbols, percent signs, commas, and surrounding
    whitespace; treats "15.4%" and "0.154" as comparable by leaving the
    percent-scale conversion to the caller's tolerance (FinQA answers are
    conventionally reported in the % scale they appear in the question).
    Returns None if no numeric value can be extracted.
    """
    import re as _re

    if raw is None:
        return None
    cleaned = str(raw).strip().replace(",", "").replace("$", "").replace("%","")
    match = _re.search(r"-?\d+\.?\d*", cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None

def finqa_exact_match(
    predictions: list[str],
    ground_truth: list[str],
    tolerance: float = 0.01,
) -> EvalResult:
    """
    FinQA numerical exact-match accuracy — RQ5 primary metric.

    Follows the FinQA benchmark convention (Chen et al.): a prediction is
    "correct" if its normalised numeric value matches the ground truth
    answer within `tolerance` (absolute), not by string equality — FinQA
    answers vary in formatting ("15.4%" vs "15.40%" vs "0.154") while
    representing the same value.

    Args:
      predictions:  model-produced answer strings, one per FinQA question.
      ground_truth: FinQA Verified [D1] ground truth answer strings.
      tolerance:    absolute tolerance for numeric match. Default 0.01
                    follows FinQA's own reported evaluation tolerance.

    Used in: RQ5 baseline (no RAG, commit 38) vs RAG-grounded (commit 39)
             comparison — the delta between the two runs is the RQ5
             headline finding.
    Written to: results/rq5_finqa_no_rag.json, results/rq5_finqa_with_rag.json
    """
    if not predictions:
        return EvalResult("finqa_exact_match", 0.0, {"error": "empty predictions list"})
    if len(predictions) != len(ground_truth):
        return EvalResult(
            "finqa_exact_match", 0.0,
            {"error": f"length mismatch: {len(predictions)} vs {len(ground_truth)}"}
        )

    correct = 0
    unparseable = 0
    for pred, gold in zip(predictions, ground_truth):
        pred_val = _normalise_numeric_answer(pred)
        gold_val = _normalise_numeric_answer(gold)
        if pred_val is None or gold_val is None:
            unparseable += 1
            continue
        if abs(pred_val - gold_val) <= tolerance:
            correct += 1

    accuracy = correct / len(predictions)

    return EvalResult(
        metric_name="finqa_exact_match",
        value=round(accuracy, 4),
        details={
            "n": len(predictions),
            "correct": correct,
            "unparseable": unparseable,
            "tolerance": tolerance,
        },
    )

def hallucination_rate(
    hallucination_reports: list[dict[str, Any]],
) -> EvalResult:
    """
    Hallucination rate — RQ5 secondary metric.

    Aggregates per-response HHEM hallucination reports across an
    evaluation run. Reports the fraction of *claims* flagged (not just
    fraction of responses with >=1 flag) — a response with one flagged
    figure among ten sound ones is a different failure profile than a
    response where every claim is unsupported, and the claim-level rate
    captures that distinction the response-level rate would hide.

    Args:
      hallucination_reports: list of HallucinationReport.to_dict() outputs,
                              one per evaluated response.

    Used alongside finqa_exact_match for the RQ5 before/after RAG
    comparison (commits 38-39): hallucination_rate should decrease when
    RAG grounding is enabled, corroborating the exact-match improvement
    rather than the two metrics moving independently.

    Written to: results/rq5_finqa_no_rag.json, results/rq5_finqa_with_rag.json
    """
    if not hallucination_reports:
        return EvalResult(
            "hallucination_rate", 0.0,
            {"error": "no hallucination reports provided"}
        )

    total_claims = sum(r.get("n_claims", 0) for r in hallucination_reports)
    total_flagged = sum(r.get("n_flagged", 0) for r in hallucination_reports)
    responses_with_any_flag = sum(
        1 for r in hallucination_reports if r.get("n_flagged", 0) > 0
    )

    claim_level_rate = (total_flagged / total_claims) if total_claims else 0.0
    response_level_rate = responses_with_any_flag / len(hallucination_reports)

    modes = {r.get("mode", "unknown") for r in hallucination_reports}

    return EvalResult(
        metric_name="hallucination_rate",
        value=round(claim_level_rate, 4),
        details={
            "n_responses": len(hallucination_reports),
            "total_claims": total_claims,
            "total_flagged": total_flagged,
            "response_level_hallucination_rate": round(response_level_rate, 4),
            "detector_modes_seen": sorted(modes),
        },
    )
