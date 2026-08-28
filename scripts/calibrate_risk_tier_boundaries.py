"""
scripts/calibrate_risk_tier_boundaries.py — fit data-driven tier boundaries
to replace the fixed equal-width thresholds [0.2, 0.4, 0.6, 0.8].

    python scripts/calibrate_risk_tier_boundaries.py

WHY THIS EXISTS
    agents/risk_profiling_agent.py's _score_to_class divides the hybrid
    score into 5 equal-width bands, assuming hybrid_score is roughly
    uniform over [0, 1]. It isn't: rq1_gold_risk_evaluation.json's
    score_distribution_diagnostic shows it compressed to [0.09, 0.825],
    mean 0.486. Equal-width bins over a compressed distribution starve
    the outer classes — on the 200-item gold set, "aggressive" is reached
    by 1 of 200 profiles — which is a classification problem, not just a
    confidence problem: exact accuracy on the same set is 0.380.

WHAT THIS SCRIPT DOES
    Fits quintile boundaries (20/40/60/80th percentiles) on the hybrid
    score distribution of the 1,000-profile Layer-2 STRESS set
    (evaluation.datasets.load_stress_customers) — deliberately NOT the
    200-item Layer-3 gold set. The stress set has no ground-truth labels
    (Layer 2 is "robustness only" — evaluation/datasets.py enforces this),
    so fitting boundaries on it uses only the SCORE distribution, never
    an accuracy label, and keeps the gold set fully held out for
    evaluating whatever these boundaries do to classification accuracy.
    That separation is the whole point: if boundaries were fit on the
    same 200 profiles then evaluated on them, an accuracy improvement
    would partly just be curve-fitting to that one sample.

WHAT THIS SCRIPT DOES NOT DO
    It does not touch _compute_confidence's AUROC-of-correctness problem.
    Quantile boundaries fix classification balance; they do not turn a
    distance-from-boundary heuristic into a learned probability estimate.
    See scripts/calibrate_risk_confidence.py (isotonic recalibration) for
    that separate problem, and the dissertation's Section VII-D/C for why
    combining both still tops out under AUROC 0.6.

DEPLOYMENT IS OPT-IN
    RiskProfilingAgent only uses these boundaries when
    settings.risk.use_quantile_tier_boundaries is true
    (USE_QUANTILE_TIER_BOUNDARIES=true) — same opt-in pattern as
    use_trained_model and use_calibrated_confidence, for the same reason:
    this changes every downstream classification, so it does not activate
    silently just because this script has been run once.

NO LLM, NO API KEY
    Everything here is deterministic, like the confidence calibrator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import settings  # noqa: E402
from evaluation.datasets import load_gold_risk_profiles, load_stress_customers  # noqa: E402
from evaluation.risk_rubric import RISK_TIERS  # noqa: E402
from evaluation.results_io import write_results  # noqa: E402


def _score_to_class(score: float, thresholds: list[float]) -> str:
    for i, t in enumerate(thresholds):
        if score < t:
            return RISK_TIERS[i]
    return RISK_TIERS[-1]


def main() -> int:
    from agents.risk_profiling_agent import RiskProfilingAgent
    from utils.llm_client import LLMClient

    agent = RiskProfilingAgent(LLMClient(force_mock=True))

    # Fit on the stress set's SCORE distribution only — no labels used,
    # no overlap with the gold set used to evaluate below.
    stress = load_stress_customers()
    stress_scores = [
        agent._hybrid_score(r["features"])[2] for r in stress.records
    ]
    thresholds = [round(float(t), 4) for t in np.percentile(stress_scores, [20, 40, 60, 80])]
    boundaries = [0.0] + thresholds + [1.0]

    payload = {
        "method": "quintile (20/40/60/80th percentile)",
        "fit_on": {
            "dataset": "Layer-2 stress set (unlabelled, scores only)",
            "n": len(stress_scores),
            "score_min": round(min(stress_scores), 4),
            "score_max": round(max(stress_scores), 4),
            "score_mean": round(float(np.mean(stress_scores)), 4),
        },
        "thresholds": thresholds,
        "boundaries": boundaries,
        "risk_tiers": RISK_TIERS,
    }

    boundaries_path = settings.risk.tier_boundaries_path
    boundaries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(boundaries_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  boundaries saved to {boundaries_path}")

    # Held-out evaluation on the gold set — never used for fitting above.
    gold = load_gold_risk_profiles(include_original_15=False)
    truth = gold.ground_truth()
    gold_scores = [agent._hybrid_score(r["features"])[2] for r in gold.records]

    pred_eq = [agent._score_to_class(s) for s in gold_scores]
    pred_q = [_score_to_class(s, thresholds) for s in gold_scores]

    acc_eq = float(np.mean([p == t for p, t in zip(pred_eq, truth)]))
    acc_q = float(np.mean([p == t for p, t in zip(pred_q, truth)]))

    from collections import Counter
    balance_eq = dict(Counter(pred_eq))
    balance_q = dict(Counter(pred_q))

    print(f"\n  held-out gold-set accuracy (n={len(gold_scores)}):")
    print(f"    equal-width [0.2,0.4,0.6,0.8]: {acc_eq:.4f}   class balance: {balance_eq}")
    print(f"    quantile    {thresholds}: {acc_q:.4f}   class balance: {balance_q}")

    results_payload: dict[str, Any] = {
        "phase": "RQ1 supporting — tier boundary recalibration",
        "boundaries_fit": payload,
        "held_out_evaluation": {
            "dataset": "Layer-3 gold set (200 profiles, held out from fitting)",
            "accuracy_equal_width": round(acc_eq, 4),
            "accuracy_quantile": round(acc_q, 4),
            "class_balance_equal_width": balance_eq,
            "class_balance_quantile": balance_q,
        },
        "note": (
            "Fixes classification/class-balance, not the separate "
            "confidence-AUROC problem — see "
            "scripts/calibrate_risk_confidence.py and the dissertation's "
            "Section VII-D for that."
        ),
        "currently_deployed": settings.risk.use_quantile_tier_boundaries,
    }
    write_results(results_payload, "rq1_tier_boundary_recalibration.json")

    if not settings.risk.use_quantile_tier_boundaries:
        print("\n  currently DISABLED in production "
              "(settings.risk.use_quantile_tier_boundaries=False) — "
              "set USE_QUANTILE_TIER_BOUNDARIES=true to enable.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
