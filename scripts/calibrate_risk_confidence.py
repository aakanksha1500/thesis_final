"""
scripts/calibrate_risk_confidence.py — recalibrate RiskProfilingAgent's
confidence signal against the gold set it's actually wrong about.

    python scripts/calibrate_risk_confidence.py

WHY THIS EXISTS
    scripts/eval_confidence_calibration.py measured the problem:
    _compute_confidence (agents/risk_profiling_agent.py) is a decision
    margin — distance from the nearest of the five equal-width class
    boundaries — built on an implicit assumption that hybrid_score is
    roughly uniformly distributed across [0,1]. It isn't:
    rq1_gold_risk_evaluation.json's score_distribution_diagnostic shows
    hybrid_score compressed to [0.09, 0.825], mean 0.4864. That
    compression pulls scores toward 0.5 regardless of whether the
    resulting classification is correct — and 0.5 is the dead centre of
    the "moderate" bin, where the margin heuristic reports its HIGHEST
    confidence. Measured on the 200-item gold set:
    AUROC(confidence predicts correct) = 0.391 — worse than chance — and
    mean confidence when WRONG (0.544) is higher than mean confidence
    when RIGHT (0.437). settings.risk.min_confidence, already a
    production gate, is currently passing through the LESS accurate
    predictions and holding back the MORE accurate ones.

WHAT THIS SCRIPT DOES
    Fits an isotonic regression mapping raw decision-margin confidence to
    P(correct), using the SAME 200-item Layer-3 gold set (never Layers
    1/2 — evaluation/datasets.py enforces this at the dataset level for
    anything requiring ground truth). Isotonic regression is the
    textbook fix for exactly this failure mode: a monotonic relationship
    exists between score and outcome, but the wrong direction and the
    wrong shape are baked into a hand-derived formula instead of learned
    from outcomes. increasing="auto" lets sklearn detect the sign from
    the data via Spearman correlation rather than hard-coding the
    inversion this docstring already describes — the fix should be
    validated by the data, not assumed to have the sign this docstring
    expects.

HONEST EVALUATION, NOT SELF-GRADING
    Fitting and evaluating a calibrator on the same 200 points would
    overstate the fix — isotonic regression can memorise noise in a
    small sample, especially with n=200. 5-fold cross-validation reports
    AUROC on held-out folds only: each fold's calibrator sees the other
    4 folds during fitting and is scored on the fold it never saw. The
    DEPLOYED calibrator (saved to
    data/processed/risk_confidence_calibrator.pkl) is then refit on all
    200 points, which is standard practice once CV has shown the
    approach generalises — but the reported AUROC in this file's
    headline_metrics is the CV number, not a number computed on the
    deployed calibrator's own training data.

DEPLOYMENT IS OPT-IN
    RiskProfilingAgent only applies this calibrator when
    settings.risk.use_calibrated_confidence is true
    (USE_CALIBRATED_CONFIDENCE=true) — mirroring use_trained_model's
    existing opt-in pattern, for the same reason its own log message
    gives: this changes every confidence-dependent number (the approval
    gate's actual behaviour, every RQ1 confidence figure), so it does
    not activate silently just because this script has been run once.

NO LLM, NO API KEY
    Everything here is deterministic, like RQ1 and
    eval_confidence_calibration.py itself.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from config.settings import ROOT_DIR, settings  # noqa: E402
from evaluation.datasets import load_gold_risk_profiles  # noqa: E402
from evaluation.results_io import write_results  # noqa: E402

CV_FOLDS = 5
# Fixed and documented rather than left to numpy's global state, so a
# re-run reproduces the exact same fold assignment and therefore the
# exact same CV numbers reported here.
CV_SEED = 20260817


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """Same Mann-Whitney U identity as eval_confidence_calibration.py's
    _auroc — duplicated rather than imported because that script keeps
    its helpers private (no __all__, module-level functions), and two
    ~10-line identical implementations is a smaller coupling risk than
    reaching into another script's internals."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg))


def _gate_utility(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    above = [k for s, k in zip(scores, labels) if s >= threshold]
    below = [k for s, k in zip(scores, labels) if s < threshold]
    return {
        "n_at_or_above": len(above),
        "accuracy_at_or_above": round(float(np.mean(above)), 4) if above else None,
        "n_below": len(below),
        "accuracy_below": round(float(np.mean(below)), 4) if below else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    args = ap.parse_args()

    from agents.risk_profiling_agent import RiskProfilingAgent
    from utils.llm_client import LLMClient

    dataset = load_gold_risk_profiles(include_original_15=False)
    agent = RiskProfilingAgent(LLMClient(force_mock=True))
    truth = dataset.ground_truth()

    raw_confidence: list[float] = []
    correct: list[int] = []
    for record, expected in zip(dataset.records, truth):
        _ml, _rule, hybrid = agent._hybrid_score(record["features"])
        predicted = agent._score_to_class(hybrid)
        raw_confidence.append(agent._compute_confidence(hybrid))
        correct.append(int(predicted == expected))

    X = np.array(raw_confidence, dtype=float)
    y = np.array(correct, dtype=int)
    n = len(X)
    print(f"Gold set: {n} profiles, {int(y.sum())} correctly classified "
          f"({y.mean():.3f} accuracy)")

    # ---- Honest, held-out evaluation via k-fold CV ----
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=CV_SEED)
    cv_calibrated_scores: list[float] = []
    cv_labels: list[int] = []
    fold_signs: list[bool] = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        fold_cal = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing="auto",
        )
        fold_cal.fit(X[train_idx], y[train_idx])
        cv_calibrated_scores.extend(fold_cal.predict(X[test_idx]).tolist())
        cv_labels.extend(y[test_idx].tolist())
        fold_signs.append(bool(fold_cal.increasing_))
        print(f"  fold {fold_i}/{args.cv_folds}: train n={len(train_idx)} "
              f"test n={len(test_idx)} increasing={fold_cal.increasing_}")

    if len(set(fold_signs)) > 1:
        print(
            "\n  !! WARNING: folds disagreed on the sign of the "
            "confidence-correctness relationship (increasing True in "
            f"some folds, False in others: {fold_signs}). The inversion "
            "may not be as stable across resamples as the headline "
            "AUROC=0.391 finding suggested — read the per-fold numbers "
            "above, not just the aggregate, before trusting this fix.\n"
        )

    auroc_raw = _auroc(raw_confidence, [bool(c) for c in correct])
    auroc_calibrated_cv = _auroc(cv_calibrated_scores, [bool(c) for c in cv_labels])

    # ---- Deployed calibrator: refit on ALL 200 points ----
    deployed = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing="auto",
    )
    deployed.fit(X, y)

    calibrator_path = settings.risk.confidence_calibrator_path
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrator_path, "wb") as f:
        pickle.dump({
            "calibrator": deployed,
            "fit_on": "gold_risk_profiles (rubric-labelled 200, "
                      "include_original_15=False)",
            "fit_n": n,
            "increasing": bool(deployed.increasing_),
            "cv_auroc_raw": round(auroc_raw, 4),
            "cv_auroc_calibrated": round(auroc_calibrated_cv, 4),
        }, f)

    # ---- Gate utility at the CURRENT threshold, before vs after ----
    threshold = settings.risk.min_confidence
    deployed_calibrated_all = deployed.predict(X).tolist()
    gate_before = _gate_utility(raw_confidence, correct, threshold)
    gate_after = _gate_utility(deployed_calibrated_all, correct, threshold)

    payload = {
        "phase": 3,
        "research_question": (
            "RQ1 — confidence recalibration (fix for the inverted "
            "calibration finding in rq1_confidence_calibration.json)"
        ),
        "agent": "RiskProfilingAgent",
        "requires_llm": False,
        "dataset": dataset.describe(),
        "method": {
            "calibrator": "sklearn.isotonic.IsotonicRegression(increasing='auto')",
            "input": "raw decision-margin confidence (_compute_confidence)",
            "output": "P(correct) — a genuine calibrated probability, "
                      "unlike the raw margin (see that method's own "
                      "docstring for why ECE was not reportable on it)",
            "cv_folds": args.cv_folds,
            "cv_seed": CV_SEED,
            "fold_signs_agree": len(set(fold_signs)) == 1,
            "deployed_calibrator_path": str(
                calibrator_path.relative_to(ROOT_DIR)
            ),
            "deployed_calibrator_fit_on": (
                "all 200 points (refit after CV validated the approach — "
                "see honest_evaluation_note)"
            ),
        },
        "headline_metrics": {
            "auroc_raw_confidence": round(auroc_raw, 4),
            "auroc_calibrated_confidence_cv_held_out": round(auroc_calibrated_cv, 4),
            "honest_evaluation_note": (
                "auroc_calibrated_confidence_cv_held_out is computed ONLY "
                "on points each fold's calibrator never trained on — not "
                "a number computed on the deployed calibrator's own "
                "training data, which would overstate the fix. Compare "
                "against 0.391 in rq1_confidence_calibration.json (below "
                "chance). Isotonic regression cannot manufacture "
                "discriminative signal that isn't in the raw score's "
                "rank-ordering — it can only correctly ORIENT and SHAPE "
                "whatever monotonic relationship already exists, so this "
                "is a ceiling-respecting improvement, not an inflated one."
            ),
        },
        "gate_utility_at_current_threshold": {
            "threshold": threshold,
            "before_raw_confidence": gate_before,
            "after_calibrated_confidence": gate_after,
            "note": (
                "before_raw_confidence reproduces "
                "rq1_confidence_calibration.json's approval_gate_utility "
                "finding (passes lower-accuracy predictions, holds back "
                "higher-accuracy ones). after_calibrated_confidence "
                "applies the SAME threshold value to the calibrated "
                "score — but calibrated confidence is now a genuine "
                "P(correct) estimate, so 0.6 means something different "
                "post-calibration than it did as a raw margin threshold. "
                "This script does not retune the threshold itself; "
                "compare the two accuracy splits here, then decide "
                "separately whether 0.6 is still the right cut for a "
                "calibrated probability."
            ),
        },
        "deployment": {
            "opt_in_setting": "settings.risk.use_calibrated_confidence "
                               "(env: USE_CALIBRATED_CONFIDENCE)",
            "currently_enabled": settings.risk.use_calibrated_confidence,
            "why_opt_in": (
                "Mirrors use_trained_model's existing opt-in pattern — "
                "this changes every confidence-dependent number in the "
                "system, so it should not activate just because this "
                "file now exists on disk. Set "
                "USE_CALIBRATED_CONFIDENCE=true once these CV numbers "
                "have been reviewed."
            ),
        },
    }

    path = write_results(payload, "rq1_confidence_recalibration.json")

    print(f"\n  AUROC raw confidence:            {auroc_raw:.3f}")
    print(f"  AUROC calibrated (CV held-out):  {auroc_calibrated_cv:.3f}")
    print(f"\n  gate @ {threshold} BEFORE: "
          f"passes n={gate_before['n_at_or_above']} "
          f"acc={gate_before['accuracy_at_or_above']}, "
          f"holds n={gate_before['n_below']} "
          f"acc={gate_before['accuracy_below']}")
    print(f"  gate @ {threshold} AFTER:  "
          f"passes n={gate_after['n_at_or_above']} "
          f"acc={gate_after['accuracy_at_or_above']}, "
          f"holds n={gate_after['n_below']} "
          f"acc={gate_after['accuracy_below']}")
    print(f"\n  calibrator saved to {calibrator_path.relative_to(ROOT_DIR)}")
    print(f"  currently "
          f"{'ENABLED' if settings.risk.use_calibrated_confidence else 'DISABLED'} "
          f"in production (settings.risk.use_calibrated_confidence)")
    print(f"  wrote {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())