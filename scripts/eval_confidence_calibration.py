"""
scripts/eval_confidence_calibration.py — is the system's confidence honest?

    python scripts/eval_confidence_calibration.py

WHY THIS EXISTS
    The literature review's Gap 4 argues, from Takayanagi et al. and Li
    et al., that a financial AI can pass every accuracy check while
    producing mis-calibrated trust — users trusting it in the scenario
    types it handles worst — and that this failure is "invisible to
    current evaluation practice". Decision X3 makes calibrated trust an
    explicit design goal: explanations should produce trust proportional
    to actual advice quality, not maximised satisfaction.

    Until now that was a cited claim. With the Layer-3 gold set it
    becomes measurable on this system, because for the first time there
    are 200 independently-labelled profiles where correctness is known.

    The question this script answers: WHEN THE MODEL IS WRONG, DOES IT
    SAY SO? A system with 0.38 accuracy that reliably flags its own
    errors is, under X3, behaving better than one with 0.60 accuracy and
    flat confidence — because the first can be governed by an approval
    gate and the second cannot.

WHAT IT REPORTS
    discrimination   AUROC of confidence as a predictor of correctness.
                     0.5 = confidence is noise; 1.0 = confidence
                     perfectly separates right from wrong answers. This
                     is the headline: it is threshold-free and it is the
                     property X3 actually needs.
    separation       mean confidence when correct minus mean confidence
                     when incorrect, with a bootstrap CI. If the interval
                     excludes 0, the system's self-assessment carries
                     real signal.
    gate utility     accuracy above vs below settings.risk.min_confidence
                     (0.6). This is the operational number: it says what
                     the approval gate actually buys.
    reliability      binned confidence vs observed accuracy, so
                     over/under-confidence is visible per band.

    NOTE ON WHY THERE IS NO ECE HEADLINE
    _compute_confidence returns distance from the nearest class boundary,
    normalised to [0,1]. It is a decision-margin, NOT a predicted
    probability of correctness, so Expected Calibration Error — which
    assumes the score is a probability — would be a category error. The
    reliability table is reported for inspection; the headline metrics
    are discrimination and separation, which are meaningful for a margin.

NO LLM, NO API KEY
    Everything here is deterministic, like RQ1 itself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR, settings  # noqa: E402
from evaluation.bootstrap import DEFAULT_RESAMPLES, bootstrap_metric  # noqa: E402
from evaluation.datasets import load_gold_risk_profiles  # noqa: E402
from evaluation.results_io import write_results  # noqa: E402
from evaluation.risk_rubric import RISK_TIERS  # noqa: E402


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """
    AUROC via the Mann-Whitney U identity: the probability that a randomly
    chosen correct prediction carries higher confidence than a randomly
    chosen incorrect one. Ties count half.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos for n in neg
    )
    return wins / (len(pos) * len(neg))


def _reliability_table(
    confidences: list[float], correct: list[bool], n_bins: int = 5
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = [
            (c, k) for c, k in zip(confidences, correct)
            if (c >= lo and c < hi) or (hi == 1.0 and c == 1.0)
        ]
        if not in_bin:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": len(in_bin),
            "mean_confidence": round(float(np.mean([c for c, _ in in_bin])), 4),
            "observed_accuracy": round(float(np.mean([k for _, k in in_bin])), 4),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = ap.parse_args()

    from agents.risk_profiling_agent import RiskProfilingAgent
    from utils.llm_client import LLMClient

    dataset = load_gold_risk_profiles(include_original_15=False)
    agent = RiskProfilingAgent(LLMClient(force_mock=True))

    truth = dataset.ground_truth()
    confidences: list[float] = []
    correct: list[bool] = []
    within_one: list[bool] = []

    for record, expected in zip(dataset.records, truth):
        _ml, _rule, hybrid = agent._hybrid_score(record["features"])
        predicted = agent._score_to_class(hybrid)
        confidences.append(agent._compute_confidence(hybrid))
        correct.append(predicted == expected)
        within_one.append(
            abs(RISK_TIERS.index(predicted) - RISK_TIERS.index(expected)) <= 1
        )

    auroc_exact = _auroc(confidences, correct)
    auroc_within = _auroc(confidences, within_one)

    conf_correct = [c for c, k in zip(confidences, correct) if k]
    conf_wrong = [c for c, k in zip(confidences, correct) if not k]

    pairs = list(zip(confidences, correct))

    def _separation(sample: list[tuple[float, bool]]) -> float:
        yes = [c for c, k in sample if k]
        no = [c for c, k in sample if not k]
        if not yes or not no:
            raise ValueError("degenerate resample")
        return float(np.mean(yes) - np.mean(no))

    separation_ci = bootstrap_metric(
        pairs, _separation, n_resamples=args.resamples)

    threshold = settings.risk.min_confidence
    above = [k for c, k in zip(confidences, correct) if c >= threshold]
    below = [k for c, k in zip(confidences, correct) if c < threshold]

    payload = {
        "phase": 3,
        "research_question": "RQ1 / RQ3 supporting — trust calibration (Gap 4, X3)",
        "agent": "RiskProfilingAgent",
        "requires_llm": False,
        "dataset": dataset.describe(),
        "question": (
            "When the risk classifier is wrong, does its stated confidence "
            "fall? Decision X3 requires trust proportional to advice "
            "quality; this measures whether the system's own signal "
            "supports that."
        ),
        "headline_metrics": {
            "auroc_confidence_predicts_exact_correctness": round(auroc_exact, 4),
            "auroc_confidence_predicts_within_one_tier": round(auroc_within, 4),
            "mean_confidence_when_correct": round(float(np.mean(conf_correct)), 4) if conf_correct else None,
            "mean_confidence_when_incorrect": round(float(np.mean(conf_wrong)), 4) if conf_wrong else None,
            "separation": separation_ci.to_dict(),
            "separation_significant_at_95pct": not (
                separation_ci.ci_low <= 0.0 <= separation_ci.ci_high
            ),
        },
        "interpretation_guide": {
            "auroc_0.5": "confidence is noise — the system cannot tell when it is wrong",
            "auroc_0.6_to_0.7": "weak but real signal",
            "auroc_above_0.7": "confidence is a usable governance signal",
            "why_it_matters": (
                "Under X3 a low-accuracy system that reliably flags its own "
                "errors is preferable to a higher-accuracy system with flat "
                "confidence: only the first can be governed by an approval "
                "gate or a disclosure. This metric decides which one this is."
            ),
        },
        "approval_gate_utility": {
            "threshold": threshold,
            "n_at_or_above": len(above),
            "accuracy_at_or_above": round(float(np.mean(above)), 4) if above else None,
            "n_below": len(below),
            "accuracy_below": round(float(np.mean(below)), 4) if below else None,
            "note": (
                "settings.risk.min_confidence is already used as a gate in "
                "production. This is what that gate buys: accuracy on the "
                "predictions it lets through versus the ones it holds back."
            ),
        },
        "reliability_table": _reliability_table(confidences, correct),
        "confidence_distribution": {
            "mean": round(float(np.mean(confidences)), 4),
            "std": round(float(np.std(confidences)), 4),
            "min": round(float(np.min(confidences)), 4),
            "max": round(float(np.max(confidences)), 4),
        },
        "n": len(confidences),
        "n_correct": int(sum(correct)),
        "methodological_note": (
            "Confidence and correctness are computed on the Layer-3 gold "
            "set only. Correctness requires independent labels, so this "
            "analysis cannot be run on Layers 1 or 2."
        ),
    }

    path = write_results(payload, "rq1_confidence_calibration.json")

    print(f"Confidence calibration on {len(confidences)} gold profiles "
          f"({sum(correct)} correct)\n")
    print(f"  AUROC (exact correctness)     {auroc_exact:.3f}")
    print(f"  AUROC (within one tier)       {auroc_within:.3f}")
    print(f"  mean confidence when correct  "
          f"{np.mean(conf_correct):.3f}" if conf_correct else "  n/a")
    print(f"  mean confidence when wrong    "
          f"{np.mean(conf_wrong):.3f}" if conf_wrong else "  n/a")
    print(f"  separation                    {separation_ci.format()}  "
          f"{'significant' if payload['headline_metrics']['separation_significant_at_95pct'] else 'NOT significant'}")
    print(f"\n  gate at confidence >= {threshold}:")
    print(f"    passes: n={len(above):<4} accuracy="
          f"{np.mean(above):.3f}" if above else "    passes: none")
    print(f"    held:   n={len(below):<4} accuracy="
          f"{np.mean(below):.3f}" if below else "    held: none")
    print("\n  reliability:")
    for row in payload["reliability_table"]:
        if row["n"]:
            print(f"    {row['bin']}  n={row['n']:<4} "
                  f"mean_conf={row['mean_confidence']:.3f}  "
                  f"observed_acc={row['observed_accuracy']:.3f}")
    print(f"\n  wrote {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())