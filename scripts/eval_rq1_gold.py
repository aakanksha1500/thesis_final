"""
scripts/eval_rq1_gold.py — RQ1 on the Layer-3 gold-standard set.

    python scripts/eval_rq1_gold.py
    python scripts/eval_rq1_gold.py --resamples 2000

WHAT THIS RUNS ON
    data/evaluation/gold_risk_profiles.json — 200 profiles (40 per risk
    class) labelled by evaluation/risk_rubric.py, plus the 15 original
    hand-labelled profiles kept separate and reported separately.

    Nothing else. The 254 operational customers (Layer 1) and the 1,000
    stress profiles (Layer 2) have no independent labels and are not
    touched by this script; evaluation/datasets.py would raise if they
    were. Their robustness numbers come from scripts/eval_rq1_stress.py.

NO LLM IS INVOLVED
    RiskProfilingAgent's classification path — _hybrid_score() and
    _score_to_class() — is fully deterministic: a trained RandomForest
    over four GMSC features, a coefficient-based preference score, and a
    rule layer, combined at fixed weights and cut into five equal-width
    bins. The LLM only writes the plain-English rationale afterwards.
    So these numbers are reproducible offline and identical in mock and
    real mode, which is why this is the one RQ that can be regenerated
    without an API key.

THREE PREDICTORS, NOT ONE
    hybrid     0.6 * ml + 0.4 * rule — what the system actually ships
    rule_only  the rule layer alone — the ablation the original RQ1
               evaluation already reported
    ml_only    the ML layer alone — added here because with 200 labelled
               profiles it becomes possible to say which component the
               hybrid's performance actually comes from, which n=15 could
               not support

THE DIAGNOSTIC THAT MATTERS FOR THE VIVA
    `rubric_vs_rule_layer_agreement` reports how often the labelling
    rubric and the agent's rule layer agree. Both encode the same
    regulatory constructs, so this will be well above chance, and hiding
    that would be worse than reporting it. Read it as: "here is how much
    shared structure exists between the instrument and one component of
    the system, stated up front." The ML layer shares nothing with the
    rubric, and the hybrid-vs-rule-only delta is what isolates it.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR, settings  # noqa: E402
from evaluation.bootstrap import (  # noqa: E402
    DEFAULT_RESAMPLES,
    bootstrap_paired_metric,
    detectable_effect_note,
)
from evaluation.datasets import load_gold_risk_profiles  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    accuracy_value,
    auc_roc,
    classification_report,
    f1_risk_classification,
    hybrid_vs_rule_only_delta,
    macro_f1_value,
    risk_alignment_rate,
)
from evaluation.results_io import write_results  # noqa: E402
from evaluation.risk_rubric import RISK_TIERS  # noqa: E402
from evaluation.risk_rubric import label as rubric_label


def _predict_all(agent, records: list[dict]) -> dict[str, list]:
    """Run the three deterministic predictors over every record."""
    out: dict[str, list] = {
        "hybrid": [], "rule_only": [], "ml_only": [],
        "hybrid_score": [], "ml_score": [], "rule_score": [],
    }
    for record in records:
        features = record["features"]
        ml, rule, hybrid = agent._hybrid_score(features)
        out["hybrid"].append(agent._score_to_class(hybrid))
        out["rule_only"].append(agent._score_to_class(rule))
        out["ml_only"].append(agent._score_to_class(ml))
        out["hybrid_score"].append(hybrid)
        out["ml_score"].append(ml)
        out["rule_score"].append(rule)
    return out


def _evaluate_block(
    name: str,
    records: list[dict],
    agent,
    resamples: int,
) -> dict[str, Any]:
    """Full metric block for one set of labelled records."""
    if not records:
        return {"name": name, "n": 0, "note": "no records"}

    truth = [r["ground_truth_risk_class"] for r in records]
    preds = _predict_all(agent, records)

    report = classification_report(preds["hybrid"], truth, labels=RISK_TIERS)
    rar = risk_alignment_rate(preds["hybrid"], truth)
    f1_legacy = f1_risk_classification(preds["hybrid"], truth)
    auc = auc_roc(preds["hybrid_score"], truth)
    delta = hybrid_vs_rule_only_delta(preds["hybrid"], preds["rule_only"], truth)

    cis = {
        "accuracy": bootstrap_paired_metric(
            preds["hybrid"], truth, accuracy_value, n_resamples=resamples,
        ).to_dict(),
        "macro_f1": bootstrap_paired_metric(
            preds["hybrid"], truth,
            lambda p, g: macro_f1_value(p, g, RISK_TIERS),
            n_resamples=resamples,
        ).to_dict(),
        "risk_alignment_rate": bootstrap_paired_metric(
            preds["hybrid"], truth,
            lambda p, g: risk_alignment_rate(p, g).value,
            n_resamples=resamples,
        ).to_dict(),
    }

    ablation = {
        variant: {
            "accuracy": round(accuracy_value(preds[variant], truth), 4),
            "macro_f1": round(macro_f1_value(preds[variant], truth, RISK_TIERS), 4),
            "risk_alignment_rate": risk_alignment_rate(preds[variant], truth).value,
        }
        for variant in ("hybrid", "rule_only", "ml_only")
    }

    # How much of the rubric's structure is shared with the rule layer.
    rubric_preds = [rubric_label(r["features"]).risk_class for r in records]
    rule_agreement = sum(
        a == b for a, b in zip(rubric_preds, preds["rule_only"])
    ) / len(records)
    ml_agreement = sum(
        a == b for a, b in zip(rubric_preds, preds["ml_only"])
    ) / len(records)

    # Score-distribution diagnostic. _score_to_class cuts [0,1] into five
    # equal-width bins, so a predictor whose scores cluster near 0.5 can
    # only ever reach the middle tiers regardless of how well it ranks
    # profiles. Reporting the distribution turns "low accuracy" into a
    # diagnosable statement about WHERE the scores live.
    import numpy as _np

    scores = _np.array(preds["hybrid_score"], dtype=float)
    bins = {
        "0.0-0.2 (conservative)": int(((scores >= 0.0) & (scores < 0.2)).sum()),
        "0.2-0.4 (moderately_conservative)": int(((scores >= 0.2) & (scores < 0.4)).sum()),
        "0.4-0.6 (moderate)": int(((scores >= 0.4) & (scores < 0.6)).sum()),
        "0.6-0.8 (moderately_aggressive)": int(((scores >= 0.6) & (scores < 0.8)).sum()),
        "0.8-1.0 (aggressive)": int((scores >= 0.8).sum()),
    }

    return {
        "name": name,
        "n": len(records),
        "class_distribution": dict(Counter(truth)),
        "predicted_class_distribution": dict(Counter(preds["hybrid"])),
        "score_distribution_diagnostic": {
            "hybrid_score_min": round(float(scores.min()), 4),
            "hybrid_score_max": round(float(scores.max()), 4),
            "hybrid_score_mean": round(float(scores.mean()), 4),
            "hybrid_score_std": round(float(scores.std()), 4),
            "counts_per_class_band": bins,
            "interpretation": (
                "_score_to_class cuts [0,1] into five equal-width bands. "
                "If the observed score range is narrower than [0,1], the "
                "outer tiers are unreachable in practice and per-class "
                "recall for conservative/aggressive is bounded above by "
                "the binning, not by the model's ranking ability. Compare "
                "the near-1.0 AUC-ROC (a ranking metric, threshold-free) "
                "with the per-class recall (threshold-dependent) to see "
                "whether the problem is ranking or calibration."
            ),
        },
        "metrics": {
            "classification_report": report.to_dict(),
            "risk_alignment_rate": rar.to_dict(),
            "f1_macro_legacy_metric": f1_legacy.to_dict(),
            "auc_roc": auc.to_dict(),
            "hybrid_vs_rule_only_delta": delta.to_dict(),
        },
        "confidence_intervals_95pct": cis,
        "component_ablation": ablation,
        "shared_structure_diagnostic": {
            "rubric_vs_rule_layer_agreement": round(rule_agreement, 4),
            "rubric_vs_ml_layer_agreement": round(ml_agreement, 4),
            "interpretation": (
                "Agreement between the labelling rubric and each component "
                "of the system, reported so the reader can see how much "
                "construct overlap exists rather than inferring it. The "
                "rubric and the rule layer both encode MiFID II / CBI "
                "suitability constructs, so their agreement is expected to "
                "be above chance (0.20 for 5 balanced classes). The ML "
                "layer is trained on GMSC distress data and shares no "
                "structure with the rubric."
            ),
        },
        "per_profile": [
            {
                "profile_id": r["profile_id"],
                "provenance": r["provenance"],
                "ground_truth": t,
                "hybrid_pred": hp,
                "rule_pred": rp,
                "ml_pred": mp,
                "hybrid_score": round(hs, 4),
                "correct": hp == t,
                "within_one_tier": abs(
                    RISK_TIERS.index(hp) - RISK_TIERS.index(t)
                ) <= 1,
            }
            for r, t, hp, rp, mp, hs in zip(
                records, truth, preds["hybrid"], preds["rule_only"],
                preds["ml_only"], preds["hybrid_score"],
            )
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = ap.parse_args()

    from agents.risk_profiling_agent import RiskProfilingAgent
    from utils.llm_client import LLMClient

    dataset = load_gold_risk_profiles(include_original_15=True)
    rubric_records = [
        r for r in dataset.records
        if r["provenance"] != "hand_labelled_original_15"
    ]
    original_records = [
        r for r in dataset.records
        if r["provenance"] == "hand_labelled_original_15"
    ]

    print(f"Gold set: {len(dataset)} profiles "
          f"({len(rubric_records)} rubric-labelled + "
          f"{len(original_records)} original hand-labelled)")

    agent = RiskProfilingAgent(LLMClient(force_mock=True))
    model_mode = "trained_model" if agent._ml_model is not None else "heuristic_proxy"
    print(f"  model mode: {model_mode}")

    primary = _evaluate_block(
        "gold_rubric_labelled_200", rubric_records, agent, args.resamples)
    original = _evaluate_block(
        "original_hand_labelled_15", original_records, agent, args.resamples)

    report = primary["metrics"]["classification_report"]["details"]
    ci = primary["confidence_intervals_95pct"]

    payload = {
        "phase": 3,
        "research_question": "RQ1",
        "agent": "RiskProfilingAgent",
        "model_mode": model_mode,
        "requires_llm": False,
        "dataset": dataset.describe(),
        "evaluation_configuration": {
            "predictor": "_hybrid_score -> _score_to_class (deterministic)",
            "ml_weight": settings.risk.ml_weight,
            "rule_weight": settings.risk.rule_weight,
            "capacity_weight": settings.risk.capacity_weight,
            "class_boundaries": [0.2, 0.4, 0.6, 0.8],
            "bootstrap_resamples": args.resamples,
            "confidence_level": 0.95,
            "sklearn_runtime_version": __import__("sklearn").__version__,
            "sklearn_pickle_version": (
                agent._ml_model.get("sklearn_version")
                if isinstance(agent._ml_model, dict) else None
            ),
        },
        "primary_result": primary,
        "original_15_result": original,
        "statistical_power": detectable_effect_note(len(rubric_records), 0.5),
        "methodological_guarantees": [
            "Labels were assigned by evaluation/risk_rubric.py before this "
            "script ran and were not modified afterwards.",
            "No profile was excluded on the basis of the model's output.",
            "No model prediction was used as ground truth anywhere.",
            "Layer-1 (real) and Layer-2 (stress) data are excluded from "
            "this file entirely — they have no independent labels.",
        ],
        "supersedes": (
            "phase3_risk_baseline.json (n=15, no confidence intervals, no "
            "confusion matrix, no per-class support)"
        ),
    }

    path = write_results(payload, "rq1_gold_risk_evaluation.json")

    print(f"\n  PRIMARY (n={primary['n']}, 40 per class)")
    print(f"    accuracy            {report['accuracy']:.3f}  "
          f"[{ci['accuracy']['ci_low']:.3f}, {ci['accuracy']['ci_high']:.3f}]")
    print(f"    macro F1            {report['macro_avg']['f1']:.3f}  "
          f"[{ci['macro_f1']['ci_low']:.3f}, {ci['macro_f1']['ci_high']:.3f}]")
    print(f"    weighted F1         {report['weighted_avg']['f1']:.3f}")
    print(f"    risk alignment      "
          f"{primary['metrics']['risk_alignment_rate']['value']:.3f}  "
          f"[{ci['risk_alignment_rate']['ci_low']:.3f}, "
          f"{ci['risk_alignment_rate']['ci_high']:.3f}]")
    print(f"    AUC-ROC (macro OVR) {primary['metrics']['auc_roc']['value']:.3f}")
    print("\n    per class:")
    for tier in RISK_TIERS:
        stats = report["per_class"][tier]
        print(f"      {tier:<26} P={stats['precision']:.3f} "
              f"R={stats['recall']:.3f} F1={stats['f1']:.3f} "
              f"n={stats['support']}")
    print("\n    component ablation:")
    for variant, stats in primary["component_ablation"].items():
        print(f"      {variant:<12} acc={stats['accuracy']:.3f} "
              f"macroF1={stats['macro_f1']:.3f} "
              f"RAR={stats['risk_alignment_rate']:.3f}")
    diag = primary["shared_structure_diagnostic"]
    print(f"\n    rubric vs rule layer agreement: "
          f"{diag['rubric_vs_rule_layer_agreement']:.3f}")
    print(f"    rubric vs ML layer agreement:   "
          f"{diag['rubric_vs_ml_layer_agreement']:.3f}")

    if original["n"]:
        orig_report = original["metrics"]["classification_report"]["details"]
        print("\n  ORIGINAL 15 (hand-labelled, reported separately)")
        print(f"    accuracy {orig_report['accuracy']:.3f}   "
              f"macro F1 {orig_report['macro_avg']['f1']:.3f}   "
              f"RAR {original['metrics']['risk_alignment_rate']['value']:.3f}")

    print(f"\n  wrote {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
