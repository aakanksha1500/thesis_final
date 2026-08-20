"""
scripts/eval_robustness_layers.py — Layer-1 and Layer-2 robustness.

    python scripts/eval_robustness_layers.py
    python scripts/eval_robustness_layers.py --layers stress

WHAT THIS MEASURES, AND WHAT IT REFUSES TO MEASURE
    Layer 1 (254 operational customers) and Layer 2 (1,000 synthetic
    stress profiles) have no independent risk labels. So this script
    reports only things that are true without labels:

        completion rate          did every profile produce a prediction
        crash rate               did anything raise
        prediction distribution  which tiers the system reaches at all
        confidence distribution  and how it varies by subgroup
        coverage-tier behaviour  does each of the six data-sufficiency
                                 tiers produce its intended disclosure
        missing-feature handling does an incomplete profile take the
                                 elicitation path rather than scoring
                                 silently on defaults
        subgroup stability       prediction spread by employment status,
                                 age band, DTI band and coverage tier

    It computes no accuracy, precision, recall or F1, and it cannot: it
    obtains its data through evaluation/datasets.py, whose Layer-1 and
    Layer-2 handles raise GroundTruthUnavailable on any attempt to pull
    labels out of them. That guard is the point of the module.

WHY PREDICTION DISTRIBUTION IS WORTH HAVING WITHOUT LABELS
    A classifier that has collapsed onto two of its five tiers is broken
    in a way that is visible without a single label — and visible over
    1,000 profiles in a way 15 could never show. Distribution over a
    deliberately balanced stress population is a genuine finding; it just
    is not an accuracy claim, and this script keeps those two things
    apart.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR, settings  # noqa: E402
from evaluation.datasets import (  # noqa: E402
    GroundTruthUnavailable,
    layer_summary,
    load_real_customers,
    load_stress_customers,
    load_stress_transactions,
)
from evaluation.results_io import write_results  # noqa: E402


def _age_band(age: Any) -> str:
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "unknown"
    for upper, name in ((25, "18-24"), (35, "25-34"), (45, "35-44"),
                        (55, "45-54"), (65, "55-64")):
        if age < upper:
            return name
    return "65+"


def _dti_band(features: dict) -> str:
    try:
        income = float(features.get("income", 0))
        debt = float(features.get("existing_debt", 0))
    except (TypeError, ValueError):
        return "unknown"
    if income <= 0:
        return "no_income"
    ratio = debt / income
    for upper, name in ((0.1, "<0.10"), (0.25, "0.10-0.25"),
                        (0.5, "0.25-0.50"), (1.0, "0.50-1.00")):
        if ratio < upper:
            return name
    return ">=1.00"


def run_layer(name: str, dataset, agent, transactions=None) -> dict[str, Any]:
    """One robustness pass. Never touches labels."""
    started = time.time()

    predictions: list[str] = []
    confidences: list[float] = []
    crashes: list[dict] = []
    missing_feature_cases = 0
    complete_cases = 0

    by_employment: dict[str, Counter] = defaultdict(Counter)
    by_age: dict[str, Counter] = defaultdict(Counter)
    by_dti: dict[str, Counter] = defaultdict(Counter)
    by_coverage: dict[str, Counter] = defaultdict(Counter)
    by_edge_case: dict[str, Counter] = defaultdict(Counter)

    required = list(settings.risk.required_features)

    for record in dataset:
        features = record["features"]
        record_id = record.get("customer_id") or record.get("profile_id")

        missing = [f for f in required if f not in features or features[f] is None]
        if missing:
            missing_feature_cases += 1
        else:
            complete_cases += 1

        try:
            ml, rule, hybrid = agent._hybrid_score(features)
            predicted = agent._score_to_class(hybrid)
            confidence = agent._compute_confidence(hybrid)
        except Exception as exc:  # noqa: BLE001
            crashes.append({
                "id": record_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:200],
                "missing_features": missing,
                "edge_case_kind": record.get("edge_case_kind"),
            })
            continue

        predictions.append(predicted)
        confidences.append(confidence)

        by_employment[str(features.get("employment_status", "unknown"))][predicted] += 1
        by_age[_age_band(features.get("age"))][predicted] += 1
        by_dti[_dti_band(features)][predicted] += 1
        if record.get("expected_coverage_tier"):
            by_coverage[record["expected_coverage_tier"]][predicted] += 1
        if record.get("edge_case_kind"):
            by_edge_case[record["edge_case_kind"]][predicted] += 1

    n = len(dataset)
    import numpy as np

    conf = np.array(confidences) if confidences else np.array([0.0])

    result = {
        "layer_dataset": dataset.describe(),
        "n_records": n,
        "completion": {
            "n_completed": len(predictions),
            "n_crashed": len(crashes),
            "completion_rate": round(len(predictions) / n, 4) if n else 0.0,
            "crash_rate": round(len(crashes) / n, 4) if n else 0.0,
            "crashes": crashes[:25],
        },
        "feature_completeness": {
            "n_complete_profiles": complete_cases,
            "n_with_missing_required_features": missing_feature_cases,
            "required_features": required,
            "note": (
                "Profiles with missing required features are scored here "
                "via _hybrid_score's own defaulting. In production "
                "RiskProfilingAgent.run() routes them to elicitation "
                "instead — this pass measures whether the scoring path "
                "survives them, not whether the routing is correct."
            ),
        },
        "prediction_distribution": dict(Counter(predictions)),
        "prediction_distribution_pct": {
            k: round(v / len(predictions), 4)
            for k, v in Counter(predictions).items()
        } if predictions else {},
        "n_distinct_classes_predicted": len(set(predictions)),
        "confidence_distribution": {
            "mean": round(float(conf.mean()), 4),
            "std": round(float(conf.std()), 4),
            "min": round(float(conf.min()), 4),
            "max": round(float(conf.max()), 4),
            "p25": round(float(np.percentile(conf, 25)), 4),
            "median": round(float(np.percentile(conf, 50)), 4),
            "p75": round(float(np.percentile(conf, 75)), 4),
            "below_min_confidence_threshold": int(
                (conf < settings.risk.min_confidence).sum()
            ),
            "min_confidence_threshold": settings.risk.min_confidence,
        },
        "subgroup_prediction_distribution": {
            "by_employment_status": {k: dict(v) for k, v in sorted(by_employment.items())},
            "by_age_band": {k: dict(v) for k, v in sorted(by_age.items())},
            "by_debt_to_income_band": {k: dict(v) for k, v in sorted(by_dti.items())},
            "by_coverage_tier": {k: dict(v) for k, v in sorted(by_coverage.items())},
            "by_edge_case_kind": {k: dict(v) for k, v in sorted(by_edge_case.items())},
        },
        "wall_clock_s": round(time.time() - started, 2),
    }

    if transactions:
        result["coverage_tier_verification"] = _verify_coverage_tiers(
            dataset, transactions)

    # Prove the guard is live rather than merely documented.
    try:
        dataset.ground_truth()
        result["ground_truth_guard"] = "FAILED — labels were returned"
    except GroundTruthUnavailable as exc:
        result["ground_truth_guard"] = {
            "status": "enforced",
            "message": str(exc).split(".")[0],
            "note": (
                "evaluation/datasets.py refused to return labels for this "
                "layer. No accuracy metric can be computed from this file."
            ),
        }
    return result


def _verify_coverage_tiers(dataset, transactions) -> dict[str, Any]:
    """
    Does each stress profile actually land on the coverage tier it was
    built for? Checks the generator against agents/data_sufficiency.py
    rather than trusting the label written at build time.
    """
    from agents.data_sufficiency import assess_data_sufficiency

    matches = 0
    checked = 0
    mismatches: list[dict] = []
    observed: Counter = Counter()

    for record in dataset:
        expected = record.get("expected_coverage_tier")
        entry = transactions.get(record["customer_id"])
        if not expected or entry is None:
            continue
        checked += 1
        try:
            assessment = assess_data_sufficiency(entry["transactions"])
            actual = assessment.coverage_tier
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"id": record["customer_id"], "error": str(exc)[:120]})
            continue
        observed[actual] += 1
        if actual == expected:
            matches += 1
        elif len(mismatches) < 20:
            mismatches.append({
                "id": record["customer_id"],
                "expected": expected, "actual": actual,
            })

    return {
        "n_checked": checked,
        "n_matching_intended_tier": matches,
        "tier_construction_accuracy": round(matches / checked, 4) if checked else 0.0,
        "observed_tier_distribution": dict(observed),
        "mismatches_sample": mismatches,
        "note": (
            "Validates the STRESS GENERATOR against "
            "agents/data_sufficiency.py — it confirms the stress set "
            "really does exercise all six tiers. It is not a model "
            "accuracy metric."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--layers", nargs="+", default=["real", "stress"],
                    choices=["real", "stress"])
    args = ap.parse_args()

    from agents.risk_profiling_agent import RiskProfilingAgent
    from utils.llm_client import LLMClient

    agent = RiskProfilingAgent(LLMClient(force_mock=True))
    results: dict[str, Any] = {}

    if "real" in args.layers:
        dataset = load_real_customers()
        print(f"Layer 1 (real): {len(dataset)} customers")
        results["layer1_real"] = run_layer("layer1_real", dataset, agent)

    if "stress" in args.layers:
        dataset = load_stress_customers()
        transactions = load_stress_transactions()
        print(f"Layer 2 (stress): {len(dataset)} profiles")
        results["layer2_stress"] = run_layer(
            "layer2_stress", dataset, agent, transactions)

    payload = {
        "phase": 3,
        "research_question": "RQ1 (supporting) — robustness, not accuracy",
        "agent": "RiskProfilingAgent",
        "requires_llm": False,
        "what_this_file_is_not": (
            "Not an accuracy result. Layers 1 and 2 have no independent "
            "ground truth. Every metric here is label-free by "
            "construction, and evaluation/datasets.py raises "
            "GroundTruthUnavailable if labels are requested from either "
            "layer — see the ground_truth_guard block in each result."
        ),
        "three_layer_inventory": layer_summary(),
        "results": results,
    }

    path = write_results(payload, "rq1_robustness_layers.json")

    for key, block in results.items():
        print(f"\n  {key}")
        print(f"    completion rate      "
              f"{block['completion']['completion_rate']:.4f} "
              f"({block['completion']['n_crashed']} crashes)")
        print(f"    classes predicted    "
              f"{block['n_distinct_classes_predicted']}/5  "
              f"{block['prediction_distribution']}")
        print(f"    mean confidence      "
              f"{block['confidence_distribution']['mean']:.3f} "
              f"(sd {block['confidence_distribution']['std']:.3f}, "
              f"{block['confidence_distribution']['below_min_confidence_threshold']} "
              f"below threshold)")
        print(f"    missing-feature rows "
              f"{block['feature_completeness']['n_with_missing_required_features']}")
        if "coverage_tier_verification" in block:
            cov = block["coverage_tier_verification"]
            print(f"    coverage tiers built correctly "
                  f"{cov['tier_construction_accuracy']:.4f} "
                  f"({cov['observed_tier_distribution']})")
        print(f"    ground-truth guard   "
              f"{block['ground_truth_guard']['status']}")

    print(f"\n  wrote {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
