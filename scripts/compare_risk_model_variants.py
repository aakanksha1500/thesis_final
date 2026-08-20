"""
Day 1b — compare the heuristic-only risk classifier against the trained
RandomForest-backed hybrid classifier on the same 15-profile RQ1 fixture.

WHY THIS EXISTS
    "Run RQ1 both ways and keep both files" only produces two JSON blobs
    sitting in a folder. Nothing reads them side by side. This is the
    script that turns them into the actual heuristic-vs-trained
    comparison RQ1 is supposed to answer — headline metrics side by
    side, the delta, and every profile where the two disagree.

USAGE
    # 1. Generate both files first (see README below or Day 1b in
    #    BUILD_PLAN.md) — this script only reads, it doesn't run agents.
    # 2. Then:
    python scripts/compare_risk_model_variants.py

    # Non-default paths / provider directory:
    python scripts/compare_risk_model_variants.py \\
        --heuristic results/groq/phase3_risk_heuristic.json \\
        --trained   results/groq/phase3_risk_trained.json

GENERATING THE TWO INPUT FILES
    USE_TRAINED_RISK_MODEL=false LLM_CACHE=record \\
      python scripts/regenerate_evidence.py --only rq1
    mv results/groq/phase3_risk_baseline.json results/groq/phase3_risk_heuristic.json

    USE_TRAINED_RISK_MODEL=true LLM_CACHE=record \\
      python scripts/regenerate_evidence.py --only rq1
    mv results/groq/phase3_risk_baseline.json results/groq/phase3_risk_trained.json

    Both runs are LLM-free (test_rq1_full_fixture_evaluation_and_write_results
    only exercises RiskProfilingAgent._hybrid_score, which is deterministic
    feature scoring — no API calls, no cost, either way).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "results" / "groq"

# (results_payload["metrics"] key, display label)
METRICS: list[tuple[str, str]] = [
    ("risk_alignment_rate", "Risk Alignment Rate"),
    ("f1_macro", "Macro F1"),
    ("auc_roc", "AUC-ROC (macro OVR)"),
    ("hybrid_vs_rule_only_delta", "Hybrid vs Rule-only delta"),
]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Missing: {path}\n"
            f"Generate it first — see the module docstring at the top of "
            f"this file, or Day 1b in BUILD_PLAN.md."
        )
    return json.loads(path.read_text())


def compare(heuristic_path: Path, trained_path: Path) -> dict[str, Any]:
    h = _load(heuristic_path)
    t = _load(trained_path)

    # Sanity check: catch "forgot to flip the env var before the second
    # run" immediately rather than silently comparing two identical runs.
    if h.get("model_mode") != "heuristic_proxy":
        print(
            f"  WARNING: {heuristic_path.name} has "
            f"model_mode={h.get('model_mode')!r}, expected "
            f"'heuristic_proxy'. Was USE_TRAINED_RISK_MODEL actually "
            f"false when this ran?"
        )
    if t.get("model_mode") != "trained_model":
        print(
            f"  WARNING: {trained_path.name} has "
            f"model_mode={t.get('model_mode')!r}, expected "
            f"'trained_model'. Was USE_TRAINED_RISK_MODEL actually "
            f"true when this ran?"
        )

    rows = []
    for key, label in METRICS:
        hv = h["metrics"][key]["value"]
        tv = t["metrics"][key]["value"]
        rows.append({
            "metric": label,
            "heuristic": hv,
            "trained": tv,
            "delta": round(tv - hv, 4),
        })

    # Per-profile disagreements. Fixture order is fixed (RQ1_FIXTURE in
    # test_risk_profiling_agent.py), so zip-by-position is safe as long
    # as the fixture itself hasn't changed between the two runs.
    h_profiles = h.get("per_profile", [])
    t_profiles = t.get("per_profile", [])
    disagreements = []
    if len(h_profiles) == len(t_profiles):
        for hp, tp in zip(h_profiles, t_profiles):
            if hp["hybrid_pred"] != tp["hybrid_pred"]:
                disagreements.append({
                    "expected": hp["expected"],
                    "heuristic_pred": hp["hybrid_pred"],
                    "trained_pred": tp["hybrid_pred"],
                    "features": hp["features"],
                })
    else:
        print(
            "  WARNING: fixture length differs between the two files "
            f"({len(h_profiles)} vs {len(t_profiles)}) — skipping "
            "per-profile comparison."
        )

    return {
        "circularity_and_supersession_warning": (
            "SUPERSEDED — both inputs to this comparison come from the "
            "same 15-item fixture (RQ1_FIXTURE) and the same author's "
            "scoring, so neither run is independent validation of the "
            "other, and this comparison is a diagnostic of how the two "
            "scorers disagree, not a reportable RQ1 result. Cite "
            "rq1_gold_risk_evaluation.json (n=200, independently "
            "rubric-labelled) for RQ1 itself."
        ),
        "metrics": rows,
        "disagreements": disagreements,
        "heuristic_meta": h.get("_meta", {}),
        "trained_meta": t.get("_meta", {}),
    }


def print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("  RQ1 — heuristic vs trained-model comparison")
    print("=" * 64)
    print(f"\n  !! {result['circularity_and_supersession_warning']}")

    print(f"\n  {'Metric':<28}{'Heuristic':>12}{'Trained':>12}{'Δ':>10}")
    for row in result["metrics"]:
        print(
            f"  {row['metric']:<28}{row['heuristic']:>12.3f}"
            f"{row['trained']:>12.3f}{row['delta']:>+10.3f}"
        )
    print(
        "\n  Note: the 'Hybrid vs Rule-only delta' row is itself a delta "
        "metric (hybrid score minus rule-only score, within a single "
        "run) — its own Δ column above is the second-order comparison "
        "of that metric across the two runs."
    )

    n = len(result["disagreements"])
    print(f"\n  Disagreements: {n} / 15 profiles")
    for d in result["disagreements"]:
        print(
            f"    expected={d['expected']:<24} "
            f"heuristic={d['heuristic_pred']:<24} "
            f"trained={d['trained_pred']}"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--heuristic", type=Path,
        default=DEFAULT_DIR / "phase3_risk_heuristic.json",
    )
    ap.add_argument(
        "--trained", type=Path,
        default=DEFAULT_DIR / "phase3_risk_trained.json",
    )
    ap.add_argument(
        "--out", type=Path,
        default=DEFAULT_DIR / "phase3_risk_comparison.json",
        help="Where to write the machine-readable comparison",
    )
    args = ap.parse_args()

    result = compare(args.heuristic, args.trained)
    print_report(result)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"  Written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())