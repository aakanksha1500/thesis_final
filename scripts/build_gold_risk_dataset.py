"""
scripts/build_gold_risk_dataset.py — builds the Layer-3 gold-standard set.

    python scripts/build_gold_risk_dataset.py
    python scripts/build_gold_risk_dataset.py --per-class 40 --seed 20260812

WHAT IT PRODUCES
    data/evaluation/gold_risk_profiles.json
        200 profiles, 40 per risk class, each carrying the rubric
        arithmetic that assigned its label, PLUS the 15 original
        hand-labelled profiles preserved verbatim with their own
        provenance tag and their own label untouched.

LABEL-FIRST CONSTRUCTION, AND WHY IT IS NOT CIRCULAR
    The naive way to build a balanced evaluation set is to generate
    profiles, run the model, and keep 40 of each predicted class. That
    produces a set on which the model scores 100% by construction, and
    it is the single most common way an evaluation gets quietly
    invalidated.

    This script never runs the model. For each target class it:
      1. picks a (capacity_band, tolerance_band) cell of the suitability
         matrix that maps to that class;
      2. samples a demographically coherent profile from broad priors;
      3. asks evaluation/risk_rubric.py — which imports nothing from
         agents/ — what that profile's bands are;
      4. keeps the profile only if the bands land in the intended cell.

    Step 3 is a CHECK, not a fit. A profile is accepted or discarded on
    the rubric's arithmetic alone; whether the model would agree is
    unknown at build time and irrelevant to acceptance. Rejection
    sampling from broad priors is what keeps within-class diversity —
    the accepted profiles vary in age, income, employment, dependents,
    debt ratio and horizon, they are not one template with the numbers
    nudged.

DEMOGRAPHIC COHERENCE
    Profiles are constrained to be internally plausible before the rubric
    ever sees them: nobody aged 70 is a student, retirees do not have
    45-year investment horizons, and horizon is drawn as a function of
    remaining working life. An incoherent profile is not a hard test
    case, it is a nonsense one, and a supervisor is entitled to ask what
    a label on it would even mean.

SEED AND REPRODUCIBILITY
    One fixed seed (default 20260812), one numpy Generator, no global RNG
    use. Re-running this script on any machine reproduces the same 200
    profiles byte-for-byte. The output file records the seed, the rubric
    version and the git SHA at build time.

IF YOU EVER CHANGE THE RUBRIC
    Rebuild this file from scratch and discard every result computed
    against the old version. Do not re-label existing profiles: that is
    the path by which labels start drifting toward whatever the model
    happens to do.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR  # noqa: E402
from evaluation.risk_rubric import (  # noqa: E402
    CELLS_BY_TIER,
    RISK_TIERS,
    RUBRIC_VERSION,
    label,
    rubric_documentation,
)

OUTPUT_PATH = ROOT_DIR / "data" / "evaluation" / "gold_risk_profiles.json"
ORIGINAL_FIXTURE_FILE = ROOT_DIR / "tests" / "unit" / "test_risk_profiling_agent.py"

DEFAULT_SEED = 20260812
DEFAULT_PER_CLASS = 40

TIER_CODE = {
    "conservative": "CONS",
    "moderately_conservative": "MCON",
    "moderate": "MODT",
    "moderately_aggressive": "MAGG",
    "aggressive": "AGGR",
}

EMPLOYMENT_BY_AGE: dict[str, tuple[int, int]] = {
    # employment_status -> (min_age, max_age) it is plausible at
    "student": (18, 30),
    "employed": (18, 68),
    "self_employed": (22, 70),
    "self-employed": (22, 70),
    "part_time": (18, 70),
    "unemployed": (18, 64),
    "retired": (60, 85),
}


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def load_original_15() -> list[dict]:
    """
    Read RQ1_FIXTURE out of the original test file by parsing the source,
    not by importing it.

    Importing a test module drags in pytest's import mode, the rootdir
    calculation and conftest side effects — the exact fragility
    tests/unit/test_core_intent_evaluation.py already documents having
    been bitten by. ast.literal_eval on the assignment node has none of
    those dependencies and cannot execute anything.
    """
    if not ORIGINAL_FIXTURE_FILE.exists():
        print(f"  ! {ORIGINAL_FIXTURE_FILE} not found — skipping original 15")
        return []

    tree = ast.parse(ORIGINAL_FIXTURE_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "RQ1_FIXTURE":
                value = node.value if isinstance(node, ast.Assign) else node.value
                return ast.literal_eval(value)
    return []


def _sample_profile(rng: np.random.Generator) -> dict:
    """
    One demographically coherent profile drawn from broad priors.

    Deliberately broad: the rubric filter downstream is what selects the
    target cell, so widening these priors widens within-class diversity
    at the cost of a few more rejected draws, which are free.
    """
    age = int(rng.integers(18, 81))

    plausible = [
        emp for emp, (lo, hi) in EMPLOYMENT_BY_AGE.items()
        if lo <= age <= hi
    ]
    # Both spellings of self-employed are in the pool on purpose: the
    # agent's rule layer branches on both, so the gold set must contain
    # both to exercise that branch.
    employment = str(rng.choice(plausible))

    if employment == "student":
        income = float(np.round(rng.uniform(8_000, 24_000), 2))
    elif employment == "unemployed":
        income = float(np.round(rng.uniform(10_000, 26_000), 2))
    elif employment == "retired":
        income = float(np.round(rng.uniform(14_000, 70_000), 2))
    elif employment == "part_time":
        income = float(np.round(rng.uniform(12_000, 45_000), 2))
    else:
        income = float(np.round(rng.uniform(20_000, 175_000), 2))

    dependents = int(rng.integers(0, 6))
    debt_ratio = float(rng.uniform(0.0, 0.95))
    existing_debt = float(np.round(income * debt_ratio, 2))

    # Horizon is bounded by remaining working/investing life, then
    # widened by a uniform draw so it is not a deterministic function of
    # age (which would make age and horizon collinear across the set).
    max_horizon = int(np.clip(78 - age, 1, 40))
    investment_horizon = int(rng.integers(1, max_horizon + 1))

    loss_tolerance = int(rng.integers(1, 6))
    financial_knowledge_score = int(rng.integers(1, 6))

    return {
        "age": age,
        "income": income,
        "employment_status": employment,
        "dependents": dependents,
        "existing_debt": existing_debt,
        "investment_horizon": investment_horizon,
        "loss_tolerance": loss_tolerance,
        "financial_knowledge_score": financial_knowledge_score,
    }


def build_gold_profiles(per_class: int, seed: int) -> tuple[list[dict], dict]:
    """Rejection-sample `per_class` profiles for each of the five tiers."""
    rng = np.random.default_rng(seed)

    profiles: list[dict] = []
    seen: set[tuple] = set()
    stats: dict[str, dict] = {}

    for tier in RISK_TIERS:
        cells = sorted(CELLS_BY_TIER[tier])
        # Split the quota evenly across every matrix cell that produces
        # this tier, so a two-cell tier is not silently 40 copies of the
        # easier cell to hit.
        quotas = [per_class // len(cells)] * len(cells)
        for i in range(per_class - sum(quotas)):
            quotas[i] += 1

        tier_stats = {"attempts": 0, "accepted": 0, "by_cell": {}}

        for cell, quota in zip(cells, quotas):
            accepted = 0
            attempts = 0
            max_attempts = quota * 20_000

            while accepted < quota and attempts < max_attempts:
                attempts += 1
                features = _sample_profile(rng)
                verdict = label(features)

                if (verdict.capacity_band, verdict.tolerance_band) != cell:
                    continue
                if verdict.risk_class != tier:  # defensive: matrix drift
                    continue

                key = tuple(sorted(features.items()))
                if key in seen:
                    continue
                seen.add(key)

                accepted += 1
                idx = len(profiles) + 1
                profiles.append({
                    "profile_id": f"GOLD_{TIER_CODE[tier]}_{accepted:03d}",
                    "sequence": idx,
                    "features": features,
                    "ground_truth_risk_class": tier,
                    "provenance": "rubric_labelled_v" + RUBRIC_VERSION,
                    "label_source": "evaluation/risk_rubric.py",
                    "label_assigned_before_model_run": True,
                    "construction_cell": {
                        "capacity_band": cell[0],
                        "tolerance_band": cell[1],
                    },
                    "rubric": verdict.to_dict(),
                })

            tier_stats["attempts"] += attempts
            tier_stats["accepted"] += accepted
            tier_stats["by_cell"][f"{cell[0]}|{cell[1]}"] = {
                "quota": quota, "accepted": accepted, "attempts": attempts,
            }
            if accepted < quota:
                print(
                    f"  ! {tier} cell {cell}: only {accepted}/{quota} after "
                    f"{attempts} attempts"
                )

        stats[tier] = tier_stats
        print(
            f"  {tier:<26} {tier_stats['accepted']:>3}/{per_class} "
            f"accepted from {tier_stats['attempts']:>6} draws"
        )

    return profiles, stats


def compare_rubric_to_original_15(original: list[dict]) -> dict:
    """
    Does the rubric reproduce the 15 hand-assigned labels?

    This is a validity check on the INSTRUMENT, run once and reported
    whatever it says. A low agreement rate would be a finding about the
    rubric (or about the original hand labelling) and would need to be
    discussed, not fixed by adjusting either one.
    """
    rows = []
    agree = 0
    for i, item in enumerate(original, start=1):
        human = item["expected"]
        verdict = label(item["features"])
        match = verdict.risk_class == human
        agree += match
        rows.append({
            "index": i,
            "hand_label": human,
            "rubric_label": verdict.risk_class,
            "agree": match,
            "capacity_band": verdict.capacity_band,
            "tolerance_band": verdict.tolerance_band,
        })

    n = len(original)
    adjacent = 0
    for row in rows:
        gap = abs(RISK_TIERS.index(row["hand_label"]) - RISK_TIERS.index(row["rubric_label"]))
        if gap <= 1:
            adjacent += 1

    return {
        "n": n,
        "exact_agreement": round(agree / n, 4) if n else None,
        "within_one_tier_agreement": round(adjacent / n, 4) if n else None,
        "n_exact_agreements": agree,
        "per_profile": rows,
        "interpretation": (
            "Agreement between two INDEPENDENT labelling instruments (a "
            "human applying CBI suitability rules by hand in the original "
            "fixture, and the band matrix in evaluation/risk_rubric.py). "
            "Neither was adjusted to improve this number. It is reported "
            "as a validity check on the rubric, not as a model result."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    print(f"Building gold set: {args.per_class} per class x "
          f"{len(RISK_TIERS)} classes, seed={args.seed}")
    print(f"Labelling instrument: evaluation/risk_rubric.py v{RUBRIC_VERSION} "
          f"(imports nothing from agents/)\n")

    profiles, stats = build_gold_profiles(args.per_class, args.seed)

    original = load_original_15()
    print(f"\n  original hand-labelled profiles preserved: {len(original)}")
    comparison = compare_rubric_to_original_15(original) if original else {}

    for i, item in enumerate(original, start=1):
        verdict = label(item["features"])
        profiles.append({
            "profile_id": f"ORIG_HAND_{i:03d}",
            "sequence": len(profiles) + 1,
            "features": item["features"],
            # The hand label is the ground truth for these 15. The rubric's
            # opinion is recorded alongside for comparison and is NOT
            # allowed to overwrite it.
            "ground_truth_risk_class": item["expected"],
            "provenance": "hand_labelled_original_15",
            "label_source": (
                "tests/unit/test_risk_profiling_agent.py::RQ1_FIXTURE — "
                "hand-assigned by applying CBI suitability rules manually, "
                "predates evaluation/risk_rubric.py"
            ),
            "label_assigned_before_model_run": True,
            "rubric_label_for_comparison_only": verdict.risk_class,
            "rubric": verdict.to_dict(),
        })

    counts: dict[str, int] = {}
    for p in profiles:
        counts[p["ground_truth_risk_class"]] = counts.get(p["ground_truth_risk_class"], 0) + 1

    payload = {
        "dataset_name": "gold_risk_profiles_v1",
        "layer": "layer3_gold",
        "purpose": (
            "The ONLY dataset in this project licensed for accuracy, "
            "precision, recall, F1 and confusion-matrix reporting on risk "
            "classification."
        ),
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "generator": "scripts/build_gold_risk_dataset.py",
            "seed": args.seed,
            "per_class_target": args.per_class,
            "rubric_version": RUBRIC_VERSION,
            "construction": (
                "Label-first rejection sampling. For each target class a "
                "suitability-matrix cell is chosen, demographically "
                "coherent profiles are drawn from broad priors, and a "
                "profile is kept only if the independent rubric places it "
                "in that cell. The model is never run during construction "
                "and no profile is ever discarded for being predicted "
                "incorrectly."
            ),
            "independence_guarantees": [
                "No model prediction was consulted at any point.",
                "risk_model.pkl is never loaded by this script.",
                "evaluation/risk_rubric.py imports nothing from agents/.",
                "Labels are fixed at build time and are never revised.",
                "No profile is filtered out on the basis of model output.",
            ],
            "sampling_statistics": stats,
            "class_counts": counts,
            "original_15_preserved": len(original),
        },
        "rubric": rubric_documentation(),
        "rubric_vs_original_15_agreement": comparison,
        "profiles": profiles,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote {len(profiles)} profiles -> "
          f"{args.output.relative_to(ROOT_DIR)}")
    print(f"  class balance: {counts}")
    if comparison:
        print(f"  rubric vs original 15: "
              f"exact={comparison['exact_agreement']:.3f}  "
              f"within-1-tier={comparison['within_one_tier_agreement']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
