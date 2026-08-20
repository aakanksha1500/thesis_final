"""
scripts/build_stress_customer_base.py — builds the Layer-2 stress set.

    python scripts/build_stress_customer_base.py
    python scripts/build_stress_customer_base.py --count 1000 --seed 20260812

WHAT IT PRODUCES
    data/evaluation/stress_customers.json     1,000 profiles
    data/evaluation/stress_transactions.json  matching histories

WHAT THIS SET IS FOR, AND WHAT IT IS NOT FOR
    FOR: robustness and stress testing. Does the pipeline complete on
    every profile? Does it crash on a debt-to-income ratio of 3.0, an
    income of zero, a missing loss_tolerance? Is the confidence
    distribution stable across coverage tiers? Does every one of the six
    data-sufficiency tiers produce the disclosure it is supposed to?

    NOT FOR: accuracy, precision, recall or F1. Each profile carries a
    `construction_target_class` recording which rubric cell it was
    sampled to fill. That is a record of how the profile was BUILT. It is
    not an independent observation of what the profile IS, and
    evaluation/datasets.py deliberately refuses to expose it as ground
    truth — load_stress_customers() returns a dataset whose layer has
    has_ground_truth=False, so calling ground_truth() on it raises.

    The distinction matters because the sampler and the model share
    inputs. A "97% accuracy on 1,000 synthetic customers" headline built
    from construction targets measures the sampler, and a viva examiner
    will find that in about two questions.

DESIGN — WHY STRATIFIED AND NOT JUST RANDOM
    scripts/generate_synthetic_customer_base.py (the existing Layer-1
    load-test generator) samples coverage tiers from a probability mix,
    which is right for simulating a population but wrong for stress
    testing: the rare tiers get a handful of profiles and the tails go
    untested. This uses a full factorial instead —

        5 target risk classes x 6 coverage tiers = 30 strata
        30 profiles per stratum                  = 900 profiles
        + 100 deliberate edge cases              = 1,000

    — so every tier gets 150 profiles and every (class, tier)
    combination gets 30, regardless of how improbable that combination
    would be in a real population. Rarity is the point.

THE 100 EDGE CASES
    Boundary and adversarial inputs, each tagged with `edge_case_kind`,
    so a failure can be attributed instead of just counted: zero income,
    debt exceeding income several times over, minimum and maximum age,
    one-year and forty-year horizons, extreme tolerance and knowledge
    values, and profiles with required features MISSING entirely (which
    exercises RiskProfilingAgent._check_missing_features and the
    elicitation path rather than the scoring path).

REPRODUCIBILITY
    One seed (default 20260812), one numpy Generator, no global RNG use.
    Transaction histories are drawn from the same generator, so the whole
    artefact — profiles and histories — reproduces byte-for-byte.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR  # noqa: E402
from evaluation.risk_rubric import CELLS_BY_TIER, RISK_TIERS, label  # noqa: E402
from scripts.generate_transactions import (  # noqa: E402
    _discretionary,
    _food,
    _healthcare,
    _housing,
    _insurance,
    _transport,
    _utilities,
)

CUSTOMERS_PATH = ROOT_DIR / "data" / "evaluation" / "stress_customers.json"
TRANSACTIONS_PATH = ROOT_DIR / "data" / "evaluation" / "stress_transactions.json"

DEFAULT_SEED = 20260812
DEFAULT_COUNT = 1000
N_EDGE_CASES = 100

# (tier_name, months_of_history) — the month counts that land on each tier
# of agents/data_sufficiency.py::_COVERAGE_TIERS. Ranges are sampled
# within, so "low" is not always exactly 2 months.
COVERAGE_TIERS: list[tuple[str, tuple[int, int]]] = [
    ("insufficient", (0, 0)),
    ("minimal", (1, 1)),
    ("low", (2, 3)),
    ("medium", (4, 6)),
    ("medium_high", (7, 11)),
    ("high", (12, 12)),
]

EMPLOYMENT_POOL = [
    "employed", "self_employed", "self-employed", "part_time",
    "unemployed", "retired", "student",
]

EMPLOYMENT_BY_AGE: dict[str, tuple[int, int]] = {
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
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip() or "unknown"
    except Exception:
        return "unknown"


def _sample_profile(rng: np.random.Generator) -> dict:
    """A demographically coherent profile from broad priors."""
    age = int(rng.integers(18, 81))
    plausible = [
        emp for emp in EMPLOYMENT_POOL
        if EMPLOYMENT_BY_AGE[emp][0] <= age <= EMPLOYMENT_BY_AGE[emp][1]
    ]
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

    max_horizon = int(np.clip(78 - age, 1, 40))
    return {
        "age": age,
        "income": income,
        "employment_status": employment,
        "dependents": int(rng.integers(0, 6)),
        "existing_debt": float(np.round(income * rng.uniform(0.0, 0.95), 2)),
        "investment_horizon": int(rng.integers(1, max_horizon + 1)),
        "loss_tolerance": int(rng.integers(1, 6)),
        "financial_knowledge_score": int(rng.integers(1, 6)),
    }


def _sample_for_target_class(
    rng: np.random.Generator, target: str, max_attempts: int = 40_000
) -> tuple[dict, bool]:
    """
    Draw until the rubric places the profile in `target`.

    Returns (features, hit_target). On the rare miss the profile is kept
    anyway with hit_target=False — a stress set should not silently drop
    a hard-to-reach corner of the feature space, and the flag keeps the
    stratum honest about what it actually contains.
    """
    cells = set(CELLS_BY_TIER[target])
    for _ in range(max_attempts):
        features = _sample_profile(rng)
        verdict = label(features)
        if (verdict.capacity_band, verdict.tolerance_band) in cells:
            return features, True
    return _sample_profile(rng), False


def _edge_case_profile(rng: np.random.Generator, kind: str) -> dict:
    """Boundary and adversarial inputs. See module docstring."""
    base = _sample_profile(rng)

    if kind == "zero_income":
        base["income"] = 0.0
        base["existing_debt"] = float(np.round(rng.uniform(0, 20_000), 2))
    elif kind == "debt_exceeds_income":
        base["existing_debt"] = float(np.round(base["income"] * rng.uniform(1.5, 4.0), 2))
    elif kind == "zero_debt":
        base["existing_debt"] = 0.0
    elif kind == "minimum_age":
        base["age"] = 18
        base["employment_status"] = "student"
        base["income"] = float(np.round(rng.uniform(0, 15_000), 2))
    elif kind == "maximum_age":
        base["age"] = 80
        base["employment_status"] = "retired"
        base["investment_horizon"] = 1
    elif kind == "very_high_income":
        base["income"] = float(np.round(rng.uniform(250_000, 1_000_000), 2))
    elif kind == "many_dependents":
        base["dependents"] = int(rng.integers(6, 11))
    elif kind == "shortest_horizon":
        base["investment_horizon"] = 1
    elif kind == "longest_horizon":
        base["age"] = int(rng.integers(18, 35))
        base["investment_horizon"] = 40
    elif kind == "extreme_tolerance_low":
        base["loss_tolerance"] = 1
        base["financial_knowledge_score"] = 1
    elif kind == "extreme_tolerance_high":
        base["loss_tolerance"] = 5
        base["financial_knowledge_score"] = 5
    elif kind == "contradictory_low_capacity_high_tolerance":
        base["income"] = float(np.round(rng.uniform(12_000, 20_000), 2))
        base["existing_debt"] = float(np.round(base["income"] * 0.8, 2))
        base["dependents"] = 4
        base["loss_tolerance"] = 5
        base["financial_knowledge_score"] = 5
    elif kind == "missing_loss_tolerance":
        base.pop("loss_tolerance", None)
    elif kind == "missing_income":
        base.pop("income", None)
    elif kind == "missing_two_features":
        base.pop("financial_knowledge_score", None)
        base.pop("dependents", None)
    return base


EDGE_CASE_KINDS = [
    "zero_income", "debt_exceeds_income", "zero_debt", "minimum_age",
    "maximum_age", "very_high_income", "many_dependents",
    "shortest_horizon", "longest_horizon", "extreme_tolerance_low",
    "extreme_tolerance_high", "contradictory_low_capacity_high_tolerance",
    "missing_loss_tolerance", "missing_income", "missing_two_features",
]


def _months_for_tier(rng: np.random.Generator, tier: str) -> int:
    for name, (lo, hi) in COVERAGE_TIERS:
        if name == tier:
            return int(rng.integers(lo, hi + 1))
    return 12


def _history(
    rng: np.random.Generator, monthly_income: float, months: int
) -> list[dict]:
    """
    A transaction history covering exactly `months` distinct months.

    Built as a full year then filtered to the first `months` calendar
    months, which is how scripts/generate_synthetic_customer_base.py does
    it — keeping seasonal structure (insurance renewals, utility
    seasonality) intact rather than generating a flat stub.
    """
    if months <= 0:
        return []

    full_year = (
        _housing(monthly_income, rng)
        + _food(monthly_income, rng, sigma=0.2)
        + _utilities(monthly_income, rng, seasonal_amplitude=0.25)
        + _insurance(monthly_income, rng, annual_fraction=0.02,
                     renewal_month=int(rng.integers(1, 13)))
        + _discretionary(monthly_income, rng, poisson_lambda=3.0,
                         sigma=0.35, spike_probability=0.05)
        + _transport(monthly_income, rng)
        + _healthcare(monthly_income, rng)
    )
    full_year.sort(key=lambda t: t["date"])
    keep = set(range(1, months + 1))
    return [t for t in full_year if int(t["date"][5:7]) in keep]


def build(count: int, seed: int) -> tuple[list[dict], dict, dict]:
    rng = np.random.default_rng(seed)

    n_population = count - N_EDGE_CASES
    strata = [(cls, tier) for cls in RISK_TIERS for tier, _ in COVERAGE_TIERS]
    per_stratum = n_population // len(strata)
    remainder = n_population - per_stratum * len(strata)

    customers: list[dict] = []
    transactions: dict[str, dict] = {}
    stats = {
        "per_stratum": per_stratum,
        "n_strata": len(strata),
        "target_class_misses": 0,
        "coverage_tier_counts": {},
        "employment_counts": {},
        "construction_class_counts": {},
    }

    index = 0
    for stratum_i, (target_class, tier) in enumerate(strata):
        quota = per_stratum + (1 if stratum_i < remainder else 0)
        for _ in range(quota):
            index += 1
            features, hit = _sample_for_target_class(rng, target_class)
            if not hit:
                stats["target_class_misses"] += 1

            months = _months_for_tier(rng, tier)
            monthly_income = float(features.get("income", 0.0)) / 12.0
            txns = _history(rng, monthly_income, months)

            customer_id = f"STRESS_{index:05d}"
            customers.append({
                "customer_id": customer_id,
                "features": features,
                # NOT ground truth. See module docstring and
                # evaluation/datasets.py::load_stress_customers.
                "construction_target_class": target_class,
                "construction_target_hit": hit,
                "expected_coverage_tier": tier,
                "expected_months_of_history": months,
                "stratum": f"{target_class}|{tier}",
                "edge_case_kind": None,
                "layer": "layer2_stress",
            })
            transactions[customer_id] = {
                "customer_id": customer_id,
                "monthly_income": round(monthly_income, 2),
                "scenario": "stress_test_population",
                "expected_coverage_tier": tier,
                "transactions": txns,
            }

            stats["coverage_tier_counts"][tier] = stats["coverage_tier_counts"].get(tier, 0) + 1
            emp = features.get("employment_status", "unknown")
            stats["employment_counts"][emp] = stats["employment_counts"].get(emp, 0) + 1
            stats["construction_class_counts"][target_class] = (
                stats["construction_class_counts"].get(target_class, 0) + 1
            )

    # Edge cases — cycled through the kinds so every kind gets equal weight.
    for i in range(N_EDGE_CASES):
        index += 1
        kind = EDGE_CASE_KINDS[i % len(EDGE_CASE_KINDS)]
        features = _edge_case_profile(rng, kind)
        tier = COVERAGE_TIERS[i % len(COVERAGE_TIERS)][0]
        months = _months_for_tier(rng, tier)
        monthly_income = float(features.get("income", 0.0)) / 12.0

        customer_id = f"STRESS_EDGE_{i + 1:03d}"
        customers.append({
            "customer_id": customer_id,
            "features": features,
            "construction_target_class": None,
            "construction_target_hit": None,
            "expected_coverage_tier": tier,
            "expected_months_of_history": months,
            "stratum": f"edge_case|{tier}",
            "edge_case_kind": kind,
            "layer": "layer2_stress",
        })
        transactions[customer_id] = {
            "customer_id": customer_id,
            "monthly_income": round(monthly_income, 2),
            "scenario": "stress_test_edge_case",
            "expected_coverage_tier": tier,
            "transactions": _history(rng, monthly_income, months),
        }
        stats["coverage_tier_counts"][tier] = stats["coverage_tier_counts"].get(tier, 0) + 1

    stats["n_edge_cases"] = N_EDGE_CASES
    stats["edge_case_kinds"] = EDGE_CASE_KINDS
    stats["n_total"] = len(customers)
    return customers, transactions, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    print(f"Building Layer-2 stress set: {args.count} profiles, "
          f"seed={args.seed}")
    customers, transactions, stats = build(args.count, args.seed)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "generator": "scripts/build_stress_customer_base.py",
        "seed": args.seed,
        "design": (
            "Full factorial: 5 construction target classes x 6 "
            "data-sufficiency coverage tiers, 30 profiles per stratum, "
            "plus 100 tagged edge cases."
        ),
        "usage_restriction": (
            "ROBUSTNESS AND STRESS TESTING ONLY. construction_target_class "
            "records which rubric cell each profile was sampled to fill "
            "and is NOT ground truth. evaluation/datasets.py refuses to "
            "expose it as such. Any accuracy, precision, recall or F1 "
            "number computed from this file would be measuring the "
            "sampler, not the model."
        ),
        "statistics": stats,
    }

    CUSTOMERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOMERS_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset_name": "stress_customers_v1",
            "layer": "layer2_stress",
            "provenance": provenance,
            "customers": customers,
        }, fh, indent=2)

    with open(TRANSACTIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset_name": "stress_transactions_v1",
            "layer": "layer2_stress",
            "provenance": {k: v for k, v in provenance.items() if k != "statistics"},
            "transactions": transactions,
        }, fh, separators=(",", ":"))

    n_txn = sum(len(v["transactions"]) for v in transactions.values())
    print(f"\nWrote {len(customers)} profiles -> "
          f"{CUSTOMERS_PATH.relative_to(ROOT_DIR)}")
    print(f"Wrote {n_txn} transactions -> "
          f"{TRANSACTIONS_PATH.relative_to(ROOT_DIR)} "
          f"({TRANSACTIONS_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  coverage tiers: {stats['coverage_tier_counts']}")
    print(f"  employment:     {stats['employment_counts']}")
    print(f"  target misses:  {stats['target_class_misses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
