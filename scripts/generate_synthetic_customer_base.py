"""
scripts/generate_synthetic_customer_base.py

Generates N synthetic customers (default 100) with a COMPLETE
RiskConfig.required_features record each, plus a matching TransactionStore
history — specifically so there's a large enough, independently-addressable
customer base for scripts/run_concurrency_demo.py to actually exercise
concurrent sessions against, rather than repeatedly hammering the same 3
demo customers.

WHY THIS IS SYNTHETIC, NOT scripts/preprocess_customers.py's REAL DATA
    Checked before writing this: data/raw/german_credit.data and data/raw/
    give_me_some_credit.csv (what preprocess_customers.py needs) are not
    present in this repository at all — that script would produce zero
    rows as-is. Even with those files, its own code deliberately leaves
    loss_tolerance and financial_knowledge_score blank (neither raw
    dataset captures psychometric preference at all) — RiskProfilingAgent
    could never classify a customer built from real-dataset preprocessing
    alone; that's WHY data/demo_customers.json's 3 profiles are hand-
    completed rather than machine-generated. So "at least 100 customers,
    ready for the live pipeline" genuinely needs a generator that produces
    all 8 required_features directly — this is that, clearly labelled as
    synthetic rather than pretending to be derived from a real dataset it
    isn't.

WHY PURELY ADDITIVE, SAME PRINCIPLE AS add_demo_customer_transactions.py
    Never touches the 3 real DEMO_* customers or the 4 TXN_* research
    scenarios — different ID namespace entirely (SYNTH_NNNNN), and both
    the CustomerStore write (via its own safe/atomic save()) and the
    TransactionStore merge (read-modify-write, same technique as the
    demo-customer script) leave everything already on disk untouched.

WHY THE COVERAGE/DENSITY MIX IS DELIBERATE, NOT UNIFORM
    All 100 customers having full 12-month history would make
    scripts/run_concurrency_demo.py's BudgetAgent calls uniformly
    "complete" — realistic-looking but uninteresting, and it wouldn't
    exercise the questionnaire fallback path under concurrent load at
    all. ~45% full year, ~30% partial (medium tier), ~15% dormant
    (stale pattern), ~10% no history at all — the same tiers agents/
    data_sufficiency.py already names, just distributed across a
    population instead of one customer per --persona flag.

USAGE
    python scripts/generate_synthetic_customer_base.py
    python scripts/generate_synthetic_customer_base.py --count 250 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR  # noqa: E402
from data.customer_store import CustomerStore  # noqa: E402
from scripts.generate_transactions import (  # noqa: E402
    _discretionary,
    _food,
    _healthcare,
    _housing,
    _insurance,
    _transport,
    _utilities,
)

TRANSACTIONS_PATH = ROOT_DIR / "data" / "processed" / "transactions.json"
SEED = 500  # distinct from generate_transactions.py's 42 and
            # add_demo_customer_transactions.py's 100 — independent,
            # reproducible, never coincides with either.

EMPLOYMENT_CHOICES = ["employed", "self-employed", "unemployed", "retired", "student"]
EMPLOYMENT_WEIGHTS = [0.62, 0.12, 0.06, 0.15, 0.05]

# (weight, months_to_keep_or_None, tier_label) — None means "all 12".
#
# minimal/low/medium_high were missing entirely until the coverage audit
# flagged it: agents/data_sufficiency.py names six tiers (insufficient,
# minimal, low, medium, medium_high, high) but this mix only ever produced
# four of them, so two of six were structurally unreachable in the real
# pipeline data regardless of population size — not under-sampled, actually
# impossible to observe. months_to_keep values below are chosen so the
# resulting calendar SPAN (not a count of months) lands each tier where
# agents/data_sufficiency.py's _COVERAGE_TIERS staircase expects it:
# {1}->span 1 (minimal), {1,2,3}->span 3 (low), {1..7}->span 7 (medium_high).
_COVERAGE_MIX = [
    (0.30, None, "high"),
    (0.20, {1, 2, 3, 4, 5}, "medium"),
    (0.10, {1, 2, 3, 4, 5, 6, 7}, "medium_high"),
    (0.10, {1, 6, 12}, "high coverage, low density"),
    (0.10, {1}, "minimal"),
    (0.10, {1, 2, 3}, "low"),
    (0.10, set(), "insufficient"),
]


def _month(date_str: str) -> int:
    return int(date_str[5:7])


def _sample_customer(rng: np.random.Generator, index: int) -> dict:
    customer_id = f"SYNTH_{index:05d}"

    age = int(np.clip(rng.normal(42, 14), 18, 80))
    # Income loosely age-correlated (career progression, then a step down
    # around typical retirement age) — a flat random income wouldn't
    # produce anything BudgetAgent/RiskProfilingAgent would find internally
    # inconsistent, but it also wouldn't look like a real population.
    base_income = 22000 + min(age, 55) * 650
    if age >= 65:
        base_income *= 0.55  # pension-era step down
    income = round(float(np.clip(rng.lognormal(np.log(base_income), 0.35), 15000, 180000)), 2)

    employment_status = str(rng.choice(EMPLOYMENT_CHOICES, p=EMPLOYMENT_WEIGHTS))
    if age >= 66 and employment_status not in ("retired",):
        employment_status = "retired"  # keep the population internally coherent

    dependents = int(np.clip(rng.poisson(1.0), 0, 5))
    existing_debt = round(float(income * np.clip(rng.normal(0.18, 0.12), 0.0, 0.6)), 2)
    # Investment horizon inversely related to age, floor of 1 year.
    investment_horizon = int(np.clip(round((75 - age) * rng.uniform(0.15, 0.35)), 1, 35))
    loss_tolerance = int(np.clip(round(rng.normal(3.0, 1.0)), 1, 5))
    financial_knowledge_score = int(np.clip(round(rng.normal(3.0, 1.0)), 1, 5))

    return {
        "customer_id": customer_id,
        "age": age,
        "income": income,
        "employment_status": employment_status,
        "dependents": dependents,
        "existing_debt": existing_debt,
        "investment_horizon": investment_horizon,
        "loss_tolerance": loss_tolerance,
        "financial_knowledge_score": financial_knowledge_score,
    }


def _sample_transactions(rng: np.random.Generator, monthly_income: float) -> tuple[list[dict], str]:
    weights = [w for w, _, _ in _COVERAGE_MIX]
    idx = rng.choice(len(_COVERAGE_MIX), p=weights)
    _, months_to_keep, tier_label = _COVERAGE_MIX[idx]

    if months_to_keep is not None and len(months_to_keep) == 0:
        return [], tier_label  # "insufficient" — genuinely no history

    full_year = (
        _housing(monthly_income, rng)
        + _food(monthly_income, rng, sigma=0.2)
        + _utilities(monthly_income, rng, seasonal_amplitude=0.25)
        + _insurance(monthly_income, rng, annual_fraction=0.02, renewal_month=int(rng.integers(1, 13)))
        + _discretionary(monthly_income, rng, poisson_lambda=3.0, sigma=0.35, spike_probability=0.05)
        + _transport(monthly_income, rng)
        + _healthcare(monthly_income, rng)
    )
    full_year.sort(key=lambda t: t["date"])
    txns = (
        full_year if months_to_keep is None
        else [t for t in full_year if _month(t["date"]) in months_to_keep]
    )
    return txns, tier_label


def generate(count: int, seed: int = SEED) -> tuple[list[dict], dict[str, dict]]:
    rng = np.random.default_rng(seed)
    customers: list[dict] = []
    transactions: dict[str, dict] = {}

    for i in range(1, count + 1):
        profile = _sample_customer(rng, i)
        txns, tier_label = _sample_transactions(rng, profile["income"] / 12)
        customers.append(profile)
        transactions[profile["customer_id"]] = {
            "customer_id": profile["customer_id"],
            "monthly_income": round(profile["income"] / 12, 2),
            "scenario": "synthetic_load_test_population",
            "expected_coverage_tier": tier_label,
            "transactions": txns,
        }

    return customers, transactions


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--transactions-path", type=Path, default=TRANSACTIONS_PATH)
    args = ap.parse_args()

    customers, new_transactions = generate(args.count, seed=args.seed)

    store = CustomerStore()
    for profile in customers:
        customer_id = profile.pop("customer_id")
        store.save(customer_id, profile, source_dataset="synthetic_load_test")

    if not args.transactions_path.exists():
        print(
            f"{args.transactions_path} doesn't exist — run "
            f"`python scripts/generate_transactions.py` first."
        )
        return 1
    existing = json.loads(args.transactions_path.read_text())
    existing.setdefault("customers", {}).update(new_transactions)
    args.transactions_path.write_text(json.dumps(existing, indent=2))

    tier_counts: dict[str, int] = {}
    for entry in new_transactions.values():
        tier_counts[entry["expected_coverage_tier"]] = (
            tier_counts.get(entry["expected_coverage_tier"], 0) + 1
        )

    print(f"Generated {len(customers)} synthetic customers: "
          f"SYNTH_{1:05d} .. SYNTH_{args.count:05d}")
    print(f"Persisted to CustomerStore (data/processed/customers.csv)")
    print(f"Persisted transaction histories to {args.transactions_path}")
    print(f"Coverage tier distribution: {tier_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())