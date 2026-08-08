"""
scripts/add_demo_customer_transactions.py

Gives the 3 real CustomerStore demo customers (DEMO_GC_042, DEMO_GC_107,
DEMO_GMSC_318 — data/demo_customers.json) their own entries in
TransactionStore, so a real customer_id lookup through the live pipeline
(the API, or run_demo.py without --persona) actually finds something,
instead of the empty-list "checked, nothing there" result that was the
best Orchestrator._load_customer_transactions() could honestly do before
this script ran.

WHY THIS IS A SEPARATE SCRIPT, NOT AN EDIT TO generate_transactions.py
    generate_transactions.py's four TXN_* scenarios (LUMPY, BENCHMARK,
    INCOME_CHANGE, NEGATIVE) are purpose-built for RQ1 — the aggregation-
    window research question — and tests/unit/test_budget_agent.py does
    exact-string lookups against those names
    (customer_results["customer_id"] == "TXN_LUMPY", etc.). Renaming or
    folding new customers into that script's SCENARIOS dict risks
    perturbing committed RQ1 evidence and breaking that test outright —
    confirmed by grepping for the dependency before writing a line of
    this, not assumed.

    Equally, renaming data/demo_customers.json's IDs to match the TXN_*
    scheme was considered and rejected: tests/integration/
    test_existing_customer_baseline.py has DEMO_CUSTOMER_IDS as a literal
    constant driving a baseline test, and run_demo.py's --persona system
    keys off these exact IDs too.

    So: purely additive. This script never touches SCENARIOS, never
    calls generate_transactions.generate_all()/main(), and never
    modifies the four existing TXN_* entries in transactions.json — it
    reuses that file's own category-generator functions (imported, not
    reimplemented) to build NEW entries, then merges them into the
    EXISTING file under the customers' own real IDs.

WHY EACH CUSTOMER GETS A DIFFERENT COVERAGE/DENSITY PATTERN
    Not arbitrary — matches data/demo_customers.json's own profile for
    each, using the same coverage/density mechanism run_demo.py's
    --persona system already demonstrates (agents/data_sufficiency.py),
    just now tied to a real, independently-lookupable customer_id instead
    of a demo-script flag:
      DEMO_GC_042   42, employed, 1 dependent    -> full 12 months (high tier)
      DEMO_GC_107   29, employed, no dependents  -> 5 consecutive months (medium tier)
      DEMO_GMSC_318 61, RETIRED                  -> 3 scattered months across
                                                     the full year (high coverage,
                                                     low density — the dormant-
                                                     account case density_score
                                                     exists to catch, and the
                                                     one this customer's own
                                                     profile — retired, oldest
                                                     of the three — narratively
                                                     fits best)

    All three are built from the SAME low-variance, benchmark-conforming
    composition scripts/generate_transactions.py's own TXN_BENCHMARK
    uses (housing/food/utilities/insurance/discretionary via HBS
    fractions) — these represent an individual customer's real spending,
    not a deliberately stressed RQ1 scenario, so there is no reason for
    them to look different in kind from the control case.

IDEMPOTENT
    Re-running this only replaces these 3 customers' own entries; every
    other entry in the file (including anything else added later) is
    read back and preserved untouched.

USAGE
    python scripts/add_demo_customer_transactions.py
    python scripts/add_demo_customer_transactions.py --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR  # noqa: E402
from scripts.generate_transactions import (  # noqa: E402
    YEAR,
    _discretionary,
    _food,
    _housing,
    _insurance,
    _utilities,
)

DEMO_CUSTOMERS_PATH = ROOT_DIR / "data" / "demo_customers.json"
TRANSACTIONS_PATH = ROOT_DIR / "data" / "processed" / "transactions.json"

SEED = 100  # deliberately different from generate_transactions.py's
            # default (42) — independent, reproducible, never coincides.


def _month(date_str: str) -> int:
    return int(date_str[5:7])


def _full_year_benchmark_conforming(monthly_income: float, rng: np.random.Generator) -> list[dict]:
    """Same low-variance composition TXN_BENCHMARK uses — see module
    docstring for why these customers should look like the control case,
    not a deliberately stressed RQ1 scenario."""
    txns = (
        _housing(monthly_income, rng)
        + _food(monthly_income, rng, sigma=0.15)
        + _utilities(monthly_income, rng, seasonal_amplitude=0.2)
        + _insurance(monthly_income, rng, annual_fraction=0.02, renewal_month=3)
        + _discretionary(monthly_income, rng, poisson_lambda=3.0, sigma=0.3, spike_probability=0.0)
    )
    txns.sort(key=lambda t: t["date"])
    return txns


# (customer_id, months_to_keep, coverage_tier_this_produces, narrative)
_COVERAGE_PLAN: list[tuple[str, set[int] | None, str, str]] = [
    (
        "DEMO_GC_042", None, "high",
        "Established, active account — full 12 months of benchmark-"
        "conforming spending, matching this customer's own "
        "demo_customers.json profile (42, employed, 1 dependent).",
    ),
    (
        "DEMO_GC_107", {1, 2, 3, 4, 5}, "medium",
        "5 consecutive months of history — a newer or partially-visible "
        "account, matching this customer's own profile (29, employed, "
        "no dependents). See agents/data_sufficiency.py's coverage tiers.",
    ),
    (
        "DEMO_GMSC_318", {1, 6, 12}, "high coverage, low density",
        "3 months scattered across the full year — a dormant account: "
        "the date span still reads as ~12 months, but only 3 of them "
        "have any activity. Matches this customer's own profile (61, "
        "RETIRED, the oldest of the three) better than a contiguous "
        "slice would. This is deliberately the same coverage/density "
        "split run_demo.py's --persona stale demonstrates, now tied to "
        "a real, independently-lookupable customer_id.",
    ),
]


def build_demo_customer_entries(seed: int = SEED) -> dict[str, dict]:
    demo_customers = {
        c["customer_id"]: c for c in json.loads(DEMO_CUSTOMERS_PATH.read_text())
    }
    rng = np.random.default_rng(seed)
    entries: dict[str, dict] = {}

    for customer_id, months_to_keep, tier_label, narrative in _COVERAGE_PLAN:
        profile = demo_customers[customer_id]
        monthly_income = profile["income"] / 12
        full_year = _full_year_benchmark_conforming(monthly_income, rng)
        txns = (
            full_year if months_to_keep is None
            else [t for t in full_year if _month(t["date"]) in months_to_keep]
        )
        entries[customer_id] = {
            "customer_id": customer_id,
            "monthly_income": round(monthly_income, 2),
            "scenario": "demo_customer_own_transactions",
            "expected_coverage_tier": tier_label,
            "narrative": narrative,
            "transactions": txns,
        }

    return entries


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--transactions-path", type=Path, default=TRANSACTIONS_PATH)
    args = ap.parse_args()

    if not args.transactions_path.exists():
        print(
            f"{args.transactions_path} doesn't exist yet — run "
            f"`python scripts/generate_transactions.py` first. This script "
            f"only ADDS to that file, it never creates it from scratch."
        )
        return 1

    existing = json.loads(args.transactions_path.read_text())
    before_keys = set(existing.get("customers", {}).keys())

    new_entries = build_demo_customer_entries(seed=args.seed)
    existing.setdefault("customers", {}).update(new_entries)
    existing.setdefault("_meta", {})["demo_customer_transactions_added"] = {
        "added_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "customer_ids": list(new_entries.keys()),
        "source_script": "scripts/add_demo_customer_transactions.py",
    }

    args.transactions_path.write_text(json.dumps(existing, indent=2))

    after_keys = set(existing["customers"].keys())
    untouched = before_keys - set(new_entries.keys())
    print(f"Added/updated: {sorted(new_entries.keys())}")
    print(f"Untouched (still exactly as before): {sorted(untouched)}")
    print(f"Total customers in {args.transactions_path.name}: {len(after_keys)}")
    for customer_id, months_to_keep, tier_label, _ in _COVERAGE_PLAN:
        n = len(new_entries[customer_id]["transactions"])
        print(f"  {customer_id}: {n} transactions, expected tier={tier_label!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())