"""
Day 2a — synthetic per-customer transaction histories.

WHY THIS EXISTS
    BudgetAgent currently receives monthly_expenses as a single
    pre-computed dict — a snapshot, with no notion of *which window* that
    snapshot covers. That hides the actual finding: a 1-month window
    cannot see an annual insurance payment; a 12-month window can. This
    generator produces 12 months of dated, categorised transactions per
    customer so BudgetAgent._aggregate_transactions() has something real
    to derive 1/3/12-month estimates from — and so the gap between those
    estimates is a measurable number, not an assertion.

DESIGN — realistic structure, not invented spending levels
    Category *proportions* are anchored to settings.budget.
    ireland_hbs_benchmarks (CSO Ireland HBS 2022-23 [D11]) — the same
    benchmarks BudgetAgent compares users against, so a "within benchmark"
    customer and the HBS national average are the same numbers by
    construction, not a coincidence. Each category gets the temporal
    *structure* real spending actually has, not uniform noise around a
    monthly target:
        housing (rent/mortgage)   — monthly, fixed day, fixed amount
        food (groceries)          — weekly, lognormal amount
        utilities                 — bi-monthly, seasonal amplitude
        insurance                 — annual lump sum (the one that matters —
                                     any window under 12 months either
                                     misses it or inflates it, never close)
        entertainment (discretionary) — Poisson arrivals, heavy right tail

    "insurance" has no dedicated line in the 8-category HBS benchmark set
    used elsewhere in this codebase (housing/food/transport/utilities/
    healthcare/entertainment/savings/other) — it's carved out explicitly
    here because it's the category that demonstrates the window problem,
    not because it's independently benchmarked. It will show as an
    "unknown" category in BudgetAgent._compare_to_benchmarks(), which is
    correct: there's nothing to compare it against.

SCENARIO COVERAGE (deliberate, not random sampling)
    TXN_LUMPY          — high irregular costs: oversized insurance lump +
                          heavy-tailed discretionary spikes (a home repair,
                          an appliance). The flagship case for the
                          1-vs-12-month finding.
    TXN_BENCHMARK       — spending within benchmark on every category, low
                          variance. The control: proves the window effect
                          is real, not a modelling artefact that shows up
                          for every synthetic customer regardless.
    TXN_INCOME_CHANGE   — mid-year belt-tightening: months 1-6 at one
                          spending level, months 7-12 at a reduced one
                          (a pay cut). A 12-month average blends stale
                          "before" behaviour into a "now" estimate; here
                          the SHORTER window is the more accurate one —
                          the effect cuts both ways depending on what
                          actually changed, not "longer window always
                          wins".
    TXN_NEGATIVE        — expenses consistently exceed income. Not a
                          window-size story — a savings_rate_flag story.

USAGE
    python scripts/generate_transactions.py
    python scripts/generate_transactions.py --seed 7 --out data/processed/transactions.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config.settings import ROOT_DIR, settings  # noqa: E402

GENERATOR_VERSION = "1.0"
YEAR = 2025  # arbitrary fixed reference year — this is synthetic history,
             # not tied to the real calendar; _aggregate_transactions()
             # windows relative to the latest transaction date, not "today".

HBS = settings.budget.ireland_hbs_benchmarks


def _lognormal_mean_target(rng: np.random.Generator, target_mean: float, sigma: float) -> float:
    """
    numpy's lognormal is parameterised by (mu, sigma) of the underlying
    normal, not by the distribution's own mean. Solve for mu so
    E[X] == target_mean: mean = exp(mu + sigma^2/2).
    """
    target_mean = max(target_mean, 0.01)
    mu = math.log(target_mean) - (sigma ** 2) / 2
    return float(rng.lognormal(mu, sigma))


def _housing(monthly_income: float, rng: np.random.Generator, fraction: float | None = None) -> list[dict]:
    fraction = HBS["housing"] if fraction is None else fraction
    amount = round(monthly_income * fraction, 2)
    day = int(rng.integers(1, 4))  # fixed day each month (1st-3rd)
    return [
        {"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "housing", "amount": amount}
        for m in range(1, 13)
    ]


def _food(monthly_income: float, rng: np.random.Generator, fraction: float | None = None, sigma: float = 0.25) -> list[dict]:
    fraction = HBS["food"] if fraction is None else fraction
    monthly_target = monthly_income * fraction
    weekly_target = monthly_target / (52 / 12)
    out = []
    for m in range(1, 13):
        for week in range(4):
            day = min(1 + week * 7 + int(rng.integers(0, 3)), 28)
            amount = round(_lognormal_mean_target(rng, weekly_target, sigma), 2)
            out.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "food", "amount": amount})
    return out


def _utilities(monthly_income: float, rng: np.random.Generator, fraction: float | None = None, seasonal_amplitude: float = 0.4) -> list[dict]:
    """Bi-monthly (6 payments/year). seasonal_amplitude=0 for a flat/no-season variant."""
    fraction = HBS["utilities"] if fraction is None else fraction
    annual_target = monthly_income * 12 * fraction
    base = annual_target / 6
    out = []
    for m in range(1, 13, 2):
        # peaks in January (winter heating), troughs in July
        seasonal_mult = 1 + seasonal_amplitude * math.cos(2 * math.pi * (m - 1) / 12)
        amount = round(base * seasonal_mult, 2)
        out.append({"date": f"{YEAR}-{m:02d}-15", "category": "utilities", "amount": amount})
    return out


def _insurance(monthly_income: float, rng: np.random.Generator, annual_fraction: float, renewal_month: int = 6) -> list[dict]:
    """
    One lump payment a year. annual_fraction is a fraction of ANNUAL income
    (not monthly) — there is no HBS benchmark line for this, see module
    docstring; the fraction is an illustrative assumption, not a cited figure.
    """
    annual_income = monthly_income * 12
    amount = round(annual_income * annual_fraction, 2)
    return [{"date": f"{YEAR}-{renewal_month:02d}-01", "category": "insurance", "amount": amount}]


def _transport(monthly_income: float, rng: np.random.Generator, fraction: float | None = None, sigma: float = 0.2) -> list[dict]:
    """
    Weekly-ish (fuel, public transport top-ups) — same temporal shape as
    _food but its own noise level, since transport spend is typically
    steadier week to week than groceries.
    """
    fraction = HBS["transport"] if fraction is None else fraction
    monthly_target = monthly_income * fraction
    weekly_target = monthly_target / (52 / 12)
    out = []
    for m in range(1, 13):
        for week in range(4):
            day = min(1 + week * 7 + int(rng.integers(0, 3)), 28)
            amount = round(_lognormal_mean_target(rng, weekly_target, sigma), 2)
            out.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "transport", "amount": amount})
    return out


def _healthcare(
    monthly_income: float,
    rng: np.random.Generator,
    fraction: float | None = None,
    poisson_lambda: float = 1.2,
    sigma: float = 0.5,
) -> list[dict]:
    """
    Occasional (GP visits, pharmacy) — Poisson arrivals like _discretionary
    but a much lower rate, since healthcare is HBS's smallest category
    (5%) and genuinely sparse month to month, not a recurring bill.
    Some months legitimately have zero events, same as _discretionary.
    """
    fraction = HBS["healthcare"] if fraction is None else fraction
    monthly_target = monthly_income * fraction
    out = []
    for m in range(1, 13):
        n_events = int(rng.poisson(poisson_lambda))
        if n_events == 0:
            continue
        per_event_target = monthly_target / n_events
        for _ in range(n_events):
            day = int(rng.integers(1, 29))
            amount = round(_lognormal_mean_target(rng, per_event_target, sigma), 2)
            out.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "healthcare", "amount": amount})
    return out


def _discretionary(
    monthly_income: float,
    rng: np.random.Generator,
    fraction: float | None = None,
    poisson_lambda: float = 3.0,
    sigma: float = 0.6,
    spike_probability: float = 0.0,
    spike_multiplier: float = 4.0,
) -> list[dict]:
    """
    Poisson arrivals per month, lognormal amounts (heavy right tail via
    sigma). spike_probability > 0 injects an occasional large one-off
    (a repair, an appliance) — the "irregular cost" half of TXN_LUMPY.
    """
    fraction = HBS["entertainment"] if fraction is None else fraction
    monthly_target = monthly_income * fraction
    out = []
    for m in range(1, 13):
        n_events = int(rng.poisson(poisson_lambda))
        if n_events == 0:
            continue
        per_event_target = monthly_target / n_events
        for _ in range(n_events):
            day = int(rng.integers(1, 29))
            amount = round(_lognormal_mean_target(rng, per_event_target, sigma), 2)
            out.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "entertainment", "amount": amount})
        if rng.random() < spike_probability:
            spike_day = int(rng.integers(1, 29))
            spike_amount = round(per_event_target * spike_multiplier, 2)
            out.append({"date": f"{YEAR}-{m:02d}-{spike_day:02d}", "category": "entertainment", "amount": spike_amount})
    return out


def _generate_lumpy(rng: np.random.Generator) -> tuple[dict, list[dict]]:
    monthly_income = 4200.0
    txns = (
        _housing(monthly_income, rng)
        + _food(monthly_income, rng, sigma=0.3)
        + _utilities(monthly_income, rng, seasonal_amplitude=0.5)
        + _insurance(monthly_income, rng, annual_fraction=0.055, renewal_month=11)
        + _discretionary(monthly_income, rng, poisson_lambda=3.5, sigma=0.9, spike_probability=0.25, spike_multiplier=5.0)
        + _transport(monthly_income, rng)
        + _healthcare(monthly_income, rng)
    )
    meta = {
        "monthly_income": monthly_income,
        "scenario": "high_irregular_costs",
        "narrative": (
            "Oversized annual insurance renewal (5.5% of annual income, "
            "one lump in November) plus heavy-tailed discretionary "
            "spikes. A 1-month window misses it entirely unless it "
            "happens to land in November (100% underestimate). A "
            "3-month window that DOES include November doesn't dilute "
            "the cost toward the truth — it roughly quadruples it, "
            "since a once-a-year payment gets wrongly treated as if it "
            "recurs every 3 months. Short windows on a lump cost are "
            "not 'a bit off either way' — they're either a complete "
            "miss or a large overstatement, never close. Only the "
            "12-month window sees the true annualised rate."
        ),
    }
    return meta, txns


def _generate_benchmark(rng: np.random.Generator) -> tuple[dict, list[dict]]:
    monthly_income = 4500.0
    txns = (
        _housing(monthly_income, rng)
        + _food(monthly_income, rng, sigma=0.15)
        + _utilities(monthly_income, rng, seasonal_amplitude=0.2)
        + _insurance(monthly_income, rng, annual_fraction=0.02, renewal_month=3)
        + _discretionary(monthly_income, rng, poisson_lambda=3.0, sigma=0.3, spike_probability=0.0)
        + _transport(monthly_income, rng, sigma=0.1)
        + _healthcare(monthly_income, rng, sigma=0.2)
    )
    meta = {
        "monthly_income": monthly_income,
        "scenario": "entirely_within_benchmark",
        "narrative": (
            "Every category tracks its HBS benchmark fraction with low "
            "variance and a modest insurance line. Control case: if "
            "window size barely moves the estimate here while it moves "
            "it a lot for TXN_LUMPY, the effect is about lumpiness, not "
            "an artefact of the generator itself."
        ),
    }
    return meta, txns


def _generate_income_change(rng: np.random.Generator) -> tuple[dict, list[dict]]:
    income_before, income_after = 4000.0, 2800.0
    change_month = 7  # belt-tightening starts here

    # Committed costs (housing, utilities, insurance, transport, healthcare)
    # don't move with a pay cut on the timescale of a single year — only
    # discretionary and food spending adjust.
    txns = (
        _housing(income_before, rng, fraction=HBS["housing"])
        + _utilities(income_before, rng, seasonal_amplitude=0.3)
        + _insurance(income_before, rng, annual_fraction=0.025, renewal_month=2)
    )
    for m in range(1, 13):
        income_this_month = income_before if m < change_month else income_after
        food_fraction = HBS["food"] if m < change_month else HBS["food"] * 0.75
        disc_fraction = HBS["entertainment"] if m < change_month else HBS["entertainment"] * 0.4
        for week in range(4):
            day = min(1 + week * 7 + int(rng.integers(0, 3)), 28)
            weekly_target = income_this_month * food_fraction / (52 / 12)
            amount = round(_lognormal_mean_target(rng, weekly_target, 0.25), 2)
            txns.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "food", "amount": amount})
        n_events = int(rng.poisson(2.5))
        if n_events:
            monthly_disc_target = income_this_month * disc_fraction
            per_event_target = monthly_disc_target / n_events
            for _ in range(n_events):
                day = int(rng.integers(1, 29))
                amount = round(_lognormal_mean_target(rng, per_event_target, 0.5), 2)
                txns.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "entertainment", "amount": amount})

    # Appended after the loop, not with the other committed costs above —
    # keeps the food/discretionary RNG draws bit-identical to before this
    # category was added, since they're unaffected by two extra calls that
    # now happen only after all of THEIR draws are already consumed.
    txns += _transport(income_before, rng) + _healthcare(income_before, rng)

    meta = {
        "monthly_income": income_after,  # current income — what a 1-month view sees
        "monthly_income_before_month": change_month,
        "monthly_income_history": {"before": income_before, "after": income_after, "change_month": change_month},
        "scenario": "mid_year_income_change",
        "narrative": (
            f"Pay cut in month {change_month} ({income_before} -> "
            f"{income_after}/month); food and discretionary spending "
            f"drop in response, housing/utilities/insurance don't. Here "
            f"the SHORTER window is more accurate: a 12-month average "
            f"blends five months of stale pre-cut behaviour into an "
            f"estimate of current reality. The window effect cuts both "
            f"ways depending on what actually changed."
        ),
    }
    return meta, txns


def _generate_negative(rng: np.random.Generator) -> tuple[dict, list[dict]]:
    monthly_income = 2200.0
    # Fractions deliberately sum well above 1.0 — housing alone at 55%
    # (benchmark 28%) plus everything else running hot. Margin matters:
    # at a sum close to 1.0, Poisson's occasional zero-event months in
    # _discretionary (~8% chance per month at lambda=2.5) can pull the
    # realised average back under 100% by chance. This needs to be
    # robustly negative across reruns/seeds, not sitting on the boundary.
    txns = (
        _housing(monthly_income, rng, fraction=0.55)
        + _food(monthly_income, rng, fraction=0.24, sigma=0.2)
        + _utilities(monthly_income, rng, fraction=0.10, seasonal_amplitude=0.3)
        + _insurance(monthly_income, rng, annual_fraction=0.02, renewal_month=11)
        + _discretionary(monthly_income, rng, fraction=0.25, poisson_lambda=3.0, sigma=0.4)
        + _transport(monthly_income, rng)
        + _healthcare(monthly_income, rng)
    )
    meta = {
        "monthly_income": monthly_income,
        "scenario": "negative_disposable_income",
        "narrative": (
            "Housing alone is 55% of income (benchmark 28%); total "
            "expense fractions sum to ~1.16 of income by construction, "
            "so expenses exceed income in most months regardless of "
            "aggregation window. Not a window-size story — a "
            "savings_rate_flag story."
        ),
    }
    return meta, txns


SCENARIOS = {
    "TXN_LUMPY": _generate_lumpy,
    "TXN_BENCHMARK": _generate_benchmark,
    "TXN_INCOME_CHANGE": _generate_income_change,
    "TXN_NEGATIVE": _generate_negative,
}


# ── Variants (2-5) of each named scenario ────────────────────────────────
#
# WHY THESE EXIST
#   The coverage audit's own standard: "if 5 is insufficient to cover the
#   relevant variations, add more — 5 is a minimum, not a target" applies
#   here too, and n=1 per scenario means there was no way to tell whether
#   TXN_LUMPY's/TXN_BENCHMARK's/etc. behaviour generalised or was an
#   artefact of that one synthetic profile. These are genuine parameter
#   variations (income level, timing, magnitude) preserving each
#   scenario's qualitative character, not copies with the name changed.
#
#   TXN_LUMPY/_BENCHMARK/_INCOME_CHANGE/_NEGATIVE (variant 1, above) are
#   UNCHANGED — tests/unit/test_budget_agent.py asserts specific numeric
#   thresholds against their exact output (e.g. lumpy_insurance_3m_error
#   > 50.0), so their generator functions and parameters are untouched;
#   only new variant IDs are added alongside them.

def _generate_lumpy_variant(
    rng: np.random.Generator, income: float, insurance_fraction: float,
    renewal_month: int, disc_lambda: float, disc_sigma: float,
    spike_probability: float, spike_multiplier: float, food_sigma: float,
    utilities_amplitude: float,
) -> tuple[dict, list[dict]]:
    txns = (
        _housing(income, rng)
        + _food(income, rng, sigma=food_sigma)
        + _utilities(income, rng, seasonal_amplitude=utilities_amplitude)
        + _insurance(income, rng, annual_fraction=insurance_fraction, renewal_month=renewal_month)
        + _discretionary(income, rng, poisson_lambda=disc_lambda, sigma=disc_sigma,
                          spike_probability=spike_probability, spike_multiplier=spike_multiplier)
        + _transport(income, rng)
        + _healthcare(income, rng)
    )
    meta = {
        "monthly_income": income,
        "scenario": "high_irregular_costs",
        "narrative": (
            f"Variant of TXN_LUMPY at a different income level (EUR{income:.0f}/month) "
            f"and insurance renewal (month {renewal_month}, {insurance_fraction:.1%} of "
            f"annual income) — same lumpy-cost lesson, different profile, to check the "
            f"1-vs-12-month finding isn't an artefact of one specific customer."
        ),
    }
    return meta, txns


def _generate_benchmark_variant(
    rng: np.random.Generator, income: float, food_sigma: float,
    utilities_amplitude: float, insurance_fraction: float, renewal_month: int,
) -> tuple[dict, list[dict]]:
    txns = (
        _housing(income, rng)
        + _food(income, rng, sigma=food_sigma)
        + _utilities(income, rng, seasonal_amplitude=utilities_amplitude)
        + _insurance(income, rng, annual_fraction=insurance_fraction, renewal_month=renewal_month)
        + _discretionary(income, rng, poisson_lambda=3.0, sigma=0.3, spike_probability=0.0)
        + _transport(income, rng, sigma=0.1)
        + _healthcare(income, rng, sigma=0.2)
    )
    meta = {
        "monthly_income": income,
        "scenario": "entirely_within_benchmark",
        "narrative": (
            f"Variant of TXN_BENCHMARK at EUR{income:.0f}/month — still low-variance, "
            f"still tracking HBS fractions, different income level."
        ),
    }
    return meta, txns


def _generate_income_change_variant(
    rng: np.random.Generator, income_before: float, income_after: float, change_month: int,
) -> tuple[dict, list[dict]]:
    txns = (
        _housing(income_before, rng, fraction=HBS["housing"])
        + _utilities(income_before, rng, seasonal_amplitude=0.3)
        + _insurance(income_before, rng, annual_fraction=0.025, renewal_month=2)
    )
    for m in range(1, 13):
        income_this_month = income_before if m < change_month else income_after
        food_fraction = HBS["food"] if m < change_month else HBS["food"] * (income_after / income_before)
        disc_fraction = HBS["entertainment"] if m < change_month else HBS["entertainment"] * (income_after / income_before)
        for week in range(4):
            day = min(1 + week * 7 + int(rng.integers(0, 3)), 28)
            weekly_target = income_this_month * food_fraction / (52 / 12)
            amount = round(_lognormal_mean_target(rng, weekly_target, 0.25), 2)
            txns.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "food", "amount": amount})
        n_events = int(rng.poisson(2.5))
        if n_events:
            monthly_disc_target = income_this_month * disc_fraction
            per_event_target = monthly_disc_target / n_events
            for _ in range(n_events):
                day = int(rng.integers(1, 29))
                amount = round(_lognormal_mean_target(rng, per_event_target, 0.5), 2)
                txns.append({"date": f"{YEAR}-{m:02d}-{day:02d}", "category": "entertainment", "amount": amount})
    txns += _transport(income_before, rng) + _healthcare(income_before, rng)

    direction = "raise" if income_after > income_before else "pay cut"
    meta = {
        "monthly_income": income_after,
        "monthly_income_before_month": change_month,
        "monthly_income_history": {"before": income_before, "after": income_after, "change_month": change_month},
        "scenario": "mid_year_income_change",
        "narrative": (
            f"Variant of TXN_INCOME_CHANGE: a {direction} in month {change_month} "
            f"(EUR{income_before:.0f} -> EUR{income_after:.0f}/month). Food and "
            f"discretionary spending move with it; housing/utilities/insurance/"
            f"transport/healthcare don't. Same window-size lesson as the original "
            f"(shorter window more accurate here), tested against a different "
            f"magnitude/direction/timing of change."
        ),
    }
    return meta, txns


def _generate_negative_variant(
    rng: np.random.Generator, income: float, housing_fraction: float,
    food_fraction: float, utilities_fraction: float, disc_fraction: float,
    disc_lambda: float, disc_sigma: float, insurance_fraction: float, renewal_month: int,
) -> tuple[dict, list[dict]]:
    txns = (
        _housing(income, rng, fraction=housing_fraction)
        + _food(income, rng, fraction=food_fraction, sigma=0.2)
        + _utilities(income, rng, fraction=utilities_fraction, seasonal_amplitude=0.3)
        + _insurance(income, rng, annual_fraction=insurance_fraction, renewal_month=renewal_month)
        + _discretionary(income, rng, fraction=disc_fraction, poisson_lambda=disc_lambda, sigma=disc_sigma)
        + _transport(income, rng)
        + _healthcare(income, rng)
    )
    total_fraction = (
        housing_fraction + food_fraction + utilities_fraction + disc_fraction
        + insurance_fraction + HBS["transport"] + HBS["healthcare"]
    )
    meta = {
        "monthly_income": income,
        "scenario": "negative_disposable_income",
        "narrative": (
            f"Variant of TXN_NEGATIVE at EUR{income:.0f}/month — total expense "
            f"fractions sum to ~{total_fraction:.2f} of income by construction "
            f"(housing {housing_fraction:.0%}, food {food_fraction:.0%}, "
            f"discretionary {disc_fraction:.0%}), a different overspend profile "
            f"than the housing-dominated original, to check the savings_rate_flag "
            f"finding isn't specific to one dominant driver."
        ),
    }
    return meta, txns


# (income, insurance_fraction, renewal_month, disc_lambda, disc_sigma,
#  spike_probability, spike_multiplier, food_sigma, utilities_amplitude)
_LUMPY_VARIANTS = [
    (3100.0, 0.040, 3, 3.0, 0.80, 0.20, 4.0, 0.25, 0.40),
    (5600.0, 0.070, 7, 4.0, 1.00, 0.30, 6.0, 0.35, 0.55),
    (2650.0, 0.060, 1, 3.2, 0.85, 0.15, 4.5, 0.28, 0.45),
    (6800.0, 0.048, 9, 3.8, 0.95, 0.35, 5.5, 0.32, 0.50),
]

# (income, food_sigma, utilities_amplitude, insurance_fraction, renewal_month)
_BENCHMARK_VARIANTS = [
    (3000.0, 0.12, 0.15, 0.018, 6),
    (5200.0, 0.18, 0.25, 0.022, 9),
    (2400.0, 0.15, 0.18, 0.020, 1),
    (6000.0, 0.10, 0.22, 0.025, 12),
]

# (income_before, income_after, change_month)
_INCOME_CHANGE_VARIANTS = [
    (3500.0, 2500.0, 4),   # earlier, similarly-proportioned cut
    (5000.0, 4200.0, 9),   # smaller, later cut
    (2800.0, 1900.0, 5),   # severe cut, low starting income
    (3200.0, 4400.0, 6),   # a raise, not a cut — tests the opposite direction
]

# (income, housing_fraction, food_fraction, utilities_fraction, disc_fraction,
#  disc_lambda, disc_sigma, insurance_fraction, renewal_month)
_NEGATIVE_VARIANTS = [
    (2600.0, 0.30, 0.35, 0.12, 0.40, 3.5, 0.50, 0.02, 5),   # food/discretionary-led, not housing
    (3000.0, 0.35, 0.20, 0.12, 0.35, 3.0, 0.40, 0.02, 8),   # multiple moderate overspends, no single driver
    (1800.0, 0.65, 0.22, 0.11, 0.22, 2.8, 0.40, 0.02, 2),   # severe case, low income
    (3500.0, 0.42, 0.20, 0.10, 0.14, 3.0, 0.35, 0.02, 10),  # mild — closer to the boundary
]

for _i, _p in enumerate(_LUMPY_VARIANTS, start=2):
    SCENARIOS[f"TXN_LUMPY_{_i}"] = (lambda rng, p=_p: _generate_lumpy_variant(rng, *p))
for _i, _p in enumerate(_BENCHMARK_VARIANTS, start=2):
    SCENARIOS[f"TXN_BENCHMARK_{_i}"] = (lambda rng, p=_p: _generate_benchmark_variant(rng, *p))
for _i, _p in enumerate(_INCOME_CHANGE_VARIANTS, start=2):
    SCENARIOS[f"TXN_INCOME_CHANGE_{_i}"] = (lambda rng, p=_p: _generate_income_change_variant(rng, *p))
for _i, _p in enumerate(_NEGATIVE_VARIANTS, start=2):
    SCENARIOS[f"TXN_NEGATIVE_{_i}"] = (lambda rng, p=_p: _generate_negative_variant(rng, *p))


def generate_all(seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    customers = {}
    for customer_id, generator_fn in SCENARIOS.items():
        meta, txns = generator_fn(rng)
        txns.sort(key=lambda t: t["date"])
        customers[customer_id] = {
            "customer_id": customer_id,
            **meta,
            "transactions": txns,
        }

    return {
        "_meta": {
            "generated": True,
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hbs_anchoring_source": (
                "config.settings.BudgetConfig.ireland_hbs_benchmarks "
                "(CSO Ireland HBS 2022-23 [D11])"
            ),
            "hbs_benchmarks_used": HBS,
            "scenarios_covered": list(SCENARIOS.keys()),
            "year_reference": YEAR,
            "note": (
                "Synthetic — category proportions anchored to real HBS "
                "benchmarks, temporal structure (frequency/distribution "
                "per category) is a modelling assumption, documented per "
                "category in this file, not itself sourced from HBS."
            ),
        },
        "customers": customers,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "data" / "processed" / "transactions.json")
    args = ap.parse_args()

    data = generate_all(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))

    print(f"Generated {len(data['customers'])} customer transaction histories -> {args.out}")
    for cid, record in data["customers"].items():
        n = len(record["transactions"])
        total = sum(t["amount"] for t in record["transactions"])
        print(f"  {cid}: {n} transactions, €{total:,.2f} total over 12 months ({record['scenario']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())