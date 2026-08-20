"""
scripts/measure_planner_determinism.py — does temperature=0 actually mean
the planner routes the same message the same way twice, on THIS model,
right now?

    EVAL_LIVE_API=1 python scripts/measure_planner_determinism.py
    EVAL_LIVE_API=1 python scripts/measure_planner_determinism.py --repeats 30
    python scripts/measure_planner_determinism.py --dry-run   # smoke test, no LLM

WHY THIS SCRIPT EXISTS
    config/settings.py's PlannerConfig docstring states, about temperature:
    "0.0, deliberately... the same message on the same context should
    route the same way twice." That claim is asserted, not measured,
    anywhere in this codebase — no test or script calls the planner
    repeatedly on identical input and checks. A prior debugging session
    (not this codebase snapshot — see this script's own results file for
    what IS measured here) reported empirically finding 1-in-5 identical
    calls producing a different plan, which would make that docstring
    claim false for Groq's hosted inference specifically. Server-side
    non-determinism at temperature=0 is not unique to this project or a
    bug in this code — it's a documented property of some hosted LLM
    inference stacks (batching, kernel scheduling, floating-point
    non-associativity across batch compositions can all vary
    call-to-call even with temperature=0 and a fixed seed, when the
    provider doesn't expose or guarantee output-level determinism). But
    this codebase had no way to confirm, refute, or quantify that claim
    for itself — only a remembered number from a different session. This
    script exists to produce a current, reproducible, cite-able number
    instead of relying on that memory.

WHAT THIS DOES NOT DO
    It does not decide how the system SHOULD handle whatever it finds.
    If temperature=0 does turn out to be non-deterministic here, that's a
    genuine design question with real trade-offs across at least three
    directions — documenting it as a known limitation and doing nothing
    else, retrying and falling back to the static table on disagreement
    (added latency and cost, and does not itself guarantee agreement),
    or majority-voting across N calls (more latency and cost again, for
    a probabilistic rather than absolute guarantee) — and this script
    deliberately stops at measuring, not choosing between them. See this
    run's output for what the numbers actually show before deciding.

METHOD
    Reuses SCENARIOS from scripts/eval_planner_vs_static.py rather than
    defining new ones — same well-considered fixtures (a returning user
    with risk_class already cached, a narrow question, a multi-topic
    question), imported, not duplicated, so a change to those scenarios
    can't silently drift out of sync with this measurement. For each
    scenario, calls Planner.plan() REPEATS times with byte-identical
    (message, context, intent) and records every resulting step
    sequence — not just whether it matches the first call, so a plan
    that alternates between three different sequences across repeats is
    visible as three, not flattened into "disagreed once."

COST
    n_scenarios x REPEATS planner calls, ~600 tokens each per
    scripts/eval_planner_vs_static.py's own estimate. Default 12
    scenarios x 20 repeats = 240 calls, ~150k tokens. Use --scenarios to
    run a subset first if that's more than you want to spend confirming
    this.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from evaluation.results_io import write_results  # noqa: E402
from orchestrator.planner import Planner  # noqa: E402
from scripts.eval_planner_vs_static import SCENARIOS  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402


def _sequence_key(steps: tuple[str, ...] | list[str]) -> str:
    """A hashable, human-readable key for one observed sequence, used to
    count distinct outcomes without caring about list vs tuple identity."""
    return " -> ".join(steps) if steps else "(empty)"


def measure_scenario(planner: Planner, scenario: dict, repeats: int) -> dict:
    observed: Counter[str] = Counter()
    raw_sequences: list[list[str]] = []
    acceptance: Counter[bool] = Counter()
    rejection_codes: Counter[str] = Counter()

    for _ in range(repeats):
        plan = planner.plan(scenario["message"], dict(scenario["context"]), scenario["intent"])
        observed[_sequence_key(plan.steps)] += 1
        raw_sequences.append(list(plan.steps))
        acceptance[plan.accepted] += 1
        rejection_codes.update(plan.rejection_codes)

    n_distinct = len(observed)
    modal_sequence, modal_count = observed.most_common(1)[0]
    return {
        "scenario_id": scenario["id"],
        "message": scenario["message"],
        "repeats": repeats,
        "n_distinct_sequences_observed": n_distinct,
        "is_deterministic_across_these_repeats": n_distinct == 1,
        "modal_sequence": modal_sequence,
        "modal_sequence_frequency": round(modal_count / repeats, 4),
        "all_observed_sequences_with_counts": dict(observed),
        "n_accepted": acceptance.get(True, 0),
        "n_fell_back_to_static": acceptance.get(False, 0),
        "rejection_codes_seen": dict(rejection_codes),
        "raw_sequences_in_call_order": raw_sequences,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repeats", type=int, default=20,
                    help="calls per scenario with byte-identical input")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="scenario IDs to run (default: all from "
                         "eval_planner_vs_static.SCENARIOS)")
    ap.add_argument("--dry-run", action="store_true",
                    help="1 repeat per scenario, no LLM cost check — "
                         "confirms wiring, not determinism")
    args = ap.parse_args()

    llm = LLMClient()
    is_mock = llm.mode == "mock"
    if is_mock:
        print("\n  !! LLM is in MOCK mode — every call returns the same "
              "canned response, so this will trivially report "
              "'deterministic' regardless of what Groq actually does. "
              "This run cannot be used as evidence either way; results "
              "are written with is_placeholder=true. Set EVAL_LIVE_API=1 "
              "for a real measurement.\n")

    scenarios = SCENARIOS
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s["id"] in wanted]
        missing = wanted - {s["id"] for s in scenarios}
        if missing:
            print(f"  unknown scenario ids, skipping: {sorted(missing)}")

    repeats = 1 if args.dry_run else args.repeats
    print(f"  temperature={settings.planner.temperature}  "
          f"model={getattr(llm, 'model', 'unknown')}  "
          f"mode={llm.mode}  repeats={repeats}  "
          f"scenarios={len(scenarios)}")

    planner = Planner(llm)
    results = []
    for sc in scenarios:
        r = measure_scenario(planner, sc, repeats)
        results.append(r)
        flag = "DETERMINISTIC" if r["is_deterministic_across_these_repeats"] else "NON-DETERMINISTIC"
        print(f"  {sc['id']:28s} {flag:18s} "
              f"{r['n_distinct_sequences_observed']} distinct sequence(s) "
              f"over {repeats} calls "
              f"(modal: {r['modal_sequence_frequency']:.0%})")

    n_scenarios = len(results)
    n_deterministic = sum(r["is_deterministic_across_these_repeats"] for r in results)
    total_calls = sum(r["repeats"] for r in results)
    total_distinct_outcomes = sum(r["n_distinct_sequences_observed"] for r in results)
    # Overall "how often did a call disagree with that scenario's own
    # modal (most common) outcome" — the same shape of number a prior
    # session's "1 in 5" finding would have been.
    total_non_modal_calls = sum(
        r["repeats"] - r["all_observed_sequences_with_counts"][r["modal_sequence"]]
        for r in results
    )

    payload = {
        "phase": 4,
        "research_question": (
            "Empirical check of PlannerConfig's temperature=0 "
            "reproducibility claim (config/settings.py) — see this "
            "script's module docstring for why this had never been "
            "directly measured in this codebase before."
        ),
        "requires_llm": True,
        "config": {
            "temperature": settings.planner.temperature,
            "repeats_per_scenario": repeats,
            "n_scenarios": n_scenarios,
        },
        "headline": {
            "n_scenarios_fully_deterministic": n_deterministic,
            "n_scenarios_showing_variation": n_scenarios - n_deterministic,
            "fraction_of_calls_disagreeing_with_their_own_modal_outcome": (
                round(total_non_modal_calls / total_calls, 4) if total_calls else None
            ),
            "note": (
                "fraction_of_calls_disagreeing_with_their_own_modal_outcome "
                "is the same shape of number as the prior session's "
                "remembered '1 in 5' finding — compare directly, but "
                "treat this run's number as the current, reproducible one "
                "and the remembered figure as superseded by it, not "
                "confirmed by it, since they're different sampling runs "
                "against a provider that may itself have changed."
            ),
        },
        "scenarios": results,
        "decision_not_made_here": (
            "See this script's module docstring, 'WHAT THIS DOES NOT "
            "DO' — documenting-only, retry-with-fallback, and "
            "majority-voting are three different directions with real "
            "cost/latency trade-offs, and this script does not pick one."
        ),
    }

    path = write_results(payload, "planner_determinism_measurement.json", llm.mode)
    print(f"\n  {n_deterministic}/{n_scenarios} scenarios fully deterministic "
          f"over {repeats} repeats each")
    if total_calls:
        print(f"  {total_non_modal_calls}/{total_calls} calls "
              f"({total_non_modal_calls / total_calls:.1%}) disagreed with "
              f"their own scenario's most common outcome")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())