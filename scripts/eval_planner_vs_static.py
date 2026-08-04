#!/usr/bin/env python3
"""
Planner vs static routing table — the Day 4-5 evaluation (RQ4).

THE QUESTION
    Does an LLM planner beat a hand-written routing table in a regulated
    advisory setting, and how often does it propose something invalid?

    Both halves matter. A planner that agrees with the table on every
    scenario has added a model call and 600 tokens per turn for nothing. A
    planner that disagrees usefully — a shorter plan when the user asked a
    narrow question, an extra agent when the message spans two topics — has
    earned its place. And a planner that proposes invalid plans at a
    measurable rate is a finding about LLM planning in regulated routing,
    which is worth reporting either way.

WHAT IS MEASURED
    plan_validity_rate      accepted proposals / attempts
    agreement_rate          accepted plans identical to the static table
                            (computed over ACCEPTED plans only — a rejected
                            plan agreeing with nothing is not a disagreement,
                            it is a rejection, and mixing the two would let a
                            high rejection rate masquerade as high agreement
                            once fallbacks are counted as "agreeing")
    rejection_counts        why it failed, by machine-readable code
    mean_steps              plan length per arm — the cost proxy
    static_contextual_gaps  how often the TABLE proposes a step whose
                            preconditions this scenario cannot meet. The
                            baseline is not perfect either, and a comparison
                            that only scores the new component is not a
                            comparison.

WHY THE SCENARIOS ARE FIXED AND COMMITTED
    A planner evaluated on messages chosen after seeing its behaviour proves
    nothing. This list is written once, covers each routing bucket, and
    deliberately includes the cases where the table is known to be crude:
    narrow questions it over-serves, multi-topic questions it under-serves,
    and returning users whose risk class is already cached.

RUNNING
    # honest numbers — one planner call per scenario
    EVAL_LIVE_API=1 LLM_CACHE=record python scripts/eval_planner_vs_static.py

    # replay a recorded run, no API calls, no cost
    LLM_CACHE=replay python scripts/eval_planner_vs_static.py

    # pipeline smoke test only — see the banner it prints
    python scripts/eval_planner_vs_static.py

COST
    One planner call per scenario. ~600 tokens each; the scenario set below
    is well under 20k tokens total. The static arm costs nothing — it is a
    dict lookup.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.payloads import STATIC_SEQUENCES  # noqa: E402
from evaluation.results_io import write_results  # noqa: E402
from orchestrator.planner import (  # noqa: E402
    Planner,
    PlanValidator,
    available_context_keys,
)
from utils.llm_client import LLMClient  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


# ── the scenario set ───────────────────────────────────────────────────────
# Each entry: id, message, the routing bucket the classifier assigns, the
# context available at that point, and a one-line note on what it probes.

_KNOWN_USER = {
    "user_features": {"age": 35, "income": 55000, "existing_debt": 5000,
                      "dependents": 1, "loss_tolerance": 0.4,
                      "investment_horizon": 10, "financial_knowledge_score": 3},
    "monthly_income": 4500,
    "monthly_expenses": 2800,
}

_RETURNING_USER = {
    **_KNOWN_USER,
    "risk_profile": {"risk_class": "balanced", "confidence": 0.82},
}

_NEW_USER: dict[str, Any] = {"user_features": {}, "risk_profile": None}

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "S01_small_talk",
        "message": "Hi, what can you help me with?",
        "intent": "conversational_only",
        "context": _NEW_USER,
        "probes": "trivial route — planner should not invent specialists",
    },
    {
        "id": "S02_out_of_scope",
        "message": "What's the weather in Galway tomorrow?",
        "intent": "conversational_only",
        "context": _NEW_USER,
        "probes": "refusal route",
    },
    {
        "id": "S03_risk_only",
        "message": "How risky an investor am I?",
        "intent": "risk_profiling",
        "context": _KNOWN_USER,
        "probes": "single-specialist route",
    },
    {
        "id": "S04_investment_known_user",
        "message": "I have 20,000 euro saved. What should I invest it in?",
        "intent": "investment",
        "context": _KNOWN_USER,
        "probes": "the canonical chain — risk must precede investment",
    },
    {
        "id": "S05_investment_new_user",
        "message": "What should I invest in?",
        "intent": "investment",
        "context": _NEW_USER,
        "probes": "nothing known — does the planner still chain correctly?",
    },
    {
        "id": "S06_investment_returning_user",
        "message": "Any new product suggestions for me?",
        "intent": "investment",
        "context": _RETURNING_USER,
        "probes": "risk_class already cached — the table re-runs risk, "
                  "the planner need not",
    },
    {
        "id": "S07_budget",
        "message": "Where is my money going each month?",
        "intent": "budget",
        "context": _KNOWN_USER,
        "probes": "budget route",
    },
    {
        "id": "S08_budget_no_data",
        "message": "Can you review my spending?",
        "intent": "budget",
        "context": _NEW_USER,
        "probes": "table proposes BudgetAgent with no expense data — this is "
                  "where the static arm should score a contextual gap",
    },
    {
        "id": "S09_full_advisory",
        "message": "I know nothing about finance. Tell me what to do with my money.",
        "intent": "full_advisory",
        "context": _KNOWN_USER,
        "probes": "longest route — is the planner willing to run four agents?",
    },
    {
        "id": "S10_explanation_request",
        "message": "Why did you say I was a balanced investor?",
        "intent": "explanation_request",
        "context": _RETURNING_USER,
        "probes": "explanation-only route",
    },
    {
        "id": "S11_narrow_question",
        "message": "Just list me three low-risk funds, no explanation needed.",
        "intent": "investment",
        "context": _RETURNING_USER,
        "probes": "the table always appends Explainability; a shorter plan "
                  "here would be a genuine planner win",
    },
    {
        "id": "S12_multi_topic",
        "message": "Can you look at both my spending and whether I should invest?",
        "intent": "investment",
        "context": _KNOWN_USER,
        "probes": "spans two routes; the table picks one, the planner can "
                  "include BudgetAgent",
    },
]


def _sequences_equal(a, b) -> bool:
    return list(a) == list(b)


def run(scenarios: list[dict], planner: Planner) -> dict[str, Any]:
    validator = PlanValidator()
    rows: list[dict[str, Any]] = []
    rejection_counter: Counter[str] = Counter()

    for sc in scenarios:
        context = sc["context"]
        available = available_context_keys(context)

        plan = planner.plan(sc["message"], context, sc["intent"])
        static_steps = STATIC_SEQUENCES.get(
            sc["intent"], STATIC_SEQUENCES["conversational_only"]
        )

        # Score the baseline on the same axis as the planner: a table entry
        # whose preconditions this scenario cannot meet is a routing miss,
        # and it is only visible if we check it.
        static_gaps = [
            p.as_dict() for p in validator.validate(list(static_steps), available)
        ]

        rejection_counter.update(plan.rejection_codes)

        rows.append({
            "id": sc["id"],
            "message": sc["message"],
            "intent": sc["intent"],
            "probes": sc["probes"],
            "available_context_keys": sorted(available),
            "planner_proposed": list(plan.proposed),
            "planner_accepted": plan.accepted,
            "planner_steps": list(plan.steps),
            "planner_rejections": [r.as_dict() for r in plan.rejections],
            "planner_tokens": plan.llm_tokens,
            "static_steps": list(static_steps),
            "static_contextual_gaps": static_gaps,
            "agrees_with_static": (
                plan.accepted and _sequences_equal(plan.steps, static_steps)
            ),
        })

    accepted = [r for r in rows if r["planner_accepted"]]
    agreed = [r for r in accepted if r["agrees_with_static"]]
    planner_lengths = [len(r["planner_steps"]) for r in rows]
    static_lengths = [len(r["static_steps"]) for r in rows]

    summary = {
        "scenarios": len(rows),
        "plan_validity_rate": round(len(accepted) / len(rows), 4) if rows else 0.0,
        "accepted": len(accepted),
        "fallbacks": len(rows) - len(accepted),
        # Denominator is ACCEPTED, not all scenarios — see module docstring.
        "agreement_rate_over_accepted": (
            round(len(agreed) / len(accepted), 4) if accepted else None
        ),
        "disagreements": [
            {"id": r["id"], "planner": r["planner_steps"], "static": r["static_steps"]}
            for r in accepted if not r["agrees_with_static"]
        ],
        "rejection_counts": dict(rejection_counter),
        "mean_steps_planner": round(sum(planner_lengths) / len(rows), 3) if rows else 0,
        "mean_steps_static": round(sum(static_lengths) / len(rows), 3) if rows else 0,
        "planner_tokens_total": sum(r["planner_tokens"] for r in rows),
        "static_scenarios_with_contextual_gaps": sum(
            1 for r in rows if r["static_contextual_gaps"]
        ),
    }
    return {"summary": summary, "scenarios": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="rq4_planner_vs_static.json",
                    help="filename under results/<llm_mode>/")
    ap.add_argument("--no-write", action="store_true",
                    help="print the summary, write nothing")
    args = ap.parse_args()

    from config.settings import settings
    if not settings.planner.enabled:
        print("PLANNER_ENABLED=false — the planner arm would short-circuit to "
              "the static table and the comparison would compare the baseline "
              "with itself. Unset it (or set true) and re-run.", file=sys.stderr)
        return 2

    client = LLMClient()
    if client.mode == "mock":
        bar = "!" * 74
        print(f"\n{bar}")
        print(" LLMClient is in MOCK mode. Mock responses are not JSON, so every")
        print(" proposal will be rejected as malformed_json and every arm will")
        print(" fall back. This run tests the PIPELINE, not the planner.")
        print(" For real numbers:")
        print("   EVAL_LIVE_API=1 LLM_CACHE=record "
              "python scripts/eval_planner_vs_static.py")
        print(f"{bar}\n")

    planner = Planner(client)
    payload = run(SCENARIOS, planner)
    payload["planner_stats"] = planner.stats.as_dict()
    payload["llm_mode"] = client.mode
    payload["mock_mode_warning"] = (
        "Mock mode: all proposals rejected as malformed_json; "
        "these numbers describe the fallback path only."
        if client.mode == "mock" else None
    )

    print(json.dumps(payload["summary"], indent=2))
    for row in payload["scenarios"]:
        if row["planner_accepted"] and not row["agrees_with_static"]:
            print(f"  DISAGREE {row['id']}: planner={row['planner_steps']} "
                  f"static={row['static_steps']}")

    if not args.no_write:
        path = write_results(payload, args.out)
        print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
