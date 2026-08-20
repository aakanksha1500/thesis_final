"""
verify_before_after.py — demonstrate each mechanism fix against the ORIGINAL code.

Run this from inside either tree:

    cd "<pristine tree>"  && python3 ../verify_before_after.py
    cd "<fixed tree>"     && python3 ../verify_before_after.py

It probes the three mechanism-level bugs directly, using only symbols that
exist in BOTH trees, so the same script runs against pre-fix and post-fix code
and the difference in output is the evidence.

Why this exists rather than a pytest that "fails before":
    The fixes introduce new modules (orchestrator/routing_policy.py,
    agents/transaction_contract.py, ...). A pytest file importing them cannot
    even be COLLECTED against the original tree — the run aborts at import
    with ModuleNotFoundError, which proves nothing about behaviour. This
    script imports nothing new and exercises the old seams.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("APPROVAL_DB_PATH", "/tmp/vba_approvals.db")
os.environ.setdefault("MEMORY_DB_PATH", "/tmp/vba_memory.db")
sys.path.insert(0, os.getcwd())

RESULTS: list[tuple[str, str, str]] = []


def record(probe: str, observed: str, verdict: str) -> None:
    RESULTS.append((probe, observed, verdict))
    print(f"  [{verdict:4}] {probe}\n         {observed}")


# PROBE 1 — does a moderately-confident specialist intent reach the specialist?
def probe_routing_gate() -> None:
    from orchestrator.orchestrator import Orchestrator, RoutingDecision
    from utils.llm_client import LLMClient

    orch = Orchestrator(LLMClient(force_mock=True), session_id="vba-routing")

    # Force the classifier to return a CORRECT intent at 0.55 confidence —
    # below the old 0.65 threshold, above the new accept_at of 0.55.
    conv = orch._agents["ConversationalAgent"]
    conv.classify_only = lambda msg: ("risk_profiling", 0.55)
    try:
        from orchestrator.routing_policy import IntentSignal

        conv.classify_signal = lambda msg: IntentSignal(
            intent="risk_profiling",
            raw_confidence=0.55,
            alternatives={"risk_profiling": 0.55, "general_query": 0.05},
        )
    except ImportError:
        pass          # original tree: only classify_only exists

    routing, intent, conf = orch._classify_intent("Can you assess my risk profile?")

    if routing is RoutingDecision.RISK_PROFILING:
        record(
            "Correct intent at confidence 0.55 routes to the specialist",
            f"routing={routing.value} intent={intent}",
            "PASS",
        )
    else:
        record(
            "Correct intent at confidence 0.55 routes to the specialist",
            f"routing={routing.value} (intent '{intent}' was DISCARDED by the "
            f"confidence gate)",
            "FAIL",
        )


# PROBE 2 — does the classifier's own detected intent survive a rejection?
def probe_intent_preserved_in_audit() -> None:
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient

    orch = Orchestrator(LLMClient(force_mock=True), session_id="vba-audit")
    conv = orch._agents["ConversationalAgent"]
    conv.classify_only = lambda msg: ("risk_profiling", 0.10)
    try:
        from orchestrator.routing_policy import IntentSignal

        conv.classify_signal = lambda msg: IntentSignal(
            intent="risk_profiling", raw_confidence=0.10,
            alternatives={"risk_profiling": 0.10},
        )
    except ImportError:
        pass

    orch._classify_intent("something vague")
    outcome = getattr(orch, "last_routing_outcome", None)

    if outcome is not None and outcome.to_audit().get("detected_intent") == "risk_profiling":
        record(
            "A gate-rejected intent is still recorded for audit",
            f"detected_intent={outcome.to_audit()['detected_intent']} "
            f"routed_intent={outcome.to_audit()['routed_intent']} "
            f"band={outcome.to_audit()['band']}",
            "PASS",
        )
    else:
        record(
            "A gate-rejected intent is still recorded for audit",
            "no routing evidence recorded — a gate-forced fallback is "
            "indistinguishable in the log from a genuine conversational "
            "classification",
            "FAIL",
        )


# PROBE 3 — does ExplainabilityAgent invent a risk class on upstream failure?
def probe_hallucination_on_upstream_failure() -> None:
    from explainability.explainability_agent import ExplainabilityAgent
    from utils.llm_client import LLMClient

    agent = ExplainabilityAgent(LLMClient(force_mock=True))
    result = agent.run({
        "risk_agent_payload": {
            "status": "needs_input",
            "needs": ["income", "existing_debt"],
        },
    })
    risk_class = result.payload.get("risk_class")
    confidence = result.payload.get("confidence")

    if risk_class == "moderate" and confidence == 0.7:
        record(
            "No risk class is asserted when upstream reported needs_input",
            f"asserted risk_class={risk_class!r} at confidence={confidence} "
            f"— the RQ4 S02 hallucination, reproduced exactly",
            "FAIL",
        )
    elif risk_class in (None, "", "unknown"):
        record(
            "No risk class is asserted when upstream reported needs_input",
            f"refused: status={result.payload.get('status')!r} "
            f"risk_class={risk_class!r} confidence={confidence!r}",
            "PASS",
        )
    else:
        record(
            "No risk class is asserted when upstream reported needs_input",
            f"unexpected: risk_class={risk_class!r} confidence={confidence!r}",
            "FAIL",
        )


# PROBE 4 — is a plan blocked only on elicitable data rejected or repaired?
def probe_planner_data_gap() -> None:
    from orchestrator.planner import PlanValidator

    validator = PlanValidator()
    steps = ["RiskProfilingAgent", "InvestmentAgent"]

    if hasattr(validator, "repair"):
        repaired, elicited, problems = validator.repair(steps, available=set())
        if not problems and repaired[0] == "ConversationalAgent":
            record(
                "A plan short only of user-supplied data is repaired, not rejected",
                f"repaired={repaired} elicits={elicited}",
                "PASS",
            )
        else:
            record(
                "A plan short only of user-supplied data is repaired, not rejected",
                f"repair returned problems={[str(p) for p in problems]}",
                "FAIL",
            )
    else:
        problems = validator.validate(steps, available=set())
        record(
            "A plan short only of user-supplied data is repaired, not rejected",
            f"no repair() exists; validate() rejects with "
            f"{[p.code.value for p in problems]} and the plan is replaced by "
            f"the static table, which needs the same missing key",
            "FAIL",
        )


# PROBE 5 — end-to-end routing over the RQ4 scenario set, no LLM
def probe_rq4_routing_end_to_end() -> None:
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient

    seed = {
        "age": 41, "income": 4200, "existing_debt": 620, "dependents": 1,
        "employment_status": "employed", "investment_horizon": 12,
        "loss_tolerance": "moderate", "financial_knowledge_score": 3,
    }
    scenarios = [
        ("What is my account balance?", "conversational_only", False),
        ("Can you assess my financial risk profile?", "risk_profiling", True),
        ("I want to invest my savings for retirement.", "investment", True),
        ("Help me understand my monthly spending.", "budget", True),
        ("What is the current exchange rate?", "conversational_only", False),
        ("Can you recommend a low-risk investment product?", "investment", True),
        ("How much risk can I afford to take?", "risk_profiling", True),
        ("Where am I spending more than average in Ireland?", "budget", True),
        ("Give me a complete financial review please.", "full_advisory", True),
    ]

    correct = 0
    multi_agent = 0
    for i, (message, expected, seeded) in enumerate(scenarios):
        orch = Orchestrator(LLMClient(force_mock=True), session_id=f"vba-rq4-{i}")
        if seeded:
            orch._session_state["user_features"].update(seed)
            orch._session_state["monthly_expenses"] = {
                "rent": 1400.0, "groceries": 480.0, "utilities": 190.0,
                "transport": 160.0, "entertainment": 220.0,
            }
        result = orch.process_turn(message)
        if result.routing_decision.value == expected:
            correct += 1
        specialists = {"RiskProfilingAgent", "InvestmentAgent", "BudgetAgent"}
        if len(specialists.intersection(result.agents_invoked)) >= 2:
            multi_agent += 1

    rate = correct / len(scenarios)
    verdict = "PASS" if rate >= 0.8 else "FAIL"
    record(
        "RQ4 scenarios route correctly with NO live LLM (graceful degradation)",
        f"{correct}/{len(scenarios)} correct ({rate:.0%}); "
        f"{multi_agent} turn(s) invoked 2+ specialists",
        verdict,
    )


def main() -> int:
    probes = [
        ("routing gate", probe_routing_gate),
        ("audit evidence", probe_intent_preserved_in_audit),
        ("upstream hallucination", probe_hallucination_on_upstream_failure),
        ("planner data gap", probe_planner_data_gap),
        ("rq4 end to end", probe_rq4_routing_end_to_end),
    ]
    print(f"\n  tree: {os.getcwd()}\n")
    for name, fn in probes:
        try:
            fn()
        except Exception as exc:
            record(name, f"probe raised {type(exc).__name__}: {exc}", "ERR")
            traceback.print_exc(limit=2)
        print()

    failed = sum(1 for _, _, v in RESULTS if v != "PASS")
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} probes PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
