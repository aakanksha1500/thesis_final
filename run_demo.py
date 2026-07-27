#!/usr/bin/env python3
"""
run_demo.py — a terminal entry point for the advisory pipeline.

The repository previously had no way to run the system outside pytest. This
is the smallest thing that fixes that: it builds an Orchestrator, feeds it
messages, and prints the routing decision, the agents invoked, and the reply.

USAGE
    python run_demo.py                          # scripted demo conversation
    python run_demo.py --interactive            # type your own messages
    python run_demo.py --customer DEMO_GC_042   # start from a known customer
    python run_demo.py --debug                  # DEBUG-level logs
    python run_demo.py -m "Should I invest?"    # one message and exit
"""
from __future__ import annotations

import argparse
import os
import sys


def _routings():
    from orchestrator.orchestrator import RoutingDecision
    return list(RoutingDecision)

# Feature-complete profile so RiskProfilingAgent does not refuse to classify.
# (It returns status="incomplete" if ANY required feature is missing — that is
# deliberate: it will not guess.)
DEMO_FEATURES = {
    "age": 34,
    "income": 55000,
    "employment_status": "employed",
    "dependents": 0,
    "existing_debt": 5000,
    "investment_horizon": 15,
    "loss_tolerance": 4,
    "financial_knowledge_score": 3,
}

DEMO_EXPENSES = {
    "housing": 1400, "food": 520, "transport": 260,
    "utilities": 180, "entertainment": 220, "other": 300,
}

SCRIPT = [
    "Hello, I'd like some help with my finances.",
    "Can you assess my risk profile?",
    "I want to invest my savings for retirement.",
    "Why did you recommend that?",
    "Where am I spending more than the Irish average?",
]

BAR = "=" * 78


def show(turn_no: int, message: str, result) -> None:
    print(f"\n{BAR}")
    print(f"TURN {turn_no}  ·  YOU: {message}")
    print(BAR)
    print(f"  routing    : {result.routing_decision.value}")
    print(f"  agents     : {', '.join(result.agents_invoked) or '(none)'}")
    print(f"  duration   : {result.total_duration_ms:.0f} ms")
    if result.conflicts:
        print(f"  conflicts  : {[c['type'] for c in result.conflicts]}")
    hard = [v for v in result.constraint_violations if v.get("severity") == "hard_block"]
    if hard:
        print(f"  BLOCKED    : {[v['rule_id'] for v in hard]}")
    if result.recovered_agents:
        print(f"  recovered  : {result.recovered_agents}")

    for ar in result.agent_results:
        status = ar.payload.get("status", "-")
        mark = "ok " if ar.success else "ERR"
        print(f"    [{mark}] {ar.agent_name:<22} status={status:<12} "
              f"{ar.duration_ms:6.0f}ms  {ar.tokens_used:5d} tok")
        if ar.error:
            print(f"          └─ {ar.error}")

    print(f"\n  ASSISTANT: {result.final_response}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--message", help="Send one message and exit")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive session (Ctrl-D or 'quit' to exit)")
    parser.add_argument("-c", "--customer", help="Existing customer id, e.g. DEMO_GC_042")
    parser.add_argument("-s", "--session", default="demo-session",
                        help="Session id (names the audit log file)")
    parser.add_argument("--debug", action="store_true", help="DEBUG-level logging")
    parser.add_argument("--no-features", action="store_true",
                        help="Do NOT pre-fill demo features — see the elicitation path")
    parser.add_argument("--force-routing", choices=[r.value for r in _routings()],
                        help="Bypass the LLM intent classifier and force a route. "
                             "Essential in mock mode, where the classifier cannot "
                             "classify and everything falls back to conversational_only.")
    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG"] = "true"

    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient

    client = LLMClient()
    print(f"\nLLM mode: {client.mode}"
          + ("   (no API key found — responses will be [MOCK RESPONSE]. "
             "Set a key in .env for real output.)" if client.mode == "mock" else ""))

    orch = Orchestrator(client, session_id=args.session, customer_id=args.customer)

    if args.force_routing:
        # Pin the routing decision so the full specialist pipeline can be
        # demonstrated without a live LLM. Everything downstream of routing —
        # agent execution, handoffs, conflicts, constraints — runs for real.
        from orchestrator.orchestrator import RoutingDecision
        forced = RoutingDecision(args.force_routing)
        orch._classify_intent = lambda msg: (forced, f"forced:{forced.value}", 1.0)
        print(f"Routing forced to: {forced.value}")

    if not args.no_features and not orch._session_state.get("user_features"):
        orch._session_state["user_features"] = dict(DEMO_FEATURES)
        print("Pre-filled demo user_features (use --no-features to skip).")

    def send(n: int, msg: str) -> None:
        # BudgetAgent needs expenses; the Orchestrator has no channel for them
        # yet, so they are injected into session state for the demo.
        orch._session_state.setdefault("monthly_expenses", DEMO_EXPENSES)
        orch._session_state.setdefault("monthly_income", DEMO_FEATURES["income"] / 12)
        show(n, msg, orch.process_turn(msg))

    if args.message:
        send(1, args.message)
    elif args.interactive:
        print("Interactive mode — type a message, or 'quit' to exit.")
        n = 0
        while True:
            try:
                msg = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if msg.lower() in {"quit", "exit", "q"}:
                break
            if msg:
                n += 1
                send(n, msg)
    else:
        for n, msg in enumerate(SCRIPT, start=1):
            send(n, msg)

    print(f"{BAR}\nAudit trail: {orch.audit_log.log_path}\n{BAR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
