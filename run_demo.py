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
    python run_demo.py --persona normal          # named customer archetype
    python run_demo.py --persona stale -i        # ...answered interactively
    python run_demo.py --persona normal --push   # bank-push instead of lookup
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

# DEMO_EXPENSES = {
#     "housing": 1400, "food": 520, "transport": 260,
#     "utilities": 180, "entertainment": 220, "other": 300,
# }

SCRIPT = [
    "Hello, I'd like some help with my finances.",
    "Can you assess my risk profile?",
    "I want to invest my savings for retirement.",
    "Why did you recommend that?",
    "Where am I spending more than the Irish average?",
]

BAR = "=" * 78

PERSONAS: dict[str, dict] = {
    "normal":  {"customer_id": "DEMO_GC_042",   "no_features": False},
    "average": {"customer_id": "DEMO_GC_107",   "no_features": False},
    "stale":   {"customer_id": "DEMO_GMSC_318", "no_features": False},
    "new":     {"customer_id": None,            "no_features": True},
}

def _month(date_str: str) -> int:
    return int(date_str[5:7])


def _load_persona_transactions(name: str) -> list[dict]:
    """
    Derive a persona's transaction history from the TXN_BENCHMARK scenario
    by filtering which months are kept. Deliberately NOT sourced from
    scripts/generate_transactions.py directly — that script's output also
    feeds committed dissertation evidence (results/*.json); regenerating it
    to add new scenarios risks perturbing numbers already written up.
    Filtering the existing generated file is read-only and additive.
    """
    from data.transaction_store import TransactionStore

    if name == "new":
        return []

    base = TransactionStore().lookup("TXN_BENCHMARK")
    if not base:
        print(
            "WARNING: TXN_BENCHMARK not found in data/processed/transactions.json "
            "— run `python scripts/generate_transactions.py` first. Falling back "
            "to an empty transaction history for this persona."
        )
        return []

    if name == "normal":
            return base
    if name == "average":
        # 5 consecutive months -> "medium" coverage tier, fully active
        # within that window.
        months = {1, 2, 3, 4, 5}
    elif name == "stale":
        # Scattered across the full year, not a contiguous slice: coverage
        # still spans ~12 months (first txn in Jan, last in Dec), so
        # coverage_tier reads "high" -> BudgetAgent proceeds to aggregate
        # normally despite only 3 of those months having any activity.
        # density_score/confidence_score end up low but are NOT consulted
        # by BudgetAgent's gate or by the disclosure text. See module
        # docstring — this is a verified finding, not the intuitive
        # "asks a follow-up question" behaviour.
        months = {1, 6, 12}
    else:
        raise ValueError(f"Unknown persona: {name!r}")

    filtered = [t for t in base if _month(t["date"]) in months]
    print(f"Persona {name!r}: {len(filtered)} transactions across months {sorted(months)} "
        f"(filtered from TXN_BENCHMARK's {len(base)})")
    return filtered

def _build_push_context(customer_id: str, store) -> dict:
    """
    Build a bank-push customer_context payload from an existing
    CustomerStore record, withholding loss_tolerance so
    Orchestrator._load_customer()'s proxy-derivation path runs instead of
    just re-reading a value nothing in a real bank's system would have.
    """
    record = store.lookup(customer_id)
    if record is None:
        raise SystemExit(
            f"--push: no CustomerStore record for customer_id={customer_id!r} "
            f"to build a payload from."
        )
    context = dict(record["features"])
    context.pop("loss_tolerance", None)
    context["name"] = f"Demo {customer_id}"
    return context

def show(turn_no: int, message: str, result) -> None:
    from config.settings import settings
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
    if result.skipped:
        for s in result.skipped:
            print(f"  skipped    : {s['agent']} — {s['reason']}")
    if result.collaboration_events:
        for e in result.collaboration_events:
            if e.get("retried"):
                print(f"  collab     : {e['requester']} retried after collaboration")
            else:
                mark = "resolved" if e.get("resolved") else "unresolved"
                print(f"  collab     : {e['requester']} needs {e['needs'][0]!r} "
                      f"→ {e.get('producer', '(none found)')} [{mark}]")

    if result.final_response == settings.approval.withheld_message:
        print(f"  HELD       : turn_id={result.turn_id} — awaiting approval. "
              f"Review with:\n"
              f"               python run_demo.py --list-approvals\n"
              f"               python run_demo.py --approve {result.turn_id}")

    for ar in result.agent_results:
        status = ar.payload.get("status", "-")
        mark = "ok " if ar.success else "ERR"
        print(f"    [{mark}] {ar.agent_name:<22} status={status:<12} "
              f"{ar.duration_ms:6.0f}ms  {ar.tokens_used:5d} tok")
        if ar.error:
            print(f"          \u2514\u2500 {ar.error}")

        sufficiency = ar.payload.get("data_sufficiency")
        if sufficiency:
            print(
                f"          \u2514\u2500 data_sufficiency: tier={sufficiency['coverage_tier']:<12} "
                f"coverage={sufficiency['coverage_score']:.2f} "
                f"density={sufficiency['density_score']:.2f} "
                f"confidence={sufficiency['confidence_score']:.2f}"
            )

    print(f"\n  ASSISTANT: {result.final_response}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--message", help="Send one message and exit")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Interactive session (Ctrl-D or 'quit' to exit)")
    parser.add_argument("-c", "--customer", help="Existing customer id, e.g. DEMO_GC_042")
    parser.add_argument("-p", "--persona", choices=list(PERSONAS),
                        help="Named customer archetype — sets --customer and the "
                             "transaction history together. See module docstring.")
    parser.add_argument("--push", action="store_true",
                        help="Simulate a bank push (customer_context) instead of a "
                             "CustomerStore lookup. Requires --persona (not 'new').")
    parser.add_argument("-s", "--session", default="demo-session",
                        help="Session id (names the audit log file)")
    parser.add_argument("--debug", action="store_true", help="DEBUG-level logging")
    parser.add_argument("--no-features", action="store_true",
                        help="Do NOT pre-fill demo features — see the elicitation path")
    parser.add_argument("--force-routing", choices=[r.value for r in _routings()],
                        help="Bypass the LLM intent classifier and force a route. "
                             "Essential in mock mode, where the classifier cannot "
                             "classify and everything falls back to conversational_only.")
    parser.add_argument("--list-approvals", action="store_true",
                        help="Day 8: list pending approvals across all sessions, and exit")
    parser.add_argument("--approve", metavar="TURN_ID",
                        help="Day 8: approve a held turn, deliver it, and exit")
    parser.add_argument("--reject", metavar="TURN_ID",
                        help="Day 8: reject a held turn, and exit")
    parser.add_argument("--reviewer-note", default="",
                        help="Note attached to an --approve/--reject decision")
    args = parser.parse_args()

    if args.list_approvals or args.approve or args.reject:
        from orchestrator.approvals import get_approval_store
        store = get_approval_store()

        if args.list_approvals:
            pending = store.list_pending()
            if not pending:
                print("No approvals pending.")
                return 0
            for p in pending:
                print(f"\n{BAR}")
                print(f"turn_id={p.turn_id}  session={p.session_id}  created={p.created_at}")
                print(f"  reasons: {p.reasons}")
                print(f"  draft (withheld from customer): {p.draft_response[:200]}")
            print(f"\n{BAR}\n{len(pending)} pending.")
            return 0

        if args.approve:
            approved = store.approve(args.approve, reviewer_note=args.reviewer_note)
            delivered = store.mark_delivered(args.approve)
            print(f"Approved {args.approve} (session={delivered.session_id}).")
            print(f"\n{BAR}\nDELIVERED:\n{BAR}\n{delivered.draft_response}\n")
            print(
                "If that session is running interactively, its next turn "
                "will surface this same text automatically — see "
                "Orchestrator.collect_all_approved_responses()."
            )
            return 0

        if args.reject:
            rejected = store.reject(args.reject, reviewer_note=args.reviewer_note)
            print(f"Rejected {args.reject} (session={rejected.session_id}). "
                  f"The customer will not receive this response.")
            return 0

    if args.push and not args.persona:
        parser.error("--push requires --persona (it needs a feature set to push)")
    if args.push and args.persona == "new":
        parser.error("--push doesn't apply to --persona new — there is no existing "
                      "record to push; omit --customer/--persona instead for the "
                      "unknown-customer path")
    if args.persona and args.customer and args.customer != PERSONAS[args.persona]["customer_id"]:
        print(f"Note: --persona {args.persona!r} overrides --customer "
              f"({PERSONAS[args.persona]['customer_id']!r} is used instead of "
              f"{args.customer!r})")    

    if args.debug:
        os.environ["DEBUG"] = "true"

    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    from data.customer_store import CustomerStore

    client = LLMClient()
    print(f"\nLLM mode: {client.mode}"
          + ("   (no API key found — responses will be [MOCK RESPONSE]. "
             "Set a key in .env for real output.)" if client.mode == "mock" else ""))

    # orch = Orchestrator(client, session_id=args.session, customer_id=args.customer)
    store = CustomerStore()

    customer_id = args.customer
    customer_context = None
    no_features = args.no_features

    if args.persona:
        spec = PERSONAS[args.persona]
        customer_id = spec["customer_id"]
        no_features = no_features or spec["no_features"]
        transactions = _load_persona_transactions(args.persona)

        if args.push and customer_id:
            customer_context = _build_push_context(customer_id, store)
            customer_id = f"{customer_id}-PUSH"  # distinct id: a push, not a lookup;
                                                  # also keeps update_customer_features()
                                                  # from writing back onto the curated
                                                  # demo profile mid-conversation
            print(f"Push-mode: sending customer_context for persona={args.persona!r} "
                  f"(loss_tolerance withheld \u2014 proxy will be derived)")
        elif customer_id:
            print(f"Persona: {args.persona} -> customer_id={customer_id}")
        else:
            print(f"Persona: {args.persona} -> unknown customer, elicitation from scratch")
    else:
        # No persona named: default to the same full transaction history the
        # original script effectively demonstrated, so a bare
        # `python run_demo.py` still shows a complete budget answer.
        transactions = _load_persona_transactions("normal")

    orch = Orchestrator(
        client, session_id=args.session, customer_store=store,
        customer_id=customer_id, customer_context=customer_context,
    )
 

    if args.force_routing:
        # Pin the routing decision so the full specialist pipeline can be
        # demonstrated without a live LLM. Everything downstream of routing —
        # agent execution, handoffs, conflicts, constraints — runs for real.
        from orchestrator.orchestrator import RoutingDecision
        forced = RoutingDecision(args.force_routing)
        orch._classify_intent = lambda msg: (forced, f"forced:{forced.value}", 1.0)
        print(f"Routing forced to: {forced.value}")

    if not no_features and not orch._session_state.get("user_features"):
        orch._session_state["user_features"] = dict(DEMO_FEATURES)
        print("Pre-filled demo user_features (use --no-features to skip).")

    def send(n: int, msg: str) -> None:
        # BudgetAgent needs expenses; the Orchestrator has no channel for them
        # yet, so they are injected into session state for the demo.
        # orch._session_state.setdefault("monthly_expenses", DEMO_EXPENSES)
        # orch._session_state.setdefault("monthly_income", DEMO_FEATURES["income"] / 12)
        orch._session_state.setdefault("transactions", transactions)
        show(n, msg, orch.process_turn(msg))

    if args.message:
        send(1, args.message)
    elif args.interactive:
        print("Interactive mode — type a message, or 'quit' to exit.")
        if no_features or (args.persona == "new"):
            print(
                "New/incomplete customer: the system won't volunteer what it "
                "needs until you ask it something requiring a specialist — "
                "try \"Can you assess my risk profile?\" or \"I want to "
                "invest for retirement\" to see it ask for what's missing."
            )
        n = 0
        while True:
            delivered = orch.collect_all_approved_responses()
            for text in delivered:
                print(f"\n{BAR}\nA held response was just approved and is now "
                      f"ready:\n{BAR}\n{text}\n")
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
