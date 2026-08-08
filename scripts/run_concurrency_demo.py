"""
scripts/run_concurrency_demo.py

Demonstrates and measures the pipeline under genuinely concurrent load —
N customers, each in their own session, hitting Orchestrator.process_turn()
at the same time via a real thread pool. Not a synthetic benchmark of an
isolated function: this runs the FULL pipeline (routing, planning, agent
execution, constraint validation, memory persistence, audit logging) per
thread, the same code path api/app.py's /chat endpoint uses for a real
concurrent request (FastAPI runs sync endpoints via run_in_threadpool —
this is that same execution model, just driven directly rather than
through HTTP).

WHAT THIS IS ACTUALLY CHECKING, BEYOND "IT DIDN'T CRASH"
    1. turn_id uniqueness under real concurrency — the thing an earlier,
       unrelated report ("all events getting the same turn_id") worried
       about, before turning out to be a misread session_id, not a real
       collision. This settles it with actual concurrent execution
       (uuid4-based generation, and utils/trace.py's contextvars), not
       just reasoning about it.
    2. Whether the SHARED SQLite stores (orchestrator/approvals.py,
       data/customer_memory.py) hold up under concurrent writes from many
       threads — every turn writes to customer_memory.db, and Python's
       sqlite3 default busy-timeout (5s) can still surface "database is
       locked" under enough write contention if it's ever exceeded. This
       script reports it if it happens rather than silently retrying.
    3. Routing/agent-execution diversity under load — customers are given
       different message types (risk / budget / investment / explanation)
       so this isn't 100 identical requests, which would exercise far
       less of the pipeline concurrently than a real multi-customer
       population would.

WHY MOCK MODE BY DEFAULT
    The question this script answers is "does the SYSTEM behave correctly
    under concurrency" — routing, persistence, thread safety — not "is
    the LLM's answer good", which is what the rest of this test suite
    already covers in mock mode for the same reason (see conftest.py's
    _no_live_api_in_tests). Real-mode concurrent calls to Groq/Ollama
    would also cost tokens and time for no additional signal on the
    question this script exists to answer. --real is available if you
    specifically want to see the pipeline under concurrent load against
    real providers too.

USAGE
    python scripts/run_concurrency_demo.py
    python scripts/run_concurrency_demo.py --customers 50 --workers 10
    python scripts/run_concurrency_demo.py --real --workers 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR  # noqa: E402
from data.customer_store import CustomerStore  # noqa: E402

RESULTS_PATH = ROOT_DIR / "results" / "concurrency_demo.json"

# Rotated across customers so concurrent load exercises different parts
# of the pipeline at once, not 100 copies of the same request.
MESSAGES = [
    "Can you assess my risk profile?",
    "Where does my money go?",
    "Should I invest my savings?",
    "Can you explain my risk level?",
]


def _one_customer_turn(customer_id: str, message: str, force_mock: bool) -> dict:
    """
    Runs in a worker thread. Builds its OWN Orchestrator (own session,
    own specialist/orchestrator LLMClient instances) — deliberately not
    sharing one Orchestrator across threads, since a single session is
    meant to be one customer's conversation, not a pool resource. The
    shared state that DOES cross threads (ApprovalStore, CustomerMemory
    Store, both process-wide singletons — see their own modules) is
    exactly what point 2 in the module docstring is checking.
    """
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient

    started = time.perf_counter()
    try:
        client = LLMClient(force_mock=force_mock) if force_mock else LLMClient()
        orch = Orchestrator(client, session_id=customer_id, customer_id=customer_id)
        result = orch.process_turn(message)
        return {
            "customer_id": customer_id,
            "message": message,
            "turn_id": result.turn_id,
            "routing": result.routing_decision.value,
            "agents_invoked": result.agents_invoked,
            "success": result.success,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": None,
        }
    except Exception as exc:
        return {
            "customer_id": customer_id,
            "message": message,
            "turn_id": None,
            "routing": None,
            "agents_invoked": [],
            "success": False,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--customers", type=int, default=100,
                     help="How many SYNTH_* customers to use (must already "
                          "exist — run generate_synthetic_customer_base.py "
                          "first if this is the first time).")
    ap.add_argument("--workers", type=int, default=20,
                     help="Concurrent thread pool size.")
    ap.add_argument("--real", action="store_true",
                     help="Use real LLM providers instead of mock mode — "
                          "see module docstring for why mock is the default.")
    args = ap.parse_args()

    store = CustomerStore()
    customer_ids = [
        f"SYNTH_{i:05d}" for i in range(1, args.customers + 1)
        if f"SYNTH_{i:05d}" in store
    ]
    if len(customer_ids) < args.customers:
        print(
            f"Only found {len(customer_ids)}/{args.customers} SYNTH_* "
            f"customers in CustomerStore — run "
            f"`python scripts/generate_synthetic_customer_base.py "
            f"--count {args.customers}` first if you haven't."
        )
        if not customer_ids:
            return 1

    print(f"Running {len(customer_ids)} customers through {args.workers} "
          f"concurrent workers ({'REAL' if args.real else 'mock'} LLM mode)...")

    wall_start = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _one_customer_turn, cid, MESSAGES[i % len(MESSAGES)], not args.real,
            ): cid
            for i, cid in enumerate(customer_ids)
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_clock_s = time.perf_counter() - wall_start

    # --- Point 1: turn_id uniqueness under real concurrency ---
    turn_ids = [r["turn_id"] for r in results if r["turn_id"] is not None]
    duplicate_turn_ids = len(turn_ids) - len(set(turn_ids))

    # --- Point 2: any SQLite contention surfaced as an actual error? ---
    locked_errors = [
        r for r in results
        if r["error"] and "locked" in r["error"].lower()
    ]

    succeeded = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    durations = [r["duration_ms"] for r in succeeded]

    routing_breakdown: dict[str, int] = {}
    for r in succeeded:
        routing_breakdown[r["routing"]] = routing_breakdown.get(r["routing"], 0) + 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "real" if args.real else "mock",
        "workers": args.workers,
        "total_customers": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "wall_clock_s": round(wall_clock_s, 2),
        "throughput_requests_per_s": round(len(results) / wall_clock_s, 2) if wall_clock_s else None,
        "turn_id_uniqueness": {
            "total_turn_ids": len(turn_ids),
            "unique_turn_ids": len(set(turn_ids)),
            "duplicates_found": duplicate_turn_ids,
        },
        "sqlite_lock_errors_under_concurrency": len(locked_errors),
        "latency_ms": {
            "mean": round(statistics.mean(durations), 1) if durations else None,
            "median": round(statistics.median(durations), 1) if durations else None,
            "p95": round(_percentile(durations, 0.95), 1) if durations else None,
            "max": round(max(durations), 1) if durations else None,
        },
        "routing_breakdown": routing_breakdown,
        "errors": [r for r in failed],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print()
    print(f"{'='*70}")
    print(f"CONCURRENCY DEMO — {summary['total_customers']} customers, "
          f"{args.workers} workers, {summary['mode']} mode")
    print(f"{'='*70}")
    print(f"  wall clock         : {summary['wall_clock_s']}s")
    print(f"  throughput         : {summary['throughput_requests_per_s']} req/s")
    print(f"  succeeded / failed : {summary['succeeded']} / {summary['failed']}")
    print(f"  turn_id collisions : {duplicate_turn_ids} "
          f"(out of {len(turn_ids)} turn_ids — {'CLEAN' if duplicate_turn_ids == 0 else 'FOUND COLLISIONS'})")
    print(f"  sqlite lock errors : {len(locked_errors)}")
    if durations:
        print(f"  latency (ms)       : mean={summary['latency_ms']['mean']} "
              f"median={summary['latency_ms']['median']} "
              f"p95={summary['latency_ms']['p95']} max={summary['latency_ms']['max']}")
    print(f"  routing breakdown  : {routing_breakdown}")
    if failed:
        print(f"\n  {len(failed)} failure(s):")
        for r in failed[:10]:
            print(f"    {r['customer_id']}: {r['error']}")
    print(f"\nFull results: {RESULTS_PATH}")
    print(f"{'='*70}")

    return 0 if not failed and duplicate_turn_ids == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
