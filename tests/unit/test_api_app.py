"""
Day 10 (UI) tests: api/app.py's HTTP contract.

WHAT THESE TESTS ARE FOR
    Not the underlying pipeline — Days 6-9's own test files cover that.
    This file is specifically about the SERIALISATION boundary: does the
    JSON api/app.py returns actually have the shape static/index.html's
    JS expects, does GET /trace/{turn_id} actually return what tracing
    captured, and does the approval cycle (gate -> approve -> delivered
    on the next /chat call) work over real HTTP, not just Python calls.

    Every test here uses TestClient against the REAL app object (the
    same one uvicorn would serve) — not a hand-built substitute — so a
    passing test means the actual wiring works, not just that the
    pieces would work if wired correctly.

WHY EACH TEST USES A DISTINCT session_id
    api/app.py's SessionRegistry is a module-level singleton (deliberately
    — see its own docstring on why one registry must be shared, not
    reconstructed per request). That singleton persists across every test
    in this file, since nothing re-imports api.app between tests. Reusing
    a session_id across two unrelated tests would leak one test's
    Orchestrator (and its conversation_history, computed risk_profile,
    ...) into the next — so every test picks its own.

RUNNING
    python -m pytest tests/unit/test_api_app.py -v
"""
from __future__ import annotations

import pytest
#import fastapi
#import httpx

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agents.base_agent import AgentResult  # noqa: E402
from utils import trace  # noqa: E402


@pytest.fixture
def client(audit_tmp_dir):
    import api.app as appmod
    return TestClient(appmod.app), appmod


def _rig_investment_turn(appmod, session_id, expected_return_pct=50.0):
    """
    Same technique as test_approval_gate.py's integration tests: force a
    known InvestmentAgent payload so the gate fires deterministically,
    without depending on mock-mode hallucination-flagging noise.
    """
    from orchestrator.orchestrator import RoutingDecision

    orch = appmod._registry.get_or_create(session_id, customer_id=f"cust-{session_id}")
    orch._session_state["user_features"] = {
        "age": 34, "income": 55000, "employment_status": "employed",
        "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
        "loss_tolerance": 4, "financial_knowledge_score": 3,
    }
    orch._classify_intent = lambda msg: (RoutingDecision.INVESTMENT, "forced", 1.0)
    real_execute = orch._execute_agent

    def fake_execute(agent_name, context):
        if agent_name == "InvestmentAgent":
            payload = {
                "status": "complete", "risk_class": context.get("risk_class", "moderate"),
                "shortlist": [{"name": "TestFund", "category": "corporate_bond",
                              "product_id": "X1", "expected_return_pct": expected_return_pct}],
                "synthesis": "a recommendation", "deliverable": True,
                "hallucination_flagged": False,
            }
            return AgentResult(agent_name=agent_name, success=True, payload=payload), False
        return real_execute(agent_name, context)

    orch._execute_agent = fake_execute
    return orch


# ── Basic contract ──────────────────────────────────────────────────────

class TestBasicContract:

    def test_root_serves_the_html_file(self, client):
        c, _ = client
        r = c.get("/")
        assert r.status_code == 200
        assert "<title>HALO" in r.text

    def test_chat_returns_the_shape_the_frontend_expects(self, client):
        c, _ = client
        r = c.post("/chat", json={"session_id": "contract-1", "message": "hello"})
        assert r.status_code == 200
        data = r.json()
        expected_keys = {
            "session_id", "turn_id", "routing_decision", "final_response",
            "awaiting_approval", "success", "total_duration_ms", "plan",
            "agents_invoked", "agent_results", "skipped", "collaboration_events",
            "conflicts", "constraint_violations", "recovered_agents",
            "explanation_layers", "audit_ref", "delivered_approvals",
        }
        assert expected_keys <= set(data.keys())
        assert set(data["plan"].keys()) == {"source", "accepted", "steps"}
        assert isinstance(data["agent_results"], list)
        assert isinstance(data["explanation_layers"], list)
        assert isinstance(data["delivered_approvals"], list)

    def test_agent_result_payload_is_included_in_full(self, client):
        """The whole point of this panel is showing what an agent
        produced — a trimmed/summary payload would defeat it."""
        c, appmod = client
        _rig_investment_turn(appmod, "contract-2", expected_return_pct=3.0)  # stays under the gate
        r = c.post("/chat", json={"session_id": "contract-2", "message": "should I invest?"})
        data = r.json()
        inv = next(a for a in data["agent_results"] if a["agent_name"] == "InvestmentAgent")
        assert inv["payload"]["shortlist"][0]["name"] == "TestFund"

    def test_audit_ref_names_a_real_session_file_and_line(self, client):
        c, _ = client
        r = c.post("/chat", json={"session_id": "contract-3", "message": "hello"})
        ref = r.json()["audit_ref"]
        assert ref.startswith("session_contract-3.jsonl:")
        assert int(ref.split(":")[-1]) > 0

    def test_session_is_reused_across_calls_not_recreated(self, client):
        c, appmod = client
        c.post("/chat", json={"session_id": "contract-4", "message": "first"})
        c.post("/chat", json={"session_id": "contract-4", "message": "second"})
        orch = appmod._registry.get_or_create("contract-4")
        assert orch._session_state["turn_count"] == 2


# ── Approvals router reuse (Day 8 -> Day 10) ───────────────────────────

class TestApprovalsRouterReuse:
    """
    Confirms api/approvals_router.py's routes are reachable on the MAIN
    app at their bare paths (/approvals, not /api/approvals or similar)
    — the specific thing that broke on the first attempt at including it
    (app.include_router(other_app.router) silently included nothing but
    the auto-generated docs routes; fixed by exporting an APIRouter
    instead — see that module's docstring).
    """

    def test_approvals_list_reachable_on_the_main_app(self, client):
        c, _ = client
        r = c.get("/approvals")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_approve_reachable_on_the_main_app(self, client):
        c, appmod = client
        _rig_investment_turn(appmod, "reuse-1")
        r = c.post("/chat", json={"session_id": "reuse-1", "message": "should I invest?"})
        turn_id = r.json()["turn_id"]
        assert r.json()["awaiting_approval"] is True

        r2 = c.post(f"/approvals/{turn_id}/approve", json={"reviewer_note": "fine"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "approved"


# ── The full gate -> approve -> deliver cycle, over HTTP ───────────────

class TestApprovalCycleOverHTTP:

    def test_gated_turn_withholds_and_records_reasons(self, client):
        c, appmod = client
        _rig_investment_turn(appmod, "cycle-1")
        r = c.post("/chat", json={"session_id": "cycle-1", "message": "should I invest?"})
        data = r.json()
        assert data["awaiting_approval"] is True

        r2 = c.get(f"/approvals/{data['turn_id']}")
        assert r2.status_code == 200
        assert any("expected_return_pct" in reason for reason in r2.json()["reasons"])

    def test_approved_response_is_delivered_on_the_next_chat_call(self, client):
        """
        The build plan's own demo line — "approve it, watch it deliver"
        — as an HTTP test. This is the exact gap found and fixed while
        building this file: api/app.py did not originally call
        Orchestrator.collect_all_approved_responses() at all, so this
        would have failed (delivered_approvals always []) before that fix.
        """
        c, appmod = client
        _rig_investment_turn(appmod, "cycle-2")
        r1 = c.post("/chat", json={"session_id": "cycle-2", "message": "should I invest?"})
        turn_id = r1.json()["turn_id"]
        assert r1.json()["delivered_approvals"] == []

        c.post(f"/approvals/{turn_id}/approve", json={"reviewer_note": "ok"})

        # A route that won't itself re-gate, so this turn's own
        # delivered_approvals reflects ONLY what turn 1 left pending.
        from orchestrator.orchestrator import RoutingDecision
        orch = appmod._registry.get_or_create("cycle-2")
        orch._classify_intent = lambda msg: (RoutingDecision.CONVERSATIONAL_ONLY, "forced", 1.0)

        r2 = c.post("/chat", json={"session_id": "cycle-2", "message": "thanks"})
        assert len(r2.json()["delivered_approvals"]) == 1

    def test_rejected_response_is_never_delivered(self, client):
        c, appmod = client
        _rig_investment_turn(appmod, "cycle-3")
        r1 = c.post("/chat", json={"session_id": "cycle-3", "message": "should I invest?"})
        turn_id = r1.json()["turn_id"]

        c.post(f"/approvals/{turn_id}/reject", json={"reviewer_note": "not suitable"})

        from orchestrator.orchestrator import RoutingDecision
        orch = appmod._registry.get_or_create("cycle-3")
        orch._classify_intent = lambda msg: (RoutingDecision.CONVERSATIONAL_ONLY, "forced", 1.0)
        r2 = c.post("/chat", json={"session_id": "cycle-3", "message": "thanks"})
        assert r2.json()["delivered_approvals"] == []

    def test_clean_turn_is_never_gated(self, client):
        c, appmod = client
        _rig_investment_turn(appmod, "cycle-4", expected_return_pct=3.0)
        r = c.post("/chat", json={"session_id": "cycle-4", "message": "should I invest?"})
        assert r.json()["awaiting_approval"] is False


# ── Trace capture ────────────────────────────────────────────────────────

class TestTraceEndpoint:

    def test_trace_is_populated_after_a_real_chat_call(self, client):
        """
        api/app.py force-enables TRACE=true, TRACE_FORMAT=json at import
        (see its module docstring) — this test would fail with events==[]
        if that override were ever removed, which is exactly the failure
        mode it exists to prevent.
        """
        c, _ = client
        assert trace.trace_config.enabled is True
        assert trace.trace_config.format == "json"

        r = c.post("/chat", json={"session_id": "trace-1", "message": "hello"})
        turn_id = r.json()["turn_id"]

        r2 = c.get(f"/trace/{turn_id}")
        assert r2.status_code == 200
        events = r2.json()
        assert len(events) > 0
        assert all(e["turn_id"] == turn_id for e in events)

    def test_unknown_turn_id_returns_empty_list_not_404(self, client):
        c, _ = client
        r = c.get("/trace/never-happened")
        assert r.status_code == 200
        assert r.json() == []

    def test_events_are_scoped_to_their_own_turn(self, client):
        c, _ = client
        r1 = c.post("/chat", json={"session_id": "trace-2", "message": "first"})
        r2 = c.post("/chat", json={"session_id": "trace-2", "message": "second"})
        t1, t2 = r1.json()["turn_id"], r2.json()["turn_id"]
        assert t1 != t2

        events1 = trace.get_captured_events(t1)
        events2 = trace.get_captured_events(t2)
        assert all(e["turn_id"] == t1 for e in events1)
        assert all(e["turn_id"] == t2 for e in events2)