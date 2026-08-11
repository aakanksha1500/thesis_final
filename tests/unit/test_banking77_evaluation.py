"""
Human approval gate tests (Day 8).

WHAT THESE TESTS ARE FOR
    Three layers, tested separately then together:
    1. ApprovalStore's state machine in isolation — the SQLite persistence
       and the legal-transition rules, with nothing else involved.
    2. Orchestrator._check_approval_gate()'s four trigger conditions,
       each fed a synthetic agent_results/violations/conflicts shape
       directly — no LLM, no real agent execution.
    3. The full path through a real process_turn(): a gate firing must
       actually withhold the response, and the whole approve-then-
       collect cycle must hand back the exact original draft.

    _execute_agent is stubbed for the integration tests, same principle
    test_dynamic_routing.py and test_agent_collaboration.py use — what's
    under test is the gate's wiring into the turn, not any real agent's
    behaviour.

WHY LOW_CONFIDENCE_AGGRESSIVE IS CHECKED VIA `conflicts`, NOT `risk_class`
    ConflictResolver already downgrades risk_class to "moderate" for
    routing before the gate ever runs (orchestrator/conflict_resolver.py)
    — checking risk_class directly here would never see the aggressive
    classification that triggered the downgrade in the first place. See
    settings.approval's docstring and _check_approval_gate()'s.

RUNNING
    python -m pytest tests/unit/test_approval_gate.py -v
"""
from __future__ import annotations

import pytest

from agents.base_agent import AgentResult
from config.settings import settings
from orchestrator.approvals import (
    ApprovalNotFoundError,
    ApprovalStore,
    ApprovalTransitionError,
    PendingApproval,
    set_approval_store,
)


@pytest.fixture
def approval_store(tmp_path, monkeypatch):
    """A fresh, isolated ApprovalStore, injected as the process-wide
    singleton so Orchestrator instances built during a test use it too."""
    monkeypatch.setattr(settings.approval, "db_path", tmp_path / "approvals.db", raising=False)
    store = ApprovalStore(db_path=tmp_path / "approvals.db")
    set_approval_store(store)
    yield store
    set_approval_store(None)


def _orch(session_id: str):
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    return Orchestrator(LLMClient(force_mock=True), session_id=session_id)


def _pending(turn_id="t1", session_id="s1", **overrides) -> PendingApproval:
    defaults = dict(
        turn_id=turn_id, session_id=session_id, reasons=["hard_block"],
        draft_response="the real answer", agent_results=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return PendingApproval(**defaults)


# ── ApprovalStore: state machine in isolation ───────────────────────────

class TestApprovalStoreStateMachine:

    def test_create_then_get_round_trips(self, approval_store):
        p = _pending(reasons=["hard_block", "hallucination detector flagged a claim"])
        approval_store.create(p)
        fetched = approval_store.get("t1")
        assert fetched.status == "pending"
        assert fetched.reasons == p.reasons
        assert fetched.draft_response == p.draft_response

    def test_get_unknown_turn_returns_none_not_raise(self, approval_store):
        assert approval_store.get("nope") is None

    def test_pending_to_approved_to_delivered_is_legal(self, approval_store):
        approval_store.create(_pending())
        approved = approval_store.approve("t1", reviewer_note="fine")
        assert approved.status == "approved"
        assert approved.reviewer_note == "fine"
        assert approved.decided_at is not None
        delivered = approval_store.mark_delivered("t1")
        assert delivered.status == "delivered"

    def test_pending_to_rejected_is_legal_and_terminal(self, approval_store):
        approval_store.create(_pending())
        rejected = approval_store.reject("t1", reviewer_note="not suitable")
        assert rejected.status == "rejected"
        with pytest.raises(ApprovalTransitionError):
            approval_store.mark_delivered("t1")
        with pytest.raises(ApprovalTransitionError):
            approval_store.approve("t1")

    def test_cannot_re_approve_an_already_approved_turn(self, approval_store):
        approval_store.create(_pending())
        approval_store.approve("t1")
        with pytest.raises(ApprovalTransitionError):
            approval_store.approve("t1")

    def test_cannot_deliver_a_still_pending_turn(self, approval_store):
        approval_store.create(_pending())
        with pytest.raises(ApprovalTransitionError):
            approval_store.mark_delivered("t1")

    def test_transition_on_unknown_turn_raises_not_found(self, approval_store):
        with pytest.raises(ApprovalNotFoundError):
            approval_store.approve("never-created")

    def test_list_pending_excludes_decided_turns(self, approval_store):
        approval_store.create(_pending(turn_id="a"))
        approval_store.create(_pending(turn_id="b"))
        approval_store.approve("a")
        pending = approval_store.list_pending()
        assert [p.turn_id for p in pending] == ["b"]

    def test_list_approved_is_exactly_the_awaiting_delivery_set(self, approval_store):
        approval_store.create(_pending(turn_id="a"))
        approval_store.create(_pending(turn_id="b"))
        approval_store.approve("a")
        approved = approval_store.list_approved()
        assert [p.turn_id for p in approved] == ["a"]

    def test_lists_are_scoped_by_session_id(self, approval_store):
        approval_store.create(_pending(turn_id="a", session_id="s1"))
        approval_store.create(_pending(turn_id="b", session_id="s2"))
        assert [p.turn_id for p in approval_store.list_pending(session_id="s1")] == ["a"]
        assert [p.turn_id for p in approval_store.list_pending(session_id="s2")] == ["b"]

    def test_reasons_and_agent_results_survive_the_json_round_trip(self, approval_store):
        p = _pending(
            reasons=["hard_block constraint violation"],
            agent_results=[{"agent_name": "InvestmentAgent", "success": True,
                            "payload": {"shortlist": [{"name": "X"}]}}],
        )
        approval_store.create(p)
        fetched = approval_store.get("t1")
        assert fetched.agent_results[0]["payload"]["shortlist"][0]["name"] == "X"


# ── Orchestrator._check_approval_gate(): each trigger, in isolation ────

class TestApprovalGateTriggers:

    def test_hard_block_violation_triggers(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-1")
        reasons = orch._check_approval_gate(
            [], [{"severity": "hard_block", "rule_id": "R001"}], [],
        )
        assert reasons and "hard_block" in reasons[0]

    def test_soft_violation_does_not_trigger(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-2")
        reasons = orch._check_approval_gate(
            [], [{"severity": "warning", "rule_id": "R002"}], [],
        )
        assert reasons == []

    def test_low_confidence_aggressive_conflict_triggers(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-3")
        reasons = orch._check_approval_gate(
            [], [], [{"type": "LOW_CONFIDENCE_AGGRESSIVE"}],
        )
        assert reasons and "aggressive" in reasons[0]

    def test_other_conflict_types_do_not_trigger(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-4")
        reasons = orch._check_approval_gate(
            [], [], [{"type": "RISK_PRODUCT_MISMATCH"}],
        )
        assert reasons == []

    def test_hallucination_flagged_triggers(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-5")
        inv = AgentResult(agent_name="InvestmentAgent", success=True,
                          payload={"hallucination_flagged": True, "shortlist": []})
        reasons = orch._check_approval_gate([inv], [], [])
        assert reasons and "hallucination" in reasons[0]

    def test_expected_return_above_ceiling_triggers(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-6")
        inv = AgentResult(agent_name="InvestmentAgent", success=True,
                          payload={"shortlist": [
                              {"name": "RiskyFund", "expected_return_pct": 99.0},
                          ]})
        reasons = orch._check_approval_gate([inv], [], [])
        assert reasons and "RiskyFund" in reasons[0]

    def test_expected_return_within_ceiling_does_not_trigger(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-7")
        inv = AgentResult(agent_name="InvestmentAgent", success=True,
                          payload={"shortlist": [
                              {"name": "SafeFund", "expected_return_pct": 3.0},
                          ]})
        assert orch._check_approval_gate([inv], [], []) == []

    def test_multiple_triggers_all_reported(self, approval_store, audit_tmp_dir):
        orch = _orch("gate-8")
        inv = AgentResult(agent_name="InvestmentAgent", success=True,
                          payload={"hallucination_flagged": True, "shortlist": []})
        reasons = orch._check_approval_gate(
            [inv], [{"severity": "hard_block"}], [{"type": "LOW_CONFIDENCE_AGGRESSIVE"}],
        )
        assert len(reasons) == 3

    def test_disabled_config_suppresses_every_trigger(self, approval_store, audit_tmp_dir, monkeypatch):
        monkeypatch.setattr(settings.approval, "enabled", False)
        orch = _orch("gate-9")
        inv = AgentResult(agent_name="InvestmentAgent", success=True,
                          payload={"hallucination_flagged": True,
                                   "shortlist": [{"name": "X", "expected_return_pct": 99.0}]})
        reasons = orch._check_approval_gate(
            [inv], [{"severity": "hard_block"}], [{"type": "LOW_CONFIDENCE_AGGRESSIVE"}],
        )
        assert reasons == []

    def test_individual_trigger_can_be_switched_off(self, approval_store, audit_tmp_dir, monkeypatch):
        monkeypatch.setattr(settings.approval, "gate_on_hard_block", False)
        orch = _orch("gate-10")
        reasons = orch._check_approval_gate(
            [], [{"severity": "hard_block"}], [],
        )
        assert reasons == []


# ── Full integration: process_turn() withholds, approves, delivers ─────

class TestApprovalGateIntegration:

    @staticmethod
    def _orch_with_fake_investment(session_id, shortlist):
        from orchestrator.orchestrator import Orchestrator, RoutingDecision
        from utils.llm_client import LLMClient

        orch = Orchestrator(LLMClient(force_mock=True), session_id=session_id)
        orch._session_state["user_features"] = {
            "age": 34, "income": 55000, "employment_status": "employed",
            "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
            "loss_tolerance": 4, "financial_knowledge_score": 3,
        }
        # These features run through the REAL RiskProfilingAgent (only
        # InvestmentAgent is faked below), so shortlist fixtures below use
        # "corporate_bond" specifically because it's valid under BOTH
        # "moderate" and "moderately_aggressive" (RISK_PRODUCT_ALLOW) — this
        # fixed input currently classifies as moderately_aggressive, but the
        # point of these tests is the approval gate's return-threshold
        # trigger, not the exact tier a given calibration produces. A
        # category valid under only one tier would make these tests silently
        # depend on RiskProfilingAgent's exact calibration (see
        # scripts/recalibrate_risk_model_age_neutral.py for why that number
        # moved once already) — ConflictResolver strips anything outside the
        # assigned tier's allowed set before the gate ever sees it, so an
        # incompatible category masks the gate logic entirely rather than
        # failing loudly.
        orch._classify_intent = lambda msg: (RoutingDecision.INVESTMENT, "forced", 1.0)

        real_execute = orch._execute_agent

        def fake_execute(agent_name, context):
            if agent_name == "InvestmentAgent":
                payload = {
                    "status": "complete", "risk_class": context.get("risk_class", "moderate"),
                    "shortlist": shortlist, "synthesis": "a recommendation",
                    "deliverable": True, "hallucination_flagged": False,
                }
                return AgentResult(agent_name=agent_name, success=True, payload=payload), False
            return real_execute(agent_name, context)

        orch._execute_agent = fake_execute
        return orch

    def test_gated_turn_withholds_the_real_response(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-1",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        assert result.final_response == settings.approval.withheld_message
        assert result.final_response != "a recommendation"

    def test_gated_turn_creates_a_pending_row_matching_the_turn(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-2",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        pending = approval_store.get(result.turn_id)
        assert pending is not None
        assert pending.status == "pending"
        assert pending.session_id == "int-2"
        assert any("expected_return_pct" in r for r in pending.reasons)
        # The withheld draft — not the placeholder the customer saw:
        assert pending.draft_response != settings.approval.withheld_message

    def test_clean_turn_is_never_gated(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-3",
            [{"name": "SafeBond", "category": "corporate_bond",
              "product_id": "X2", "expected_return_pct": 2.5}],
        )
        result = orch.process_turn("Should I invest?")
        assert result.final_response != settings.approval.withheld_message
        assert approval_store.list_pending(session_id="int-3") == []

    def test_approve_then_collect_returns_the_original_draft(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-4",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        pending = approval_store.get(result.turn_id)

        approval_store.approve(result.turn_id, reviewer_note="checked, fine")
        collected = orch.collect_approved_response(result.turn_id)

        assert collected == pending.draft_response
        assert approval_store.get(result.turn_id).status == "delivered"

    def test_collect_before_approval_returns_none(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-5",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        assert orch.collect_approved_response(result.turn_id) is None

    def test_collect_after_rejection_returns_none(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-6",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        approval_store.reject(result.turn_id, reviewer_note="not suitable")
        assert orch.collect_approved_response(result.turn_id) is None

    def test_collect_all_approved_responses_returns_only_this_sessions(
        self, approval_store, audit_tmp_dir,
    ):
        orch = self._orch_with_fake_investment(
            "int-7",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        # A gated turn from a DIFFERENT session must not leak into this one.
        approval_store.create(_pending(turn_id="other-session-turn", session_id="someone-else"))
        approval_store.approve("other-session-turn")

        approval_store.approve(result.turn_id)
        collected = orch.collect_all_approved_responses()
        assert len(collected) == 1
        assert approval_store.get("other-session-turn").status == "approved"  # untouched

    def test_gate_and_decision_are_both_audit_logged(self, approval_store, audit_tmp_dir):
        orch = self._orch_with_fake_investment(
            "int-8",
            [{"name": "Bond", "category": "corporate_bond",
              "product_id": "X1", "expected_return_pct": 50.0}],
        )
        result = orch.process_turn("Should I invest?")
        approval_store.approve(result.turn_id, reviewer_note="ok")

        events = [r["event_type"] for r in orch.audit_log.read_all()]
        assert "APPROVAL_GATE" in events
        assert "APPROVAL_DECISION" in events
        assert events.count("SESSION_INIT") == 1  # not duplicated by the decision


# ── Optional FastAPI router — skipped entirely if fastapi isn't installed ──

class TestApprovalsRouter:

    def test_router_full_cycle(self, approval_store):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient
        import api.approvals_router as router_mod

        approval_store.create(_pending(turn_id="api1"))
        client = TestClient(router_mod.app)

        listed = client.get("/approvals").json()
        assert [p["turn_id"] for p in listed] == ["api1"]

        approved = client.post(
            "/approvals/api1/approve", json={"reviewer_note": "fine"}
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "approved"

        again = client.post("/approvals/api1/approve", json={})
        assert again.status_code == 409

        missing = client.get("/approvals/does-not-exist")
        assert missing.status_code == 404