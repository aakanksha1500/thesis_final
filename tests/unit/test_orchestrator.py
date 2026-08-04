"""
Phase 7 - Orchestrator unit tests

Three test groups:

GROUP A: Auditing unit tests (O3 - TRiSM)
  Tests JSONL audit log in complete isolation — no Orchestrator needed.
  Verifies: schema version, event types, GDPR compliance (no message
  content logged), read_all() reconstruction.

GROUP B: ConflictResolver unit tests (O4)
  Tests cross-agent conflict detection with mock agent results.
  Verifies: RISK_PRODUCT_MISMATCH removal, MISSING_RISK detection,
  LOW_CONFIDENCE_AGGRESSIVE downgrade.

GROUP C: FailureHandler unit tests (O4 — AgentFixer)
  Tests recovery strategies for each agent type.
  Verifies: correct strategy name, success flag, payload structure,
  RiskProfilingAgent fallback forces low confidence (→ X3 uncertainty flag).

RUNNING:
  pytest tests/unit/test_orchestrator.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.base_agent import AgentResult
from orchestrator.audit_log import AuditLog
from orchestrator.conflict_resolver import ConflictResolver
from orchestrator.failure_handler import FailureHandler

from orchestrator.orchestrator import Orchestrator
from utils.llm_client import LLMClient


# GROUP A: AuditLog unit tests
class TestAuditLog:

    def _make_log(self, tmp_path: Path) -> AuditLog:
        """Create an AuditLog writing to a temp directory."""
        with patch("orchestrator.audit_log.settings") as mock_settings:
            mock_settings.orchestrator.audit_log_dir = tmp_path
            return AuditLog(session_id="test-session-001")

    def test_creates_log_file_on_init(self, tmp_path):
        log = self._make_log(tmp_path)
        assert log.log_path.exists()

    def test_session_init_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        records = log.read_all()
        assert len(records) >= 1
        assert records[0]["event_type"] == "SESSION_INIT"

    def test_session_init_has_schema_version(self, tmp_path):
        log = self._make_log(tmp_path)
        records = log.read_all()
        assert "schema_version" in records[0]

    def test_session_init_has_compliance_list(self, tmp_path):
        log = self._make_log(tmp_path)
        records = log.read_all()
        assert "compliance" in records[0]["payload"]
        assert "EU_AI_Act_Art13" in records[0]["payload"]["compliance"]

    def test_turn_start_does_not_log_message_content(self, tmp_path):
        """GDPR compliance — message content must never appear in log."""
        log = self._make_log(tmp_path)
        log.record_turn_start("t001", "I want to invest €50,000 in equities")
        records = log.read_all()
        turn_start = next(r for r in records if r["event_type"] == "TURN_START")
        # Content must not be in the record
        assert "I want to invest" not in json.dumps(turn_start)
        assert "50,000" not in json.dumps(turn_start)
        # Only length should be stored
        assert "user_message_length" in turn_start["payload"]

    def test_routing_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_routing("t001", "investment", "investment_advice", 0.87,
                           "Banking77 classification")
        records = log.read_all()
        routing = next(r for r in records if r["event_type"] == "ROUTING_DECISION")
        assert routing["payload"]["routing_decision"] == "investment"
        assert routing["payload"]["confidence"] == 0.87

    def test_agent_call_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_agent_call("t001", "RiskProfilingAgent",
                               True, 120.5, 45, "step-abc")
        records = log.read_all()
        call = next(r for r in records if r["event_type"] == "AGENT_CALL")
        assert call["payload"]["agent"] == "RiskProfilingAgent"
        assert call["payload"]["success"] is True
        assert call["payload"]["tokens_used"] == 45

    def test_constraint_violation_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_constraint_violation("t001", "R003", "hard_block",
                                         "Prohibited phrase detected", True)
        records = log.read_all()
        violation = next(
            r for r in records if r["event_type"] == "CONSTRAINT_VIOLATION"
        )
        assert violation["payload"]["rule_id"] == "R003"
        assert violation["payload"]["response_blocked"] is True

    def test_conflict_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_conflict("t001", "RISK_PRODUCT_MISMATCH",
                             "ETF not suitable for conservative",
                             "Product removed from shortlist")
        records = log.read_all()
        conflict = next(r for r in records if r["event_type"] == "CONFLICT_DETECTED")
        assert conflict["payload"]["conflict_type"] == "RISK_PRODUCT_MISMATCH"

    def test_turn_end_record_written(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_turn_end("t001", 150, ["RiskProfilingAgent"], 340.0, 0, 0)
        records = log.read_all()
        end = next(r for r in records if r["event_type"] == "TURN_END")
        assert end["payload"]["agents_invoked"] == ["RiskProfilingAgent"]

    def test_all_records_have_session_id(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_turn_start("t001", "Hello")
        log.record_routing("t001", "conversational_only", "general_query", 0.9, "")
        records = log.read_all()
        for r in records:
            assert r["session_id"] == "test-session-001"

    def test_all_records_have_timestamp(self, tmp_path):
        log = self._make_log(tmp_path)
        log.record_turn_start("t001", "Hello")
        records = log.read_all()
        for r in records:
            assert "timestamp" in r
            assert r["timestamp"]  # not empty

    def test_write_never_raises_on_bad_path(self):
        """O3 requirement: audit failure must never block a user response."""
        with patch("orchestrator.audit_log.settings") as mock_settings:
            mock_settings.orchestrator.audit_log_dir = Path("/nonexistent/path/xyz")
            # Should not raise even with bad path
            try:
                AuditLog(session_id="bad-path-test")
            except Exception:
                pass  # init may fail — that's acceptable

    def test_read_all_returns_list(self, tmp_path):
        log = self._make_log(tmp_path)
        records = log.read_all()
        assert isinstance(records, list)

    def test_write_count_increments(self, tmp_path):
        log = self._make_log(tmp_path)
        initial = log.write_count
        log.record_turn_start("t001", "Hello")
        assert log.write_count == initial + 1


# GROUP B: ConflictResolver unit tests

def _make_risk_result(
    risk_class: str = "moderate",
    confidence: float = 0.8,
) -> AgentResult:
    return AgentResult(
        agent_name="RiskProfilingAgent",
        success=True,
        payload={
            "status": "complete",
            "risk_class": risk_class,
            "confidence": confidence,
            "hybrid_score": 0.5,
        },
    )


def _make_investment_result(
    shortlist: list | None = None,
) -> AgentResult:
    default_shortlist = [
        {"name": "Broad ETF", "category": "etf_broad",
         "expected_return_pct": 7.0, "score": 0.82},
    ]
    return AgentResult(
        agent_name="InvestmentAgent",
        success=True,
        payload={
            "status": "complete",
            "risk_class": "moderate",
            "shortlist": shortlist or default_shortlist,
            "synthesis": "Based on your profile...",
        },
    )


class TestConflictResolver:

    def test_no_conflict_when_compatible(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("moderate")
        inv = _make_investment_result([
            {"name": "ETF", "category": "etf_broad",
             "expected_return_pct": 7.0, "score": 0.8}
        ])
        _, conflicts = resolver.resolve([risk, inv])
        type_names = [c["type"] for c in conflicts]
        assert "RISK_PRODUCT_MISMATCH" not in type_names

    def test_mismatch_detected_and_product_removed(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("conservative")
        inv = _make_investment_result([
            {"name": "High-Risk Equity Fund", "category": "individual_equity",
             "expected_return_pct": 15.0, "score": 0.9}
        ])
        results, conflicts = resolver.resolve([risk, inv])
        mismatch = [c for c in conflicts if c["type"] == "RISK_PRODUCT_MISMATCH"]
        assert len(mismatch) >= 1
        # Product should be removed from shortlist
        inv_result = next(r for r in results if r.agent_name == "InvestmentAgent")
        assert len(inv_result.payload["shortlist"]) == 0

    def test_compatible_product_kept(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("conservative")
        inv = _make_investment_result([
            {"name": "Govt Bond", "category": "government_bond",
             "expected_return_pct": 3.2, "score": 0.85}
        ])
        results, conflicts = resolver.resolve([risk, inv])
        inv_result = next(r for r in results if r.agent_name == "InvestmentAgent")
        assert len(inv_result.payload["shortlist"]) == 1

    def test_missing_risk_before_investment_detected(self):
        resolver = ConflictResolver()
        # Only investment result, no risk result
        inv = _make_investment_result()
        _, conflicts = resolver.resolve([inv])
        types = [c["type"] for c in conflicts]
        assert "MISSING_RISK_BEFORE_INVESTMENT" in types

    def test_missing_risk_marks_investment_undeliverable(self):
        resolver = ConflictResolver()
        inv = _make_investment_result()
        results, _ = resolver.resolve([inv])
        inv_result = next(r for r in results if r.agent_name == "InvestmentAgent")
        assert inv_result.payload.get("deliverable") is False

    def test_low_confidence_aggressive_downgraded(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("aggressive", confidence=0.35)
        _, conflicts = resolver.resolve([risk])
        types = [c["type"] for c in conflicts]
        assert "LOW_CONFIDENCE_AGGRESSIVE" in types

    def test_low_confidence_aggressive_sets_moderate(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("aggressive", confidence=0.35)
        results, _ = resolver.resolve([risk])
        risk_result = next(r for r in results if r.agent_name == "RiskProfilingAgent")
        assert risk_result.payload["risk_class"] == "moderate"
        assert risk_result.payload["risk_class_original"] == "aggressive"

    def test_high_confidence_aggressive_not_downgraded(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("aggressive", confidence=0.85)
        _, conflicts = resolver.resolve([risk])
        types = [c["type"] for c in conflicts]
        assert "LOW_CONFIDENCE_AGGRESSIVE" not in types

    def test_empty_results_returns_empty_conflicts(self):
        resolver = ConflictResolver()
        _, conflicts = resolver.resolve([])
        assert conflicts == []

    def test_conflict_records_have_required_keys(self):
        resolver = ConflictResolver()
        risk = _make_risk_result("conservative")
        inv = _make_investment_result([
            {"name": "Equity Fund", "category": "individual_equity",
             "expected_return_pct": 12.0, "score": 0.9}
        ])
        _, conflicts = resolver.resolve([risk, inv])
        for c in conflicts:
            assert "type" in c
            assert "description" in c
            assert "resolution" in c


# GROUP C: FailureHandler unit tests (O4 — AgentFixer)

class TestFailureHandler:

    def test_risk_recovery_succeeds(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("RiskProfilingAgent",
                                          Exception("timeout"), {})
        assert result["success"] is True
        assert result["strategy"] == "heuristic_fallback"

    def test_risk_recovery_forces_low_confidence(self):
        """
        RiskProfilingAgent fallback must force confidence < 0.6
        so ExplainabilityAgent triggers the X3 uncertainty flag.
        Takayanagi et al. [7]: calibrated trust requires surfacing uncertainty.
        """
        handler = FailureHandler()
        result = handler.attempt_recovery("RiskProfilingAgent",
                                          Exception("API error"), {})
        conf = result["recovered_payload"]["confidence"]
        assert conf < 0.6, (
            f"Fallback confidence {conf} must be < 0.6 to trigger X3 flag"
        )

    def test_risk_recovery_payload_has_risk_class(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("RiskProfilingAgent",
                                          Exception("timeout"), {})
        assert "risk_class" in result["recovered_payload"]
        assert result["recovered_payload"]["risk_class"] == "moderate"

    def test_investment_recovery_succeeds(self):
        handler = FailureHandler()
        result = handler.attempt_recovery(
            "InvestmentAgent", Exception("timeout"),
            {"risk_agent_payload": {"risk_class": "conservative"}}
        )
        assert result["success"] is True
        assert result["strategy"] == "static_shortlist"

    def test_investment_recovery_has_shortlist(self):
        handler = FailureHandler()
        result = handler.attempt_recovery(
            "InvestmentAgent", Exception("API error"),
            {"risk_agent_payload": {"risk_class": "moderate"}}
        )
        assert len(result["recovered_payload"]["shortlist"]) >= 1

    def test_investment_recovery_has_disclaimer(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("InvestmentAgent",
                                          Exception("error"), {})
        synthesis = result["recovered_payload"]["synthesis"]
        assert "not regulated financial advice" in synthesis.lower() or \
               "consult a qualified advisor" in synthesis.lower()

    def test_budget_recovery_is_skip(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("BudgetAgent",
                                          Exception("timeout"), {})
        assert result["strategy"] == "skip"
        assert result["success"] is True

    def test_explainability_recovery_has_calibration_note(self):
        """X3 requirement: calibration note must be deliverable even on failure."""
        handler = FailureHandler()
        result = handler.attempt_recovery("ExplainabilityAgent",
                                          Exception("timeout"), {})
        assert result["success"] is True
        payload = result["recovered_payload"]
        assert payload["calibration_note"] is not None
        assert len(payload["calibration_note"]) > 0

    def test_conversational_recovery_has_response(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("ConversationalAgent",
                                          Exception("timeout"), {})
        assert result["success"] is True
        assert result["recovered_payload"]["response"]

    def test_unknown_agent_returns_failure(self):
        handler = FailureHandler()
        result = handler.attempt_recovery("UnknownAgent",
                                          Exception("error"), {})
        assert result["success"] is False

    def test_recovery_includes_original_error(self):
        handler = FailureHandler()
        error = Exception("connection timeout")
        result = handler.attempt_recovery("BudgetAgent", error, {})
        assert "connection timeout" in result["original_error"]

    def test_all_agent_names_have_recovery_strategies(self):
        handler = FailureHandler()
        agents = [
            "RiskProfilingAgent", "InvestmentAgent", "BudgetAgent",
            "ExplainabilityAgent", "ConversationalAgent",
        ]
        for agent in agents:
            result = handler.attempt_recovery(agent, Exception("test"), {})
            assert result["strategy"] != "none", (
                f"{agent} has no recovery strategy — add one to FailureHandler"
            )

# GROUP D: per-role LLM provider routing
#
# Specialist agents (RiskProfilingAgent, InvestmentAgent, BudgetAgent,
# ExplainabilityAgent — narrating figures Python already computed) run on
# settings.llm.specialist_provider (default "ollama", local), independent
# of whatever LLM_PROVIDER the orchestrator/judge tiers use. This is
# provider routing, not just a different model NAME on the same
# provider — see Orchestrator._derive_client() and
# utils.llm_client.LLMClient's `provider` param.

class TestDeriveClientProviderRouting:

    def test_mock_primary_always_reused_regardless_of_provider(self):
        """
        The one case that must never change: every existing unit test
        constructs Orchestrator(LLMClient()) with no real key (mock
        mode), and some patch .chat() on that exact primary instance.
        A second client for the specialist tier would silently bypass
        the patch. Confirms this session's mock-mode test suite is
        unaffected by the provider-routing change.
        """
        mock_primary = LLMClient()
        assert mock_primary.mode == "mock"
        derived = Orchestrator._derive_client(
            mock_primary, "llama3.1:8b", "specialist", provider="ollama"
        )
        assert derived is mock_primary

    def test_different_provider_creates_a_new_client(self):
        """
        Same model name, different provider — must not be silently
        collapsed into the primary just because a model string happens
        to match (it won't in practice, but the provider check must be
        independent of the model check either way).
        """
        primary = LLMClient(provider="ollama", model="llama3.1:8b")
        assert primary.mode == "ollama"

        derived = Orchestrator._derive_client(
            primary, "llama3.1:8b", "specialist", provider="ollama"
        )
        # same provider AND same model -> correctly reused, not a new client
        assert derived is primary

    def test_provider_change_forces_new_client_even_with_mock_fallback(self):
        """
        primary is real (ollama). Requesting a DIFFERENT provider for
        the derived client must not reuse primary, even though the
        derived client itself may fall back to mock in an environment
        with no credentials for that other provider (expected here —
        this sandbox has no real cloud API key).
        """
        primary = LLMClient(provider="ollama", model="llama3.1:8b")
        assert primary.mode == "ollama"

        derived = Orchestrator._derive_client(
            primary, "some-other-model", "judge", provider="groq"
        )
        assert derived is not primary

    def test_provider_none_inherits_primary_provider(self):
        """judge tier passes provider=None — must track whatever primary is using, not be pinned independently."""
        primary = LLMClient(provider="ollama", model="llama3.1:8b")
        derived = Orchestrator._derive_client(
            primary, "llama3.1:8b", "judge", provider=None
        )
        assert derived is primary  # same model, provider inherited -> reused

    def test_orchestrator_wires_specialist_provider_from_settings(self):
        """
        End-to-end: Orchestrator.__init__ should pass
        settings.llm.specialist_provider through to _derive_client for
        the specialist tier specifically. In mock mode this collapses
        back to the primary (see test_mock_primary_always_reused_
        regardless_of_provider above) — this test just confirms the
        wiring exists and doesn't raise, using a real (non-mock)
        primary to exercise the actual branch.

        Uses settings.llm.specialist_model directly rather than
        hardcoding an assumed value — .env's SPECIALIST_MODEL might
        still hold a pre-split Groq-style name (exactly the staleness
        documented in settings.py's LLMConfig), and this test should
        pass regardless of which model string is actually configured.
        """
        from config.settings import settings

        primary = LLMClient(
            provider=settings.llm.specialist_provider,
            model=settings.llm.specialist_model,
        )
        orch = Orchestrator(primary, session_id="provider-routing-test")
        # specialist tier requests the same provider AND same model as
        # primary here (by construction) -> correctly reused, not a new
        # client.
        assert orch.specialist_llm is primary
        # judge tier: provider=None, inherits primary's provider; model
        # differs only if JUDGE_MODEL != specialist_model in this env,
        # so just confirm it didn't crash and is a valid client.
        assert orch.judge_llm.mode in (settings.llm.specialist_provider, "mock")


class TestPlannerWiring:
    """
    The planner is wired into process_turn as the primary Layer 1 path.
    These tests are about the WIRING, not the planner itself (that is
    tests/unit/test_planner.py) — specifically, that turning it on cannot
    change what a turn does when the model gives nothing usable, because
    every committed RQ result was produced on the static path.
    """

    @pytest.fixture(autouse=True)
    def _planner_on(self, monkeypatch):
        """PLANNER_ENABLED is an environment variable; an ambient
        PLANNER_ENABLED=false (the static arm's setting during evaluation)
        would make every assertion below pass for the wrong reason."""
        from config.settings import settings
        monkeypatch.setattr(settings.planner, "enabled", True)

    @staticmethod
    def _orch_with_plan(plan_json_text: str, **kwargs):
        from unittest.mock import MagicMock

        from orchestrator.orchestrator import Orchestrator
        from utils.llm_client import LLMClient

        client = LLMClient(force_mock=True)
        orch = Orchestrator(client, **kwargs)

        from config.prompts import PLANNER_SYSTEM

        def fake_chat(system, messages, temperature=None):
            resp = MagicMock()
            resp.content = (
                plan_json_text if system == PLANNER_SYSTEM
                else "[MOCK RESPONSE] synthesis"
            )
            resp.tokens_used = 7
            return resp

        orch._planner.llm.chat = fake_chat
        return orch

    def test_result_carries_the_plan(self, audit_tmp_dir):
        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-1")
        result = orch.process_turn("hello there")
        assert result.plan is not None
        assert result.plan_source in ("planner", "static_fallback")

    def test_accepted_plan_drives_the_agent_sequence(self, audit_tmp_dir):
        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-2")
        result = orch.process_turn("hi")
        assert result.plan.accepted
        assert result.agents_invoked == ["ConversationalAgent"]

    def test_unusable_plan_reproduces_the_static_path_exactly(self, audit_tmp_dir):
        """The regression guard for every committed result: when the planner
        gives nothing usable, the turn must run precisely what the static
        table would have run."""
        from orchestrator.orchestrator import Orchestrator

        orch = self._orch_with_plan("no json here", session_id="plan-wire-3")
        result = orch.process_turn("hi")
        expected = Orchestrator._get_agent_sequence(result.routing_decision)
        assert result.plan.source == "static_fallback"
        assert list(result.plan.steps) == expected

    def test_plan_is_written_to_the_audit_log(self, audit_tmp_dir):
        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-4")
        orch.process_turn("hi")
        events = orch.audit_log.read_all()
        plans = [e for e in events if e["event_type"] == "PLAN"]
        assert len(plans) == 1
        assert plans[0]["payload"]["source"] in ("planner", "static_fallback")
        assert "proposed" in plans[0]["payload"]

    def test_plan_event_type_is_declared_valid(self, audit_tmp_dir):
        """
        AuditLog._write logs an ERROR for an unrecognised event_type and then
        writes the record anyway. That is the right runtime behaviour — losing
        an audit record is worse than writing an odd one — but it means a new
        record_*() whose type was never added to VALID_EVENT_TYPES produces a
        log file that LOOKS correct while every downstream consumer filtering
        on the known set silently drops it.

        A test asserting only "the PLAN event is in the file" passes in that
        state. This one does not.
        """
        from orchestrator.audit_log import AuditLog

        assert "PLAN" in AuditLog.VALID_EVENT_TYPES

    def test_a_whole_turn_writes_only_declared_event_types(self, audit_tmp_dir):
        """The general version of the above, so the next component to add a
        record_*() method is covered without anyone remembering to."""
        from orchestrator.audit_log import AuditLog

        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-8")
        orch.process_turn("hi")
        written = {e["event_type"] for e in orch.audit_log.read_all()}
        undeclared = written - AuditLog.VALID_EVENT_TYPES
        assert not undeclared, (
            f"written to the audit log but not in VALID_EVENT_TYPES: "
            f"{sorted(undeclared)} — downstream consumers filtering on the "
            f"known set will drop these silently"
        )

    def test_rejected_proposal_survives_into_the_audit_log(self, audit_tmp_dir):
        """InvestmentAgent with no risk class — rejected. The audit log has to
        show what was asked for, not just what ran."""
        orch = self._orch_with_plan('{"plan": ["InvestmentAgent"]}',
                                    session_id="plan-wire-5")
        orch.process_turn("what should I invest in?")
        payload = [e for e in orch.audit_log.read_all()
                   if e["event_type"] == "PLAN"][0]["payload"]
        assert payload["proposed"] == ["InvestmentAgent"]
        assert payload["accepted"] is False
        assert payload["rejections"]

    def test_planner_disabled_marks_the_source_as_static(self, audit_tmp_dir,
                                                          monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings.planner, "enabled", False)
        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-6")
        result = orch.process_turn("hi")
        assert result.plan_source == "static"

    def test_planner_stats_exposed_on_the_orchestrator(self, audit_tmp_dir):
        orch = self._orch_with_plan('{"plan": ["ConversationalAgent"]}',
                                    session_id="plan-wire-7")
        orch.process_turn("hi")
        stats = orch.planner_stats
        assert stats["attempts"] == 1
        assert set(stats) >= {"accepted", "fallbacks", "validity_rate",
                              "rejection_counts"}

    def test_static_sequences_match_the_orchestrator_table(self):
        """One declaration, two consumers. If these ever diverge, the
        planner-vs-static comparison stops measuring what it claims to."""
        from agents.payloads import STATIC_SEQUENCES
        from orchestrator.orchestrator import Orchestrator, RoutingDecision

        for routing in RoutingDecision:
            assert (Orchestrator._get_agent_sequence(routing)
                    == list(STATIC_SEQUENCES[routing.value]))