"""
Audit-trail regressions — narrow, targeted tests pinning specific bugs
that were found and fixed, so they don't reappear silently.

WHY THIS FILE EXISTS, AND WHY SEPARATE FROM test_explainability_agent.py /
test_session_evaluation.py
    test_explainability_agent.py exercises the three ablation conditions
    (RQ3) against a hand-crafted, COMPLETE risk payload — it was never
    testing what happens when an upstream agent's payload documents its
    own failure, because that wasn't the question that file exists to
    answer. test_session_evaluation.py drives full multi-turn sessions
    through process_turn() but never specifically checked what
    ExplainabilityAgent does with an incomplete RiskProfilingAgent
    result, or what the audit log actually contains at the byte level.
    Both gaps are real: the second one is how a hallucinating turn
    (RQ4 scenario S02 — RiskProfilingAgent reported status="incomplete",
    ExplainabilityAgent still asserted risk_class="moderate",
    confidence=0.7) passed 592 other tests and was only caught by an
    LLM-as-judge reading the transcript after the fact. This file exists
    so the specific bug, and the audit-log guarantees the fix touches,
    have their own direct, fast, judge-free coverage.

THE FIX BEING PINNED (explainability/explainability_agent.py)
    risk_payload.get("risk_class", "moderate") used to default to a
    specific, fabricated classification whenever risk_class was absent —
    true BOTH when no risk profiling was ever attempted this turn AND
    when RiskProfilingAgent ran and explicitly reported it could not
    classify. INCOMPLETE_STATUSES (re-exported from
    agents.payloads.PARTIAL_STATUSES — see that constant's own docstring
    for why it's imported rather than duplicated) now gates this: a
    status in that set means risk_class/confidence are left honestly
    empty rather than defaulted.

RUNNING
    pytest tests/unit/test_audit_regressions.py -v
    No live API key needed — mock mode throughout.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.payloads import PARTIAL_STATUSES
from explainability.explainability_agent import INCOMPLETE_STATUSES, ExplainabilityAgent
from orchestrator.audit_log import AuditLog
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient


def make_agent() -> ExplainabilityAgent:
    """ExplainabilityAgent in mock mode. Duplicated from
    test_explainability_agent.py's own make_agent() rather than imported
    from it — this file should not depend on that one's internals."""
    return ExplainabilityAgent(LLMClient())


# The exact shape RiskProfilingAgent.run() returns when
# _check_missing_features() finds gaps — agents/risk_profiling_agent.py,
# the payload branch that sets status="incomplete". No risk_class key at
# all, which is exactly what made the old default fire.
INCOMPLETE_RISK_PAYLOAD: dict = {
    "status": "incomplete",
    "missing_features": [
        "age", "income", "employment_status", "dependents",
        "existing_debt", "investment_horizon", "loss_tolerance",
        "financial_knowledge_score",
    ],
    "message": (
        "I don't have enough information yet to assess your risk "
        "profile — I still need a few details about your situation "
        "before I can give you an accurate answer."
    ),
}

# A genuine, successful RiskProfilingAgent payload — same shape as
# test_explainability_agent.py's SAMPLE_CONTEXT, trimmed to what this
# file's tests actually read. No "status" key at all on a real success
# payload (see risk_profiling_agent.py) — status="complete" also appears
# in practice; both must be treated as "trust this data".
COMPLETE_RISK_PAYLOAD: dict = {
    "risk_class": "moderately_aggressive",
    "confidence": 0.81,
    "hybrid_score": 0.63,
    "feature_importance": {
        "loss_tolerance": {"value": 4, "shap_impact": 0.09},
    },
    "rationale": "Classified moderately_aggressive based on stated loss tolerance.",
}


class TestIncompleteStatusesIsSharedNotDuplicated:
    """
    Pins the relationship itself, not just its effect. If a future edit
    to agents/payloads.py adds or removes a status from PARTIAL_STATUSES
    without anyone touching explainability_agent.py, this is the test
    that notices — the whole point of importing rather than duplicating
    the list was so there's exactly one thing to keep in sync, but
    "exactly one thing" still needs a test asserting it stayed one thing.
    """

    def test_incomplete_statuses_is_the_same_set_as_partial_statuses(self):
        assert INCOMPLETE_STATUSES == frozenset(PARTIAL_STATUSES)

    def test_incomplete_status_is_a_member(self):
        # The one value RiskProfilingAgent actually produces. If this
        # ever stops being true, the fix below stops firing for the
        # exact case it was written for.
        assert "incomplete" in INCOMPLETE_STATUSES


class TestExplainabilityAgentDoesNotHallucinateOnIncompleteRisk:
    """
    The S02 regression, isolated: ExplainabilityAgent.run() called
    directly with a context whose risk_agent_payload documents a
    RiskProfilingAgent failure. No Orchestrator involved — if this fails,
    the bug is back in ExplainabilityAgent itself, not in how something
    upstream wires it in.
    """

    def test_risk_class_is_empty_not_moderate(self):
        agent = make_agent()
        result = agent.run({"risk_agent_payload": INCOMPLETE_RISK_PAYLOAD})
        assert result.payload["risk_class"] == ""
        assert result.payload["risk_class"] != "moderate"

    def test_confidence_is_zero_not_the_old_default(self):
        agent = make_agent()
        result = agent.run({"risk_agent_payload": INCOMPLETE_RISK_PAYLOAD})
        assert result.payload["confidence"] == 0.0
        assert result.payload["confidence"] != 0.7

    def test_unavailable_reason_names_the_actual_status(self):
        agent = make_agent()
        result = agent.run({"risk_agent_payload": INCOMPLETE_RISK_PAYLOAD})
        assert result.payload["risk_class_unavailable_reason"] == "incomplete"

    def test_shap_layer_is_skipped_not_narrated_on_nothing(self):
        # feature_importance doesn't exist on an incomplete payload
        # either — shap_summary is {} and Layer A's own
        # "if cfg.use_shap and shap_summary" guard should skip it
        # regardless of this fix, but worth pinning alongside risk_class
        # since both come from the same missing data.
        agent = make_agent()
        result = agent.run({"risk_agent_payload": INCOMPLETE_RISK_PAYLOAD})
        assert result.payload["shap_narrative"] is None
        assert "shap" not in result.payload["layers_applied"]

    def test_does_not_raise_and_still_produces_a_response(self):
        # The honest-absence path still has to produce SOMETHING —
        # calibration_note/full_explanation should degrade gracefully,
        # not crash, given risk_class="".
        agent = make_agent()
        result = agent.run({"risk_agent_payload": INCOMPLETE_RISK_PAYLOAD})
        assert result.success
        assert result.payload["full_explanation"]
        assert result.payload["calibration_note"]


class TestExplainabilityAgentStillTrustsCompleteRiskData:
    """
    The other side of the same fix: a real, successful risk_class must
    NOT get swept into the same "honestly absent" treatment. This is the
    test that would catch an overcorrection — e.g. a future edit that
    widens the INCOMPLETE_STATUSES check to something that also matches
    a legitimate status value.
    """

    def test_real_risk_class_passes_through_unchanged(self):
        agent = make_agent()
        result = agent.run({"risk_agent_payload": COMPLETE_RISK_PAYLOAD})
        assert result.payload["risk_class"] == "moderately_aggressive"
        assert result.payload["confidence"] == 0.81
        assert result.payload["risk_class_unavailable_reason"] is None

    def test_status_complete_is_also_trusted(self):
        # RiskProfilingAgent's real success payload sets status="complete"
        # explicitly (not just an absent status key) — confirm that
        # value specifically isn't caught by INCOMPLETE_STATUSES either.
        payload = dict(COMPLETE_RISK_PAYLOAD, status="complete")
        agent = make_agent()
        result = agent.run({"risk_agent_payload": payload})
        assert result.payload["risk_class"] == "moderately_aggressive"
        assert result.payload["risk_class_unavailable_reason"] is None


class TestExplainabilityAgentBudgetOnlyPathUnaffected:
    """
    The pre-existing budget_only branch (risk_class = "" for a different
    reason — no risk data was ever in scope, not that it failed) sits
    right next to the new incomplete-status branch. Confirms the two
    don't interfere: budget_only's own empty-risk_class case still works,
    and doesn't get an unavailable_reason meant for the OTHER case.
    """

    def test_budget_only_still_produces_empty_risk_class(self):
        agent = make_agent()
        result = agent.run({
            "budget_agent_payload": {
                "benchmark_comparison": {"housing": "above"},
                "monthly_expenses": {"housing": 1200.0},
            },
        })
        assert result.payload["risk_class"] == ""

    def test_budget_only_does_not_claim_a_risk_unavailable_reason(self):
        # risk_payload was never present at all here — this isn't the
        # incomplete-status case, so the reason field should stay None,
        # not be mistaken for "risk profiling failed".
        agent = make_agent()
        result = agent.run({
            "budget_agent_payload": {
                "benchmark_comparison": {"housing": "above"},
                "monthly_expenses": {"housing": 1200.0},
            },
        })
        assert result.payload["risk_class_unavailable_reason"] is None


class TestAuditLogEventSchemas:
    """
    Direct tests against AuditLog's own methods, no Orchestrator involved.
    Pins the exact payload shape for the two events that are conditional
    on settings flags (HALLUCINATION_FLAG, RAG_CITATIONS — see
    Orchestrator._cache_agent_output's settings.hallucination.
    log_flagged_to_audit / settings.rag.log_citations_to_audit gates) and
    are therefore easy to stop firing without any single end-to-end test
    noticing, since neither is on the "happy path" most tests exercise.
    """

    @pytest.fixture
    def audit_log(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_settings:
            mock_settings.orchestrator.audit_log_dir = tmp_path
            return AuditLog(session_id="audit-schema-test")

    def test_hallucination_flag_records_score_and_threshold(self, audit_log):
        audit_log.record_hallucination_flag(
            turn_id="t1",
            agent_name="InvestmentAgent",
            claim="This ETF has never posted a loss.",
            score=0.12,
            threshold=0.5,
            mode="hhem",
        )
        flags = [r for r in audit_log.read_all() if r["event_type"] == "HALLUCINATION_FLAG"]
        assert len(flags) == 1
        payload = flags[0]["payload"]
        assert payload["agent"] == "InvestmentAgent"
        assert payload["claim"] == "This ETF has never posted a loss."
        assert payload["score"] == 0.12
        assert payload["threshold"] == 0.5
        assert payload["detector_mode"] == "hhem"

    def test_citations_record_query_alongside_each_source(self, audit_log):
        # query is logged specifically so a citation can be re-verified
        # after the fact (see record_citations' own docstring) — a
        # regression that dropped the query but kept the citations would
        # silently break that, worth asserting on its own.
        audit_log.record_citations(
            turn_id="t1",
            agent_name="ExplainabilityAgent",
            query="early repayment charge rules",
            citations=[{
                "source": "CBI Consumer Protection Code",
                "text": "...",
                "relevance": 0.83,
                "document_set": "regulatory",
                "doc_id": "D12",
            }],
        )
        cites = [r for r in audit_log.read_all() if r["event_type"] == "RAG_CITATIONS"]
        assert len(cites) == 1
        assert cites[0]["payload"]["query"] == "early repayment charge rules"
        assert cites[0]["payload"]["citations"][0]["source"] == "CBI Consumer Protection Code"
        assert cites[0]["payload"]["citations"][0]["doc_id"] == "D12"

    def test_turn_start_logs_length_only_not_content(self, audit_log):
        message = "a private, specific customer message"
        audit_log.record_turn_start("t1", message)
        starts = [r for r in audit_log.read_all() if r["event_type"] == "TURN_START"]
        assert starts[0]["payload"]["user_message_length"] == len(message)
        assert "user_message" not in starts[0]["payload"]
        assert message not in str(starts[0]["payload"])

    def test_every_record_carries_session_and_turn_id(self, audit_log):
        # O3 / TRiSM's core requirement (see this module's own docstring):
        # every record traceable back to a specific session and turn.
        audit_log.record_turn_start("t1", "x")
        records = audit_log.read_all()
        non_init = [r for r in records if r["event_type"] != "SESSION_INIT"]
        assert non_init  # something was actually recorded
        for r in non_init:
            assert r["session_id"] == "audit-schema-test"
            assert r["turn_id"] == "t1"


class TestOrchestratorLevelAuditRegressions:
    """
    The full-pipeline versions of the two regressions above: reproduces
    the actual S02 scenario end-to-end through process_turn() (not just
    ExplainabilityAgent in isolation), and confirms the GDPR content
    guarantee at the raw log-file level rather than against one method's
    return value.

    WHY THIS CLASS EXISTS SEPARATELY FROM THE ISOLATED ExplainabilityAgent
    TESTS ABOVE, AND ISN'T REDUNDANT WITH THEM
        It caught a second, deeper bug the isolated tests physically
        could not: those construct context={"risk_agent_payload":
        INCOMPLETE_RISK_PAYLOAD} by hand, which assumes the orchestrator
        actually hands ExplainabilityAgent the failed payload. It didn't
        — Orchestrator._publish() returned {} for any failed agent call
        (a blanket success gate), so a failed RiskProfilingAgent's
        "status": "incomplete" payload never reached context at all; what
        ExplainabilityAgent actually saw was whatever
        _execute_plan()'s cross-turn setdefault had already put there
        (last turn's cache, or {} on a first turn) — indistinguishable
        from risk profiling never having been attempted, which is exactly
        the ambiguity the ExplainabilityAgent-level fix was meant to
        resolve and couldn't, because the information never arrived.
        Fixed in Orchestrator._publish(): the full payload under
        _AGENT_PAYLOAD_CONTEXT_KEY now publishes regardless of success
        (flattened capability keys like "risk_class" stay success-only,
        so a failure still can't make a downstream agent think a real
        classification exists — see _publish()'s own docstring for the
        full reasoning). This class's first test is the one that caught
        the gap; the isolated ExplainabilityAgent tests above remain
        useful for fast, judge-free coverage of ExplainabilityAgent's own
        logic, they just aren't sufficient on their own to catch a bug in
        what hands data to it.
    """

    @pytest.fixture
    def orch(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_settings:
            mock_settings.orchestrator.audit_log_dir = tmp_path
            return Orchestrator(
                LLMClient(force_mock=True), session_id="audit-regressions-eval",
            )

    def test_publish_hands_failed_risk_payload_to_context_not_nothing(self, orch):
        # Direct test of Orchestrator._publish() itself, the actual
        # mechanism this regression lived in — isolated from
        # ExplainabilityAgent, from process_turn(), from routing. A
        # failed RiskProfilingAgent call must still publish its full
        # payload under risk_agent_payload; only the flattened
        # capability keys (risk_class, confidence) should stay
        # success-gated.
        class _FakeFailedResult:
            success = False
            payload = {"status": "incomplete", "missing_features": ["age"]}

        published = orch._publish("RiskProfilingAgent", _FakeFailedResult())
        assert published.get("risk_agent_payload") == _FakeFailedResult.payload
        # The flattened key must NOT appear — a failure publishing
        # "risk_class" would make available_context_keys() think a real
        # classification exists when it doesn't.
        assert "risk_class" not in published

    def test_s02_scenario_no_seeded_customer_does_not_hallucinate_risk_class(self, orch):
        # No customer_id, no seeded user_features — RiskProfilingAgent
        # has nothing to work with and reports status="incomplete", the
        # same starting condition as RQ4 scenario S02.
        orch._classify_intent = lambda msg: (
            RoutingDecision.RISK_PROFILING, "risk_profiling", 0.99,
        )
        result = orch.process_turn("Can you assess my financial risk profile?")

        risk_result = next(
            r for r in result.agent_results if r.agent_name == "RiskProfilingAgent"
        )
        assert risk_result.payload.get("status") == "incomplete"

        explain_result = next(
            (r for r in result.agent_results if r.agent_name == "ExplainabilityAgent"),
            None,
        )
        assert explain_result is not None, (
            "ExplainabilityAgent should still run in the risk_profiling "
            "sequence even when RiskProfilingAgent came back incomplete "
            "— it's the honesty of what it says that this test checks, "
            "not whether it runs at all."
        )
        assert explain_result.payload["risk_class"] == ""
        assert "moderate" not in explain_result.payload["risk_class"]

    def test_user_message_content_never_appears_in_raw_audit_log(self, orch):
        # The full-pipeline version of TestAuditLogEventSchemas'
        # test_turn_start_logs_length_only_not_content — checks the
        # guarantee holds across an ENTIRE turn (routing, agent calls,
        # synthesis), not just the one TURN_START record, so a stray
        # debug payload logged anywhere else in the turn would also be
        # caught.
        canary = "XYZZY_CANARY_ACCOUNT_BALANCE_IS_EXACTLY_87342_EUR"
        orch.process_turn(canary)
        raw_content = orch.audit_log.log_path.read_text(encoding="utf-8")
        assert canary not in raw_content