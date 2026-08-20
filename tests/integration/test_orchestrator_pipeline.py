"""
Phase 7 — Orchestrator integration tests (RQ4).

This file runs the full HALO pipeline end-to-end across 10 scripted
scenarios and measures four RQ4 metrics:
  - routing_accuracy        (HALO Layer 1 correctness)
  - component_synergy_score (CSS — Raza et al. [1])
  - tool_utilisation_efficacy (TUE — Raza et al. [1])
  - step_progress_rate      (AgentBoard E1 — Ma et al.)

Plus Agent-as-Judge scores (E2 — Zhuge et al.) for qualitative evaluation.

Writes: results/rq4_mas_coherence.json

MOCK MODE vs REAL MODE:
  Mock mode: LLM returns [MOCK RESPONSE] for all calls. Routing will
  default to CONVERSATIONAL_ONLY for most queries (intent classifier
  cannot classify without a real LLM). CSS and TUE are still meaningful
  because they measure pipeline structure, not LLM quality.
  routing_accuracy will be 0.0 in mock mode — this is expected and
  documented in the results JSON.

  Real mode: Full routing + agent execution + synthesis. All metrics
  are meaningful. Expected routing_accuracy ≥ 0.70 with Gemini-2.0.

RUNNING:
  pytest tests/integration/test_orchestrator_pipeline.py -v -s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import settings
from evaluation.agent_judge import AgentJudge
from evaluation.metrics import (
    component_synergy_score,
    routing_accuracy,
    step_progress_rate,
    tool_utilisation_efficacy,
)
from evaluation.results_io import write_results
from orchestrator.orchestrator import Orchestrator, RoutingDecision
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


# 10 scripted scenarios with expected routing decisions

SCENARIOS = [
    {
        "id": "S01",
        "message": "What is my account balance?",
        "expected_routing": "conversational_only",
        "description": "Simple general query — no specialist needed",
    },
    {
        "id": "S02",
        "message": "Can you assess my financial risk profile?",
        "expected_routing": "risk_profiling",
        "description": "Direct risk profiling request",
    },
    {
        "id": "S03",
        "message": "I want to invest my savings for retirement.",
        "expected_routing": "investment",
        "description": "Investment query — requires risk first",
    },
    {
        "id": "S04",
        "message": "Help me understand my monthly spending.",
        "expected_routing": "budget",
        "description": "Budget analysis request",
    },
    {
        "id": "S05",
        "message": "Why did you recommend that ETF?",
        "expected_routing": "explanation_request",
        "description": "Explanation request for prior recommendation",
    },
    {
        "id": "S06",
        "message": "What is the current exchange rate?",
        "expected_routing": "conversational_only",
        "description": "Out-of-scope query (FX rate not in advisory scope)",
    },
    {
        "id": "S07",
        "message": "Can you recommend a low-risk investment product?",
        "expected_routing": "investment",
        "description": "Product suggestion → investment routing",
    },
    {
        "id": "S08",
        "message": "How much risk can I afford to take?",
        "expected_routing": "risk_profiling",
        "description": "Risk capacity question → risk profiling",
    },
    {
        "id": "S09",
        "message": "Where am I spending more than average in Ireland?",
        "expected_routing": "budget",
        "description": "HBS benchmark query → budget agent",
    },
    {
        "id": "S10",
        "message": "How did you decide my risk class?",
        "expected_routing": "explanation_request",
        "description": "Explanation of classification",
    },
]


# Helper

def make_orchestrator(tmp_path: Path) -> Orchestrator:
    """Create an Orchestrator with audit log in temp directory."""
    client = LLMClient()
    with patch("orchestrator.audit_log.settings") as mock_settings:
        mock_settings.orchestrator.audit_log_dir = tmp_path
        orch = Orchestrator(llm_client=client, session_id="integration-test-001")
    return orch


# Unit-level integration tests

class TestOrchestratorInit:

    def test_orchestrator_initialises(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="test-001")
        assert orch.session_id == "test-001"

    def test_all_agents_registered(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="test-001")
        expected_agents = {
            "ConversationalAgent", "RiskProfilingAgent",
            "InvestmentAgent", "BudgetAgent", "ExplainabilityAgent",
        }
        assert set(orch._agents.keys()) == expected_agents

    def test_session_state_initialised(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="test-001")
        assert "conversation_history" in orch._session_state
        assert "user_features" in orch._session_state
        assert orch._session_state["turn_count"] == 0


class TestAgentSequences:

    def _get_sequence(self, routing: RoutingDecision, tmp_path: Path) -> list[str]:
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="seq-test")
        return orch._get_agent_sequence(routing)

    def test_conversational_only_sequence(self, tmp_path):
        seq = self._get_sequence(RoutingDecision.CONVERSATIONAL_ONLY, tmp_path)
        assert seq == ["ConversationalAgent"]

    def test_risk_sequence_includes_explainability(self, tmp_path):
        seq = self._get_sequence(RoutingDecision.RISK_PROFILING, tmp_path)
        assert "RiskProfilingAgent" in seq
        assert "ExplainabilityAgent" in seq
        assert seq[-1] == "ExplainabilityAgent"

    def test_investment_sequence_correct_order(self, tmp_path):
        seq = self._get_sequence(RoutingDecision.INVESTMENT, tmp_path)
        assert seq.index("RiskProfilingAgent") < seq.index("InvestmentAgent")
        assert seq.index("InvestmentAgent") < seq.index("ExplainabilityAgent")

    def test_full_advisory_includes_all_agents(self, tmp_path):
        seq = self._get_sequence(RoutingDecision.FULL_ADVISORY, tmp_path)
        for agent in ["RiskProfilingAgent", "InvestmentAgent",
                      "BudgetAgent", "ExplainabilityAgent"]:
            assert agent in seq

    def test_explainability_always_last_in_non_conv(self, tmp_path):
        for routing in [
            RoutingDecision.RISK_PROFILING,
            RoutingDecision.INVESTMENT,
            RoutingDecision.FULL_ADVISORY,
        ]:
            seq = self._get_sequence(routing, tmp_path)
            if "ExplainabilityAgent" in seq:
                assert seq[-1] == "ExplainabilityAgent", (
                    f"ExplainabilityAgent must be last in {routing.value} sequence"
                )


class TestProcessTurnStructure:

    def test_process_turn_returns_orchestrator_result(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        from orchestrator.orchestrator import OrchestratorResult
        assert isinstance(result, OrchestratorResult)

    def test_result_has_session_id(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        assert result.session_id == "struct-test"

    def test_result_has_turn_id(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        assert result.turn_id
        assert len(result.turn_id) > 0

    def test_result_has_routing_decision(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        assert isinstance(result.routing_decision, RoutingDecision)

    def test_result_has_agents_invoked(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        assert isinstance(result.agents_invoked, list)
        assert len(result.agents_invoked) >= 1

    def test_result_has_final_response(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        result = orch.process_turn("Hello")
        assert isinstance(result.final_response, str)
        assert len(result.final_response) > 0

    def test_turn_count_increments(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        orch.process_turn("Hello")
        orch.process_turn("Tell me more")
        assert orch._session_state["turn_count"] == 2

    def test_conversation_history_grows(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="struct-test")
        orch.process_turn("Hello")
        orch.process_turn("What is risk profiling?")
        history = orch._session_state["conversation_history"]
        assert len(history) == 4  # 2 user + 2 assistant

    def test_audit_log_written_after_turn(self, tmp_path):
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="audit-test")
        orch.process_turn("Hello")
        records = orch.audit_log.read_all()
        event_types = {r["event_type"] for r in records}
        assert "TURN_START" in event_types
        assert "ROUTING_DECISION" in event_types
        assert "TURN_END" in event_types


# RQ4 integration evaluation — 10 scenarios
@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests

class TestRQ4Evaluation:

    def test_rq4_pipeline_evaluation_and_write_results(self, tmp_path):
        """
        Run 10 scripted scenarios through the full HALO pipeline.
        Measures: routing_accuracy, CSS, TUE, step_progress_rate.
        Writes: results/rq4_mas_coherence.json

        In mock mode: routing_accuracy = 0.0 (expected — LLM cannot classify).
        In real mode: routing_accuracy ≥ 0.70 (expected with Gemini-2.0-flash).

        DIAGNOSTIC INSTRUMENTATION (read-only — no orchestrator/ or agents/
        source changed, no routing behaviour changed):

        A misroute to CONVERSATIONAL_ONLY can currently come from at least
        three different places, and this file previously couldn't tell them
        apart:
          (a) the classifier genuinely, confidently says "general_query" —
              a real classification decision, nothing wrong;
          (b) Orchestrator._classify_intent()'s confidence gate
              (routing_confidence_threshold, config/settings.py) discards a
              *different*, non-conversational intent because its
              self-reported confidence fell below threshold;
          (c) ConversationalAgent._classify_intent() silently falls back to
              ("general_query", 0.5) because the LLM call raised, or its
              response didn't contain parseable JSON — 0.5 is always below
              the 0.65 default threshold, so this is indistinguishable from
              (a)/(b) downstream without extra instrumentation.

        Every one of the above already leaves a trace that this test just
        wasn't reading:
          - AuditLog.record_routing() already logs `intent` and `confidence`
            for every turn (orchestrator/audit_log.py) — read back here from
            audit_records instead of being left unread.
          - ConversationalAgent already exposes `last_classification_error`
            / `last_classification_error_status`, set only when path (c)
            above actually fires — read directly off the live agent
            instance after each turn.
          - ConversationalAgent.questionnaire_active tells us whether
            Orchestrator.process_turn()'s questionnaire short-circuit skipped
            _classify_intent() entirely this turn, which would otherwise
            make last_classification_error look misleadingly stale (it would
            hold whatever error, or lack of one, was left over from the last
            turn classification actually ran).

        See `classify_intent_diagnostics` on each scenario result and
        `classify_intent_diagnostics_summary` at the top level.
        """
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="rq4-eval-001")

        judge = AgentJudge(orch.judge_llm)
        conv_agent = orch._agents["ConversationalAgent"]

        scenario_results = []
        actual_routings = []
        expected_routings = []
        all_step_records = []
        judge_scores = []

        for scenario in SCENARIOS:
            # Snapshot taken BEFORE the turn: if ConversationalAgent was
            # already mid-questionnaire (set by a PRIOR turn),
            # process_turn()'s short-circuit will skip _classify_intent()
            # entirely this turn — so anything read from the agent AFTER
            # the call below would describe an earlier turn, not this one.
            questionnaire_shortcircuit = getattr(
                conv_agent, "questionnaire_active", False
            )

            result = orch.process_turn(scenario["message"])

            # Snapshot taken AFTER the turn: ConversationalAgent's own
            # record of its most recent real classification attempt —
            # meaningful only when the short-circuit above did NOT fire.
            classification_error = getattr(
                conv_agent, "last_classification_error", None
            )
            classification_error_status = getattr(
                conv_agent, "last_classification_error_status", None
            )

            actual_routing = result.routing_decision.value
            actual_routings.append(actual_routing)
            expected_routings.append(scenario["expected_routing"])

            # Step records for AgentBoard (E1)
            for agent_result in result.agent_results:
                all_step_records.append(agent_result.to_step_record())

            # Agent-as-Judge evaluation (E2)
            judge_score = judge.evaluate(
                result,
                ground_truth={
                    "expected_routing": scenario["expected_routing"],
                    "scenario_id": scenario["id"],
                },
            )
            judge_scores.append(judge_score)

            scenario_results.append({
                "scenario_id": scenario["id"],
                "description": scenario["description"],
                "message": scenario["message"],
                "expected_routing": scenario["expected_routing"],
                "actual_routing": actual_routing,
                "routing_correct": actual_routing == scenario["expected_routing"],
                "agents_invoked": result.agents_invoked,
                "conflicts": result.conflicts,
                "constraint_violations": result.constraint_violations,
                "recovered_agents": result.recovered_agents,
                "duration_ms": result.total_duration_ms,
                "judge_score": judge_score,
                # --- diagnostic instrumentation (see method docstring) ---
                "turn_id": result.turn_id,
                "classify_intent_diagnostics": {
                    "classification_skipped_questionnaire_shortcircuit":
                        questionnaire_shortcircuit,
                    "last_classification_error": classification_error,
                    "last_classification_error_http_status":
                        classification_error_status,
                    "note": (
                        "classification_skipped_questionnaire_shortcircuit "
                        "reflects questionnaire_active BEFORE this turn, not "
                        "whether _classify_intent() ran THIS turn — those are "
                        "no longer the same thing now that pivot-through "
                        "exists (an off_topic reply mid-questionnaire still "
                        "calls _classify_intent()). See "
                        "inferred_questionnaire_pivot_through below, filled "
                        "in once the audit log has been read, for the "
                        "turn-specific answer."
                    ),
                    # logged_intent / logged_confidence / logged_rationale
                    # filled in below once the audit log has been read —
                    # AuditLog.record_routing() logs these for every turn
                    # regardless of which of paths (a)/(b)/(c) produced them.
                },
            })

        # Compute RQ4 metrics
        routing_result = routing_accuracy(actual_routings, expected_routings)
        step_result = step_progress_rate(all_step_records)

        # Read audit log for CSS and TUE
        audit_records = orch.audit_log.read_all()
        css_result = component_synergy_score(audit_records)
        tue_result = tool_utilisation_efficacy(audit_records)

        # Join each scenario's ROUTING_DECISION record back in by turn_id.
        # record_routing() already logs intent + confidence for every turn
        # (orchestrator/audit_log.py) — this was sitting in the audit log,
        # already read into audit_records above, and simply never consulted
        # by this test before now.
        routing_payload_by_turn = {
            r["turn_id"]: r["payload"]
            for r in audit_records
            if r.get("event_type") == "ROUTING_DECISION"
        }
        n_questionnaire_absorbed = 0
        n_questionnaire_pivoted_through = 0
        n_silent_classification_failure = 0
        n_low_confidence_override = 0
        n_confident_general_query = 0
        _threshold = settings.orchestrator.routing_confidence_threshold
        # The orchestrator's hardcoded absorb-path (Orchestrator.process_turn,
        # the questionnaire short-circuit) always logs EXACTLY this
        # (intent, confidence) pair — it's a literal tuple, not a computed
        # value. Real classification, LLM-reported, essentially never lands
        # on exactly 1.0 — confirmed empirically: the 2026-08-15 pivot-through
        # rerun logged confidence=0.97/0.99 on every turn that actually
        # pivoted, never 1.0. This is a safe, purely-observational way to
        # tell "genuinely absorbed" apart from "pivoted through" without
        # needing Orchestrator to expose questionnaire_absorbs_this_turn
        # directly, which would mean editing orchestrator.py again for a
        # test-only signal.
        _HARDCODED_ABSORB_INTENTS = {"risk_profiling", "budget_analysis"}
        for sc in scenario_results:
            logged = routing_payload_by_turn.get(sc["turn_id"], {})
            diag = sc["classify_intent_diagnostics"]
            diag["logged_intent"] = logged.get("intent")
            diag["logged_confidence"] = logged.get("confidence")
            diag["logged_rationale"] = logged.get("rationale")

            was_in_questionnaire_before_turn = diag[
                "classification_skipped_questionnaire_shortcircuit"
            ]
            looks_like_hardcoded_absorb = (
                was_in_questionnaire_before_turn
                and diag["logged_confidence"] == 1.0
                and diag["logged_intent"] in _HARDCODED_ABSORB_INTENTS
            )
            diag["inferred_questionnaire_pivot_through"] = (
                was_in_questionnaire_before_turn and not looks_like_hardcoded_absorb
            )

            if looks_like_hardcoded_absorb:
                n_questionnaire_absorbed += 1
            elif was_in_questionnaire_before_turn:
                # questionnaire_active was true going into this turn, but the
                # logged routing does NOT match the hardcoded absorb tuple —
                # classify_questionnaire_reply() found this turn off_topic,
                # pivot-through fired, and the routing/intent/confidence
                # below came from a real _classify_intent() call, same as
                # any ordinary turn.
                n_questionnaire_pivoted_through += 1
            elif diag["last_classification_error"]:
                # Path (c): call/parse failure, masquerading downstream as
                # a confident general_query.
                n_silent_classification_failure += 1
            elif (
                diag["logged_confidence"] is not None
                and diag["logged_confidence"] < _threshold
                and diag["logged_intent"] not in (
                    "general_query", "out_of_scope", None,
                )
            ):
                # Path (b): a genuine, different intent was found and then
                # discarded by the confidence gate.
                n_low_confidence_override += 1
            elif diag["logged_intent"] in ("general_query", "out_of_scope"):
                # Path (a): a real, confident conversational classification.
                n_confident_general_query += 1

        classify_intent_diagnostics_summary = {
            "n_scenarios": len(scenario_results),
            "n_questionnaire_absorbed": n_questionnaire_absorbed,
            "n_questionnaire_pivoted_through": n_questionnaire_pivoted_through,
            "n_silent_classification_failure": n_silent_classification_failure,
            "n_low_confidence_override": n_low_confidence_override,
            "n_confident_general_query": n_confident_general_query,
            "n_unclassified": (
                len(scenario_results)
                - n_questionnaire_absorbed
                - n_questionnaire_pivoted_through
                - n_silent_classification_failure
                - n_low_confidence_override
                - n_confident_general_query
            ),
            "interpretation": (
                "n_questionnaire_absorbed counts turns whose logged routing "
                "matches the orchestrator's hardcoded questionnaire "
                "short-circuit tuple exactly (confidence==1.0, intent in "
                f"{sorted(_HARDCODED_ABSORB_INTENTS)}) — inferred from the "
                "audit log, not read directly off Orchestrator's internal "
                "questionnaire_absorbs_this_turn flag; see the loop above "
                "for why that's a reliable proxy. n_questionnaire_pivoted_"
                "through counts turns where questionnaire_active was true "
                "beforehand but the logged routing does NOT match that "
                "tuple — classify_questionnaire_reply() found the turn "
                "off_topic and it went through real classification instead. "
                "n_silent_classification_failure counts turns where "
                "_classify_intent() ran but the LLM call or JSON parse "
                "failed, silently producing (general_query, 0.5) — "
                "indistinguishable from a genuine general_query "
                "classification anywhere else in this file before this "
                "instrumentation was added. n_low_confidence_override "
                "counts turns where a different, non-conversational intent "
                "WAS found but discarded for falling below "
                f"routing_confidence_threshold ({_threshold}). "
                "n_unclassified covers cases this rollup's rules don't "
                "cleanly bucket (e.g. a genuinely low-confidence "
                "general_query) — inspect classify_intent_diagnostics on "
                "the individual scenario for those."
            ),
        }

        # Judge aggregate
        modes = [s.get("judge_mode", "unavailable") for s in judge_scores]
        measured = [s for s in judge_scores if s.get("judge_mode") == "real"]
        judge_coverage = len(measured) / len(judge_scores) if judge_scores else 0.0
        mean_judge_score = (
            sum(s["overall_score"] for s in measured) / len(measured)
            if measured else None
        )
        judge_mode = (
            "real" if judge_coverage == 1.0
            else "partial" if measured
            else (modes[0] if modes else "unavailable")
        )
        verdicts = [s.get("verdict", "flag") for s in measured]

        judge_failures = {}
        for s in judge_scores:
            m = s.get("judge_mode")
            if m and m != "real" and m not in judge_failures:
                judge_failures[m] = {
                    "reason": s.get("judge_failure_reason", ""),
                    "raw_preview": s.get("judge_raw_preview", ""),
                }

        # Write results
        results_payload = {
            "phase": 7,
            "agent": "Orchestrator",
            "llm_mode": orch.llm.mode,
            "n_scenarios": len(SCENARIOS),
            "note": (
                "routing_accuracy=0.0 in mock mode (expected — intent classifier "
                "requires real LLM). All other metrics are meaningful in both modes."
            ),
            "metrics": {
                "routing_accuracy": routing_result.to_dict(),
                "step_progress_rate": step_result.to_dict(),
                "component_synergy_score": css_result.to_dict(),
                "tool_utilisation_efficacy": tue_result.to_dict(),
                "agent_judge": {
                    "metric": "agent_judge_overall",
                    "value": (round(mean_judge_score, 3)
                              if mean_judge_score is not None else None),
                    "judge_mode": judge_mode,
                    "judge_model": settings.llm.judge_model,
                    "judge_coverage": round(judge_coverage, 3),
                    "n_measured": len(measured),
                    "n_scenarios": len(judge_scores),
                    "mode_distribution": {m: modes.count(m) for m in set(modes)},
                    "failures": judge_failures,
                    "verdict_distribution": {
                        v: verdicts.count(v) for v in set(verdicts)
                    },
                    "interpretation": (
                        "value is the mean over MEASURED turns only. "
                        "judge_coverage < 1.0 means some turns were not judged; "
                        "coverage 0.0 means the 5-dimension trajectory score was "
                        "NOT measured in this run and must not be reported as a "
                        "finding."
                    ),
                },
            },
            "scenarios": scenario_results,
            "classify_intent_diagnostics_summary": classify_intent_diagnostics_summary,
            "audit_log_path": str(orch.audit_log.log_path),
        }

        results_path = write_results(
            results_payload, "rq4_mas_coherence.json", orch.llm.mode
        )

        print(f"\n[RQ4] routing_accuracy:          {routing_result.value:.3f}")
        print(f"[RQ4] step_progress_rate:        {step_result.value:.3f}")
        print(f"[RQ4] component_synergy_score:   {css_result.value:.3f}")
        print(f"[RQ4] tool_utilisation_efficacy: {tue_result.value:.3f}")
        if mean_judge_score is None:
            print(f"[RQ4] agent_judge_overall:       NOT MEASURED "
                  f"(mode={judge_mode}, 0/{len(judge_scores)} turns judged)")
            for m, detail in judge_failures.items():
                print(f"[RQ4]   {m}: {detail['reason'][:110]}")
                if detail["raw_preview"]:
                    print(f"[RQ4]   raw: {detail['raw_preview'][:110]!r}")
        else:
            print(f"[RQ4] agent_judge_overall:       {mean_judge_score:.3f} "
                  f"({judge_mode}, {len(measured)}/{len(judge_scores)} turns judged, "
                  f"model={settings.llm.judge_model})")
        print(
            f"[RQ4] classify_intent diagnostics: "
            f"{n_confident_general_query} confident-general-query, "
            f"{n_low_confidence_override} low-confidence-override, "
            f"{n_silent_classification_failure} silent-classification-failure, "
            f"{n_questionnaire_absorbed} questionnaire-absorbed, "
            f"{n_questionnaire_pivoted_through} questionnaire-pivoted-through, "
            f"{classify_intent_diagnostics_summary['n_unclassified']} unclassified"
        )
        print(f"[RQ4] Results written to {results_path}")

        # Structural assertions — always pass regardless of mode
        assert 0.0 <= routing_result.value <= 1.0
        assert 0.0 <= step_result.value <= 1.0
        assert 0.0 <= css_result.value <= 1.0
        assert 0.0 <= tue_result.value <= 1.0
        assert len(scenario_results) == len(SCENARIOS)
        # Structural check on the new diagnostic fields — always passes
        # regardless of mode, mirrors the assertions above.
        assert all(
            "classify_intent_diagnostics" in s for s in scenario_results
        )
        assert all(
            "logged_intent" in s["classify_intent_diagnostics"]
            for s in scenario_results
        )

class TestRQ4MultiAgentCoordination:
    """
    Complementary to TestRQ4Evaluation above — not a replacement for it,
    and not claiming to be a "more correct" version of it.

    WHY THIS CLASS EXISTS
        TestRQ4Evaluation's 10 scenarios deliberately start from a blank
        session (no customer_id) to test COLD-START ROUTING — can the
        system tell what a first, context-free message is about. That
        design has a side effect worth naming directly, checked against
        a real run rather than assumed: 9 of 10 scenarios invoked
        exactly one agent (the conversational_only fallback); only S02
        ever invoked two. component_synergy_score=1.00 and
        tool_utilisation_efficacy=1.00 in that file are real numbers,
        but they are "0 conflicts across a sample that almost never had
        the chance to conflict" — not strong evidence of multi-agent
        coordination specifically, because the scenarios that would
        exercise it structurally couldn't occur with no seeded profile.

        This class seeds a COMPLETE, real customer profile via
        customer_id (data/processed/customers.csv's DEMO_* rows, which
        also carry transaction history in data/processed/
        transactions.json — checked directly, not assumed, before
        writing these scenarios) before the turn runs. RiskProfilingAgent
        then succeeds immediately instead of triggering elicitation, and
        BudgetAgent has enough transaction history not to need a
        questionnaire either — which is what lets full_advisory actually
        route all four agents (RiskProfilingAgent, InvestmentAgent,
        BudgetAgent, ExplainabilityAgent — agents/payloads.py's
        STATIC_SEQUENCES) in a single turn, the condition CSS/TUE need
        to be measuring something under.

    WHY ROUTING IS FORCED HERE, UNLIKE TestRQ4Evaluation
        TestRQ4Evaluation lets real classification run because routing
        accuracy IS what it measures. This class measures something
        different — coordination GIVEN a routing decision — so
        _classify_intent is monkeypatched to the scenario's intended
        route, isolating that variable. This also means every scenario
        here runs correctly in force_mock=True with no API key: the
        specialist agents' scoring logic (RiskProfilingAgent,
        InvestmentAgent, BudgetAgent) is deterministic regardless of LLM
        mode, and what this class checks — agent counts, handoffs,
        conflicts — is structural, not narrative content. Narrative
        quality is TestRQ4Evaluation's agent-judge's job, not this one's.

    WHAT THIS DOES NOT DO
        It does not manufacture a conflict to prove conflict-resolution
        works. conflicts/constraint_violations/recovered_agents are
        recorded as OBSERVED per scenario, never asserted empty or
        non-empty — forcing an assertion either way without independent
        evidence of what SHOULD happen would be inventing ground truth,
        not measuring behaviour. The one hard assertion this class does
        make is that the expected agents actually ran: if that fails,
        the seeded profile isn't as complete as this docstring claims,
        and the scenario needs fixing, not the assertion loosened.
    """

    MULTI_AGENT_SCENARIOS = [
        {
            "id": "M01",
            "description": "Full advisory review, complete existing profile",
            "customer_id": "DEMO_GC_042",  # moderate; complete features + transactions
            "message": "Can you give me a full review — my risk profile, "
                       "some investment ideas, and how my budget looks?",
            "expected_routing": "full_advisory",
            "expected_agents": ["RiskProfilingAgent", "InvestmentAgent",
                                "BudgetAgent", "ExplainabilityAgent"],
        },
        {
            "id": "M02",
            "description": "Investment request, complete existing profile",
            "customer_id": "DEMO_GC_107",  # moderately_aggressive
            "message": "What investment products would suit me right now?",
            "expected_routing": "investment",
            "expected_agents": ["RiskProfilingAgent", "InvestmentAgent",
                                "ExplainabilityAgent"],
        },
        {
            "id": "M03",
            "description": "Budget explanation, complete existing profile",
            "customer_id": "DEMO_GMSC_318",  # conservative, retired
            "message": "Can you break down my spending and explain where "
                       "I could cut back?",
            "expected_routing": "budget",
            "expected_agents": ["BudgetAgent", "ExplainabilityAgent"],
        },
        {
            "id": "M04",
            "description": "Full advisory, second profile — different risk "
                           "tier than M01, same routing decision",
            "customer_id": "DEMO_GMSC_318",
            "message": "I'd like the full picture — risk, investing, and "
                       "budgeting, all of it.",
            "expected_routing": "full_advisory",
            "expected_agents": ["RiskProfilingAgent", "InvestmentAgent",
                                "BudgetAgent", "ExplainabilityAgent"],
        },
    ]

    def test_seeded_profiles_actually_enable_multi_agent_turns(self, tmp_path):
        """
        Writes: results/{mode}/rq4_multi_agent_coordination.json

        Structural checks only — see class docstring for why conflicts/
        constraint_violations are recorded, not asserted.
        """
        scenario_results = []
        all_audit_records = []

        for scenario in self.MULTI_AGENT_SCENARIOS:
            with patch("orchestrator.audit_log.settings") as mock_s:
                mock_s.orchestrator.audit_log_dir = tmp_path
                orch = Orchestrator(
                    LLMClient(force_mock=True),
                    session_id=f"rq4-multiagent-{scenario['id']}",
                    customer_id=scenario["customer_id"],
                )

            _routing = RoutingDecision(scenario["expected_routing"])
            orch._classify_intent = lambda msg, _r=_routing, _i=scenario["expected_routing"]: (
                _r, _i, 0.95,
            )

            result = orch.process_turn(scenario["message"])
            agents_invoked = result.agents_invoked
            expected_set = set(scenario["expected_agents"])
            invoked_set = set(agents_invoked)

            all_audit_records.extend(orch.audit_log.read_all())

            scenario_results.append({
                "scenario_id": scenario["id"],
                "description": scenario["description"],
                "customer_id": scenario["customer_id"],
                "message": scenario["message"],
                "expected_routing": scenario["expected_routing"],
                "actual_routing": result.routing_decision.value,
                "expected_agents": scenario["expected_agents"],
                "agents_invoked": agents_invoked,
                "n_agents_invoked": len(agents_invoked),
                "all_expected_agents_ran": expected_set.issubset(invoked_set),
                "unexpected_agents": sorted(invoked_set - expected_set),
                "missing_expected_agents": sorted(expected_set - invoked_set),
                "conflicts": result.conflicts,
                "constraint_violations": result.constraint_violations,
                "recovered_agents": result.recovered_agents,
                "duration_ms": result.total_duration_ms,
            })

        css_result = component_synergy_score(all_audit_records)
        tue_result = tool_utilisation_efficacy(all_audit_records)

        n_multi_agent = sum(1 for s in scenario_results if s["n_agents_invoked"] > 1)
        n_all_expected_ran = sum(1 for s in scenario_results if s["all_expected_agents_ran"])
        total_conflicts = sum(len(s["conflicts"]) for s in scenario_results)
        total_violations = sum(len(s["constraint_violations"]) for s in scenario_results)

        results_payload = {
            "phase": 7,
            "research_question": (
                "RQ4 (supplementary) — multi-agent coordination under "
                "conditions where multiple agents actually run together, "
                "complementing TestRQ4Evaluation's cold-start routing "
                "scenarios (see this class's own docstring for why that "
                "file's CSS/TUE=1.00 numbers are under-tested)"
            ),
            "n_scenarios": len(self.MULTI_AGENT_SCENARIOS),
            "metrics": {
                "component_synergy_score": css_result.to_dict(),
                "tool_utilisation_efficacy": tue_result.to_dict(),
            },
            "coordination_coverage": {
                "n_scenarios_invoking_multiple_agents": n_multi_agent,
                "n_scenarios_of_total": len(self.MULTI_AGENT_SCENARIOS),
                "n_scenarios_where_all_expected_agents_ran": n_all_expected_ran,
                "total_conflicts_observed": total_conflicts,
                "total_constraint_violations_observed": total_violations,
                "note": (
                    "n_scenarios_invoking_multiple_agents is the number "
                    "this class exists to move off 1/10 (TestRQ4Evaluation's "
                    "real figure) — compare directly. conflicts/violations "
                    "are OBSERVED counts, not targets: 0 here means these "
                    "particular seeded profiles didn't produce a "
                    "disagreement between agents, not that conflict "
                    "resolution was exercised and passed."
                ),
            },
            "scenarios": scenario_results,
        }

        results_path = write_results(
            results_payload, "rq4_multi_agent_coordination.json", orch.llm.mode,
        )

        print(f"\n[RQ4-multiagent] scenarios invoking >1 agent: "
              f"{n_multi_agent}/{len(self.MULTI_AGENT_SCENARIOS)} "
              f"(TestRQ4Evaluation's cold-start scenarios: 1/10)")
        print(f"[RQ4-multiagent] all expected agents ran: "
              f"{n_all_expected_ran}/{len(self.MULTI_AGENT_SCENARIOS)}")
        print(f"[RQ4-multiagent] component_synergy_score: {css_result.value:.3f}")
        print(f"[RQ4-multiagent] tool_utilisation_efficacy: {tue_result.value:.3f}")
        print(f"[RQ4-multiagent] conflicts observed: {total_conflicts}, "
              f"constraint_violations observed: {total_violations}")
        print(f"[RQ4-multiagent] Results written to {results_path}")

        # Hard assertion — see class docstring for why this one, unlike
        # conflicts/violations, is not just observed-and-reported.
        for s in scenario_results:
            assert s["all_expected_agents_ran"], (
                f"{s['scenario_id']}: expected {s['expected_agents']} to all "
                f"run, only got {s['agents_invoked']} — the seeded profile "
                f"for {s['customer_id']} is not as complete as this test "
                f"assumes. Fix the scenario/profile, not this assertion."
            )
        assert 0.0 <= css_result.value <= 1.0
        assert 0.0 <= tue_result.value <= 1.0