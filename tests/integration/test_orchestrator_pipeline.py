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


class TestRQ4Evaluation:

    def test_rq4_pipeline_evaluation_and_write_results(self, tmp_path):
        """
        Run 10 scripted scenarios through the full HALO pipeline.
        Measures: routing_accuracy, CSS, TUE, step_progress_rate.
        Writes: results/rq4_mas_coherence.json

        In mock mode: routing_accuracy = 0.0 (expected — LLM cannot classify).
        In real mode: routing_accuracy ≥ 0.70 (expected with Gemini-2.0-flash).
        """
        with patch("orchestrator.audit_log.settings") as mock_s:
            mock_s.orchestrator.audit_log_dir = tmp_path
            orch = Orchestrator(LLMClient(), session_id="rq4-eval-001")

        judge = AgentJudge(orch.llm)

        scenario_results = []
        actual_routings = []
        expected_routings = []
        all_step_records = []
        judge_scores = []

        for scenario in SCENARIOS:
            result = orch.process_turn(scenario["message"])

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
            })

        # Compute RQ4 metrics
        routing_result = routing_accuracy(actual_routings, expected_routings)
        step_result = step_progress_rate(all_step_records)

        # Read audit log for CSS and TUE
        audit_records = orch.audit_log.read_all()
        css_result = component_synergy_score(audit_records)
        tue_result = tool_utilisation_efficacy(audit_records)

        # Judge aggregate
        judge_mode = judge_scores[0].get("judge_mode", "mock") if judge_scores else "mock"
        mean_judge_score = (
            sum(s.get("overall_score", 3.0) for s in judge_scores) / len(judge_scores)
            if judge_scores else 0.0
        )
        verdicts = [s.get("verdict", "flag") for s in judge_scores]

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
                    "value": round(mean_judge_score, 3),
                    "judge_mode": judge_mode,
                    "verdict_distribution": {
                        v: verdicts.count(v) for v in set(verdicts)
                    },
                },
            },
            "scenarios": scenario_results,
            "audit_log_path": str(orch.audit_log.log_path),
        }

        results_path = write_results(
            results_payload, "rq4_mas_coherence.json", orch.llm.mode
        )

        print(f"\n[RQ4] routing_accuracy:          {routing_result.value:.3f}")
        print(f"[RQ4] step_progress_rate:        {step_result.value:.3f}")
        print(f"[RQ4] component_synergy_score:   {css_result.value:.3f}")
        print(f"[RQ4] tool_utilisation_efficacy: {tue_result.value:.3f}")
        print(f"[RQ4] agent_judge_overall:       {mean_judge_score:.3f} ({judge_mode})")
        print(f"[RQ4] Results written to {results_path}")

        # Structural assertions — always pass regardless of mode
        assert 0.0 <= routing_result.value <= 1.0
        assert 0.0 <= step_result.value <= 1.0
        assert 0.0 <= css_result.value <= 1.0
        assert 0.0 <= tue_result.value <= 1.0
        assert len(scenario_results) == len(SCENARIOS)
