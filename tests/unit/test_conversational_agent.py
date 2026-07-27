"""
Phase 2 — ConversationalAgent evaluation.

Three test groups:

GROUP A: Unit tests (no LLM needed)
  Tests agent logic — slot management, escalation decision, result structure —
  entirely in mock mode. These never call an LLM and never fail due to
  API availability.

GROUP B: Intent accuracy evaluation (mock mode)
  Simulates Banking77 [D5] classification across a fixed 20-query fixture.
  In mock mode the LLM returns "[MOCK RESPONSE]" so all predictions land
  in "general_query". This gives us a baseline score of 0/20 intentionally
  — it proves the metric function works and gives a floor measurement to
  compare against when a real LLM key is available.

GROUP C: Slot fill rate evaluation (logic-only, no LLM)
  Tests the slot tracking logic directly without LLM calls.
  Uses hand-crafted session fixtures representing realistic multi-turn
  conversations.

RUNNING:
  Mock mode (no API key):
    pytest tests/unit/test_conversational_agent.py -v

  Real mode (OPENAI_API_KEY in .env):
    pytest tests/unit/test_conversational_agent.py -v -m real

  Generate results JSON:
    pytest tests/unit/test_conversational_agent.py -v --tb=short
    (results written to results/phase2_conversational_baseline.json)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.conversational_agent import INTENT_BUCKETS, ConversationalAgent
from evaluation.metrics import EvalResult, intent_accuracy, slot_fill_rate
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


# Helpers

def make_agent(mock_response: str = "[MOCK RESPONSE]") -> ConversationalAgent:
    """Return a ConversationalAgent in mock mode (no API key required)."""
    client = LLMClient()
    assert client.mode == "mock", "Tests must run in mock mode without API key"
    return ConversationalAgent(client)

def make_real_agent() -> ConversationalAgent:
    """Return a ConversationalAgent using the real API key from .env."""
    from dotenv import load_dotenv
    load_dotenv()
    client = LLMClient()
    return ConversationalAgent(client)


def make_agent_with_responses(responses: list[str]) -> ConversationalAgent:
    """
    Return an agent whose LLM calls return a fixed sequence of responses.
    Useful for testing multi-turn slot extraction and intent classification
    without any API call.
    """
    client = LLMClient()
    response_iter = iter(responses)

    def fake_chat(system, messages, temperature=None):
        content = next(response_iter, "[MOCK RESPONSE]")
        mock_resp = MagicMock()
        mock_resp.content = content
        mock_resp.tokens_used = 0
        return mock_resp

    client.chat = fake_chat
    return ConversationalAgent(client)


# GROUP A: Unit tests — logic only, no LLM

class TestSlotManagement:

    def test_slots_empty_on_init(self):
        agent = make_agent()
        assert agent.slots == {}

    def test_update_slots_stores_valid_slot(self):
        agent = make_agent()
        agent.update_slots({"age": 35})
        assert agent.slots["age"] == 35

    def test_update_slots_ignores_unknown_slot(self):
        agent = make_agent()
        agent.update_slots({"unknown_field": "value"})
        assert "unknown_field" not in agent.slots

    def test_update_slots_ignores_none_value(self):
        agent = make_agent()
        agent.update_slots({"age": None})
        assert "age" not in agent.slots

    def test_update_slots_overwrites_existing(self):
        agent = make_agent()
        agent.update_slots({"age": 30})
        agent.update_slots({"age": 35})
        assert agent.slots["age"] == 35

    def test_get_missing_slots_all_missing(self):
        agent = make_agent()
        missing = agent.get_missing_slots(["age", "income"])
        assert set(missing) == {"age", "income"}

    def test_get_missing_slots_partially_collected(self):
        agent = make_agent()
        agent.update_slots({"age": 35})
        missing = agent.get_missing_slots(["age", "income"])
        assert missing == ["income"]

    def test_get_missing_slots_all_collected(self):
        agent = make_agent()
        agent.update_slots({"age": 35, "income": 60000})
        missing = agent.get_missing_slots(["age", "income"])
        assert missing == []

    def test_multiple_valid_slots_stored(self):
        agent = make_agent()
        agent.update_slots({"age": 40, "income": 75000, "investment_goal": "retirement"})
        assert len(agent.slots) == 3


class TestEscalationLogic:

    def test_escalation_needed_for_investment_intent(self):
        agent = make_agent()
        assert agent._needs_escalation("investment_advice", confidence=0.9) is True

    def test_no_escalation_for_general_query(self):
        agent = make_agent()
        assert agent._needs_escalation("general_query", confidence=0.9) is False

    def test_no_escalation_below_confidence_threshold(self):
        agent = make_agent()
        # Intent warrants escalation but confidence too low — ask for clarification
        assert agent._needs_escalation("investment_advice", confidence=0.4) is False

    def test_escalation_block_structure(self):
        agent = make_agent()
        agent.update_slots({"age": 35})
        block = agent._build_escalation_block("investment_advice", "I want to invest")
        assert block["escalate"] is True
        assert block["intent"] == "investment_advice"
        assert "collected_slots" in block
        assert block["collected_slots"]["age"] == 35
        assert block["user_message"] == "I want to invest"

    def test_escalation_block_contains_current_slots(self):
        agent = make_agent()
        agent.update_slots({"age": 42, "income": 80000})
        block = agent._build_escalation_block("risk_profiling", "Assess my risk")
        assert len(block["collected_slots"]) == 2


class TestAgentResult:

    def test_run_returns_agent_result(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        from agents.base_agent import AgentResult
        assert isinstance(result, AgentResult)

    def test_result_has_agent_name(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        assert result.agent_name == "ConversationalAgent"

    def test_result_payload_has_required_keys(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        for key in ("intent", "confidence", "escalation_needed", "response",
                    "collected_slots", "turn_count"):
            assert key in result.payload, f"Missing key: {key}"

    def test_empty_message_returns_error_result(self):
        agent = make_agent()
        result = agent.run({"user_message": ""})
        assert result.success is False
        assert result.error is not None

    def test_turn_count_increments(self):
        agent = make_agent()
        agent.run({"user_message": "Hello"})
        agent.run({"user_message": "Tell me more"})
        assert agent.turn_count == 2

    def test_history_updated_after_run(self):
        agent = make_agent()
        agent.run({"user_message": "Hello"})
        assert len(agent.history) == 2   # user turn + assistant turn

    def test_result_step_record_has_required_fields(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        step = result.to_step_record()
        for key in ("step_id", "agent", "completed", "duration_ms"):
            assert key in step

    def test_duration_ms_is_positive(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        assert result.duration_ms >= 0.0

    def test_routing_context_has_intent(self):
        agent = make_agent()
        result = agent.run({"user_message": "Hello"})
        assert "intent" in result.routing_context


class TestIntentBuckets:

    def test_all_escalation_intents_are_valid_buckets(self):
        from agents.conversational_agent import INTENT_ALIASES
        from config.settings import settings
        for intent in settings.conversational.escalation_intents:
            resolved = INTENT_ALIASES.get(intent, intent)
            assert resolved in INTENT_BUCKETS, (
                f"Escalation intent '{intent}' (resolved: '{resolved}') not in INTENT_BUCKETS"
            )

    def test_intent_buckets_non_empty(self):
        for bucket, classes in INTENT_BUCKETS.items():
            assert len(classes) > 0, f"Bucket '{bucket}' has no classes"


# GROUP B: Intent accuracy evaluation — Banking77 fixture

# Fixed 20-query Banking77-style fixture with labelled routing buckets.
# Ground truth manually assigned from Banking77 intent taxonomy [D5].
BANKING77_FIXTURE: list[dict] = [
    {"message": "What is my account balance?",             "bucket": "general_query"},
    {"message": "I want to invest my savings.",            "bucket": "investment_advice"},
    {"message": "Can you assess my financial risk?",       "bucket": "risk_profiling"},
    {"message": "I need to make a budget plan.",           "bucket": "budget_analysis"},
    {"message": "What credit card should I get?",          "bucket": "product_suggestion"},
    {"message": "Why did you recommend that fund?",        "bucket": "explanation_request"},
    {"message": "I need legal advice about my mortgage.",  "bucket": "out_of_scope"},
    {"message": "What is the current exchange rate?",      "bucket": "general_query"},
    {"message": "Help me plan for retirement.",            "bucket": "investment_advice"},
    {"message": "How much risk can I take?",               "bucket": "risk_profiling"},
    {"message": "Where am I spending too much?",           "bucket": "budget_analysis"},
    {"message": "Tell me about savings accounts.",         "bucket": "investment_advice"},
    {"message": "How did you calculate that?",             "bucket": "explanation_request"},
    {"message": "Can you recommend an ETF?",               "bucket": "investment_advice"},
    {"message": "My card is about to expire.",             "bucket": "general_query"},
    {"message": "What are my investment options?",         "bucket": "investment_advice"},
    {"message": "Help me understand my spending.",         "bucket": "budget_analysis"},
    {"message": "I want a low-risk investment.",           "bucket": "risk_profiling"},
    {"message": "Can I get a loan?",                       "bucket": "product_suggestion"},
    {"message": "Explain your reasoning.",                 "bucket": "explanation_request"},
]


class TestIntentAccuracyEvaluation:
    """
    Runs intent classification on the Banking77 fixture and measures accuracy.
    In mock mode: all predictions are 'general_query' (LLM returns mock string).
    The test does NOT fail on low accuracy — it records whatever score the
    current LLM achieves. The important thing is that the metric runs and
    the JSON is written.
    """

    def test_intent_accuracy_metric_runs(self):
        """Metric function works with valid inputs."""
        preds = ["general_query"] * 5
        gold  = ["general_query"] * 5
        result = intent_accuracy(preds, gold)
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.value <= 1.0

    def test_intent_accuracy_perfect(self):
        preds = ["general_query", "investment_advice"]
        gold  = ["general_query", "investment_advice"]
        result = intent_accuracy(preds, gold)
        assert result.value == 1.0

    def test_intent_accuracy_zero(self):
        preds = ["general_query", "general_query"]
        gold  = ["investment_advice", "risk_profiling"]
        result = intent_accuracy(preds, gold)
        assert result.value == 0.0

    def test_intent_accuracy_length_mismatch_returns_error(self):
        result = intent_accuracy(["a"], ["a", "b"])
        assert result.value == 0.0
        assert "error" in result.details

    def test_intent_accuracy_empty_returns_error(self):
        result = intent_accuracy([], [])
        assert "error" in result.details

    def test_intent_accuracy_has_per_bucket_breakdown(self):
        preds = ["general_query", "investment_advice", "general_query"]
        gold  = ["general_query", "general_query",    "general_query"]
        result = intent_accuracy(preds, gold)
        assert "per_bucket_accuracy" in result.details

    def test_banking77_fixture_evaluation_and_write_results(self):
        """
        Full evaluation run on the 20-query Banking77 fixture.
        Writes results/phase2_conversational_baseline.json.
        In mock mode: accuracy will be low (LLM can't classify without real API).
        In real mode: expected accuracy 0.70+ based on GPT-4o-mini benchmark.
        """
        agent = make_real_agent()

        predictions = []
        for item in BANKING77_FIXTURE:
            result = agent.run({"user_message": item["message"]})
            time.sleep(2)
            predictions.append(result.payload.get("intent", "general_query"))

        gold = [item["bucket"] for item in BANKING77_FIXTURE]
        eval_result = intent_accuracy(predictions, gold)

        # Write to results/ for dissertation evidence
        RESULTS_DIR.mkdir(exist_ok=True)
        results_payload = {
            "phase": 2,
            "agent": "ConversationalAgent",
            "dataset": "Banking77_fixture_20queries",
            "llm_mode": agent.llm.mode,
            **eval_result.to_dict(),
            "fixture": [
                {"message": item["message"], "gold": item["bucket"], "pred": pred}
                for item, pred in zip(BANKING77_FIXTURE, predictions)
            ],
        }
        results_path = write_results(
            results_payload, "phase2_conversational_baseline.json", agent.llm.mode
        )

        print(f"\n[Phase 2 Eval] Intent accuracy: {eval_result.value:.3f}")
        print(f"[Phase 2 Eval] Results written to {results_path}")

        # Assertion: metric ran and produced a valid score
        # NOT asserting a minimum score — this is a measurement, not a pass/fail gate.
        assert 0.0 <= eval_result.value <= 1.0


# GROUP C: Slot fill rate evaluation

class TestSlotFillRateMetric:

    def test_metric_runs_with_valid_sessions(self):
        sessions = [
            {"collected_slots": {"age": 35, "income": 60000}, "escalated": True},
            {"collected_slots": {"age": 40}, "escalated": True},
        ]
        result = slot_fill_rate(sessions, required_slots=["age", "income"])
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.value <= 1.0

    def test_all_slots_collected(self):
        sessions = [
            {"collected_slots": {"age": 35, "income": 60000}, "escalated": True},
        ]
        result = slot_fill_rate(sessions, required_slots=["age", "income"])
        assert result.value == 1.0

    def test_no_slots_collected(self):
        sessions = [{"collected_slots": {}, "escalated": True}]
        result = slot_fill_rate(sessions, required_slots=["age", "income"])
        assert result.value == 0.0

    def test_partial_slots_collected(self):
        sessions = [
            {"collected_slots": {"age": 35}, "escalated": True},
        ]
        result = slot_fill_rate(sessions, required_slots=["age", "income"])
        assert result.value == pytest.approx(0.5, abs=0.01)

    def test_empty_sessions_returns_error(self):
        result = slot_fill_rate([], required_slots=["age"])
        assert "error" in result.details

    def test_per_slot_fill_in_details(self):
        sessions = [
            {"collected_slots": {"age": 35, "income": 60000}, "escalated": True},
            {"collected_slots": {"age": 40, "income": None}, "escalated": True},
        ]
        result = slot_fill_rate(sessions, required_slots=["age", "income"])
        assert "per_slot_fill_rate" in result.details
        assert result.details["per_slot_fill_rate"]["age"] == 1.0

    def test_slot_fill_across_multi_turn_fixture(self):
        """
        Simulates 5 escalated sessions with varying slot completeness.
        Writes slot fill metrics to results/phase2_conversational_baseline.json
        (merged with intent accuracy results above).
        """
        sessions = [
            {"collected_slots": {"age": 35, "income": 60000, "investment_goal": "growth"}, "escalated": True, "turns_taken": 3},
            {"collected_slots": {"age": 28, "income": 45000}, "escalated": True, "turns_taken": 2},
            {"collected_slots": {"age": 52, "income": 90000, "risk_tolerance": "low"}, "escalated": True, "turns_taken": 4},
            {"collected_slots": {"age": 44}, "escalated": True, "turns_taken": 2},
            {"collected_slots": {"age": 31, "income": 55000, "investment_goal": "savings"}, "escalated": True, "turns_taken": 3},
        ]
        required = ["age", "income", "investment_goal"]
        result = slot_fill_rate(sessions, required_slots=required)

        print(f"\n[Phase 2 Eval] Slot fill rate: {result.value:.3f}")
        print(f"  Per-slot: {result.details['per_slot_fill_rate']}")

        assert 0.0 <= result.value <= 1.0
        assert result.details["n_sessions"] == 5
