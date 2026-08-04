"""
Tests for the planner-vs-static evaluation harness (Day 4-5 / RQ4).

WHY A HARNESS NEEDS TESTS
    scripts/eval_planner_vs_static.py produces numbers that go into the
    thesis. An arithmetic slip in the agreement-rate denominator would not
    crash anything — it would quietly produce a plausible figure that is
    wrong, and it would be defended in the viva. These tests drive the
    harness with a scripted planner whose behaviour is known exactly, so the
    expected summary can be written down by hand and compared.

RUNNING
    python -m pytest tests/unit/test_planner_vs_static_eval.py -v
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agents.payloads import STATIC_SEQUENCES
from orchestrator.planner import Planner
from scripts.eval_planner_vs_static import SCENARIOS, run
from utils.llm_client import LLMClient


@pytest.fixture(autouse=True)
def _planner_on(monkeypatch):
    """See tests/unit/test_planner.py — the harness measures the planner, so
    an ambient PLANNER_ENABLED=false (the static arm's own setting) must not
    silently turn these into a test of the fallback."""
    from config.settings import settings
    monkeypatch.setattr(settings.planner, "enabled", True)


KNOWN_USER = {
    "user_features": {"age": 35, "income": 55000, "existing_debt": 5000,
                      "dependents": 1},
    "monthly_income": 4500,
    "monthly_expenses": 2800,
}


def scripted_planner(responses: list[str]) -> Planner:
    client = LLMClient(force_mock=True)
    it = iter(responses)

    def fake_chat(system, messages, temperature=None):
        resp = MagicMock()
        resp.content = next(it, "")
        resp.tokens_used = 10
        return resp

    client.chat = fake_chat
    return Planner(client)


def as_json(steps: list[str]) -> str:
    return json.dumps({"plan": steps, "reason": "scripted"})


def scenario(sid: str, intent: str, context: dict) -> dict:
    return {"id": sid, "message": "m", "intent": intent,
            "context": context, "probes": "test"}


class TestSummaryArithmetic:

    def test_all_accepted_and_all_agreeing(self):
        scenarios = [
            scenario("A", "investment", KNOWN_USER),
            scenario("B", "budget", KNOWN_USER),
        ]
        planner = scripted_planner([
            as_json(STATIC_SEQUENCES["investment"]),
            as_json(STATIC_SEQUENCES["budget"]),
        ])
        s = run(scenarios, planner)["summary"]
        assert s["plan_validity_rate"] == 1.0
        assert s["agreement_rate_over_accepted"] == 1.0
        assert s["disagreements"] == []
        assert s["rejection_counts"] == {}

    def test_accepted_but_disagreeing_is_not_a_rejection(self):
        """A shorter valid plan is the outcome G6 exists to produce. It must
        count as accepted AND as a disagreement — conflating the two would
        make a working planner look broken."""
        scenarios = [scenario("A", "investment", KNOWN_USER)]
        planner = scripted_planner(
            [as_json(["RiskProfilingAgent", "InvestmentAgent"])]
        )
        s = run(scenarios, planner)["summary"]
        assert s["plan_validity_rate"] == 1.0
        assert s["agreement_rate_over_accepted"] == 0.0
        assert s["disagreements"][0]["id"] == "A"
        assert s["rejection_counts"] == {}

    def test_agreement_denominator_is_accepted_not_total(self):
        """Two scenarios, one accepted-and-agreeing, one rejected. Agreement
        must be 1/1, not 1/2 — otherwise a planner that rejects everything
        and falls back would report rising agreement as it got worse."""
        scenarios = [
            scenario("A", "investment", KNOWN_USER),
            scenario("B", "budget", KNOWN_USER),
        ]
        planner = scripted_planner([
            as_json(STATIC_SEQUENCES["investment"]),
            "not json at all",
        ])
        s = run(scenarios, planner)["summary"]
        assert s["accepted"] == 1
        assert s["fallbacks"] == 1
        assert s["plan_validity_rate"] == 0.5
        assert s["agreement_rate_over_accepted"] == 1.0

    def test_fallback_steps_are_never_counted_as_agreement(self):
        """A fallback IS the static table by construction. If fallbacks were
        scored for agreement, the metric would read 100% precisely when the
        planner is contributing nothing."""
        scenarios = [scenario("A", "investment", KNOWN_USER)]
        planner = scripted_planner(["not json"])
        rows = run(scenarios, planner)["scenarios"]
        assert rows[0]["planner_steps"] == list(STATIC_SEQUENCES["investment"])
        assert rows[0]["agrees_with_static"] is False

    def test_rejection_counts_aggregate_by_code(self):
        scenarios = [
            scenario("A", "investment", KNOWN_USER),
            scenario("B", "investment", KNOWN_USER),
            scenario("C", "investment", KNOWN_USER),
        ]
        planner = scripted_planner([
            "not json",
            as_json(["InvestmentAgent"]),          # unsatisfied requires
            as_json(["NopeAgent"]),                # unknown agent
        ])
        s = run(scenarios, planner)["summary"]
        assert s["rejection_counts"]["malformed_json"] == 1
        assert s["rejection_counts"]["unsatisfied_requires"] == 1
        assert s["rejection_counts"]["unknown_agent"] == 1

    def test_agreement_rate_is_none_when_nothing_was_accepted(self):
        """Not 0.0 — zero would read as 'the planner disagreed', when in fact
        it never produced a plan to agree or disagree with."""
        scenarios = [scenario("A", "investment", KNOWN_USER)]
        s = run(scenarios, scripted_planner(["not json"]))["summary"]
        assert s["agreement_rate_over_accepted"] is None


class TestBaselineIsScoredToo:

    def test_static_table_gap_is_detected_when_context_is_empty(self):
        """The budget table entry runs BudgetAgent, which needs income and
        expenses. For a user with neither, the BASELINE is at fault, and the
        harness has to say so or the comparison is one-sided."""
        scenarios = [scenario("A", "budget", {"user_features": {}})]
        rows = run(scenarios, scripted_planner(["not json"]))["scenarios"]
        assert rows[0]["static_contextual_gaps"]
        codes = {g["code"] for g in rows[0]["static_contextual_gaps"]}
        assert "unsatisfied_requires" in codes

    def test_no_gap_when_context_is_complete(self):
        scenarios = [scenario("A", "budget", KNOWN_USER)]
        rows = run(scenarios, scripted_planner(["not json"]))["scenarios"]
        assert rows[0]["static_contextual_gaps"] == []


class TestCommittedScenarioSet:
    """The scenario list is evidence, not a fixture. These guard against it
    being quietly edited into something easier."""

    def test_every_routing_bucket_is_covered(self):
        covered = {sc["intent"] for sc in SCENARIOS}
        assert covered == set(STATIC_SEQUENCES), (
            f"uncovered routes: {set(STATIC_SEQUENCES) - covered}"
        )

    def test_scenario_ids_are_unique(self):
        ids = [sc["id"] for sc in SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_every_scenario_declares_what_it_probes(self):
        for sc in SCENARIOS:
            assert sc["probes"].strip(), f"{sc['id']} has no stated purpose"

    def test_includes_at_least_one_empty_context_scenario(self):
        """Without a new-user case the harness would only ever exercise the
        happy path, and the static arm would never show a contextual gap."""
        assert any(not sc["context"].get("user_features") for sc in SCENARIOS)

    @pytest.mark.parametrize("sid", [sc["id"] for sc in SCENARIOS])
    def test_scenario_runs_without_error(self, sid):
        sc = next(s for s in SCENARIOS if s["id"] == sid)
        out = run([sc], scripted_planner(["not json"]))
        assert out["summary"]["scenarios"] == 1
