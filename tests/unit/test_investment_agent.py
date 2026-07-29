"""
Phase 4 - InvestmentAgent evaluation, RQ2 metrics baseline.

3 test groups:
GROUP A: Ranking unit tests (no LLM, no API key)
    Tests the pure arithmatic of _normalise(), _horizon_fit_score(), and
    _rank_products(). mirrors the RiskProfilingAgent Group A design (Phase 3):
    (hybrid vs baseline recommendation comparison) depends on the ranking
    arithmethic being correct, independent of any LLM output.

GROUP B: Full run() pipeline tests (mock mode)
    Verifies the three-layer pipeline wires together correctly and produces
    a well-formed AgentResult, including the missing-risk_class guard and
    the FinancialConstraints post-hoc validation.

GROUP C: Evaluation - NDCG@3 / Precision@3 baseline on 5 fixed queries
    Each query is a (risk_class, investment_horizon) pair with a hand-labelled
    set of "known correct products" (graded relevance), assigned by applying
    the same horizon-fit-plus-cost-plus-return reasoning that _rank_products()
    implements, following the RiskProfilingAgent hand-labelled fixture pattern.
    This is a PURE InvestmentAgent baseline: no ExplainabilityAgent, no
    Orchestrator involved yet (those arrive in later phases) - it isolates ranking
    quality as its own measurable quantity. Writes result to results/rq2_investment_baseline.json.

RUNNING:
    python -m pytest tests/unit/test_investment_agent.py -v
    python -m pytest tests/unit/test_investment_agent.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.investment_agent import InvestmentAgent
from evaluation.metrics import EvalResult, ndcg_at_k, precision_at_k
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Helpers
def make_agent() -> InvestmentAgent:
    """Investment agent in mock mode — ranking logic requires no API key."""
    client = LLMClient()
    return InvestmentAgent(client)

# GROUP A: Ranking unit tests
class TestNormalise:

    def test_midpoint_gives_half(self):
        agent = make_agent()
        assert agent._normalise(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_low_bound_gives_zero(self):
        agent = make_agent()
        assert agent._normalise(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_high_bound_gives_one(self):
        agent = make_agent()
        assert agent._normalise(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_zero_range_returns_neutral_half(self):
        agent = make_agent()
        assert agent._normalise(5.0, 3.0, 3.0) == 0.5

    def test_clips_outside_range(self):
        agent = make_agent()
        assert agent._normalise(-5.0, 0.0, 10.0) == 0.0
        assert agent._normalise(15.0, 0.0, 10.0) == 1.0

class TestHorizonFitScore:

    def test_exact_match_gives_one(self):
        agent = make_agent()
        assert agent._horizon_fit_score(5, 5) == 1.0

    def test_ten_year_gap_gives_zero(self):
        agent = make_agent()
        assert agent._horizon_fit_score(0, 10) == 0.0

    def test_beyond_ten_year_gap_clips_to_zero(self):
        agent = make_agent()
        assert agent._horizon_fit_score(1, 20) == 0.0

    def test_partial_gap_gives_partial_score(self):
        agent = make_agent()
        assert agent._horizon_fit_score(5, 7) == pytest.approx(0.8)

class TestRankProducts:

    def test_empty_input_returns_empty_list(self):
        agent = make_agent()
        assert agent._rank_products([], {}) == []

    def test_every_product_gets_a_score(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        ranked = agent._rank_products(filtered, {"user_features": {"investment_horizon": 8}})
        assert all("score" in p for p in ranked)
        assert all(0.0 <= p["score"] <= 1.0 for p in ranked)

    def test_sorted_descending_by_score(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        ranked = agent._rank_products(filtered, {"user_features": {"investment_horizon": 8}})
        scores = [p["score"] for p in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_horizon_match_affects_ranking(self):
        """A product whose typical_horizon_years exactly matches the user's
        stated horizon should rank at least as high as an otherwise-similar
        product with a large horizon mismatch, all else being closer."""
        agent = make_agent()
        filtered = agent._filter_by_risk_class("conservative")
        # User horizon of 1 year should favour SAV001/TDP001/MMK001 (h=1)
        # over GOV001 (h=3), assuming comparable return/cost profiles.
        ranked = agent._rank_products(filtered, {"user_features": {"investment_horizon": 1}})
        top_ids = [p["product_id"] for p in ranked[:3]]
        assert "GOV001" not in top_ids or ranked[0]["product_id"] != "GOV001"

    def test_missing_user_features_defaults_horizon_to_five(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        ranked = agent._rank_products(filtered, {})
        assert ranked[0]["score_breakdown"]["user_investment_horizon"] == 5

    def test_score_breakdown_components_present(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        ranked = agent._rank_products(filtered, {"user_features": {"investment_horizon": 8}})
        for p in ranked:
            breakdown = p["score_breakdown"]
            assert "return_score" in breakdown
            assert "cost_score" in breakdown
            assert "horizon_fit_score" in breakdown

# GROUP B: Full run() pipeline tests
class TestRunPipeline:

    def test_missing_risk_class_returns_incomplete(self):
        agent = make_agent()
        result = agent.run({"user_features": {"investment_horizon": 10}})
        assert result.success is False
        assert result.payload["status"] == "incomplete"

    def test_complete_run_returns_success(self):
        agent = make_agent()
        result = agent.run({
            "risk_class": "moderate",
            "user_features": {"investment_horizon": 8},
        })
        assert result.success is True
        assert result.payload["status"] == "complete"

    def test_shortlist_respects_top_k(self):
        agent = make_agent()
        from config.settings import settings
        result = agent.run({
            "risk_class": "moderate",
            "user_features": {"investment_horizon": 8},
        })
        assert len(result.payload["shortlist"]) <= settings.investment.top_k

    def test_shortlist_only_contains_suitable_categories(self):
        agent = make_agent()
        from config.constraints import financial_constraints
        result = agent.run({
            "risk_class": "conservative",
            "user_features": {"investment_horizon": 1},
        })
        allowed = financial_constraints.RISK_PRODUCT_ALLOW["conservative"]
        for product in result.payload["shortlist"]:
            assert product["category"] in allowed

    def test_result_has_required_payload_keys(self):
        agent = make_agent()
        result = agent.run({
            "risk_class": "moderate",
            "user_features": {"investment_horizon": 8},
        })
        for key in ("status", "risk_class", "shortlist", "synthesis",
                    "deliverable", "constraint_violations"):
            assert key in result.payload, f"Missing key: {key}"

    def test_routing_context_has_top_product(self):
        agent = make_agent()
        result = agent.run({
            "risk_class": "moderate",
            "user_features": {"investment_horizon": 8},
        })
        assert "top_product_id" in result.routing_context

    def test_step_record_structure(self):
        agent = make_agent()
        result = agent.run({
            "risk_class": "moderate",
            "user_features": {"investment_horizon": 8},
        })
        step = result.to_step_record()
        assert step["agent"] == "InvestmentAgent"

# GROUP C: RQ2 evaluation on 5 fixed queries
RQ2_FIXTURE: list[dict] = [
    {
        "query": "conservative_horizon_1",
        "risk_class": "conservative",
        "investment_horizon": 1,
        "relevance_scores": {
            "SAV001": 1, "TDP001": 2, "MMK001": 2, "GOV001": 0,
        },
    },
    {
        "query": "moderately_conservative_horizon_5",
        "risk_class": "moderately_conservative",
        "investment_horizon": 5,
        "relevance_scores": {
            "SAV001": 0, "GOV001": 1, "CBI001": 2, "MXL001": 2,
        },
    },
    {
        "query": "moderate_horizon_8",
        "risk_class": "moderate",
        "investment_horizon": 8,
        "relevance_scores": {
            "GOV001": 0, "CB001": 1, "MXM001": 2, "ETB001": 1, "REIT001": 2,
        },
    },
    {
        "query": "moderately_aggressive_horizon_10",
        "risk_class": "moderately_aggressive",
        "investment_horizon": 10,
        "relevance_scores": {
            "CB001": 0, "ETB001": 2, "REIT001": 1, "EQF001": 1, "ETS001": 2,
        },
    },
    {
        "query": "aggressive_horizon_20",
        "risk_class": "aggressive",
        "investment_horizon": 20,
        "relevance_scores": {
            "EQF001": 0, "ETS001": 1, "EQI001": 2, "VEN001": 2,
        },
    },
]

@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests

class TestRQ2Evaluation:

    def test_precision_metric_runs(self):
        result = precision_at_k(["A", "B", "C"], {"A", "C"}, k=3)
        assert isinstance(result, EvalResult)
        assert result.value == pytest.approx(2 / 3, abs=1e-3)

    def test_ndcg_metric_runs(self):
        result = ndcg_at_k(["A", "B"], {"A": 2, "B": 1}, k=2)
        assert isinstance(result, EvalResult)
        assert result.value == pytest.approx(1.0)

    def test_ndcg_penalises_wrong_order(self):
        ideal = ndcg_at_k(["A", "B"], {"A": 2, "B": 1}, k=2)
        reversed_order = ndcg_at_k(["B", "A"], {"A": 2, "B": 1}, k=2)
        assert reversed_order.value < ideal.value

    def test_rq2_full_fixture_evaluation_and_write_results(self):
        """
        Full RQ2 evaluation on 5 fixed (risk_class, horizon) queries.
        Ranking is deterministic (no LLM involved) so results are
        meaningful in both mock and real mode.
        Writes: results/rq2_investment_baseline.json
        """
        agent = make_agent()

        per_query_results = []
        precision_values = []
        ndcg_values = []

        for item in RQ2_FIXTURE:
            filtered = agent._filter_by_risk_class(item["risk_class"])
            ranked = agent._rank_products(
                filtered,
                {"user_features": {"investment_horizon": item["investment_horizon"]}},
            )
            ranked_ids = [p["product_id"] for p in ranked]
            relevance_scores = item["relevance_scores"]
            relevant_ids = {
                pid for pid, score in relevance_scores.items() if score > 0
            }

            precision_result = precision_at_k(ranked_ids, relevant_ids, k=3)
            ndcg_result = ndcg_at_k(ranked_ids, relevance_scores, k=3)

            precision_values.append(precision_result.value)
            ndcg_values.append(ndcg_result.value)

            per_query_results.append({
                "query": item["query"],
                "risk_class": item["risk_class"],
                "investment_horizon": item["investment_horizon"],
                "ranked_ids_full": ranked_ids,
                "precision_at_3": precision_result.to_dict(),
                "ndcg_at_3": ndcg_result.to_dict(),
            })

        mean_precision = sum(precision_values) / len(precision_values)
        mean_ndcg = sum(ndcg_values) / len(ndcg_values)

        # Write results for dissertation evidence — RQ2 pre-explainability,
        # pre-orchestration InvestmentAgent baseline.
        RESULTS_DIR.mkdir(exist_ok=True)
        results_payload = {
            "phase": 4,
            "agent": "InvestmentAgent",
            "dataset": "hand_labelled_fixture_5_queries",
            "note": (
                "Pure InvestmentAgent baseline (filter + rank layers only, "
                "no LLM synthesis, no ExplainabilityAgent, no Orchestrator). "
                "RQ2 pre-explainability / pre-orchestration reference point."
            ),
            "metrics": {
                "mean_precision_at_3": round(mean_precision, 4),
                "mean_ndcg_at_3": round(mean_ndcg, 4),
            },
            "per_query": per_query_results,
        }
        results_path = write_results(results_payload, "rq2_investment_baseline.json")

        print(f"\n[Phase 4 RQ2] Mean Precision@3: {mean_precision:.3f}")
        print(f"[Phase 4 RQ2] Mean NDCG@3:      {mean_ndcg:.3f}")
        print(f"[Phase 4 RQ2] Results written to {results_path}")

        assert 0.0 <= mean_precision <= 1.0
        assert 0.0 <= mean_ndcg <= 1.0
