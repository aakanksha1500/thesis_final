"""
Phase 5 - BudgetAgent tests

Three test groups:
GROUP A: Cashflow arthmetic (no LLM, no API key)
    Tests _compute_cashflow() in complete isolation.
    Verifies: disposable = income - expenses, savings rate formula,
    expense fractions, edge cases (zero income, empty expenses).
    
GROUP B: Benchmark comparison logic (no LLM)
    Tests _compare_to_benchmarks() against the Ireland HBS benchmark
    Verifies: above/below/inline classification, tolerance band, gap_pct_points
    direction, unkown categories.
    
GROUP C: Full run() integration (mock LLM)
    Tests the full agent pipeline: input validation, cashflow + benchmark
    computation wired together, AgentResult structure.
    Writes results/phase5_budget_baseline.json to verify the benchmark

RUNNING:
    pytest tests/unit/test_budget_agent.py -v
    pytest tests/unit/test_budget_agent.py -v -s 
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.base_agent import AgentResult
from agents.budget_agent import BudgetAgent
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Hepler
def make_agent() -> BudgetAgent:
    """BudgetAgent in mock mode — all arithmetic runs without API key."""
    client = LLMClient()
    return BudgetAgent(client)

PROFILE_STRETCHED = {
    "name": "stretched_household",
    "description": "High housing costs, below-average savings, moderate income",
    "monthly_income": 3500.0,
    "monthly_expenses": {
        "housing": 1400.0,   # 40% — well above 28% benchmark
        "food": 500.0,       # 14.3% — inline
        "transport": 350.0,  # 10% — below 13% benchmark
        "utilities": 220.0,  # 6.3% — inline
        "healthcare": 100.0, # 2.9% — below 5% benchmark
        "entertainment": 180.0,  # 5.1% — inline
        "other": 300.0,      # 8.6% — inline
    },
}

PROFILE_HEALTHY = {
    "name": "healthy_household",
    "description": "Balanced spending, good savings rate, above-average income",
    "monthly_income": 5500.0,
    "monthly_expenses": {
        "housing": 1400.0,   # 25.5% — inline (under 28%)
        "food": 700.0,       # 12.7% — inline
        "transport": 600.0,  # 10.9% — inline
        "utilities": 350.0,  # 6.4% — inline
        "healthcare": 250.0, # 4.5% — inline
        "entertainment": 300.0,  # 5.5% — inline
        "other": 400.0,      # 7.3% — inline
    },
}

PROFILE_LOW_INCOME = {
    "name": "low_income_household",
    "description": "Low income, stretched across essentials, negative savings",
    "monthly_income": 2200.0,
    "monthly_expenses": {
        "housing": 900.0,    # 40.9% — well above 28%
        "food": 400.0,       # 18.2% — above 14%
        "transport": 200.0,  # 9.1% — below 13%
        "utilities": 180.0,  # 8.2% — above 7%
        "healthcare": 80.0,  # 3.6% — below 5%
        "entertainment": 100.0,  # 4.5% — inline
        "other": 200.0,      # 9.1% — inline
    },
}

# GROUP A: Cashflow arithmetic
class TestComputeCashflow:

    def test_disposable_income_correct(self):
        agent = make_agent()
        result = agent._compute_cashflow(
            monthly_income=3000.0,
            monthly_expenses={"housing": 900.0, "food": 300.0},
        )
        assert result["disposable_income"] == pytest.approx(1800.0, abs=0.01)

    def test_total_expenses_correct(self):
        agent = make_agent()
        result = agent._compute_cashflow(
            monthly_income=3000.0,
            monthly_expenses={"housing": 900.0, "food": 300.0},
        )
        assert result["total_expenses"] == pytest.approx(1200.0, abs=0.01)

    def test_savings_rate_formula(self):
        agent = make_agent()
        result = agent._compute_cashflow(
            monthly_income=4000.0,
            monthly_expenses={"housing": 1200.0, "food": 400.0},
        )
        expected_rate = (4000.0 - 1600.0) / 4000.0 * 100
        assert result["savings_rate_pct"] == pytest.approx(expected_rate, abs=0.01)

    def test_expense_fractions_sum_to_total_fraction(self):
        agent = make_agent()
        expenses = {"housing": 1000.0, "food": 500.0, "transport": 300.0}
        result = agent._compute_cashflow(monthly_income=4000.0,
                                         monthly_expenses=expenses)
        total_fraction = sum(result["expense_fractions"].values())
        assert total_fraction == pytest.approx(1800.0 / 4000.0, abs=0.001)

    def test_negative_disposable_income_allowed(self):
        """Agent should compute correctly even when user is spending more than earning."""
        agent = make_agent()
        result = agent._compute_cashflow(
            monthly_income=2000.0,
            monthly_expenses={"housing": 1800.0, "food": 500.0},
        )
        assert result["disposable_income"] < 0
        assert result["savings_rate_pct"] < 0

    def test_zero_income_returns_error(self):
        agent = make_agent()
        result = agent._compute_cashflow(monthly_income=0.0, monthly_expenses={})
        assert "error" in result

    def test_empty_expenses_gives_full_disposable(self):
        agent = make_agent()
        result = agent._compute_cashflow(monthly_income=3000.0, monthly_expenses={})
        assert result["disposable_income"] == pytest.approx(3000.0, abs=0.01)
        assert result["savings_rate_pct"] == pytest.approx(100.0, abs=0.01)

    def test_all_values_rounded_to_2dp(self):
        agent = make_agent()
        result = agent._compute_cashflow(
            monthly_income=3333.33,
            monthly_expenses={"housing": 1111.11},
        )
        assert result["disposable_income"] == round(result["disposable_income"], 2)
        assert result["total_expenses"] == round(result["total_expenses"], 2)


# GROUP B: Benchmark comparison logic
class TestCompareTooBenchmarks:

    def test_above_benchmark_classified_correctly(self):
        agent = make_agent()
        # Housing benchmark is 0.28; 0.40 is well above
        fractions = {"housing": 0.40}
        result = agent._compare_to_benchmarks(fractions)
        assert result["housing"]["label"] == "above"

    def test_below_benchmark_classified_correctly(self):
        agent = make_agent()
        # Transport benchmark is 0.13; 0.05 is well below
        fractions = {"transport": 0.05}
        result = agent._compare_to_benchmarks(fractions)
        assert result["transport"]["label"] == "below"

    def test_inline_within_tolerance(self):
        agent = make_agent()
        # Food benchmark is 0.14; 0.15 is 7% above — within 10% tolerance
        fractions = {"food": 0.15}
        result = agent._compare_to_benchmarks(fractions)
        assert result["food"]["label"] == "inline"

    def test_gap_pct_points_direction(self):
        agent = make_agent()
        # Housing: user 0.40, benchmark 0.28 → gap = +12pp
        fractions = {"housing": 0.40}
        result = agent._compare_to_benchmarks(fractions)
        assert result["housing"]["gap_pct_points"] > 0

    def test_below_benchmark_gives_negative_gap(self):
        agent = make_agent()
        # Transport: user 0.05, benchmark 0.13 → gap = -8pp
        fractions = {"transport": 0.05}
        result = agent._compare_to_benchmarks(fractions)
        assert result["transport"]["gap_pct_points"] < 0

    def test_unknown_category_labelled_unknown(self):
        agent = make_agent()
        fractions = {"gym_membership": 0.03}
        result = agent._compare_to_benchmarks(fractions)
        assert result["gym_membership"]["label"] == "unknown"
        assert result["gym_membership"]["benchmark_fraction"] is None

    def test_all_known_categories_return_benchmark_fraction(self):
        agent = make_agent()
        from config.settings import settings
        fractions = {cat: 0.10 for cat in settings.budget.ireland_hbs_benchmarks}
        result = agent._compare_to_benchmarks(fractions)
        for cat in settings.budget.ireland_hbs_benchmarks:
            assert result[cat]["benchmark_fraction"] is not None

    def test_inline_tolerance_boundary_above(self):
        agent = make_agent()
        # Food benchmark 0.14; tolerance 10% → upper = 0.154
        # 0.155 should be 'above'
        fractions = {"food": 0.155}
        result = agent._compare_to_benchmarks(fractions)
        assert result["food"]["label"] == "above"

    def test_inline_tolerance_boundary_below(self):
        agent = make_agent()
        # Food benchmark 0.14; lower = 0.126
        # 0.125 should be 'below'
        fractions = {"food": 0.125}
        result = agent._compare_to_benchmarks(fractions)
        assert result["food"]["label"] == "below"

# Savings rate flag
class TestSavingsRateFlag:

    def test_below_threshold_returns_warning(self):
        agent = make_agent()
        flag = agent._savings_rate_flag(5.0)
        assert flag is not None
        assert "10%" in flag or "savings" in flag.lower()

    def test_above_threshold_returns_none(self):
        agent = make_agent()
        assert agent._savings_rate_flag(15.0) is None

    def test_exactly_at_threshold_returns_none(self):
        agent = make_agent()
        assert agent._savings_rate_flag(10.0) is None

    def test_negative_savings_rate_returns_warning(self):
        agent = make_agent()
        flag = agent._savings_rate_flag(-5.0)
        assert flag is not None

# GROUP C: Full run() integration + Phase 5 results file
class TestBudgetAgentRun:

    def test_missing_income_returns_incomplete(self):
        agent = make_agent()
        result = agent.run({"monthly_expenses": {"housing": 500.0}})
        assert result.success is False
        assert result.payload["status"] == "incomplete"

    def test_missing_expenses_returns_incomplete(self):
        agent = make_agent()
        result = agent.run({"monthly_income": 3000.0})
        assert result.success is False
        assert result.payload["status"] == "incomplete"

    def test_income_from_user_features(self):
        """Agent reads annual income from user_features and converts to monthly."""
        agent = make_agent()
        result = agent.run({
            "user_features": {"income": 48000},
            "monthly_expenses": {"housing": 1000.0, "food": 400.0},
        })
        assert result.success is True
        assert result.payload["monthly_income"] == pytest.approx(4000.0, abs=0.01)

    def test_result_is_agent_result(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 900.0, "food": 300.0},
        })
        assert isinstance(result, AgentResult)

    def test_successful_result_has_required_keys(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3500.0,
            "monthly_expenses": {"housing": 1000.0, "food": 500.0},
        })
        for key in (
            "status", "monthly_income", "monthly_expenses",
            "total_expenses", "disposable_income", "savings_rate_pct",
            "benchmark_comparison", "recommendations_text",
        ):
            assert key in result.payload, f"Missing key: {key}"

    def test_status_is_complete_on_success(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3500.0,
            "monthly_expenses": {"housing": 1000.0, "food": 500.0},
        })
        assert result.payload["status"] == "complete"

    def test_routing_context_has_disposable_income(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3500.0,
            "monthly_expenses": {"housing": 1000.0, "food": 500.0},
        })
        assert "disposable_income" in result.routing_context
        assert "savings_rate_pct" in result.routing_context
        assert "savings_rate_healthy" in result.routing_context

    def test_savings_rate_healthy_flag_true_when_good(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 5000.0,
            "monthly_expenses": {"housing": 1200.0, "food": 600.0},
        })
        # disposable = 3200, rate = 64% → healthy
        assert result.routing_context["savings_rate_healthy"] is True

    def test_savings_rate_healthy_flag_false_when_low(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 2000.0,
            "monthly_expenses": {"housing": 1900.0},
        })
        # disposable = 100, rate = 5% → below 10% threshold
        assert result.routing_context["savings_rate_healthy"] is False

    def test_agent_name(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 900.0},
        })
        assert result.agent_name == "BudgetAgent"

    def test_step_record_structure(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 900.0},
        })
        step = result.to_step_record()
        for key in ("step_id", "agent", "completed", "duration_ms"):
            assert key in step

    def test_phase5_evaluation_and_write_results(self):
        """
        Run BudgetAgent against three representative Irish household profiles.
        Records benchmark comparison results to results/phase5_budget_baseline.json
        as qualitative dissertation evidence.
        No accuracy metric — correctness is verified by reading the JSON output
        and confirming the benchmark labels match manual expectations.
        """
        agent = make_agent()
        profiles = [PROFILE_STRETCHED, PROFILE_HEALTHY, PROFILE_LOW_INCOME]
        profile_results = []

        for profile in profiles:
            result = agent.run({
                "monthly_income": profile["monthly_income"],
                "monthly_expenses": profile["monthly_expenses"],
            })
            assert result.success is True, (
                f"Profile '{profile['name']}' failed: {result.error}"
            )

            p = result.payload
            profile_results.append({
                "profile": profile["name"],
                "description": profile["description"],
                "monthly_income": p["monthly_income"],
                "total_expenses": p["total_expenses"],
                "disposable_income": p["disposable_income"],
                "savings_rate_pct": p["savings_rate_pct"],
                "savings_rate_healthy": result.routing_context["savings_rate_healthy"],
                "benchmark_comparison": p["benchmark_comparison"],
                "savings_rate_flag": p["savings_rate_flag"],
                "recommendations_text": p["recommendations_text"],
            })

            print(
                f"\n[Phase 5] {profile['name']}: "
                f"income=€{p['monthly_income']:.0f} "
                f"disposable=€{p['disposable_income']:.0f} "
                f"savings_rate={p['savings_rate_pct']:.1f}% "
                f"healthy={result.routing_context['savings_rate_healthy']}"
            )

        # Write results file
        results_path = write_results(
                {
                    "phase": 5,
                    "agent": "BudgetAgent",
                    "dataset": "Ireland HBS 2022-23 [D11] benchmarks",
                    "note": (
                        "No RQ accuracy metric. Results validate benchmark "
                        "comparison logic against three synthetic Irish household "
                        "profiles. Manual inspection of benchmark_comparison "
                        "confirms CSO HBS data is applied correctly."
                    ),
                    "profiles": profile_results,
                },
                "phase5_budget_baseline.json",
            )
        print(f"\n[Phase 5] Results written to {results_path}")

        # Structural assertions
        assert len(profile_results) == 3

        # Stretched household: housing should be flagged 'above'
        stretched = profile_results[0]
        assert stretched["benchmark_comparison"]["housing"]["label"] == "above"

        # Low income household: savings rate should be unhealthy
        low_income = profile_results[2]
        assert low_income["savings_rate_healthy"] is False
