"""
Phase 5 - BudgetAgent tests

Three test groups:
GROUP A: Cashflow arithmetic (no LLM, no API key)
    Tests _compute_cashflow() in complete isolation.
    Verifies: disposable = income - expenses, savings rate formula,
    expense fractions, edge cases (zero income, empty expenses).

GROUP B: Benchmark comparison logic (no LLM)
    Tests _compare_to_benchmarks() against the Ireland HBS benchmark
    Verifies: above/below/inline classification, tolerance band, gap_pct_points
    direction, unknown categories.

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
from unittest.mock import patch

import pytest

from agents.base_agent import AgentResult
from agents.budget_agent import BudgetAgent
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Helper
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

@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests

# GROUP C: Full run() integration + Phase 5 results file
class TestDeterministicFallbackNarrative:
    """
    The gap this closes: data_sufficiency and clarifying_questions were
    computed and attached to the payload (Sections 1 and 3) but never
    reached the actual LLM-generated narrative text a customer reads —
    they sat in a separate payload field a caller had to know to look
    for. Confirms the prompt itself, not just the payload, carries this.
    """
    def _cashflow(self):
        return {
            "total_expenses": 2000.0, "disposable_income": 1000.0,
            "savings_rate_pct": 25.0, "expense_fractions": {},
        }

    def test_fallback_includes_sufficiency_disclosure(self):
        from agents.data_sufficiency import assess_data_sufficiency
        agent = make_agent()
        sufficiency = assess_data_sufficiency(
            [{"date": "2025-06-05", "category": "housing", "amount": 1000.0}]
        )
        text = agent._deterministic_fallback_narrative(
            self._cashflow(), None, sufficiency_result=sufficiency,
        )
        assert sufficiency.disclosure in text

    def test_fallback_lists_unverified_categories(self):
        from agents.data_sufficiency import assess_data_sufficiency
        agent = make_agent()
        sufficiency = assess_data_sufficiency(
            [{"date": "2025-06-01", "category": "insurance", "amount": 500.0}],
            monthly_income=3000.0,
        )
        text = agent._deterministic_fallback_narrative(
            self._cashflow(), None, sufficiency_result=sufficiency,
        )
        assert "insurance" in text

    def test_fallback_includes_clarifying_questions(self):
        agent = make_agent()
        text = agent._deterministic_fallback_narrative(
            self._cashflow(), None,
            clarifying_questions={"insurance": "Is this monthly, quarterly, half-yearly, or annual?"},
        )
        assert "Is this monthly, quarterly, half-yearly, or annual?" in text

    def test_fallback_includes_questionnaire_disclosure(self):
        agent = make_agent()
        text = agent._deterministic_fallback_narrative(
            self._cashflow(), None,
            questionnaire_confidence={
                "source": "self_reported", "confidence_tier": "medium",
                "disclosure": "This budget is based on your own estimates, not transaction history.",
            },
        )
        assert "based on your own estimates" in text

    def test_fallback_uses_generic_text_when_nothing_else_given(self):
        """Backward compatible default: a fully-verified, high-confidence
        run with no disclosure to add still gets a sensible fallback,
        not an empty or broken one."""
        agent = make_agent()
        text = agent._deterministic_fallback_narrative(self._cashflow(), None)
        assert "may not reflect your full financial picture" in text

    def test_run_end_to_end_fallback_still_carries_clarifying_question(self):
        """Force the LLM call to fail entirely -- the resulting
        recommendations_text must still carry the insurance clarifying
        question, not just the bare cashflow numbers."""
        agent = make_agent()
        with patch.object(agent, "_call_llm", side_effect=Exception("rate limited")):
            result = agent.run({
                "monthly_income": 3000.0,
                "transactions": [
                    {"date": f"2025-{m:02d}-01", "category": "housing", "amount": 1000.0}
                    for m in range(1, 13)
                ] + [{"date": "2025-06-01", "category": "insurance", "amount": 600.0}],
            })
        assert result.payload["status"] == "complete"
        assert "monthly, quarterly, half-yearly, or annual" in result.payload["recommendations_text"]

    def _cashflow_and_benchmark(self):
        cashflow = {
            "total_expenses": 2000.0, "disposable_income": 1000.0,
            "savings_rate_pct": 25.0, "expense_fractions": {},
        }
        benchmark = {}
        return cashflow, benchmark

    def test_prompt_includes_disclosure_when_sufficiency_result_given(self):
        from agents.data_sufficiency import assess_data_sufficiency
        agent = make_agent()
        cashflow, benchmark = self._cashflow_and_benchmark()
        sufficiency = assess_data_sufficiency(
            [{"date": "2025-06-05", "category": "housing", "amount": 1000.0}]
        )
        prompt = agent._build_synthesis_prompt(
            cashflow, benchmark, 3000.0, {"housing": 1000.0},
            sufficiency_result=sufficiency,
        )
        assert "DATA CONFIDENCE" in prompt
        assert sufficiency.disclosure in prompt

    def test_prompt_lists_unverified_categories(self):
        from agents.data_sufficiency import assess_data_sufficiency
        agent = make_agent()
        cashflow, benchmark = self._cashflow_and_benchmark()
        sufficiency = assess_data_sufficiency(
            [{"date": "2025-06-01", "category": "insurance", "amount": 500.0}],
            monthly_income=3000.0,
        )
        prompt = agent._build_synthesis_prompt(
            cashflow, benchmark, 3000.0, {"insurance": 500.0},
            sufficiency_result=sufficiency,
        )
        assert "insurance" in prompt
        assert "Not yet verified" in prompt

    def test_prompt_includes_clarifying_questions_when_given(self):
        agent = make_agent()
        cashflow, benchmark = self._cashflow_and_benchmark()
        prompt = agent._build_synthesis_prompt(
            cashflow, benchmark, 3000.0, {"insurance": 500.0},
            clarifying_questions={"insurance": "Is this monthly, quarterly, half-yearly, or annual?"},
        )
        assert "OPEN QUESTIONS" in prompt
        assert "Is this monthly, quarterly, half-yearly, or annual?" in prompt

    def test_prompt_unaffected_when_neither_given(self):
        """Backward compatibility: the explicit-monthly_expenses path
        (no transactions) has neither -- prompt must be identical to
        before this feature existed."""
        agent = make_agent()
        cashflow, benchmark = self._cashflow_and_benchmark()
        prompt = agent._build_synthesis_prompt(cashflow, benchmark, 3000.0, {"housing": 1000.0})
        assert "DATA CONFIDENCE" not in prompt
        assert "OPEN QUESTIONS" not in prompt

    def test_below_benchmark_prompt_warns_against_reflexive_cutback_advice(self):
        """
        Real-mode run produced contradictory advice for a below-benchmark
        food category: 'align with the national average (14%)' followed
        immediately by 'reduce your food expenses further' -- reducing
        further moves AWAY from 14%, not toward it. Confirms the prompt
        now tells the LLM not to reflexively suggest cutting back on
        spending that's already below benchmark.
        """
        agent = make_agent()
        cashflow, _ = self._cashflow_and_benchmark()
        benchmark = {
            "food": {
                "user_fraction": 0.10, "benchmark_fraction": 0.14,
                "label": "below", "gap_pct_points": -4.0,
            },
        }
        prompt = agent._build_synthesis_prompt(cashflow, benchmark, 3000.0, {"food": 300.0})
        assert "healthy frugality" in prompt
        assert "Don't reflexively suggest cutting back" in prompt

    def test_run_end_to_end_passes_sufficiency_into_the_prompt(self):
        """Full run() with mock LLM -- spy on _call_llm to confirm the
        actual prompt sent includes the disclosure, not just that the
        payload has the field."""
        agent = make_agent()
        with patch.object(agent, "_call_llm", wraps=agent._call_llm) as spy:
            agent.run({
                "monthly_income": 3000.0,
                "transactions": [
                    {"date": "2025-06-05", "category": "housing", "amount": 1000.0},
                ],
            })
        sent_prompt = spy.call_args[0][0]
        assert "DATA CONFIDENCE" in sent_prompt

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


# GROUP D: transaction aggregation (Day 2a)

SAMPLE_TRANSACTIONS = [
    {"date": "2025-01-05", "category": "housing", "amount": 1000.0},
    {"date": "2025-02-05", "category": "housing", "amount": 1000.0},
    {"date": "2025-03-05", "category": "housing", "amount": 1000.0},
    {"date": "2025-01-10", "category": "food", "amount": 300.0},
    {"date": "2025-02-10", "category": "food", "amount": 320.0},
    {"date": "2025-03-10", "category": "food", "amount": 310.0},
    {"date": "2025-02-01", "category": "insurance", "amount": 1200.0},  # one lump, month 2 only
]


class TestAggregateTransactions:

    def test_divisor_is_actual_history_not_requested_window_when_shorter(self):
        """
        The bug: a customer with ONE real month of data, requesting a
        12-month window (e.g. via the min_aggregation_window_months
        floor), used to get that one month's total divided by 12 —
        manufacturing 11 fictional zero-spending months and understating
        true costs twelvefold. Found via a real session run producing a
        95.8% savings rate for a customer whose true rate was ~50%.
        """
        agent = make_agent()
        one_month_only = [
            {"date": "2025-06-05", "category": "housing", "amount": 1200.0},
        ]
        result = agent._aggregate_transactions(one_month_only, months=12)
        assert result["monthly_expenses"]["housing"] == pytest.approx(1200.0)
        assert result["effective_months"] == 1
        assert result["window_months"] == 12  # requested window is still reported

    def test_effective_months_equals_requested_when_history_is_long_enough(self):
        """The Day 2a customers (full 12 months of real data) must be
        completely unaffected — effective_months == window_months
        whenever real history covers the full requested window."""
        agent = make_agent()
        twelve_months = [
            {"date": f"2025-{m:02d}-05", "category": "housing", "amount": 1000.0}
            for m in range(1, 13)
        ]
        result = agent._aggregate_transactions(twelve_months, months=12, as_of="2025-12-31")
        assert result["effective_months"] == 12
        assert result["monthly_expenses"]["housing"] == pytest.approx(1000.0)

    def test_partial_history_between_the_two_extremes(self):
        """3 real months, 12-month window requested -- divisor should be
        3, matching exactly what run() now asserts end to end."""
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=12)
        assert result["effective_months"] == 3
        assert result["monthly_expenses"]["housing"] == pytest.approx(1000.0)


    def test_empty_transactions_returns_empty_expenses(self):
        agent = make_agent()
        result = agent._aggregate_transactions([], months=3)
        assert result["monthly_expenses"] == {}
        assert result["transaction_count"] == 0

    def test_averages_over_the_window_not_just_sums(self):
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=3, as_of="2025-03-31")
        # housing: 1000+1000+1000 over 3 months = 1000/month
        assert result["monthly_expenses"]["housing"] == pytest.approx(1000.0)

    def test_one_month_window_misses_a_lump_outside_it(self):
        """
        The Day 2a finding, in miniature: insurance only appears in
        February. A window that doesn't include February sees zero.
        """
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=1, as_of="2025-03-31")
        assert "insurance" not in result["monthly_expenses"]

    def test_wider_window_sees_the_lump_averaged_down(self):
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=3, as_of="2025-03-31")
        # 1200 lump spread over 3 months = 400/month — present, but diluted
        assert result["monthly_expenses"]["insurance"] == pytest.approx(400.0)

    def test_defaults_as_of_to_latest_transaction_date(self):
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=1)
        # Latest transaction in SAMPLE_TRANSACTIONS is 2025-03-10 (food) —
        # as_of uses that exact date, not padded to month-end, since
        # padding would imply data we don't actually have.
        assert result["window_end"] == "2025-03-10"
        assert result["window_start"] == "2025-03-01"

    def test_window_uses_calendar_months_not_rolling_30_days(self):
        """
        months=1 as_of 2025-03-15 should be the whole of March
        (2025-03-01 through 2025-03-15), not "the last 30 days" — a
        transaction on 2025-02-20 is outside it either way, but this
        pins down which rule is actually being used.
        """
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=1, as_of="2025-03-15")
        assert result["window_start"] == "2025-03-01"
        assert result["window_end"] == "2025-03-15"

    def test_transaction_count_reflects_window_not_full_list(self):
        agent = make_agent()
        result = agent._aggregate_transactions(SAMPLE_TRANSACTIONS, months=1, as_of="2025-03-31")
        assert result["transaction_count"] == 2  # one housing + one food in March
        assert result["total_transaction_count"] == len(SAMPLE_TRANSACTIONS)


class TestTransactionBackedRun:

    def test_run_derives_monthly_expenses_from_transactions(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": SAMPLE_TRANSACTIONS,
            "aggregation_window_months": 12,
        })
        assert result.success is True
        # 3 months of €1000 housing = €3000 total / 3 REAL months = €1000/month.
        # This IS correct, not a bug: SAMPLE_TRANSACTIONS only has 3 months of
        # data, so a genuine 12-month average is diluted by the 9 months with
        # no recorded spending — exactly the "measurable gap" the window
        # arithmetic is supposed to expose, not paper over.
        assert result.payload["monthly_expenses"]["housing"] == pytest.approx(1000.0)
        assert result.payload["transaction_aggregation"]["window_months"] == 12
        assert result.payload["transaction_aggregation"]["effective_months"] == 3

    def test_explicit_monthly_expenses_takes_priority_over_transactions(self):
        """
        A directly-provided monthly_expenses dict is assumed deliberate
        (e.g. a test fixture, or an upstream system that already computed
        it) — transactions must not silently override it.
        """
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 500.0},
            "transactions": SAMPLE_TRANSACTIONS,
        })
        assert result.payload["monthly_expenses"] == {"housing": 500.0}
        assert result.payload["transaction_aggregation"] is None

    def test_default_window_comes_from_settings(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": SAMPLE_TRANSACTIONS,
        })
        from config.settings import settings
        assert (
            result.payload["transaction_aggregation"]["window_months"]
            == settings.budget.default_aggregation_window_months
        )


# GROUP E: periodicity ambiguity integration (Section 3)

class TestPeriodicityIntegration:
    """
    agents.periodicity_inference wired into BudgetAgent — a single
    material insurance payment observed in the window should surface a
    clarifying question, without changing monthly_expenses itself (the
    figure is still reported; it's just flagged as uncertain).
    """

    def _transactions_with_one_insurance_payment(self):
        return [
            {"date": "2025-01-05", "category": "housing", "amount": 1000.0},
            {"date": "2025-02-05", "category": "housing", "amount": 1000.0},
            {"date": "2025-03-05", "category": "housing", "amount": 1000.0},
            {"date": "2025-02-01", "category": "insurance", "amount": 600.0},  # material vs 3000 income
        ]

    def test_aggregate_transactions_flags_single_material_insurance_payment(self):
        agent = make_agent()
        result = agent._aggregate_transactions(
            self._transactions_with_one_insurance_payment(),
            months=3, as_of="2025-03-31", monthly_income=3000.0,
        )
        assert "insurance" in result["periodicity_flags"]
        assert "insurance" in result["clarifying_questions"]
        # the aggregated figure is still reported, just flagged as uncertain
        assert "insurance" in result["monthly_expenses"]

    def test_no_periodicity_check_without_monthly_income(self):
        """monthly_income=None must skip the check entirely, not crash."""
        agent = make_agent()
        result = agent._aggregate_transactions(
            self._transactions_with_one_insurance_payment(),
            months=3, as_of="2025-03-31",
        )
        assert result["periodicity_flags"] == {}
        assert result["clarifying_questions"] == {}

    def test_non_ambiguous_categories_never_flagged(self):
        """3 housing payments, perfectly monthly and unambiguous by
        category anyway — must never appear in periodicity_flags."""
        agent = make_agent()
        result = agent._aggregate_transactions(
            self._transactions_with_one_insurance_payment(),
            months=3, as_of="2025-03-31", monthly_income=3000.0,
        )
        assert "housing" not in result["periodicity_flags"]

    def test_run_surfaces_clarifying_questions_at_top_level(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": self._transactions_with_one_insurance_payment(),
            "aggregation_window_months": 3,  # clamped to 12, but still runs
        })
        assert "insurance" in result.payload["clarifying_questions"]

    def test_run_has_no_clarifying_questions_when_explicit_expenses_given(self):
        """The transactions path is what triggers periodicity checking —
        a directly-provided monthly_expenses dict has no transaction-level
        detail to check, so this must be empty, not crash."""
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 1000.0, "insurance": 600.0},
        })
        assert result.payload["clarifying_questions"] == {}

    # GROUP F: data sufficiency integration (Section 1)

class TestDataSufficiencyIntegration:

    def test_minimal_tier_still_produces_a_budget_not_a_short_circuit(self):
        """
        Only genuinely empty history hits "insufficient" (available_months
        can't go below 1 for any non-empty list under month-granularity
        coverage — see agents/data_sufficiency.py). A single month of
        real activity is "minimal" tier, not "insufficient": run() should
        still produce a budget, just with low confidence attached, not
        refuse outright.
        """
        agent = make_agent()
        single_month_txns = [
            {"date": "2025-06-05", "category": "housing", "amount": 1000.0},
            {"date": "2025-06-20", "category": "food", "amount": 200.0},
        ]
        result = agent.run({"monthly_income": 3000.0, "transactions": single_month_txns})
        assert result.payload["status"] == "complete"
        assert result.payload["data_sufficiency"]["coverage_tier"] == "minimal"

    def test_empty_transactions_list_is_insufficient(self):
        agent = make_agent()
        result = agent.run({"monthly_income": 3000.0, "transactions": []})
        assert result.payload["status"] == "insufficient_history"
        assert "data_sufficiency" in result.payload
        assert result.payload["data_sufficiency"]["coverage_tier"] == "insufficient"

    def test_complete_run_attaches_data_sufficiency_to_payload(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": SAMPLE_TRANSACTIONS,  # spans Jan-Mar = "low" tier
        })
        assert result.payload["status"] == "complete"
        assert result.payload["data_sufficiency"]["coverage_tier"] == "low"

    def test_explicit_monthly_expenses_has_no_data_sufficiency(self):
        """No transactions -> nothing to assess coverage/density against."""
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "monthly_expenses": {"housing": 1000.0},
        })
        assert result.payload["data_sufficiency"] is None

    def test_dormant_long_window_customer_still_flagged_low_confidence(self):
        """12-month span but sparse activity -- coverage tier says
        'high' but density brings overall confidence down; this must
        survive being surfaced all the way through run()'s payload."""
        agent = make_agent()
        sparse_but_long = [
            {"date": "2024-07-05", "category": "housing", "amount": 900.0},
            {"date": "2024-11-05", "category": "housing", "amount": 900.0},
            {"date": "2025-06-05", "category": "housing", "amount": 900.0},
        ]
        result = agent.run({"monthly_income": 3000.0, "transactions": sparse_but_long})
        suff = result.payload["data_sufficiency"]
        assert suff["coverage_tier"] == "high"
        assert suff["confidence_score"] < 0.3

    # GROUP G: questionnaire integration (Section 2)

_SUFFICIENT_QUESTIONNAIRE = {
    "income": 3000.0, "housing_cost": 1200.0, "rough_monthly_leftover": 500.0,
    "food_spend": 400.0, "utilities_spend": 150.0, "discretionary_spend": 200.0,
}
_PARTIAL_QUESTIONNAIRE = {"income": 3000.0, "housing_cost": 1200.0}


class TestQuestionnaireIntegration:

    def test_no_transactions_but_sufficient_questionnaire_produces_a_budget(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "questionnaire_answers": _SUFFICIENT_QUESTIONNAIRE,
        })
        assert result.payload["status"] == "complete"
        assert result.payload["monthly_expenses"]["housing"] == 1200.0
        assert result.payload["monthly_expenses"]["food"] == 400.0

    def test_questionnaire_confidence_capped_at_medium(self):
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "questionnaire_answers": _SUFFICIENT_QUESTIONNAIRE,
        })
        assert result.payload["questionnaire_confidence"]["confidence_tier"] == "medium"

    def test_partial_questionnaire_returns_insufficient_with_next_question(self):
        """Mandatory not fully answered (missing rough_monthly_leftover)
        -- must report insufficient_history AND say what to ask next,
        not just dead-end."""
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "questionnaire_answers": _PARTIAL_QUESTIONNAIRE,
        })
        assert result.payload["status"] == "insufficient_history"
        assert result.payload["next_questionnaire_question"]["slot_name"] == "rough_monthly_leftover"

    def test_empty_transactions_with_sufficient_questionnaire_uses_questionnaire(self):
        """A brand-new customer (transactions=[]) who's answered the
        questionnaire should get a budget from it, not just
        insufficient_history — the questionnaire is exactly the
        fallback for this case."""
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": [],
            "questionnaire_answers": _SUFFICIENT_QUESTIONNAIRE,
        })
        assert result.payload["status"] == "complete"

    def test_neither_transactions_nor_questionnaire_falls_through_to_incomplete(self):
        """Confirms the restructure didn't break the original plain
        caller-error path — no signal attempted at all still reports the
        original generic 'incomplete', not insufficient_history."""
        agent = make_agent()
        result = agent.run({"monthly_income": 3000.0})
        assert result.payload["status"] == "incomplete"

    def test_verified_transaction_category_overrides_questionnaire_answer(self):
        """A customer answers the questionnaire, but also has SOME real
        transaction data that happens to verify a category (e.g. 2
        months of consistent housing payments) -- the verified real
        figure should win for that category, self-report elsewhere."""
        agent = make_agent()
        two_months_housing = [
            {"date": "2025-05-05", "category": "housing", "amount": 1150.0},
            {"date": "2025-06-05", "category": "housing", "amount": 1150.0},
        ]
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": two_months_housing,
            "questionnaire_answers": _SUFFICIENT_QUESTIONNAIRE,
            "aggregation_window_months": 2,
        })
        # housing was self-reported as 1200 but transaction-verified at 1150
        assert result.payload["monthly_expenses"]["housing"] == pytest.approx(1150.0)
        # food has no transaction data at all -- self-report survives
        assert result.payload["monthly_expenses"]["food"] == 400.0

    def test_synthesis_prompt_includes_questionnaire_disclosure(self):
        agent = make_agent()
        with patch.object(agent, "_call_llm", wraps=agent._call_llm) as spy:
            agent.run({
                "monthly_income": 3000.0,
                "questionnaire_answers": _SUFFICIENT_QUESTIONNAIRE,
            })
        sent_prompt = spy.call_args[0][0]
        assert "self-reported" in sent_prompt
        assert "medium" in sent_prompt

    def test_missing_transactions_key_vs_explicit_empty_list_both_handled(self):
        """
        Regression guard: context.get("transactions") is falsy for [],
        so an earlier version of this check (truthiness, not key
        presence) silently skipped the sufficiency path entirely for an
        explicitly-empty list — exactly what TransactionStore.lookup()
        returns for a genuinely new/unknown customer — and fell through
        to the generic "incomplete" status instead of the more specific,
        actionable "insufficient_history" one.
        """
        agent = make_agent()

        no_key = agent.run({"monthly_income": 3000.0})
        assert no_key.payload["status"] == "incomplete"

        empty_list = agent.run({"monthly_income": 3000.0, "transactions": []})
        assert empty_list.payload["status"] == "insufficient_history"
        assert empty_list.payload["data_sufficiency"]["coverage_tier"] == "insufficient"

    def test_requested_window_below_minimum_is_clamped_up(self):
        """
        A proper budget plan needs at least a year of history — a request
        for a shorter window in run() doesn't get honoured silently.
        """
        agent = make_agent()
        result = agent.run({
            "monthly_income": 3000.0,
            "transactions": SAMPLE_TRANSACTIONS,
            "aggregation_window_months": 1,
        })
        from config.settings import settings
        assert (
            result.payload["transaction_aggregation"]["window_months"]
            == settings.budget.min_aggregation_window_months
        )


@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests
class TestTransactionWindowEvaluation:
    """
    Day 2a's actual deliverable: report expense estimates at 1, 3, and
    12-month windows against the 12-month figure, across the 4 scenario
    customers from scripts/generate_transactions.py. No LLM involved —
    this is pure aggregation arithmetic over synthetic data, so it costs
    nothing to run regardless of LLM_PROVIDER/EVAL_LIVE_API.
    """

    def test_window_comparison_and_write_results(self):
        from scripts.generate_transactions import generate_all

        agent = make_agent()
        data = generate_all(seed=42)
        windows = (1, 3, 12)
        customer_results = []

        for customer_id, record in data["customers"].items():
            txns = record["transactions"]
            by_window = {
                w: agent._aggregate_transactions(txns, months=w)["monthly_expenses"]
                for w in windows
            }
            ground_truth = by_window[12]

            per_category_error = {}
            for category, truth_value in ground_truth.items():
                per_category_error[category] = {}
                for w in (1, 3):
                    est = by_window[w].get(category, 0.0)
                    pct_error = (
                        abs(est - truth_value) / truth_value * 100
                        if truth_value else None
                    )
                    per_category_error[category][f"{w}m_estimate"] = est
                    per_category_error[category][f"{w}m_pct_error_vs_12m"] = (
                        round(pct_error, 1) if pct_error is not None else None
                    )
                per_category_error[category]["12m_ground_truth"] = truth_value

            customer_results.append({
                "customer_id": customer_id,
                "scenario": record["scenario"],
                "narrative": record["narrative"],
                "monthly_income": record["monthly_income"],
                "windows": by_window,
                "per_category_error": per_category_error,
            })

            print(f"\n[Transaction window] {customer_id} ({record['scenario']}):")
            for category, err in per_category_error.items():
                print(
                    f"    {category}: 12m=€{err['12m_ground_truth']:.2f}  "
                    f"1m_err={err['1m_pct_error_vs_12m']}%  "
                    f"3m_err={err['3m_pct_error_vs_12m']}%"
                )

        results_path = write_results(
            {
                "phase": 5,
                "agent": "BudgetAgent",
                "finding": (
                    "1-month aggregation windows either completely miss an "
                    "annual-lump category (insurance) or, if they happen to "
                    "catch it, wrongly extrapolate it as if it recurred "
                    "every window-length — never close to the true "
                    "annualised rate. Only a 12-month window sees it "
                    "correctly. The effect is scenario-dependent, not "
                    "universal — TXN_BENCHMARK's low-variance spending is "
                    "far more window-stable on non-lump categories, and "
                    "TXN_INCOME_CHANGE inverts the usual direction (there, "
                    "the SHORTER window is more accurate, because what "
                    "changed was income, not lumpiness)."
                ),
                "generator_seed": data["_meta"]["seed"],
                "hbs_anchoring_source": data["_meta"]["hbs_anchoring_source"],
                "customers": customer_results,
            },
            "phase5_transaction_window_comparison.json",
        )
        print(f"\n[Transaction window] Results written to {results_path}")

        # Structural + qualitative assertions
        assert len(customer_results) == 4

        lumpy = next(c for c in customer_results if c["customer_id"] == "TXN_LUMPY")
        assert "insurance" not in lumpy["windows"][1], (
            "TXN_LUMPY's insurance renewal should be invisible to a "
            "1-month window unless it happens to fall in that exact month"
        )
        lumpy_insurance_3m_error = lumpy["per_category_error"]["insurance"]["3m_pct_error_vs_12m"]
        assert lumpy_insurance_3m_error > 50.0, (
            f"Expected a large 3-month error on TXN_LUMPY's insurance "
            f"line — the renewal falls inside the trailing 3-month "
            f"window, so it gets wrongly extrapolated as if it recurs "
            f"every 3 months (~4x the true annualised rate), not "
            f"diluted toward it — got {lumpy_insurance_3m_error}%"
        )

        benchmark = next(c for c in customer_results if c["customer_id"] == "TXN_BENCHMARK")
        benchmark_food_1m_error = benchmark["per_category_error"]["food"]["1m_pct_error_vs_12m"]
        assert benchmark_food_1m_error < 40.0, (
            f"TXN_BENCHMARK is the low-variance control — expected a "
            f"modest 1-month food error, got {benchmark_food_1m_error}%"
        )