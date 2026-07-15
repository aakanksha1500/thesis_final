"""
Phase 5 - BudgetAgent

Responsibilities:
    1. Cashflow analysis - income minus expenditure, savings rate calculations.
    2. Benchmark comparison - each spending category classified as above / below / 
       inline versus the Ireland HouseholdB Budget survey
    3. LLM synthesis - plain-english recommendations grounded in the benchmarkgaps.

"""
from __future__ import annotations

import json
import time
from typing import Any

from agents.base_agent import BaseAgent, AgentResult
from config.prompts import BUDGET_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class BudgetAgent(BaseAgent):
    """
    Monthly cashflow analyser with CSO Ireland HBS benchmark comparison.
    """
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="BudgetAgent")
        self._benchmarks = settings.budget.ireland_hbs_benchmarks

    @property
    def system_prompt(self) -> str:
        return BUDGET_SYSTEM

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """Budget LLM output is plain-English — wrap directly."""
        return {"recommendations_text": raw.strip()}
    
    # Cashflow Computation
    def _compute_cashflow(
        self,
        monthly_income: float,
        monthly_expenses: dict[str, float],
    ) -> dict[str, float]:
       """
        Compute core cashflow metrics from income and expense inputs.

        Returns a dict with:
          total_expenses    — sum of all expense categories
          disposable_income — monthly_income - total_expenses
          savings_rate_pct  — disposable_income / monthly_income * 100
          expense_fractions — each category as fraction of income

        No LLM involved — pure arithmetic, fully testable.
        """ 
       if monthly_income <= 0:
            return {
                "total_expenses": 0.0,
                "disposable_income": 0.0,
                "savings_rate_pct": 0.0,
                "expense_fractions": {},
                "error": "monthly_income must be positive",
            }
       total_expenses = sum(monthly_expenses.values())
       disposable_income = monthly_income - total_expenses
       savings_rate_pct = (disposable_income / monthly_income) * 100
       
       expense_fractions = {
            category: round(amount / monthly_income, 4)
            for category, amount in monthly_expenses.items()
        }
       
       return {
            "total_expenses": round(total_expenses, 2),
            "disposable_income": round(disposable_income, 2),
            "savings_rate_pct": round(savings_rate_pct, 2),
            "expense_fractions": expense_fractions,
        }
    
    # Benchmark Comparison
    def _compare_to_benchmarks(
        self,
        expense_fractions: dict[str, float],
    ) -> dict[str, dict]:
        """
        Compare each spending category fraction to the Ireland HBS 2022-23
        national average benchmark [D11].

        Classification rule (settings.budget.inline_tolerance = 0.10):
          above   — user fraction > benchmark * (1 + tolerance)
          below   — user fraction < benchmark * (1 - tolerance)
          inline  — within ±10% of benchmark
          unknown — category not in HBS benchmarks (user-defined category)

        Returns a dict of:
          category -> {
            "user_fraction":      float,
            "benchmark_fraction": float | None,
            "label":              "above" | "below" | "inline" | "unknown",
            "gap_pct_points":     float  (user - benchmark, in percentage points)
          }

        The gap in percentage points is the key input for the LLM synthesis:
        "You spend X percentage points more than the Irish average on housing."
        """
        tolerance = settings.budget.inline_tolerance
        result = {}

        for category, user_fraction in expense_fractions.items():
            benchmark = self._benchmarks.get(category)

            if benchmark is None:
                result[category] = {
                    "user_fraction": round(user_fraction, 4),
                    "benchmark_fraction": None,
                    "label": "unknown",
                    "gap_pct_points": None,
                }
                continue

            gap = user_fraction - benchmark  # positive = spending more than average
            upper = benchmark * (1 + tolerance)
            lower = benchmark * (1 - tolerance)

            if user_fraction > upper:
                label = "above"
            elif user_fraction < lower:
                label = "below"
            else:
                label = "inline"

            result[category] = {
                "user_fraction": round(user_fraction, 4),
                "benchmark_fraction": round(benchmark, 4),
                "label": label,
                "gap_pct_points": round(gap * 100, 2),
            }

        return result

    def _savings_rate_flag(self, savings_rate_pct: float) -> str | None:
        """
        Return a warning string if savings rate is below the CBI-referenced
        minimum healthy threshold (settings.budget.min_healthy_savings_rate_pct).
        Returns None if savings rate is healthy.
        """
        threshold = settings.budget.min_healthy_savings_rate_pct
        if savings_rate_pct < threshold:
            return (
                f"Your current savings rate of {savings_rate_pct:.1f}% is below "
                f"the {threshold:.0f}% threshold generally recommended for "
                f"financial resilience. Addressing this before committing to "
                f"long-term investments is advisable."
            )
        return None
    
    # LLM synthesis prompt builder
    def _build_synthesis_prompt(
        self,
        cashflow: dict[str, float],
        benchmark_comparison: dict[str, dict],
        monthly_income: float,
        monthly_expenses: dict[str, float],
    ) -> str:
        """
        Build the structured prompt for BUDGET_SYSTEM synthesis.
        All inputs are deterministically computed — LLM only narrates them.
        """
        above_benchmark = {
            cat: data for cat, data in benchmark_comparison.items()
            if data["label"] == "above"
        }
        below_benchmark = {
            cat: data for cat, data in benchmark_comparison.items()
            if data["label"] == "below"
        }

        prompt_lines = [
            f"Monthly income: €{monthly_income:.2f}",
            f"Monthly expenses: {json.dumps(monthly_expenses, indent=2)}",
            f"Total expenses: €{cashflow['total_expenses']:.2f}",
            f"Disposable income: €{cashflow['disposable_income']:.2f}",
            f"Savings rate: {cashflow['savings_rate_pct']:.1f}%",
            "",
            "Benchmark comparison (Ireland HBS 2022-23):",
        ]

        for cat, data in benchmark_comparison.items():
            if data["benchmark_fraction"] is not None:
                prompt_lines.append(
                    f"  {cat}: {data['label'].upper()} benchmark "
                    f"(user={data['user_fraction']*100:.1f}% of income, "
                    f"benchmark={data['benchmark_fraction']*100:.1f}%, "
                    f"gap={data['gap_pct_points']:+.1f}pp)"
                )
            else:
                prompt_lines.append(
                    f"  {cat}: not in HBS benchmarks "
                    f"(user spends {data['user_fraction']*100:.1f}% of income)"
                )

        if above_benchmark:
            prompt_lines.append(
                f"\nCategories above benchmark: "
                f"{', '.join(above_benchmark.keys())}"
            )
        if below_benchmark:
            prompt_lines.append(
                f"Categories below benchmark: "
                f"{', '.join(below_benchmark.keys())}"
            )

        savings_flag = self._savings_rate_flag(cashflow["savings_rate_pct"])
        if savings_flag:
            prompt_lines.append(f"\nSAVINGS ALERT: {savings_flag}")

        prompt_lines.append(
            "\nWrite your cashflow summary and recommendations now, "
            "per your system instructions."
        )
        return "\n".join(prompt_lines)
    
    # Main entry point
    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Process one budget analysis request.

        context keys used:
          'monthly_income'    (float, required) — gross monthly income in euros
          'monthly_expenses'  (dict,  required) — category -> euros spent per month
          'user_features'     (dict,  optional) — reads income if monthly_income absent

        Returns AgentResult with payload:
          status, monthly_income, monthly_expenses, total_expenses,
          disposable_income, savings_rate_pct, benchmark_comparison,
          savings_rate_flag, recommendations_text
        """
        start_time = time.perf_counter()

        # Accept income from either direct key or user_features dict
        monthly_income: float = float(
            context.get("monthly_income")
            or (context.get("user_features") or {}).get("income", 0) / 12
        )
        monthly_expenses: dict = context.get("monthly_expenses", {})

        # Validate inputs
        if monthly_income <= 0:
            payload = {
                "status": "incomplete",
                "message": (
                    "Cannot analyse budget without monthly_income. "
                    "Provide 'monthly_income' in context or 'income' in user_features."
                ),
            }
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("[BudgetAgent] Missing monthly_income")
            return self._make_result(
                payload=payload,
                duration_ms=duration_ms,
                error="Missing monthly_income",
            )

        if not monthly_expenses:
            payload = {
                "status": "incomplete",
                "message": (
                    "Cannot analyse budget without monthly_expenses. "
                    "Provide a dict of category -> euros spent per month."
                ),
            }
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("[BudgetAgent] Missing monthly_expenses")
            return self._make_result(
                payload=payload,
                duration_ms=duration_ms,
                error="Missing monthly_expenses",
            )

        # Step 1 — Cashflow arithmetic (no LLM)
        cashflow = self._compute_cashflow(monthly_income, monthly_expenses)
        if "error" in cashflow:
            return self._make_result(
                payload={"status": "error", **cashflow},
                duration_ms=(time.perf_counter() - start_time) * 1000,
                error=cashflow["error"],
            )

        # Step 2 — Benchmark comparison (no LLM)
        benchmark_comparison = self._compare_to_benchmarks(
            cashflow["expense_fractions"]
        )

        # Step 3 — Savings rate flag (no LLM)
        savings_flag = self._savings_rate_flag(cashflow["savings_rate_pct"])

        # Step 4 — LLM synthesis
        prompt = self._build_synthesis_prompt(
            cashflow, benchmark_comparison, monthly_income, monthly_expenses
        )
        try:
            raw_text, tokens = self._call_llm(prompt)
            recommendations_text = raw_text.strip()
        except Exception as exc:
            logger.warning(
                f"[BudgetAgent] Synthesis failed: {exc} — using fallback"
            )
            recommendations_text = (
                f"Your monthly disposable income is "
                f"€{cashflow['disposable_income']:.2f} "
                f"({cashflow['savings_rate_pct']:.1f}% savings rate). "
                + (savings_flag or "Your savings rate appears healthy.")
                + " This analysis is based solely on the figures provided "
                "and may not reflect your full financial picture."
            )
            tokens = 0

        payload = {
            "status": "complete",
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "total_expenses": cashflow["total_expenses"],
            "disposable_income": cashflow["disposable_income"],
            "savings_rate_pct": cashflow["savings_rate_pct"],
            "expense_fractions": cashflow["expense_fractions"],
            "benchmark_comparison": benchmark_comparison,
            "savings_rate_flag": savings_flag,
            "recommendations_text": recommendations_text,
            "data_source": "Ireland Household Budget Survey 2022-23 [D11]",
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[BudgetAgent] income=€{monthly_income:.0f} "
            f"disposable=€{cashflow['disposable_income']:.0f} "
            f"savings_rate={cashflow['savings_rate_pct']:.1f}% "
            f"duration={duration_ms:.0f}ms"
        )

        return self._make_result(
            payload=payload,
            raw=recommendations_text,
            duration_ms=duration_ms,
            tokens=tokens,
            routing_context={
                "disposable_income": cashflow["disposable_income"],
                "savings_rate_pct": cashflow["savings_rate_pct"],
                "savings_rate_healthy": savings_flag is None,
            },
        )

