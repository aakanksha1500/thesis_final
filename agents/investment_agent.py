"""
Phase 4 - InvestmentAgent

Implements the context-aware hybrid recommendations model described in the proposed approach: 
rule-based constraints + feature-based scoring + LLM synthesis. This mirros
the RiskProfilingAgent's hybrid pattern from Phase 3.

Literature grounding:
  - Nguyen et al. [8]: rule-based constraints as a deterministic secondary
    layer alongside quantitative/ML scoring (E4) — implemented here as the
    filter-then-rank pipeline that runs before any LLM involvement.
  - Klesel & Wittmann [6]: grounding LLM output in retrieved/structured
    data rather than free generation — the shortlist IS the grounding.
  - Takayanagi et al. [7] / Li et al. [3]: calibrated trust (X3) — the
    synthesis prompt requires stated trade-offs and assumptions, not
    maximised confidence.
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import BaseAgent, AgentResult
from config.constraints import financial_constraints
from config.prompts import INVESTMENT_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

# Synthetic illustrative data for prototype development — NOT a live feed.
# Categories match config.constraints.FinancialConstraints.RISK_PRODUCT_ALLOW
# exactly, so every product is reachable by at least one risk tier and the
# filter layer has a real rule set to operate against.

IRISH_PRODUCT_CATALOGUE: list[dict[str, Any]] = [
    {
        "product_id": "SAV001",
        "name": "Instant Access Savings Account",
        "category": "savings_account",
        "provider": "Retail Bank (synthetic)",
        "expected_return_pct": 1.5,
        "expense_ratio_pct": 0.0,
        "typical_horizon_years": 1,
        "liquidity": "high",
        "description": "Instant-access deposit account, covered by the Deposit "
                        "Guarantee Scheme up to the statutory limit.",
    },
    {
        "product_id": "TDP001",
        "name": "12-Month Term Deposit",
        "category": "term_deposit",
        "provider": "Retail Bank (synthetic)",
        "expected_return_pct": 2.5,
        "expense_ratio_pct": 0.0,
        "typical_horizon_years": 1,
        "liquidity": "low",
        "description": "Fixed-term deposit; funds locked for 12 months in "
                        "exchange for a higher rate than instant access.",
    },
    {
        "product_id": "MMK001",
        "name": "Euro Money Market Fund",
        "category": "money_market",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 3.0,
        "expense_ratio_pct": 0.15,
        "typical_horizon_years": 1,
        "liquidity": "high",
        "description": "Short-duration euro money market instruments; capital "
                        "stability prioritised over growth.",
    },
    {
        "product_id": "GOV001",
        "name": "Irish Government Bond (short-dated)",
        "category": "government_bond",
        "provider": "NTMA-issued (synthetic tracking product)",
        "expected_return_pct": 3.2,
        "expense_ratio_pct": 0.1,
        "typical_horizon_years": 3,
        "liquidity": "medium",
        "description": "Short-dated sovereign bond exposure; capital value can "
                        "still fluctuate with interest rate movements.",
    },
    {
        "product_id": "GOV002",
        "name": "Irish Government Bond (long-dated)",
        "category": "government_bond",
        "provider": "NTMA-issued (synthetic tracking product)",
        "expected_return_pct": 4.0,
        "expense_ratio_pct": 0.1,
        "typical_horizon_years": 10,
        "liquidity": "medium",
        "description": "Longer-dated sovereign bond exposure; more sensitive to "
                        "interest rate changes than the short-dated equivalent.",
    },
    {
        "product_id": "CBI001",
        "name": "Investment-Grade Corporate Bond Fund",
        "category": "corporate_bond_ig",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 4.2,
        "expense_ratio_pct": 0.35,
        "typical_horizon_years": 5,
        "liquidity": "medium",
        "description": "Diversified investment-grade corporate bonds; higher "
                        "yield than government bonds with modest added credit risk.",
    },
    {
        "product_id": "MXL001",
        "name": "Low-Volatility Mixed Fund (30/70 equity/bond)",
        "category": "mixed_fund_low",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 4.5,
        "expense_ratio_pct": 0.5,
        "typical_horizon_years": 5,
        "liquidity": "medium",
        "description": "Conservative-leaning multi-asset fund; small equity "
                        "allocation for modest additional growth.",
    },
    {
        "product_id": "CB001",
        "name": "Broad Corporate Bond Fund",
        "category": "corporate_bond",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 5.0,
        "expense_ratio_pct": 0.4,
        "typical_horizon_years": 6,
        "liquidity": "medium",
        "description": "Investment-grade and higher-yield corporate bonds; "
                        "greater credit risk than the IG-only fund.",
    },
    {
        "product_id": "MXM001",
        "name": "Balanced Mixed Fund (60/40 equity/bond)",
        "category": "mixed_fund",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 6.0,
        "expense_ratio_pct": 0.6,
        "typical_horizon_years": 7,
        "liquidity": "medium",
        "description": "Balanced multi-asset fund targeting growth with "
                        "moderated volatility versus a pure equity fund.",
    },
    {
        "product_id": "ETB001",
        "name": "Broad Global Equity ETF",
        "category": "etf_broad",
        "provider": "ETF Provider (synthetic)",
        "expected_return_pct": 7.0,
        "expense_ratio_pct": 0.2,
        "typical_horizon_years": 10,
        "liquidity": "high",
        "description": "Passive tracker across developed-market large-cap "
                        "equities; broad diversification, low cost.",
    },
    {
        "product_id": "REIT001",
        "name": "Diversified European REIT Fund",
        "category": "reit",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 6.5,
        "expense_ratio_pct": 0.55,
        "typical_horizon_years": 8,
        "liquidity": "medium",
        "description": "Pooled exposure to income-generating commercial "
                        "property across multiple European markets.",
    },
    {
        "product_id": "EQF001",
        "name": "Active Global Equity Fund",
        "category": "equity_fund",
        "provider": "Asset Manager (synthetic)",
        "expected_return_pct": 8.0,
        "expense_ratio_pct": 0.9,
        "typical_horizon_years": 10,
        "liquidity": "high",
        "description": "Actively managed global equity fund seeking to "
                        "outperform a broad market index.",
    },
    {
        "product_id": "ETS001",
        "name": "Technology Sector ETF",
        "category": "etf_sector",
        "provider": "ETF Provider (synthetic)",
        "expected_return_pct": 9.0,
        "expense_ratio_pct": 0.3,
        "typical_horizon_years": 10,
        "liquidity": "high",
        "description": "Concentrated exposure to a single sector; higher "
                        "volatility than a broad-market equivalent.",
    },
    {
        "product_id": "EQI001",
        "name": "Individual Blue-Chip Equity Basket",
        "category": "individual_equity",
        "provider": "Brokerage (synthetic)",
        "expected_return_pct": 10.0,
        "expense_ratio_pct": 0.1,
        "typical_horizon_years": 10,
        "liquidity": "high",
        "description": "Direct holdings in a small number of individual "
                        "listed companies; concentration risk, no fund-level "
                        "diversification.",
    },
    {
        "product_id": "VEN001",
        "name": "Early-Stage Venture Fund",
        "category": "venture",
        "provider": "Venture Fund Manager (synthetic)",
        "expected_return_pct": 15.0,
        "expense_ratio_pct": 2.0,
        "typical_horizon_years": 12,
        "liquidity": "low",
        "description": "Illiquid pooled exposure to early-stage private "
                        "companies; highest risk and return band in the "
                        "catalogue, long lock-up periods.",
    },
]

class InvestmentAgent(BaseAgent):
    """
    Context-Aware hybrid investment product recommender
    
    Pipeline :
        Layer 1 _filter_by_risk_class()   rule-based CBI suitability filter
        Layer 2 _rank_products()          feature-based scoring, no LLM
        Layer 3 LLM synthesis             explain the ranke shortlist
        
    The LLM never sees the full catalogue and never sees rejected products -
    it only ever synthesises over what has already survived Layers 1 and 2.
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="InvestmentAgent")

    @property
    def system_prompt(self) -> str:
        return INVESTMENT_SYSTEM
    
    def _parse_response(self, raw: str) -> dict[str, Any]:
        """
        Investment recommendations are free text, so this just wraps the raw string."""
        return {"synthesis": raw.strip()}
    
    # Hybrid layer 1 — rule-based filtering
    def _filter_by_risk_class(self, risk_class: str) -> list[dict[str, Any]]:
        """
        Layer 1: keeps only catalogue products whose category is in the
        CBI-suitability allow-list for this risk_class (from
        config/constraints.py), so the ranking layer never even sees an
        unsuitable product. 
        
        Args: 
            risk_class: one of the five tiers in settings.risk.risk_classes.
        
        Returns:
            List of product dicts whose category is in the allow-set for 
            risk_class. Empty list if risk_class is unrecognised or no 
            catalogue products match (both are treated as caller errors 
            to surface explicitly rather than silently returning nothing)
        """

        allowed_categories = financial_constraints.RISK_PRODUCT_ALLOW.get(
            risk_class, set()
        )
        if not allowed_categories:
            logger.warning(
                f"[InvestmentAgent] No allowed categories for risk_class="
                f"'{risk_class}' — check settings.risk.risk_classes spelling."
            )
            return []

        filtered = [
            product for product in self.catalogue
            if product["category"] in allowed_categories
        ]
        logger.debug(
            f"[InvestmentAgent] Filtered catalogue for risk_class="
            f"'{risk_class}': {len(filtered)}/{len(self.catalogue)} products "
            f"survive (categories: {sorted(allowed_categories)})"
        )
        return filtered
    
    # Hybrid layer 2 — feature-based ranking

    def _normalised(self, value: float, low: float, high: float) -> float:
        """Min-max nomalise value into [0, 1]; guards against a zero range."""
        if high <= low:
            return 0.5
        return float(min(max((value - low) / (high - low), 0.0), 1.0))
    
    def _horizon_fit_score(
            self, product_horizon: int, user_horizon: int 
    ) -> float:
        """
        Score in [0, 1] for how closely a product's typical holding period
        matches the user's stated investment horizon; 1.0 = exact match,
        decaying linearly to 0 at a 10-year gap.
        """
        gap = abs(product_horizon - user_horizon)
        return float(max(1.0 - gap / 10.0, 0.0))
    
    def _rank_products(
        self,
        products: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Layer 2: scores each filtered product on return, cost (inverted —
        lower cost is better) and horizon fit, combines them with weights
        from settings.investment, and returns the list sorted best-first.

        Scoring components (weights from settings.investment, sum to 1.0):
          return_score       — normalised expected_return_pct across shortlist
          cost_score         — normalised (1 - expense_ratio_pct); lower cost
                                is better, so the ratio is inverted before
                                normalisation
          horizon_fit_score  — from _horizon_fit_score() against the user's
                                investment_horizon (default 5 years if absent)
                                
        Args:
            products: output of _filter_by_risk_class() — already suitable.
            context:  run() context; reads 'user_features.investment_horizon'.

        Returns:
            Products (each with an added 'score' and 'score_breakdown' key)
            sorted descending by score. Empty list in, empty list out.
        """
        if not products:
            return []
        
        user_features = context.get("user_features", {}) or {}
        user_horizon = int(user_features.get("investment_hprizon", 5))

        returns = [p["expected_return_pct"] for p in products]
        costs = [p["expense_ratio_pct"] for p in products]
        min_return, max_return = min(returns), max(returns)
        min_cost, max_cost = min(costs), max(costs)

        weights = settings.investment
        scored: list[dict[str, Any]] = []
        for product in products:
            return_score = self._normalise(
                product["expected_return_pct"], min_return, max_return
            )
            # Invert cost so that lower expense_ratio_pct scores higher.
            cost_score = 1.0 - self._normalise(
                product["expense_ratio_pct"], min_cost, max_cost
            )
            horizon_score = self._horizon_fit_score(
                product["typical_horizon_years"], user_horizon
            )

            total = (
                weights.return_weight * return_score
                + weights.cost_weight * cost_score
                + weights.horizon_fit_weight * horizon_score
            )

            enriched = dict(product)
            enriched["score"] = round(float(total), 4)
            enriched["score_breakdown"] = {
                "return_score": round(return_score, 4),
                "cost_score": round(cost_score, 4),
                "horizon_fit_score": round(horizon_score, 4),
                "user_investment_horizon": user_horizon,
            }
            scored.append(enriched)

        scored.sort(key=lambda p: p["score"], reverse=True)
        logger.debug(
            f"[InvestmentAgent] Ranked {len(scored)} products — top: "
            f"{scored[0]['product_id']} (score={scored[0]['score']})"
            if scored else "[InvestmentAgent] No products to rank"
        )
        return scored

    # Hybrid layer 3 — LLM synthesis

    def _build_synthesis_prompt(
        self,
        risk_class: str,
        shortlist: list[dict[str, Any]],
        user_features: dict[str, Any],
    ) -> str:
        """
        Builds the LLM prompt using ONLY the already-filtered, already-
        ranked shortlist — the model is never shown the full catalogue or
        any rejected product, so it has no way to reference something it
        wasn't given (a concrete anti-hallucination guardrail for this
        agent).
        """
        horizon = user_features.get("investment_horizon", "not stated")
        lines = [
            f"User risk classification: {risk_class}",
            f"User stated investment horizon: {horizon} years",
            "",
            "Ranked shortlist (already filtered for suitability, already "
            "scored — present these, do not add or remove any):",
        ]
        for rank, product in enumerate(shortlist, start=1):
            lines.append(
                f"{rank}. {product['name']} ({product['category']}) — "
                f"expected_return_pct={product['expected_return_pct']}, "
                f"expense_ratio_pct={product['expense_ratio_pct']}, "
                f"typical_horizon_years={product['typical_horizon_years']}, "
                f"score={product['score']}"
            )
        lines.append(
            "\nWrite the recommendation now, per your system instructions."
        )
        return "\n".join(lines)