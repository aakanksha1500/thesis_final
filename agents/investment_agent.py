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

import json
import time
from typing import Any

from agents.base_agent import AgentResult, BaseAgent
from config.constraints import financial_constraints
from config.prompts import INVESTMENT_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger
from utils import trace

logger = get_logger(__name__)

# Synthetic illustrative data for prototype development — NOT a live feed.
# Categories match config.constraints.FinancialConstraints.RISK_PRODUCT_ALLOW
# exactly, so every product is reachable by at least one risk tier and the
# filter layer has a real rule set to operate against.

_FALLBACK_CATALOGUE: list[dict[str, Any]] = [
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

TICKER_PROXY_MAP: dict[str, dict[str, Any]] = {
    "MMK001": {"tickers": ["BIL"], "weights": [1.0], "note": "1-3 month T-Bill proxy for cash-like money market exposure"},
    "GOV001": {"tickers": ["SHY"], "weights": [1.0], "note": "1-3 year Treasury proxy for short-dated sovereign bonds"},
    "GOV002": {"tickers": ["IEF"], "weights": [1.0], "note": "7-10 year Treasury proxy for long-dated sovereign bonds"},
    "CBI001": {"tickers": ["LQD"], "weights": [1.0], "note": "Investment-grade corporate bond index proxy"},
    "MXL001": {"tickers": ["VT", "BND"], "weights": [0.3, 0.7], "note": "30/70 global equity / aggregate bond blend"},
    "CB001": {"tickers": ["HYG"], "weights": [1.0], "note": "High-yield corporate bond index proxy"},
    "MXM001": {"tickers": ["VT", "BND"], "weights": [0.6, 0.4], "note": "60/40 global equity / aggregate bond blend"},
    "ETB001": {"tickers": ["VT"], "weights": [1.0], "note": "Total world equity index proxy"},
    "REIT001": {"tickers": ["VNQ"], "weights": [1.0], "note": "US-listed REIT index used as global REIT proxy"},
    "EQF001": {"tickers": ["VT"], "weights": [1.0], "note": "Total world equity index proxy (passive stand-in for an active fund)"},
    "ETS001": {"tickers": ["XLK"], "weights": [1.0], "note": "Technology sector index proxy"},
    "EQI001": {"tickers": ["DIA"], "weights": [1.0], "note": "Blue-chip index proxy for an individual-stock basket"},
}

def _load_catalogue_seed() -> list[dict[str, Any]]:
    """
    Read the catalogue from data/raw/product_catalogue/catalogue_seed.json.

    The seed nests illustrative figures under a "synthetic" key. This flattens
    them into the shape the rest of the agent already expects, so no ranking,
    filtering or constraint code changes — but the nesting on disk means a
    reader of the file cannot mistake an illustrative expense ratio for a
    sourced one, which a flat key invites.
    """
    path = settings.product_data.seed_path
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        products: list[dict[str, Any]] = []
        for entry in doc["products"]:
            product = {k: v for k, v in entry.items() if k != "synthetic"}
            product.update(entry.get("synthetic", {}))
            products.append(product)
        if not products:
            raise ValueError("seed file contains no products")
        logger.info(
            f"[InvestmentAgent] loaded {len(products)} products from {path.name}"
        )
        return products
    except Exception as exc:
        logger.error(
            f"[InvestmentAgent] could not read the product seed at {path} "
            f"({type(exc).__name__}: {exc}) — falling back to the {len(_FALLBACK_CATALOGUE)}"
            f"-product inline catalogue. Every RQ2 figure computed in this state "
            f"reflects a SMALLER product universe than the seed defines, so do "
            f"not compare it with results generated from the seed."
        )
        return [dict(p) for p in _FALLBACK_CATALOGUE]

IRISH_PRODUCT_CATALOGUE: list[dict[str, Any]] = _load_catalogue_seed()

def _apply_product_data(catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Layer real sourced figures over the synthetic ones, per FIELD, with a
    provenance stamp per field rather than per product.

    WHY PER-FIELD
        expected_return_source already existed, covering one field. Once a
        second source enriches a second field, a single product-level "is this
        real?" flag becomes a lie in both directions: a product with a real
        deposit rate and an illustrative expense ratio is neither "real" nor
        "synthetic". Each enriched field therefore carries its own
        <field>_source and <field>_citation, which is what lets
        ExplainabilityAgent tell a user precisely which number is illustrative.

    Gated on settings.product_data.use_real_product_data because turning it on
    changes what RQ2 measures — see ProductDataConfig.
    """
    if not settings.product_data.use_real_product_data:
        return catalogue

    from utils.product_data_client import get_product_data_client  # noqa: PLC0415

    client = get_product_data_client()
    enriched, n_sourced = [], 0

    for product in catalogue:
        product = dict(product)
        key = product.get("real_data_key")
        if key:
            fact = client.get_fact(key, field="expected_return_pct")
            if fact is not None:
                product["expected_return_pct"] = round(fact.value, 2)
                product["expected_return_source"] = f"sourced:{fact.tier}"
                product["expected_return_citation"] = (
                    f"{fact.source}, as of {fact.as_of}"
                )
                n_sourced += 1
        enriched.append(product)

    logger.info(
        f"[InvestmentAgent] product data enrichment: {n_sourced}/{len(catalogue)} "
        f"products carry a sourced expected_return_pct "
        f"(client mode={client.mode})"
    )
    return enriched

def _apply_live_pricing(catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return a copy of `catalogue` with expected_return_pct replaced by a
    live/snapshotted trailing return wherever TICKER_PROXY_MAP has an
    entry and settings.market_data yields a quote. Falls through to the
    synthetic value per-product on any failure — never raises, and never
    mutates the module-level IRISH_PRODUCT_CATALOGUE.

    Adds two provenance fields for the audit trail (O3):
      expected_return_source: "live" | "synthetic"
      pricing_note: proxy ticker(s) and period used, or None
    """
    from utils.market_data_client import market_data_client  # noqa: PLC0415

    enriched = []
    for product in catalogue:
        product = dict(product)
        product.setdefault("expected_return_source", "synthetic")
        product.setdefault("pricing_note", None)
        product.setdefault("expense_ratio_source", "synthetic")

        if product["expected_return_source"].startswith("sourced:"):
            enriched.append(product)
            continue
 

        proxy = TICKER_PROXY_MAP.get(product["product_id"])
        if proxy and settings.market_data.enabled:
            weighted_return = 0.0
            as_of = None
            missing = False
            for ticker, weight in zip(proxy["tickers"], proxy["weights"]):
                quote = market_data_client.get_trailing_return_pct(ticker)
                if quote is None:
                    missing = True
                    break
                weighted_return += quote.trailing_return_pct * weight
                as_of = quote.as_of

            if not missing:
                product["expected_return_pct"] = round(weighted_return, 2)
                product["expected_return_source"] = "live"
                product["pricing_note"] = (
                    f"{'+'.join(proxy['tickers'])} trailing "
                    f"{settings.market_data.period_days}d return, as of {as_of} "
                    f"({proxy['note']})"
                )
            else:
                logger.debug(
                    f"[InvestmentAgent] Live pricing unavailable for "
                    f"{product['product_id']} — using synthetic expected_return_pct"
                )

        enriched.append(product)
    return enriched

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
        self.catalogue = _apply_live_pricing(_apply_product_data(IRISH_PRODUCT_CATALOGUE))
 

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

    def _normalise(self, value: float, low: float, high: float) -> float:
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
        user_horizon = int(user_features.get("investment_horizon", 5))

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

    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Process one investment recommendation request.

        context keys used:
          'risk_class'            (str, required)  — from RiskProfilingAgent
          'user_features'         (dict, optional) — reads investment_horizon
          'conversation_history'  (list, optional) — unused in Phase 4

        Returns AgentResult with payload:
          status, risk_class, shortlist (ranked, top_k), synthesis,
          constraint_violations, deliverable
        """
        start_time = time.perf_counter()

        risk_class: str | None = context.get("risk_class")
        user_features: dict = context.get("user_features", {})

        if not risk_class:
            payload = {
                "status": "incomplete",
                "message": (
                    "Cannot recommend products without a risk_class. "
                    "Run RiskProfilingAgent first and pass its output "
                    "forward via the Orchestrator."
                ),
            }
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("[InvestmentAgent] Missing risk_class in context")
            return self._make_result(
                payload=payload,
                duration_ms=duration_ms,
                error="Missing risk_class",
            )

        # Layer 1 — rule-based filter
        filtered = self._filter_by_risk_class(risk_class)
        trace.emit("FILTER", f"risk={risk_class}",
                   kept=f"{len(filtered)}/{len(self.catalogue)}")
        if len(filtered) < settings.investment.min_products_after_filter:
            payload = {
                "status": "no_suitable_products",
                "risk_class": risk_class,
                "message": (
                    f"No products in the catalogue are suitable for "
                    f"risk_class='{risk_class}'."
                ),
            }
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"[InvestmentAgent] No suitable products for '{risk_class}'"
            )
            return self._make_result(
                payload=payload,
                duration_ms=duration_ms,
                error="No suitable products after filtering",
            )

        # Layer 2 — ranking
        ranked = self._rank_products(filtered, context)
        shortlist = ranked[: settings.investment.top_k]
        if shortlist:
            trace.emit("RANK", f"top={shortlist[0]['product_id']}",
                       score=shortlist[0]["score"], shortlisted=len(shortlist))

        # Layer 3 — LLM synthesis over the shortlist only
        prompt = self._build_synthesis_prompt(risk_class, shortlist, user_features)
        try:
            raw_synthesis, tokens = self._call_llm(prompt)
            synthesis = raw_synthesis.strip()
        except Exception as exc:
            logger.warning(
                f"[InvestmentAgent] Synthesis generation failed: {exc} "
                f"— using fallback"
            )
            top = shortlist[0]
            synthesis = (
                f"Based on your '{risk_class}' risk profile, the top-ranked "
                f"suitable product is {top['name']} ({top['category']}), with "
                f"an indicative expected return of {top['expected_return_pct']}% "
                f"and an expense ratio of {top['expense_ratio_pct']}%. "
                f"This is not regulated financial advice — please consult a "
                f"qualified advisor. Past performance is not indicative of "
                f"future results."
            )
            tokens = 0

        # Post-hoc deterministic validation of the LLM's own text — the same
        # FinancialConstraints layer used by RiskProfilingAgent's response
        # checking (Raza et al. [1]; O3 audit-trace principle).
        top_product = shortlist[0]
        deliverable, violations = financial_constraints.validate_response(
            response_text=synthesis,
            risk_class=risk_class,
            product_category=top_product["category"],
            claimed_return=top_product["expected_return_pct"],
        )
        trace.emit("✓ VALIDATE", "FinancialConstraints",
                   violations=len(violations), deliverable=deliverable)

        hallucination_report = None
        rag_sources: list[str] = []
        if settings.hallucination.run_inline:
            try:
                from rag.hallucination_detector import hallucination_detector
                from rag.knowledge_base import knowledge_base

                grounding_query = f"{risk_class} {top_product['category']} {top_product['name']}"
                grounding_contexts = knowledge_base.retrieve(grounding_query)
                trace.emit("TOOL", "knowledge_base.retrieve",
                           chunks=len(grounding_contexts))
                report = hallucination_detector.score_response(
                    response_text=synthesis,
                    grounding_contexts=grounding_contexts,
                )
                hallucination_report = report.to_dict()
                trace.emit("TOOL", "hallucination_detector.score_response",
                           mode=report.mode,
                           claims=hallucination_report["n_claims"],
                           flagged=hallucination_report["n_flagged"])
                rag_sources = [c["source"] for c in grounding_contexts]

                if report.hallucination_rate > 0:
                    logger.warning(
                        f"[InvestmentAgent] HHEM flagged "
                        f"{hallucination_report['n_flagged']}/{hallucination_report['n_claims']} "
                        f"claims (mode={report.mode}) in synthesis"
                    )
            except Exception as exc:
                logger.warning(f"[InvestmentAgent] Hallucination detection failed: {exc}")

        payload = {
            "status": "complete",
            "risk_class": risk_class,
            "shortlist": shortlist,
            "synthesis": synthesis,
            "deliverable": deliverable,
            "constraint_violations": [
                {
                    "rule_id": v.rule_id,
                    "description": v.description,
                    "severity": v.severity,
                }
                for v in violations
            ],
            "hallucination_report": hallucination_report,
            "hallucination_flagged": bool(
                hallucination_report and hallucination_report["n_flagged"] > 0
            ),
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[InvestmentAgent] risk_class={risk_class} "
            f"top_product={top_product['product_id']} "
            f"deliverable={deliverable} duration={duration_ms:.0f}ms"
        )

        return self._make_result(
            payload=payload,
            raw=synthesis,
            duration_ms=duration_ms,
            tokens=tokens,
            rag_sources=rag_sources,
            error=None if deliverable else "Constraint hard-block on synthesis output",
            routing_context={
                "risk_class": risk_class,
                "top_product_id": top_product["product_id"],
                "top_product_score": top_product["score"],
                "hallucination_flagged": bool(
                    hallucination_report and hallucination_report["n_flagged"] > 0
                ),
            },
        )
