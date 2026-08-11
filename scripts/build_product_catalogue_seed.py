"""
Builds data/raw/product_catalogue/catalogue_seed.json — the intended
catalogue InvestmentAgent is supposed to load, currently missing from this
environment entirely (not stale — absent), so every run has been silently
falling back to the 15-product inline catalogue in agents/investment_agent.py.

WHY THIS EXISTS
    Three problems, all traced back to this one missing file:

    1. test_catalogue_loads_from_the_seed_file fails outright — there's
       nothing to load.

    2. test_every_tier_has_more_candidates_than_the_shortlist fails: with
       top_k=3, every risk tier needs > 6 eligible products for ranking to
       mean anything. The 15-product fallback gives conservative and
       moderately_aggressive only 5 each — barely more than the shortlist
       itself. This matters most for conservative and moderately_conservative
       specifically, since those are the two tiers real customers actually
       land in most often (see the risk-model coverage audit).

    3. test_gov_and_mmk_are_in_both_enrichment_maps and
       test_authoritative_source_beats_the_ticker_proxy fail: nothing in the
       fallback catalogue carries a `real_data_key`, so the precedence rule
       between TICKER_PROXY_MAP.py's live-market proxy and an authoritative
       real_data_key source (CBI/An Post/ECB) has never actually been
       exercised by any product.

WHAT THIS DOES NOT CHANGE
    RISK_PRODUCT_ALLOW (config/constraints.py) is untouched — which product
    CATEGORY is suitable for which risk tier is a CBI-suitability-grounded
    design decision, not a data-availability problem, and loosening it here
    would be solving the wrong layer. This script only adds MORE PRODUCTS
    within the categories that already exist, in the tiers that already
    allow them.

DESIGN
    Keeps all 15 of the original product_ids and their exact category/
    field values unchanged — TICKER_PROXY_MAP and existing tests reference
    several of them (GOV001, ETB001, MMK001) directly by ID. Adds one
    additional product per category (an "*002" sibling, distinct provider,
    a realistic return/cost variation, not a copy) so every one of the 5
    risk tiers clears the > 2*top_k threshold. Adds a `real_data_key` to
    the products least likely to have a liquid market-ticker proxy
    (SAV001, TDP001 — bank deposit rates, not exchange-traded) plus GOV001
    and MMK001 specifically, so the precedence-guard tests have something
    real to exercise: an authoritative published rate AND a ticker proxy
    both mapped to the same product.

USAGE
    python scripts/build_product_catalogue_seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

OUT_PATH = settings.product_data.seed_path


# Each tuple: (product_id, name, category, provider, expected_return_pct,
#              expense_ratio_pct, typical_horizon_years, liquidity,
#              description, real_data_key_or_None)
_PRODUCTS: list[tuple] = [
    # ── originals — unchanged from agents/investment_agent.py's fallback ──
    ("SAV001", "Instant Access Savings Account", "savings_account",
     "Retail Bank (synthetic)", 1.5, 0.0, 1, "high",
     "Instant-access deposit account, covered by the Deposit Guarantee "
     "Scheme up to the statutory limit.", "anpost.demand_deposit_account"),
    ("TDP001", "12-Month Term Deposit", "term_deposit",
     "Retail Bank (synthetic)", 2.5, 0.0, 1, "low",
     "Fixed-term deposit; funds locked for 12 months in exchange for a "
     "higher rate than instant access.", "anpost.term_deposit_12m"),
    ("MMK001", "Euro Money Market Fund", "money_market",
     "Asset Manager (synthetic)", 3.0, 0.15, 1, "high",
     "Short-duration euro money market instruments; capital stability "
     "prioritised over growth.", "ecb.deposit_facility_rate"),
    ("GOV001", "Irish Government Bond (short-dated)", "government_bond",
     "NTMA-issued (synthetic tracking product)", 3.2, 0.1, 3, "medium",
     "Short-dated sovereign bond exposure; capital value can still "
     "fluctuate with interest rate movements.", "cbi.irish_govt_bond_2y"),
    ("GOV002", "Irish Government Bond (long-dated)", "government_bond",
     "NTMA-issued (synthetic tracking product)", 4.0, 0.1, 10, "medium",
     "Longer-dated sovereign bond exposure; more sensitive to interest "
     "rate changes than the short-dated equivalent.", None),
    ("CBI001", "Investment-Grade Corporate Bond Fund", "corporate_bond_ig",
     "Asset Manager (synthetic)", 4.2, 0.35, 5, "medium",
     "Diversified investment-grade corporate bonds; higher yield than "
     "government bonds with modest added credit risk.", None),
    ("MXL001", "Low-Volatility Mixed Fund (30/70 equity/bond)", "mixed_fund_low",
     "Asset Manager (synthetic)", 4.5, 0.5, 5, "medium",
     "Conservative-leaning multi-asset fund; small equity allocation for "
     "modest additional growth.", None),
    ("CB001", "Broad Corporate Bond Fund", "corporate_bond",
     "Asset Manager (synthetic)", 5.0, 0.4, 6, "medium",
     "Investment-grade and higher-yield corporate bonds; greater credit "
     "risk than the IG-only fund.", None),
    ("MXM001", "Balanced Mixed Fund (60/40 equity/bond)", "mixed_fund",
     "Asset Manager (synthetic)", 6.0, 0.6, 7, "medium",
     "Balanced multi-asset fund targeting growth with moderated "
     "volatility versus a pure equity fund.", None),
    ("ETB001", "Broad Global Equity ETF", "etf_broad",
     "ETF Provider (synthetic)", 7.0, 0.2, 10, "high",
     "Passive tracker across developed-market large-cap equities; broad "
     "diversification, low cost.", None),
    ("REIT001", "Diversified European REIT Fund", "reit",
     "Asset Manager (synthetic)", 6.5, 0.55, 8, "medium",
     "Pooled exposure to income-generating commercial property across "
     "multiple European markets.", None),
    ("EQF001", "Active Global Equity Fund", "equity_fund",
     "Asset Manager (synthetic)", 8.0, 0.9, 10, "high",
     "Actively managed global equity fund seeking to outperform a broad "
     "market index.", None),
    ("ETS001", "Technology Sector ETF", "etf_sector",
     "ETF Provider (synthetic)", 9.0, 0.3, 10, "high",
     "Concentrated exposure to a single sector; higher volatility than a "
     "broad-market equivalent.", None),
    ("EQI001", "Individual Blue-Chip Equity Basket", "individual_equity",
     "Brokerage (synthetic)", 10.0, 0.1, 10, "high",
     "Direct holdings in a small number of individual listed companies; "
     "concentration risk, no fund-level diversification.", None),
    ("VEN001", "Early-Stage Venture Fund", "venture",
     "Venture Fund Manager (synthetic)", 15.0, 2.0, 12, "low",
     "Illiquid pooled exposure to early-stage private companies; highest "
     "risk and return band in the catalogue, long lock-up periods.", None),

    # ── new — one additional product per category, so every risk tier ──
    # ── clears the (top_k * 2) minimum for meaningful ranking ──────────
    ("SAV002", "Regular Saver Account", "savings_account",
     "Credit Union (synthetic)", 2.0, 0.0, 1, "high",
     "Deposit account rewarding a fixed monthly contribution with a "
     "higher rate than instant access, within a capped balance.", None),
    ("TDP002", "6-Month Term Deposit", "term_deposit",
     "Retail Bank (synthetic)", 2.1, 0.0, 1, "low",
     "Shorter lock-up than the 12-month equivalent, at a correspondingly "
     "lower fixed rate.", None),
    ("MMK002", "Short Government Bill Fund", "money_market",
     "Asset Manager (synthetic)", 2.8, 0.12, 1, "high",
     "Pooled exposure to short-dated government treasury bills; slightly "
     "lower cost than the money market fund equivalent.", None),
    ("GOV003", "Irish Government Bond (medium-dated)", "government_bond",
     "NTMA-issued (synthetic tracking product)", 3.6, 0.1, 6, "medium",
     "Sits between the short- and long-dated equivalents on both "
     "duration and rate sensitivity.", None),
    ("CBI002", "Euro Investment-Grade Corporate Bond ETF", "corporate_bond_ig",
     "ETF Provider (synthetic)", 4.0, 0.22, 5, "high",
     "Passive, exchange-traded alternative to the actively managed IG "
     "corporate bond fund, at a lower expense ratio.", None),
    ("MXL002", "Low-Volatility Mixed Fund (20/80 equity/bond)", "mixed_fund_low",
     "Asset Manager (synthetic)", 4.1, 0.45, 5, "medium",
     "A more conservative equity/bond split than the 30/70 equivalent, "
     "for a lower expected return and lower volatility.", None),
    ("CB002", "Short-Duration High-Yield Bond Fund", "corporate_bond",
     "Asset Manager (synthetic)", 5.4, 0.5, 4, "medium",
     "Higher-yield corporate bonds with a shorter average duration, "
     "trading additional credit risk for reduced rate sensitivity.", None),
    ("MXM002", "Balanced Mixed Fund (50/50 equity/bond)", "mixed_fund",
     "Asset Manager (synthetic)", 5.6, 0.55, 7, "medium",
     "A more bond-weighted balance than the 60/40 equivalent, for a "
     "modestly lower expected return and volatility.", None),
    ("ETB002", "Broad European Equity ETF", "etf_broad",
     "ETF Provider (synthetic)", 6.7, 0.18, 10, "high",
     "Passive tracker across developed European large-cap equities — "
     "regional rather than global diversification.", None),
    ("REIT002", "Irish Commercial Property Fund", "reit",
     "Asset Manager (synthetic)", 6.0, 0.6, 8, "medium",
     "Pooled exposure to Irish commercial property specifically, rather "
     "than the pan-European spread of the diversified equivalent.", None),
    ("EQF002", "Passive Global Equity Fund", "equity_fund",
     "Asset Manager (synthetic)", 7.3, 0.4, 10, "high",
     "Lower-cost, index-tracking alternative to the actively managed "
     "global equity fund, trading outperformance potential for cost.", None),
    ("ETS002", "Healthcare Sector ETF", "etf_sector",
     "ETF Provider (synthetic)", 8.5, 0.32, 10, "high",
     "Concentrated exposure to the healthcare sector — a different "
     "single-sector bet than the technology sector equivalent.", None),
    ("EQI002", "Individual Dividend-Growth Equity Basket", "individual_equity",
     "Brokerage (synthetic)", 9.0, 0.1, 10, "high",
     "Direct holdings in a small number of established, dividend-paying "
     "companies — a lower-volatility individual-equity profile than the "
     "blue-chip growth basket.", None),
    ("VEN002", "Diversified Early-Stage Venture Fund", "venture",
     "Venture Fund Manager (synthetic)", 13.5, 2.2, 12, "low",
     "Broader portfolio of early-stage positions than the flagship "
     "venture fund, trading some upside concentration for diversification.",
     None),
]


def build() -> dict[str, Any]:
    products = []
    for (pid, name, category, provider, ret, expense, horizon, liquidity,
         desc, real_data_key) in _PRODUCTS:
        entry: dict[str, Any] = {
            "product_id": pid,
            "name": name,
            "category": category,
            "provider": provider,
            "typical_horizon_years": horizon,
            "liquidity": liquidity,
            "description": desc,
            "synthetic": {
                "expected_return_pct": ret,
                "expense_ratio_pct": expense,
            },
        }
        if real_data_key:
            entry["real_data_key"] = real_data_key
        products.append(entry)

    return {
        "_meta": {
            "note": (
                "Synthetic illustrative catalogue for prototype development, "
                "NOT a live feed — expected_return_pct/expense_ratio_pct under "
                "each product's 'synthetic' key are starting figures, replaced "
                "by agents/investment_agent.py's enrichment layers "
                "(_apply_product_data, _apply_live_pricing) wherever a real "
                "sourced figure or live ticker is available and enabled."
            ),
            "source": "scripts/build_product_catalogue_seed.py",
        },
        "products": products,
    }


def main() -> int:
    doc = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {len(doc['products'])} products -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())