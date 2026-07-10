"""
RUNNING:
  python -m pytest tests/unit/test_investment_filtering.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.investment_agent import InvestmentAgent
from utils.llm_client import LLMClient


def make_agent() -> InvestmentAgent:
    """Investment agent in mock mode — filtering logic requires no API key."""
    client = LLMClient()
    return InvestmentAgent(client)


class TestFilterByRiskClass:

    def test_conservative_investor_never_sees_individual_equity(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("conservative")
        categories = {p["category"] for p in filtered}
        assert "individual_equity" not in categories
        assert "venture" not in categories

    def test_conservative_investor_sees_only_allowed_categories(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("conservative")
        allowed = {"savings_account", "government_bond", "money_market", "term_deposit"}
        categories = {p["category"] for p in filtered}
        assert categories.issubset(allowed)
        assert len(filtered) > 0

    def test_aggressive_investor_can_see_individual_equity(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("aggressive")
        categories = {p["category"] for p in filtered}
        assert "individual_equity" in categories

    def test_aggressive_investor_never_sees_savings_account(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("aggressive")
        categories = {p["category"] for p in filtered}
        assert "savings_account" not in categories

    def test_moderate_investor_excludes_extremes_both_ways(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        categories = {p["category"] for p in filtered}
        assert "individual_equity" not in categories
        assert "venture" not in categories
        assert "savings_account" not in categories

    def test_unknown_risk_class_returns_empty_list(self):
        agent = make_agent()
        assert agent._filter_by_risk_class("not_a_real_tier") == []

    def test_every_risk_class_has_at_least_one_product(self):
        agent = make_agent()
        from config.settings import settings
        for risk_class in settings.risk.risk_classes:
            filtered = agent._filter_by_risk_class(risk_class)
            assert len(filtered) > 0, f"No products survive filtering for {risk_class}"

    def test_filtered_products_retain_full_product_dict(self):
        agent = make_agent()
        filtered = agent._filter_by_risk_class("moderate")
        for product in filtered:
            assert "product_id" in product
            assert "expected_return_pct" in product
            assert "expense_ratio_pct" in product
