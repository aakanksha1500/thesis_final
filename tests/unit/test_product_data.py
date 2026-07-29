"""
tests/unit/test_product_data.py

Covers the product-data enrichment layer added on top of MarketDataClient's
return enrichment.

THE PROPERTY THESE TESTS EXIST TO PROTECT
    Not "does enrichment work" — that is the easy half. The hard half is that
    a figure must never carry a 'sourced' provenance stamp unless it really
    was sourced, and must never be silently replaced by a less authoritative
    one. Every failure mode below produced a plausible number in some earlier
    version of this codebase.

RUNNING
    python -m pytest tests/unit/test_product_data.py -v
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from agents.investment_agent import (
    IRISH_PRODUCT_CATALOGUE,
    TICKER_PROXY_MAP,
    _apply_product_data,
)
from config.constraints import FinancialConstraints
from config.settings import settings
from utils.product_data_client import ProductDataClient


@pytest.fixture
def snapshot_path(tmp_path, monkeypatch):
    path = tmp_path / "product_snapshot.json"
    monkeypatch.setattr(settings.product_data, "snapshot_path", path, raising=False)
    return path


def write_snapshot(path, entries):
    path.write_text(json.dumps({"_meta": {}, "entries": entries}), encoding="utf-8")


def entry(value, days_old=0, field="expected_return_pct"):
    return {
        "field": field,
        "value": value,
        "as_of": (date.today() - timedelta(days=days_old)).isoformat(),
        "source": "Test source",
        "source_url": "https://example.invalid",
    }



class TestCatalogueSeed:

    def test_catalogue_loads_from_the_seed_file(self):
        assert len(IRISH_PRODUCT_CATALOGUE) > 15, (
            "the seed file should be in use; 15 means the inline fallback ran"
        )

    def test_every_category_is_reachable_by_some_risk_tier(self):
        """
        A category outside RISK_PRODUCT_ALLOW can never be recommended and
        fails silently — the filter simply returns nothing for it.
        """
        allowed = set()
        for cats in FinancialConstraints.RISK_PRODUCT_ALLOW.values():
            allowed |= set(cats)
        orphans = {p["category"] for p in IRISH_PRODUCT_CATALOGUE} - allowed
        assert not orphans, f"unreachable categories: {orphans}"

    def test_every_tier_has_more_candidates_than_the_shortlist(self):
        """
        With top_k=3 and only 4-5 candidates per tier, the ranker was nearly a
        no-op and precision@3 measured the filter, not Layer 2.
        """
        top_k = settings.investment.top_k
        for tier, cats in FinancialConstraints.RISK_PRODUCT_ALLOW.items():
            n = sum(1 for p in IRISH_PRODUCT_CATALOGUE if p["category"] in cats)
            assert n > top_k * 2, (
                f"{tier} has only {n} candidates for a top-{top_k} shortlist — "
                f"ranking is close to trivial"
            )

    def test_product_ids_are_unique(self):
        ids = [p["product_id"] for p in IRISH_PRODUCT_CATALOGUE]
        assert len(ids) == len(set(ids))



class TestProvenance:

    def test_disabled_by_default_leaves_everything_synthetic(self, monkeypatch):
        monkeypatch.setattr(settings.product_data, "use_real_product_data", False)
        out = _apply_product_data(IRISH_PRODUCT_CATALOGUE)
        assert all("expected_return_citation" not in p for p in out)

    def test_empty_snapshot_fabricates_nothing(self, snapshot_path, monkeypatch):
        """
        The shipped snapshot has value=None everywhere. That must produce
        synthetic figures, not zeros and not invented numbers.
        """
        write_snapshot(snapshot_path, {"anpost.deposit_account": entry(None)})
        client = ProductDataClient()
        assert client.get_fact("anpost.deposit_account") is None

    def test_sourced_value_carries_a_checkable_citation(self, snapshot_path):
        write_snapshot(snapshot_path, {"anpost.deposit_account": entry(0.13)})
        fact = ProductDataClient().get_fact("anpost.deposit_account")
        assert fact is not None
        assert fact.value == 0.13
        assert fact.source and fact.as_of

    def test_unknown_key_returns_none(self, snapshot_path):
        write_snapshot(snapshot_path, {})
        assert ProductDataClient().get_fact("nope.not.a.key") is None

    def test_field_mismatch_is_not_served(self, snapshot_path):
        """An expense-ratio entry must not be handed back as a return."""
        write_snapshot(snapshot_path,
                       {"k": entry(0.4, field="expense_ratio_pct")})
        assert ProductDataClient().get_fact("k", field="expected_return_pct") is None

    def test_malformed_value_degrades_instead_of_raising(self, snapshot_path):
        bad = entry("not-a-number")
        write_snapshot(snapshot_path, {"k": bad})
        assert ProductDataClient().get_fact("k") is None



class TestStaleness:

    def test_fresh_entry_is_served(self, snapshot_path):
        write_snapshot(snapshot_path, {"k": entry(2.5, days_old=10)})
        assert ProductDataClient().get_fact("k") is not None

    def test_stale_entry_is_refused_not_warned_about(self, snapshot_path):
        """
        Refusing degrades to an honestly-labelled synthetic figure. Warning
        would leave a stale number in the results file wearing a 'real'
        provenance stamp, which is strictly worse.
        """
        old = settings.product_data.max_age_days + 1
        write_snapshot(snapshot_path, {"k": entry(9.99, days_old=old)})
        assert ProductDataClient().get_fact("k") is None

    def test_missing_date_is_refused(self, snapshot_path):
        e = entry(2.5)
        e["as_of"] = None
        write_snapshot(snapshot_path, {"k": e})
        assert ProductDataClient().get_fact("k") is None



class TestSourcePrecedence:

    def test_gov_and_mmk_are_in_both_enrichment_maps(self):
        """
        Guards the assumption the precedence rule exists for. If this ever
        stops being true the collision test below silently stops testing
        anything.
        """
        keyed = {p["product_id"] for p in IRISH_PRODUCT_CATALOGUE
                 if p.get("real_data_key")}
        assert keyed & set(TICKER_PROXY_MAP), (
            "no product is in both maps — the precedence guard is untested"
        )

    def test_authoritative_source_beats_the_ticker_proxy(
        self, snapshot_path, monkeypatch
    ):
        """
        GOV001 has both an Irish sovereign entry and a US Treasury ETF proxy
        (SHY). The real Irish yield must win — otherwise a US ETF's trailing
        return would be reported under an ECB citation.
        """
        import agents.investment_agent as ia
        import utils.market_data_client as mdc
        import utils.product_data_client as pdc
        from utils.market_data_client import PriceQuote

        gov = next(p for p in IRISH_PRODUCT_CATALOGUE
                   if p["product_id"] == "GOV001")
        write_snapshot(snapshot_path, {gov["real_data_key"]: entry(2.41)})

        monkeypatch.setattr(settings.product_data, "use_real_product_data", True)
        monkeypatch.setattr(settings.market_data, "enabled", True)
        pdc.set_product_data_client(None)

        class FakeMarket:
            mode = "snapshot"

            def get_trailing_return_pct(self, ticker):
                return PriceQuote(ticker, 88.8, "2026-07-01", 365, "fake")

        monkeypatch.setattr(mdc, "market_data_client", FakeMarket(), raising=False)
        monkeypatch.setattr(mdc, "get_market_data_client",
                            lambda: FakeMarket(), raising=False)

        out = ia._apply_live_pricing(ia._apply_product_data(IRISH_PRODUCT_CATALOGUE))
        result = next(p for p in out if p["product_id"] == "GOV001")

        assert result["expected_return_pct"] == 2.41
        assert result["expected_return_source"].startswith("sourced:")

        pdc.set_product_data_client(None)

    def test_unsourced_products_still_get_the_proxy(self, monkeypatch):
        """The precedence guard must not disable proxy enrichment generally."""
        import agents.investment_agent as ia
        import utils.market_data_client as mdc
        from utils.market_data_client import PriceQuote

        monkeypatch.setattr(settings.product_data, "use_real_product_data", False)
        monkeypatch.setattr(settings.market_data, "enabled", True)

        class FakeMarket:
            mode = "snapshot"

            def get_trailing_return_pct(self, ticker):
                return PriceQuote(ticker, 88.8, "2026-07-01", 365, "fake")

        monkeypatch.setattr(mdc, "market_data_client", FakeMarket(), raising=False)

        out = ia._apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        etb = next(p for p in out if p["product_id"] == "ETB001")
        assert etb["expected_return_source"] == "live"



class TestNoMutation:

    def test_enrichment_never_mutates_the_module_catalogue(self, snapshot_path,
                                                           monkeypatch):
        write_snapshot(snapshot_path, {"anpost.deposit_account": entry(0.13)})
        monkeypatch.setattr(settings.product_data, "use_real_product_data", True)
        import utils.product_data_client as pdc
        pdc.set_product_data_client(None)

        before = json.dumps(IRISH_PRODUCT_CATALOGUE, sort_keys=True, default=str)
        _apply_product_data(IRISH_PRODUCT_CATALOGUE)
        after = json.dumps(IRISH_PRODUCT_CATALOGUE, sort_keys=True, default=str)
        assert before == after
        pdc.set_product_data_client(None)
