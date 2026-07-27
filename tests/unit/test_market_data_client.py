"""
Phase 8.5 — live fund/ETF pricing (utils/market_data_client.py) and its
wiring into agents/investment_agent.py's TICKER_PROXY_MAP enrichment.

GROUP A: MarketDataClient — snapshot path, disabled-by-default path,
         unreachable/no-dependency fallback (all never raise)
GROUP A2: _fetch_live() — the one method that actually parses a yfinance
         response. Exercised against a mocked `yfinance.Ticker` (fake
         pandas DataFrame), not a stubbed-out client, so the real
         indexing/guard logic runs. This does NOT test the real yfinance
         API — only this method's handling of what yfinance returns.
GROUP B: _apply_live_pricing() — per-product enrichment, provenance
         fields, and the deliberately-unmapped products (SAV001, TDP001,
         VEN001) staying synthetic regardless of settings.market_data
GROUP C: InvestmentAgent — catalogue enrichment happens at construction,
         default (disabled) behaviour is byte-for-byte identical to
         Phases 1-8's synthetic values — this is the test that actually
         matters for not silently breaking RQ2/RQ3 reproducibility.

None of this reaches the real yfinance API or network — this sandbox has
no route to finance.yahoo.com, and a dissertation eval run shouldn't
depend on a live external service anyway. What IS and ISN'T covered:
  - Covered (mocked at the yfinance boundary): return computation,
    empty/short-history guards, zero-price guard, exception handling.
  - NOT covered: whether yfinance's real response shape still matches
    what these fakes assume (column names, exception types) — that can
    drift with a yfinance version bump. Verify once against the real API
    on a machine with network access before relying on this for the
    dissertation (see README "Verifying live pricing" note).

RUNNING:
    python -m pytest tests/unit/test_market_data_client.py -v
"""

from __future__ import annotations

import json
import sys

import pytest

from agents.investment_agent import (
    IRISH_PRODUCT_CATALOGUE,
    TICKER_PROXY_MAP,
    _apply_live_pricing,
)
from config.settings import settings
from utils.market_data_client import MarketDataClient, PriceQuote

# GROUP A: MarketDataClient

class TestMarketDataClient:

    def test_disabled_by_default(self):
        # The whole reproducibility story rests on this being False
        # out of the box — a dissertation eval run should never
        # accidentally pick up live data just because yfinance happens
        # to be installed.
        assert settings.market_data.enabled is False

    def test_get_trailing_return_returns_none_when_disabled_and_no_snapshot(self):
        client = MarketDataClient()
        assert client.get_trailing_return_pct("VT") is None

    def test_mode_reports_unavailable_when_disabled_and_no_snapshot(self):
        client = MarketDataClient()
        assert client.mode == "unavailable"

    def test_snapshot_path_used_when_present(self, tmp_path):
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps({
            "VT": {"trailing_return_pct": 12.34, "as_of": "2026-01-01T00:00:00Z", "period_days": 365}
        }))
        original_path = settings.market_data.snapshot_path
        settings.market_data.snapshot_path = snapshot_path
        try:
            client = MarketDataClient()
            quote = client.get_trailing_return_pct("VT")
            assert quote is not None
            assert quote.trailing_return_pct == 12.34
            assert quote.source == "snapshot"
            assert client.mode == "snapshot"
        finally:
            settings.market_data.snapshot_path = original_path

    def test_snapshot_missing_ticker_falls_through_to_none(self, tmp_path):
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text(json.dumps({"VT": {"trailing_return_pct": 12.34, "as_of": "x"}}))
        original_path = settings.market_data.snapshot_path
        settings.market_data.snapshot_path = snapshot_path
        try:
            client = MarketDataClient()
            # "BIL" not in this snapshot, and enabled=False -> no live fetch
            assert client.get_trailing_return_pct("BIL") is None
        finally:
            settings.market_data.snapshot_path = original_path

    def test_corrupt_snapshot_does_not_raise(self, tmp_path):
        snapshot_path = tmp_path / "snapshot.json"
        snapshot_path.write_text("{not valid json")
        original_path = settings.market_data.snapshot_path
        settings.market_data.snapshot_path = snapshot_path
        try:
            client = MarketDataClient()  # must not raise
            assert client.get_trailing_return_pct("VT") is None
        finally:
            settings.market_data.snapshot_path = original_path

    def test_write_snapshot_skips_unfetchable_tickers_without_raising(self, tmp_path):
        # enabled=False -> _fetch_live always returns None -> snapshot ends
        # up empty, but write_snapshot must still complete and write valid JSON.
        client = MarketDataClient()
        out_path = client.write_snapshot(["VT", "BND"], path=tmp_path / "out.json")
        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data == {}



# GROUP A2: _fetch_live() — the one method that actually talks to yfinance.



class _FakeTicker:
    """Mimics yfinance.Ticker's .history() surface for a fixed close-price series."""
    def __init__(self, closes: list[float] | None, raise_exc: Exception | None = None):
        self._closes = closes
        self._raise_exc = raise_exc

    def history(self, period=None, timeout=None):
        import pandas as pd

        if self._raise_exc:
            raise self._raise_exc
        if self._closes is None:
            return pd.DataFrame()  # empty — mimics an invalid/delisted ticker
        return pd.DataFrame({"Close": self._closes})


class _FakeYfinanceModule:
    def __init__(self, ticker_factory):
        self._ticker_factory = ticker_factory

    def Ticker(self, symbol):
        return self._ticker_factory(symbol)


@pytest.fixture
def install_fake_yfinance(monkeypatch):
    """Installs a fake `yfinance` module into sys.modules for the duration
    of the test, so `import yfinance as yf` inside _fetch_live() resolves
    to our fake rather than raising ImportError (or hitting the real
    package, if it's ever installed in this environment)."""
    def _install(ticker_factory):
        fake_module = _FakeYfinanceModule(ticker_factory)
        monkeypatch.setitem(sys.modules, "yfinance", fake_module)
        return fake_module
    return _install


class TestFetchLive:

    def test_computes_trailing_return_from_close_prices(self, install_fake_yfinance):
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=[100.0, 110.0]))
        client = MarketDataClient()
        quote = client._fetch_live("VT")
        assert quote is not None
        assert quote.trailing_return_pct == pytest.approx(10.0)
        assert quote.ticker == "VT"
        assert quote.source == "yfinance"

    def test_negative_return_computed_correctly(self, install_fake_yfinance):
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=[100.0, 92.5]))
        client = MarketDataClient()
        quote = client._fetch_live("VT")
        assert quote.trailing_return_pct == pytest.approx(-7.5)

    def test_empty_history_returns_none(self, install_fake_yfinance):
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=None))
        client = MarketDataClient()
        assert client._fetch_live("DELISTED") is None

    def test_single_row_history_returns_none(self, install_fake_yfinance):
        # len(hist) < 2 guard — can't compute a trailing return from one point
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=[100.0]))
        client = MarketDataClient()
        assert client._fetch_live("VT") is None

    def test_zero_start_price_returns_none_not_divide_by_zero(self, install_fake_yfinance):
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=[0.0, 5.0]))
        client = MarketDataClient()
        assert client._fetch_live("VT") is None

    def test_yfinance_exception_caught_and_returns_none(self, install_fake_yfinance):
        install_fake_yfinance(
            lambda symbol: _FakeTicker(closes=None, raise_exc=RuntimeError("rate limited"))
        )
        client = MarketDataClient()
        assert client._fetch_live("VT") is None  # must not propagate the exception

    def test_get_trailing_return_pct_uses_fetch_live_when_enabled(self, install_fake_yfinance, monkeypatch):
        install_fake_yfinance(lambda symbol: _FakeTicker(closes=[50.0, 55.0]))
        monkeypatch.setattr(settings.market_data, "enabled", True)
        client = MarketDataClient()
        client._yf_mode = "yfinance"  # bypass _init_yfinance's own import check
        quote = client.get_trailing_return_pct("VT")
        assert quote is not None
        assert quote.trailing_return_pct == pytest.approx(10.0)


# GROUP B: _apply_live_pricing()

class TestApplyLivePricing:

    def test_disabled_returns_catalogue_unchanged_in_value(self):
        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        original_by_id = {p["product_id"]: p["expected_return_pct"] for p in IRISH_PRODUCT_CATALOGUE}
        for product in enriched:
            assert product["expected_return_pct"] == original_by_id[product["product_id"]]

    def test_disabled_marks_every_product_synthetic(self):
        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        assert all(p["expected_return_source"] == "synthetic" for p in enriched)
        assert all(p["pricing_note"] is None for p in enriched)

    def test_does_not_mutate_module_level_catalogue(self):
        before = json.dumps(IRISH_PRODUCT_CATALOGUE, sort_keys=True, default=str)
        _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        after = json.dumps(IRISH_PRODUCT_CATALOGUE, sort_keys=True, default=str)
        assert before == after

    def test_deliberately_unmapped_products_have_no_proxy_entry(self):
        # SAV001/TDP001 (deposit rates aren't market-quoted) and VEN001
        # (illiquid private exposure) must never be enrichable — this is
        # a design guarantee, not just current behaviour.
        for product_id in ("SAV001", "TDP001", "VEN001"):
            assert product_id not in TICKER_PROXY_MAP

    def test_returns_same_number_of_products_as_input(self):
        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        assert len(enriched) == len(IRISH_PRODUCT_CATALOGUE)

    def test_enabled_with_stubbed_client_uses_live_value_and_note(self, monkeypatch):
        monkeypatch.setattr(settings.market_data, "enabled", True)

        class _StubClient:
            def get_trailing_return_pct(self, ticker):
                return PriceQuote(
                    ticker=ticker, trailing_return_pct=9.99,
                    as_of="2026-07-20T00:00:00Z", period_days=365, source="yfinance",
                )

        import agents.investment_agent as ia
        monkeypatch.setattr(ia, "market_data_client", _StubClient(), raising=False)
        # _apply_live_pricing imports market_data_client locally inside the
        # function, so patch the module it actually imports from instead.
        import utils.market_data_client as mdc
        monkeypatch.setattr(mdc, "market_data_client", _StubClient())

        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        etb001 = next(p for p in enriched if p["product_id"] == "ETB001")
        assert etb001["expected_return_source"] == "live"
        assert etb001["expected_return_pct"] == 9.99
        assert "VT" in etb001["pricing_note"]

    def test_enabled_missing_quote_falls_back_to_synthetic(self, monkeypatch):
        monkeypatch.setattr(settings.market_data, "enabled", True)

        class _StubClientNoData:
            def get_trailing_return_pct(self, ticker):
                return None

        import utils.market_data_client as mdc
        monkeypatch.setattr(mdc, "market_data_client", _StubClientNoData())

        original = next(p for p in IRISH_PRODUCT_CATALOGUE if p["product_id"] == "ETB001")
        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        etb001 = next(p for p in enriched if p["product_id"] == "ETB001")
        assert etb001["expected_return_source"] == "synthetic"
        assert etb001["expected_return_pct"] == original["expected_return_pct"]

    def test_composite_proxy_blends_by_weight(self, monkeypatch):
        monkeypatch.setattr(settings.market_data, "enabled", True)

        class _StubClient:
            RETURNS = {"VT": 10.0, "BND": 2.0}

            def get_trailing_return_pct(self, ticker):
                if ticker not in self.RETURNS:
                    return None
                return PriceQuote(
                    ticker=ticker, trailing_return_pct=self.RETURNS[ticker],
                    as_of="2026-07-20T00:00:00Z", period_days=365, source="yfinance",
                )

        import utils.market_data_client as mdc
        monkeypatch.setattr(mdc, "market_data_client", _StubClient())

        enriched = _apply_live_pricing(IRISH_PRODUCT_CATALOGUE)
        # MXM001: VT/BND blended 0.6/0.4 -> 0.6*10 + 0.4*2 = 6.8
        mxm001 = next(p for p in enriched if p["product_id"] == "MXM001")
        assert mxm001["expected_return_source"] == "live"
        assert mxm001["expected_return_pct"] == pytest.approx(6.8, abs=0.01)



# GROUP C: InvestmentAgent construction

class TestInvestmentAgentCatalogueEnrichment:

    def test_default_construction_matches_synthetic_catalogue(self):
        # This is the test that actually protects RQ2/RQ3 reproducibility:
        # constructing InvestmentAgent under default settings must produce
        # a catalogue identical (in expected_return_pct) to Phases 1-8.
        from agents.investment_agent import InvestmentAgent
        from utils.llm_client import LLMClient

        agent = InvestmentAgent(LLMClient())
        original_by_id = {p["product_id"]: p["expected_return_pct"] for p in IRISH_PRODUCT_CATALOGUE}
        for product in agent.catalogue:
            assert product["expected_return_pct"] == original_by_id[product["product_id"]]

    def test_default_construction_every_product_has_provenance_fields(self):
        from agents.investment_agent import InvestmentAgent
        from utils.llm_client import LLMClient

        agent = InvestmentAgent(LLMClient())
        for product in agent.catalogue:
            assert "expected_return_source" in product
            assert "pricing_note" in product
