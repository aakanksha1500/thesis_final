"""
Phase 8c - live fund/ETF pricing, replacing IRISH_PRODUCT_CATALOGUE's syenthtic 
    expected_return_pct where a real market proxy ticker exists.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PriceQuote:
    ticker: str
    trailing_return_pct: float
    as_of: str
    period_days: int
    source: str

class MarketDataClient:
    """
    Fetches a trailing annualised return for a ticker. Three data paths,
    tried in order:
        1. Frozen snapshot at settings.market_data.snapshot_path, if present - 
            the reproducible-evaluation path.
        2. Live yfinance fetch, if settings.market_data.enabled and yfinance
            is installed.
        3. None - caller keeps the synthetic catalogue value.
        
    An in memory TTL cache sits in front of (2) so a single process
    doesn't refetch mid-run and drift.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, PriceQuote]] = {}
        self._snapshot: dict[str, Any] | None = None
        self._yf_mode = "unavailable"
        self._load_snapshot()
        if settings.market_data.enabled:
            self._init_yfinance()

    def _load_snapshot(self) -> None:
        path = settings.market_data.snapshot_path
        if path.exists():
            try:
                with open(path) as f:
                    self._snapshot = json.load(f)
                logger.info(f"[MarketDataClient] loaded price snapshot from {path}")
            except Exception as exc:
                logger.warning(f"[MarketDataClient] Failed to load snapshot {path}: {exc}")

    def _init_yfinance(self) -> None:
        try:
            import yfinance

            self._yf_mode = "yfinance"
            logger.info("[MarketDataClient] yfinance available — live pricing enabled")
        except ImportError:
            logger.warning(
                "[MarketDataClient] settings.market_data.enabled=True but "
                "yfinance is not installed — add yfinance to requirements.txt "
                "and pip install. Falling back to synthetic catalogue values."
            )

    @property
    def mode(self) -> str:
        if self._snapshot:
            return "snapshot"
        return self._yf_mode

    def get_trailing_return_pct(self, ticker: str) -> PriceQuote | None:
        """
        Return a PriceQuote for `ticker`, or None if unavailable — snapshot
        first, then live fetch (if enabled), then give up. Never raises.
        """
        if self._snapshot and ticker in self._snapshot:
            entry = self._snapshot[ticker]
            return PriceQuote(
                ticker=ticker,
                trailing_return_pct=entry["trailing_return_pct"],
                as_of=entry["as_of"],
                period_days=entry.get("period_days", settings.market_data.period_days),
                source="snapshot",
            )

        if not settings.market_data.enabled or self._yf_mode != "yfinance":
            return None

        cached = self._cache.get(ticker)
        if cached and (time.time() - cached[0]) < settings.market_data.cache_ttl_seconds:
            return cached[1]

        quote = self._fetch_live(ticker)
        if quote:
            self._cache[ticker] = (time.time(), quote)
        return quote

    def _fetch_live(self, ticker: str) -> PriceQuote | None:
        try:
            import yfinance as yf  # noqa: PLC0415

            t = yf.Ticker(ticker)
            hist = t.history(
                period=f"{settings.market_data.period_days}d",
                timeout=settings.market_data.request_timeout_seconds,
            )
            if hist.empty or len(hist) < 2:
                logger.warning(f"[MarketDataClient] No price history for '{ticker}'")
                return None

            start_price = float(hist["Close"].iloc[0])
            end_price = float(hist["Close"].iloc[-1])
            if start_price <= 0:
                return None

            trailing_return_pct = round((end_price - start_price) / start_price * 100, 2)
            return PriceQuote(
                ticker=ticker,
                trailing_return_pct=trailing_return_pct,
                as_of=datetime.now(timezone.utc).isoformat(),
                period_days=settings.market_data.period_days,
                source="yfinance",
            )
        except Exception as exc:
            logger.warning(f"[MarketDataClient] Live fetch failed for '{ticker}': {exc}")
            return None


    def write_snapshot(self, tickers: list[str], path: Path | None = None) -> Path:
        """
        Fetch live quotes for `tickers` and persist them to `path` (defaults
        to settings.market_data.snapshot_path) — the reproducible-evaluation
        workflow described in the module docstring.
        """
        path = path or settings.market_data.snapshot_path
        snapshot: dict[str, Any] = {}
        for ticker in tickers:
            quote = self._fetch_live(ticker)
            if quote:
                snapshot[ticker] = {
                    "trailing_return_pct": quote.trailing_return_pct,
                    "as_of": quote.as_of,
                    "period_days": quote.period_days,
                }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        logger.info(f"[MarketDataClient] Wrote snapshot for {len(snapshot)}/{len(tickers)} tickers to {path}")
        return path


market_data_client = MarketDataClient()
