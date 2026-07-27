"""
scripts/verify_market_data.py

Phase 8.5 — manual, one-off verification that utils/market_data_client.py's
assumptions about yfinance's response shape still hold against the REAL
API. Not part of the automated test suite (tests/unit/test_market_data_client.py
mocks yfinance.Ticker deliberately, so CI stays network-free and
deterministic) — this script is the thing that actually calls out to
Yahoo Finance.

Run this:
  - Once, after `pip install -r requirements.txt`, before trusting live
    pricing for anything in the dissertation
  - Again after any yfinance version bump — its response shape (column
    names, exception types) is not contractually stable, and a silent
    drift there would make MarketDataClient._fetch_live() start
    returning None for everything without raising an error anywhere.

Requires network access to query1/query2.finance.yahoo.com. In a
sandboxed environment (e.g. this one), that domain will need to be
added to the network egress allowlist first — otherwise every ticker
below will report BLOCKED, not FAIL, which is a network policy
difference from an actual code or data problem (see the last section
printed).

USAGE:
    python scripts/verify_market_data.py                  # default tickers
    python scripts/verify_market_data.py VT BND SHY        # specific tickers
    python scripts/verify_market_data.py --all-proxies     # every ticker
                                                             # referenced in
                                                             # TICKER_PROXY_MAP
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force-enable before importing settings-dependent modules, so
# MarketDataClient actually attempts the live path regardless of the
# environment's MARKET_DATA_ENABLED setting.
os.environ["MARKET_DATA_ENABLED"] = "true"

from config.settings import settings  # noqa: E402

DEFAULT_TICKERS = ["VT", "BND", "SHY", "IEF", "LQD", "HYG", "VNQ", "XLK", "DIA", "BIL"]

# Loose sanity bounds for a ~1yr trailing return — flags something that
# parsed successfully but looks structurally wrong (e.g. wrong column
# picked up, prices off by a stock split factor), not a hard assertion.
PLAUSIBLE_MIN_PCT = -80.0
PLAUSIBLE_MAX_PCT = 300.0


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.all_proxies:
        from agents.investment_agent import TICKER_PROXY_MAP
        return sorted({t for proxy in TICKER_PROXY_MAP.values() for t in proxy["tickers"]})
    if args.tickers:
        return args.tickers
    return DEFAULT_TICKERS


def _preflight_network_check() -> str | None:
    """
    Raw HTTP request to Yahoo Finance's API host, bypassing yfinance
    entirely. yfinance wraps a blocked/403 response into a generic
    JSONDecodeError internally (it tries to json.loads() the proxy's
    plaintext deny message), which makes the real cause indistinguishable
    from a genuine parsing failure once it reaches yfinance's exception.
    This check runs first so a network-policy block is reported as
    exactly that, not misattributed to yfinance or MarketDataClient.

    Returns an error string if blocked/unreachable, or None if the host
    is reachable (a real API error past this point is a genuine issue).
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        "https://query1.finance.yahoo.com/v8/finance/chart/VT",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return None
    except urllib.error.HTTPError as exc:
        deny_reason = exc.headers.get("x-deny-reason") if exc.headers else None
        if deny_reason == "host_not_allowed" or exc.code == 403:
            body = exc.read()[:200].decode("utf-8", errors="replace")
            return f"BLOCKED by network policy (HTTP {exc.code}): {body}"
        return None  # a non-403 HTTP error is a real API-level issue, not a block
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"BLOCKED or unreachable: {exc}"


def _check_raw_yfinance_shape(ticker: str) -> str | None:
    """
    Calls yfinance directly (bypassing MarketDataClient) to inspect the
    raw DataFrame shape MarketDataClient._fetch_live() assumes: a
    'Close' column, at least 2 rows. Returns an error string, or None if
    the shape looks as expected.
    """
    try:
        import yfinance as yf
    except ImportError:
        return "yfinance not installed — pip install -r requirements.txt"

    try:
        hist = yf.Ticker(ticker).history(
            period=f"{settings.market_data.period_days}d",
            timeout=settings.market_data.request_timeout_seconds,
        )
    except Exception as exc:
        # Distinguish a network-policy block from a genuine API/library error.
        msg = str(exc)
        if "not in allowlist" in msg or "host_not_allowed" in msg or "Forbidden" in msg:
            return f"BLOCKED by network policy: {msg}"
        return f"yfinance raised: {type(exc).__name__}: {exc}"

    if hist.empty:
        return "empty history returned — ticker may be delisted/invalid, or still blocked upstream"
    if "Close" not in hist.columns:
        return f"UNEXPECTED SHAPE: no 'Close' column — columns are {list(hist.columns)}"
    if len(hist) < 2:
        return f"UNEXPECTED SHAPE: only {len(hist)} row(s) returned for a {settings.market_data.period_days}d period"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify MarketDataClient against the real yfinance API."
    )
    parser.add_argument("tickers", nargs="*", help="Specific tickers to check (default: a fixed sample)")
    parser.add_argument(
        "--all-proxies", action="store_true",
        help="Check every ticker referenced in agents.investment_agent.TICKER_PROXY_MAP"
    )
    args = parser.parse_args()
    tickers = _resolve_tickers(args)

    # Import after MARKET_DATA_ENABLED is set, and after argparse so
    # --help doesn't pay the import cost.
    from utils.market_data_client import MarketDataClient

    print(f"MarketDataClient mode setting: enabled={settings.market_data.enabled}")
    print(f"Checking {len(tickers)} ticker(s): {', '.join(tickers)}\n")

    client = MarketDataClient()
    print(f"Client resolved mode: {client.mode}\n")

    network_blocked = _preflight_network_check()
    if network_blocked:
        print(f"Network preflight: {network_blocked}\n")

    results = []
    for ticker in tickers:
        if network_blocked:
            results.append((ticker, "BLOCKED", None, network_blocked))
            print(f"  [{'BLOCKED':28s}] {ticker:8s} —")
            continue

        shape_issue = _check_raw_yfinance_shape(ticker)
        quote = client.get_trailing_return_pct(ticker)

        if shape_issue:
            status = "SHAPE MISMATCH"
        elif quote is None:
            status = "FAIL (no quote, no shape issue detected — investigate)"
        elif not (PLAUSIBLE_MIN_PCT <= quote.trailing_return_pct <= PLAUSIBLE_MAX_PCT):
            status = f"IMPLAUSIBLE VALUE ({quote.trailing_return_pct}%)"
        else:
            status = "OK"

        results.append((ticker, status, quote, shape_issue))
        quote_str = f"{quote.trailing_return_pct:+.2f}% as of {quote.as_of}" if quote else "—"
        print(f"  [{status:28s}] {ticker:8s} {quote_str}")
        if shape_issue:
            print(f"      -> {shape_issue}")

    n_ok = sum(1 for _, status, _, _ in results if status == "OK")
    n_blocked = sum(1 for _, status, _, _ in results if status == "BLOCKED")
    n_other_fail = len(results) - n_ok - n_blocked

    print(f"\n{n_ok}/{len(results)} OK, {n_blocked} blocked by network policy, {n_other_fail} other failure(s)")

    if n_blocked == len(results):
        print(
            "\nEvery ticker was blocked at the network layer, not by yfinance or "
            "MarketDataClient — this environment's egress policy does not allow "
            "query1/query2.finance.yahoo.com. Add that host to the network "
            "settings for this environment and re-run; this is not a code issue."
        )
    elif n_other_fail > 0:
        print(
            "\nAt least one ticker failed for a reason other than network policy — "
            "see the '->' lines above. If this is a shape mismatch (missing "
            "'Close' column, unexpected row count), yfinance's response format "
            "has likely changed and utils/market_data_client.py._fetch_live() "
            "needs updating to match."
        )
        sys.exit(1)
    else:
        print("\nAll reachable tickers parsed correctly — MarketDataClient's "
              "assumptions about yfinance's response shape hold.")


if __name__ == "__main__":
    main()
