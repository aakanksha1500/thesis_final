"""
utils/product_data_client.py — real product reference data, where it exists.


WHAT THIS IS FOR

MarketDataClient already replaces one field — expected_return_pct — for the
products in TICKER_PROXY_MAP, using a yfinance trailing return. Everything
else about every product (that it exists at all, its name, provider, expense
ratio, category, liquidity) has always been synthetic.

This client extends real sourcing to a second field, for the two categories
where genuinely authoritative Irish figures are publicly available for free:

    government_bond   ECB Statistical Data Warehouse — Irish and euro-area
                      sovereign yields. A real REST API, no key, no licence
                      barrier, and the series the ECB itself publishes for
                      convergence reporting.

    savings_account   An Post State Savings and CBI retail interest rate
    term_deposit      statistics. Real, published, and stable enough that a
                      periodically refreshed snapshot is honest.


WHY ONLY THOSE TWO

The realistic alternative was to claim more coverage than can be delivered.
Expense ratios and risk indicators for UCITS funds live in PRIIPs KID
documents — PDFs, per-fund, no free API — or behind Morningstar/FE licences.
Scraping fund factsheets to populate a dissertation results table would make
the numbers less defensible, not more, because nothing would record which
figure came from which revision of which PDF.

So: two categories genuinely real end-to-end, every other product explicitly
and traceably illustrative. That is a stronger claim than fifteen products of
uncertain provenance.


THE THREE TIERS  (identical in shape to MarketDataClient, deliberately)

    1. live       fetch from the source API, if enabled and reachable
    2. snapshot   frozen JSON on disk — the reproducible-evaluation path
    3. None       caller keeps the catalogue's synthetic value

Never raises. Every fallback is logged at WARNING or above, because the
failure this codebase keeps producing is not a crash — it is a plausible
number with no indication of where it came from.


STALENESS

A deposit rate from eighteen months ago is not "real data", it is a stale
number wearing a provenance stamp. Snapshot entries older than
settings.product_data.max_age_days are refused, not warned about — refusing
degrades to the synthetic tier, which is honestly labelled, whereas warning
leaves a stale figure in the results file with a "real" source stamp on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProductFact:
    """
    One sourced field for one product.

    `source` is a human-readable citation, not a category label — it goes
    into the audit trail and, via ExplainabilityAgent, potentially in front of
    a user. "ECB SDW IRS.M.IE.L.L40.CI.0000.EUR.N.Z, as of 2026-06-30" is a
    claim someone can check. "live" is not.
    """
    field: str
    value: float
    as_of: str
    source: str
    source_url: str = ""
    tier: str = "snapshot"          # "live" | "snapshot"


# ECB Statistical Data Warehouse series. Documented here rather than inlined
# because the series key IS the provenance — anyone can paste it into
# data.ecb.europa.eu and get the same number back.
ECB_SERIES: dict[str, dict[str, str]] = {
    "ecb.ie_govt_yield_10y": {
        "series": "IRS/M.IE.L.L40.CI.0000.EUR.N.Z",
        "label": "Ireland, 10-year government bond yield (convergence criterion)",
    },
    "ecb.ea_govt_yield_10y": {
        "series": "IRS/M.U2.L.L40.CI.0000.EUR.N.Z",
        "label": "Euro area, 10-year government bond yield (convergence criterion)",
    },
    # Short-dated Irish sovereign has no equivalent convergence series. Rather
    # than substitute the 10-year and call it short-dated — which would be a
    # fabrication with a real-looking citation attached — this key resolves
    # only from the snapshot, where a curator has to state what they used.
    "ecb.ie_govt_yield_short": {
        "series": "",
        "label": "Ireland, short-dated sovereign yield (snapshot-only — no ECB convergence series exists)",
    },
}

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"


class ProductDataClient:
    """Reference data for catalogue products. Same tiering as MarketDataClient."""

    def __init__(self) -> None:
        self._snapshot: dict[str, Any] = {}
        self._snapshot_meta: dict[str, Any] = {}
        self._live_mode = "disabled"
        self._load_snapshot()
        if settings.product_data.enabled:
            self._probe_live()

    # snapshot 
    def _load_snapshot(self) -> None:
        path = settings.product_data.snapshot_path
        if not path.exists():
            logger.info(
                f"[ProductDataClient] no snapshot at {path} — every product "
                f"keeps its synthetic figures. Create one with "
                f"scripts/fetch_product_data.py"
            )
            return
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            self._snapshot = doc.get("entries", {})
            self._snapshot_meta = doc.get("_meta", {})
            populated = sum(
                1 for e in self._snapshot.values() if e.get("value") is not None
            )
            logger.info(
                f"[ProductDataClient] loaded snapshot from {path} — "
                f"{populated}/{len(self._snapshot)} entries populated"
            )
            if populated == 0:
                logger.warning(
                    "[ProductDataClient] the snapshot exists but every entry is "
                    "empty. This is the shipped template — fill it in from the "
                    "source URLs, or run scripts/fetch_product_data.py. Until "
                    "then every figure remains synthetic."
                )
        except Exception as exc:
            logger.warning(f"[ProductDataClient] failed to read {path}: {exc}")

    def _probe_live(self) -> None:
        try:
            import requests  # noqa: F401, PLC0415

            self._live_mode = "ecb_sdw"
            logger.info("[ProductDataClient] requests available — live ECB fetch enabled")
        except ImportError:
            logger.warning(
                "[ProductDataClient] PRODUCT_DATA_ENABLED=true but `requests` is "
                "not installed — falling back to the snapshot. "
                "pip install requests to enable live fetching."
            )

    @property
    def mode(self) -> str:
        """Reported into results/_meta.subsystem_modes, like every other client."""
        if self._live_mode == "ecb_sdw":
            return "livesnapshot" if self._snapshot else "live"
        if self._snapshot:
            return "snapshot"
        return "synthetic"

    # staleness 
    def _is_fresh(self, as_of: str | None) -> bool:
        if not as_of:
            return False
        try:
            observed = date.fromisoformat(as_of[:10])
        except ValueError:
            return False
        age = (datetime.now(timezone.utc).date() - observed).days
        return age <= settings.product_data.max_age_days

    # the one public method 
    def get_fact(self, real_data_key: str, field: str = "expected_return_pct") -> ProductFact | None:
        """
        Resolve one sourced field, or None. Never raises.

        None is a completely normal outcome — it means "no real figure is
        available for this product", and the caller keeps its synthetic value
        with a synthetic provenance stamp. That is the designed behaviour for
        the twenty-odd catalogue entries that have no free authoritative
        source, not an error path.
        """
        if not real_data_key:
            return None

        if self._live_mode == "ecb_sdw" and real_data_key in ECB_SERIES:
            fact = self._fetch_ecb(real_data_key, field)
            if fact is not None:
                return fact
            logger.info(
                f"[ProductDataClient] live fetch failed for {real_data_key} — "
                f"trying snapshot"
            )

        entry = self._snapshot.get(real_data_key)
        if not entry or entry.get("value") is None:
            return None

        if entry.get("field", "expected_return_pct") != field:
            return None

        if not self._is_fresh(entry.get("as_of")):
            logger.warning(
                f"[ProductDataClient] snapshot entry {real_data_key!r} is dated "
                f"{entry.get('as_of')}, older than "
                f"{settings.product_data.max_age_days} days — REFUSING it and "
                f"falling back to the synthetic value. A stale rate carrying a "
                f"'real' provenance stamp is worse than an honest illustrative "
                f"one. Refresh with scripts/fetch_product_data.py"
            )
            return None

        try:
            return ProductFact(
                field=field,
                value=float(entry["value"]),
                as_of=entry["as_of"],
                source=entry.get("source", "snapshot"),
                source_url=entry.get("source_url", ""),
                tier="snapshot",
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                f"[ProductDataClient] snapshot entry {real_data_key!r} is malformed: {exc}"
            )
            return None

    # live 
    def _fetch_ecb(self, real_data_key: str, field: str) -> ProductFact | None:
        """
        One observation from the ECB Statistical Data Warehouse.

        Yields are published as percentages already, so no unit conversion —
        stated explicitly because a silent x100 here would put a 400% expected
        return into a shortlist, which R001 would then hard-block, and the
        turn would fail for a reason nobody could trace back to this line.
        """
        spec = ECB_SERIES.get(real_data_key)
        if not spec or not spec["series"]:
            return None
        if field != "expected_return_pct":
            return None

        url = f"{ECB_BASE}/{spec['series']}"
        try:
            import requests  # noqa: PLC0415

            resp = requests.get(
                url,
                params={"lastNObservations": "1", "format": "jsondata"},
                timeout=settings.product_data.timeout_seconds,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            payload = resp.json()

            series = payload["dataSets"][0]["series"]
            observations = next(iter(series.values()))["observations"]
            idx, value = next(iter(sorted(observations.items(), key=lambda kv: int(kv[0]))))
            value = float(value[0])

            periods = (payload["structure"]["dimensions"]["observation"][0]["values"])
            as_of = periods[int(idx)]["id"] if int(idx) < len(periods) else periods[-1]["id"]

            return ProductFact(
                field=field,
                value=round(value, 2),
                as_of=as_of,
                source=f"ECB SDW {spec['series']} — {spec['label']}",
                source_url=f"{url}?lastNObservations=1",
                tier="live",
            )
        except Exception as exc:
            logger.warning(
                f"[ProductDataClient] ECB fetch failed for {real_data_key} "
                f"({type(exc).__name__}: {exc})"
            )
            return None


# lazy singleton, matching the R14 pattern used everywhere else 
_product_data_client: "ProductDataClient | None" = None


def get_product_data_client() -> "ProductDataClient":
    global _product_data_client
    if _product_data_client is None:
        _product_data_client = ProductDataClient()
    return _product_data_client


def set_product_data_client(instance: "ProductDataClient | None") -> None:
    """Inject a substitute (or None to reset). For tests and ablations."""
    global _product_data_client
    _product_data_client = instance


def __getattr__(name: str):
    if name == "product_data_client":
        return get_product_data_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
 