#!/usr/bin/env python3
"""
scripts/fetch_product_data.py — refresh data/raw/product_catalogue/product_snapshot.json

Two kinds of entry live in that snapshot, and the difference is not a detail:

  FETCHABLE   ECB Statistical Data Warehouse series. A real REST API, no key.
              This script updates them automatically.

  MANUAL      An Post State Savings and CBI retail interest rate statistics.
              Published, authoritative, and available only as HTML pages and
              PDFs. This script will NOT scrape them. It prints the source URL
              and what to fill in, and you transcribe the figure along with
              the date you read it.

WHY NOT JUST SCRAPE THE MANUAL ONES
    A scraper against a marketing page is a silent-failure machine: the layout
    changes, the selector matches the wrong number, and a plausible rate lands
    in a dissertation results table with an authoritative-looking citation
    attached. Transcribing a handful of numbers by hand once a quarter is
    slower and far more defensible. It also forces a human to notice the An
    Post total-return-vs-annual-rate trap, which no selector would catch.

USAGE
    python scripts/fetch_product_data.py              # dry run — show what would change
    python scripts/fetch_product_data.py --write      # fetch and write
    python scripts/fetch_product_data.py --check      # staleness report, exit 1 if stale
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from utils.product_data_client import ECB_BASE, ECB_SERIES  # noqa: E402

BAR = "=" * 76


def fetch_ecb(series: str) -> tuple[float, str] | None:
    """Return (value, as_of) for one ECB series, or None. Never raises."""
    try:
        import requests
    except ImportError:
        print("    requests is not installed — pip install requests")
        return None

    url = f"{ECB_BASE}/{series}"
    try:
        resp = requests.get(
            url,
            params={"lastNObservations": "1", "format": "jsondata"},
            timeout=settings.product_data.timeout_seconds,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        payload = resp.json()
        obs = next(iter(payload["dataSets"][0]["series"].values()))["observations"]
        idx, val = next(iter(sorted(obs.items(), key=lambda kv: int(kv[0]))))
        periods = payload["structure"]["dimensions"]["observation"][0]["values"]
        as_of = periods[int(idx)]["id"] if int(idx) < len(periods) else periods[-1]["id"]
        # ECB publishes monthly periods as "2026-06". Pad to a full date so the
        # staleness check can parse it with date.fromisoformat.
        if len(as_of) == 7:
            as_of = f"{as_of}-01"
        return float(val[0]), as_of
    except Exception as exc:
        print(f"    fetch failed: {type(exc).__name__}: {exc}")
        return None


def age_days(as_of: str | None) -> int | None:
    if not as_of:
        return None
    try:
        return (datetime.now(timezone.utc).date() - date.fromisoformat(as_of[:10])).days
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--write", action="store_true", help="Actually write the snapshot")
    ap.add_argument("--check", action="store_true",
                    help="Report staleness only; exit 1 if anything is stale or empty")
    args = ap.parse_args()

    path = settings.product_data.snapshot_path
    if not path.exists():
        print(f"No snapshot at {path}")
        return 1

    doc = json.loads(path.read_text(encoding="utf-8"))
    entries = doc["entries"]
    max_age = settings.product_data.max_age_days

    print(f"\n{BAR}\n CURRENT SNAPSHOT  ({path.relative_to(ROOT)})\n{BAR}")
    stale: list[str] = []
    empty: list[str] = []
    for key, entry in entries.items():
        value, as_of = entry.get("value"), entry.get("as_of")
        days = age_days(as_of)
        if value is None:
            state = "EMPTY  -> product stays synthetic"
            empty.append(key)
        elif days is None:
            state = "BAD DATE -> refused"
            stale.append(key)
        elif days > max_age:
            state = f"STALE ({days}d > {max_age}d) -> refused"
            stale.append(key)
        else:
            state = f"ok ({days}d old)"
        shown = "     -" if value is None else f"{value:>6.2f}%"
        print(f"  {key:<42} {shown}  {state}")

    if args.check:
        usable = len(entries) - len(empty) - len(stale)
        print(f"\n  {len(empty)} empty, {len(stale)} stale, {usable} usable")
        return 1 if (empty or stale) else 0

    print(f"\n{BAR}\n FETCHABLE (ECB Statistical Data Warehouse)\n{BAR}")
    updated = 0
    for key, entry in entries.items():
        if not entry.get("fetchable"):
            continue
        spec = ECB_SERIES.get(key)
        if not spec or not spec["series"]:
            continue
        print(f"\n  {key}\n    {spec['series']} - {spec['label']}")
        result = fetch_ecb(spec["series"])
        if result is None:
            continue
        value, as_of = result
        print(f"    -> {value:.2f}%  as of {as_of}")
        if args.write:
            entry["value"] = round(value, 2)
            entry["as_of"] = as_of
            updated = 1

    manual = [(k, e) for k, e in entries.items() if not e.get("fetchable")]
    print(f"\n{BAR}\n MANUAL - transcribe these yourself\n{BAR}")
    print("  No API exists for these. Open each URL, read the current figure,")
    print("  and set both 'value' and 'as_of' (the date YOU read it).\n")
    for key, entry in manual:
        print(f"  {key}")
        print(f"    {entry.get('source_url', '(no url)')}")
        note = entry.get("source", "")
        if "NOTE:" in note:
            print(f"    !  {note[note.index('NOTE:'):]}")
        print()

    if args.write and updated:
        doc["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"{BAR}\n  wrote {updated} fetched value(s) -> {path.relative_to(ROOT)}\n{BAR}")
    elif args.write:
        print(f"{BAR}\n  nothing fetched - snapshot unchanged\n{BAR}")
    else:
        print(f"{BAR}\n  dry run - nothing written. Re-run with --write.\n{BAR}")

    print("\n  Then enable enrichment for an evaluation run:")
    print("    USE_REAL_PRODUCT_DATA=true python scripts/regenerate_evidence.py --only rq2\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())