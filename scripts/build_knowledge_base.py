"""
Phase 8 - builds and persists the RAG vector store index

Referenced from scripts/download_datasets.py's phase8_downloads() docstring
as the automated fetch setup for CBI open data. Run this AFTER
'python scripts/download_datasets.py --phase 8' so anu manually-downloaded
EU Digital Financial CSVs and FINQA splits are picked up too - set if the
corresponding raw data isn't present, so this script always produces a usable index.

Usage:
    python scripts/build_knowledge_base.py                # build + persist
    python scripts/build_knowledge_base.py --no-persist    # build, print
                                                            # summary, don't
                                                            # write to disk
    python scripts/build_knowledge_base.py --stats-only    # just report
                                                            # the currently
                                                            # persisted index
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import settings
from rag.knowledge_base import KnowledgeBase


def _print_summary(kb: KnowledgeBase) -> None:
    by_set = Counter(doc.document_set for doc in kb.store._documents)
    print("\n=== Knowledge base summary ===")
    print(f"    Embedder mode: {kb.embedder.mode}")
    print(f"    Store backend: {kb.store._backend}")
    print(f"    Total chunks: {len(kb.store)}")
    for document_set in settings.rag.document_sets:
        print(f"    {document_set:22s} {by_set.get(document_set, 0)} chunks")
    print()

def _refresh_market_snapshot() -> None:
    """
    fetch live prices for every ticker in TICKER_PROXY_MAP and
    write them to settings.market_data.snapshot_path, so subsequent
    evaluation runs read from the frozen snapshot rather than fetching
    live (reproducibility — see utils/market_data_client.py docstring).
    """
    from agents.investment_agent import TICKER_PROXY_MAP  # noqa: E402
    from utils.market_data_client import market_data_client  # noqa: E402

    tickers = sorted({t for proxy in TICKER_PROXY_MAP.values() for t in proxy["tickers"]})
    print(f"Fetching live prices for {len(tickers)} tickers: {', '.join(tickers)}")
    if not settings.market_data.enabled:
        print(
            "WARNING: settings.market_data.enabled is False "
            "(MARKET_DATA_ENABLED not set) — fetch will likely return no "
            "quotes. Set MARKET_DATA_ENABLED=true and re-run."
        )
    path = market_data_client.write_snapshot(tickers)
    print(f"Snapshot written to {path}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (or inspect) the finadvise-mas RAG knowledge base."
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Build the index in memory and print a summary, but dont write to disk"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Load the currently persisted index (if any) and print a summary - no rebuild."
        )
    parser.add_argument(
        "--refresh-market-snapshot", action="store_true",
        help="Fetch live prices for every ticker in "
             "agents.investment_agent.TICKER_PROXY_MAP via yfinance and write "
             "them to settings.market_data.snapshot_path (Phase 8.5). "
             "Independent of the RAG index build — requires "
             "MARKET_DATA_ENABLED=true and yfinance installed; does not "
             "require --no-persist/--stats-only and can be combined with "
             "either, but the KB build still runs unless --stats-only is set."
    )
    args = parser.parse_args()

    if args.refresh_market_snapshot:
        _refresh_market_snapshot()
        if args.stats_only:
            return

    kb = KnowledgeBase()

    if args.stats_only:
        kb.ensure_built()
        _print_summary(kb)
        return

    print("Building RAG knowledge base over D1-D4...")
    print("    CBI Open Data Portal — live API attempt, seed fallback")
    print("    EU Digital Finance Platform — data/raw/eu_digital_finance/, seed fallback")
    print("    FinQA Original — data/raw/finqa_original_train.json, seed fallback")
    print("    FinQA Verified — data/raw/finqa_verified/, seed fallback₹")
    kb.rebuild()
    _print_summary(kb)

    if not args.no_persist:
        kb.persist()
        print(f"Index persisted to {settings.rag.index_dir}")
    else:
        print("--no-persist set - index not written to disk.")


if __name__ == "__main__":
    main()