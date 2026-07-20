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
from pathlib import Path

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
        print(f"    {document_set} {by_set.get(document_set, 0)} chunks")
    print()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build (or inspect) th finadvise-mas RAG knowledge base."
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="Build the index in memory and print a summary, but dont write to disk"
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Load the currently persisted index (if any) and print a sumamry - no rebuild."
    )
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.stats_only:
        kb.ensure_built()
        _print_summary(kb)
        return
    
    print("Building RAG knowledge base over D1-D4...")
    print(f"    CBI Open Data Portal — live API attempt, seed fallback")
    print(f"    EU Digital Finance Platform — data/raw/eu_digital_finance/, seed fallback")
    print(f"    FinQA Original — data/raw/finqa_original_train.json, seed fallback")
    print(f"    FinQA Verified — data/raw/finqa_verified/, seed fallback₹")
    kb.rebuild()
    _print_summary(kb)

    if not args.no_persist:
        kb.persist()
        print(f"Index persisted to {settings.rag.index_dir}")
    else:
        print("--no-persist set - index not written to disk.")


if __name__ == "__main__":
    main()