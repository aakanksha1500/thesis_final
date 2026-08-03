#!/usr/bin/env python3
"""
diagnose_retrieval.py — is BROKEN retrieval a corpus problem or a ranking one?

verify_real_pipeline reports the top hit for a deposit-guarantee query as an
off-topic FinQA chunk. Two very different causes produce that symptom, and they
need opposite fixes:

  CORPUS   the Irish regulatory text is not in the index at all, so no ranking
           change can help. Fix the download / build.

  RANKING  the text IS there but 4,360 FinQA chunks out-score 15 regulatory
           ones by chance similarity. Fix retrieval scoping.

This distinguishes them. Read-only — changes nothing.

    python diagnose_retrieval.py
"""
from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
logging.disable(logging.INFO)

QUERIES = [
    "how much of my savings is protected if the bank fails",
    "deposit guarantee scheme protection limit",
    "what disclosures must an investment firm give a retail client",
]

REGULATORY = ("regulatory", "cbi_open_data", "eu_digital_finance")
BAR = "=" * 78


def main() -> int:
    from config.settings import settings
    from rag.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    kb.ensure_built()
    docs = kb.store._documents

    print(f"\n{BAR}\n CORPUS COMPOSITION\n{BAR}")
    counts = collections.Counter(d.document_set for d in docs)
    total = sum(counts.values())
    for name, n in counts.most_common():
        print(f"  {name:<24} {n:>6}  {n / total:>6.1%}")
    reg_n = sum(counts[s] for s in REGULATORY)
    print(f"\n  regulatory (D3+D4)       {reg_n:>6}  {reg_n / total:>6.1%}")
    print(f"  min_relevance_score      {settings.rag.min_relevance_score}")
    print(f"  top_k_citations          {settings.rag.top_k_citations}")

    print(f"\n{BAR}\n DOES THE CONTENT EXIST AT ALL?\n{BAR}")
    needles = ["deposit guarantee", "100,000", "compensation scheme"]
    found = [d for d in docs if any(n.lower() in d.text.lower() for n in needles)]
    print(f"  {len(found)} chunk(s) mention {needles}")
    for d in found[:5]:
        print(f"    [{d.document_set}] {d.text[:88]}")
    if not found:
        print("    -> CORPUS PROBLEM. The content is absent; ranking cannot fix it.")

    print(f"\n{BAR}\n RETRIEVAL, UNFILTERED vs REGULATORY-ONLY\n{BAR}")
    verdicts = []
    for q in QUERIES:
        print(f"\n  query: {q!r}")

        unfiltered = kb.retrieve(q, top_k=5)
        print("    unfiltered:")
        if not unfiltered:
            print("      (nothing above min_relevance_score)")
        for h in unfiltered:
            print(f"      {h['relevance']:.3f}  {h['document_set']:<20} {h['text'][:58]}")

        scoped = kb.retrieve(q, top_k=5, document_sets=list(REGULATORY))
        print("    regulatory-only:")
        if not scoped:
            print("      (no regulatory chunk above min_relevance_score)")
        for h in scoped:
            print(f"      {h['relevance']:.3f}  {h['document_set']:<20} {h['text'][:58]}")

        top_is_reg = bool(unfiltered) and unfiltered[0]["document_set"] in REGULATORY
        verdicts.append((q, top_is_reg, bool(scoped),
                         scoped[0]["relevance"] if scoped else None,
                         unfiltered[0]["relevance"] if unfiltered else None))

    print(f"\n{BAR}\n VERDICT\n{BAR}")
    any_reg = any(v[2] for v in verdicts)
    any_top = any(v[1] for v in verdicts)

    for q, top_is_reg, reg_exists, reg_score, top_score in verdicts:
        if not reg_exists:
            state = "no regulatory hit at all"
        elif top_is_reg:
            state = "regulatory hit ranks first — fine"
        else:
            state = (f"regulatory hit exists ({reg_score:.3f}) but loses to "
                     f"{top_score:.3f}")
        print(f"  {q[:52]:<54} {state}")

    print()
    if not any_reg:
        print("  CORPUS PROBLEM.")
        print("  The regulatory corpora are effectively absent from retrieval.")
        print("  D4 (EU) is currently SYNTHETIC seed paragraphs and D3 (CBI) is")
        print("  11 chunks. Fix the source data before measuring RQ5:")
        print("      python scripts/download_datasets.py")
        print("      rm -rf data/embeddings/rag_index && python scripts/build_knowledge_base.py")
    elif not any_top:
        print("  RANKING PROBLEM.")
        print("  The regulatory text IS retrievable but is out-scored by FinQA,")
        print("  which is ~98% of the corpus. retrieve() already supports")
        print("  document_sets — the callers do not use it. Scoping compliance")
        print("  queries to D3+D4 is a small change and defensible: a financial")
        print("  QA dataset is not a regulatory source and should not be cited")
        print("  as one.")
    else:
        print("  MIXED — some queries rank regulatory text first, some do not.")
        print("  Scoping by document_set is still the right fix; the corpus")
        print("  imbalance makes unfiltered retrieval unreliable rather than")
        print("  uniformly wrong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
