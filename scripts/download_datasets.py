"""
Usage:
    python scripts/download_datasets.py --phase <phase>

Dataset inventory (from literature review):
    [D1] FinQA Verified      -   (RAG + RQ5 hallucination eval)
    [D2] FinQA Unverified    -   (RQ5 FinQA exact match basseline)
    [D3] CBI Open Data       -   (RAG knowledge base, Irish rates)
    [D4] EU Digital Finance  -   (RAG knowledge base, EU macro)
    [D5] Banking77           -   (Conversational agent intent classification)
    [D6] MultiWOZ            -   (multi-turn dialogue trajectory reference)
    [D7] German Credit       -   (RiskProfilingAgent ML training)
    [D8] GiveMeSomeCredit    -   (RiskProfilingAgent ML training)
    [D9] Bank Marketing      -   (supplementary risk features)
    [D10] Personal Finance   -   (BudgetAgent income/expenditure)
    [D11] Ireland HBS        -   (BudgetAgent Irish spending benchmarks) 
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

def phase1_check() -> None:
    """
    Phase 1: No dataset needed to be downloaded.
    Verify the data directory structure is correct.
    """
    print(f"[Phase 1] Checking data directory structure...")
    for subdir in ("raw", "processed", "embeddings"):
        path = ROOT / "data" / subdir
        path.mkdir(parents=True, exist_ok=True)
        print(f" OK data/{subdir}/")
    print("[Phase 1] Structure check complete. No downloads required yet\n.")


PHASE_FUNCTIONS = {
    1: [phase1_check],
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for the project - gated by build phase."
        )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--phase",
        type=int,
        choices=range(1, 9),
        help="Download all datasets required up to and including this phase."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets."
    )
    args = parser.parse_args()

    phase = 8 if args.all else args.phase
    print(f"Dataset downloader - up to Phase {phase}\n")
    functions = PHASE_FUNCTIONS.get(phase, [phase1_check])
    seen = set()
    unique_fns = []
    for fn in functions:
        if fn not in seen:
            seen.add(fn)
            unique_fns.append(fn)

    for fn in unique_fns:
        fn()

    print("Download complete. Check messages above for any FAIL or manual steps.\n")

if __name__ == "__main__":
    main()