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

def phase2_downloads() -> None:
    """
    Phase 2: ConversationalAgent.
    Downloads Banking77 via HuggingFace datasets library.
    MultiWOZ is used as a reference architecture - not trained on directly.
    """
    print("[Phase 2] Downloading Banking77 - intent classification dataset...")
    try:
        from datasets import load_dataset   
        ds = load_dataset("mteb/banking77", split="train")
        save_path = DATA_RAW / "banking77"
        ds.save_to_disk(str(save_path))
        print(f" OK Banking77 saved to {save_path} ({len(ds)} samples)")
    except ImportError:
        print(" SKIP 'datasets' package not installed. "
              "Add datasets>=2.19.0 to requirements.txt and re-run.")
    except Exception as e:
        print(f" FAIL Banking77 download failed: {e}")
    
    print("[Phase 2] - used as a dialogue reference architecture.")
    print()

def phase3_downloads() -> None:
    """
    Phase 3 — RiskProfilingAgent training data.
    German Credit (D7), GiveMeSomeCredit (D8), Bank Marketing (D9).
    """
    print("[Phase 3] Downloading risk profiling datasets...")

    # D7 — German Credit Data (UCI ML Repository)
    print("  [D7] German Credit Data...")
    try:
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        dest = DATA_RAW / "german_credit.data"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
            print(f"  OK  German Credit saved to {dest}")
        else:
            print(f"  SKIP  {dest} already exists")
    except Exception as e:
        print(f"  FAIL  German Credit: {e}")

    # D9 — Bank Marketing (UCI ML Repository)
    print("  [D9] Bank Marketing Dataset...")
    try:
        import urllib.request
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        dest = DATA_RAW / "bank.zip"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
            import zipfile
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(DATA_RAW / "bank_marketing")
            print(f"  OK  Bank Marketing extracted to data/raw/bank_marketing/")
        else:
            print(f"  SKIP  {dest} already exists")
    except Exception as e:
        print(f"  FAIL  Bank Marketing: {e}")

    # D8 — GiveMeSomeCredit (Kaggle — requires manual download)
    print("  [D8] GiveMeSomeCredit — requires Kaggle account.")
    print("       Manual download: https://www.kaggle.com/c/GiveMeSomeCredit")
    print("       Place cs-training.csv in data/raw/give_me_some_credit/")
    print()

def phase5_downloads() -> None:
    """
    Phase 5 — BudgetAgent data.
    Personal Finance Dataset (D10) via HuggingFace.
    Ireland HBS (D11) — manual download from data.gov.ie.
    """
    print("[Phase 5] Downloading budget agent datasets...")

    # D10 — Personal Finance Dataset
    print("  [D10] Personal Finance Dataset (HuggingFace)...")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("mihirinamdar/personal-finance-dataset", split="train")
        save_path = DATA_RAW / "personal_finance"
        ds.save_to_disk(str(save_path))
        print(f"  OK  Personal Finance saved ({len(ds)} examples)")
    except ImportError:
        print("  SKIP  'datasets' package not installed.")
    except Exception as e:
        print(f"  FAIL  Personal Finance: {e}")

    # D11 — Ireland Household Budget Survey (manual — data.gov.ie)
    print("  [D11] Ireland Household Budget Survey 2022–23.")
    print("        Manual download: https://data.gov.ie/dataset/hbs07-average-weekly-household-expenditure")
    print("        Place the CSV in data/raw/ireland_hbs/")
    print()

def phase8_downloads() -> None:
    """
    Phase 8 — RAG knowledge base + FinQA for RQ5 hallucination evaluation.
    [D1] FinQA Verified, [D2] FinQA Original, [D3] CBI Open Data, [D4] EU Digital Finance.
    """
    print("[Phase 8] Downloading RAG and evaluation datasets...")

    # D1 — FinQA Verified
    print("  [D1] FinQA Verified (HuggingFace)...")
    try:
        from datasets import load_dataset  # noqa: PLC0415
        ds = load_dataset("Aiera/finqa-verified", split="train")
        save_path = DATA_RAW / "finqa_verified"
        ds.save_to_disk(str(save_path))
        print(f"  OK  FinQA Verified saved ({len(ds)} examples)")
    except Exception as e:
        print(f"  FAIL  FinQA Verified: {e}")

    # D2 — FinQA Original
    print("  [D2] FinQA Original...")
    try:
        import urllib.request, zipfile  # noqa: PLC0415
        url = "https://github.com/czyssrs/FinQA/raw/main/dataset/train.json"
        dest = DATA_RAW / "finqa_original_train.json"
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
            print(f"  OK  FinQA Original train split saved")
        else:
            print(f"  SKIP  {dest} already exists")
    except Exception as e:
        print(f"  FAIL  FinQA Original: {e}")

    # D3 — CBI Open Data (automated via their API)
    print("  [D3] CBI Open Data Portal — Irish interest rates.")
    print("       API: https://opendata.centralbank.ie")
    print("       Automated fetch implemented in scripts/build_knowledge_base.py (Phase 8).")

    # D4 — EU Digital Finance Platform
    print("  [D4] EU Digital Finance Platform Datasets.")
    print("       Manual download: https://digital-finance-platform.ec.europa.eu/data-hub/datasets")
    print("       Place downloaded CSVs in data/raw/eu_digital_finance/")
    print()


PHASE_FUNCTIONS = {
    1: [phase1_check],
    2: [phase1_check, phase2_downloads],
    3: [phase1_check, phase2_downloads, phase3_downloads],
    5: [phase1_check, phase2_downloads, phase3_downloads, phase5_downloads],
    8: [phase1_check, phase2_downloads, phase3_downloads, phase5_downloads, phase8_downloads],
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