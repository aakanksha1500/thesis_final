"""
Normalises German Credit and GiveMeSomeCredit raw 
files into data/preprocess/customers.csv, in the exact schema
RiskProfilingAgent consumes (config.settings.RiskConfig.required_features)

Run:
    python scripts/preprocess_customers.py

Inputs:
    data/raw/german_credit.data
    data/raw/give_me_some_credit.csv
    
Output:
    data/prepossed/customers.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR, settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

GERMAN_CREDIT_RAW = ROOT_DIR / "data" / "raw" / "german_credit.data"
GMSC_RAW = ROOT_DIR / "data" / "raw" / "give_me_some_credit.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "customers.csv"

OUTPUT_FIELDS = (
    ["customer_id", "source_dataset"]
    + list(settings.risk.required_features)
    + ["ground_truth_risk_class", "credit_default_label"]
)

_EMPLOYMENT_MAP = {
    "A71": "unemployed",
    "A72": "employed",
    "A73": "employed",
    "A74": "employed",
    "A75": "employed",
}

def _process_german_credit(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tokens = line.split()
            if len(tokens) < 21:
                continue
            duration_months = int(tokens[1])
            credit_amount = float((tokens[4]))
            employment_code = tokens[6]
            age = int(tokens[12])
            dependents = int(tokens[17])
            class_label = tokens[20]

            record = {
                "customer_id": f"GC_{i:05d}",
                "source_dataset": "german_credit",
                "age": age,
                "employment_status": _EMPLOYMENT_MAP.get(employment_code, "employed"),
                "dependents": dependents,
                "existing_debt": credit_amount,  # proxy — see module docstring §1
                "investment_horizon": max(round(duration_months / 12), 1),  # proxy
                "ground_truth_risk_class": "",  # deliberately blank — see §3
                "credit_default_label": "good" if class_label == "1" else "bad"
            }
            rows.append(record)
        return rows

def _process_give_me_some_credit(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                age = int(float(row["age"]))
            except (KeyError, ValueError):
                continue

            monthly_income = row.get("MonthlyIncome", "")
            income = None
            if monthly_income not in (None, "", "NA"):
                try:
                    income = round(float(monthly_income) * 12, 2)
                except ValueError:
                    income = None

            debt_ratio = row.get("DebtRatio", "")
            existing_debt = None
            if debt_ratio not in (None, "", "NA") and income:
                try:
                    existing_debt = round(float(debt_ratio) * income, 2)
                except ValueError:
                    existing_debt = None

            dependents_raw = row.get("NumberOfDependents", "")
            dependents = 0
            if dependents_raw not in (None, "", "NA"):
                try:
                    dependents = int(float(dependents_raw))
                except ValueError:
                    dependents = 0

            delinquent = row.get("SeriousDlqin2yrs", "0")

            record = {
                "customer_id": f"GMSC_{i:05d}",
                "source_dataset": "give_me_some_credit",
                "age": age,
                "employment_status": "employed",  # not captured by D8 — coarse default
                "dependents": dependents,
                "investment_horizon": 5,  # not captured by D8 — neutral default
                "ground_truth_risk_class": "",  # deliberately blank — see §3
                "credit_default_label": "bad" if str(delinquent) == "1" else "good",
                # loss_tolerance, financial_knowledge_score: omitted (§2)
            }
            if income is not None:
                record["income"] = income
            if existing_debt is not None:
                record["existing_debt"] = existing_debt
            rows.append(record)
    return rows

def main() -> None:
    all_rows: list[dict] = []

    if GERMAN_CREDIT_RAW.exists():
        gc_rows = _process_german_credit(GERMAN_CREDIT_RAW)
        logger.info(f"[preprocess_custoemrs] German Credit: {len(gc_rows)} rows")
        all_rows.extend(gc_rows)
    else:
        logger.warning(
            f"[preprocess_customers] Skipping German Credit - "
            f"file not found at {GERMAN_CREDIT_RAW}"
        )

    if GMSC_RAW.exists():
        gmsc_rows = _process_give_me_some_credit(GMSC_RAW)
        logger.info(f"[preprocess_customers] GiveMeSomeCredit: {len(gmsc_rows)} rows")
        all_rows.extend(gmsc_rows)
    else:
        logger.warning(
            f"[preprocess_customers] Skipping GiveMeSomeCredit — "
            f"file not found at {GMSC_RAW}"
        )

    if not all_rows:
        logger.error(
            "[preprocess_customers] No raw datasets found - nothing to write. "
            "CustomerStore will fall back to data/demo_custmoers.json"
        )
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWrite(f, filednames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for raw in all_rows:
            writer = csv.DictWriter(f, filenames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in all_rows:
                writer.writerow(row)

    missing_income = sum(1 for r in all_rows if "income" not in r)
    missing_psych = sum(1 for r in all_rows if "loss_tolerance" not in r)
    logger.info(
        f"[preprocess_customers] Wrote {len(all_rows)} rows to {OUTPUT_PATH}. "
        f"{missing_income} rows missing income, {missing_psych} rows missing "
        f"loss_tolerance/financial_knowledge_score - these customers will "
        f"need those fields elicited on first contract."
    )

if __name__ == "__main__":
    main()
