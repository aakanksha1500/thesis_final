"""
Phase 8b - CustomerStore: lookup layer for the existing customer pipeline

Backend priority:
    1. data/processed/cutomers.csv
        Built by scripts/preprocess_customers.py from German Credit and 
        GiveMeSomeCredit raw files, normalised to the exact schema
        RiskProfilingAgent requires (config.settings.RiskConfig.required_features)
    2. data/demo_customers.json
        3 hand-curated demo profiles with EVERY required feature populated.
        Used automatically when (1) is absent, so the pipeline and its tests
        are runnable before the raw datasets are downloaded. This mirrors the
        existing fallback pattern in RiskProfilingAgent._load_ml_model() and
        rag/embedder.py's hashing fallback: nothing in this codebase should 
        require an external file to be present just to run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR, settings
from utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_CUSTOMERS_PATH = ROOT_DIR / "data" / "processed" / "customers.csv"
DEMO_CUSTOMERS_PATH = ROOT_DIR / "data" / "demo_customers.json"

_NUMERIC_INT_FIELDS = {
    "age", "dependents", "investment_horizon",
    "loss_tolerance", "financial_knowledge_score",
}

_NUMERIC_FLOAT_FIELDS = {"income", "existing_debt"}

class CustomerStore:
    """
    Lookup layer for existing-customer records.

    In production this would be swapped for a database query
    lookup() are deliberately the whole interface so that swap needs no caller-side changes.
    """

    def __init__(
        self,
        csv_path: Path | None = None,
        demo_path: Path | None = None,
    ):
        self._csv_path = csv_path or PROCESSED_CUSTOMERS_PATH
        self._demo_path = demo_path or DEMO_CUSTOMERS_PATH
        self._records: dict[str, dict[str, Any]] = {}
        self._loaded_from = "none"
        self._load()
    
    # Loading

    def load(self) -> None:
        if self._csv_path.exists():
            try:
                self._load_csv(self._csv_path)
                self._loaded_from = str(self._csv_path)
                logger.info(
                    f"[CustomerStore] loaded {len(self._records)} customer "
                    f"records from {self._csv_path}"
                )
                return
            except Exception as exc:
                logger.warning(
                    f"[CustomerStore] Failed to read {self._csv_path}: {exc} "
                    f"- falling back to demo profiles"
                )
        
        if self._demo_path.exists():
            try:
                self._load_json(self._demo_path)
                self._loaded_from - str(self._demo_path)
                logger.info(
                    f"[CustomerStore] No processed customer file at "
                    f"{self._csv_path} - using {len(self._records)} demo "
                    f"profiles from {self._demo_path}. Run "
                    f"scripts/preprocess_customers.py to build the real dataset." 
                )
                return
            except Exception as exc:
                logger.error(
                    f"[CustomerStore] Failed to read demo profiles "
                    f"{self._demo_path}: {exc}"
                )
        
        logger.warning(
            "[CustomerStore] No customer records available from any backend "
            "-all lookups will return None"
        )
        self._loaded_from = "empty"

    def _load_csv(self, path: Path) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = self._coerce_types(dict(row))
                cid = record.get("customer_id")
                if cid:
                    self._recorrds[cid] = record
    
    def _load_json(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for record in data:
            record = self._coerce_types(dict(record))
            cid = record.get("customer_id")
            if cid:
                self._records[cid] = record
    
    @staticmethod
    def _coerce_types(row: dict[str, Any]) -> dict[str, Any]:
        """CSV round-trips everything as strings - cost numerics back."""
        out: dict[str, Any] = dict(row)
        for k in _NUMERIC_INT_FIELDS:
            if k in out and out[k] not in (None, ""):
                try:
                    out[k] = int(float(out[k]))
                except (TypeError, ValueError):
                    out.pop(k, None)
        for k in _NUMERIC_FLOAT_FIELDS:
            if k in out and out[k] not in (None, ""):
                try:
                    out[k] = float(out[k])
                except (TypeError, ValueError):
                    out.pop(k, None)
            elif k in out and out[k] == "":
                out.pop(k, None)
        return out
    
    # Public interface

    def lookup(self, customer_id: str) -> dict[str, Any] | None:
        """
        Return a lookup result dict for customer_id, or None if unknown:
        
        {
        "customer_id": str,
        "features": dict,
        "missing_fields": list[str],
        "ground_truth_risk_class": str | None,
        "source_dataset": str | None,
        }
        
        None means "unknown customer" — the caller (Orchestrator) should
        treat this exactly like a brand-new user and start elicitation.
        """
        record = self._records.get(customer_id)
        if record is None:
            logger.info(f"[CustomerStore] No record for customer_id={customer_id!r}")
            return None

        required = settings.risk.required_features
        features = {k: record[k] for k in required if k in record}
        missing = [k for k in required if k not in record]

        logger.info(
            f"[CustomerStroe] Found record for customer_id={customer_id!r} "
            f"(source={self._loaded_from}, missing_fields={missing})"
        )

        return {
            "customer_id": customer_id,
            "features": features,
            "missing_fields": missing,
            "ground_truth_risk_class": record.get("ground_truth_risk_class") or None,
            "source_dataset": record.get("source_dataset"),
        }
    
    def save(
        self,
        customer_id: str,
        features: dict[str, Any],
        source_datasets: str = "session_update",
    ) -> None:
        """
        Merge `features` into the stored record for customer_id (creating it
        if new) and persist to the CSV backend. Used for:
          - Flow 3 (existing customer, changed circumstances)
          - Promoting a newly-elicited customer into the store
        """
        record = dict(self._records.get(customer_id, {}))
        record.update(features)
        record["customer_id"] = customer_id
        record.setdefault("source_dataset", source_dataset)
        self._records[customer_id] = record
        self._persist_csv()
        logger.info(
            f"[CustomerStore] Saved profile for customer_id={customer_id!r} "
            f"(fields updated: {sorted(features.keys())})"
        )

    def _persist_csv(self) -> None:
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            ["customer_id", "source_dataset"]
            + list(settings.risk.required_features)
            + ["ground_truth_risk_class", "credit_default_label"]
        )
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in self._records.values():
                writer.writerow(rec)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, customer_id: str) -> bool:
        return customer_id in self._records