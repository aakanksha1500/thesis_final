"""
Phase 8b — CustomerStore: lookup/save layer for the existing-customer
pipeline (locked down before the API layer — see Phase 9).

Backend priority:
  1. data/processed/customers.csv
     Built by scripts/preprocess_customers.py from German Credit [D7] and
     GiveMeSomeCredit [D8] raw files, normalised to the exact schema
     RiskProfilingAgent requires (config.settings.RiskConfig.required_features).
  2. data/demo_customers.json
     3 hand-curated demo profiles, IDs prefixed `DEMO_` (e.g. DEMO_GC_042) —
     DELIBERATELY not `GC_00042`. scripts/preprocess_customers.py assigns
     German Credit rows IDs as `GC_{line_index:05d}` and GiveMeSomeCredit
     rows as `GMSC_{line_index:05d}`. If a demo ID ever matched a real
     auto-generated one, and CustomerStore is CSV-first, the real
     (incomplete — see notes below) row would silently SHADOW the curated
     demo profile the moment customers.csv exists. The `DEMO_` prefix
     guarantees that can never happen, regardless of dataset size.
     Used automatically when (1) is absent, so the pipeline and its tests
     are runnable before the raw datasets are downloaded. This mirrors the
     existing fallback pattern in RiskProfilingAgent._load_ml_model() and
     rag/embedder.py's hashing fallback: nothing in this codebase should
     require an external file to be present just to run.
"""

from __future__ import annotations

import csv
import os
import tempfile
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
    Lookup/save layer for existing-customer records.

    In production this would be swapped for a database query
    (e.g. Postgres). lookup()/save() are deliberately the whole interface
    so that swap needs no caller-side changes.
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

    def _load(self) -> None:
        if self._csv_path.exists():
            try:
                self._load_csv(self._csv_path)
                self._loaded_from = str(self._csv_path)
                logger.info(
                    f"[CustomerStore] Loaded {len(self._records)} customer "
                    f"records from {self._csv_path}"
                )
                return
            except Exception:
                logger.exception(
                    f"[CustomerStore] Failed to read {self._csv_path} "
                    f"— falling back to demo profiles. Full traceback above."
                )

        if self._demo_path.exists():
            try:
                self._load_json(self._demo_path)
                self._loaded_from = str(self._demo_path)
                logger.info(
                    f"[CustomerStore] No processed customer file at "
                    f"{self._csv_path} — using {len(self._records)} demo "
                    f"profiles from {self._demo_path}. Run "
                    f"scripts/preprocess_customers.py to build the real dataset."
                )
                return
            except Exception:
                logger.exception(
                    f"[CustomerStore] Failed to read demo profiles "
                    f"{self._demo_path} — falling back to empty store. "
                    f"Full traceback above; do not treat this as recoverable "
                    f"without investigating, it means every 'known customer' "
                    f"lookup will silently return None."
                )

        logger.warning(
            "[CustomerStore] No customer records available from any backend "
            "— all lookups will return None (treated as new customers)."
        )
        self._loaded_from = "empty"

    def _load_csv(self, path: Path) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = self._coerce_types(dict(row))
                cid = record.get("customer_id")
                if cid:
                    self._records[cid] = record

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
        """CSV round-trips everything as strings — cast numerics back."""
        out: dict[str, Any] = dict(row)
        for k in _NUMERIC_INT_FIELDS:
            if k in out and out[k] not in (None, ""):
                try:
                    out[k] = int(float(out[k]))
                except (TypeError, ValueError):
                    out.pop(k, None)
            elif k in out and out[k] == "":
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
            "features": dict,           # only the RiskConfig.required_features present
            "missing_fields": list[str],  # required_features NOT present at all
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
            f"[CustomerStore] Found record for customer_id={customer_id!r} "
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
        source_dataset: str = "session_update",
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
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self._csv_path.parent),
            prefix=f".{self._csv_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in self._records.values():
                    writer.writerow(rec)
                f.flush()
                os.fsync(f.fileno())   # data on disk before the rename, not just in the page cache
            os.replace(tmp_path, self._csv_path)
        except BaseException:
            # BaseException, not Exception: KeyboardInterrupt during a write is
            # exactly the case that used to leave a truncated store behind.
            tmp_path.unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, customer_id: str) -> bool:
        return customer_id in self._records