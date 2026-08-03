"""
Day 2a — TransactionStore: lookup layer for per-customer transaction
histories, mirroring CustomerStore's pattern (data/customer_store.py).

Backend: data/processed/transactions.json, built by
scripts/generate_transactions.py. Gitignored (generated data, like
data/processed/customers.csv and data/processed/risk_model.pkl) — must
be regenerated locally before it's available. An absent file means
empty lookups, not an exception: nothing in this codebase should require
an external file to be present just to run (mirrors CustomerStore's own
fallback philosophy, and RiskProfilingAgent._load_ml_model()'s).

WHERE THIS FITS
    scripts/generate_transactions.py writes the file.
    TransactionStore (here) reads it back.
    BudgetAgent._aggregate_transactions() turns a raw transaction list
    into monthly_expenses at a given window size.
    Nothing currently wires TransactionStore into the live Orchestrator
    conversation flow the way CustomerStore is wired for risk features
    (Orchestrator._load_customer) — that's a genuine next step, not yet
    built. Today this is consulted directly (tests, scripts), the same
    way CustomerStore can be used standalone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR, settings
from utils.logger import get_logger

logger = get_logger(__name__)

TRANSACTIONS_PATH = ROOT_DIR / settings.budget.transactions_path


class TransactionStore:
    """
    Lookup layer for per-customer transaction histories.

    In production this would be swapped for a database query (e.g. the
    same store CustomerStore would eventually front). lookup() is
    deliberately the whole interface so that swap needs no caller-side
    changes.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or TRANSACTIONS_PATH
        self._customers: dict[str, dict[str, Any]] = {}
        self._meta: dict[str, Any] = {}
        self._loaded_from = "none"
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning(
                f"[TransactionStore] No transactions file at {self._path} — "
                f"all lookups will return an empty list. Run "
                f"scripts/generate_transactions.py to build it."
            )
            self._loaded_from = "empty"
            return
        try:
            data = json.loads(self._path.read_text())
            self._customers = data.get("customers", {})
            self._meta = data.get("_meta", {})
            self._loaded_from = str(self._path)
            logger.info(
                f"[TransactionStore] Loaded {len(self._customers)} customer "
                f"transaction histories from {self._path} "
                f"(generator_version={self._meta.get('generator_version')}, "
                f"seed={self._meta.get('seed')})"
            )
        except Exception:
            logger.exception(
                f"[TransactionStore] Failed to read {self._path} — falling "
                f"back to empty store. Full traceback above; do not treat "
                f"this as recoverable without investigating, it means every "
                f"transaction lookup will silently return []."
            )
            self._customers = {}
            self._loaded_from = "empty"

    def lookup(self, customer_id: str) -> list[dict[str, Any]]:
        """
        Return the transaction list for customer_id — each item
        {"date": "YYYY-MM-DD", "category": str, "amount": float} — or []
        if the customer has no transaction history on file. [] is a
        normal, expected result (e.g. a brand-new customer), not an
        error condition — callers should treat it exactly like "nothing
        to aggregate", not "something went wrong".
        """
        record = self._customers.get(customer_id)
        if record is None:
            logger.info(
                f"[TransactionStore] No transaction history for "
                f"customer_id={customer_id!r}"
            )
            return []
        return record.get("transactions", [])

    def scenario_for(self, customer_id: str) -> str | None:
        """The named scenario this customer demonstrates, if any (see
        scripts/generate_transactions.py's SCENARIOS) — mainly useful for
        tests and the evidence-generation evaluation, not production."""
        record = self._customers.get(customer_id)
        return record.get("scenario") if record else None

    def __len__(self) -> int:
        return len(self._customers)

    def __contains__(self, customer_id: str) -> bool:
        return customer_id in self._customers