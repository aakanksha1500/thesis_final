"""
TransactionStore tests — uses a temp file, not the real generated
data/processed/transactions.json, so these pass whether or not
scripts/generate_transactions.py has been run.
"""
from __future__ import annotations

import json

import pytest

from data.transaction_store import TransactionStore

SAMPLE_DATA = {
    "_meta": {"generated": True, "seed": 42, "generator_version": "1.0"},
    "customers": {
        "TXN_TEST": {
            "customer_id": "TXN_TEST",
            "monthly_income": 3000.0,
            "scenario": "test_scenario",
            "narrative": "for tests only",
            "transactions": [
                {"date": "2025-01-05", "category": "housing", "amount": 900.0},
                {"date": "2025-02-05", "category": "housing", "amount": 900.0},
            ],
        },
    },
}


@pytest.fixture
def populated_store(tmp_path):
    path = tmp_path / "transactions.json"
    path.write_text(json.dumps(SAMPLE_DATA))
    return TransactionStore(path=path)


class TestTransactionStore:

    def test_missing_file_returns_empty_lookup_not_an_exception(self, tmp_path):
        store = TransactionStore(path=tmp_path / "does_not_exist.json")
        assert store.lookup("ANYONE") == []
        assert len(store) == 0

    def test_lookup_known_customer_returns_transactions(self, populated_store):
        txns = populated_store.lookup("TXN_TEST")
        assert len(txns) == 2
        assert txns[0]["category"] == "housing"

    def test_lookup_unknown_customer_returns_empty_list(self, populated_store):
        assert populated_store.lookup("NOT_A_REAL_CUSTOMER") == []

    def test_contains(self, populated_store):
        assert "TXN_TEST" in populated_store
        assert "NOBODY" not in populated_store

    def test_len(self, populated_store):
        assert len(populated_store) == 1

    def test_scenario_for_known_customer(self, populated_store):
        assert populated_store.scenario_for("TXN_TEST") == "test_scenario"

    def test_scenario_for_unknown_customer_is_none(self, populated_store):
        assert populated_store.scenario_for("NOBODY") is None

    def test_corrupt_file_falls_back_to_empty_not_a_crash(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{ this is not valid json")
        store = TransactionStore(path=path)
        assert store.lookup("ANYONE") == []
        assert len(store) == 0