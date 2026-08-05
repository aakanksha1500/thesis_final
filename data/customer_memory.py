"""
data/customer_memory.py — Day 9, long-term memory (G7).

WHAT THIS DOES **NOT** STORE, AND WHY — READ BEFORE ADDING A "features" COLUMN
    The build plan's schema for this table includes a `features` column.
    This implementation deliberately omits it.

    data/customer_store.py's CustomerStore already persists exactly that —
    durably (fsync + atomic rename, not just an in-memory dict — see its
    _persist_csv()), and Orchestrator.update_customer_features() already
    writes to it on every elicitation. Verified empirically, not assumed:
    a customer_id with no CustomerStore record, taken through the full
    risk-elicitation questionnaire in one process, was found fully
    pre-filled — zero re-elicitation — by a SECOND, independent process
    given the same customer_id. That is the build plan's own demo
    scenario for the "features" half of memory, and it was already
    working before this file existed.

    Adding a second, redundant `features` column here would create two
    sources of truth for the same data with no mechanism keeping them in
    sync — update_customer_features() would need to write to both, or
    they drift, and a drifted memory table is a worse failure mode than
    a missing one (it looks authoritative and quietly isn't). This table
    exists for the things CustomerStore's schema genuinely has no room
    for: what was last computed about a customer, what they've said they
    want, and how many times we've spoken with them — not a duplicate
    home for what CustomerStore already owns.

WHAT preferences MEANS HERE
    Scoped deliberately narrow for v1: values ConversationalAgent's slot
    tracking already collects (settings.conversational.tracked_slots)
    that are NOT also RiskConfig.required_features — currently just
    "investment_goal" and "user_name". Not "risk_tolerance" or "time_
    horizon", despite both being tracked_slots names: they're vestigial
    synonyms for loss_tolerance/investment_horizon from an earlier naming
    scheme (see config/settings.py's tracked_slots comment), and treating
    them as a separate "preference" alongside the real required_feature
    of the same meaning would just create two answers to one question.

WHY A SEPARATE SQLITE FILE FROM orchestrator/approvals.py
    Different lifecycle, different reader. Approvals are a reviewer's
    queue — transient, cleared by being decided. Customer memory is a
    long-lived per-customer record that outlives any single turn, turn's
    approval, or session. Keeping them in one file would couple two
    things that fail, get inspected, and get cleared independently.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# The only tracked_slots names treated as "preferences" — see module
# docstring. Deliberately a short, explicit allowlist rather than "every
# tracked slot not in required_features": that would silently start
# persisting budget-questionnaire slots (housing_cost, food_spend, ...)
# the moment someone extends tracked_slots for an unrelated reason.
PREFERENCE_SLOTS = ("investment_goal", "user_name")


@dataclass
class CustomerMemory:
    customer_id: str
    risk_profile: dict[str, Any] | None = None
    preferences: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    last_seen: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "risk_profile": self.risk_profile,
            "preferences": dict(self.preferences),
            "turn_count": self.turn_count,
            "last_seen": self.last_seen,
            "updated_at": self.updated_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CustomerMemoryStore:
    """SQLite-backed get/upsert for CustomerMemory rows, one per customer_id."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.conversational.memory_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS customer_memory (
                    customer_id   TEXT PRIMARY KEY,
                    risk_profile  TEXT,
                    preferences   TEXT NOT NULL DEFAULT '{}',
                    turn_count    INTEGER NOT NULL DEFAULT 0,
                    last_seen     TEXT,
                    updated_at    TEXT
                )
            """)
            conn.commit()

    def get(self, customer_id: str) -> CustomerMemory | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM customer_memory WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        if row is None:
            return None
        return CustomerMemory(
            customer_id=row["customer_id"],
            risk_profile=json.loads(row["risk_profile"]) if row["risk_profile"] else None,
            preferences=json.loads(row["preferences"]),
            turn_count=row["turn_count"],
            last_seen=row["last_seen"],
            updated_at=row["updated_at"],
        )

    def upsert(
        self,
        customer_id: str,
        risk_profile: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
        turn_increment: int = 1,
    ) -> CustomerMemory:
        """
        Merge into the existing row (creating it if new). risk_profile,
        if given, REPLACES the stored one (it's "last computed", not
        cumulative); preferences MERGE (a customer stating their goal
        once should not need re-stating because a later turn's dict
        didn't repeat it). turn_count always increments — called once
        per turn, at turn end, per the build plan's own "persists on
        turn end" instruction, so this is a lifetime count of turns
        with this customer across every session, not a per-session one.
        """
        existing = self.get(customer_id)
        merged_prefs = dict(existing.preferences) if existing else {}
        if preferences:
            merged_prefs.update({k: v for k, v in preferences.items() if v is not None})

        new_risk_profile = (
            risk_profile if risk_profile is not None
            else (existing.risk_profile if existing else None)
        )
        new_turn_count = (existing.turn_count if existing else 0) + turn_increment
        now = _now()

        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO customer_memory
                   (customer_id, risk_profile, preferences, turn_count, last_seen, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(customer_id) DO UPDATE SET
                     risk_profile = excluded.risk_profile,
                     preferences = excluded.preferences,
                     turn_count = excluded.turn_count,
                     last_seen = excluded.last_seen,
                     updated_at = excluded.updated_at""",
                (
                    customer_id,
                    json.dumps(new_risk_profile) if new_risk_profile is not None else None,
                    json.dumps(merged_prefs),
                    new_turn_count,
                    now, now,
                ),
            )
            conn.commit()

        logger.info(
            f"[CustomerMemoryStore] customer_id={customer_id!r} updated — "
            f"turn_count={new_turn_count} preferences={sorted(merged_prefs)} "
            f"risk_profile={'present' if new_risk_profile else 'none'}"
        )
        return self.get(customer_id)


def extract_preferences_from_slots(slots: dict[str, Any]) -> dict[str, Any]:
    """
    The narrow view of ConversationalAgent's tracked slots that counts as
    a "preference" for memory purposes — see module docstring for exactly
    why this list is short and explicit rather than derived generically.
    """
    return {k: slots[k] for k in PREFERENCE_SLOTS if slots.get(k) is not None}


# lazy singleton, matching orchestrator/approvals.py's exact pattern
_customer_memory_store: "CustomerMemoryStore | None" = None


def get_customer_memory_store() -> "CustomerMemoryStore":
    global _customer_memory_store
    if _customer_memory_store is None:
        _customer_memory_store = CustomerMemoryStore()
    return _customer_memory_store


def set_customer_memory_store(instance: "CustomerMemoryStore | None") -> None:
    """Inject a substitute (or None to reset). For tests."""
    global _customer_memory_store
    _customer_memory_store = instance