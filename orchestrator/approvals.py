"""
orchestrator/approvals.py — Day 8, human approval gate.

State machine on the turn:
    PROCESSING -> AWAITING_APPROVAL -> APPROVED / REJECTED -> DELIVERED

PROCESSING is never persisted — it's just "process_turn() is still
running". A PendingApproval row is created only once a turn actually
clears the gate (Orchestrator._check_approval_gate() returned non-empty
reasons); a turn that never gates has no row here at all and is
conceptually "delivered" the instant process_turn() returns. Everything
below is about the turns that DID gate.

WHY SQLITE, AND WHY ONE SHARED FILE ACROSS SESSIONS
    AuditLog is deliberately per-session (logs/audit/session_<id>.jsonl) —
    it's a trace of one conversation. A reviewer's queue is the opposite
    shape: "everything pending, across every customer, right now" — GET
    /approvals in the build plan's own wording. A single shared table is
    the natural fit, and SQLite (stdlib, no new dependency) is enough for
    a queue a prototype's reviewer polls; it is not trying to be a
    production approvals service.

WHY THIS DOESN'T LIVE INSIDE Orchestrator
    Same reasoning as AuditLog and ConflictResolver before it: an
    approval can be decided by a process that never touches the
    Orchestrator instance that created it at all (a reviewer hitting the
    API, or run_demo.py's --approve/--reject flags in a second terminal
    while the first one is still running). The store has to be
    reachable on its own.
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

STATUSES = ("pending", "approved", "rejected", "delivered")

# Only these transitions are legal. Enforced in code, not just documented,
# because a silently-accepted illegal transition here (approving something
# already rejected, delivering something never approved) is exactly the
# kind of "plausible-looking but wrong" state this system tries hard
# everywhere else not to produce.
_LEGAL_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"approved", "rejected"},
    "approved": {"delivered"},
    "rejected": set(),
    "delivered": set(),
}


class ApprovalTransitionError(Exception):
    """Raised on an illegal status transition (see _LEGAL_TRANSITIONS)."""


class ApprovalNotFoundError(Exception):
    """Raised when a turn_id has no PendingApproval row."""


@dataclass
class PendingApproval:
    """
    Shape matches the build plan exactly, plus decided_at (when status
    left "pending") — needed to report turnaround time and not otherwise
    inferable once the row has moved on.
    """
    turn_id: str
    session_id: str
    reasons: list[str]
    draft_response: str          # withheld from the customer until delivered
    agent_results: list[dict]    # what the reviewer needs to judge
    created_at: str
    status: str = "pending"      # pending | approved | rejected | delivered
    reviewer_note: str = ""
    decided_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "reasons": list(self.reasons),
            "draft_response": self.draft_response,
            "agent_results": list(self.agent_results),
            "created_at": self.created_at,
            "status": self.status,
            "reviewer_note": self.reviewer_note,
            "decided_at": self.decided_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalStore:
    """SQLite-backed CRUD + state transitions for PendingApproval rows."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or settings.approval.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    turn_id         TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    reasons         TEXT NOT NULL,
                    draft_response  TEXT NOT NULL,
                    agent_results   TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    reviewer_note   TEXT NOT NULL DEFAULT '',
                    decided_at      TEXT
                )
            """)
            conn.commit()

    # -- create / read --------------------------------------------------

    def create(self, pending: PendingApproval) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO pending_approvals
                   (turn_id, session_id, reasons, draft_response,
                    agent_results, created_at, status, reviewer_note, decided_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pending.turn_id, pending.session_id,
                    json.dumps(pending.reasons), pending.draft_response,
                    json.dumps(pending.agent_results), pending.created_at,
                    pending.status, pending.reviewer_note, pending.decided_at,
                ),
            )
            conn.commit()
        logger.info(
            f"[ApprovalStore] turn={pending.turn_id} session={pending.session_id} "
            f"gated — reasons={pending.reasons}"
        )

    def get(self, turn_id: str) -> PendingApproval | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM pending_approvals WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        return self._row_to_dataclass(row) if row else None

    def list_pending(self, session_id: str | None = None) -> list[PendingApproval]:
        return self._list(status="pending", session_id=session_id)

    def list_approved(self, session_id: str | None = None) -> list[PendingApproval]:
        """Approved but not yet delivered — what a live session should
        check for and hand back to the customer on its next turn."""
        return self._list(status="approved", session_id=session_id)

    def list_all(self, session_id: str | None = None) -> list[PendingApproval]:
        return self._list(status=None, session_id=session_id)

    def _list(
        self, status: str | None, session_id: str | None,
    ) -> list[PendingApproval]:
        query = "SELECT * FROM pending_approvals"
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dataclass(r) for r in rows]

    # -- state transitions -----------------------------------------------

    def _transition(
        self, turn_id: str, new_status: str, reviewer_note: str = "",
    ) -> PendingApproval:
        pending = self.get(turn_id)
        if pending is None:
            raise ApprovalNotFoundError(f"no pending approval for turn_id={turn_id!r}")
        allowed = _LEGAL_TRANSITIONS[pending.status]
        if new_status not in allowed:
            raise ApprovalTransitionError(
                f"turn={turn_id} is {pending.status!r} — cannot move to "
                f"{new_status!r} (allowed from here: {sorted(allowed) or 'none'})"
            )
        with closing(self._connect()) as conn:
            conn.execute(
                """UPDATE pending_approvals
                   SET status = ?, reviewer_note = ?, decided_at = ?
                   WHERE turn_id = ?""",
                (new_status, reviewer_note or pending.reviewer_note, _now(), turn_id),
            )
            conn.commit()
        logger.info(f"[ApprovalStore] turn={turn_id} {pending.status} -> {new_status}")

        # Audit-log the DECISION here, not in each caller (CLI, API router,
        # tests) — this is the one choke point every decision path goes
        # through, so logging it here means it can't be forgotten by a
        # future caller the way a repeated-in-three-places call could be.
        # Deliberately reopens AuditLog(session_id=pending.session_id)
        # rather than requiring a live Orchestrator: the reviewer deciding
        # this is very often a different process from the one that
        # created the turn (see this module's docstring).
        if new_status in ("approved", "rejected"):
            try:
                from orchestrator.audit_log import AuditLog
                AuditLog(
                    session_id=pending.session_id, write_init=False,
                ).record_approval_decision(
                    turn_id=turn_id, decision=new_status,
                    reviewer_note=reviewer_note or pending.reviewer_note,
                )
            except Exception as exc:
                # Same principle AuditLog._write() itself follows: an audit
                # failure must never block the actual decision from taking
                # effect.
                logger.error(
                    f"[ApprovalStore] failed to audit-log decision for "
                    f"turn={turn_id}: {exc}"
                )

        return self.get(turn_id)

    def approve(self, turn_id: str, reviewer_note: str = "") -> PendingApproval:
        return self._transition(turn_id, "approved", reviewer_note)

    def reject(self, turn_id: str, reviewer_note: str = "") -> PendingApproval:
        return self._transition(turn_id, "rejected", reviewer_note)

    def mark_delivered(self, turn_id: str) -> PendingApproval:
        """
        Called once the customer-facing side has actually handed
        draft_response back — see Orchestrator.collect_approved_response().
        Not folded into approve() itself: approving and delivering can be
        arbitrarily far apart (a reviewer approves at 2pm; the customer
        doesn't ask again, or reconnect, until 6pm), and DELIVERED means
        "the customer has now seen it", which approve() cannot know.
        """
        return self._transition(turn_id, "delivered")

    def _row_to_dataclass(self, row: sqlite3.Row) -> PendingApproval:
        return PendingApproval(
            turn_id=row["turn_id"],
            session_id=row["session_id"],
            reasons=json.loads(row["reasons"]),
            draft_response=row["draft_response"],
            agent_results=json.loads(row["agent_results"]),
            created_at=row["created_at"],
            status=row["status"],
            reviewer_note=row["reviewer_note"],
            decided_at=row["decided_at"],
        )


# lazy singleton, matching the pattern used by market_data_client.py and
# product_data_client.py — a shared resource, injectable in tests via
# set_approval_store(fresh_instance_or_None).
_approval_store: "ApprovalStore | None" = None


def get_approval_store() -> "ApprovalStore":
    global _approval_store
    if _approval_store is None:
        _approval_store = ApprovalStore()
    return _approval_store


def set_approval_store(instance: "ApprovalStore | None") -> None:
    """Inject a substitute (or None to reset). For tests."""
    global _approval_store
    _approval_store = instance