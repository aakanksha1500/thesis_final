"""
Phase 7 - TRiSM-compliant JSONL audit log

Committed BEFORE the Orchestrator so the audit log can be
tested in isolation. This mirrors the dissertation argument: auditability
is a design property that must be provably present before the system
runs, not a feature added after evaluation.

TRiSM requires:
    - Every routing decisions is logged with rationale.
    - Every agent call is logged with success/failure and duration.
    - Every constraint violation is logged with severity.
    - Every cross-agent conflict detected and resolved is logged.
    - All records are machine-readable (JSONL) and self-describing.
    - Each record carries session_id + turn_id for trace reconstruction.

EU AI Act requirements addressed:
    - Traceability: full decision chain reconstructable from logs alone.
    - Human oversight: audit trail enables post-hoc review of any decision.
    - GDPR: user message content is NOT logged - only messgae length.
    Slot values collected during the session ARE logged (user provided them
    for the advisory services: this is within the purpose limitation).

Log location: logs/audit/session_<session_id>.jsonl
Format: one JSON object per line, newline-delimited.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Schema version — increment when event structure changes
SCHEMA_VERSION = "1.0"

def _make_event(
        session_id: str,
        event_type: str,
        payload: dict,
        turn_id: str | None = None,
) -> dict:
    """
    Build a self-describing audit event record.
    Every field needed to reconstruct the trace is present in the record
    itself - no external lookup required.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,
        "turn_id": turn_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }

class AuditLog:
    """
    Per-session JSONL audit logger.

    One instance per Orchestrator session.
    All writes are append only - rrec ords are never modified or deleted.
    If a write fails, it is logged to the appliation logger but does not raise -
    audit failure must never block a user response.
    """

    # Valid event types - enforced to prevent typos in event_type strings.
    VALID_EVENT_TYPES = {
        "SESSION_INIT",
        "TURN_START",
        "ROUTING_DECISION",
        "PLAN",
        "AGENT_CALL",
        "AGENT_FAILURE",
        "AGENT_RETRY",
        "CONFLICT_DETECTED",
        "CONFLICT_RESOLVED",
        "CONSTRAINT_VIOLATION",
        "SYNTHESIS",
        "TURN_END",
        "CUSTOMER_LOAD",
        "HALLUCINATION_FLAG",
        "APPROVAL_GATE",
        "APPROVAL_DECISION",
    }

    def __init__(self, session_id: str, write_init: bool = True) -> None:
        self.session_id = session_id
        self._log_dir = settings.orchestrator.audit_log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"session_{session_id}.jsonl"
        self._write_count = 0

        # Write session initialisation record
        if write_init:
            self._write(_make_event(
                session_id=session_id,
                event_type="SESSION_INIT",
                payload={
                    "schema_version": SCHEMA_VERSION,
                    "compliance": ["EU_AI_Act_Art13", "CBI_MRM", "GDPR_Art5"],
                    "note": (
                        "User message content is never logged (GDPR Art5 purpose limitation). "
                        "Only message length and extracted slot values are recorded."
                    ),
                },
            ))
            logger.info(
                f"[AuditLog] Session {session_id} initialised -> {self._log_path}"
            )

    def _write(self, record: dict) -> None:
        """Append one record to the JSONL file. Never raises."""
        event_type = record.get("event_type")
        if event_type not in self.VALID_EVENT_TYPES:
            logger.error(
                f"[AuditLog] Unknown event_type {event_type!r} — recorded anyway, "
                f"but nothing downstream will match it. Add it to "
                f"VALID_EVENT_TYPES. Known: {sorted(self.VALID_EVENT_TYPES)}"
            )
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            self._write_count += 1
        except Exception as exc:
            logger.error(
                f"[Auditlog] write failed (session={self.session_id}): {exc}. "
                f"Continuing - audit failure must not block user response."
            )


    # Pre-event logging methods
    # Called by Orchestrator at each stage of HALO pipeline.
    def record_turn_start(self, turn_id: str, user_message: str) -> None:
        """
        Log start of a conversational turn.
        GDPR: only messgae length is stored, not content.
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="TURN_START",
            payload={
                "user_message_length": len(user_message),
                "gdpr_note": "Message content not logged per GDPR Arts.",
            },
        ))

    def record_customer_load(
        self,
        customer_id: str,
        known: bool,
        fields_loaded: int,
        missing_fields: list[str],
    ) -> None:
        """
        Log an existing-customer lookup at session start.
        GDPR: only the identifier, count, and field NAMES are logged -
        never the feature VALUES (income, debt, etc.), matching the
        content-minimisation approach used by record_turn_start().
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id="session_start",
            event_type="CUSTOMER_LOAD",
            payload={
                "customer_id": customer_id,
                "known": known,
                "fields_loaded": fields_loaded,
                "missing_fields": missing_fields,
            },
        ))

    def record_routing(
            self,
            turn_id: str,
            routing_decision: str,
            intent: str,
            confidence: float,
            rationale: str,
    ) -> None:
        """
        Log routing decision with rationale (O3 — every decision traceable).
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="ROUTING_DECISION",
            payload={
                "routing_decision": routing_decision,
                "intent": intent,
                "confidence": confidence,
                "rationale": rationale,
            },
        ))
    def record_plan(
            self,
            turn_id: str,
            plan: dict,
    ) -> None:
        """
        Log the Layer 1 plan — G6, Day 4-5.

        SEPARATE FROM record_routing ON PURPOSE
            ROUTING_DECISION answers "what kind of request was this?".
            PLAN answers "who was therefore going to run, who decided that,
            and what was refused?". Folding the second into the first would
            make the planner's rejected proposals unrecoverable from the audit
            trail, and the rejected proposal is the evidence for the
            planner-vs-static comparison.

            It also matters for TRiSM: in a regulated setting, "an LLM chose
            this execution path and here is the deterministic check it
            passed" is the auditable claim. It needs its own record.
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="PLAN",
            payload=plan,
        ))

    def record_agent_call(
        self,
        turn_id: str,
        agent_name: str,
        success: bool,
        duration_ms: float,
        tokens_used: int,
        step_id: str,
        error: str | None = None,
    ) -> None:
        """Log one agent execution — success or failure, always recorded."""
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="AGENT_CALL",
            payload={
                "agent": agent_name,
                "step_id": step_id,
                "success": success,
                "duration_ms": round(duration_ms, 1),
                "tokens_used": tokens_used,
                "error": error,
            },
        ))

    def record_agent_failure(
        self,
        turn_id: str,
        agent_name: str,
        error: str,
        recovery_strategy: str,
        recovery_success: bool,
    ) -> None:
        """Log agent failure and recovery attempt (O4 — AgentFixer trace)."""
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="AGENT_FAILURE",
            payload={
                "agent": agent_name,
                "error": error,
                "recovery_strategy": recovery_strategy,
                "recovery_success": recovery_success,
            },
        ))

    def record_conflict(
        self,
        turn_id: str,
        conflict_type: str,
        description: str,
        resolution: str,
    ) -> None:
        """Log cross-agent conflict detection and resolution."""
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="CONFLICT_DETECTED",
            payload={
                "conflict_type": conflict_type,
                "description": description,
                "resolution": resolution,
            },
        ))

    def record_constraint_violation(
        self,
        turn_id: str,
        rule_id: str,
        severity: str,
        description: str,
        blocked: bool,
    ) -> None:
        """Log constraint violation — hard blocks always recorded."""
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="CONSTRAINT_VIOLATION",
            payload={
                "rule_id": rule_id,
                "severity": severity,
                "description": description,
                "response_blocked": blocked,
            },
        ))

    def record_hallucination_flag(
        self,
        turn_id: str,
        agent_name: str,
        claim: str,
        score: float,
        threshold: float,
        mode: str,
    ) -> None:
        """
        Log an HHEM-flagged claim (Phase 8 — E4 probabilistic layer,
        alongside record_constraint_violation's deterministic layer above).
        Mirrors the same "every decision traceable" TRiSM principle (O3):
        a claim scoring below threshold is recorded with the score and
        threshold that triggered the flag, not just a boolean.
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="HALLUCINATION_FLAG",
            payload={
                "agent": agent_name,
                "claim": claim,
                "score": round(score, 4),
                "threshold": threshold,
                "detector_mode": mode,
            },
        ))

    def record_turn_end(
        self,
        turn_id: str,
        response_length: int,
        agents_invoked: list[str],
        total_duration_ms: float,
        hard_blocks: int,
        conflicts_resolved: int,
    ) -> None:
        """Log turn completion summary."""
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="TURN_END",
            payload={
                "response_length": response_length,
                "agents_invoked": agents_invoked,
                "total_duration_ms": round(total_duration_ms, 1),
                "hard_blocks": hard_blocks,
                "conflicts_resolved": conflicts_resolved,
                "total_records_this_session": self._write_count,
            },
        ))

    def record_approval_gate(
        self,
        turn_id: str,
        reasons: list[str],
    ) -> None:
        """
        Day 8 — a turn was held for human review before delivery. The
        REASONS are logged here, in the audit trail; they are never sent
        to the customer (see ApprovalConfig.withheld_message) — the
        customer-facing side of a gate is deliberately uninformative about
        why, the reviewer-facing side is deliberately specific.
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="APPROVAL_GATE",
            payload={"reasons": reasons},
        ))

    def record_approval_decision(
        self,
        turn_id: str,
        decision: str,
        reviewer_note: str,
    ) -> None:
        """
        Day 8 — a reviewer approved or rejected a held turn. Logged
        separately from record_approval_gate() on purpose: the gate event
        and the decision on it can be arbitrarily far apart in time (a
        different process entirely, in the API case — see
        orchestrator/approvals.py), and TRiSM's audit requirement is that
        BOTH the gate and the decision on it are independently traceable,
        not just the outcome.
        """
        self._write(_make_event(
            session_id=self.session_id,
            turn_id=turn_id,
            event_type="APPROVAL_DECISION",
            payload={"decision": decision, "reviewer_note": reviewer_note},
        ))

    def read_all(self) -> list[dict]:
        """
        Read all records for this session.
        Used by evaluation (component_synergy_score, step_progress_rate)
        and by tests to assert audit structure.
        """
        records = []
        try:
            with self._log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.error(f"[AuditLog] Read failed: {exc}")
        return records

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def write_count(self) -> int:
        return self._write_count
