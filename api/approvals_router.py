"""
api/approvals_router.py — Day 8, GET /approvals and POST /approvals/{turn_id}/approve|reject.

SOFT DEPENDENCY ON FASTAPI, DELIBERATELY
    Every other optional heavyweight dependency in this codebase
    (sentence-transformers, faiss, torch, requests) degrades gracefully
    when missing — a warning at import time, not a crash. This module
    does the equivalent: importing it without fastapi installed raises
    one clear ImportError explaining what to install, rather than a
    router that silently exists but can't actually run.

    The core mechanism has NO dependency on this file at all —
    orchestrator/approvals.py's ApprovalStore, and run_demo.py's
    --list-approvals/--approve/--reject flags, are the actual reviewer
    workflow this prototype was built and demonstrated with, using
    nothing beyond the standard library. This router is a thin,
    optional HTTP face on the exact same store, for whoever eventually
    builds Day 10's UI against it.

RUNNING
    pip install fastapi uvicorn
    uvicorn api.approvals_router:app --reload

    GET  /approvals?session_id=...        -> list pending (all sessions if omitted)
    POST /approvals/{turn_id}/approve     -> body: {"reviewer_note": "..."}
    POST /approvals/{turn_id}/reject      -> body: {"reviewer_note": "..."}
"""
from __future__ import annotations

try:
    from fastapi import APIRouter, FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without fastapi installed
    raise ImportError(
        "api/approvals_router.py needs fastapi (pydantic comes with it). "
        "pip install fastapi uvicorn to use this endpoint. The core "
        "approval mechanism itself does not need either — see "
        "orchestrator/approvals.py and run_demo.py's --list-approvals/"
        "--approve/--reject flags."
    ) from exc

from orchestrator.approvals import (
    ApprovalNotFoundError,
    ApprovalTransitionError,
    get_approval_store,
)

router = APIRouter()


class DecisionBody(BaseModel):
    reviewer_note: str = ""


@router.get("/approvals")
def list_approvals(session_id: str | None = None) -> list[dict]:
    """Every turn currently awaiting review, oldest first. Filter to one
    session with ?session_id=..., or omit it for the full cross-customer
    queue — see orchestrator/approvals.py's module docstring for why this
    is one shared queue rather than per-session, unlike the audit log."""
    store = get_approval_store()
    return [p.to_dict() for p in store.list_pending(session_id=session_id)]


@router.get("/approvals/{turn_id}")
def get_approval(turn_id: str) -> dict:
    store = get_approval_store()
    pending = store.get(turn_id)
    if pending is None:
        raise HTTPException(status_code=404, detail=f"no approval for turn_id={turn_id!r}")
    return pending.to_dict()


@router.post("/approvals/{turn_id}/approve")
def approve(turn_id: str, body: DecisionBody = DecisionBody()) -> dict:
    store = get_approval_store()
    try:
        approved = store.approve(turn_id, reviewer_note=body.reviewer_note)
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return approved.to_dict()


@router.post("/approvals/{turn_id}/reject")
def reject(turn_id: str, body: DecisionBody = DecisionBody()) -> dict:
    store = get_approval_store()
    try:
        rejected = store.reject(turn_id, reviewer_note=body.reviewer_note)
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return rejected.to_dict()
app = FastAPI(
    title="HALO Approvals",
    version="1.0",
    description="Day 8 human approval gate — the reviewer-facing queue.",
)
app.include_router(router)