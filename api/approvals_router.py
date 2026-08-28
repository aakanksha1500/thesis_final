"""
HTTP routes for the reviewer approval queue.
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
    """Lists turns awating review, oldest first. Pass session_id to filter to one
    session, or ."""
    store = get_approval_store()
    return [p.to_dict() for p in store.list_pending(session_id=session_id)]


@router.get("/approvals/{turn_id}")
def get_approval(turn_id: str) -> dict:
    """Fetch one pending approval by turn_id, or 404 if it doesn't exist."""
    store = get_approval_store()
    pending = store.get(turn_id)
    if pending is None:
        raise HTTPException(status_code=404, detail=f"no approval for turn_id={turn_id!r}")
    return pending.to_dict()


@router.post("/approvals/{turn_id}/approve")
def approve(turn_id: str, body: DecisionBody = DecisionBody()) -> dict:
    """Approve a held turn; optional reviewer_note is stored with the decision."""
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
    """Reject a held turn; optional reviewer_note is stored with the decision."""
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