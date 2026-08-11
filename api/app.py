"""
FastAPI server for the web UI.

ENDPOINTS
    POST /chat              -> one turn result, JSON (see _serialise_result)
    GET  /trace/{turn_id}   -> captured utils.trace events for that turn
    GET  /approvals         -> from api/approvals_router.py, included whole
    POST /approvals/{id}/approve|reject  -> likewise
    GET  /                  -> static/index.html

RUNNING
    pip install fastapi uvicorn
    uvicorn api.app:app --reload
    open http://127.0.0.1:8000/
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without fastapi installed
    raise ImportError(
        "api/app.py needs fastapi (pydantic comes with it) and, to actually "
        "run, uvicorn. pip install fastapi uvicorn. Nothing else in this "
        "codebase needs either — run_demo.py is the dependency-free way to "
        "exercise the same pipeline this app is a UI on top of."
    ) from exc

import api.approvals_router as approvals_router
from api.session_registry import SessionRegistry
from orchestrator.orchestrator import AgentResult, OrchestratorResult
# from orchestrator.approvals import get_approval_store
from utils import trace
from utils.logger import get_logger

logger = get_logger(__name__)

# See module docstring's "WHY TRACING IS FORCE-ENABLED" note.
trace.trace_config.enabled = True
trace.trace_config.format = "json"

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="HALO",
    version="1.0",
    description="Day 10 TRiSM demonstrator — the trust artefacts, not just the chat.",
)
app.include_router(approvals_router.router)

# One registry, process-wide — see api/session_registry.py's own docstring
# on why this must not be constructed per-request.
_registry = SessionRegistry()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    customer_id: str | None = None
    customer_context: dict[str, Any] | None = None


def _serialise_agent_result(r: AgentResult) -> dict[str, Any]:
    """
    Full payload included, deliberately — the whole point of this panel
    is showing what an agent actually produced (SHAP values, shortlist,
    hallucination report, ...), not a status summary. 
    """
    return {
        "agent_name": r.agent_name,
        "step_id": r.step_id,
        "success": r.success,
        "tokens_used": r.tokens_used,
        "error": r.error,
        "payload": r.payload,
    }


def _serialise_result(result: OrchestratorResult) -> dict[str, Any]:
    """
    Turn an OrchestratorResult into the JSON the UI panels read: plan, explanation layers, 
    constraints and audit reference.
    """
    explanation_layers: list[str] = []
    citations: list[dict[str, Any]] = []
    for r in result.agent_results:
        if r.agent_name == "ExplainabilityAgent" and r.success:
            explanation_layers = list(r.payload.get("layers_applied") or [])
            citations = list(r.payload.get("rag_citations") or [])

    awaiting_approval = result.final_response == _withheld_message()

    return {
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "routing_decision": result.routing_decision.value,
        "final_response": result.final_response,
        "awaiting_approval": awaiting_approval,
        "success": result.success,
        "total_duration_ms": round(result.total_duration_ms, 1),
        "plan": {
            "source": result.plan_source,
            "accepted": bool(result.plan and result.plan.accepted),
            "steps": list(result.plan.steps) if result.plan else [],
        },
        "agents_invoked": result.agents_invoked,
        "agent_results": [_serialise_agent_result(r) for r in result.agent_results],
        "skipped": result.skipped,
        "collaboration_events": result.collaboration_events,
        "conflicts": result.conflicts,
        "constraint_violations": result.constraint_violations,
        "recovered_agents": result.recovered_agents,
        "explanation_layers": explanation_layers,
        "citations": citations,
        # Set by chat() below, which is the caller that actually has the
        # live AuditLog instance (result itself carries no reference to
        # it) — the line number a reviewer would jump to first.
        "audit_ref": None,
    }


def _withheld_message() -> str:
    from config.settings import settings
    return settings.approval.withheld_message


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """
    Handles one turn. Reuse the Orchestrator for this session_id; customer_id and 
    customer_context only apply on the session's first call.
    """
    orch = _registry.get_or_create(
        req.session_id, customer_id=req.customer_id, customer_context=req.customer_context,
    )
    delivered = orch.collect_all_approved_responses()

    result = orch.process_turn(req.message)
    payload = _serialise_result(result)
    payload["audit_ref"] = f"{orch.audit_log.log_path.name}:{orch.audit_log.write_count}"
    payload["delivered_approvals"] = delivered
    return payload


@app.get("/trace/{turn_id}")
def get_trace(turn_id: str) -> list[dict]:
    """
    Returns the trace events for one turn, in order. An empty list is normal and means no
    such turn ran in this server process.
    """
    return trace.get_captured_events(turn_id)


@app.get("/")
def index() -> FileResponse:
    path = STATIC_DIR / "index.html"
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"static/index.html not found at {path} — see static/ in the repo root.",
        )
    return FileResponse(path)