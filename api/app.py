"""
api/app.py — Day 10, the UI's server.

FastAPI + one HTML file. Not a product — a TRiSM demonstrator: what makes
it worth showing is that it exposes the trust artefacts (plan, explanation
layers, constraints, approval state, audit reference), not just the chat
text. See static/index.html for the actual panel.

SOFT DEPENDENCY ON FASTAPI — see api/approvals_router.py's docstring for
the reasoning; this file follows the identical pattern. Unlike that
router, this file's entire purpose requires FastAPI to exist at all, so
there's no "core mechanism works without it" claim to make here — the
core mechanism is run_demo.py, and it remains fully independent of
everything in this file.

ENDPOINTS
    POST /chat              -> one turn result, JSON (see _serialise_result)
    GET  /trace/{turn_id}   -> captured utils.trace events for that turn
    GET  /approvals         -> from api/approvals_router.py, included whole
    POST /approvals/{id}/approve|reject  -> likewise
    GET  /                  -> static/index.html

WHY TRACING IS FORCE-ENABLED AT STARTUP, NOT LEFT TO .env
    TRACE defaults to false everywhere else in this codebase (near-zero
    cost when off, per utils/trace.py's own docstring) — correct for
    run_demo.py, where trace output is just console noise most of the
    time. This server's whole purpose is serving GET /trace/{turn_id};
    leaving that empty because nobody remembered to export TRACE=true
    would make the one endpoint that makes this a "TRiSM demonstrator"
    rather than a plain chat box silently do nothing. So this module
    overrides trace_config at import time — deliberate, not a default
    left lying around.

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
from orchestrator.approvals import get_approval_store
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
    hallucination report, ...), not a status summary. AgentResult.
    to_step_record() (agents/base_agent.py) exists for a different
    consumer (the planned AgentBoard step evaluator) and drops payload
    entirely, which is exactly the part this UI needs.
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
    One JSON shape covering every panel in the mockup (see module
    docstring): PLAN, EXPLANATION LAYERS (read straight from
    ExplainabilityAgent's own payload.layers_applied — already exactly
    what that panel needs, no separate extraction), CONSTRAINTS,
    approval state, and an audit reference. Not derived from captured
    trace events — those are supplementary detail (GET /trace/{turn_id}),
    not the primary data source, so this endpoint still returns
    something complete even when TRACE-based capture is empty (a fresh
    server restart, or a turn that ran before this process's tracing
    was live).
    """
    explanation_layers: list[str] = []
    for r in result.agent_results:
        if r.agent_name == "ExplainabilityAgent" and r.success:
            explanation_layers = list(r.payload.get("layers_applied") or [])

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
    One turn. Reuses (never creates a second) Orchestrator per session_id
    via SessionRegistry — customer_id/customer_context only take effect
    on that session's FIRST call, exactly as SessionRegistry documents.

    Checks for newly-delivered approvals FIRST, same as run_demo.py's -i
    loop does before its own input() prompt — a reviewer deciding a held
    turn (elsewhere: another tab, run_demo.py --approve, the API
    directly) should surface here on this session's next message, not
    require the customer to somehow know to ask again. Without this,
    "approve it, watch it deliver" (the build plan's own demo line) would
    only actually work from the CLI, not from this UI — found by tracing
    through what the UI's own copy promises, not assumed to already work.
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
    Raw captured trace events for one turn, emission order. [] is a
    normal, valid response — it means no turn with this id has run in
    THIS server process since it started (captures are in-memory, not
    persisted — see utils/trace.py's module docstring), not an error;
    left as 200 rather than 404 for exactly that reason.
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