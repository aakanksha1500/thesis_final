"""
Phase 8b - SessionRegistry: holds one Orchestrator instance per
concurrent user, so the prototype can demostrate mutiple customers
seeking advice at the same time without their sessions interfacing.

This exists because Orchestrator itself is stateful per-session but 
it otherwise stateless in how it's constructed - nothing about it assumes
there's only ever one instance alive. SessionRegistry just gives that
multi-instance usage a single, obvious place to live, instead of leaving
it implicit in whatever calls Orchestrator() directly.

In production this responsibility would sit inside the API layer's request
handling; this in-memory version is intentionally the simplest thing that
lets one demo process serve several concurrent browser sessions.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from data.customer_store import CustomerStore
from orchestrator.orchestrator import Orchestrator
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionEntry:
    orchestrator: Orchestrator
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)


class SessionRegistry:
    """
    Thread-safe in-memory registry of active Orchestrator sessions.
    
    One registry instance should be shared process-wide - NOT one per
    request, or you lose the whole point of it.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        customer_store: CustomerStore | None = None,
        idle_timeout_seconds: float = 3600.0,
    ):
        self._llm_client = llm_client or LLMClient()
        self._customer_store = customer_store or CustomerStore()
        self._idle_timeout = idle_timeout_seconds
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        session_id: str,
        customer_id: str | None = None,
        customer_context: dict[str, Any] | None = None,
    ) -> Orchestrator:
        """
        Return the existing Orchestrator for session_id, or create a new one.

        customer_id / customer_context are only used on FIRST creation of
        this session_id - once a session exists, its customer identity is 
        fixed for the life of that session.
        """
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_active_at = time.time()
                return entry.orchestrator
            
            orch = Orchestrator(
                self._llm_client,
                session_id=session_id,
                customer_id=customer_id,
                customer_context=customer_context,
                customer_store=self._customer_store,
            )
            self._sessions[session_id] = SessionEntry(orchestrator=orch)
            logger.info(
                f"[SessionRegistry] Created session={session_id!r} "
                f"(customer_id={customer_id!r}, "
                f"push_mode={customer_context is not None}). "
                f"Active sessions: {len(self._sessions)}"
            )
            return orch
    
    def end_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
            if existed:
                logger.info(f"[SessionRegistry] Ended session={session_id!r}")
            return existed

    def active_session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def sweep_idle_sessions(self) -> int:
        """Remove sessions inactive longer than idle_timeout_seconds. Returns count removed."""
        cutoff = time.time() - self._idle_timeout
        with self._lock:
            idle = [sid for sid, e in self._sessions.items() if e.last_active_at < cutoff]
            for sid in idle:
                del self._sessions[sid]
        if idle:
            logger.info(f"[SessionRegistry] Swept {len(idle)} idle session(s): {idle}")
        return len(idle)

    def __len__(self) -> int:
        return len(self._sessions)
    