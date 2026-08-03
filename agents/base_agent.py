"""
Added before ConversationalAgent so that the contract every agent must
satisfy is settled first.

Three things BaseAgent provides:
1. AgentResult envelope
    Every agent returns the same dataclass regardless of what it omputes.
    The step_id and duration_ms fields feed AgentBoard step-level evaluation
2._call_llm() wrapper
    Centralises timing, error logging, and the mock/real switch. Agents never
    call self.llm.chat() directly, they call self._call_llm() so latency is
    always measured and errors are always caught at one place.
3. Abstract interface
    system_prompt, _parse_response, and run() must be implemented by every
    subclass. Python will raise TypeError at import time if they are missing -
    catching interface violations before any test runs.
"""

from __future__ import annotations

import abc
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agents import payloads
from utils import trace
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AgentResult:
    """
    Standard envelope every agent.run() returns, regardless of what the
    agent actually computed. Keeping this identical across agents is what
    lets the evaluation layer (AgentBoard-style step metrics) and the
    Orchestrator process any agent's output the same way.
    """

    # agent_name        -> which agent produced this (for step_records/logs)
    # step_id           -> unique per-call id; links this result to a step
    #                      record for process-level evaluation (E1/E2)
    # success           -> False only if something raised; run() itself
    #                      never raises, it captures the error here instead
    # payload           -> the actual agent-specific output (risk_class,
    #                      recommendation text, etc.)
    # duration_ms       -> wall-clock latency, used for cost/latency analysis
    # tokens_used       -> LLM token spend, used for cost analysis
    # rag_sources_used  -> citations, feeds the RAG explainability layer (X2b)
    #                      (currently always empty — no RAG layer exists yet)
    # routing_context   -> small dict the Orchestrator would read to decide
    #                      what to call next (not consumed anywhere yet,
    #                      since there's no Orchestrator)

    agent_name: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    success: bool = True
    payload: dict[str, Any] =field(default_factory=dict)
    raw_llm_output: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    rag_sources_used: list[str] = field(default_factory=list)
    routing_context: dict[str, Any] = field(default_factory=dict)

    def to_step_record(self) -> dict:
        """
        Converts this result into the flat dict format the (planned)
        AgentBoard-style step evaluator expects: one record per agent
        invocation, so failures can be localised to a specific step
        instead of only seeing a wrong final answer.
        """
        return {
            "step_id": self.step_id,
            "agent": self.agent_name,
            "completed": self.success,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }
class BaseAgent(abc.ABC):
    """
    Abstract parent for every specialist agent. Subclasses MUST implement
    system_prompt, _parse_response(), and run() — Python raises TypeError
    at instantiation time if any is missing, which catches an incomplete
    agent before it's ever run.

    Subclass must implement:
        - system_prompt: property returning the agent's system prompt string
        - _parse_response(): converts raw LLM string output to structured dict
        - run(): main entry point callled by th eOrchestrator
    """

    def __init__(self, llm_client: LLMClient, name: str):
        self.llm = llm_client
        self.name = name
        self._call_count = 0

    # Abstract interface -  must be implemented by every subclass

    @property
    @abc.abstractmethod
    def system_prompt(self) -> str:
        """
        Return this agent's system prompt from config/prompts.py.
        """
        ...

    @abc.abstractmethod
    def _parse_response(self, raw_output: str) -> dict[str, Any]:
        """
        Parse raw LLM string output into a structured payload dict.
        Called inside run() after every LLM call.
        Should never raise - return {"parse_error": raw} on failure.
        """
        ...

    @abc.abstractmethod
    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Execute this agent's task.

        Args:
            context:  dict assembled by the Orchestrator containing at minimum:
                'user_message' (str), 'conversation_history' (list),
                and any outputs from agents that run before this one.

            Returns:
                AgentResult with success = True on completion, success=False on error.
                Never raises - errors are captured in AgentResult.error.
        """
        ...

    # Shared utilities - available to all subclasses

    def _call_llm(
            self,
            user_message: str,
            extra_messages: list[dict] | None = None,
            temperature: float | None = None,
            system_override: str | None = None,
    ) -> tuple[str, int]:
        """
        The only path an agent should use to talk to the LLM. Centralising
        this means latency timing and error handling happen in exactly one
        place instead of being duplicated (and inconsistently applied) in
        every agent.

        Args:
            user_message: The primary user-turn content for this call.
            extra_messages: Optional prior turns to pretend (for multi-turn context).
            temperature: Per-call override; None uses client default.
            system_override: Use this system message instead of
                self.system_prompt for this one call. For a lightweight,
                narrowly-scoped call (e.g. classification) where the
                agent's full persona/instruction system prompt is mostly
                irrelevant overhead — see INTENT_CLASSIFIER_SYSTEM.

        Returns:
            (response_text, tokens_used)

        Raises:
            Exception from LLMClient on API failure - let it propogate so the
            Orchestrator's failure handler can act.
        """

        messages = list(extra_messages or [])
        messages.append({"role": "user", "content": user_message})

        start = time.perf_counter()
        trace.emit("PROMPT",
                   (user_message.strip().splitlines() or [""])[0][:48],
                   chars=len(user_message), temp=temperature)

        response = self.llm.chat(
            system=system_override if system_override is not None else self.system_prompt,
            messages=messages,
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1

        trace.emit("LLM", f"← {response.tokens_used} tok",
                   duration_ms=round(elapsed_ms),
                   model=response.model, mode=self.llm.mode)

        logger.debug(
            f"[{self.name}] LLM call #{self._call_count} "
            f"- {elapsed_ms:.0f}ms - {response.tokens_used} tokens"
        )
        return response.content, response.tokens_used

    def _make_result(
            self,
            payload: dict,
            raw: str = "",
            duration_ms: float = 0.0,
            tokens: int = 0,
            rag_sources: list[str] | None = None,
            routing_context: dict | None = None,
            error: str | None = None,
    ) -> AgentResult:
        """
        Convenience constructor so run() methods don't have to build an
        AgentResult by hand every time — keeps each agent's run() focused
        on its own logic instead of envelope bookkeeping.
        """
        payloads.enforce(self.name, payload, success=error is None)
        return AgentResult(
            agent_name = self.name,
            success = error is None,
            payload = payload,
            raw_llm_output = raw,
            duration_ms = duration_ms,
            tokens_used = tokens,
            error = error,
            rag_sources_used = rag_sources or [],
            routing_context = routing_context or {},
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, calls={self._call_count})"