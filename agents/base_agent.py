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
    subclass. Python will raise TyprError at import time if they are missing -
    catching interface violations before any test runs.
"""

from __future__ import annotations
import time
import abc
import uuid
from dataclasses import dataclass, field
from typing import Any
from utils.logger import get_logger
from utils.llm_client import LLMClient

logger = get_logger(__name__)

@dataclass
class AgentResult:
    """
    Standardised result envelope returned by every agent's run method
    
    Fields used by the evaluation layer:
        - step_id: Unique per run() call; links to AgentBoard step records
        - duration_ms: wall-clock time for this agent's full execution
        - tokens_used: LLM tokens consumed; tracked for cost analysis
        - routing_context - metadata the Orchestrator reads for routing decisions
    """

    agent_name: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    success: bool = True
    playload: dict[str, Any] =field(default_factory=dict)
    raw_llm_output: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: str | None = None
    rag_sources_used: list[str] = field(default_factory=list)
    routing_context: dict[str, Any] = field(default_factory=dict)

    def to_step_record(self) -> dict:
        """
        Serialise to the format expected by evaluation.
        Called by the Orchestrator after each agent execution.
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
    Abstract base class for all specialist agents.
    
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
    ) -> tuple[str, int]:
        """
        Invoke the LLM with timing and error capture.
        
        Args:
            user_message: The primary user-turn content for this call.
            extra_messages: Optional prior turns to pretend (for multi-turn context).
            temperature: Per-call override; None uses client default.
            
        Returns:
            (response_text, tokens_used)
            
        Raises:
            Exception from LLMClient on API failure - let it propogate so the
            Orchestrator's failure handler can act.
        """

        messages = list(extra_messages or [])
        messages.append({"role": "user", "content": user_message})

        start = time.perf_counter()
        response = self.llm.chat(
            system=self.system_prompt,
            messages=messages,
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1

        logger.debug(
            f"[{self.name}] LLM call #{self._call_count} "
            f"- {elapsed_ms:.0f}ms - {response.tokens_used} tokens"
        )
        return response.content, response.tokens_used
    
    def _make_result(
            self,
            payload: dict,
            raw: str = "",
            duratin_ms: float = 0.0,
            tokens: int = 0,
            rag_sources: list[str] | None = None,
            routing_context: dict | None = None,
            error: str | None = None,
    ) -> AgentResult:
        """
        Conveniance constructor for AgentResult - keeps run() method clean.
        """
        return AgentResult(
            agent_name = self.name,
            success = error is None,
            payload = payload,
            raw_llm_output = raw,
            duration_ms = duratin_ms,
            tokens_used = tokens,
            error = error,
            rag_sources_used = rag_sources or [],
            routing_context = routing_context or {},
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, calls={self._call_count})"
    