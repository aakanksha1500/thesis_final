"""
LLM Abstraction Layer - the single point of contact between the system
and any external language model API.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class LLMResponse:
    """
    Standardised response envelope returned by every LLMClient.chat() call.
    Agents depend on this type - not a provider spefic response objects.
    """
    content: str 
    tokens_used: int
    model: str

class LLMClient:
    """
    Wrapper class around LLM provider APIs.
    
    Initialisation is intentionally cheap and safe - it will never raise
    even if the API key is missing. Missing key = mock mode, logged at
    WARNING level.

    Args:
        model: Override the default model string. If None, reads from 
               ORCHESTRATOR_MODEL env var, else falls back to gpt-4o-mini.
        temprature: Default sampling temprature. Agents may override
                    per-call by passing trmprature = to chat().
                    max_tokens: HArd calling on response length.   
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPRATURE = 0.2
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str | None = None,
        temprature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.model = model or os.getenv("ORCHESTRATOR_MODEL", self.DEFAULT_MODEL)
        self.temprature = temprature if temprature is not None else self.DEFAULT_TEMPRATURE
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        
        self._client = None
        self._mode = "mock"
        self._init_client()

    def _init_client(self) -> None:
        """
        Attempt to initialise the real OpenAI client.
        Falls back to mode silently if:
            - OpenAI package not installed yet
            - OPENAI_API_KEY not set in .env
        """

        api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            logger.warning(
                "[LLMClient] OPENAI_API_KEY not set. LLMClient will run in mock mode."
                "Set it in .env to activate real LLM API calls."
            )
            return
        
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._mode = "openai"
            logger.info(f"[LLMClient] Initialised in REAL mode — model={self.model}")
        except ImportError:
            logger.warning(
                "[LLMClient] OpenAI package not installed. LLMClient will run in mock mode."
                "Add it to requirements.txt and run pip install -r requirements.txt."
            )

    def chat(
            self,
            system: str,
            messages: list[dict],
            temprature: float | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            system: System prmpt string (agent-specific, from config/prompts.py)
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            temprature: Per-call override. If None, uses client default.

        Returns:
            LLMResponse with content, tokens_used, and model name.
        
        Raises:
            Exception: Only in REAL mode if the API call fails. Mock mode never raises.
        """
        temp = temprature if temprature is not None else self.temprature

        if self._mode == "mock" or self._client is None:
            return self._mock_response(system, messages)
        
        all_messages = [{"role": "system", "content": system}] + messages
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=all_messages,
                temperature=temp,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            logger.debug(f"[LLMClient] {tokens} tokens used -  model = {self.model}")
            return LLMResponse(content=content, tokens_used=tokens_used, model=self.model)
        except Exception as e:
            logger.error(f"[LLMClient] API call failed: {e}")
            raise

    def _mock_response(self, system: str, messages: list[dict]) -> LLMResponse:
            """
            Return a clearly labelled mock response for development and testing.
            Content encodes the system prompt role so tests can assert routing 
            without any API calls.
            """
            # Extarct first 60 chars of system prompt to identify which agent called
            role_hint = system[:60].replace("\n", " ")
            last_user = next(
                (m["content"][:40] for m in reversed(messages) if m.get("role") == "user"),
                "no user message",
            )
            mock_content = (
                f"[MOCK RESPONSE] "
                f"Agent role: '{role_hint}...' | "
                f"User said: '{last_user}...' | "
                f"Set OPENAI_API_KEY in .env for real API responses."
            )
            return LLMResponse(content=mock_content, tokens_used=0, model="mock")
    
    @property
    def mode(self) -> str:
        """Returns 'mock' or 'openai' - useful for test assertions."""
        return self._mode
    
    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, mode={self._mode!r})"
    