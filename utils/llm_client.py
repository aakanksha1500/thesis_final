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
            self._client = openai.OpenAI(api_key=api_key)
            self._mode = "openai"
            logger.info(f"[LLMClient] Initialised in real mode - model {self.model}.")
        except ImportError:
            logger.warning(
                "[LLMClient] OpenAI package not installed. LLMClient will run in mock mode."
                "Add it to requirements.txt and run pip install -r requirements.txt."
            )