"""
LLM Abstraction Layer - the single point of contact between the system
and any external language model API.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from config.settings import settings
from utils import llm_cache
from utils.logger import get_logger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_TEST_MAX_RETRY_SLEEP = 5.0

@dataclass
class LLMResponse:
    """
    Standardised response envelope returned by every LLMClient.chat() call.
    Agents depend on this type - not a provider specific response objects.
    """
    content: str
    tokens_used: int
    model: str

def _is_retryable(exc: Exception) -> bool:
    """
    True for transient failures only. A 429 or 5xx is worth another go; a 400
    (malformed request) or 401 (bad key) is not.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status in _RETRYABLE_STATUS:
        return True

    text = str(exc).lower()
    return any(
        s in text
        for s in (
            "rate limit",
            "429",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "503",
            "502",
            "overloaded"
        )
    )

def _max_retry_sleep() -> float:
    import os as _os  # noqa: PLC0415
    return _TEST_MAX_RETRY_SLEEP if _os.getenv("PYTEST_CURRENT_TEST") else 120.0

def _retry_delay(exc: Exception, attempt: int) -> float:
    """
    Honour Retry-After if present; otherwise exponential backoff with jitter.
    """
    import re as _re

    ceiling = _max_retry_sleep()

    match = _re.search(r"try again in (\d+)m([\d.]+)s", str(exc))
    if match:
        return min(float(match.group(1)) * 60 + float(match.group(2)), ceiling)
    match = _re.search(r"try again in ([\d.]+)s", str(exc))

    if match:
        return min(float(match.group(1)), ceiling)

    return min(2.0 ** attempt, 30.0, ceiling) + random.uniform(0, 0.5)

class LLMClient:
    """
    Wrapper class around LLM provider APIs.

    Initialisation is intentionally cheap and safe - it will never raise
    even if the API key is missing. Missing key = mock mode, logged at
    WARNING level.

    Args:
        model: Override the default model string. If None, reads from
               ORCHESTRATOR_MODEL env var, else falls back to gpt-4o-mini.
        temperature: Default sampling temperature. Agents may override
                    per-call by passing tempreature = to chat().
                    max_tokens: Hard on response length.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        force_mock: bool = False,
    ):
        self.model = model or os.getenv("ORCHESTRATOR_MODEL", self.DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        self._client = None
        self._mode = "mock"
        self._force_mock = force_mock
        self._init_client()

    _PROVIDER_BASE_URLS = {
        "openai": "https://api.openai.com/v1",  # default OpenAI endpoint
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "groq": "https://api.groq.com/openai/v1",
        "together": "https://api.together.xyz/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": "http://localhost:11434/v1",  # local; override with OLLAMA_BASE_URL
    }

    # Providers that don't require a real API key. Ollama's OpenAI-compatible
    # server doesn't check the key at all, but the openai SDK still requires
    # a non-empty string to construct the client.
    _NO_AUTH_PROVIDERS = {"ollama"}

    def _init_client(self) -> None:
        """
        Attempt to initialise the real LLM client.
        Falls back to mode silently if:
            - OpenAI package not installed yet
            - relevant API key not set in .env
            - LLM_PROVIDER names an unknown provider
        Provider is selected via the LLM_PROVIDER env var (default: "openai").
        Each provider reads its key from a provider-specific env var:
          openai     -> OPENAI_API_KEY
          groq       -> GROQ_API_KEY
          together   -> TOGETHER_API_KEY
          openrouter -> OPENROUTER_API_KEY
          ollama     -> none required (local server, no auth).
                        Base URL defaults to http://localhost:11434/v1,
                        override with OLLAMA_BASE_URL. Model name should be
                        whatever you've pulled, e.g. "llama3.1:8b".
        """
        if self._force_mock:
            logger.info("[LLMClient] force_mock=True — running in MOCK mode regardless of environment.")
            return
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider not in self._PROVIDER_BASE_URLS:
            logger.warning(
                f"[LLMClient] Unknown LLM_PROVIDER={provider!r} — running in MOCK mode. "
                f"Known providers: {list(self._PROVIDER_BASE_URLS)}"
            )
            return
        key_env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(key_env_var, "")
        if not api_key and provider in self._NO_AUTH_PROVIDERS:
            api_key = "ollama"  # dummy — the local server ignores this value
        if not api_key:
            logger.warning(
                f"[LLMClient] {key_env_var} not set — running in MOCK mode. "
                f"Set it in .env to activate real LLM calls via {provider}."
            )
            return

        try:
            import openai
            base_url = (
                os.getenv("OLLAMA_BASE_URL", self._PROVIDER_BASE_URLS[provider])
                if provider == "ollama"
                else self._PROVIDER_BASE_URLS[provider]
            )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            self._mode = provider
            logger.info(f"[LLMClient] Initialised in REAL mode — provider={provider} model={self.model}")
        except ImportError:
            logger.warning(
                "[LLMClient] OpenAI package not installed. LLMClient will run in mock mode."
                "Add it to requirements.txt and run pip install -r requirements.txt."
            )

    def chat(
            self,
            system: str,
            messages: list[dict],
            temperature: float | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            system: System prompt string (agent-specific, from config/prompts.py)
            messages: List of {"role": "user"|"assistant", "content": str} dicts.
            temperature: Per-call override. If None, uses client default.

        Returns:
            LLMResponse with content, tokens_used, and model name.

        Raises:
            Exception: Only in REAL mode if the API call fails. Mock mode never raises.
        """
        temp = temperature if temperature is not None else self.temperature

        if self._mode == "mock" or self._client is None:
            return self._mock_response(system, messages)

        all_messages = [{"role": "system", "content": system}] + messages

        cache_key = ""

        if llm_cache.enabled():
            cache_key = llm_cache.make_key(self.model, system, messages, temp)
            cached = llm_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"[LLMClient] cache HIT {cache_key[:12]}")
                return LLMResponse(**cached)
            if llm_cache.is_replay_miss_blocking():
                raise RuntimeError(
                    f"LLM_CACHE=replay and no cached response for {cache_key[:12]}"
                )


        last_exc: Exception | None = None
        for attempt in range(settings.llm.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=all_messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    timeout=settings.llm.timeout_seconds,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= settings.llm.max_retries or not _is_retryable(exc):
                    raise
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    f"[LLMClient] attempt {attempt + 1}/"
                    f"{settings.llm.max_retries + 1} failed "
                    f"({type(exc).__name__}) — retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise last_exc  # type: ignore[misc]

        try:
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            logger.debug(f"[LLMClient] {tokens_used} tokens used -  model = {self.model}")

            if cache_key:
                llm_cache.put(
                    cache_key,
                    {"content": content, "tokens_used": tokens_used, "model": self.model},
                    request_meta={"model": self.model, "temperature": temp,
                                  "system_preview": system[:120]},
                )

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
            # Extract first 60 chars of system prompt to identify which agent called
            role_hint = system[:60].replace("\n", " ").strip()
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