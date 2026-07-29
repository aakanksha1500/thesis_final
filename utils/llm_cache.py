"""
WHY
    Two problems at once.

    COST. One investment turn costs ~3,600 tokens; a 10-scenario RQ4 run costs
    ~56,000 against a 100,000/day quota. Re-running an evaluation after a code
    change that did not touch prompts should not cost anything.

    REPRODUCIBILITY — the more important one. An LLM is non-deterministic even
    at temperature 0. Without a cache, nobody (including you next month, or an
    examiner) can regenerate the exact numbers in results/*.json. With a
    committed cache, the evaluation reproduces them byte-for-byte, offline.

MODES  (LLM_CACHE env var)
    off      default. No caching. Every call hits the API.
    record   call the API and store every response. Use for the run that
             produces your final numbers.
    replay   serve from cache only. A miss does NOT call the API — it is
             counted and logged loudly (see below).
    auto     serve from cache when present, otherwise call and record.
             Best for day-to-day development.

WHY A REPLAY MISS DOES NOT RAISE
    Every agent here wraps its LLM call in try/except and falls back to
    deterministic template text. Raising on a cache miss would therefore be
    swallowed, and the run would look successful while quietly using templates


KEY
    sha256 over (model, temperature, system, messages). Deliberately NOT
    including timestamps, session ids or turn ids — those change every run and
    would make every entry a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(
    os.getenv("LLM_CACHE_DIR", str(ROOT_DIR / "data" / "llm_cache"))
)
MODE = os.getenv("LLM_CACHE", "off").lower()

_VALID_MODES = {"off", "record", "replay", "auto"}

if MODE not in _VALID_MODES:
    logger.warning(
        f"[LLMCache] Unknown LLM_CACHE={MODE!r} — treating as 'off'. "
        f"Valid: {sorted(_VALID_MODES)}"
    )
    MODE = "off"

_lock = threading.Lock()
_stats = {
    "hits": 0,
    "misses": 0,
    "writes": 0,
    "errors": 0,
}


def enabled() -> bool:
    """Return True if caching is enabled."""
    return MODE != "off"


def make_key(
    model: str,
    system: str,
    messages: list[dict],
    temperature: float | None,
) -> str:
    """
    Stable hash of everything that determines the response.

    Keys are sorted so dict ordering can never produce two entries
    for the same logical call.
    """
    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(key: str) -> Path:
    """
    Two-level directory fan-out to avoid huge directories.
    """
    return CACHE_DIR / key[:2] / f"{key}.json"


def get(key: str) -> dict[str, Any] | None:
    """
    Return a cached response dict, or None.

    Never raises.
    """
    if MODE not in ("replay", "auto"):
        return None

    try:
        path = _path(key)

        if not path.exists():
            with _lock:
                _stats["misses"] += 1

            if MODE == "replay":
                logger.error(
                    f"[LLMCache] REPLAY MISS {key[:12]} — no cached response. "
                    f"This call will NOT be made; the agent will fall back to "
                    f"template text. Re-record with LLM_CACHE=record."
                )

            return None

        entry = json.loads(path.read_text(encoding="utf-8"))

        with _lock:
            _stats["hits"] += 1

        return entry["response"]

    except Exception as exc:
        with _lock:
            _stats["errors"] += 1

        logger.warning(f"[LLMCache] Read failed for {key[:12]}: {exc}")
        return None


def put(
    key: str,
    response: dict[str, Any],
    request_meta: dict[str, Any] | None = None,
) -> None:
    """
    Store a response.

    Never raises—a cache failure must not break a turn.
    """
    if MODE not in ("record", "auto"):
        return

    try:
        path = _path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "request": request_meta or {},
                    "response": response,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        with _lock:
            _stats["writes"] += 1

    except Exception as exc:
        with _lock:
            _stats["errors"] += 1

        logger.warning(f"[LLMCache] Write failed for {key[:12]}: {exc}")


def is_replay_miss_blocking() -> bool:
    """
    In strict replay mode, a cache miss means we must NOT call the API.
    """
    return MODE == "replay"


def stats() -> dict[str, Any]:
    """
    Return a snapshot of cache statistics for embedding in results metadata.
    """
    with _lock:
        s = dict(_stats)

    s["mode"] = MODE
    s["cache_dir"] = str(CACHE_DIR)

    total = s["hits"] + s["misses"]
    s["hit_rate"] = round(s["hits"] / total, 4) if total else None

    # Indicates whether this was a clean replay run.
    s["clean_replay"] = (
        MODE == "replay"
        and s["misses"] == 0
        and s["errors"] == 0
    )

    return s


def reset_stats() -> None:
    """Reset hit/miss/write/error counters."""
    with _lock:
        for key in _stats:
            _stats[key] = 0
