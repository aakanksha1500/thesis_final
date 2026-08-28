"""
Developer-facing execution tracer. Complements — does NOT replace —
orchestrator/audit_log.py.

  audit_log.py  = the legal record. JSONL, GDPR-minimised, written to disk,
                  read after the fact by a compliance reviewer.
  trace.py      = the development view. Human-readable, nested, timed,
                  written to the terminal while you work.

Enable with TRACE=true in .env. Off by default and near-zero cost when off.
"""
from __future__ import annotations

import contextvars
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import OrderedDict

# Correlation IDs. contextvars (not globals) so concurrent sessions in the
# same process never interleave each other's identity — which is exactly the
# failure mode api/session_registry.py makes possible.
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("sid", default="-")
_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar("tid", default="-")
_depth: contextvars.ContextVar[int] = contextvars.ContextVar("depth", default=0)


@dataclass
class TraceConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("TRACE", "false").lower() == "true"
    )
    format: str = field(default_factory=lambda: os.getenv("TRACE_FORMAT", "human"))
    stream = sys.stdout
    colour: bool = field(
        default_factory=lambda: os.getenv("TRACE_COLOUR", "true").lower() == "true"
        and sys.stdout.isatty()
    )
    max_captured_turns: int = 200


trace_config = TraceConfig()

_C = {"dim": "\033[2m", "cyan": "\033[36m", "green": "\033[32m",
      "yellow": "\033[33m", "red": "\033[31m", "bold": "\033[1m",
      "reset": "\033[0m"}

_captured: "OrderedDict[str, list[dict]]" = OrderedDict()


def _capture(turn_id: str, record: dict) -> None:
    if turn_id not in _captured:
        _captured[turn_id] = []
        while len(_captured) > trace_config.max_captured_turns:
            _captured.popitem(last=False)
    _captured[turn_id].append(record)


def get_captured_events(turn_id: str) -> list[dict]:
    """Every event captured for one turn, in emission order. [] if tracing
    wasn't enabled (or wasn't in json format) when that turn ran — not an
    error, just nothing was ever captured for it."""
    return list(_captured.get(turn_id, []))


def clear_captured_events(turn_id: str | None = None) -> None:
    """Drop one turn's captured events, or every turn's if turn_id is
    None. For tests, and for an API server that wants to bound memory
    more aggressively than max_captured_turns alone."""
    if turn_id is None:
        _captured.clear()
    else:
        _captured.pop(turn_id, None)


def _paint(text: str, colour: str) -> str:
    if not trace_config.colour:
        return text
    return f"{_C.get(colour, '')}{text}{_C['reset']}"


def set_session(session_id: str) -> None:
    _session_id.set(session_id)


def new_turn(turn_id: str | None = None) -> str:
    tid = turn_id or str(uuid.uuid4())[:8]
    _turn_id.set(tid)
    _depth.set(0)
    return tid


def emit(event: str, message: str = "", **fields) -> None:
    """Emit one trace event at the current nesting depth. Never raises."""
    if not trace_config.enabled:
        return
    try:
        if trace_config.format == "json":
            record = {
                "ts": time.time(),
                "session_id": _session_id.get(),
                "turn_id": _turn_id.get(),
                "depth": _depth.get(),
                "event": event,
                "message": message,
                **fields,
            }
            _capture(_turn_id.get(), record)
            print(json.dumps(record, default=str), file=trace_config.stream)
            return

        indent = "│   " * _depth.get()
        extras = "  ".join(f"{k}={v}" for k, v in fields.items())
        line = f" {indent}├─ {_paint(event.ljust(9), 'cyan')} {message}"
        if extras:
            line += _paint(f"   {extras}", "dim")
        print(line, file=trace_config.stream)
    except Exception:
        pass  # a broken tracer must never break the pipeline


@contextmanager
def span(event: str, message: str = "", **fields):
    """
    A timed, nested block. Emits on entry, indents everything inside, and
    emits a duration on exit — including on exception, so a failure still
    reports where it got to and how long it took.
    """
    if not trace_config.enabled:
        yield
        return

    emit(event, message, **fields)
    depth = _depth.get()
    _depth.set(depth + 1)
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        ms = (time.perf_counter() - start) * 1000
        _depth.set(depth)
        emit("✘ FAIL", f"{message} — {type(exc).__name__}: {exc}", duration_ms=round(ms))
        raise
    else:
        ms = (time.perf_counter() - start) * 1000
        _depth.set(depth)
        emit("✔ OK", message, duration_ms=round(ms))


def turn_banner(turn_id: str, session_id: str, turn_no: int, message: str) -> None:
    if not trace_config.enabled:
        return
    bar = "═" * 76
    preview = message if len(message) <= 68 else message[:65] + "..."
    print(f"\n{_paint(bar, 'bold')}", file=trace_config.stream)
    print(f" TURN {turn_id}  ·  session {session_id}  ·  turn #{turn_no}",
          file=trace_config.stream)
    print(f" {_paint('» ' + preview, 'dim')}", file=trace_config.stream)
    print(_paint(bar, "bold"), file=trace_config.stream)
