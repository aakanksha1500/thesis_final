"""
Single writer for every results/*.json file.

Two problems this solves:

QW6 - MODE SEGREGATION.
    Results used to be written straight into results/. A mock-mode run
    therefore silently overwrote real-mode dissertation evidence. Results now
    go to results/<llm_mode>/, so a mock run can never clobber a real one.

QW7 - PROVENANCE.
    Every file now carries a _meta block recording exactly what produced it:
    git commit, prompt version, which LLM, and - critically - which fallback
    mode each subsystem was in. A number produced by the hashing embedder and
    the lexical hallucination heuristic is not comparable to one produced by
    sentence-transformers and HHEM, and previously nothing recorded which you
    had.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR

RESULTS_ROOT = ROOT_DIR / "results"


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT_DIR, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"

def _llm_cache_stats() -> dict[str, Any]:
    """Cache hit/miss counters, so a replayed run is identifiable as such."""
    try:
        from utils import llm_cache
        return llm_cache.stats()
    except Exception:
        return {"mode": "unknown"}

def _subsystem_modes() -> dict[str, str]:
    """
    Report which mode each optional subsystem is in, WITHOUT constructing
    anything expensive. Only already-imported singletons are inspected - this
    must never trigger a model download just to write a results file.
    """
    import sys as _sys
    modes: dict[str, str] = {}
    kb = _sys.modules.get("rag.knowledge_base")
    if kb is not None:
        try:
            modes["embedder"] = kb.knowledge_base.embedder.mode
            modes["vector_store"] = kb.knowledge_base.store._backend
        except Exception:
            pass
    hd = _sys.modules.get("rag.hallucination_detector")
    if hd is not None:
        try:
            modes["hallucination_detector"] = hd.hallucination_detector.mode
        except Exception:
            pass
    pd_ = _sys.modules.get("utils.product_data_client")
    if pd_ is not None:
        try:
            modes["product_data"] = pd_.get_product_data_client().mode
        except Exception:
            pass
    md = _sys.modules.get("utils.market_data_client")
    if md is not None:
        try:
            modes["market_data"] = md.market_data_client.mode
        except Exception:
            pass
    return modes


def build_meta(llm_mode: str | None = None) -> dict[str, Any]:
    from config.prompts import PROMPT_VERSION
    from config.settings import settings

    if llm_mode is None:
        from utils.llm_client import LLMClient
        llm_mode = LLMClient().mode

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "prompt_version": PROMPT_VERSION,
        "llm_mode": llm_mode,
        "orchestrator_model": settings.llm.orchestrator_model,
        "specialist_model": settings.llm.specialist_model,
        "judge_model": settings.llm.judge_model,
        "environment": settings.environment,
        "subsystem_modes": _subsystem_modes(),
        "llm_cache": _llm_cache_stats(),
        "product_catalogue_mode": (
            "real_where_available"
            if settings.product_data.use_real_product_data else "synthetic"
        ),
        "explainability_ablation": {
            "use_shap": settings.explainability.use_shap,
            "use_rag_citation": settings.explainability.use_rag_citation,
            "use_counterfactual": settings.explainability.use_counterfactual,
        },
    }


def write_results(
    payload: dict[str, Any],
    filename: str,
    llm_mode: str | None = None,
) -> Path:
    """
    Write `payload` to results/<llm_mode>/<filename> with a _meta block, and
    return the path. Never overwrites a different mode's results.
    """
    meta = build_meta(llm_mode)
    out_dir = RESULTS_ROOT / meta["llm_mode"]
    out_dir.mkdir(parents=True, exist_ok=True)

    from config.settings import settings as _settings
    if _settings.product_data.use_real_product_data:
        stem, dot, ext = filename.rpartition(".")
        filename = f"{stem}__realproducts{dot}{ext}" if dot else f"{filename}__realproducts"

    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"_meta": meta, **payload}, f, indent=2, default=str)
    return path
