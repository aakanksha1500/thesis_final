#!/usr/bin/env python3
"""
One command that regenerates every results/*.json the dissertation cites, in
dependency order, with provenance, cost accounting, and a diff against what was
there before.


WHY THIS SCRIPT EXISTS

Twelve patches changed things that MOVE THE NUMBERS:

    06  a trained risk model replaced the heuristic          → RQ1
    10  model tiers, judge dimensions, retry on 429          → RQ4
    11  R23 prohibited-phrase regex, R24 advisory-scoped
        disclaimers, R33 explainability on the budget route  → RQ2, RQ3, RQ4
    12  an eighth intent bucket (full_advisory)              → RQ4 routing

Every committed results file therefore describes code that no longer exists.
Regenerating them one at a time, by hand, over several days, on an LLM whose
output is non-deterministic, is how a results table ends up internally
inconsistent — half the numbers from one commit, half from another, and no
record of which.

So this script does four things a manual re-run cannot:

  1. ARCHIVES the previous results before overwriting anything. A regeneration
     that destroys the evidence you were going to compare against is worse than
     no regeneration.

  2. ORDERS the stages by dependency. The risk model must be trained before RQ1
     evaluates it; the knowledge base must be built before RQ5 measures RAG
     grounding. Running RQ1 first silently measures the OLD model.

  3. RECORDS every LLM call (LLM_CACHE=record), so the whole suite replays
     offline, byte-for-byte, for free. This is what makes the numbers
     reproducible by an examiner rather than merely reported by you.

  4. DIFFS old against new and prints what moved, so "RQ4 routing accuracy fell
     4 points" is something you notice here rather than in your viva.


USAGE

    python scripts/regenerate_evidence.py --plan          # show stages + cost, run nothing
    python scripts/regenerate_evidence.py --preflight     # check readiness only
    python scripts/regenerate_evidence.py                 # full regeneration (records cache)
    python scripts/regenerate_evidence.py --only rq4      # one stage
    python scripts/regenerate_evidence.py --skip rq5      # everything but one
    python scripts/regenerate_evidence.py --replay        # re-run from cache, zero cost
    python scripts/regenerate_evidence.py --compare-only  # diff archive vs current, no runs

BUDGET
    Groq's free tier is 100,000 tokens/day. A full regeneration is estimated
    below at ~85,000. That is close enough to the ceiling that --plan exists:
    look at it before you start, and use --only to split across two days if
    you would rather not risk a 429 halfway through RQ4.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
ARCHIVE = ROOT / "results_archive"

BAR = "═" * 78
SUB = "─" * 78



# The stage graph

@dataclass
class Stage:
    key: str
    label: str
    command: list[str]
    produces: list[str]
    est_tokens: int
    rq: str
    # Stages that must have succeeded first. Enforced, not documented — a
    # wrong ORDER here silently evaluates a stale artefact and the result
    # looks perfectly valid.
    depends_on: list[str] = field(default_factory=list)
    notes: str = ""


PYTEST = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider"]


STAGES: list[Stage] = [
    Stage(
        key="index",
        label="Rebuild RAG index with the current embedder",
        command=[sys.executable, "scripts/build_knowledge_base.py"],
        produces=[],
        est_tokens=0,
        rq="—",
        notes="No LLM. Must precede RQ5 and any RAG-citation measurement: an "
              "index built by a different embedder scores pure noise, and "
              "nothing errors when it does.",
    ),
    Stage(
        key="riskmodel",
        label="Train the distress model (GMSC [D8])",
        command=[sys.executable, "scripts/train_risk_model.py"],
        produces=["rq1_risk_model_training.json"],
        est_tokens=0,
        rq="RQ1",
        notes="No LLM. Must precede RQ1 — otherwise RQ1 evaluates whatever "
              ".pkl happens to be on disk, which may be from an older feature "
              "derivation.",
    ),
    Stage(
        key="rq1",
        label="RQ1 — risk classification (hybrid vs rule-only)",
        command=PYTEST + ["tests/unit/test_risk_profiling_agent.py::TestRQ1Evaluation"],
        produces=["phase3_risk_baseline.json"],
        est_tokens=2_000,
        rq="RQ1",
        depends_on=["riskmodel"],
    ),
    Stage(
        key="rq2",
        label="RQ2 — investment shortlist quality",
        command=PYTEST + ["tests/unit/test_investment_agent.py::TestRQ2Evaluation"],
        produces=["rq2_investment_baseline.json"],
        est_tokens=12_000,
        rq="RQ2",
        depends_on=["riskmodel"],
        notes="Moved by R23 (prohibited-phrase regex now catches evasions the "
              "substring check missed).",
    ),
    Stage(
        key="rq3",
        label="RQ3 — explainability ablation (3 conditions)",
        command=PYTEST + ["tests/unit/test_explainability_agent.py::TestRQ3AblationEvaluation"],
        produces=["rq3_shap_only.json", "rq3_shap_rag.json", "rq3_full_stack.json"],
        est_tokens=9_000,
        rq="RQ3",
        depends_on=["index"],
        notes="Moved by R33 — the budget route now emits an explainability "
              "layer, so ablation coverage changed.",
    ),
    Stage(
        key="rq4",
        label="RQ4 — multi-agent coherence (10 scenarios + judge)",
        command=PYTEST + ["tests/integration/test_orchestrator_pipeline.py::TestRQ4Evaluation"],
        produces=["rq4_mas_coherence.json"],
        est_tokens=52_000,
        rq="RQ4",
        depends_on=["riskmodel", "index"],
        notes="The expensive one, and the one four separate patches moved: "
              "R5 judge dimensions, R10 model tiers, R24 disclaimer scoping, "
              "R7 the eighth intent bucket.",
    ),
    Stage(
        key="rq5",
        label="RQ5 — FinQA grounding (with and without RAG)",
        command=PYTEST + ["tests/unit/test_rq5_finqa_evaluation.py::TestRQ5FinQAEvaluation"],
        produces=["rq5_finqa_no_rag.json", "rq5_finqa_with_rag.json"],
        est_tokens=6_000,
        rq="RQ5",
        depends_on=["index"],
    ),
    Stage(
        key="intent",
        label="Banking77 intent-classification baseline",
        command=PYTEST + ["tests/unit/test_conversational_agent.py::TestIntentAccuracyEvaluation"],
        produces=["phase2_conversational_baseline.json"],
        est_tokens=3_000,
        rq="RQ4 (supporting)",
        notes="Moved by R7 — an eighth bucket changes the confusion matrix. "
              "Check full_advisory is not stealing from investment_advice.",
    ),
    Stage(
        key="budget",
        label="Budget benchmark baseline (Ireland HBS [D11])",
        command=PYTEST + ["tests/unit/test_budget_agent.py::TestBudgetAgentRun"],
        produces=["phase5_budget_baseline.json"],
        est_tokens=1_500,
        rq="— (validates benchmark logic)",
    ),
    Stage(
        key="customer",
        label="Existing-customer + push-mode baselines",
        command=PYTEST + [
            "tests/integration/test_existing_customer_baseline.py",
            "tests/integration/test_push_mode_baseline.py",
        ],
        produces=["phase8b_existing_customer_baseline.json",
                  "phase8b_push_mode_baseline.json"],
        est_tokens=4_000,
        rq="— (flow coverage)",
    ),
]

STAGE_BY_KEY = {s.key: s for s in STAGES}



# Preflight — refuse to produce evidence from a degraded pipeline

def preflight(require_llm: bool = True) -> list[str]:
    """
    Return a list of blocking problems. Empty means go.

    Every check here corresponds to a bug that already happened once and
    produced plausible-looking numbers: a missing key silently mocked every
    response, a stale index silently scored noise, a missing .pkl silently ran
    the heuristic. None of them raised.
    """
    problems: list[str] = []

    # 1. .env actually loaded
    try:
        from config.settings import settings  # noqa: F401
    except Exception as exc:
        return [f"cannot import config.settings: {exc}"]

    # 2. LLM is in real mode
    if require_llm:
        from utils.llm_client import LLMClient
        client = LLMClient()
        if client.mode == "mock":
            problems.append(
                "LLMClient is in MOCK mode — no API key found. Every 'result' "
                "would be [MOCK RESPONSE] text scored as if it were real. "
                "Set LLM_PROVIDER and the matching *_API_KEY in .env."
            )
        else:
            print(f"  ok    LLM provider = {client.mode}, model = {client.model}")

    # 3. Trained risk model present and loadable
    from config.settings import settings
    model_path = Path(settings.risk.model_path)
    if not model_path.exists():
        problems.append(
            f"no trained risk model at {model_path} — RQ1 would measure the "
            f"heuristic while the write-up implies a trained model. "
            f"Run: python scripts/train_risk_model.py"
        )
    else:
        print(f"  ok    risk model present ({model_path.name})")

    # 4. Index exists and records which embedder built it
    index_dir = ROOT / "data" / "embeddings" / "rag_index"
    meta_file = index_dir / "meta.json"
    if not meta_file.exists():
        problems.append(
            f"no RAG index at {index_dir} — run: python scripts/build_knowledge_base.py"
        )
    else:
        meta = json.loads(meta_file.read_text())
        stamped = meta.get("embedder_model") or meta.get("embedder_mode")
        if not stamped:
            problems.append(
                "the RAG index records no embedder, so there is no way to know "
                "whether its vectors match your current one. If they do not, "
                "every relevance score is noise and nothing errors. "
                "Rebuild: rm -rf data/embeddings/rag_index && "
                "python scripts/build_knowledge_base.py"
            )
        else:
            print(f"  ok    index stamped with embedder = {stamped}")
        if meta.get("format_version", 1) < 2:
            print("  warn  index is a legacy pickle index (format v1); "
                  "rebuilding it also migrates to the safe JSON+npy format")

    # 5. Working tree clean enough for provenance to mean anything
    sha, dirty = _git_state()
    if dirty:
        print("  warn  working tree is DIRTY — every results file will be "
              "stamped <sha>-dirty and cannot be reproduced from a commit. "
              "Commit first if these are your final numbers.")
    else:
        print(f"  ok    working tree clean at {sha}")

    return problems


def _git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT, stderr=subprocess.DEVNULL, timeout=5).decode().strip())
        return sha, dirty
    except Exception:
        return "unknown", False



# Archiving

def archive_existing() -> Path | None:
    """
    Copy results/ aside before anything overwrites it.

    Not optional and not prompted for. The single most expensive mistake
    available here is regenerating on top of the numbers you were going to
    compare against — the old run cost real tokens and may not be reproducible
    if prompts changed.
    """
    if not RESULTS.exists() or not any(RESULTS.rglob("*.json")):
        print("  (no existing results to archive)")
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha, dirty = _git_state()
    dest = ARCHIVE / f"{stamp}_{sha}{'-dirty' if dirty else ''}"
    shutil.copytree(RESULTS, dest)
    n = len(list(dest.rglob("*.json")))
    print(f"  archived {n} result files → results_archive/{dest.name}/")
    return dest


def latest_archive() -> Path | None:
    if not ARCHIVE.exists():
        return None
    runs = sorted(p for p in ARCHIVE.iterdir() if p.is_dir())
    return runs[-1] if runs else None



# Comparison

# Metrics worth surfacing per file. Kept explicit rather than diffing every
# key: a full diff of a results file is dominated by per-scenario detail and
# the headline number gets lost in it.
HEADLINE_KEYS = [
    "routing_accuracy", "css", "tue", "step_progress_rate",
    "f1", "macro_f1", "accuracy", "exact_match", "hallucination_rate",
    "rar", "hybrid_vs_rule_only_delta", "tps", "tci",
    "constraint_violations", "n_violations", "auc_advisory_features",
    "slot_fill_rate", "recovery_rate",
]


def _flatten(obj, prefix="") -> dict:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_meta":
                continue
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = obj
    return out


def _headline(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    flat = _flatten(data)
    return {
        k: v for k, v in flat.items()
        if any(k.split(".")[-1] == h for h in HEADLINE_KEYS)
    }


def compare(archive_dir: Path | None) -> list[dict]:
    """Diff headline metrics: archive → current. Returns rows for the manifest."""
    if archive_dir is None:
        print("  (nothing to compare against — this is the first archived run)")
        return []

    rows = []
    for new_path in sorted(RESULTS.rglob("*.json")):
        if new_path.name == "EVIDENCE_MANIFEST.json":
            continue        # the manifest describes the run, it is not evidence
        rel = new_path.relative_to(RESULTS)
        old_path = archive_dir / rel
        if not old_path.exists():
            rows.append({"file": str(rel), "metric": "—", "old": None,
                         "new": None, "status": "NEW FILE"})
            continue
        old_m, new_m = _headline(old_path), _headline(new_path)
        for key in sorted(set(old_m) | set(new_m)):
            o, n = old_m.get(key), new_m.get(key)
            if o is None or n is None:
                status = "added" if o is None else "removed"
            elif abs(n - o) < 1e-9:
                status = "same"
            else:
                status = "CHANGED"
            rows.append({"file": str(rel), "metric": key,
                         "old": o, "new": n, "status": status})
    return rows


def print_comparison(rows: list[dict]) -> None:
    changed = [r for r in rows if r["status"] in ("CHANGED", "NEW FILE", "added", "removed")]
    print(f"\n{BAR}\n COMPARISON — previous run → this run\n{BAR}")
    if not rows:
        return
    if not changed:
        print("  no headline metric moved.")
        print("  Worth a second look: four patches were supposed to move these.")
        return

    current_file = None
    for r in changed:
        if r["file"] != current_file:
            current_file = r["file"]
            print(f"\n  {current_file}")
        o, n = r["old"], r["new"]
        if o is None or n is None:
            print(f"    {r['metric']:<42} {r['status']}")
        else:
            delta = n - o
            arrow = "▲" if delta > 0 else "▼"
            print(f"    {r['metric']:<42} {o:>10.4f} → {n:>10.4f}  {arrow} {delta:+.4f}")

    same = len(rows) - len(changed)
    print(f"\n  {len(changed)} metric(s) moved, {same} unchanged.")



# Running

def run_stage(stage: Stage, env: dict, dry: bool) -> dict:
    print(f"\n{SUB}\n  {stage.key.upper():<10} {stage.label}")
    if stage.notes:
        print(f"             {stage.notes}")
    print(f"             est. {stage.est_tokens:,} tokens · {stage.rq}")
    print(SUB)

    if dry:
        print("             (--plan: not executed)")
        return {"key": stage.key, "status": "planned", "duration_s": 0}

    start = time.perf_counter()
    proc = subprocess.run(stage.command, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - start
    status = "ok" if proc.returncode == 0 else "FAILED"
    print(f"\n             {status} in {elapsed:.0f}s")

    produced = []
    for name in stage.produces:
        hits = list(RESULTS.rglob(name))
        produced.extend(str(p.relative_to(RESULTS)) for p in hits)

    return {
        "key": stage.key, "label": stage.label, "rq": stage.rq,
        "status": status, "returncode": proc.returncode,
        "duration_s": round(elapsed, 1), "produced": produced,
        "est_tokens": stage.est_tokens,
    }


def cache_summary() -> dict:
    """Read llm_cache stats out of the freshest results file's _meta block."""
    newest, newest_mtime = None, -1.0
    for p in RESULTS.rglob("*.json"):
        if p.stat().st_mtime > newest_mtime:
            newest, newest_mtime = p, p.stat().st_mtime
    if newest is None:
        return {}
    try:
        return json.loads(newest.read_text()).get("_meta", {}).get("llm_cache", {})
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="Show the stage graph and cost, run nothing")
    ap.add_argument("--preflight", action="store_true", help="Readiness checks only")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="Run only these stages")
    ap.add_argument("--skip", nargs="+", metavar="KEY", default=[], help="Skip these stages")
    ap.add_argument("--replay", action="store_true",
                    help="LLM_CACHE=replay — reproduce from cache, zero API cost")
    ap.add_argument("--compare-only", action="store_true",
                    help="Diff the latest archive against current results, run nothing")
    ap.add_argument("--no-archive", action="store_true",
                    help="Do not archive first (you will not be able to compare)")
    ap.add_argument("--force", action="store_true", help="Run even if preflight fails")
    args = ap.parse_args()

    print(f"\n{BAR}\n EVIDENCE REGENERATION\n{BAR}")

    if args.compare_only:
        print_comparison(compare(latest_archive()))
        return 0

    selected = [s for s in STAGES
                if (not args.only or s.key in args.only) and s.key not in args.skip]
    if not selected:
        print("  nothing selected.")
        return 1

    # Dependency check. A stage whose prerequisite is not in this run is not
    # necessarily wrong — the prerequisite may have run yesterday — but it is
    # worth saying out loud, because the failure mode is a valid-looking number
    # computed from a stale artefact.
    chosen = {s.key for s in selected}
    for s in selected:
        missing = [d for d in s.depends_on if d not in chosen]
        if missing:
            print(f"  note  {s.key} depends on {missing}, not included in this run — "
                  f"it will use whatever is already on disk.")

    print(f"\n  Stages : {', '.join(s.key for s in selected)}")
    total = sum(s.est_tokens for s in selected)
    print(f"  Est.   : {total:,} tokens"
          + ("  (LLM_CACHE=replay → 0 actual)" if args.replay else
             f"  of a 100,000/day free-tier quota  ({total/1000:.0f}%)"))
    if total > 90_000 and not args.replay:
        print("  WARN   this is close to the daily ceiling. Consider splitting "
              "with --only, e.g. run rq4 on its own.")

    print(f"\n{SUB}\n  PREFLIGHT\n{SUB}")
    needs_llm = any(s.est_tokens > 0 for s in selected) and not args.replay
    problems = preflight(require_llm=needs_llm)
    if problems:
        print("\n  BLOCKED:")
        for p in problems:
            print(f"    ✗ {p}")
        # --plan is diagnostic: it must still show you the stage graph and the
        # cost, precisely so you can look at it BEFORE fixing the environment.
        if args.plan:
            print("\n  (--plan: showing the graph anyway)")
            for s in selected:
                run_stage(s, {}, dry=True)
            return 1
        if not args.force:
            print("\n  Fix these, or re-run with --force to proceed anyway "
                  "(the results will not be trustworthy).")
            return 1
        print("\n  --force given: continuing despite the above.")
    else:
        print("\n  preflight clean.")

    if args.preflight:
        return 0
    if args.plan:
        for s in selected:
            run_stage(s, {}, dry=True)
        return 0

    print(f"\n{SUB}\n  ARCHIVE\n{SUB}")
    archived = None if args.no_archive else archive_existing()

    env = dict(os.environ)
    env["LLM_CACHE"] = "replay" if args.replay else "record"
    env["PYTHONPATH"] = str(ROOT)
    print(f"\n  LLM_CACHE={env['LLM_CACHE']}")

    records, run_start = [], time.perf_counter()
    for stage in selected:
        rec = run_stage(stage, env, dry=False)
        records.append(rec)
        if rec["status"] == "FAILED":
            print(f"\n  {stage.key} failed. Stopping — later stages would be "
                  f"generated from a half-updated state.\n"
                  f"  Fix, then resume with: --only "
                  f"{' '.join(s.key for s in selected[selected.index(stage):])}")
            break

    wall = time.perf_counter() - run_start
    rows = compare(archived)
    print_comparison(rows)

    cache = cache_summary()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_state()[0],
        "git_dirty": _git_state()[1],
        "llm_cache_mode": env["LLM_CACHE"],
        "llm_cache_stats": cache,
        "wall_clock_s": round(wall, 1),
        "archived_to": str(archived.relative_to(ROOT)) if archived else None,
        "stages": records,
        "metric_changes": [r for r in rows if r["status"] != "same"],
    }
    manifest_path = RESULTS / "EVIDENCE_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{BAR}")
    print(f" DONE  {ok}/{len(records)} stages ok · {wall/60:.1f} min")
    if cache:
        print(f"       cache: {cache.get('hits',0)} hits, {cache.get('misses',0)} misses"
              + ("  ← CLEAN REPLAY" if cache.get("clean_replay") else ""))
    print("       manifest → results/EVIDENCE_MANIFEST.json")
    print(BAR)
    print("\n Next: commit results/ AND data/llm_cache/ together. The cache is\n"
          " what lets anyone (including an examiner) reproduce these exact\n"
          " numbers offline:  python scripts/regenerate_evidence.py --replay\n")

    return 0 if all(r["status"] == "ok" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())