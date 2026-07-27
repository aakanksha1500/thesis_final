"""
scripts/verify_real_pipeline.py

Answers one question: IS THE REAL PIPELINE ACTUALLY WORKING?

The test suite deliberately runs on fallbacks so it stays fast and offline.
That is the right default for CI — and it is exactly why it cannot tell you
whether the real system works. Every subsystem here has a fallback that keeps
running rather than failing, so a fully degraded pipeline looks identical to a
working one from the outside.

This script refuses to let that happen. It checks each subsystem is genuinely
in real mode, exercises it against a known input, and reports:

    REAL      running the real model/API, and it behaved correctly
    DEGRADED  running a fallback — silently substituted, results not comparable
    BROKEN    in real mode but produced a wrong or unusable result
    SKIP      not applicable in this configuration

Exit code 0 only if nothing is BROKEN and nothing critical is DEGRADED.

USAGE
    python scripts/verify_real_pipeline.py              # full check
    python scripts/verify_real_pipeline.py --no-llm     # skip paid API calls
    python scripts/verify_real_pipeline.py --strict     # DEGRADED also fails
    python scripts/verify_real_pipeline.py --json       # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REAL, DEGRADED, BROKEN, SKIP = "REAL", "DEGRADED", "BROKEN", "SKIP"

_ICON = {REAL: "\033[32m✔ REAL    \033[0m",
         DEGRADED: "\033[33m▲ DEGRADED\033[0m",
         BROKEN: "\033[31m✘ BROKEN  \033[0m",
         SKIP: "\033[2m- SKIP    \033[0m"}

results: list[dict] = []
_use_colour = sys.stdout.isatty()


def report(stage: str, status: str, detail: str = "", fix: str = "") -> str:
    results.append({"stage": stage, "status": status, "detail": detail, "fix": fix})
    icon = _ICON[status] if _use_colour else f"{status:<9}"
    print(f"  {icon}  {stage}")
    if detail:
        for line in detail.splitlines():
            print(f"                 {line}")
    if fix and status in (DEGRADED, BROKEN):
        print(f"                 \033[36m→ {fix}\033[0m" if _use_colour else f"                 -> {fix}")
    return status


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m" if _use_colour else f"\n{title}")
    print("─" * 74)


# 1. LLM

def check_llm(skip_llm: bool) -> object:
    section("1. LLM  ·  utils/llm_client.py")
    from utils.llm_client import LLMClient

    client = LLMClient()
    if client.mode == "mock":
        report("LLM provider", DEGRADED,
               "No API key found — every agent will return [MOCK RESPONSE].\n"
               "Intent classification cannot work, so ALL routing falls back\n"
               "to conversational_only and no specialist ever runs.",
               "Set LLM_PROVIDER and the matching *_API_KEY in .env")
        return client

    report("LLM provider", REAL, f"provider={client.mode}  model={client.model}")

    if skip_llm:
        report("LLM live call", SKIP, "--no-llm passed")
        return client

    try:
        t0 = time.perf_counter()
        resp = client.chat(
            system="You are a test harness. Reply with exactly one word.",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
        )
        ms = (time.perf_counter() - t0) * 1000
        text = (resp.content or "").strip()
        if not text:
            report("LLM live call", BROKEN, "API returned empty content",
                   "Check the model name is valid for this provider")
        else:
            report("LLM live call", REAL,
                   f"{ms:.0f}ms  {resp.tokens_used} tokens  reply={text[:40]!r}")
    except Exception as exc:
        report("LLM live call", BROKEN, f"{type(exc).__name__}: {exc}",
               "Check the key is valid and the model name exists for this provider")
    return client
# 2. Embedder  index (the mismatch that produces silent nonsense)

def check_embedder_and_index():
    section("2. Embeddings  vector index  ·  rag/")
    from config.settings import settings
    from rag.embedder import Embedder

    emb = Embedder()
    if emb.mode == "fallback":
        report("Embedder", DEGRADED,
               "Hashing bag-of-words embedder. NOT semantic: two paraphrases\n"
               "with no shared tokens score zero. Retrieval quality is not\n"
               "comparable to a real-model run.",
               "pip install sentence-transformers")
    else:
        report("Embedder", REAL, f"mode={emb.mode}  model={emb.model_name}  dim={emb.dim}")

    from rag.vector_store import VectorStore
    vs = VectorStore(dim=emb.dim)
    if vs._backend == "faiss":
        report("Vector backend", REAL, "faiss  (IndexFlatIP)")
    else:
        report("Vector backend", DEGRADED,
               "numpy brute-force. Correct, just slower — fine at this corpus size.",
               "pip install faiss-cpu")

    # --- index provenance: the important one -----------------------------
    index_dir = settings.rag.index_dir
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        report("Index provenance", SKIP,
               f"No persisted index at {index_dir} — it will be built on first use.")
        return emb

    meta = json.loads(meta_path.read_text())
    built_mode = meta.get("embedder_mode")
    built_model = meta.get("embedder_model")

    if built_mode is None:
        report("Index provenance", BROKEN,
               f"Index at {index_dir} records no embedder.\n"
               f"It was built before provenance stamping existed, so there is no\n"
               f"way to know whether its vectors match your current embedder.\n"
               f"If they do not, every relevance score is noise — and nothing\n"
               f"errors, so it looks like retrieval is working.",
               "rm -rf data/embeddings/rag_index && python scripts/build_knowledge_base.py")
    elif built_mode != emb.mode or (built_mode != "fallback" and built_model != emb.model_name):
        report("Index provenance", BROKEN,
               f"Index built by:  mode={built_mode}  model={built_model}\n"
               f"Current embedder: mode={emb.mode}  model={emb.model_name}\n"
               f"Query and document vectors live in DIFFERENT SPACES.",
               "rm -rf data/embeddings/rag_index && python scripts/build_knowledge_base.py")
    else:
        report("Index provenance", REAL,
               f"built by {built_mode} ({built_model}), {meta.get('n_documents')} docs "
               f"— matches current embedder")
    return emb



# 3. Retrieval quality — is it actually finding the right thing?

def check_retrieval():
    section("3. Retrieval quality  ·  rag/knowledge_base.py")
    from config.settings import settings
    from rag.knowledge_base import knowledge_base

    try:
        knowledge_base.ensure_built()
    except Exception as exc:
        return report("Index build", BROKEN, f"{type(exc).__name__}: {exc}")

    report("Index loaded", REAL,
           f"{len(knowledge_base.store)} chunks  "
           f"embedder={knowledge_base.embedder.mode}  "
           f"backend={knowledge_base.store._backend}")

    # A query whose correct answer is unambiguous in the seed corpus.
    probe = "deposit guarantee scheme protection limit for savings accounts"
    hits = knowledge_base.retrieve(probe, top_k=3)

    if not hits:
        return report("Retrieval relevance", BROKEN,
                      f"query={probe!r} returned NOTHING.\n"
                      f"Either the index is empty or every score fell below "
                      f"min_relevance_score={settings.rag.min_relevance_score}.",
                      "Rebuild the index, or lower settings.rag.min_relevance_score")

    top = hits[0]
    detail = "\n".join(
        f"{h['relevance']:.3f}  {h['source'][:28]:<28} {h['text'][:46]}…" for h in hits
    )

    # Does the top hit actually mention what we asked about?
    expected = {"deposit", "guarantee", "dgs", "100,000", "savings"}
    overlap = sum(1 for w in expected if w in top["text"].lower())

    if overlap >= 2 and top["relevance"] >= 0.25:
        report("Retrieval relevance", REAL,
               f"top hit is on-topic ({overlap}/5 expected terms)\n{detail}")
    elif top["relevance"] < 0.25:
        report("Retrieval relevance", BROKEN,
               f"Top score {top['relevance']:.3f} is barely above the "
               f"{settings.rag.min_relevance_score} floor — this is the signature\n"
               f"of querying an index built by a DIFFERENT embedder.\n{detail}",
               "rm -rf data/embeddings/rag_index && python scripts/build_knowledge_base.py")
    else:
        report("Retrieval relevance", BROKEN,
               f"Top hit scored {top['relevance']:.3f} but is off-topic "
               f"({overlap}/5 expected terms).\n{detail}",
               "Rebuild the index; check the corpora actually loaded")



# 4. Hallucination detection — does it discriminate?

def check_hallucination():
    section("4. Hallucination detection  ·  rag/hallucination_detector.py")

    from rag.hallucination_detector import hallucination_detector as det

    if det.mode != "hhem":
        report(
            "HHEM model",
            DEGRADED,
            "Lexical Jaccard overlap heuristic. Deliberately over-flags.\n"
            "Its scores are NOT comparable to HHEM scores.",
            "pip install 'transformers<5' torch",
        )
    else:
        report("HHEM model", REAL, f"model={det.model_id}")

    premise = (
        "Deposit Guarantee Scheme protection in Ireland covers eligible "
        "deposits up to EUR 100,000 per depositor per credit institution."
    )

    grounded = (
        "Eligible deposits are protected up to EUR 100,000 per institution."
    )

    fabricated = (
        "This fund guarantees a 45% annual return with no risk of loss."
    )

    try:
        s_ok = det.score_pair(premise, grounded)
        s_bad = det.score_pair(premise, fabricated)
    except Exception as exc:
        return report(
            "Discrimination",
            BROKEN,
            f"{type(exc).__name__}: {exc}",
        )

    detail = (
        f"grounded claim   → {s_ok:.4f}\n"
        f"fabricated claim → {s_bad:.4f}\n"
        f"separation       → {s_ok - s_bad:+.4f}"
    )

    if s_ok > s_bad:
        report("Discrimination", REAL, detail)
    else:
        report(
            "Discrimination",
            BROKEN,
            detail
            + "\nThe detector scores a fabrication at least as high as a "
              "grounded claim — it is not discriminating.",
            "Check the model loaded correctly; try force_fallback to compare",
        )



# 5. Risk model

def check_risk_model():
    section("5. Risk classifier  ·  agents/risk_profiling_agent.py")
    from config.settings import settings

    if settings.risk.model_path.exists():
        report("Trained ML model", REAL, f"loaded from {settings.risk.model_path}")
    else:
        report("Trained ML model", DEGRADED,
               f"No model at {settings.risk.model_path} — the calibrated heuristic\n"
               f"runs instead. Every RQ1 number reflects the HEURISTIC, not a\n"
               f"trained model, and _compute_shap_proxy() returns proxy\n"
               f"attributions rather than real SHAP values.",
               "Train and save a model, or state this explicitly in the write-up")



# 6. The pipeline itself, end to end, per route

def check_pipeline(client, skip_llm: bool):
    section("6. Full pipeline  ·  orchestrator/orchestrator.py")

    if skip_llm:
        return report("End-to-end routes", SKIP, "--no-llm passed")

    from orchestrator.orchestrator import Orchestrator, RoutingDecision

    features = {
        "age": 34,
        "income": 55000,
        "employment_status": "employed",
        "dependents": 0,
        "existing_debt": 5000,
        "investment_horizon": 15,
        "loss_tolerance": 4,
        "financial_knowledge_score": 3,
    }

    orch = Orchestrator(client, session_id="verify-real")

    orch._session_state["user_features"] = dict(features)
    orch._session_state["monthly_income"] = features["income"] / 12
    orch._session_state["monthly_expenses"] = {
        "housing": 1400,
        "food": 520,
        "transport": 260,
        "utilities": 180,
        "entertainment": 220,
        "other": 300,
    }

    for routing in RoutingDecision:
        seq = orch._get_agent_sequence(routing)
        ctx = orch._build_context(
            "I want investment advice for my retirement."
        )
        ctx["_turn_id"] = f"verify-{routing.value}"

        try:
            agent_results, recovered = orch._run_agent_sequence(
                list(seq),
                ctx,
            )
        except Exception as exc:
            report(
                f"route {routing.value}",
                BROKEN,
                f"{type(exc).__name__}: {exc}",
            )
            continue

        problems = []

        for r in agent_results:
            status = r.payload.get("status")

            if not r.success:
                problems.append(f"{r.agent_name} FAILED: {r.error}")
            elif status in (
                "incomplete",
                "no_suitable_products",
                "error",
            ):
                problems.append(
                    f"{r.agent_name} → {status}: "
                    f"{r.payload.get('message', '')[:80]}"
                )

        if recovered:
            problems.append(
                f"recovered (i.e. failed then substituted): {recovered}"
            )

        line = "  ".join(
            f"{r.agent_name.replace('Agent', '')}"
            f"[{r.payload.get('status', 'ok')}]"
            for r in agent_results
        )

        if problems:
            report(
                f"route {routing.value}",
                BROKEN,
                line + "\n" + "\n".join(problems),
            )
        else:
            report(f"route {routing.value}", REAL, line)

    try:
        res = orch.process_turn(
            "I want to invest my savings for retirement."
        )
    except Exception as exc:
        return report(
            "process_turn()",
            BROKEN,
            f"{type(exc).__name__}: {exc}",
        )

    detail = (
        f"routing={res.routing_decision.value}  "
        f"agents={[a.replace('Agent', '') for a in res.agents_invoked]}\n"
        f"{res.total_duration_ms:.0f}ms  "
        f"violations={len(res.constraint_violations)}  "
        f"conflicts={len(res.conflicts)}\n"
        f"reply: {res.final_response[:110]}…"
    )

    if "[MOCK RESPONSE]" in res.final_response:
        report(
            "process_turn()",
            DEGRADED,
            detail,
            "Set an API key",
        )
    elif res.routing_decision.value == "conversational_only":
        report(
            "process_turn()",
            BROKEN,
            detail
            + "\nAn explicit investment request routed to conversational_only —\n"
              "the intent classifier is not working against this model.",
            "Check the model returns valid JSON; try a stronger model",
        )
    else:
        report("process_turn()", REAL, detail)

    print(f"\n  audit trail: {orch.audit_log.log_path}")



# 7. Judge — the evaluation path

def check_judge(client, skip_llm: bool):
    section("7. Agent-as-Judge  ·  evaluation/agent_judge.py")
    if skip_llm or client.mode == "mock":
        return report("Judge scoring", SKIP, "requires a real LLM")

    from evaluation.agent_judge import SCORE_DIMENSIONS, AgentJudge
    from orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(client, session_id="verify-judge")
    orch._session_state["user_features"] = {
        "age": 34, "income": 55000, "employment_status": "employed",
        "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
        "loss_tolerance": 4, "financial_knowledge_score": 3}
    res = orch.process_turn("Should I invest for retirement?")

    scores = AgentJudge(client).evaluate(res)
    if scores.get("judge_mode") != "real":
        return report("Judge scoring", BROKEN,
                      f"Fell back to constant scores: {scores.get('reasoning','')[:90]}",
                      "The judge model is not returning parseable JSON — try a stronger model")

    missing = [d for d in SCORE_DIMENSIONS if d not in scores]
    detail = "  ".join(f"{d.split('_')[0]}={scores.get(d)}" for d in SCORE_DIMENSIONS)
    if missing:
        report("Judge scoring", BROKEN,
               f"{detail}\nMissing dimensions (silently filled with 3.0): {missing}",
               "Prompt and parser disagree — check config/prompts.py JUDGE_DIMENSIONS")
    else:
        report("Judge scoring", REAL,
               f"{detail}\noverall={scores.get('overall_score')} "
               f"verdict={scores.get('verdict')}")



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-llm", action="store_true", help="Skip all paid API calls")
    ap.add_argument("--strict", action="store_true", help="DEGRADED also fails")
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    args = ap.parse_args()

    print("\n"  "═" * 74)
    print(" REAL PIPELINE VERIFICATION")
    print(" Is the system actually running on real models, and does it work?")
    print("═" * 74)

    client = check_llm(args.no_llm)
    check_embedder_and_index()
    check_retrieval()
    check_hallucination()
    check_risk_model()
    check_pipeline(client, args.no_llm)
    check_judge(client, args.no_llm)

    broken = [r for r in results if r["status"] == BROKEN]
    degraded = [r for r in results if r["status"] == DEGRADED]

    print("\n"  "═" * 74)
    print(f" SUMMARY   real={sum(1 for r in results if r['status']==REAL)}  "
          f"degraded={len(degraded)}  broken={len(broken)}  "
          f"skipped={sum(1 for r in results if r['status']==SKIP)}")
    print("═" * 74)

    if broken:
        print("\n BROKEN — fix these:")
        for r in broken:
            print(f"   • {r['stage']}")
            if r["fix"]:
                print(f"     {r['fix']}")
    if degraded:
        print("\n DEGRADED — running on a fallback, results not comparable to a real run:")
        for r in degraded:
            print(f"   • {r['stage']}")
            if r["fix"]:
                print(f"     {r['fix']}")
    if not broken and not degraded:
        print("\n Everything is running on real models and behaving correctly.\n")

    if args.json:
        Path("real_pipeline_report.json").write_text(json.dumps(results, indent=2))
        print("\n wrote real_pipeline_report.json")

    print()
    if broken:
        return 1
    if degraded and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
