"""
scripts/eval_rq5_finqa_verified.py — RQ5 on the real FinQA Verified split.

    python scripts/eval_rq5_finqa_verified.py                 # all conditions
    python scripts/eval_rq5_finqa_verified.py --conditions no_rag rag_oracle
    python scripts/eval_rq5_finqa_verified.py --limit 10       # smoke test

WHAT CHANGED, AND WHY IT HAD TO
    The previous RQ5 evaluation (tests/unit/test_rq5_finqa_evaluation.py)
    reported exact_match 1.00 with RAG against 0.40 without, over 10
    items. Reading that file's own code: `_with_rag_predict` is a
    hand-written arithmetic function that computes the correct answer
    from the context, and `_no_rag_predict` is a hand-written function
    documented as "always subtracts, regardless of what the question
    actually asks for". There is even a unit test
    (test_with_rag_predict_matches_every_fixture_item) asserting the
    grounded predictor gets 10/10.

    So the 1.00-vs-0.40 gap was a property of two fixture functions, not
    a measurement of the system, and the fixture questions were synthetic
    FinQA-*style* items rather than FinQA. Neither condition ever called
    an LLM. That result cannot be defended in a viva, so this script
    replaces it: real split, real model, same 91 questions in every
    condition.

    The legacy file is left in place and still runs — its outputs are
    retagged as superseded rather than deleted, so the earlier numbers
    remain auditable.

THE THREE CONDITIONS (identical questions, in identical order)
    no_rag        Question only. No context of any kind.
    rag_retrieved Context retrieved by the project's own embedder and
                  vector store from a corpus built out of all 91 source
                  documents. Retrieval can and does fail — that is the
                  point; this is the end-to-end system property.
    rag_oracle    The item's own gold context, handed over directly.
                  Retrieval cannot fail. This is the ceiling, and the gap
                  between it and rag_retrieved is the retrieval cost.

    Reporting all three separates two claims that the old binary design
    conflated: "grounding helps" (oracle vs no_rag) and "our retrieval
    finds the right grounding" (retrieved vs oracle).

LEAKAGE CONTROL
    The retrieval corpus is built from the `context` field ONLY and is
    held in a fresh in-memory VectorStore. It never touches
    data/embeddings/rag_index, and it never indexes the `answer` field —
    which matters because rag/knowledge_base.py::_load_finqa_verified
    indexes verified items as "Question ... Verified answer: ...", i.e.
    with the label in the retrievable text. Retrieving from that corpus
    would score answer lookup, not reasoning.

MOCK MODE
    Without a live API key the answering model is LLMClient's mock, which
    does not do arithmetic. The script still runs end to end (that is how
    the harness is tested), but every results file it writes in that mode
    carries "is_placeholder": true and a headline warning. Do not report
    a mock-mode RQ5 number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.bootstrap import (  # noqa: E402
    bootstrap_difference,
    bootstrap_proportion,
    detectable_effect_note,
)
from evaluation.finqa_data import (  # noqa: E402
    answer_leakage_report,
    assert_no_train_contamination,
    load_finqa_verified,
)
from evaluation.metrics import (  # noqa: E402
    _normalise_numeric_answer,
    finqa_exact_match,
    hallucination_rate,
)
from evaluation.results_io import write_results  # noqa: E402
from scripts.eval_banking77_stratified import _is_rate_limit  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402

CONDITIONS = ("no_rag", "rag_retrieved", "rag_oracle")

CHECKPOINT_PATH = ROOT_DIR / "data" / "processed" / "rq5_finqa_verified_checkpoint.json"

FINQA_SYSTEM_PROMPT = (
    "You answer numerical questions about company financial filings. "
    "You are terse and you never guess at figures you have not been given."
)


def _fingerprint(condition: str, llm_mode: str, model: str) -> str:
    """
    Identity of a resumable run, for ONE condition.

    Same lesson eval_banking77_stratified.py's _fingerprint learned "the
    hard way" applies here, doubly over: llm_mode and model are included
    so a mock-mode checkpoint can never be silently resumed by a live
    run (mock predictions reported as real-model results), and condition
    is included so a no_rag checkpoint can never be mistaken for
    rag_oracle progress — the three conditions process the exact same 91
    item_ids, and without condition in the hash a resumed rag_oracle run
    could silently reuse no_rag's answers.
    """
    raw = f"{condition}|{llm_mode}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_checkpoint(condition: str, fingerprint: str) -> dict[str, dict]:
    """
    One file, one top-level key per condition, so all three conditions'
    progress lives in one place (matching the three per-condition RESULT
    files they already produce) rather than three separate checkpoint
    files that could get out of sync with each other.
    """
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    condition_payload = payload.get(condition)
    if not condition_payload or condition_payload.get("fingerprint") != fingerprint:
        return {}
    return {p["item_id"]: p for p in condition_payload.get("predictions", [])}


def _save_checkpoint(condition: str, fingerprint: str, predictions: dict[str, dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Read-modify-write the whole file so saving THIS condition's
    # progress can never clobber a different condition's already-saved
    # progress sitting in the same file.
    try:
        payload = (
            json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            if CHECKPOINT_PATH.exists() else {}
        )
    except Exception:
        payload = {}
    payload[condition] = {
        "fingerprint": fingerprint,
        "n_predictions": len(predictions),
        "predictions": list(predictions.values()),
    }
    CHECKPOINT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TokenBudgetThrottle:
    """
    Proactive token-budget pacing, sized against Groq's OWN reported TPM
    limit rather than a guess.

    WHY THIS EXISTS
        A live run of this script hit repeated RateLimitErrors with
        growing wait times (78s, then 114s, then capping at 120s — see
        utils/llm_client.py's _retry_delay, which parses Groq's own
        "try again in Xs" text rather than backing off blindly). That
        pattern looked like a daily quota exhausted, but checking
        Groq's actual rate-limit headers (curl against
        api.groq.com/openai/v1/chat/completions) showed
        x-ratelimit-limit-tokens: 8000 with a sub-second reset when
        unused — an 8K TOKENS-PER-MINUTE budget, not a daily one, and
        barely any of the 1000-request daily allowance touched. A single
        rag_oracle/rag_retrieved call sends up to 6000 characters of
        filing context (1500+ tokens) — three or four of those fired
        back-to-back exhausts an 8K/minute budget completely, and every
        call after that pays for it in long reactive waits. This throttle
        prevents the exhaustion instead of recovering from it.

    WHY IT TRACKS REAL USAGE, NOT AN ESTIMATE
        LLMResponse.tokens_used (utils/llm_client.py) is the token count
        Groq's own response reports — record() uses that number, not a
        character-count guess, so the rolling budget this throttle
        enforces stays accurate even if prompt/response lengths vary a
        lot between items (they do — no_rag calls are a few hundred
        tokens, rag_oracle calls can be several thousand).

    WHY A SLEEP-BEFORE-CALL DESIGN, NOT A FIXED PER-CALL DELAY
        A fixed sleep (Banking77's --throttle, scripts/
        eval_banking77_stratified.py) sized for the worst case (a
        rag_oracle call) would needlessly slow down every no_rag call
        too, which are cheap enough to fire several per second under an
        8K/minute budget. This only sleeps when firing the NEXT call
        would actually risk crossing the budget within the trailing 60s
        window, so pacing adapts to what each condition actually costs.
    """

    def __init__(self, tpm_budget: int, safety_margin: float = 0.85):
        # safety_margin leaves headroom for Groq's own bookkeeping and
        # this estimate both being slightly off, rather than pacing
        # right up against the measured ceiling.
        self.budget = int(tpm_budget * safety_margin)
        self.window_s = 60.0
        self._events: list[tuple[float, int]] = []  # (timestamp, tokens_used)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._events = [(t, n) for t, n in self._events if t > cutoff]

    def wait_if_needed(self, expected_tokens: int) -> None:
        """Call BEFORE firing a request. Sleeps only if the rolling total
        for the trailing 60s would exceed budget once this call's
        (estimated) tokens are added."""
        now = time.time()
        self._prune(now)
        used = sum(n for _, n in self._events)
        if self._events and used + expected_tokens > self.budget:
            oldest_t = self._events[0][0]
            sleep_for = max(0.0, (oldest_t + self.window_s) - now) + 0.5
            print(f"    (pacing: sleeping {sleep_for:.1f}s to stay under "
                  f"the {self.budget}-token/min budget — {used} used of "
                  f"trailing-60s allowance)")
            time.sleep(sleep_for)
            self._prune(time.time())

    def record(self, tokens_used: int) -> None:
        """Call AFTER a request completes, with the real token count from
        the response — not the pre-call estimate."""
        self._events.append((time.time(), tokens_used))

ANSWER_PROMPT_NO_CONTEXT = """You are a financial analyst answering a question from a company filing.

Question: {question}

Give ONLY the final numeric answer, with a percent sign if the answer is a percentage. No working, no explanation, no units other than % — just the number."""

ANSWER_PROMPT_WITH_CONTEXT = """You are a financial analyst answering a question from a company filing.

Filing extract:
{context}

Question: {question}

Using ONLY the filing extract above, first state the figures you used in one short sentence, then give the final numeric answer on its own line prefixed with "ANSWER:". Use a percent sign if the answer is a percentage."""


def _split_answer(raw: str) -> tuple[str, str]:
    """
    Separate the model's stated working from its final answer.

    The working is what the hallucination detector scores (it is the
    claim-bearing text with an entailment path back to the context); the
    answer is what exact-match scores. A bare number has nothing for an
    NLI-style detector to verify, which is the same reasoning the legacy
    fixture documented for restating source figures.
    """
    text = (raw or "").strip()
    match = re.search(r"ANSWER\s*:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().split("\n")[0].strip(), text
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return (lines[-1] if lines else ""), text


def build_retrieval_corpus(items, chunk_chars: int = 900, overlap: int = 150):
    """
    A fresh in-memory index over the 91 source documents.

    Chunked because FinQA contexts are long filing extracts and a
    single-vector-per-document index retrieves almost nothing useful.
    Deliberately constructed here rather than via KnowledgeBase so that
    (a) nothing is persisted and (b) no other document set — least of all
    the answer-bearing finqa_verified set — can end up in it.
    """
    from rag.embedder import Embedder
    from rag.vector_store import Document, VectorStore

    embedder = Embedder()
    chunks: list[Document] = []
    texts: list[str] = []

    for item in items:
        context = item.context
        step = max(chunk_chars - overlap, 1)
        pieces = [context[i:i + chunk_chars] for i in range(0, len(context), step)] or [""]
        for j, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append(Document(
                doc_id=f"{item.item_id}-chunk{j:02d}",
                text=piece,
                source=f"FinQA Verified [D1] {item.item_id}",
                document_set="finqa_verified_contexts_only",
            ))
            texts.append(piece)

    vectors = embedder.encode(texts)
    store = VectorStore(dim=len(vectors[0]) if vectors else 384)
    store.add(chunks, vectors)
    return embedder, store


def retrieve(embedder, store, question: str, top_k: int = 3) -> list[dict[str, Any]]:
    hits = store.search(embedder.encode_one(question), top_k=top_k)
    return [
        {"text": doc.text, "source": doc.source, "relevance": round(float(score), 4),
         "doc_id": doc.doc_id}
        for doc, score in hits
    ]


def run_condition(
    condition: str,
    items,
    llm: LLMClient,
    detector,
    retrieval=None,
    throttle: TokenBudgetThrottle | None = None,
) -> dict[str, Any]:
    """
    One pass over every item. Same items, same order, in all conditions.

    RESUMABLE — see CHECKPOINT_PATH / _fingerprint / _load_checkpoint /
    _save_checkpoint above for the mechanism, which mirrors
    eval_banking77_stratified.py's proven pattern exactly (that script's
    own comment explains why the checkpoint save has to be in a finally
    block, not just after the loop: KeyboardInterrupt derives from
    BaseException, so a plain except Exception around the loop body
    would never see Ctrl+C, and interrupting a run would silently
    discard everything completed since the last periodic save).

    A rate limit is treated as a clean stop, not a crash: caught,
    checkpointed, and returned with rate_limited=True so main() knows
    not to burn the same wall trying the next condition immediately.
    """
    fingerprint = _fingerprint(condition, llm.mode, llm.model)
    done = _load_checkpoint(condition, fingerprint)
    if done:
        print(f"    {condition}: resuming from checkpoint — "
              f"{len(done)}/{len(items)} already done")

    started = time.time()
    rate_limited = False
    try:
        for n, item in enumerate(items, start=1):
            if item.item_id in done:
                continue

            contexts: list[dict[str, Any]] = []

            if condition == "no_rag":
                prompt = ANSWER_PROMPT_NO_CONTEXT.format(question=item.question)
                context_used = ""
            elif condition == "rag_oracle":
                context_used = item.context[:6000]
                prompt = ANSWER_PROMPT_WITH_CONTEXT.format(
                    context=context_used, question=item.question)
                contexts = [{"text": context_used,
                             "source": f"FinQA Verified [D1] {item.item_id} (gold context)"}]
            else:  # rag_retrieved
                embedder, store = retrieval
                hits = retrieve(embedder, store, item.question, top_k=3)
                own = any(h["doc_id"].startswith(item.item_id) for h in hits)
                context_used = "\n\n".join(h["text"] for h in hits)[:6000]
                prompt = ANSWER_PROMPT_WITH_CONTEXT.format(
                    context=context_used or "(no relevant extract retrieved)",
                    question=item.question)
                contexts = [{"text": h["text"], "source": h["source"]} for h in hits]

            if throttle is not None:
                estimated_tokens = len(prompt) // 4 + 150
                throttle.wait_if_needed(estimated_tokens)

            try:
                response = llm.chat(
                    system=FINQA_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw = response.content
                if throttle is not None:
                    throttle.record(response.tokens_used)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit(exc):
                    # Same reasoning as eval_banking77_stratified.py: a
                    # tokens-per-day ceiling does not clear by waiting,
                    # and LLMClient's own retry already burned up to
                    # 120s x 3 attempts finding that out. Stop cleanly
                    # rather than keep walking into the same wall.
                    rate_limited = True
                    print(
                        f"\n    {condition}: RATE LIMITED at item "
                        f"{n}/{len(items)}.\n"
                        f"    {len(done)} predictions are safely "
                        f"checkpointed for this condition.\n"
                        f"    Re-run the SAME command once your limit "
                        f"clears — it resumes automatically. Do not "
                        f"change the model or the --tpm-budget in a way "
                        f"that changes llm.model: both are in the "
                        f"checkpoint fingerprint.\n"
                        f"    Underlying error: {str(exc)[:160]}\n"
                    )
                    break
                raw = f"ERROR: {exc}"

            answer, working = _split_answer(raw if isinstance(raw, str) else str(raw))
            report = detector.score_response(working, contexts, threshold=0.5)

            pred_val = _normalise_numeric_answer(answer)
            gold_val = _normalise_numeric_answer(item.answer)
            correct = (
                pred_val is not None and gold_val is not None
                and abs(pred_val - gold_val) <= 0.01
            )

            done[item.item_id] = {
                "item_id": item.item_id,
                "question": item.question,
                "gold_answer": item.answer,
                "predicted_answer": answer,
                "correct": bool(correct),
                "call_failed": raw.startswith("ERROR:") if isinstance(raw, str) else False,
                "retrieved_own_document": (
                    None if condition != "rag_retrieved"
                    else any(c["source"].endswith(f"{item.item_id}") for c in contexts)
                ),
                "context_chars_used": len(context_used),
                # Cached alongside the per-item summary above (not just
                # in it) because hallucination_rate() below needs the
                # FULL report for every item, not the summary fields —
                # without this, a resumed run's hallucination_rate would
                # silently only reflect items processed AFTER the resume.
                "hallucination_report": report.to_dict(),
            }

            if n % 10 == 0:
                _save_checkpoint(condition, fingerprint, done)
                print(f"    {condition}: {n}/{len(items)} "
                      f"({sum(1 for p in done.values() if p['correct'])} "
                      f"correct so far, {time.time() - started:.0f}s elapsed)")
    finally:
        # Runs on success, on exception, on the clean rate-limit break,
        # AND on Ctrl+C (KeyboardInterrupt is a BaseException, which the
        # `except Exception` above never sees) — see this function's
        # docstring for why this has to be a finally, not just code after
        # the loop.
        _save_checkpoint(condition, fingerprint, done)

    ordered = [done[item.item_id] for item in items if item.item_id in done]
    predictions = [p["predicted_answer"] for p in ordered]
    ground_truth = [p["gold_answer"] for p in ordered]
    reports = [p["hallucination_report"] for p in ordered]
    per_item = [
        {k: v for k, v in p.items() if k != "hallucination_report"}
        for p in ordered
    ]
    retrieval_hits = sum(
        1 for p in ordered if p.get("retrieved_own_document")
    )

    em = finqa_exact_match(predictions, ground_truth)
    hr = hallucination_rate(reports)
    successes = [p["correct"] for p in per_item]
    n_call_failures = sum(p["call_failed"] for p in per_item)

    result = {
        "condition": condition,
        "n_items": len(items),
        "n_items_completed": len(ordered),
        "rate_limited": rate_limited,
        "metrics": {
            "finqa_exact_match": em.to_dict(),
            "hallucination_rate": hr.to_dict(),
        },
        "confidence_intervals": {
            "finqa_exact_match": bootstrap_proportion(successes).to_dict(),
        },
        "call_failures": {
            "n": n_call_failures,
            "item_ids": [p["item_id"] for p in per_item if p["call_failed"]],
            "note": (
                "Items where the LLM call itself failed (all retries in "
                "utils/llm_client.py exhausted, typically a sustained "
                "rate-limit) rather than the model answering incorrectly "
                "— predicted_answer is an error string for these, scored "
                "as wrong by finqa_exact_match same as any other miss. "
                "Non-zero here means the reported exact_match is "
                "pessimistic relative to the model's actual capability on "
                "those items, not a fair failure. See TokenBudgetThrottle "
                "in this script for the pacing meant to keep this at 0."
                if n_call_failures else
                "No calls failed — TokenBudgetThrottle kept every request "
                "under Groq's rate limit for the full run."
            ),
        },
        "per_item": per_item,
        "wall_clock_s": round(time.time() - started, 1),
        "_successes": successes,
    }
    if len(ordered) < len(items):
        result["INCOMPLETE_RUN_NOTE"] = (
            f"Only {len(ordered)}/{len(items)} items completed for this "
            f"condition — metrics above are computed on the partial set, "
            f"not the full 91. {'Stopped due to a rate limit; ' if rate_limited else ''}"
            f"re-run the same command to resume and complete the "
            f"remaining {len(items) - len(ordered)} item(s) via the "
            f"checkpoint at {CHECKPOINT_PATH.name}."
        )
    if condition == "rag_retrieved":
        result["retrieval_diagnostics"] = {
            "n_items_where_own_document_retrieved": retrieval_hits,
            "own_document_retrieval_rate": (
                round(retrieval_hits / len(ordered), 4) if ordered else None
            ),
            "top_k": 3,
            "note": (
                "Fraction of questions for which at least one chunk of the "
                "question's OWN source document appeared in the top-3. This "
                "is the retrieval ceiling for this condition: an item whose "
                "own document was not retrieved cannot be answered from "
                "grounding, no matter how good the reader model is."
                + (" Computed over the completed subset only — see "
                   "INCOMPLETE_RUN_NOTE."
                   if len(ordered) < len(items) else "")
            ),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N items (smoke test only)")
    ap.add_argument(
        "--tpm-budget", type=int, default=8000,
        help=(
            "Tokens-per-minute budget to pace calls against. Default "
            "8000 is what api.groq.com/openai/v1/chat/completions "
            "actually reported (x-ratelimit-limit-tokens header) for "
            "openai/gpt-oss-120b when this flag was added. Check your "
            "own current limit by inspecting that header on any call "
            "if your model or account tier differs — see this script's "
            "TokenBudgetThrottle docstring."
        ),
    )
    ap.add_argument(
        "--no-throttle", action="store_true",
        help="Disable pacing entirely and rely only on reactive retry "
             "(utils/llm_client.py) — not recommended above a handful of "
             "items; this is what produced the original rate-limit run.",
    )
    args = ap.parse_args()

    items, provenance = load_finqa_verified()
    if args.limit:
        items = items[:args.limit]
        provenance["n_items_used"] = len(items)
        provenance["sampling"] = (
            f"SMOKE TEST — first {len(items)} items only, NOT a reportable run"
        )

    print(f"FinQA Verified: {len(items)} items "
          f"(split={provenance['split']}, verified={provenance['is_verified_split']})")

    contamination = assert_no_train_contamination(items)
    leakage = answer_leakage_report(items)
    print(f"  train-contamination check: "
          f"{contamination.get('n_overlapping_question_strings', 'n/a')} "
          f"overlapping question strings, 0 train items evaluated")
    print(f"  gold answer appears verbatim in own context: "
          f"{leakage['n_answers_appearing_verbatim_in_own_context']}/{leakage['n_items']}")

    llm = LLMClient()
    is_mock = llm.mode == "mock"
    if is_mock:
        print("\n  !! LLM is in MOCK mode — results will be written with "
              "is_placeholder=true and must not be reported.\n")

    throttle = None if args.no_throttle else TokenBudgetThrottle(args.tpm_budget)
    if throttle is not None:
        print(f"  pacing calls to stay under ~{throttle.budget} tokens/min "
              f"({args.tpm_budget} measured limit x 0.85 safety margin)")
    else:
        print("  --no-throttle set: pacing disabled, relying on reactive "
              "retry only")

    from rag.hallucination_detector import HallucinationDetector
    detector = HallucinationDetector()

    retrieval = None
    if "rag_retrieved" in args.conditions:
        print("  building retrieval corpus from the 91 contexts "
              "(contexts only — answers are never indexed)...")
        retrieval = build_retrieval_corpus(items)
        print(f"    indexed {len(retrieval[1])} chunks, "
              f"embedder mode={retrieval[0].mode}")

    results: dict[str, dict] = {}
    for condition in args.conditions:
        print(f"\n  running condition: {condition}")
        results[condition] = run_condition(
            condition, items, llm, detector, retrieval, throttle,
        )
        em = results[condition]["metrics"]["finqa_exact_match"]["value"]
        ci = results[condition]["confidence_intervals"]["finqa_exact_match"]
        n_failed = results[condition]["call_failures"]["n"]
        n_completed = results[condition]["n_items_completed"]
        n_total = results[condition]["n_items"]
        print(f"    exact_match = {em:.3f} "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
              + (f"  (partial: {n_completed}/{n_total} items)"
                 if n_completed < n_total else "")
              + (f"  !! {n_failed} call(s) failed — see call_failures"
                 if n_failed else ""))
        if results[condition]["rate_limited"]:
            print(f"\n  Stopping here rather than immediately hitting the "
                  f"same rate limit on the next condition. Re-run this "
                  f"exact command once your limit clears — every "
                  f"condition's progress so far, including this one, is "
                  f"checkpointed and will resume automatically.")
            break

    # Paired differences over identical items.
    comparisons = {}
    for a, b in (("rag_oracle", "no_rag"),
                 ("rag_retrieved", "no_rag"),
                 ("rag_oracle", "rag_retrieved")):
        if a not in results or b not in results:
            continue
        if results[a]["n_items_completed"] != results[b]["n_items_completed"]:
            # One condition finished (or resumed further) than the other
            # — a real possibility now that a condition can stop partway
            # on a rate limit. bootstrap_difference(paired=True) would
            # raise ValueError on mismatched lengths rather than silently
            # misalign items, which is correct, but crashing the whole
            # comparisons block here isn't — skip this ONE comparison
            # with a clear reason instead, and re-run once both
            # conditions are complete for a real number.
            comparisons[f"{a}_minus_{b}"] = {
                "skipped": True,
                "reason": (
                    f"{a} has {results[a]['n_items_completed']} items "
                    f"completed, {b} has "
                    f"{results[b]['n_items_completed']} — a paired "
                    f"comparison needs matching completed sets. Re-run "
                    f"once both conditions are fully resumed."
                ),
            }
            continue
        diff = bootstrap_difference(
            results[a]["_successes"], results[b]["_successes"], paired=True,
        )
        comparisons[f"{a}_minus_{b}"] = {
            **diff.to_dict(),
            "significant_at_95pct": not (diff.ci_low <= 0.0 <= diff.ci_high),
            "pairing": "paired — identical items in both conditions",
        }

    dataset_block = {
        **provenance,
        "train_contamination_check": contamination,
        "answer_leakage_check": leakage,
        "conditions_share_identical_items": True,
        "condition_item_order": "file order, identical across conditions",
    }

    for condition, payload in results.items():
        successes = payload.pop("_successes")
        write_results(
            {
                "phase": 8,
                "research_question": "RQ5",
                "condition": condition,
                "is_placeholder": is_mock,
                "placeholder_reason": (
                    "LLM ran in mock mode; the mock model does not perform "
                    "numerical reasoning. Re-run with GROQ_API_KEY set."
                ) if is_mock else None,
                "dataset": dataset_block,
                "evaluation_configuration": {
                    # llm.model, NOT settings.llm.specialist_model: this
                    # harness builds a bare LLMClient(), which resolves to
                    # ORCHESTRATOR_MODEL. Naming the specialist model here
                    # would have misattributed every RQ5 result to a model
                    # that never answered a question.
                    "answering_model": llm.model,
                    "llm_mode": llm.mode,
                    "temperature": 0.0,
                    "grounding": {
                        "no_rag": "none",
                        "rag_retrieved": "top-3 chunks retrieved from a "
                                         "contexts-only corpus of all 91 items",
                        "rag_oracle": "the item's own gold context",
                    }[condition],
                    "exact_match_tolerance": 0.01,
                    "hallucination_threshold": 0.5,
                },
                "statistical_power": detectable_effect_note(len(items)),
                **payload,
                "condition_comparisons": comparisons,
                "supersedes": (
                    "rq5_finqa_no_rag.json / rq5_finqa_with_rag.json — "
                    "10-item synthetic fixture answered by hand-written "
                    "simulator functions, not by the model"
                ),
            },
            f"rq5_finqa_verified_{condition}.json",
        )
        payload["_successes"] = successes

    print("\n  condition comparisons (paired bootstrap, 1000 resamples):")
    for name, block in comparisons.items():
        print(f"    {name:<32} {block['point_estimate']:+.3f} "
              f"[{block['ci_low']:+.3f}, {block['ci_high']:+.3f}] "
              f"{'significant' if block['significant_at_95pct'] else 'not significant'}")

    print(f"\n  wrote results/{llm.mode}/rq5_finqa_verified_*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())