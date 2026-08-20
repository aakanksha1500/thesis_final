"""
scripts/eval_banking77_stratified.py — Banking77 bucket classification at n=500.

    EVAL_LIVE_API=1 python scripts/eval_banking77_stratified.py
    python scripts/eval_banking77_stratified.py --n 200 --seed 20260812
    python scripts/eval_banking77_stratified.py --dry-run   # sampling only, no LLM

WHY THIS IS A SCRIPT AND NOT A TEST
    The evaluation it replaces lived in
    tests/unit/test_banking77_evaluation.py. In the snapshot this work
    started from, that file's contents are byte-identical to
    tests/unit/test_approval_gate.py — the Banking77 harness had been
    overwritten by a copy of the approval-gate tests at some point on
    2026-08-11, and only its outputs (results/mock/phase2_banking77_
    bucket_evaluation.json, data/processed/banking77_eval_checkpoint.json)
    survived. This is a rebuild, and it lives in scripts/ where the other
    evaluation entry points live, with the test file reduced to guarding
    the harness rather than containing it.

WHAT IT MEASURES
    n=500 utterances, stratified across all 77 fine-grained intents,
    classified with ONE LLM call each via
    ConversationalAgent.classify_only() — not run(), which would cost
    three calls per item for output this evaluation does not read.

    Reported: overall bucket accuracy with a 1,000-resample bootstrap CI;
    per-bucket precision/recall/F1 with support; the full 8x8 confusion
    matrix; and false-positive leakage into the five buckets Banking77
    cannot label, which is the safety-relevant number (a customer-service
    utterance routed to InvestmentAgent).

    See evaluation/banking77_data.py for why the train split is a
    legitimate evaluation pool for a zero-shot classifier, and why five
    of the eight buckets have no ground truth here by construction.

CHECKPOINTING
    500 live calls is long enough that a rate-limit or a dropped
    connection halfway through is a real risk. Predictions are written to
    data/processed/banking77_stratified_checkpoint.json as they complete,
    keyed by a fingerprint of (split, n, seed, prompt version, LLM mode,
    model), and a re-run with the same fingerprint resumes rather than
    restarting. LLM mode and model are part of the key so that a mock
    run's checkpoint can never be resumed by a live run — see
    _fingerprint().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR  # noqa: E402
from evaluation.banking77_data import (  # noqa: E402
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    assert_no_prompt_leakage,
    load_banking77,
    stratified_sample,
)
from evaluation.bootstrap import bootstrap_paired_metric  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    accuracy_value,
    classification_report,
    intent_accuracy,
)
from evaluation.results_io import write_results  # noqa: E402

CHECKPOINT_PATH = ROOT_DIR / "data" / "processed" / "banking77_stratified_checkpoint.json"

# The five buckets Banking77 has no ground truth for, by construction.
# A prediction landing in one of these on a customer-service utterance is
# always a false positive, and is the safety-relevant failure mode.
ADVISORY_BUCKETS = (
    "risk_profiling", "investment_advice", "budget_analysis",
    "full_advisory", "out_of_scope",
)

def _class_imbalance_warning(report, all_buckets: list[str]) -> dict | None:
    """
    Flags when raw accuracy is dominated by one bucket's population share
    rather than genuine per-bucket discrimination.

    WHY THIS EXISTS
        A run of this script produced bucket accuracy 0.84 against macro
        F1 0.16 — general_query alone was 168/200 items, and 4 of the 8
        buckets had zero support. Both numbers were already computed and
        printed side by side (see the loop below); nothing forced a
        reader to notice the gap between them, or to read macro F1 at
        all if they only wanted "the accuracy number" for a table. This
        turns that gap into a structured, unmissable field instead of
        something a reader has to independently compute and interpret.

    Returns None when the sample isn't skewed enough for this to be a
    real concern — this should fire on a genuinely degenerate run, not
    on ordinary variation in bucket frequency.
    """
    per_class = report.details["per_class"]
    supports = {b: per_class[b]["support"] for b in all_buckets if b in per_class}
    total_support = sum(supports.values())
    if total_support == 0:
        return None

    majority_bucket = max(supports, key=supports.get)
    majority_share = supports[majority_bucket] / total_support
    macro_f1 = report.details["macro_avg"]["f1"]
    accuracy = report.value
    gap = accuracy - macro_f1
    zero_support_buckets = [b for b in all_buckets if supports.get(b, 0) == 0]

    if majority_share < 0.5 and gap < 0.25 and not zero_support_buckets:
        return None

    zero_support_note = (
        f", and {len(zero_support_buckets)} bucket(s) have zero support "
        f"({', '.join(zero_support_buckets)})"
        if zero_support_buckets else ""
    )
    return {
        "majority_bucket": majority_bucket,
        "majority_bucket_share_of_support": round(majority_share, 4),
        "buckets_with_zero_support": zero_support_buckets,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy_minus_macro_f1_gap": round(gap, 4),
        "interpretation": (
            f"{majority_share:.0%} of this sample's ground truth is a "
            f"single bucket ({majority_bucket}){zero_support_note}. A "
            f"classifier that always predicted {majority_bucket!r} would "
            f"score close to this run's {accuracy:.3f} accuracy without "
            f"discriminating anything. macro_f1 ({macro_f1:.3f}) is not "
            f"inflated the same way — report it alongside, or instead of, "
            f"raw accuracy here. See buckets_without_ground_truth for "
            f"which buckets this dataset cannot evaluate at all."
        ),
    }


def _fingerprint(split: str, n: int, seed: int, llm_mode: str, model: str) -> str:
    """
    Identity of a resumable run.

    llm_mode and model are in here for a reason found the hard way: with
    a fingerprint of (split, n, seed, prompt_version) only, a mock-mode
    run leaves a checkpoint that a later LIVE run happily resumes —
    silently reporting mock predictions as real-model results. The two
    runs must never share a checkpoint.
    """
    import hashlib

    from config.prompts import PROMPT_VERSION
    raw = f"{split}|{n}|{seed}|{PROMPT_VERSION}|{llm_mode}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _is_rate_limit(exc: Exception) -> bool:
    """
    Is this a provider rate limit rather than a real failure?

    Matched on the message because the provider SDK is reached through
    LLMClient's own retry wrapper, which re-raises a plain Exception
    rather than the SDK's typed RateLimitError.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "rate_limit", "too many requests",
                       "quota", "tokens per day", "tpd")
    )


def _estimate_tokens_per_call() -> int:
    """
    Rough input-token cost of one classify_only() call.

    Built from the real prompt pieces rather than guessed, so it tracks
    any future edit to INTENT_BUCKET_GUIDE. ~4 chars per token is close
    enough for a budget warning; this is not billing arithmetic.
    """
    try:
        from agents.conversational_agent import INTENT_BUCKET_GUIDE
        from config.prompts import INTENT_CLASSIFIER_SYSTEM

        guide = "\n".join(f"  {b}: {d}" for b, d in INTENT_BUCKET_GUIDE.items())
        chars = len(INTENT_CLASSIFIER_SYSTEM) + len(guide) + 200
        return chars // 4 + 50          # + utterance and the JSON reply
    except Exception:                    # noqa: BLE001
        return 800


def _load_checkpoint(fingerprint: str) -> dict[str, dict]:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        payload = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("fingerprint") != fingerprint:
        return {}
    return {p["item_id"]: p for p in payload.get("predictions", [])}


def _save_checkpoint(fingerprint: str, predictions: dict[str, dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({
        "fingerprint": fingerprint,
        "n_predictions": len(predictions),
        "predictions": list(predictions.values()),
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report the sample, make no LLM calls")
    ap.add_argument("--throttle", type=float, default=2.0,
                    help="seconds to wait between calls. 2.0 keeps a "
                         "single-threaded run under a 30 RPM limit.")
    args = ap.parse_args()

    items, provenance = load_banking77()
    print(f"Banking77: {len(items)} mapped utterances "
          f"(split={provenance['split']})")

    sample, sampling = stratified_sample(items, n=args.n, seed=args.seed)
    print(f"  stratified sample: n={len(sample)} across "
          f"{sampling['n_strata']} intent strata "
          f"({sampling['min_per_stratum']}-{sampling['max_per_stratum']} per stratum)")
    for bucket, shares in sampling["bucket_balance_preserved"].items():
        print(f"    {bucket:<22} population {shares['population_share']:.3f} "
              f"-> sample {shares['sample_share']:.3f}")

    leakage = assert_no_prompt_leakage(sample)
    print(f"  prompt-leakage check: "
          f"{leakage.get('n_sampled_utterances_found_in_prompt', 'n/a')} "
          f"sampled utterances found in the classifier prompt")
    est_per_call = _estimate_tokens_per_call()
    print(f"  token budget: ~{est_per_call} tokens/call x {len(sample)} "
          f"= ~{est_per_call * len(sample):,} tokens total")
    print(f"    at 100K tokens/day that is ~{100_000 // est_per_call} calls/day; "
          f"at 200K, ~{200_000 // est_per_call}; at 500K, ~{500_000 // est_per_call}")
    print("    check your own limits at console.groq.com/settings/limits")

    if args.dry_run:
        print("\n--dry-run: sample built, no LLM calls made.")
        return 0

    from agents.conversational_agent import INTENT_BUCKETS, ConversationalAgent
    from utils.llm_client import LLMClient

    llm = LLMClient()
    agent = ConversationalAgent(llm)
    is_mock = llm.mode == "mock"
    if is_mock:
        print("\n  !! LLM is in MOCK mode — the intent classifier requires a "
              "real LLM. Results will carry is_placeholder=true.\n")

    fingerprint = _fingerprint(
        provenance["split"], args.n, args.seed, llm.mode, llm.model)
    done = _load_checkpoint(fingerprint)
    if done:
        print(f"  resuming from checkpoint: {len(done)}/{len(sample)} already done")

    started = time.time()
    rate_limited = False
    try:
        for i, item in enumerate(sample, start=1):
            if item.item_id in done:
                continue
            try:
                bucket, confidence = agent.classify_only(item.text)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit(exc):
                    # A tokens-per-DAY ceiling does not clear by waiting.
                    # LLMClient backs off up to 120s per attempt, so
                    # letting this retry burns hours and still fails.
                    # Stop cleanly and let the checkpoint carry the run
                    # into tomorrow instead.
                    rate_limited = True
                    print(
                        f"\n  RATE LIMITED at item {i}/{len(sample)}.\n"
                        f"  {len(done)} predictions are safely checkpointed.\n"
                        f"  If this is a per-day limit, re-run the SAME "
                        f"command tomorrow — it resumes automatically.\n"
                        f"  Do not change --n, --seed or the model: all "
                        f"three are in the checkpoint fingerprint, and "
                        f"changing any of them restarts the run.\n"
                        f"  Underlying error: {str(exc)[:160]}\n"
                    )
                    break
                print(f"    ! {item.item_id} failed: {exc}")
                raise
            done[item.item_id] = {
                "item_id": item.item_id,
                "text": item.text,
                "banking77_label": item.banking77_label,
                "true_bucket": item.true_bucket,
                "predicted_bucket": bucket,
                "confidence": confidence,
            }
            if i % 25 == 0:
                _save_checkpoint(fingerprint, done)
                print(f"    {i}/{len(sample)} classified "
                      f"({time.time() - started:.0f}s elapsed)")
            if args.throttle > 0:
                time.sleep(args.throttle)
    finally:
        # Runs on success, on exception AND on Ctrl+C. KeyboardInterrupt
        # derives from BaseException, so the `except Exception` above
        # never saw it — before this was a finally block, interrupting a
        # run silently discarded every call completed since the last
        # multiple of 25.
        _save_checkpoint(fingerprint, done)

    ordered = [done[item.item_id] for item in sample if item.item_id in done]
    predictions = [p["predicted_bucket"] for p in ordered]
    ground_truth = [p["true_bucket"] for p in ordered]
    all_buckets = list(INTENT_BUCKETS.keys())

    report = classification_report(predictions, ground_truth, labels=all_buckets)
    legacy_accuracy = intent_accuracy(predictions, ground_truth)
    ci = bootstrap_paired_metric(predictions, ground_truth, accuracy_value)
    imbalance_warning = _class_imbalance_warning(report, all_buckets)

    labelled_buckets = sorted({p["true_bucket"] for p in ordered})
    leakage_counts = {
        bucket: sum(1 for p in ordered if p["predicted_bucket"] == bucket)
        for bucket in ADVISORY_BUCKETS
    }
    n_leaked = sum(leakage_counts.values())

    payload = {
        "phase": 2,
        "research_question": "RQ4 (supporting) — intent routing accuracy",
        "component": "ConversationalAgent.classify_only",
        "is_placeholder": is_mock,
        "placeholder_reason": (
            "Intent classification requires a real LLM; the mock client "
            "returns a fixed bucket. Re-run with GROQ_API_KEY set."
        ) if is_mock else None,
        "dataset": {
            **provenance,
            "sampling": sampling,
            "prompt_leakage_check": leakage,
            "labels_modified": False,
            "labels_note": (
                "Bucket labels come from data/banking77_bucket_map.json "
                "exactly as committed. No label was changed for this "
                "evaluation, and no item was excluded on the basis of "
                "whether the model got it right."
            ),
        },
        "evaluation_configuration": {
            "classifier": "ConversationalAgent.classify_only (1 LLM call/item)",
            "llm_mode": llm.mode,
            "n_classified": len(ordered),
            "checkpoint_fingerprint": fingerprint,
            "throttle_seconds": args.throttle,
            "run_complete": len(ordered) == len(sample),
            "partial_run_reason": (
                f"stopped early by a provider rate limit after "
                f"{len(ordered)}/{len(sample)} items"
            ) if rate_limited or len(ordered) < len(sample) else None,
        },
        "metrics": {
            "bucket_accuracy": {
                **report.to_dict(),
                "class_imbalance_warning": imbalance_warning,
            },
            "intent_accuracy_legacy_metric": legacy_accuracy.to_dict(),
        },
        "confidence_intervals": {
            "bucket_accuracy": ci.to_dict(),
        },
        "buckets_with_ground_truth": labelled_buckets,
        "buckets_without_ground_truth": [
            b for b in all_buckets if b not in labelled_buckets
        ],
        "class_imbalance_warning": imbalance_warning,
        "false_positive_leakage_into_advisory_buckets": {
            "counts": leakage_counts,
            "n_total": n_leaked,
            "rate": round(n_leaked / len(ordered), 4) if ordered else 0.0,
            "why_this_is_the_relevant_number": (
                "Banking77 provides no positive examples for the five "
                "advisory buckets, so their precision/recall are "
                "undefined. What IS measurable is how often a "
                "customer-service utterance is misrouted into one of "
                "them — each such case sends a banking query to a "
                "specialist advisory agent in production."
            ),
        },
        "per_item": ordered,
        "wall_clock_s": round(time.time() - started, 1),
        "supersedes": (
            "results/mock/phase2_banking77_bucket_evaluation.json (n=200, "
            "harness lost when tests/unit/test_banking77_evaluation.py was "
            "overwritten) and the 20-item BANKING77_FIXTURE in "
            "tests/unit/test_conversational_agent.py"
        ),
    }

    path = write_results(payload, "phase2_banking77_stratified.json")

    print(f"\n  bucket accuracy: {report.value:.3f} "
          f"[{ci.ci_low:.3f}, {ci.ci_high:.3f}]  (n={len(ordered)})")
    print(f"  macro F1 over 8 buckets: {report.details['macro_avg']['f1']:.3f}")
    if imbalance_warning is not None:
        print(
            f"\n  !! CLASS IMBALANCE WARNING — {imbalance_warning['majority_bucket']!r} "
            f"is {imbalance_warning['majority_bucket_share_of_support']:.0%} of ground "
            f"truth. Report macro_f1 ({imbalance_warning['macro_f1']:.3f}) alongside "
            f"accuracy ({imbalance_warning['accuracy']:.3f}), not accuracy alone.\n"
        )
    for bucket in labelled_buckets:
        stats = report.details["per_class"][bucket]
        print(f"    {bucket:<22} P={stats['precision']:.3f} "
              f"R={stats['recall']:.3f} F1={stats['f1']:.3f} "
              f"n={stats['support']}")
    print(f"  leakage into advisory buckets: {n_leaked}/{len(ordered)} "
          f"({leakage_counts})")
    print(f"  wrote {path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
