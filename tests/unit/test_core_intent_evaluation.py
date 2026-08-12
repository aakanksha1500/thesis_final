"""
Core advisory intent evaluation — the corpus Banking77 structurally cannot
provide.

WHY THIS EXISTS
    data/banking77_bucket_map.json documents that 5 of this system's 8
    intent buckets (risk_profiling, investment_advice, budget_analysis,
    full_advisory, out_of_scope) get ZERO ground-truth labels from
    Banking77 — it's a retail-banking customer-service dataset, and these
    5 buckets describe financial-ADVISORY conversations Banking77 was
    never built to contain (see test_banking77_evaluation.py's own
    docstring, which already flags this gap without filling it).

    So the system's actual core purpose — the 5 buckets that route to a
    specialist agent rather than a customer-service fallback — had no
    comparably-sized, real-utterance evaluation corpus at all. A handful
    of hand-written examples lived in agents/conversational_agent.py's
    INTENT_BUCKET_GUIDE (there to prompt the LLM, not to evaluate it) and
    scripts/eval_planner_vs_static.py's 12 routing scenarios (there to
    test the Planner's sequencing, not classification accuracy in
    isolation). Neither is a dedicated evaluation set.

CORPUS
    ~22 hand-authored utterances per bucket (110 total) — deliberately
    varied register (formal/casual/anxious/terse), length, and phrasing
    strategy, not parameter-swapped templates. Hand-authored because,
    unlike the customer-service buckets, no public dataset of financial-
    advisory conversation openers exists to draw from — the same
    situation RQ1_FIXTURE (15 hand-labelled risk profiles) and
    scripts/eval_planner_vs_static.py's SCENARIOS are already in, elsewhere
    in this codebase.

TWO TEST CLASSES
    TestCoreIntentFixture — pure data-shape checks (no LLM), runs anywhere.
    TestCoreIntentBucketEvaluation — the real evaluation. Needs
        EVAL_LIVE_API=1 (see conftest._no_live_api_in_tests) and costs
        real tokens (~110 classify_only() calls, ~3-4k tokens). Not run in
        this session for the same reason test_banking77_evaluation.py's
        live class isn't — sandbox has no LLM API key. The corpus and
        fixture-shape checks below are fully verified without one.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from agents.conversational_agent import INTENT_BUCKETS, ConversationalAgent
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

ALL_BUCKETS = list(INTENT_BUCKETS.keys())


def _build_confusion_and_metrics(
    predictions: list[dict[str, Any]],
) -> tuple[dict, dict]:
    """
    Deliberately a local copy of test_banking77_evaluation.py's function of
    the same purpose, not a cross-test-file import of it — that import
    depends on pytest's import mode / rootdir / package __init__.py
    presence in a way that's proven environment-fragile (works under some
    Python/pytest combinations, ImportError under others, for the exact
    same committed code). This file stays fully self-contained instead.

    predictions: [{"true_bucket": str, "predicted_bucket": str}, ...]
    Returns (confusion_matrix, per_bucket_metrics).
    """
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        confusion[p["true_bucket"]][p["predicted_bucket"]] += 1

    per_bucket = {}
    for bucket in ALL_BUCKETS:
        tp = confusion[bucket][bucket]
        true_count = sum(confusion[bucket].values())
        pred_count = sum(confusion[b].get(bucket, 0) for b in confusion)
        precision = tp / pred_count if pred_count else None
        recall = tp / true_count if true_count else None
        per_bucket[bucket] = {
            "true_count_in_sample": true_count,
            "predicted_count": pred_count,
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
        }

    return {k: dict(v) for k, v in confusion.items()}, per_bucket

CORE_INTENT_FIXTURE: list[dict[str, str]] = [
    # ── risk_profiling ──────────────────────────────────────────────
    {"text": "What's my risk profile?", "true_bucket": "risk_profiling"},
    {"text": "Can you assess my risk tolerance?", "true_bucket": "risk_profiling"},
    {"text": "Am I a cautious investor or more of a risk-taker?", "true_bucket": "risk_profiling"},
    {"text": "How risky an investor am I?", "true_bucket": "risk_profiling"},
    {"text": "Would you say I'm conservative or aggressive with money?", "true_bucket": "risk_profiling"},
    {"text": "I want to know my investor profile before I put any money anywhere.", "true_bucket": "risk_profiling"},
    {"text": "Assess my risk tolerance please", "true_bucket": "risk_profiling"},
    {"text": "What kind of investor am I based on my situation?", "true_bucket": "risk_profiling"},
    {"text": "Can you tell me how much risk I can handle?", "true_bucket": "risk_profiling"},
    {"text": "Where do I sit on the risk spectrum?", "true_bucket": "risk_profiling"},
    {"text": "Rate my appetite for investment risk", "true_bucket": "risk_profiling"},
    {"text": "I've never really thought about my risk tolerance, can you work it out?", "true_bucket": "risk_profiling"},
    {"text": "Am I too cautious with my money?", "true_bucket": "risk_profiling"},
    {"text": "How would you classify me as an investor?", "true_bucket": "risk_profiling"},
    {"text": "Can you figure out if I'm risk averse?", "true_bucket": "risk_profiling"},
    {"text": "What's my investor type?", "true_bucket": "risk_profiling"},
    {"text": "I keep hearing about 'risk profiles' - what's mine?", "true_bucket": "risk_profiling"},
    {"text": "Given my age and income, how much risk should I be taking?", "true_bucket": "risk_profiling"},
    {"text": "Is my attitude to risk more conservative or moderate?", "true_bucket": "risk_profiling"},
    {"text": "Can you profile my appetite for investment losses?", "true_bucket": "risk_profiling"},
    {"text": "How much risk can I actually afford to take on?", "true_bucket": "risk_profiling"},
    {"text": "My friend got told she was 'moderate risk' - what would I be?", "true_bucket": "risk_profiling"},

    # ── investment_advice ───────────────────────────────────────────
    {"text": "Should I invest my savings?", "true_bucket": "investment_advice"},
    {"text": "What should I invest in?", "true_bucket": "investment_advice"},
    {"text": "How can I grow my money?", "true_bucket": "investment_advice"},
    {"text": "Is now a good time to invest?", "true_bucket": "investment_advice"},
    {"text": "I have some money saved up, what should I do with it?", "true_bucket": "investment_advice"},
    {"text": "What kind of return could I realistically get?", "true_bucket": "investment_advice"},
    {"text": "Can you recommend an investment for me?", "true_bucket": "investment_advice"},
    {"text": "I want to start investing but don't know where to begin", "true_bucket": "investment_advice"},
    {"text": "Is putting money in a fund a good idea for me?", "true_bucket": "investment_advice"},
    {"text": "Should I be putting my savings into something that grows instead of just sitting in the bank?", "true_bucket": "investment_advice"},
    {"text": "What are my options if I want to invest 10,000 euro?", "true_bucket": "investment_advice"},
    {"text": "I want my money to work harder for me", "true_bucket": "investment_advice"},
    {"text": "Would an ETF suit me?", "true_bucket": "investment_advice"},
    {"text": "I'm thinking about investing for retirement, what would you suggest?", "true_bucket": "investment_advice"},
    {"text": "Can you suggest something with a decent return?", "true_bucket": "investment_advice"},
    {"text": "What's a good investment for someone my age?", "true_bucket": "investment_advice"},
    {"text": "I've got a lump sum from an inheritance, what should I do with it?", "true_bucket": "investment_advice"},
    {"text": "Is it worth investing right now or should I wait?", "true_bucket": "investment_advice"},
    {"text": "What would you put my savings into?", "true_bucket": "investment_advice"},
    {"text": "I want to start building wealth, where do I start?", "true_bucket": "investment_advice"},
    {"text": "Can you help me pick an investment product?", "true_bucket": "investment_advice"},
    {"text": "Should I invest a lump sum or drip-feed it in monthly?", "true_bucket": "investment_advice"},

    # ── budget_analysis ─────────────────────────────────────────────
    {"text": "Where does my money go?", "true_bucket": "budget_analysis"},
    {"text": "What am I spending on?", "true_bucket": "budget_analysis"},
    {"text": "Show me my budget", "true_bucket": "budget_analysis"},
    {"text": "Am I overspending?", "true_bucket": "budget_analysis"},
    {"text": "How much do I have left each month?", "true_bucket": "budget_analysis"},
    {"text": "Can you break down my spending for me?", "true_bucket": "budget_analysis"},
    {"text": "I feel like I'm always broke by the end of the month, why?", "true_bucket": "budget_analysis"},
    {"text": "What's my disposable income?", "true_bucket": "budget_analysis"},
    {"text": "Help me understand my monthly spending", "true_bucket": "budget_analysis"},
    {"text": "Is my grocery spend normal for my income?", "true_bucket": "budget_analysis"},
    {"text": "Am I saving enough each month?", "true_bucket": "budget_analysis"},
    {"text": "I want a clearer picture of my finances - where's it all going?", "true_bucket": "budget_analysis"},
    {"text": "Can you check if I'm spending too much on any one thing?", "true_bucket": "budget_analysis"},
    {"text": "How does my spending compare to a typical household?", "true_bucket": "budget_analysis"},
    {"text": "I'm struggling to make ends meet, what am I doing wrong?", "true_bucket": "budget_analysis"},
    {"text": "I'm behind on some payments, what should I look at first?", "true_bucket": "budget_analysis"},
    {"text": "What happens if I miss a bill payment?", "true_bucket": "budget_analysis"},
    {"text": "Can you tell me if my rent is eating up too much of my income?", "true_bucket": "budget_analysis"},
    {"text": "I want to cut back but don't know where", "true_bucket": "budget_analysis"},
    {"text": "What's my savings rate looking like?", "true_bucket": "budget_analysis"},
    {"text": "Can you look at my transactions and tell me what's normal and what's not?", "true_bucket": "budget_analysis"},
    {"text": "I never seem to have anything left over, can you see why?", "true_bucket": "budget_analysis"},

    # ── full_advisory ───────────────────────────────────────────────
    {"text": "I don't know where to start", "true_bucket": "full_advisory"},
    {"text": "Can you give me a complete review of my finances?", "true_bucket": "full_advisory"},
    {"text": "I'm new to all this, where do I begin?", "true_bucket": "full_advisory"},
    {"text": "I want a full picture of my finances", "true_bucket": "full_advisory"},
    {"text": "Can you look at everything - my spending, my investments, all of it?", "true_bucket": "full_advisory"},
    {"text": "I've never managed my money properly, can you help me get on top of it?", "true_bucket": "full_advisory"},
    {"text": "Where should someone in my position even start?", "true_bucket": "full_advisory"},
    {"text": "I want a complete financial health check", "true_bucket": "full_advisory"},
    {"text": "Can you walk me through my whole financial situation?", "true_bucket": "full_advisory"},
    {"text": "I feel completely lost with money, can you help me sort it all out?", "true_bucket": "full_advisory"},
    {"text": "What's the first thing I should be doing with my finances?", "true_bucket": "full_advisory"},
    {"text": "I just want an overall sense of how I'm doing financially", "true_bucket": "full_advisory"},
    {"text": "Can you give me the full works - budget, investing, risk, all of it?", "true_bucket": "full_advisory"},
    {"text": "I've come into some money and I have no idea what to do with any of it", "true_bucket": "full_advisory"},
    {"text": "Give me a general assessment of my finances", "true_bucket": "full_advisory"},
    {"text": "I want to get serious about my money but don't know the first step", "true_bucket": "full_advisory"},
    {"text": "Can you look at my whole situation and tell me what I should be doing?", "true_bucket": "full_advisory"},
    {"text": "I need help getting my finances in order generally", "true_bucket": "full_advisory"},
    {"text": "What should someone starting from scratch focus on first?", "true_bucket": "full_advisory"},
    {"text": "I want a holistic plan for my money", "true_bucket": "full_advisory"},
    {"text": "Can we go through everything from the beginning?", "true_bucket": "full_advisory"},
    {"text": "I have no financial plan at all, can you help me build one?", "true_bucket": "full_advisory"},

    # ── out_of_scope ────────────────────────────────────────────────
    {"text": "Can you give me legal advice about a dispute with my landlord?", "true_bucket": "out_of_scope"},
    {"text": "What medication should I take for a headache?", "true_bucket": "out_of_scope"},
    {"text": "Can you help me write a will?", "true_bucket": "out_of_scope"},
    {"text": "I need legal advice about my mortgage contract.", "true_bucket": "out_of_scope"},
    {"text": "What's the weather like tomorrow?", "true_bucket": "out_of_scope"},
    {"text": "I think I broke my arm, what should I do?", "true_bucket": "out_of_scope"},
    {"text": "Can you recommend a good divorce lawyer?", "true_bucket": "out_of_scope"},
    {"text": "What's the capital of France?", "true_bucket": "out_of_scope"},
    {"text": "Can you diagnose this rash for me?", "true_bucket": "out_of_scope"},
    {"text": "I'm having chest pains, is that serious?", "true_bucket": "out_of_scope"},
    {"text": "Can you help me with my visa application?", "true_bucket": "out_of_scope"},
    {"text": "What's a good recipe for chicken curry?", "true_bucket": "out_of_scope"},
    {"text": "My neighbour is being a nuisance, what are my legal rights?", "true_bucket": "out_of_scope"},
    {"text": "Can you write my CV for me?", "true_bucket": "out_of_scope"},
    {"text": "I need advice on custody arrangements for my kids", "true_bucket": "out_of_scope"},
    {"text": "What symptoms does the flu have?", "true_bucket": "out_of_scope"},
    {"text": "Can you tell me if I have a case for a personal injury claim?", "true_bucket": "out_of_scope"},
    {"text": "I need immigration advice", "true_bucket": "out_of_scope"},
    {"text": "What's the best treatment for anxiety?", "true_bucket": "out_of_scope"},
    {"text": "Can you help plan my wedding?", "true_bucket": "out_of_scope"},
    {"text": "I'm being evicted, what are my tenant rights?", "true_bucket": "out_of_scope"},
    {"text": "Can you give me medical advice about my prescription?", "true_bucket": "out_of_scope"},
]

_CORE_BUCKETS = (
    "risk_profiling", "investment_advice", "budget_analysis",
    "full_advisory", "out_of_scope",
)


class TestCoreIntentFixture:
    """Pure data-shape checks — no LLM, no dataset file required. Runs anywhere."""

    def test_every_entry_targets_one_of_the_five_core_buckets(self):
        for item in CORE_INTENT_FIXTURE:
            assert item["true_bucket"] in _CORE_BUCKETS

    def test_every_core_bucket_has_at_least_twenty_examples(self):
        from collections import Counter
        counts = Counter(item["true_bucket"] for item in CORE_INTENT_FIXTURE)
        for bucket in _CORE_BUCKETS:
            assert counts[bucket] >= 20, (
                f"{bucket} has only {counts[bucket]} examples — a handful of "
                f"hand-written examples was the exact gap this corpus exists "
                f"to close"
            )

    def test_no_duplicate_utterances(self):
        texts = [item["text"] for item in CORE_INTENT_FIXTURE]
        assert len(texts) == len(set(texts))

    def test_total_size_is_comparable_to_the_banking77_checkpoint(self):
        """
        data/processed/banking77_eval_checkpoint.json covers the other 3
        buckets with 135 utterances. This corpus should be in the same
        order of magnitude for the 5 buckets Banking77 can't reach —
        that comparability is the entire point.
        """
        assert len(CORE_INTENT_FIXTURE) >= 100


@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests
class TestCoreIntentBucketEvaluation:
    """
    The real evaluation — needs EVAL_LIVE_API=1 and a working LLM key (real
    tokens, ~3-4k for 110 classify_only() calls). Not run in this session;
    fixture-shape checks above are covered without touching the API.
    """

    def test_classification_and_write_results(self):
        client = LLMClient()
        agent = ConversationalAgent(client)

        predictions = []
        for item in CORE_INTENT_FIXTURE:
            pred_bucket, confidence = agent.classify_only(item["text"])
            if agent.last_classification_error is not None:
                pytest.skip(
                    f"Stopped at {len(predictions)}/{len(CORE_INTENT_FIXTURE)} "
                    f"-- classification failure, not a real result: "
                    f"{agent.last_classification_error}"
                )
            predictions.append({
                "text": item["text"],
                "true_bucket": item["true_bucket"],
                "predicted_bucket": pred_bucket,
                "confidence": confidence,
            })

        confusion, per_bucket = _build_confusion_and_metrics(predictions)
        correct = sum(1 for p in predictions if p["predicted_bucket"] == p["true_bucket"])
        accuracy = correct / len(predictions)

        results_path = write_results(
            {
                "phase": 2,
                "research_question": (
                    "Core advisory intent classification — the 5 buckets "
                    "Banking77 has zero ground truth for"
                ),
                "dataset": (
                    "Hand-authored corpus, tests/unit/test_core_intent_"
                    "evaluation.py::CORE_INTENT_FIXTURE, n="
                    f"{len(CORE_INTENT_FIXTURE)}"
                ),
                "cost_control": "classify_only() -- 1 LLM call/sample",
                "overall_accuracy": round(accuracy, 4),
                "per_bucket_metrics": {
                    b: per_bucket[b] for b in _CORE_BUCKETS
                },
                "confusion_matrix": {
                    b: confusion.get(b, {}) for b in _CORE_BUCKETS
                },
                "predictions": predictions,
            },
            "phase2c_core_intent_baseline.json",
        )
        print(f"\n[CoreIntent] Results written to {results_path}")
        print(f"[CoreIntent] Overall accuracy: {accuracy:.1%} ({correct}/{len(predictions)})")

        assert len(predictions) == len(CORE_INTENT_FIXTURE)