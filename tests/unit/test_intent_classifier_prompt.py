"""
Regression test for the intent classifier prompt fix.

WHAT THIS GUARDS AGAINST
    "where does my money go?" — a textbook budget_analysis question — was
    misclassified because the live classifier prompt showed the LLM raw
    Banking77 label slugs ("budget_analysis: cashflow, budget, disposable_
    income") instead of natural phrasing, so there was nothing for a
    colloquial question to lexically match. See agents/conversational_
    agent.py's INTENT_BUCKET_GUIDE docstring for the full diagnosis.

    This does NOT re-test classification accuracy end-to-end — that
    needs a real LLM call, which this suite deliberately runs without
    (see conftest.py's _no_live_api_in_tests). What it guards is the
    PROMPT ITSELF: that INTENT_BUCKET_GUIDE stays in sync with
    INTENT_BUCKETS, and that _classify_intent() actually builds its
    prompt from the descriptive guide, not the raw label slugs it used
    to (and could easily be changed back to by an unrelated edit).

RUNNING
    python -m pytest tests/unit/test_intent_classifier_prompt.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

from agents.conversational_agent import (
    INTENT_BUCKET_GUIDE,
    INTENT_BUCKETS,
    ConversationalAgent,
)


class TestIntentBucketGuideConsistency:

    def test_guide_covers_exactly_the_same_buckets_as_intent_buckets(self):
        assert set(INTENT_BUCKET_GUIDE) == set(INTENT_BUCKETS)

    def test_every_description_is_substantive_not_a_placeholder(self):
        for bucket, description in INTENT_BUCKET_GUIDE.items():
            assert len(description) > 40, (
                f"{bucket}'s description looks too short to be useful "
                f"classifier signal: {description!r}"
            )

    def test_intent_buckets_itself_is_unchanged_by_this_fix(self):
        """
        INTENT_BUCKETS holds the raw Banking77 label slugs the evaluation
        harness (test_banking77_evaluation.py) depends on — this fix must
        never touch its values, only add a separate guide used by the
        live prompt.
        """
        assert INTENT_BUCKETS["budget_analysis"] == [
            "cashflow", "budget", "disposable_income", "spending_breakdown",
        ]


class TestClassifierPromptUsesNaturalLanguage:

    def _captured_prompt(self, user_message: str) -> str:
        agent = ConversationalAgent(MagicMock())
        captured = {}

        def fake_call_llm(prompt, temperature=0.0, system_override=None):
            captured["prompt"] = prompt
            return '{"intent": "budget_analysis", "confidence": 0.9}', 10

        agent._call_llm = fake_call_llm
        agent._classify_intent(user_message)
        return captured["prompt"]

    def test_prompt_contains_natural_phrasing_not_raw_label_slugs(self):
        prompt = self._captured_prompt("where does my money go?")
        assert "where does my money go" in prompt.lower()
        # The old prompt's only signal for this bucket was this bare,
        # comma-joined slug list with no natural language around it —
        # confirm that exact shape is gone, not just that new text exists.
        assert "budget_analysis: cashflow, budget, disposable_income" not in prompt

    def test_every_bucket_guide_description_appears_in_the_prompt(self):
        prompt = self._captured_prompt("hello")
        for description in INTENT_BUCKET_GUIDE.values():
            assert description in prompt


class TestExplanationRequestVsGeneralQueryDisambiguation:
    """
    Regression test for the second prompt-diagnosis fix, grounded directly
    in data/processed/banking77_eval_checkpoint.json's real failures (see
    the coverage audit, Part 7): 53 of 110 general_query utterances were
    misrouted to explanation_request, and 21 more to product_suggestion,
    almost all of them "why is my transfer/payment/card doing X" questions
    that lexically resemble explanation_request's own examples ("why did
    you recommend that") or mention a card, without being either.

    This does NOT re-test classification accuracy end-to-end — same
    limitation as TestClassifierPromptUsesNaturalLanguage above, real
    verification needs a live LLM run (see
    scripts/regenerate_evidence.py --only intent). What it guards is that
    the disambiguating language actually reaches the prompt, and that a
    future edit can't quietly narrow it back to the version that produced
    the 34.8% accuracy figure.
    """

    def _captured_prompt(self, user_message: str) -> str:
        agent = ConversationalAgent(MagicMock())
        captured = {}

        def fake_call_llm(prompt, temperature=0.0, system_override=None):
            captured["prompt"] = prompt
            return '{"intent": "general_query", "confidence": 0.9}', 10

        agent._call_llm = fake_call_llm
        agent._classify_intent(user_message)
        return captured["prompt"]

    def test_explanation_request_excludes_general_why_questions(self):
        prompt = self._captured_prompt("why is my transfer still pending")
        guide = INTENT_BUCKET_GUIDE["explanation_request"]
        assert "never a general" in guide.lower() or "not a general" in guide.lower()
        assert "this assistant" in guide.lower()
        assert guide in prompt

    def test_general_query_explicitly_covers_why_phrased_status_questions(self):
        guide = INTENT_BUCKET_GUIDE["general_query"]
        assert "why" in guide.lower()
        assert "why is my transfer still pending" in guide.lower()

    def test_product_suggestion_excludes_existing_card_status_questions(self):
        guide = INTENT_BUCKET_GUIDE["product_suggestion"]
        assert "new to them" in guide.lower() or "already have" in guide.lower()