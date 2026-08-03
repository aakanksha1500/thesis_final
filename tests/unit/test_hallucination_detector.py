"""
Phase 8 — HHEM hallucination detector.

GROUP A: extract_claims() — sentence splitting, disclaimer filtering
GROUP B: HallucinationDetector.score_pair() — fallback lexical-overlap
         heuristic behaviour (no premise, identical text, unrelated text)
GROUP C: score_response() — full report assembly, threshold flagging,
         aggregate rate calculation

Runs entirely against the lexical-overlap FALLBACK mode (transformers/
torch not installed in this environment) — same rationale as
test_rag_knowledge_base.py testing the hashing embedder fallback. If
transformers+torch ARE installed, HallucinationDetector.mode will
legitimately report "hhem" instead and these tests still pass (they
assert on report *shape* and threshold *behaviour*, not on the specific
scores a lexical heuristic produces).

RUNNING:
    python -m pytest tests/unit/test_hallucination_detector.py -v
"""

from __future__ import annotations

from rag.hallucination_detector import (
    HallucinationDetector,
    extract_claims,
)

# GROUP A: extract_claims()

class TestExtractClaims:

    def test_splits_into_sentences(self):
        text = "First claim here. Second claim here. Third claim here."
        claims = extract_claims(text)
        assert len(claims) == 3

    def test_filters_disclaimer_boilerplate(self):
        text = (
            "The fund has an expected return of 6.0 percent. "
            "This is not regulated financial advice. "
            "Please consult a qualified advisor. "
            "Past performance is not indicative of future results."
        )
        claims = extract_claims(text)
        assert len(claims) == 1
        assert "6.0 percent" in claims[0]

    def test_filters_near_empty_fragments(self):
        text = "Yes. The fund returns 6.0 percent annually with moderate risk."
        claims = extract_claims(text)
        assert all(len(c) >= 8 for c in claims)

    def test_respects_max_claims(self):
        text = " ".join(f"This is claim number {i} about the fund." for i in range(20))
        claims = extract_claims(text, max_claims=5)
        assert len(claims) == 5

    def test_empty_text_returns_empty_list(self):
        assert extract_claims("") == []


# GROUP B: score_pair() — fallback behaviour

class TestScorePairFallback:

    def test_identical_premise_and_hypothesis_scores_high(self):
        detector = HallucinationDetector(force_fallback=True)
        text = "The fund has an expected annual return of six percent"
        score = detector.score_pair(text, text)
        assert score > 0.8

    def test_empty_premise_scores_as_unverifiable(self):
        detector = HallucinationDetector(force_fallback=True)
        score = detector.score_pair("", "the fund returns six percent annually")
        assert score == 0.3

    def test_unrelated_premise_scores_low(self):
        detector = HallucinationDetector(force_fallback=True)
        premise = "Central Bank guidance on deposit protection schemes"
        hypothesis = "the equity fund returned forty five percent last year"
        score = detector.score_pair(premise, hypothesis)
        assert score < 0.5

    def test_score_is_bounded(self):
        detector = HallucinationDetector(force_fallback=True)
        score = detector.score_pair("some grounding text about bonds", "a claim about bonds")
        assert 0.0 <= score <= 1.0


# GROUP B2: init_error reporting
#
# mode alone can't distinguish "packages not installed" from "packages
# installed but the load itself failed" (network, version incompatibility,
# disk space, etc.), and those need different fixes — see
# scripts/verify_real_pipeline.py's check_hallucination().

class TestInitErrorReporting:

    def test_force_fallback_never_attempts_init_so_no_error_recorded(self):
        detector = HallucinationDetector(force_fallback=True)
        assert detector.mode == "fallback"
        assert detector.init_error is None

    def test_init_error_consistent_with_mode(self):
        """
        Environment-agnostic: holds whether this runs somewhere HHEM loads
        successfully, fails to load, or transformers/torch aren't
        installed at all. Real-mode init either succeeds (mode="hhem",
        no error to record) or was attempted and failed (mode="fallback",
        and the reason must be recorded — never silently empty).
        """
        detector = HallucinationDetector()
        if detector.mode == "hhem":
            assert detector.init_error is None
        else:
            assert detector.init_error is not None
            assert len(detector.init_error) > 0


# GROUP C: score_response()

class TestScoreResponse:

    def test_no_claims_returns_perfect_score(self):
        detector = HallucinationDetector(force_fallback=True)
        report = detector.score_response("Not regulated financial advice.", [])
        assert report.hallucination_rate == 0.0
        assert report.mean_score == 1.0
        assert report.claim_scores == []

    def test_grounded_claim_not_flagged(self):
        detector = HallucinationDetector(force_fallback=True)
        text = "The fund has moderate risk with diversified bond exposure."
        contexts = [{
            "text": "The fund has moderate risk with diversified bond exposure across issuers.",
            "source": "Test Source",
        }]
        report = detector.score_response(text, contexts, threshold=0.5)
        assert report.claim_scores[0].score > 0.5
        assert report.claim_scores[0].flagged is False

    def test_unsupported_claim_flagged_with_no_context(self):
        detector = HallucinationDetector(force_fallback=True)
        text = "The fund guarantees a fifty percent annual return with zero risk."
        report = detector.score_response(text, [], threshold=0.85)
        assert report.claim_scores[0].flagged is True

    def test_hallucination_rate_is_fraction_flagged(self):
        detector = HallucinationDetector(force_fallback=True)
        text = (
            "The fund has moderate risk with diversified bond exposure. "
            "The fund guarantees unlimited returns with absolutely no risk whatsoever."
        )
        contexts = [{
            "text": "The fund has moderate risk with diversified bond exposure across issuers.",
            "source": "Test Source",
        }]
        report = detector.score_response(text, contexts, threshold=0.5)
        assert 0.0 <= report.hallucination_rate <= 1.0
        assert report.hallucination_rate == report.to_dict()["hallucination_rate"]

    def test_to_dict_shape(self):
        detector = HallucinationDetector(force_fallback=True)
        text = "The fund has an expected return of six percent annually."
        report = detector.score_response(text, [])
        d = report.to_dict()
        assert set(d.keys()) >= {
            "mode", "hallucination_rate", "mean_score", "n_claims", "n_flagged", "claims"
        }
        assert d["n_claims"] == len(report.claim_scores)

    def test_disclaimer_only_response_has_zero_claims(self):
        detector = HallucinationDetector(force_fallback=True)
        text = (
            "This is not regulated financial advice. "
            "Please consult a qualified advisor. "
            "Past performance is not indicative of future results."
        )
        report = detector.score_response(text, [])
        assert report.hallucination_rate == 0.0
        assert len(report.claim_scores) == 0


# GROUP D: premise sanity check (Day 1c, BUILD_PLAN.md)
#
# A claim lifted verbatim from its own premise must score highly — that's
# the check that would have caught the RQ5 test-harness bug. What was
# actually being scored as a "claim" there sometimes included the fixture
# question sentence alongside the answer; an interrogative sentence has no
# truth value to check against a premise, so it scored low almost by
# construction, unrelated to whether the numeric answer was grounded.
#
# Deliberately NOT force_fallback: runs against whichever mode is active
# (real HHEM if transformers/torch are installed, fallback otherwise). The
# RQ5 numbers were produced in real HHEM mode, so a fallback-only test
# would not have caught this — the assertion holds under both modes,
# since a verbatim/near-verbatim claim should score high regardless of
# which detector backs it.

class TestPremiseSanityCheck:

    def test_verbatim_claim_scores_above_threshold(self):
        detector = HallucinationDetector()
        premise = (
            "Term deposit principal is EUR 10,000, annual rate 2.5 percent, "
            "simple interest for one year."
        )
        claim = "Term deposit principal is EUR 10,000, annual rate 2.5 percent."
        score = detector.score_pair(premise, claim)
        assert score > 0.8, (
            f"A claim copied verbatim from its own premise scored "
            f"{score:.3f} (mode={detector.mode}). If this fails, whatever "
            f"is being passed as 'premise' upstream is not the actual "
            f"grounding text."
        )

    def test_interrogative_sentence_is_not_conflated_with_a_claim(self):
        """
        Locks in the actual Day 1c fix: extract_claims() should not be
        asked to verify a question. Regression guard against
        test_rq5_finqa_evaluation.py reintroducing item["question"] into
        response_text.
        """
        from rag.hallucination_detector import extract_claims
        response_text = "The answer is 250."
        claims = extract_claims(response_text)
        assert claims == ["The answer is 250."]
        assert not any(c.strip().endswith("?") for c in claims)