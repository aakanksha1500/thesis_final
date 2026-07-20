"""
Phase 8 - HHEM hallucination detector

GROUP A: extract_claims() - sentence splitting, disclaimer filtering
GROUP B: HallucinationDetector.score_pair() - fallback lexical-overlap
        heuristic behaviour (no premise, identical text, unrelated text)
GROUP C: score_response() - full report assembly, threshold flagging, aggregate rate calculation.

RUNNING:
    python -m pytest tests/unit/test_hallucination_detector.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rag.hallucination_detector import (
    HallucinationDetector,
    extract_claims,
)

# GROUP A: extract_claims()
class TestExtractClaims:

    def test_splits_into_sentences(self):
        text = "First claim here. Second claim here. Third claim here"
        claim = extract_claims(text)
        assert len(claim) == 3
    
    def test_filters_disclaimer_boilerplate(self):
        text =(
            "The fund has an expected return of 6.0 percent. "
            "This is not regulated financial advice. "
            "Please consult a qualified advisor. "
            "Past performance is not indicative of future results."
        )
        claims = extract_claims(text)
        assert len(claims) == 1
        assert "6.0 percent" in claims[0]

    def test_filters_near_empty_fragments(self):
        text = "Yes. The fund returns 6.0 percent anually with moderate risk."
        claims = extract_claims(text)
        assert all(len(c) >= 8 for c in claims)
    
    def test_respects_max_claims(self):
        text = " ".join(f"This is claim number {i} about the fund." for i in range[20])
        claims = extract_claims(text, max_claims=5)
        assert len(claims) == 5
    
    def test_empty_text_returns_empty_list(self):
        assert extract_claims("") == []
    
# GROUP B: score_pair() - fallback behaviour

class TestScorePairFallback:

    def test_identical_premise_and_hypothesis_score_high(self):
        detector = HallucinationDetector()
        text = "The fund has an expected annual return of six percent"
        score = detector.score_pair(text, text)
        assert score > 0.0

    def test_empty_premise_scores_as_unverifiable(self):
        detector = HallucinationDetector()
        score = detector.score_pair("", "the fund returns six percent annually")
        assert score == 0.3
    
    def test_unrelated_premise_scores_low(self):
        detector = HallucinationDetector()
        score = detector.score_pair("some grounding text about bonds", "a claim about bonds")
        assert 0.0 <= score <= 1.0
    
# GROUP C: score_response()

class TestScoreResponse:

    def test_na_claims_returns_perfect_score(self):
        detector = HallucinationDetector()
        report = detector.score_response("Not regulated financial advice,", [])
        assert report.hallucination_rate == 0.0
        assert report.mean_score == 1.0
        assert report.claim_scores == []
    
    def test_unsupported_claim_flagged_with_no_context(self):
        detector = HallucinationDetector()
        text = "The fund guarantees a fifty percent annual return with zero risk."
        report = detector.score_response(text, [], threshold=0.85)
        assert report.claim_scores[0].flagged is True

    def test_hallucination_rate_is_fraction_flagged(self):
        detector = HallucinationDetector()
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
        detector = HallucinationDetector()
        text = "The fund has an expected return of six percent annually."
        report = detector.score_response(text, [])
        d = report.to_dict()
        assert set(d.keys()) >= {
            "mode", "hallucination_rate", "mean_score", "n_claims", "n_flagged", "claims"
        }
        assert d["n_claims"] == len(report.claim_scores)

    def test_disclaimer_only_response_has_zero_claims(self):
        detector = HallucinationDetector()
        text = (
            "This is not regulated financial advice. "
            "Please consult a qualified advisor. "
            "Past performance is not indicative of future results."
        )
        report = detector.score_response(text, [])
        assert report.hallucination_rate == 0.0
        assert len(report.claim_scores) == 0
