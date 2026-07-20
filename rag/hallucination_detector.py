"""
Phase 8 - HHEM hallucination detector: the probabilistic half of the 
two-layer hallucination mitigation approach, running in combination with the 
deterministic constraint rules already in place.

Claims scoring below settings.hallucination.hhem_threshold (0.85) are 
flagged in the audit log and can be surfaced in the ExplainabilityAgent 
calibration note.

HHEM scores a pair for factual consistency: does the
hypothesis (a claim from the synthesis) follow from the premise (retrieved
grounding context)? Output is a scalar in [0, 1], 1 = fully consistent.

FALLBACK mode: a deterministic lexical-
overlap heuristic — token Jaccard similarity between claim and the best-
matching grounding context, scaled into a plausible HHEM-like range. This
is intentionally conservative (flags more, not fewer, claims) since a
missed hallucination is worse than a false positive in this domain — same
threshold-asymmetry rationale documented in config/settings.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\d+\.\d+[a-z0-9]+")

_NON_CLAIM_PATTERNS = [
    "not regulated financial advice",
    "consult a qualified advisor",
    "past performance is not",
]


@dataclass
class ClaimScore:
    """HHEM score for one extracted claim from an agent's synthesis text."""
    claim: str
    score: float
    flagged: bool
    best_context: str | None = None
    best_context_source: str | None = None


@dataclass
class HallucinationReport:
    """Aggregate result for one response, over all extracted claims."""
    claim_scores: list[ClaimScore] = field(default_factory=list)
    hallucination_rate: float = 0.0
    mean_score: float = 1.0
    mode: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "n_claims": len(self.claim_scores),
            "n_flagged": sum(1 for c in self.claim_scores if c.flagged),
            "claims": [
                {
                    "claim": c.claim,
                    "score": round(c.score, 4),
                    "flagged": c.flagged,
                    "best_context_source": c.best_context_source,
                }
                for c in self.claim_scores
            ],
        }


def extract_claims(text: str, max_claims: int | None = None) -> list[str]:
    """
    Split synthesis text into sentence-level claims for scoring.
    Filters out CBI-mandated disclaimer boilerplate (not a factual claim
    that can hallucinate) and near-empty fragments.
    """
    max_claims = max_claims or settings.hallucination.max_claims_per_response
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    claims = []
    for s in sentences:
        lowered = s.lower()
        if any(p in lowered for p in _NON_CLAIM_PATTERNS):
            continue
        if len(s) < 8:
            continue
        claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims


class HallucinationDetector:
    """
    Wraps vectara/hallucination_evaluation_model. Initialisation never
    raises — falls back to the lexical-overlap heuristic if transformers
    or torch are not installed, or if the model fails to download/load.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.hallucination.model_id
        self._model = None
        self._mode = "fallback"
        self._init_model()

    def _init_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._mode = "hhem"
            logger.info(f"[HallucinationDetector] Initialised in Real mode - model={self.model_id}")
        except ImportError:
            logger.warning(
                "[HallucinationDetector] transformers/torch not installed - "
                "running FALLBACK mode (lexical-overlap heuristic). Add "
                "transformers>=4.40.0 and torch>=2.2.0 to requirements.txt "
                "and pip install for real HHEM scoring."
            )
        except Exception as exc:
            logger.warning(
                f"[HallucinationDetector] Failed to load '{self.model_id}': "
                f"{exc} - falling back to lexical-overlap heuristic."
            )

    @property
    def mode(self) -> str:
        return self._mode

    def score_pair(self, premise: str, hypothesis: str) -> float:
        """
        Score one (premise, hypothesis) pair. Return a consistency score
        in [0, 1], 1 = hypothesis fully supported by premise.
        """
        if self._mode == "hhem" and self._model is not None:
            try:
                # HHEM's predict() takes a list of (premise, hypothesis) pairs
                # and returns consistency scores in [0, 1]
                scores = self._model.predict([(premise, hypothesis)])
                return float(scores[0])
            except Exception as exc:
                logger.warning(
                    f"[HallucinationDetector] HHEM inference failed: {exc} "
                    f"- using lexical fallback for this pair."
                )
        return self._lexical_overlap_score(premise, hypothesis)

    @staticmethod
    def _lexical_overlap_score(premise: str, hypothesis: str) -> float:
        """
        Deterministic Jaccard token-overlap heuristic, scaled to mimic
        HHEM's operating range. No premise (empty grounding context) is
        scored as unsupported (0.3) rather than 0.0 - an unscored claim is
        not proven false, but it is unverifiable, which should still
        surface as a flag under the default 0.85 threshold.
        """
        if not premise or not premise.strip():
            return 0.3

        p_tokens = set(_TOKEN_RE.findall(premise.lower()))
        h_tokens = set(_TOKEN_RE.findall(hypothesis.lower()))
        if not h_tokens:
            return 0.5

        overlap = len(p_tokens & h_tokens)
        jaccard = overlap / len(p_tokens | h_tokens) if (p_tokens | h_tokens) else 0.0
        # Jaccard on short sentences is naturally low even for genuinely
        # grounded claims; rescale so a moderate overlap (~0.25 Jaccard)
        # lands near the 0.85 threshold rather than always flagging.
        scaled = min(1.0, jaccard * 3.0)
        return round(scaled, 4)

    def score_response(
            self,
            response_text: str,
            grounding_contexts: list[dict[str, Any]],
            threshold: float | None = None,
    ) -> HallucinationReport:
        """
        Extract claims from 'response_text' and score each against the best-
        matching grounding context (typically RAG citations retrieved for
        the same response - see agents/investment_agent.py wiring).

        Args:
            response_text: The agent's synthesis text to check.
            grounding_contexts: List of {"text": str, "source": str, ...}
                                dicts - e.g. KnowledgeBase.retrieve() output.
                                If empty, every claim is scored against an
                                empty premise.
            threshold: Defaults to settings.hallucination.hhem_threshold.

        Returns:
            HallucinationReport with per-claim scores and an aggregate rate.
        """
        threshold = threshold if threshold is not None else settings.hallucination.hhem_threshold
        claims = extract_claims(response_text)

        if not claims:
            return HallucinationReport(claim_scores=[], hallucination_rate=0.0, mean_score=1.0, mode=self._mode)

        claim_scores: list[ClaimScore] = []
        for claim in claims:
            best_score = -1.0
            best_ctx: dict[str, Any] | None = None
            contexts = grounding_contexts or [{"text": "", "source": None}]
            for ctx in contexts:
                score = self.score_pair(ctx.get("text", ""), claim)
                if score > best_score:
                    best_score = score
                    best_ctx = ctx

            flagged = best_score < threshold
            claim_scores.append(ClaimScore(
                claim=claim,
                score=best_score,
                flagged=flagged,
                best_context=(best_ctx or {}).get("text"),
                best_context_source=(best_ctx or {}).get("source"),
            ))

        n_flagged = sum(1 for c in claim_scores if c.flagged)
        mean_score = sum(c.score for c in claim_scores) / len(claim_scores)

        return HallucinationReport(
            claim_scores=claim_scores,
            hallucination_rate=n_flagged / len(claim_scores),
            mean_score=mean_score,
            mode=self._mode,
        )


hallucination_detector = HallucinationDetector()