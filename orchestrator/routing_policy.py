"""
The routing decision: given a classified intent and its confidence, decide
whether to accept it, ask for confirmation, or fall back to a generic
reply. Pure function, no I/O — see RoutingPolicy.decide().

WHY THIS FILE EXISTS (fixes audit flaw #1)

    Three separate defects were compressed into those two lines.

    (a) A CORRECT classification was thrown away. The isolated classifier
        scores 96.4% on the five advisory buckets, but embedded in
        process_turn() it routed correctly 30% of the time, and every
        misroute landed on conversational_only. The gate, not the
        classifier, was the failure.

    (b) "The classifier could not answer" and "the classifier answered with
        low confidence" were the same value. ConversationalAgent._classify_
        intent returns ("general_query", 0.5) on a JSON parse failure, on an
        API error, and in mock mode. 0.5 < 0.65, so an infrastructure
        failure was silently indistinguishable from a genuinely ambiguous
        user message — and both were reported as a routing decision rather
        than as an outage.

    (c) The gating score was the LLM's own self-reported confidence, which
        is not calibrated against anything. Nothing in the system had ever
        checked whether 0.65 on that scale corresponded to any particular
        accuracy.

    The replacement separates the three concerns:

        classify   → produces an IntentSignal that can be ABSTAIN (no answer)
                     as distinct from a low-scoring answer
        calibrate  → maps a raw self-reported score to P(correct) using a
                     fitted calibrator, or the identity map with an explicit
                     `calibrated=False` marker when none is fitted
        decide     → applies a three-band policy over the calibrated score

    and the decision records WHY it decided, so a misroute is now
    attributable from the audit log without needing the tmpdir JSONL that
    the original RQ4 audit pointed at and did not ship.

THE THREE-BAND POLICY
    accept  (p >= accept_at)            route to the specialist
    clarify (clarify_at <= p < accept_at)
                                        route to the specialist ANYWAY, but
                                        mark the turn low-confidence so the
                                        approval gate sees it; for buckets
                                        listed in `confirm_before_acting`,
                                        ask a targeted one-line confirmation
                                        naming the detected intent instead
    reject  (p < clarify_at)            conversational fallback, with the
                                        rejected candidate preserved

    The middle band is the entire point. The old code had no middle band:
    everything under one threshold became a generic chat reply that named no
    intent, which is both a worse user experience and unrecoverable
    downstream. Routing a moderately-confident "assess my risk profile" to
    RiskProfilingAgent and letting the approval gate hold the output is
    strictly safer than answering it as small talk, because the specialist
    path is the one with the constraint checks, the disclaimers and the
    human review queue attached.

DETERMINISTIC LEXICAL PRIOR
    The LLM is not the only signal available. `lexical_prior()` is a
    deterministic keyword scorer over the same bucket taxonomy. It is used
    two ways:

      - as a TIE-BREAKER when the LLM's top-2 are close together, and
      - as the SOLE signal when the LLM abstained (parse failure, API error,
        mock mode).

    This is what makes routing degrade gracefully instead of collapsing to
    conversational_only, and it is what makes routing testable offline with
    no API key — which is why the RQ4 harness can now exercise multi-agent
    paths at all (see audit flaw #8).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Band",
    "IntentSignal",
    "RoutingOutcome",
    "RoutingPolicy",
    "ConfidenceCalibrator",
    "lexical_prior",
    "ABSTAINED",
]


# A sentinel score meaning "the classifier produced no usable answer".
# Distinct from 0.0, which would mean "answered, with no confidence at all".
ABSTAINED = None


class Band(str, Enum):
    ACCEPT = "accept"
    CLARIFY = "clarify"
    REJECT = "reject"
    ABSTAIN = "abstain"


# Deterministic lexical prior
#
# Weights are intentionally crude and intentionally visible. This is a
# fallback and a tie-breaker, not a model; anything more elaborate here
# would be an unlabelled second classifier competing with the first one.
_LEXICON: dict[str, tuple[tuple[str, float], ...]] = {
    "risk_profiling": (
        (r"\brisk (profile|profiling|appetite|tolerance|assessment)\b", 3.0),
        (r"\bhow much risk\b", 3.0),
        (r"\b(assess|evaluate|work out|figure out).{0,20}\brisk\b", 2.5),
        (r"\brisk\b", 1.0),
        (r"\b(conservative|aggressive|moderate)\b", 0.8),
        (r"\binvestment horizon\b", 1.2),
        (r"\bfinancial goals?\b", 1.0),
    ),
    "investment_advice": (
        (r"\b(what|which|where) should i invest\b", 3.0),
        (r"\b(invest|investing|investment)\b", 1.5),
        (r"\b(portfolio|fund|funds|etf|isa|equities|shares|stocks?|bonds?)\b", 1.4),
        (r"\b(returns?|yield|interest rate)\b", 0.9),
        (r"\bsavings? (account|rate|product)\b", 1.2),
        (r"\brecommend\b", 0.8),
    ),
    "budget_analysis": (
        (r"\bbudget(ing)?\b", 2.5),
        (r"\b(spending|expenses?|outgoings?|cash ?flow)\b", 1.8),
        (r"\bdisposable income\b", 2.2),
        (r"\bsavings rate\b", 1.8),
        (r"\bwhere (is|does) my money go(ing)?\b", 2.5),
        (r"\bafford\b", 1.0),
    ),
    "product_suggestion": (
        (r"\b(which|what) (card|account|product)\b", 2.5),
        (r"\b(card|cheque|checque|chequebook)\b", 1.2),
        (r"\bopen an? account\b", 1.5),
    ),
    "explanation_request": (
        (r"\bwhy (did|do|does|are|is|was)\b", 3.0),
        (r"\b(explain|explanation)\b", 2.5),
        (r"\bhow did you\b", 3.0),
        (r"\bwhat made you\b", 2.5),
        (r"\bon what basis\b", 2.5),
        (r"\breasoning\b", 1.8),
        (r"\bjustify\b", 1.8),
    ),
    "full_advisory": (
        (r"\bwhere (do i|to) (start|begin)\b", 3.0),
        (r"\b(complete|full|whole|overall|holistic|entire) (financial )?"
         r"(review|plan|picture|situation|position|finances)\b", 3.0),
        (r"\bnew to (all )?(this|investing|finance|money)\b", 2.5),
        (r"\breview my (whole |entire |overall )?(financial|finances|money)\b", 2.8),
        (r"\bhelp me with my finances\b", 2.2),
        (r"\bsort out my (money|finances)\b", 2.2),
        (r"\beverything (about|to do with) my (money|finances)\b", 2.2),
    ),
    "general_query": (
        (r"\b(balance|statement)\b", 1.8),
        (r"\bexchange rate\b", 2.0),
        (r"\b(expire|expiry|expired)\b", 1.5),
        (r"\b(hello|hi|hey|thanks|thank you)\b", 1.2),
        (r"\bwhat is my\b", 0.8),
    ),
    "out_of_scope": (
        (r"\b(legal|lawyer|solicitor|sue)\b", 2.0),
        (r"\b(medical|doctor|health insurance claim)\b", 2.0),
        (r"\bcomplain(t|ing)?\b", 1.5),
    ),
}


def lexical_prior(message: str) -> dict[str, float]:
    """
    Score every bucket against a deterministic keyword lexicon.

    Returns a dict of bucket -> score in [0, 1], softmax-normalised so it is
    comparable with a self-reported LLM confidence. An all-zero raw score
    (nothing matched) returns an empty dict rather than a uniform
    distribution, because "no evidence" and "evidence for everything
    equally" are different states and only the first should abstain.
    """
    text = (message or "").lower()
    raw: dict[str, float] = {}
    for bucket, patterns in _LEXICON.items():
        score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text):
                score += weight
        if score > 0:
            raw[bucket] = score

    if not raw:
        return {}

    # Temperature 1.5 deliberately flattens this. It is a prior, and a
    # confident-looking prior would overrule the LLM in the tie-break path,
    # which is not what it is for.
    temperature = 1.5
    mx = max(raw.values())
    exp = {k: math.exp((v - mx) / temperature) for k, v in raw.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


@dataclass
class IntentSignal:
    """
    What the classifier actually produced, including the ability to say
    nothing at all.

    intent:
        Best-guess bucket, or None if the classifier abstained.
    raw_confidence:
        The classifier's own self-reported score, or ABSTAINED (None).
        NEVER substitute a number here for a failure — that conflation is
        audit flaw #1(b) and this field's type is the fix.
    alternatives:
        Optional full distribution over buckets, when the classifier can
        provide one. Enables margin-based ambiguity detection, which is a
        far better signal than an absolute threshold.
    source:
        "llm", "lexical", "llm+lexical", or "none".
    error:
        Populated when the classifier failed, so an outage is reportable as
        an outage rather than as a routing statistic.
    """
    intent: str | None
    raw_confidence: float | None
    alternatives: dict[str, float] = field(default_factory=dict)
    source: str = "llm"
    error: str | None = None

    @property
    def abstained(self) -> bool:
        return self.intent is None or self.raw_confidence is ABSTAINED

    @property
    def margin(self) -> float | None:
        """
        Top-1 minus top-2. None when fewer than two candidates are known.

        Margin is the ambiguity signal an absolute threshold was standing in
        for. "risk 0.55 / investment 0.05" is a confident answer with a low
        absolute score; "risk 0.55 / investment 0.52" is a genuinely
        ambiguous one. The old single threshold treated them identically.
        """
        if len(self.alternatives) < 2:
            return None
        ordered = sorted(self.alternatives.values(), reverse=True)
        return ordered[0] - ordered[1]


@dataclass
class RoutingOutcome:
    """
    A routing decision plus the full reason it was made.

    `detected_intent` survives even when the policy declines to act on it.
    That is the single most important property of this dataclass: under the
    old code the detected intent was discarded at the gate, so an audit log
    could not distinguish "classifier said conversational" from "classifier
    said risk_profiling and the gate overrode it". Every RQ4 misroute was
    the second case and nothing recorded it.
    """
    intent: str | None
    band: Band
    routed_intent: str | None
    calibrated_confidence: float | None
    raw_confidence: float | None
    margin: float | None
    source: str
    reason: str
    needs_confirmation: bool = False
    confirmation_prompt: str | None = None
    degraded: bool = False
    error: str | None = None

    def to_audit(self) -> dict[str, Any]:
        return {
            "detected_intent": self.intent,
            "routed_intent": self.routed_intent,
            "band": self.band.value,
            "raw_confidence": self.raw_confidence,
            "calibrated_confidence": self.calibrated_confidence,
            "margin": self.margin,
            "source": self.source,
            "reason": self.reason,
            "needs_confirmation": self.needs_confirmation,
            "degraded": self.degraded,
            "error": self.error,
        }


class ConfidenceCalibrator:
    """
    Maps a raw self-reported confidence to an empirical P(correct).

    WHY (fixes audit flaw #7 at the routing layer)
        A threshold is only meaningful on a calibrated scale. The system
        compared a self-reported LLM score against 0.65 and an approval gate
        compared a decision-margin against 0.6, and neither number had ever
        been checked against observed accuracy. On the risk model the check
        was eventually run and the signal came out INVERSELY calibrated
        (AUROC 0.391 — worse than chance), which means the gate was
        systematically holding back the more accurate predictions.

        This class is deliberately minimal: an isotonic (monotone
        step-function) fit stored as plain JSON, no sklearn dependency at
        inference time, so it can be loaded in any environment including the
        API process. Fitting happens offline in
        scripts/fit_confidence_calibrators.py.

    IMPORTANT — the identity default is explicit, not silent.
        With no fitted artefact, `transform` returns the raw score and
        `is_fitted` is False. Callers are expected to surface that, so that
        an uncalibrated deployment is a visible state rather than one that
        looks identical to a calibrated one. This is the same principle as
        IntentSignal.abstained.
    """

    def __init__(self, knots: list[tuple[float, float]] | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        self._knots = sorted(knots or [])
        self.metadata = metadata or {}

    @property
    def is_fitted(self) -> bool:
        return len(self._knots) >= 2

    def transform(self, raw: float | None) -> float | None:
        if raw is ABSTAINED:
            return None
        raw = float(raw)
        if not self.is_fitted:
            return raw
        xs = [k[0] for k in self._knots]
        ys = [k[1] for k in self._knots]
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if raw <= xs[i]:
                x0, x1 = xs[i - 1], xs[i]
                y0, y1 = ys[i - 1], ys[i]
                if x1 == x0:
                    return y1
                t = (raw - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return ys[-1]

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"knots": [list(k) for k in self._knots], "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConfidenceCalibrator":
        return cls(
            knots=[(float(a), float(b)) for a, b in d.get("knots", [])],
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceCalibrator":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except Exception:
            return cls()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    # -- fitting (offline; numpy-free so it can live next to inference) --
    @classmethod
    def fit_isotonic(
        cls,
        scores: Iterable[float],
        correct: Iterable[bool],
        metadata: dict[str, Any] | None = None,
    ) -> "ConfidenceCalibrator":
        """
        Pool-Adjacent-Violators isotonic regression of correctness on score.

        Isotonic rather than Platt because we do not want to assume the raw
        score is even monotonically related to correctness with a sigmoid
        shape — on this system's risk model it turned out to be monotone in
        the WRONG direction, and a flexible monotone fit surfaces that as a
        near-flat calibration curve rather than hiding it inside fitted
        sigmoid parameters.
        """
        pairs = sorted(zip([float(s) for s in scores], [1.0 if c else 0.0 for c in correct]))
        if len(pairs) < 2:
            return cls(metadata=metadata)

        # Blocks are (max_x, weight, mean_value). PAVA merges adjacent blocks
        # whenever the sequence of means is decreasing, until it is monotone
        # non-decreasing. Each block's max_x becomes a knot, so the resulting
        # step function is right-continuous over the observed score range.
        blocks: list[list[float]] = []
        for x, y in pairs:
            blocks.append([x, 1.0, y])
            while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
                x2, w2, v2 = blocks.pop()
                x1, w1, v1 = blocks.pop()
                merged_w = w1 + w2
                blocks.append([
                    max(x1, x2),
                    merged_w,
                    (v1 * w1 + v2 * w2) / merged_w,
                ])

        # Deduplicate on x, keeping the last (highest) fitted value.
        dedup: dict[float, float] = {}
        for x, _w, v in blocks:
            dedup[x] = v

        # A single knot is not an interpolable curve; treat as unfitted.
        if len(dedup) < 2:
            return cls(metadata={**(metadata or {}), "degenerate": True})

        return cls(knots=sorted(dedup.items()), metadata=metadata)


@dataclass
class RoutingPolicy:
    """
    The three-band decision, with every threshold named and defaulted
    conservatively.

    accept_at / clarify_at:
        Operate on the CALIBRATED score when a calibrator is fitted, and on
        the raw score otherwise. The defaults are deliberately far apart so
        the clarify band is wide: in an advisory system the expensive error
        is answering a specialist question as small talk, not asking one
        confirming question.

    min_margin:
        When the classifier supplies a distribution, a top-1/top-2 gap below
        this is treated as ambiguous regardless of absolute score.

    confirm_before_acting:
        Buckets where acting on a clarify-band signal without asking is not
        acceptable. Deliberately small. Note what is NOT in it: the whole
        point of the fix is that most clarify-band traffic still reaches the
        specialist.

    lexical_fallback:
        When True (default), an abstained LLM falls back to the
        deterministic lexical prior instead of collapsing to
        conversational_only. This is what makes offline/mock evaluation of
        routing meaningful rather than trivially zero.
    """
    accept_at: float = 0.55
    clarify_at: float = 0.25
    min_margin: float = 0.10
    confirm_before_acting: frozenset[str] = frozenset({"full_advisory"})
    lexical_fallback: bool = True
    lexical_accept_at: float = 0.30
    calibrator: ConfidenceCalibrator = field(default_factory=ConfidenceCalibrator)

    def decide(self, signal: IntentSignal, message: str = "") -> RoutingOutcome:
        """
        Turn an IntentSignal into a RoutingOutcome. Pure function; no I/O,
        no LLM calls, fully unit-testable.
        """
        # -- Case 1: classifier abstained ------------------------------
        if signal.abstained:
            if self.lexical_fallback:
                prior = lexical_prior(message)
                if prior:
                    best = max(prior, key=prior.get)
                    score = prior[best]
                    if score >= self.lexical_accept_at and best not in ("out_of_scope",):
                        return RoutingOutcome(
                            intent=best,
                            band=Band.CLARIFY,
                            routed_intent=best,
                            calibrated_confidence=score,
                            raw_confidence=None,
                            margin=_margin_of(prior),
                            source="lexical",
                            reason=(
                                "LLM classifier abstained "
                                f"({signal.error or 'no answer'}); routed on the "
                                f"deterministic lexical prior at {score:.2f}. "
                                "Turn marked degraded."
                            ),
                            degraded=True,
                            error=signal.error,
                        )
            return RoutingOutcome(
                intent=None,
                band=Band.ABSTAIN,
                routed_intent=None,
                calibrated_confidence=None,
                raw_confidence=None,
                margin=None,
                source="none",
                reason=(
                    "Classifier produced no usable answer and no lexical "
                    f"evidence was found ({signal.error or 'no answer'}). "
                    "This is an availability failure, not a routing result — "
                    "it must not be counted as a conversational_only "
                    "classification in any accuracy metric."
                ),
                degraded=True,
                error=signal.error,
            )

        calibrated = self.calibrator.transform(signal.raw_confidence)
        margin = signal.margin

        # -- Case 2: ambiguous by margin, regardless of absolute score --
        if margin is not None and margin < self.min_margin:
            ordered = sorted(signal.alternatives.items(), key=lambda kv: -kv[1])
            top2 = [k for k, _ in ordered[:2]]
            return RoutingOutcome(
                intent=signal.intent,
                band=Band.CLARIFY,
                routed_intent=signal.intent,
                calibrated_confidence=calibrated,
                raw_confidence=signal.raw_confidence,
                margin=margin,
                source=signal.source,
                reason=(
                    f"Top-2 buckets {top2} separated by only {margin:.3f} "
                    f"(< {self.min_margin}); genuinely ambiguous. Routed to "
                    f"'{signal.intent}' with confirmation."
                ),
                needs_confirmation=True,
                confirmation_prompt=_confirmation_for(signal.intent),
            )

        score = calibrated if calibrated is not None else 0.0

        # -- Case 3: accept --------------------------------------------
        if score >= self.accept_at:
            return RoutingOutcome(
                intent=signal.intent,
                band=Band.ACCEPT,
                routed_intent=signal.intent,
                calibrated_confidence=calibrated,
                raw_confidence=signal.raw_confidence,
                margin=margin,
                source=signal.source,
                reason=f"Calibrated confidence {score:.2f} >= accept_at {self.accept_at}.",
            )

        # -- Case 4: clarify band — STILL ROUTES ------------------------
        if score >= self.clarify_at:
            needs_confirm = signal.intent in self.confirm_before_acting
            return RoutingOutcome(
                intent=signal.intent,
                band=Band.CLARIFY,
                routed_intent=signal.intent,
                calibrated_confidence=calibrated,
                raw_confidence=signal.raw_confidence,
                margin=margin,
                source=signal.source,
                reason=(
                    f"Calibrated confidence {score:.2f} in clarify band "
                    f"[{self.clarify_at}, {self.accept_at}). Routing to "
                    f"'{signal.intent}' anyway and marking the turn for the "
                    "approval gate — answering a specialist question as small "
                    "talk is the more expensive error."
                    + (" Confirmation requested first." if needs_confirm else "")
                ),
                needs_confirmation=needs_confirm,
                confirmation_prompt=_confirmation_for(signal.intent) if needs_confirm else None,
            )

        # -- Case 5: reject --------------------------------------------
        return RoutingOutcome(
            intent=signal.intent,
            band=Band.REJECT,
            routed_intent="general_query",
            calibrated_confidence=calibrated,
            raw_confidence=signal.raw_confidence,
            margin=margin,
            source=signal.source,
            reason=(
                f"Calibrated confidence {score:.2f} < clarify_at "
                f"{self.clarify_at}. Falling back to conversation. Detected "
                f"intent '{signal.intent}' is preserved in this record and in "
                "the audit log rather than discarded."
            ),
        )


def _margin_of(dist: dict[str, float]) -> float | None:
    if len(dist) < 2:
        return None
    ordered = sorted(dist.values(), reverse=True)
    return ordered[0] - ordered[1]


_CONFIRMATION_TEXT = {
    "risk_profiling": "Just to check — would you like me to run a risk profile for you?",
    "investment_advice": "Just to check — are you asking for investment suggestions?",
    "budget_analysis": "Just to check — would you like me to look at your budget?",
    "product_suggestion": "Just to check — are you looking for a product recommendation?",
    "explanation_request": "Just to check — would you like me to explain how I reached that?",
    "full_advisory": (
        "That sounds like a full review of your finances — risk, budget and "
        "investments together. Shall I go through all three?"
    ),
}


def _confirmation_for(intent: str | None) -> str | None:
    if intent is None:
        return None
    return _CONFIRMATION_TEXT.get(intent)
