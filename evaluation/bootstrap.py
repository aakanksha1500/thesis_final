"""
evaluation/bootstrap.py — non-parametric confidence intervals.

WHY
    Every headline number in results/ was a bare point estimate. At the
    fixture sizes this project was using (n=5 for RQ2, n=10 for RQ5, n=15
    for RQ1), "mean_precision_at_3 = 0.4" means "2 of 5 items hit" — one
    relabelled item moves it by 0.2, and nothing in the results file said
    so. A point estimate with no interval invites exactly one viva
    question, and there is no good answer to it.

    The percentile bootstrap (Efron 1979) is the right tool here: it makes
    no distributional assumption, it works for non-smooth statistics like
    macro-F1 and nDCG where an analytic standard error is awkward, and it
    degrades honestly — at n=5 it returns a comically wide interval, which
    is the correct thing for it to do.

WHAT IS RESAMPLED
    Items, with replacement, n_items at a time — the unit of independent
    observation. For classification metrics that means resampling
    (prediction, ground_truth) PAIRS, so a resample preserves the pairing.
    Resampling predictions and labels independently would destroy the
    correlation being measured and produce meaningless intervals.

REPRODUCIBILITY
    Every function takes an explicit `seed` (default 42) and uses its own
    numpy Generator instance. No global RNG state is touched, so calling a
    bootstrap never perturbs a caller's own random stream, and two runs of
    the same evaluation produce byte-identical intervals.

A NOTE ON DEGENERATE RESAMPLES
    A bootstrap resample of a 5-class problem can easily contain zero
    instances of some class. Macro-F1 is then computed over fewer classes
    than the full sample has. That is a real property of the estimator at
    small n, not a bug, and it is one reason the intervals below are wide.
    `n_degenerate_resamples` is reported so the reader can see how often
    it happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_CONFIDENCE = 0.95


@dataclass
class BootstrapCI:
    """A point estimate with its bootstrap interval and full settings."""
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_resamples: int
    n_items: int
    seed: int
    method: str = "percentile"
    n_degenerate_resamples: int = 0

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": round(self.point_estimate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "ci_width": round(self.width, 4),
            "confidence": self.confidence,
            "method": f"{self.method} bootstrap",
            "n_resamples": self.n_resamples,
            "n_items": self.n_items,
            "seed": self.seed,
            "n_degenerate_resamples": self.n_degenerate_resamples,
        }

    def format(self) -> str:
        return (
            f"{self.point_estimate:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}]"
        )


def bootstrap_metric(
    items: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """
    Percentile bootstrap CI for any statistic computed over a list of items.

    Args:
        items:       the unit of observation, one entry per evaluated item.
        statistic:   items -> float. Called once on the full sample for the
                     point estimate, then n_resamples times on resamples.
        n_resamples: 1000 by default. Below ~500 the percentile endpoints
                     are themselves noticeably noisy.
        confidence:  0.95 -> the 2.5th and 97.5th percentiles.
        seed:        fixed so the interval is reproducible.

    A statistic that raises on a degenerate resample (e.g. a resample with
    one class only) has that resample skipped and counted, rather than
    taking down the whole evaluation.
    """
    n = len(items)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, confidence, n_resamples, 0, seed)

    point = float(statistic(items))

    rng = np.random.default_rng(seed)
    values: list[float] = []
    degenerate = 0

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [items[i] for i in idx]
        try:
            values.append(float(statistic(resample)))
        except Exception:
            degenerate += 1

    if not values:
        return BootstrapCI(point, point, point, confidence, n_resamples, n,
                           seed, n_degenerate_resamples=degenerate)

    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(values, alpha * 100))
    high = float(np.percentile(values, (1.0 - alpha) * 100))

    return BootstrapCI(
        point_estimate=point,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_resamples=n_resamples,
        n_items=n,
        seed=seed,
        n_degenerate_resamples=degenerate,
    )


def bootstrap_paired_metric(
    predictions: Sequence[Any],
    ground_truth: Sequence[Any],
    statistic: Callable[[list, list], float],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """
    Bootstrap CI for a metric of the form f(predictions, ground_truth).

    Resamples INDEX POSITIONS, so each resample keeps every prediction
    with its own label. This is the function to use for accuracy,
    macro-F1, precision, recall and risk-alignment rate.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} labels — a paired bootstrap is undefined"
        )

    pairs = list(zip(predictions, ground_truth))

    def _stat(sample: Sequence[Any]) -> float:
        preds = [p for p, _ in sample]
        truth = [g for _, g in sample]
        return statistic(preds, truth)

    return bootstrap_metric(
        pairs, _stat, n_resamples=n_resamples,
        confidence=confidence, seed=seed,
    )


def bootstrap_proportion(
    successes: Sequence[bool],
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """
    CI for a simple proportion (exact-match rate, hallucination rate,
    routing accuracy). `successes` is one bool per evaluated item.
    """
    items = [bool(s) for s in successes]
    return bootstrap_metric(
        items,
        lambda sample: float(np.mean(sample)) if len(sample) else 0.0,
        n_resamples=n_resamples, confidence=confidence, seed=seed,
    )


def bootstrap_difference(
    successes_a: Sequence[bool],
    successes_b: Sequence[bool],
    paired: bool = True,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """
    CI for (mean(a) - mean(b)) — the interval that actually answers "does
    RAG grounding help?".

    paired=True (the default, and the correct setting for an ablation run
    over the SAME items) resamples item indices once and applies them to
    both conditions, so the per-item correlation between conditions is
    preserved. Treating two conditions over identical items as independent
    samples throws away that pairing and inflates the interval.

    If the resulting interval excludes 0, the difference is significant at
    the stated confidence level.
    """
    if paired and len(successes_a) != len(successes_b):
        raise ValueError(
            "paired difference needs equal-length conditions; got "
            f"{len(successes_a)} and {len(successes_b)}"
        )

    if paired:
        deltas = [
            float(a) - float(b) for a, b in zip(successes_a, successes_b)
        ]
        return bootstrap_metric(
            deltas,
            lambda sample: float(np.mean(sample)) if len(sample) else 0.0,
            n_resamples=n_resamples, confidence=confidence, seed=seed,
        )

    a = [bool(x) for x in successes_a]
    b = [bool(x) for x in successes_b]
    point = float(np.mean(a)) - float(np.mean(b))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        ra = rng.integers(0, len(a), size=len(a))
        rb = rng.integers(0, len(b), size=len(b))
        values.append(
            float(np.mean([a[i] for i in ra])) -
            float(np.mean([b[i] for i in rb]))
        )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapCI(
        point_estimate=point,
        ci_low=float(np.percentile(values, alpha * 100)),
        ci_high=float(np.percentile(values, (1.0 - alpha) * 100)),
        confidence=confidence,
        n_resamples=n_resamples,
        n_items=min(len(a), len(b)),
        seed=seed,
        method="percentile (unpaired)",
    )


def detectable_effect_note(n: int, baseline_rate: float = 0.5) -> dict[str, Any]:
    """
    What a sample of this size can actually resolve — reported alongside
    small-n evaluations so "n=91" comes with a number rather than an
    apology.

    Uses the normal approximation to the binomial for a two-sided test at
    alpha=0.05 with 80% power. Approximate by construction; it is a
    sanity scale, not an inferential claim.
    """
    if n <= 0:
        return {"n": n, "note": "no items"}

    z_alpha, z_beta = 1.96, 0.84
    p = baseline_rate
    se = float(np.sqrt(p * (1 - p) / n))
    mde = (z_alpha + z_beta) * se

    return {
        "n": n,
        "baseline_rate_assumed": p,
        "standard_error": round(se, 4),
        "minimum_detectable_difference": round(float(min(mde, 1.0)), 4),
        "alpha": 0.05,
        "power": 0.80,
        "interpretation": (
            f"With n={n}, a two-sided test at alpha=0.05 with 80% power "
            f"can detect a difference of about "
            f"{min(mde, 1.0) * 100:.1f} percentage points against a "
            f"{p * 100:.0f}% baseline. Smaller true differences are "
            f"likely to be missed."
        ),
    }
