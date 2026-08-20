"""
evaluation/risk_rubric.py — the INDEPENDENT labelling instrument.

WHY THIS MODULE EXISTS, AND WHY IT IMPORTS NOTHING FROM agents/
    A ground-truth label produced by the system under test is not ground
    truth, it is a fixed point. phase3_risk_heuristic.json in this
    repository shows exactly what that looks like: macro-F1 = 1.000,
    because the "labels" and the predictions came from the same rule.

    So this module is the labelling instrument, and it is deliberately
    firewalled from the thing being labelled:

      * it imports NOTHING from agents/ (enforced by a test — see
        tests/unit/test_evaluation_layers.py::TestRubricIndependence);
      * it never loads risk_model.pkl, never computes a SHAP value, and
        never sees a model prediction;
      * it is an ORDINAL BAND MATRIX, not a continuous weighted score.
        RiskProfilingAgent computes 0.6*ml + 0.4*rule on [0,1] and cuts
        the result into five equal-width bins. This computes two ordinal
        bands from integer points and looks the tier up in a 3x3 table.
        Different functional form, different failure modes.

WHAT IT SHARES WITH THE AGENT, STATED PLAINLY
    Both encode the same regulatory constructs, because there is only one
    set of them: MiFID II Art. 25(2) and the CBI's suitability guidance
    both require a firm to assess (a) the client's ability to bear loss
    — "risk capacity", objective and financial — and (b) the client's
    attitude to risk — "risk tolerance", subjective and psychometric —
    and to recommend on the LOWER of the two. Any honest instrument for
    this task will use income, debt, dependents, employment, horizon and
    stated loss tolerance, because those are the inputs the regulation
    names.

    That shared construct space means agreement between this rubric and
    the agent's rule layer is EXPECTED to be well above chance. It is not
    evidence of circularity, and it is also not evidence that the agent
    is correct. To keep that visible rather than buried, every evaluation
    that uses this rubric also reports `rubric_vs_rule_layer_agreement`
    as a diagnostic, so a reader can see the size of the shared structure
    instead of guessing at it.

THE TWO AXES

    RISK CAPACITY (ability to bear loss) — integer points, objective only.
        Debt-to-income        DTI < 0.10            +2
                              0.10 <= DTI < 0.25    +1
                              0.25 <= DTI <= 0.50    0
                              DTI > 0.50            -2
        Income                < EUR 25,000          -1
                              25,000-59,999          0
                              60,000-99,999         +1
                              >= EUR 100,000        +2
        Dependents            0                     +1
                              1-2                    0
                              3+                    -1
        Employment            employed              +1
                              self-employed /
                              part-time / student    0
                              retired / unemployed  -1
        Investment horizon    < 3 years             -2
                              3-5 years              0
                              6-10 years            +1
                              > 10 years            +2

        Range [-7, +8].  low: <= 0   medium: 1-4   high: >= 5

        Horizon sits on the CAPACITY axis, not the tolerance axis, because
        time to recover from a drawdown is a property of the client's
        circumstances, not of their attitude — the same reason ESMA's
        suitability guidelines group holding period with capacity.

    RISK TOLERANCE (willingness) — integer points, self-reported only.
        loss_tolerance (1-5)          (value - 3) * 2     -> [-4, +4]
        financial_knowledge (1-5)     (value - 3) * 1     -> [-2, +2]

        Range [-6, +6].  low: <= -2   medium: -1..+1   high: >= +2

        loss_tolerance is weighted twice financial_knowledge because
        knowledge is a comprehension check, not a preference: a client who
        understands products well but dislikes losses is not thereby a
        higher-risk client. Knowledge only moves the client at the margin.

    THE SUITABILITY MATRIX (capacity caps tolerance)

                       tol_low                 tol_med                    tol_high
        cap_low        conservative            conservative               moderately_conservative
        cap_med        moderately_conservative moderate                   moderately_aggressive
        cap_high       moderate                moderately_aggressive      aggressive

        Read down a column: capacity always moves the tier. Read across a
        row: tolerance always moves the tier, but can never lift a
        low-capacity client past moderately_conservative. That asymmetry
        IS the regulatory rule — willingness cannot exceed ability — and
        it is the single most important property of this table.

PROVENANCE
    Written for the gold-standard evaluation set (Layer 3). The thresholds
    above were fixed BEFORE any profile was generated and before any model
    was run against them, and have not been adjusted since. If a future
    change to them is ever needed, the gold set must be rebuilt and
    re-labelled from scratch, and the old results discarded — not the
    other way round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RUBRIC_VERSION = "1.0"

RISK_TIERS: list[str] = [
    "conservative",
    "moderately_conservative",
    "moderate",
    "moderately_aggressive",
    "aggressive",
]

# Employment strings this rubric recognises. Both spellings of
# self-employed appear in this codebase (agents/risk_profiling_agent.py
# checks for both), so both are recognised here too.
_EMPLOYMENT_POINTS: dict[str, int] = {
    "employed": 1,
    "self_employed": 0,
    "self-employed": 0,
    "part_time": 0,
    "part-time": 0,
    "student": 0,
    "retired": -1,
    "unemployed": -1,
}

CAPACITY_BANDS: list[tuple[int, str]] = [(0, "low"), (4, "medium"), (8, "high")]
TOLERANCE_BANDS: list[tuple[int, str]] = [(-2, "low"), (1, "medium"), (6, "high")]

SUITABILITY_MATRIX: dict[tuple[str, str], str] = {
    ("low", "low"): "conservative",
    ("low", "medium"): "conservative",
    ("low", "high"): "moderately_conservative",
    ("medium", "low"): "moderately_conservative",
    ("medium", "medium"): "moderate",
    ("medium", "high"): "moderately_aggressive",
    ("high", "low"): "moderate",
    ("high", "medium"): "moderately_aggressive",
    ("high", "high"): "aggressive",
}

# Which (capacity_band, tolerance_band) cells produce each tier. Used by
# scripts/build_gold_risk_dataset.py to construct profiles LABEL-FIRST,
# and asserted against SUITABILITY_MATRIX by the test suite so the two can
# never drift apart.
CELLS_BY_TIER: dict[str, list[tuple[str, str]]] = {}
for _cell, _tier in SUITABILITY_MATRIX.items():
    CELLS_BY_TIER.setdefault(_tier, []).append(_cell)


@dataclass
class RubricLabel:
    """
    A rubric decision plus the full arithmetic that produced it.

    Every gold profile stores one of these. A viva question of the form
    "why is this profile labelled aggressive?" is answered by reading
    `capacity_breakdown` and `tolerance_breakdown` — not by re-running
    anything, and certainly not by consulting the model.
    """
    risk_class: str
    capacity_points: int
    capacity_band: str
    tolerance_points: int
    tolerance_band: str
    capacity_breakdown: dict[str, int] = field(default_factory=dict)
    tolerance_breakdown: dict[str, int] = field(default_factory=dict)
    rubric_version: str = RUBRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_class": self.risk_class,
            "rubric_version": self.rubric_version,
            "capacity": {
                "points": self.capacity_points,
                "band": self.capacity_band,
                "breakdown": self.capacity_breakdown,
            },
            "tolerance": {
                "points": self.tolerance_points,
                "band": self.tolerance_band,
                "breakdown": self.tolerance_breakdown,
            },
        }


def _band(points: int, bands: list[tuple[int, str]]) -> str:
    """First band whose upper bound the points fall at or below."""
    for upper, name in bands:
        if points <= upper:
            return name
    return bands[-1][1]


def capacity_points(features: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Objective ability-to-bear-loss points. See module docstring."""
    income = float(features.get("income", 0.0))
    debt = float(features.get("existing_debt", 0.0))
    dti = debt / income if income > 0 else 1.0

    if dti < 0.10:
        dti_pts = 2
    elif dti < 0.25:
        dti_pts = 1
    elif dti <= 0.50:
        dti_pts = 0
    else:
        dti_pts = -2

    if income < 25_000:
        income_pts = -1
    elif income < 60_000:
        income_pts = 0
    elif income < 100_000:
        income_pts = 1
    else:
        income_pts = 2

    dependents = int(features.get("dependents", 0))
    if dependents == 0:
        dep_pts = 1
    elif dependents <= 2:
        dep_pts = 0
    else:
        dep_pts = -1

    employment = str(features.get("employment_status", "employed"))
    emp_pts = _EMPLOYMENT_POINTS.get(employment, 0)

    horizon = float(features.get("investment_horizon", 5))
    if horizon < 3:
        hor_pts = -2
    elif horizon <= 5:
        hor_pts = 0
    elif horizon <= 10:
        hor_pts = 1
    else:
        hor_pts = 2

    breakdown = {
        "debt_to_income": dti_pts,
        "income": income_pts,
        "dependents": dep_pts,
        "employment_status": emp_pts,
        "investment_horizon": hor_pts,
    }
    return sum(breakdown.values()), breakdown


def tolerance_points(features: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Self-reported willingness points. See module docstring."""
    loss_tolerance = int(features.get("loss_tolerance", 3))
    knowledge = int(features.get("financial_knowledge_score", 3))

    breakdown = {
        "loss_tolerance": (loss_tolerance - 3) * 2,
        "financial_knowledge_score": (knowledge - 3) * 1,
    }
    return sum(breakdown.values()), breakdown


def label(features: dict[str, Any]) -> RubricLabel:
    """
    Assign a ground-truth risk tier from features alone.

    This function is the ONLY place gold labels come from. It is pure: no
    model, no randomness, no I/O, no global state. Calling it twice with
    the same features always returns the same tier, on any machine, in
    any Python version.
    """
    cap_pts, cap_breakdown = capacity_points(features)
    tol_pts, tol_breakdown = tolerance_points(features)

    cap_band = _band(cap_pts, CAPACITY_BANDS)
    tol_band = _band(tol_pts, TOLERANCE_BANDS)

    return RubricLabel(
        risk_class=SUITABILITY_MATRIX[(cap_band, tol_band)],
        capacity_points=cap_pts,
        capacity_band=cap_band,
        tolerance_points=tol_pts,
        tolerance_band=tol_band,
        capacity_breakdown=cap_breakdown,
        tolerance_breakdown=tol_breakdown,
    )


def rubric_documentation() -> dict[str, Any]:
    """
    Machine-readable description of the instrument, embedded verbatim in
    every gold dataset file and every results file that uses it — so a
    results JSON is self-describing years after the fact, without needing
    this source file alongside it.
    """
    return {
        "rubric_version": RUBRIC_VERSION,
        "instrument": "two-axis MiFID II / CBI suitability band matrix",
        "axes": {
            "risk_capacity": {
                "kind": "objective (financial circumstances)",
                "inputs": [
                    "income", "existing_debt", "dependents",
                    "employment_status", "investment_horizon",
                ],
                "point_range": [-7, 8],
                "bands": {"low": "<= 0", "medium": "1 to 4", "high": ">= 5"},
            },
            "risk_tolerance": {
                "kind": "subjective (self-reported preference)",
                "inputs": ["loss_tolerance", "financial_knowledge_score"],
                "point_range": [-6, 6],
                "bands": {"low": "<= -2", "medium": "-1 to 1", "high": ">= 2"},
            },
        },
        "matrix": {f"{c}|{t}": tier for (c, t), tier in SUITABILITY_MATRIX.items()},
        "governing_principle": (
            "Capacity caps tolerance: a low-capacity client cannot be "
            "labelled above moderately_conservative regardless of stated "
            "willingness (MiFID II Art. 25(2); CBI suitability guidance)."
        ),
        "independence": (
            "Imports nothing from agents/. Never loads risk_model.pkl, "
            "never computes SHAP, never observes a model prediction. "
            "Labels are assigned before any model is run and are never "
            "revised in response to model output."
        ),
        "known_shared_structure": (
            "Shares regulatory CONSTRUCTS (income, debt, dependents, "
            "employment, horizon, stated tolerance) with the agent's rule "
            "layer, because the regulation names those inputs. Differs in "
            "functional form (ordinal band matrix vs continuous weighted "
            "score with equal-width binning) and shares nothing with the "
            "agent's ML layer. Agreement with the rule layer is reported "
            "as a diagnostic in every evaluation that uses this rubric."
        ),
    }
