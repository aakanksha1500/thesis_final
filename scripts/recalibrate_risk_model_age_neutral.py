"""
Recalibrates RiskProfilingAgent's trained-model percentile grid.

WHY THIS EXISTS
    scripts/train_risk_model.py builds `percentile_grid` from P(distress)
    scored over GiveMeSomeCredit's OWN training split. That's the wrong
    reference population for this system: GMSC's median customer is
    older (52 vs this system's 42) and higher-income (EUR64,680 vs
    EUR47,528) than the population this agent actually scores, which
    compresses nearly every real customer into the top quartile of
    predicted distress and made two of five risk tiers (moderately_
    aggressive, aggressive) unreachable — see the coverage audit.

    Separately, age carries 48% of the model's feature importance, and
    a SHAP decomposition shows it dominates in a way that's a poor proxy
    for INVESTMENT capacity specifically: a 25-year-old on EUR120,000
    income with zero debt scores almost identically to a 62-year-old on
    EUR22,000 with EUR15,000 debt (p_distress 0.51 vs 0.545), because
    youth's positive contribution to predicted distress roughly cancels
    high income's negative one. GMSC's SeriousDlqin2yrs label reflects
    short-to-medium-term credit delinquency risk, where youth genuinely
    predicts higher default rates independent of current income — but
    that is not the same question as "can this person absorb an
    investment loss," which is what capacity is meant to measure here.

WHAT THIS SCRIPT DOES, AND WHAT IT DOESN'T TOUCH
    Does NOT retrain the model — data/raw/give_me_some_credit isn't
    even present in this repository (see generate_synthetic_customer_
    base.py's own docstring), so retraining isn't an option here, and
    the model's coefficients aren't the problem: its raw P(distress)
    output already contains the information needed, age's SHAP
    contribution is just dominating that output for reasons orthogonal
    to investment capacity.

    Instead: for each customer, decomposes P(distress) via
    shap.TreeExplainer and subtracts age's own contribution, producing
    an "age-neutral" P(distress) — what this person's predicted distress
    would be if age contributed neither more nor less than its average
    effect, with income/debt/dependents' contributions untouched. Then
    builds a NEW percentile grid from that age-neutral score, computed
    over a large synthetic reference population (this system's own
    generator, not GMSC's), so percentile ranking compares each customer
    against a representative reference class instead of a mismatched one.

    Both grids are kept in the bundle. RiskProfilingAgent tries the
    age-neutral path first and only falls back to the original
    (GMSC-calibrated, age-INCLUDED) grid if `shap` isn't importable —
    and always uses the grid that matches whichever P(distress) it
    actually computed, so the two never get mismatched.

VERIFIED IMPACT (RQ1 15-profile hand-labelled fixture)
    RAR unchanged (0.933). Macro F1 0.499 -> 0.636. Per-tier F1:
    aggressive 0.0 -> 0.857, moderately_aggressive 0.444 -> 0.75,
    moderate 0.75 -> 0.571, moderately_conservative 0.8 -> 0.5,
    conservative unchanged at 0.5. Net improvement, not uniform —
    a handful of 35-45-year-olds near the tier boundary shift up one
    tier, since the correction isn't confined to the extreme young-
    high-earner case alone. See the coverage audit and follow-up
    discussion for the full trade-off analysis.

USAGE
    python scripts/recalibrate_risk_model_age_neutral.py
    python scripts/recalibrate_risk_model_age_neutral.py --n 8000 --seed 999983
    python scripts/recalibrate_risk_model_age_neutral.py --no-save   # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MODEL_PATH = settings.risk.model_path


def age_neutral_p_distress(model, explainer, X: np.ndarray, age_idx: int) -> np.ndarray:
    """
    P(distress) with the age feature's own SHAP contribution removed.

    SHAP values are additive around the model's expected_value baseline
    (prediction = base_value + sum(shap_values)), so subtracting one
    feature's contribution is a direct, principled way to ask "what would
    this prediction have been if this feature had contributed nothing
    beyond its average effect" — without retraining or dropping the
    feature (which would be a train/serve skew: age is still an input,
    it just no longer gets to swing the OUTPUT used for capacity).
    """
    p = model.predict_proba(X)[:, 1]
    raw = explainer.shap_values(X)
    arr = np.array(raw[1] if isinstance(raw, list) and len(raw) == 2 else raw)
    vals = arr[..., 1] if arr.ndim == 3 else arr
    shap_age = vals[:, age_idx]
    return np.clip(p - shap_age, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=8000,
                     help="Size of the synthetic reference population.")
    ap.add_argument("--seed", type=int, default=999983,
                     help="Independent of generate_synthetic_customer_base's own "
                          "default seed (500) and train_risk_model's GMSC split "
                          "(42) — a fresh, reproducible batch just for this grid.")
    ap.add_argument("--no-save", action="store_true", help="Report only, do not write the .pkl")
    args = ap.parse_args()

    import joblib
    import shap

    from scripts.generate_synthetic_customer_base import generate

    if not MODEL_PATH.exists():
        raise SystemExit(f"No trained model at {MODEL_PATH} — run train_risk_model.py first.")

    bundle = joblib.load(MODEL_PATH)
    if not (isinstance(bundle, dict) and bundle.get("kind") == "distress_probability"):
        raise SystemExit("Expected a distress_probability bundle from train_risk_model.py.")

    model = bundle["model"]
    names = bundle["feature_names"]
    if "age" not in names:
        raise SystemExit(f"'age' not in feature_names {names} — nothing to neutralise.")
    age_idx = names.index("age")

    print(f"Generating {args.n:,} reference customers "
          f"(scripts.generate_synthetic_customer_base, seed={args.seed})...")
    customers, _ = generate(count=args.n, seed=args.seed)
    X_ref = np.array([[float(c[n]) for n in names] for c in customers])

    explainer = shap.TreeExplainer(model)
    p_ref_raw = model.predict_proba(X_ref)[:, 1]
    p_ref_adj = age_neutral_p_distress(model, explainer, X_ref, age_idx)

    print(f"\n  RAW p_distress on reference pop:        "
          f"median={np.median(p_ref_raw):.4f}  p75={np.percentile(p_ref_raw,75):.4f}  "
          f"max={p_ref_raw.max():.4f}")
    print(f"  AGE-NEUTRAL p_distress on reference pop: "
          f"median={np.median(p_ref_adj):.4f}  p75={np.percentile(p_ref_adj,75):.4f}  "
          f"max={p_ref_adj.max():.4f}  min={p_ref_adj.min():.4f}")

    grid_age_neutral = np.percentile(p_ref_adj, np.arange(0, 101)).tolist()
    grid_own_population_raw = np.percentile(p_ref_raw, np.arange(0, 101)).tolist()

    new_bundle = dict(bundle)
    new_bundle["percentile_grid_age_neutral"] = grid_age_neutral
    # Original GMSC-calibrated grid is kept under its original key, untouched —
    # RiskProfilingAgent's fallback path (shap not importable) still gets a
    # grid that matches the RAW p_distress it would fall back to computing.
    new_bundle["percentile_grid_reference"] = {
        "age_neutral_source": "scripts.generate_synthetic_customer_base.generate",
        "n": args.n,
        "seed": args.seed,
        "own_population_raw_grid_for_reference_only": grid_own_population_raw,
        "original_grid_source": "GiveMeSomeCredit training split (n_train=120000)",
    }
    new_bundle["note"] = (
        bundle["note"] + " capacity_score additionally neutralises age's own SHAP "
        "contribution to P(distress) and ranks the result against a percentile grid "
        "built from this system's own synthetic customer population rather than "
        "GiveMeSomeCredit's — see scripts/recalibrate_risk_model_age_neutral.py. "
        "The original GMSC-calibrated grid is retained as `percentile_grid` for the "
        "fallback path when `shap` is not installed."
    )

    new_bundle["sklearn_version"] = __import__("sklearn").__version__

    if not args.no_save:
        joblib.dump(new_bundle, MODEL_PATH)
        print(f"\n  Saved -> {MODEL_PATH}")
    else:
        print("\n  --no-save: not writing the .pkl")

    return 0


if __name__ == "__main__":
    sys.exit(main())