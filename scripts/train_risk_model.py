"""
Trains the ML half of RiskProfilingAgent's hybrid score.

WHAT THIS TRAINS, AND WHY IT IS NOT A RISK-TIER CLASSIFIER

There are no risk-tier labels in this project's data. scripts/preprocess_
customers.py sets `ground_truth_risk_class` to "" for every row of both source
datasets, deliberately — neither German Credit nor GiveMeSomeCredit records
whether someone is a "conservative" or "aggressive" investor. Those are not
facts a credit dataset contains.

So a supervised 5-class tier classifier cannot be trained. The tempting
shortcut — generate tier labels with the existing _rule_score() and train on
those — must be avoided: the model would simply learn the rule, and
hybrid_vs_rule_only_delta would then measure approximation error rather than
any added value. That would quietly invalidate RQ1.

What this script trains instead is defensible and uses a real, independent
label: a model of FINANCIAL DISTRESS, from GiveMeSomeCredit's
`SeriousDlqin2yrs` (90 days delinquent within two years).

That maps onto the risk framework as CAPACITY, not preference:

    risk CAPACITY   objective; how much loss the balance sheet absorbs  → LEARNED HERE
    risk TOLERANCE  subjective; how much loss the person can stomach    → self-reported
    risk KNOWLEDGE  financial literacy                                  → self-reported

which is exactly the distinction data/psychometric_proxy.py already draws.

The difference is the measurable price of the advisory setting.


USAGE
    python scripts/train_risk_model.py                 # train  save  report
    python scripts/train_risk_model.py --no-save       # report only
    python scripts/train_risk_model.py --model rf      # rf (default) | gb
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ROOT_DIR, settings  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

GMSC_RAW = ROOT_DIR / "data" / "raw" / "give_me_some_credit" / "cs-training.csv"
MODEL_PATH = settings.risk.model_path

# The ONLY features an advisory conversation can realistically obtain. Order
# matters: RiskProfilingAgent rebuilds the vector from this list at inference.
ADVISORY_FEATURES = ["age", "income", "existing_debt", "dependents"]

# Credit-bureau extras — used ONLY to measure the ceiling, never deployed.
BUREAU_EXTRAS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
]


def _num(value: str) -> float:
    """GMSC uses '' and 'NA' for missing. Everything else should parse."""
    if value in (None, "", "NA"):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_gmsc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X_advisory, X_full, y).

    Feature derivation matches scripts/preprocess_customers.py exactly:
        income        = MonthlyIncome * 12
        existing_debt = DebtRatio * income
    """
    if not GMSC_RAW.exists():
        raise SystemExit(
            f"Missing {GMSC_RAW}\n"
            "Run: python scripts/download_datasets.py --phase 3"
        )

    X_adv, X_full, y = [], [], []

    with open(GMSC_RAW, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            age = _num(row["age"])
            if not np.isfinite(age):
                continue

            monthly = _num(row["MonthlyIncome"])
            income = monthly * 12 if np.isfinite(monthly) else float("nan")

            debt_ratio = _num(row["DebtRatio"])
            debt = debt_ratio * income if np.isfinite(income) else float("nan")

            deps = _num(row["NumberOfDependents"])
            deps = 0.0 if not np.isfinite(deps) else deps

            adv = [age, income, debt, deps]

            X_adv.append(adv)
            X_full.append(adv + [_num(row[k]) for k in BUREAU_EXTRAS])
            y.append(int(row["SeriousDlqin2yrs"]))

    return np.array(X_adv), np.array(X_full), np.array(y)


def build_model(kind: str):
    """
    RandomForest by default, specifically because shap.TreeExplainer supports
    it well — RiskProfilingAgent._compute_shap_proxy() upgrades from proxy
    attributions to real SHAP values when a tree model is present, and that
    upgrade is the point of training a model at all for RQ3's Layer A.
    """
    if kind == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,            # shallow: 4 features, and depth invites overfit
        min_samples_leaf=50,    # smooths probabilities — matters, we USE them
        class_weight="balanced",  # 6.7% positives
        random_state=42,
        n_jobs=-1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-save", action="store_true", help="Report only, do not write the .pkl")
    ap.add_argument("--model", choices=["rf", "gb"], default="rf")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    print("Loading GiveMeSomeCredit [D8]...")
    X_adv, X_full, y = load_gmsc()
    print(f"  {len(y):,} rows, {y.mean():.1%} positive (90 days delinquent)\n")

    Xtr, Xte, ytr, yte = train_test_split(
        X_adv, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Median imputation. Medians are stored in the bundle so inference imputes
    # identically — a different fill value at inference is a silent train/serve
    # skew, and this model's whole job is to output a calibrated probability.
    medians = np.nanmedian(Xtr, axis=0)
    def impute(X):
        X = X.copy()
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(medians, idx[1])
        return X

    Xtr_i, Xte_i = impute(Xtr), impute(Xte)

    print(f"Training {args.model} on {ADVISORY_FEATURES}...")
    model = build_model(args.model).fit(Xtr_i, ytr)
    p_te = model.predict_proba(Xte_i)[:, 1]
    auc_adv = roc_auc_score(yte, p_te)
    ap_adv = average_precision_score(yte, p_te)

    # The ceiling: same model, plus features no chatbot can ask for.
    Xf_tr, Xf_te, _, _ = train_test_split(
        X_full, y, test_size=args.test_size, stratify=y, random_state=42
    )
    med_f = np.nanmedian(Xf_tr, axis=0)
    def imp_f(X):
        X = X.copy()
        idx = np.where(~np.isfinite(X))
        X[idx] = np.take(med_f, idx[1])
        return X
    ceiling = build_model(args.model).fit(imp_f(Xf_tr), ytr)
    auc_ceiling = roc_auc_score(yte, ceiling.predict_proba(imp_f(Xf_te))[:, 1])

    # Percentile grid over TRAINING probabilities. RiskProfilingAgent maps an
    # individual's P(distress) to its percentile in this distribution, which
    # spreads a 6.7%-base-rate probability across a usable [0,1] capacity
    # score. Without it every score would bunch near 1.0 and the tier
    # thresholds would never separate anyone.
    p_train = model.predict_proba(Xtr_i)[:, 1]
    percentile_grid = np.percentile(p_train, np.arange(0, 101)).tolist()

    print(f"\n  AUC  (advisory features: {', '.join(ADVISORY_FEATURES)})  = {auc_adv:.4f}")
    print(f"  AP   (advisory features)                                = {ap_adv:.4f}")
    print(f"  AUC  ( credit-bureau features, NOT obtainable)         = {auc_ceiling:.4f}")
    print(f"  Information gap                                          = {auc_ceiling - auc_adv:.4f}")

    importances = dict(zip(ADVISORY_FEATURES,
                           [round(float(v), 4) for v in model.feature_importances_]))
    print(f"\n  Feature importances: {importances}")

    bundle = {
        "kind": "distress_probability",
        "model": model,
        "feature_names": list(ADVISORY_FEATURES),
        "imputation_medians": [float(m) for m in medians],
        "percentile_grid": percentile_grid,
        "metrics": {
            "auc_advisory_features": round(float(auc_adv), 4),
            "average_precision": round(float(ap_adv), 4),
            "auc_all_features_ceiling": round(float(auc_ceiling), 4),
            "information_gap": round(float(auc_ceiling - auc_adv), 4),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "positive_rate": round(float(y.mean()), 4),
        },
        "feature_importances": importances,
        "training_data": "GiveMeSomeCredit [D8] SeriousDlqin2yrs",
        "sklearn_estimator": type(model).__name__,
        "note": (
            "Predicts P(financial distress), NOT a risk tier — no tier labels "
            "exist in this project's data. RiskProfilingAgent converts this to "
            "a capacity score and blends it with self-reported preference "
            "features (loss_tolerance, investment_horizon, "
            "financial_knowledge_score), which have no training data and remain "
            "coefficient-based."
        ),
    }

    if not args.no_save:
        import joblib
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        print(f"\n  Saved -> {MODEL_PATH}")

        try:
            from evaluation.results_io import write_results
            out = write_results(
                {"phase": 3, "component": "risk_model_training", **{
                    k: bundle[k] for k in
                    ("metrics", "feature_importances", "feature_names",
                     "training_data", "sklearn_estimator", "kind", "note")}},
                "rq1_risk_model_training.json",
            )
            print(f"  Metrics -> {out}")
        except Exception as exc:
            logger.warning(f"Could not write results file: {exc}")

    print("\nNext: python -m pytest tests/unit/test_risk_profiling_agent.py -v -s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
