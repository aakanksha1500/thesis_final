"""
Classifies a customer into one of five risk tiers (conservative through
aggressive) by blending a trained ML model with deterministic rules.

THE BLEND
    hybrid_score = ml_weight * ml_score + rule_weight * rule_score  (0.6 / 0.4)
    ml_score itself blends a trained distress-model capacity score with a
    self-reported preference score (loss tolerance, horizon, financial
    knowledge). See _hybrid_score(), _capacity_score(), _rule_score().

AGE-NEUTRAL CAPACITY CORRECTION
    Age carries ~48% of the underlying model's feature importance — a
    valid signal for its original label (credit delinquency) but a poor
    proxy for investment-loss capacity. _capacity_score() subtracts age's
    own SHAP contribution from P(distress) before scoring it against a
    separately-fit percentile grid. See scripts/recalibrate_risk_model_
    age_neutral.py for why. Falls back to the raw, age-inclusive score if
    SHAP or the age-neutral grid isn't available.

TIER BOUNDARIES
    _score_to_class() and _compute_confidence() support both fixed
    equal-width boundaries (default) and a fitted quantile scheme
    (USE_QUANTILE_TIER_BOUNDARIES=true) — see _load_tier_boundaries().
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from agents.base_agent import AgentResult, BaseAgent
from config.prompts import RISK_PROFILING_SYSTEM
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class RiskProfilingAgent(BaseAgent):
    """
    Context-Aware Hybrid risk classifier.

    Scoring architecture:
        ml_score - trained sklearn model (Random Forest / XGBoost)
                    Falls back to calibrated heuristic if no model file found.
        rule_score - deterministic CBI-grounded rules. Always runs regardless
                    of model availability (Nguyen et al)
        hybrid - weighted combination per settings.risk weights.

    SHAP proxy output feeds ExplainabilityAgent Layer A
    """

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="RiskProfilingAgent")
        self._sklearn_version_mismatch: dict | None = None
        self._ml_model = self._load_ml_model()
        self._confidence_calibrator = self._load_confidence_calibrator()
        self._tier_boundaries = self._load_tier_boundaries()

    @property
    def system_prompt(self) -> str:
        return RISK_PROFILING_SYSTEM

    def _parse_response(self, raw: str) -> dict[str, Any]:
        """
        RiskProfilingAgent LLM output is a plain-English rationale paragraph.
        No JSON parsing needed
        """
        return {"rationale": raw.strip()}

    # ML model loading

    def _load_ml_model(self):
        """
        Load trained sklearn model from disk if available.
        Returns None if not found - heuristic fallback activates automatically.
        """
        model_path = settings.risk.model_path
        if model_path.exists() and not settings.risk.use_trained_model:
            logger.info(
                f"[RiskProfilingAgent] A trained model exists at {model_path} but "
                f"settings.risk.use_trained_model is False — using the heuristic. "
                f"Set USE_TRAINED_RISK_MODEL=true to enable it (this changes every "
                f"RQ1 number, so it is deliberately opt-in)."
            )
            return None
        if model_path.exists():
            try:
                import joblib
                model = joblib.load(model_path)
                if isinstance(model, dict) and model.get("kind") == "distress_probability":
                    m = model.get("metrics", {})
                    logger.info(
                        f"[RiskProfilingAgent] Loaded trained DISTRESS model from "
                        f"{model_path} — {model.get('sklearn_estimator')} on "
                        f"{model.get('feature_names')}, "
                        f"AUC={m.get('auc_advisory_features')} "
                        f"(bureau ceiling {m.get('auc_all_features_ceiling')}). "
                        f"Preference features remain coefficient-based."
                    )
                else:
                    if isinstance(model, dict) and model.get("kind") == "distress_probability":
                        m = model.get("metrics", {})
                        logger.info(
                            f"[RiskProfilingAgent] Loaded trained DISTRESS model from "
                            f"{model_path} — {model.get('sklearn_estimator')} on "
                            f"{model.get('feature_names')}, "
                            f"AUC={m.get('auc_advisory_features')} "
                            f"(bureau ceiling {m.get('auc_all_features_ceiling')}). "
                            f"Preference features remain coefficient-based."
                        )
                    else:
                        logger.info(
                            f"[RiskProfileAgent] loaded trained ML model from {model_path}"
                        )
                self._sklearn_version_mismatch = None
                runtime_version = __import__("sklearn").__version__
                pickled_version = (
                    model.get("sklearn_version") if isinstance(model, dict) else None
                )
                if pickled_version is not None and pickled_version != runtime_version:
                    self._sklearn_version_mismatch = {
                        "pickled_under": pickled_version,
                        "running_under": runtime_version,
                    }
                    logger.warning(
                        f"[RiskProfilingAgent] {model_path} was pickled under "
                        f"scikit-learn {pickled_version} but this process is "
                        f"running {runtime_version}. RQ1 numbers from this run "
                        f"are not guaranteed to reproduce byte-for-byte "
                        f"elsewhere. Regenerate before final reporting: "
                        f"python scripts/train_risk_model.py (then "
                        f"scripts/recalibrate_risk_model_age_neutral.py if "
                        f"that step is part of your pipeline — check for "
                        f"percentile_grid_age_neutral in the bundle to tell "
                        f"whether it already was). See README 'Known "
                        f"limitations'."
                    )
                elif pickled_version is None and isinstance(model, dict):
                    logger.warning(
                        f"[RiskProfilingAgent] {model_path} has no recorded "
                        f"sklearn_version (pickled before this field existed) "
                        f"— cannot check for a version mismatch. Regenerating "
                        f"with the current scripts/train_risk_model.py will "
                        f"add it."
                    )
                return model
            except Exception as exc:
                logger.warning(
                    f"[RiskProfilingAgent] Could not load ML model: {exc} "
                    f"- falling back to heuristic"
                )
        else:
            logger.info(
                "[RiskProfilingAgent] No trained model found at "
                f"{model_path} - using heuristic proxy. "
                "Train the model with scripts/train_risk_model.py"
            )
        return None

    def _capacity_score(self, features: dict[str, Any], bundle: dict) -> float:
        """
        Convert the trained model's P(financial distress) into a capacity score
        in [0, 1], where 1 = greatest capacity to absorb loss.
        """
        import numpy as np

        names = bundle["feature_names"]
        medians = bundle["imputation_medians"]

        row = []
        for name, median in zip(names, medians):
            raw = features.get(name)
            try:
                value = float(raw)
                if not np.isfinite(value):
                    value = float(median)
            except (TypeError, ValueError):
                value = float(median)
            row.append(value)

        X = np.array([row])
        p_distress = float(bundle["model"].predict_proba(X)[0][1])

        p_used = p_distress
        grid = bundle["percentile_grid"]
        mode = "raw"

        age_neutral_grid = bundle.get("percentile_grid_age_neutral")
        if age_neutral_grid is not None and "age" in names:
            try:
                import shap

                explainer = shap.TreeExplainer(bundle["model"])
                raw_shap = explainer.shap_values(X)
                arr = np.array(
                    raw_shap[1]
                    if isinstance(raw_shap, list) and len(raw_shap) == 2
                    else raw_shap
                )
                vals = arr[0][..., 1] if arr.ndim == 3 else arr[0]
                vals = np.asarray(vals, dtype=float).ravel()[: len(names)]
                shap_age = float(vals[names.index("age")])

                p_used = float(np.clip(p_distress - shap_age, 0.0, 1.0))
                grid = age_neutral_grid
                mode = "age_neutral"

            except ImportError:
                logger.warning(
                    "[RiskProfilingAgent] shap not installed — capacity falls "
                    "back to raw (age-inclusive) P(distress) against the "
                    "original GMSC-calibrated grid. pip install shap"
                )
            except Exception as exc:
                logger.warning(
                    f"[RiskProfilingAgent] Age-neutral capacity computation "
                    f"failed: {exc} — falling back to raw P(distress)"
                )

        percentile = float(np.searchsorted(grid, p_used)) / 100.0
        capacity = 1.0 - min(max(percentile, 0.0), 1.0)

        logger.debug(
            f"[RiskProfilingAgent] mode={mode} P(distress)={p_distress:.4f} "
            f"P(used)={p_used:.4f} → percentile={percentile:.2f} "
            f"→ capacity={capacity:.4f}"
        )

        return capacity

    def _preference_score(self, features: dict[str, Any]) -> float:
        """
        Score self-reported preference features.
        """
        score = 0.5
        score += (float(features.get("loss_tolerance", 3)) - 3) * 0.08
        score += (float(features.get("investment_horizon", 5)) - 5) * 0.02
        score += (
            float(features.get("financial_knowledge_score", 3)) - 3
        ) * 0.04

        return float(np.clip(score, 0.0, 1.0))
    # Scoring Components

    def _ml_score(self, features: dict[str, Any]) -> float:
        """
        Return a risk appetite score in [0, 1].
        0.0 = most conservative, 1.0 = most aggressive.
        """
        if (
            isinstance(self._ml_model, dict)
            and self._ml_model.get("kind") == "distress_probability"
        ):
            try:
                capacity = self._capacity_score(features, self._ml_model)
                preference = self._preference_score(features)

                w = settings.risk.capacity_weight

                score = w * capacity + (1.0 - w) * preference

                logger.debug(
                    f"[RiskProfilingAgent] ML score {score:.4f} = "
                    f"{w:.2f}*capacity({capacity:.4f}) + "
                    f"{1 - w:.2f}*preference({preference:.4f})"
                )

                return float(np.clip(score, 0.0, 1.0))

            except Exception as exc:
                logger.warning(
                    f"[RiskProfilingAgent] Trained-model scoring failed: {exc} "
                    f"— falling back to heuristic for this call"
                )
        if self._ml_model is not None and not isinstance(self._ml_model, dict):
            try:
                feature_vector = np.array([[
                    float(features.get(f, 0))
                    for f in settings.risk.required_features
                ]])
                proba = self._ml_model.predict_proba(feature_vector)[0]
                # Weight each class by its position in the risk spectrum
                class_weights = np.linspace(0.0, 1.0, len(proba))
                score = float(np.dot(proba, class_weights))
                logger.debug(
                    f"[RiskProfilingAgent] ML model score: {score:.4f} "
                    f"(probabilities: {proba.round(3)})"
                )
                return score
            except Exception as exc:
                logger.warning(
                    f"[RiskProfilingAgent] ML inference error: {exc} "
                    f"- falling back to heuristic for this call"
                )

        score = 0.5 # neutral starting point

        age = float(features.get("age", 40))
        horizon = float(features.get("investment_horizon", 5))
        tolerance = float(features.get("loss_tolerance", 3))
        knowledge = float(features.get("financial_knowledge_score", 3))
        income = max(float(features.get("income", 50000)), 1.0)
        debt = float(features.get("existing_debt", 0))
        debt_ratio = debt / income

        score += (tolerance -3) * 0.08 # dominant factor
        score += (horizon -5) * 0.02 # each year beyond 5 adds capacity
        score += (knowledge - 3) * 0.04 #financial literacy effect
        score -= (age - 35) * 0.005 # lifecycle age penalty
        score -= debt_ratio * 0.20 # debt burden penalty

        return float(np.clip(score, 0.0, 1.0))

    def _rule_score(self, features: dict[str, Any]) -> float:
        """
        Deterministic rule-based score grounded in CBI model risk guidelines.
        Returns [0, 1] risk score.

        Rules:
            R1: High debt-to-income (>50%) -> conservative floor
            R2: Unemployed / retired -> conservative constraint
            R3: Many dependents -> reduce capacity (each dependent -5%)
            R4: Low income (<€25k) -> conservative bias
            R5: Low debt and stable employment -> slight increase
        """

        score = 0.5

        employment = features.get("employment_status", "employed")
        dependents = int(features.get("dependents", 0))
        income = float(features.get("income", 50000))
        debt = float(features.get("existing_debt", 0))
        debt_to_income = debt / max(income, 1.0)

        # R1: Debt burden
        if debt_to_income > 0.5:
            score -= 0.20
            logger.debug(
                "[RiskProfilingAgent] Rule R1: high DTI -> -0.20"
            )
        elif debt_to_income < 0.1:
            score += 0.05
            logger.debug("[RiskProfilingAgent] Rule R1: low DTI -> +0.05")

        # R2: Employment status
        if employment in ("unemployed", "retired"):
            score -= 0.20
            logger.debug(
                f"[RiskProfilingAgent] Rule R2: {employment} -> -0.20"
            )
        elif employment in ("self_employed", "self-employed"):
            score -= 0.05
            logger.debug(f"[RiskProfilingAgent] Rule R2: {employment} -> -0.05")

        # R3: Dependents
        dependent_penalty = min(dependents * 0.05, 0.20)
        score -= dependent_penalty
        if dependent_penalty > 0:
            logger.debug(
                f"[RiskProfilingAgent] Rule R3: {dependents} dependents -> "
                f"-{dependent_penalty:.2f}"
            )

        # R4: Low income
        if income < 25000:
            score -= 0.10
            logger.debug("[RiskProfilingAgent] Rule R4: low income -> -0.10")
        if debt_to_income < 0.1 and employment == "employed":
            score += 0.05
            logger.debug(
                "[RiskProfilingAgent] Rule R5: low DTI + stable employment "
                "-> +0.05"
            )


        return float(np.clip(score, 0.0, 1.0))

    def _hybrid_score(self, features: dict[str, Any]) -> tuple[float, float, float]:
        """
        Combine ML and rule scores using configured weights.
        Returns (ml_score, rule_score, hybrid_score).
        """
        ml = self._ml_score(features)
        rule = self._rule_score(features)
        hybrid = (
            settings.risk.ml_weight * ml +
            settings.risk.rule_weight * rule
        )
        return ml, rule, float(np.clip(hybrid, 0.0, 1.0))

    def _score_to_class(self, score: float) -> str:
        """
        Map continuous [0, 1] hybrid score to 5-class risk tier.

        Uses settings.risk.use_quantile_tier_boundaries to choose between
        two threshold sources — see _load_tier_boundaries for why the
        equal-width default starves the outer classes.
        """
        classes = settings.risk.risk_classes
        thresholds = self._tier_boundaries or [0.2, 0.4, 0.6, 0.8]
        for i, threshold in enumerate(thresholds):
            if score < threshold:
                return classes[i]
        return classes[-1]

    def _compute_confidence(self, hybrid_score: float) -> float:
        """
        Confidence estimate based on distance from the nearest class
        boundary, normalised by that bin's own half-width so a wide bin
        (common under quantile boundaries, where bin widths are no
        longer equal) doesn't get compressed toward 0 or 1 relative to a
        narrow one. Falls back to the original fixed 0.1 half-width when
        using the equal-width [0.2, 0.4, 0.6, 0.8] thresholds, so this is
        unchanged from before unless use_quantile_tier_boundaries is on.

        This formula's deeper limitation — margin-from-boundary is not
        the same thing as a learned probability of correctness — is not
        fixed by either boundary scheme. See _load_confidence_calibrator
        and Section VII-D of the dissertation.
        """
        thresholds = self._tier_boundaries or [0.2, 0.4, 0.6, 0.8]
        boundaries = [0.0] + list(thresholds) + [1.0]

        bin_idx = 0
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= hybrid_score <= boundaries[i + 1]:
                bin_idx = i
                break
        lo, hi = boundaries[bin_idx], boundaries[bin_idx + 1]
        half_width = (hi - lo) / 2 or 1e-9
        dist_to_boundary = min(hybrid_score - lo, hi - hybrid_score)
        confidence = min(dist_to_boundary / half_width, 1.0)
        return round(float(confidence), 4)

    def _load_tier_boundaries(self) -> list[float] | None:
        """
        Load quantile-fitted tier boundaries from disk if enabled and
        available (scripts/calibrate_risk_tier_boundaries.py produces
        this file, fit on the unlabelled stress set — never the gold set
        used to evaluate it).

        Returns None if disabled, or if the file doesn't exist yet — the
        fixed [0.2, 0.4, 0.6, 0.8] thresholds are used as-is in either
        case, with a warning so a run that silently fell back doesn't
        look identical to one that deliberately chose to. Mirrors
        _load_confidence_calibrator's opt-in pattern exactly.
        """
        if not settings.risk.use_quantile_tier_boundaries:
            return None
        path = settings.risk.tier_boundaries_path
        if not path.exists():
            logger.warning(
                f"[RiskProfilingAgent] use_quantile_tier_boundaries=True but "
                f"no boundaries file at {path} — falling back to fixed "
                f"[0.2, 0.4, 0.6, 0.8]. Run "
                f"scripts/calibrate_risk_tier_boundaries.py to produce it."
            )
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            thresholds = data["thresholds"]
            logger.info(
                f"[RiskProfilingAgent] Loaded quantile tier boundaries: "
                f"{thresholds} (fit on {data['fit_on']['n']} stress profiles)"
            )
            return thresholds
        except Exception as exc:
            logger.warning(
                f"[RiskProfilingAgent] Failed to load tier boundaries from "
                f"{path}: {exc} — falling back to fixed [0.2, 0.4, 0.6, 0.8]."
            )
            return None

    def _load_confidence_calibrator(self):
        """
        Load the isotonic-regression confidence calibrator from disk if
        available and settings.risk.use_calibrated_confidence is true.

        Returns None if disabled, or if the file doesn't exist yet (run
        scripts/calibrate_risk_confidence.py to produce it) — the raw
        margin from _compute_confidence is used as-is in either case,
        with a warning so a run that silently fell back doesn't look
        identical to one that deliberately chose to.
        """
        if not settings.risk.use_calibrated_confidence:
            return None
        path = settings.risk.confidence_calibrator_path
        if not path.exists():
            logger.warning(
                f"[RiskProfilingAgent] settings.risk.use_calibrated_confidence "
                f"is true but no calibrator found at {path} — falling back to "
                f"raw decision-margin confidence (known inversely calibrated, "
                f"see _compute_confidence). Run "
                f"scripts/calibrate_risk_confidence.py first."
            )
            return None
        try:
            import pickle
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            logger.info(
                f"[RiskProfilingAgent] Loaded confidence calibrator from "
                f"{path} — fit on {bundle.get('fit_n')} gold profiles, "
                f"CV AUROC raw={bundle.get('cv_auroc_raw')} -> "
                f"calibrated={bundle.get('cv_auroc_calibrated')}."
            )
            return bundle["calibrator"]
        except Exception as exc:
            logger.warning(
                f"[RiskProfilingAgent] Failed to load confidence calibrator "
                f"from {path}: {exc}. Falling back to raw confidence."
            )
            return None

    def _apply_confidence_calibration(self, raw_confidence: float) -> float:
        """
        Map raw decision-margin confidence through the fitted calibrator,
        if one is loaded — otherwise return raw_confidence unchanged.
        Kept as a separate step (not folded into _compute_confidence) so
        the raw value is always still computable/loggable on its own.
        """
        if self._confidence_calibrator is None:
            return raw_confidence
        calibrated = self._confidence_calibrator.predict([raw_confidence])[0]
        return round(float(calibrated), 4)

    def _compute_shap_proxy(
            self,
            features: dict[str, Any],
            ml_score: float,
    ) -> dict[str, dict]:
        """
        SHAP feature attribution proxy for ExplainabilityAgent Layer A.

        With a trained sklearn model, this would call shap.TreeExplainer.
        Without a model, produces calibrated proxy attribution based on the
        known directional relationships from German Credit [D7].

        Output format consumed by ExplainabilityAgent (Phase 6):
            {feature_name: {"value": user_value, "shap_impact": float}}

        +ve shap_impact -> pushes toward aggressive.
        -ve shap_impact -> pushes toward conservative.
        """
        if (
            isinstance(self._ml_model, dict)
            and self._ml_model.get("kind") == "distress_probability"
        ):
            try:
                import shap

                bundle = self._ml_model
                names = bundle["feature_names"]
                medians = bundle["imputation_medians"]

                row = []

                for name, median in zip(names, medians):
                    try:
                        value = float(features.get(name))
                        if not np.isfinite(value):
                            value = float(median)
                    except (TypeError, ValueError):
                        value = float(median)

                    row.append(value)

                explainer = shap.TreeExplainer(bundle["model"])
                raw = explainer.shap_values(np.array([row]))

                arr = np.array(
                    raw[1]
                    if isinstance(raw, list) and len(raw) == 2
                    else raw
                )

                vals = (
                    arr[0][..., 1]
                    if arr.ndim == 3
                    else arr[0]
                )

                vals = np.asarray(vals, dtype=float).ravel()[: len(names)]

                age_neutralised = (
                    "age" in names
                    and bundle.get("percentile_grid_age_neutral") is not None
                )

                attributions = {}

                for name, value, shap_value in zip(names, row, vals):
                    impact = 0.0 if (name == "age" and age_neutralised) else float(-shap_value)
                    attributions[name] = {
                        "value": features.get(name, value),
                        "shap_impact": round(impact, 4),
                        "source": "shap",
                    }

                for name, impact in (
                    (
                        "loss_tolerance",
                        (float(features.get("loss_tolerance", 3)) - 3) * 0.08,
                    ),
                    (
                        "investment_horizon",
                        (float(features.get("investment_horizon", 5)) - 5) * 0.02,
                    ),
                    (
                        "financial_knowledge_score",
                        (
                            float(features.get("financial_knowledge_score", 3)) - 3
                        )
                        * 0.04,
                    ),
                ):
                    if name not in attributions:
                        attributions[name] = {
                            "value": features.get(name, "N/A"),
                            "shap_impact": round(impact, 4),
                            "source": "proxy",
                        }

                logger.debug(
                    f"[RiskProfilingAgent] SHAP: {len(names)} learned + "
                    f"{len(attributions) - len(names)} proxy attributions"
                )

                return attributions

            except ImportError:
                logger.warning(
                    "[RiskProfilingAgent] shap not installed — a trained model is "
                    "present but explanations fall back to proxy attributions. "
                    "pip install shap"
                )

            except Exception as exc:
                logger.warning(
                    f"[RiskProfilingAgent] SHAP computation failed: {exc} "
                    f"— using proxy"
                )
        if self._ml_model is not None and not isinstance(self._ml_model, dict):
            try:
                import shap
                explainer = shap.TreeExplainer(self._ml_model)
                feature_vector = np.array([[
                    float(features.get(f, 0))
                    for f in settings.risk.required_features
                ]])
                shap_values = explainer.shap_values(feature_vector)
                # For multi-class: use the class with highest probability
                if isinstance(shap_values, list):
                    shap_vals = shap_values[int(ml_score * 4)]
                else:
                    shap_vals = shap_values[0]
                return {
                    feat: {
                        "value": features.get(feat, "N/A"),
                        "shap_impact": round(float(val), 4),
                        "source": "proxy",
                    }
                    for feat, val in zip(settings.risk.required_features, shap_vals[0])
                }
            except Exception as exc:
                logger.debug(
                    f"[RiskProfilingAgent] SHAP computation failed: {exc} "
                    f"— using proxy"
                )

        # Proxy attributions (heuristic fallback)
        income = max(float(features.get("income", 50000)), 1.0)
        attributions = {
            "loss_tolerance": (float(features.get("loss_tolerance", 3)) - 3) * 0.08,
            "investment_horizon": (float(features.get("investment_horizon", 5)) - 5) * 0.02,
            "financial_knowledge_score": (float(features.get("financial_knowledge_score", 3)) - 3) * 0.04,
            "age": -(float(features.get("age", 40)) - 35) * 0.005,
            "existing_debt": -(float(features.get("existing_debt", 0)) / income) * 0.20,
        }
        return {
            feat: {
                "value": features.get(feat, "N/A"),
                "shap_impact": round(val, 4),
                "source": "proxy",
            }
            for feat, val in attributions.items()
        }

    def _check_missing_features(self, features: dict) -> list[str]:
        """Return list of required features not yet present in the feature dict."""
        return [
            f for f in settings.risk.required_features
            if f not in features or features[f] is None
        ]


    # Main entry point

    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Process one risk profiling request.

        context keys used:
          'user_features'         (dict, required) — feature values
          'conversation_history'  (list, optional) — for LLM rationale context

        Returns AgentResult with payload:
          status, risk_class, ml_score, rule_score, hybrid_score,
          confidence, feature_importance (SHAP), rationale, missing_features
        """
        start_time = time.perf_counter()

        features: dict = context.get("user_features", {})

        # Check for missing required features before attempting classification
        missing = self._check_missing_features(features)
        if missing:
            payload = {
                "status": "incomplete",
                "missing_features": missing,
                "message": (
                    "I don't have enough information yet to assess your risk "
                    "profile — I still need a few details about your "
                    "situation before I can give you an accurate answer."
                ),
            }
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                f"[RiskProfilingAgent] Missing features: {missing}"
            )
            return self._make_result(
                payload=payload,
                duration_ms=duration_ms,
                error=f"Missing features: {missing}",
            )

        # Hybrid scoring
        ml_score, rule_score, hybrid = self._hybrid_score(features)
        risk_class = self._score_to_class(hybrid)
        raw_confidence = self._compute_confidence(hybrid)
        confidence = self._apply_confidence_calibration(raw_confidence)
        shap_proxy = self._compute_shap_proxy(features, ml_score)

        # Flag low confidence explicitly (X3 — Takayanagi et al. [7])
        confidence_flag = (
            "LOW_CONFIDENCE — classification near tier boundary. "
            "Treat as indicative only."
            if confidence < settings.risk.min_confidence
            else None
        )

        # LLM generates the plain-English rationale (Artusi et al. [10])
        rationale_prompt = (
            f"The user has been classified as a '{risk_class}' investor "
            f"with a hybrid score of {hybrid:.3f} and confidence {confidence:.2f}.\n\n"
            f"Key feature contributions:\n"
            + "\n".join(
                f"  {feat}: value={info['value']}, "
                f"impact={'+' if info['shap_impact'] > 0 else ''}{info['shap_impact']}"
                for feat, info in shap_proxy.items()
            )
            + f"\n\nUser features: {json.dumps(features, indent=2)}\n\n"
            f"Write a plain-English rationale (max 80 words) for a retail investor. "
            f"Explain what drove the classification. End with one sentence about confidence."
        )

        try:
            raw_rationale, tokens = self._call_llm(rationale_prompt)
            rationale = raw_rationale.strip()
        except Exception as exc:
            logger.warning(
                f"[RiskProfilingAgent] Rationale generation failed: {exc} "
                f"— using fallback"
            )
            rationale = (
                f"Based on your financial profile, you have been classified as a "
                f"'{risk_class}' investor. Key factors include your stated loss "
                f"tolerance, investment horizon, and current debt level. "
                f"Confidence: {confidence:.0%}."
            )
            tokens = 0

        payload = {
            "status": "complete",
            "risk_class": risk_class,
            "ml_score": round(ml_score, 4),
            "rule_score": round(rule_score, 4),
            "hybrid_score": round(hybrid, 4),
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "confidence_calibrated": self._confidence_calibrator is not None,
            "ml_model_sklearn_version_mismatch": self._sklearn_version_mismatch,
            "ml_score_basis": (
                {
                    "trained_on": self._ml_model.get("training_data"),
                    "sklearn_estimator": self._ml_model.get("sklearn_estimator"),
                    "note": self._ml_model.get("note"),
                }
                if isinstance(self._ml_model, dict)
                else {
                    "trained_on": None,
                    "sklearn_estimator": None,
                    "note": (
                        "No trained model loaded — ml_score is the "
                        "coefficient-based heuristic fallback "
                        "(_ml_score's heuristic branch), not a "
                        "model prediction of any kind."
                    ),
                }
            ),
            "confidence_flag": confidence_flag,
            "feature_importance": shap_proxy,
            "rationale": rationale,
            "missing_features": [],
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[RiskProfilingAgent] risk_class={risk_class} "
            f"hybrid={hybrid:.3f} confidence={confidence:.3f} "
            + (f"(raw={raw_confidence:.3f}, calibrated) "
               if self._confidence_calibrator is not None else "")
            + f"duration={duration_ms:.0f}ms"
        )


        return self._make_result(
            payload=payload,
            raw=rationale,
            duration_ms=duration_ms,
            tokens=tokens,
            routing_context={
                "risk_class": risk_class,
                "confidence": confidence,
                "hybrid_score": round(hybrid, 4),
            },
        )