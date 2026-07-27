"""
Phase 3 - RiskProfilingAgent.

Implements the Context-Aware Hybrid scoring model described in the Literature Review:
    hybrid_score = 0.6 * ml_score + 0.4 * rule_score

Literature grounding:
  - Nguyen et al. [8]: rule-based constraints as a secondary deterministic
    layer alongside ML — the 0.4 rule_weight implements this directly.
  - Takayanagi et al. [7]: confidence must be stated explicitly (X3 goal).
    min_confidence threshold in RiskConfig enforces this.
  - Klesel & Wittmann [6]: SHAP attributions are an independent XAI layer.
    _compute_shap_proxy() produces the feature importance dict that
    ExplainabilityAgent (Phase 6) will consume via Layer A (X2a).

Datasets used for ML training (Phase 3 training run):
  [D7] German Credit Data — credit risk labels + behavioural features
  [D8] GiveMeSomeCredit — probability of financial distress features
  [D9] Bank Marketing — supplementary employment/contact features   
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
        self._ml_model = self._load_ml_model()

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
        Model is trained in Phase 3 training script on German Credit [D7]
        and GiveMeSomeCredit. [D8] datasets.
        """
        model_path = settings.risk.model_path
        if model_path.exists():
            try:
                import joblib
                model = joblib.load(model_path)
                logger.info(
                    f"[RiskProfileAgent] loaded trained ML model from {model_path}"
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

    # Scoring Components

    def _ml_score(self, features: dict[str, Any]) -> float:
        """
        Return a risk appetite score in [0, 1].
        0.0 = most conservative, 1.0 = most aggressive.
        
        With trained model: uses predict_proba() weighted by class index.
        Without model: calibratedheuristic proxy based on feature relationships
        established in German Credit [D7] literature.
        
        The heuristic is not a replacement for the trained model - it is a 
        development scaffold that keeps the pipeline runnable before training.
        RQ1 evaluation uses the trained model scores.
        """
        if self._ml_model is not None:
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
        
        This component always runs - it is the regulatory floor layer
        described in Nguyen et al.: multi-agent rule-based constraints
        reduce hallucination rates by preventing ML from recommending
        products that violate hard suitability rules.
        
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
        elif employment == "self_employed":
            score -= 0.05
            logger.debug("[RiskProfilingAgent] Rule R2: self_employed -> -0.05")

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
        Thresholds divide [0, 1] into 5 equal bands of width 0.2
        """
        classes = settings.risk.risk_classes
        thresholds = [0.2, 0.4, 0.6, 0.8]
        for i, threshold in enumerate(thresholds):
            if score < threshold:
                return classes[i]
        return classes[-1]

    def _compute_confidence(self, hybrid_score: float) -> float:
        """
        Confidence estimate based on distance from the nearest class boundary.
        Score near a boundary (0.2, 0.4, 0.6, 0.8) -> lower confidence.
        Score near a class centre (0.1, 0.3, 0.5, 0.7, 0.9) -> higher confidence.
        
        This directly implements the X3 trust calibration goal from
        Takayanagi et l. [7] and Liao et al. [3]: confidence must reflect
        genuine uncertainty, not always  return a high number.
        """

        boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        # Find distance to nearest boundary
        distances = [abs(hybrid_score -b) for b in boundaries]
        min_distance = min(distances)
        # Normalise: max possible distance from boundary is 0.1 (class midpoint)
        confidence = min(min_distance / 0.1, 1.0)
        return round(float(confidence), 4)

    def _compute_shap_proxy(
            self,
            features: dict[str, Any],
            ml_score: float,
    ) -> dict[str, dict]:
        """
        SHAP feature attribution proxy for ExplainabilityAgent Layer A (X2).
        
        With a trained sklearn model, this would call shap.TreeExplainer.
        Without a model, produces calibrated proxy attribution based on the
        known directional relationships from German Credit [D7].
        
        Output format consumed by ExplainabilityAgent (Phase 6):
            {feature_name: {"value": user_value, "shap_impact": float}}
        
        +ve shap_impact -> pushes toward aggressive.
        -ve shap_impact -> pushes toward conservative.
        """
        if self._ml_model is not None:
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
                    f"Cannot classify risk profile: missing features {missing}. "
                    f"Please collect these via ConversationalAgent before calling "
                    f"RiskProfilingAgent."
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
        confidence = self._compute_confidence(hybrid)
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
            "confidence_flag": confidence_flag,
            "feature_importance": shap_proxy,
            "rationale": rationale,
            "missing_features": [],
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[RiskProfilingAgent] risk_class={risk_class} "
            f"hybrid={hybrid:.3f} confidence={confidence:.3f} "
            f"duration={duration_ms:.0f}ms"
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




