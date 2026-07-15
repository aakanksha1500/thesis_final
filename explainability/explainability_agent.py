"""
Phase 6 - ExplainabilityAgent
Addresses RQ3: XAI effectiveness - 3-layer ablation study.

Decision X1 (Kawakami et al. [ACM IUI 2025]): in-pipeline, not post-hoc.
This agent runs INSIDE the pipeline before every response is delivered
to the user - not as a retrospective explainer. This is the core
architectural commitment that distinguishes this system from post-hoc
wrappers like LIME or SHAP dashboards.

THREE-LAYER STACK (Decision X2):
    Layer A: SHAP feature attributions
    Layer B: RAG source citations
    LAyer C: Counterfactual NL rationale
    
Each evaluation commit re-runs the same test fixutre and records the 
transperancy_perception_Score and trust_calibration_index. The three
JSON files constitute the ablation table for dissertation

To switch ablation conditions:
    In config/settings.py, toggle ExplainabilityConfig booleans.
    Re-run: pytest tests/unit/text_explainability_agent.py -v -s
"""

from __future__ import annotations

import json
import time
from typing import Any

from agents.base_agent import BaseAgent, AgentResult
from config.prompts import (
    EXPLAINABILITY_CALIBRATION_PROMPT,
    EXPLAINABILITY_COUNTERFACTUAL_PROMPT,
    EXPLAINABILITY_SHAP_PROMPT,
)
from config.settings import settings
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

class ExplainabilityAgent(BaseAgent):
    """
    In-pipeline XAI module
    
    Consumes output from RiskProfilingAgent and InvestmentAgent and
    wraps them with a layered explanation before delivery to the user.
    
    Layer activation is controlled by settings.explainability booleans,
    not by constructor arguments, so ablation conditions are reproducible 
    from config alone
    """
    def __inti__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="ExplainabilityAgent")

    @property
    def system_prompt(self) -> str:
        return EXPLAINABILITY_SHAP_PROMPT
    
    def _parse_response(self, raw: str) -> dict[str, Any]:
        return {"explanation": raw.strip()}
    
    def _call_with_system(
            self, system: str, user_message: str, temprature: float = 0.2
    ) -> tuple[str, int]:
        """
        Variant of _Call_llm that accepts an explicit system prompt.
        Used by XAI layers that each have their own system prompt
        (SHAP, counterfactual, calibration) rather than sharing the 
        agent-level system_prompt.
        """
        response = self.llm.chat(
            system=system,
            messages=[{"role": "user", "content": user_message}],
            temprature=temprature,
        )
        self._call_count += 1
        return response.content, response.tokens_used
    

    # Layer A - SHAP feature attribution narrative
    # always active across all ablation conditions.

    def _generate_shap_narratice(
            self,
            shap_summary: dict[str, dict],
            risk_class: str,
    ) -> str:
        """
        Convert SHAP attribution dict from RiskProfilingAgent into a 
        plain-English narrative for the user.
        
        shap_summary format (from RiskProfilingAgent._compute_shap_proxy()):
            {feature_name: {"value": user_value, "shap_impact": float}}
            
        Sorted by absolute impact — top 3 features form the narrative.
        LLM generates the text; SHAP data is the grounding so the LLM
        cannot fabricate feature contributions (Klesel & Wittmann [6]).
        """
        if not shap_summary:
            return(
                "Insufficient feature data to produce an attribution "
                "explanation for this classification."
            )
        
        # Sort by absolute impact, take top 3
        sorted_features = sorted(
            shap_summary.items(),
            key=lambda x: abs(x[1].get("shap_impact", 0)),
            reverse=True,
        )[:3]

        # Format for LLM prompt
        attribution_text = "\n".join(
            f"  {feat}:  value={info['value']}, "
            f"impact={'positive (-> more aggressive)' if info['shap_impact'] > 0 else 'negative (-> more conservative)'}, "
            f"magnitude={abs(info['shap_impact']):.4f}"
            for feat, info in sorted_features
        )
        prompt = (
            f"Feature attributions for '{risk_class}' classification:\n"
            f"{attribution_text}\n\n"
            f"Write the plain-English attribution explanation now."
        )

        try:
            raw, _ = self._call_with_system(
                EXPLAINABILITY_SHAP_PROMPT, prompt, temprature=0.2
            )
            return raw.strip()
        except Exception as exc:
            logger.warning(f"[ExplainablityAgent] SHAP narrative failed: {exc}")
            # Deterministic fallback - no LLM needed
            top_feat, top_info = sorted_features[0]
            direction = "toward a higher" if top_info["shap_impact"] >0 else "toward a lower"
            return (
                f"The most influential factor in your risk classification was "
                f"your {top_feat.replace('_', ' ')} (value: {top_info['value']}), "
                f"which pushed {direction} risk tier."
            )
    
    def _get_top_shap_feature(
            self, shap_summary: dict[str, dict]
    ) -> tuple[str, dict] | tuple[None, None]:
        """Return the feature with the highest absolute SHAP impact."""
        if not shap_summary:
            return None, None
        return max(
            shap_summary.items(),
            key=lambda x: abs(x[1].get("shap_impact", 0)),
        )
    

    # Layer B - RG source citations - paused until phase 8
    def _get_rag_citations(
            self, context: dict[str, Any]
    ) -> list[dict]:
        """
        Retrieve source citations for factual claims in the recommendation
        
        The stub is intentional: it means the ablation condition B
        (shap + rag_citation) can be committed now with use_rag_citation=True
        and a real score of 0 citations — establishing the baseline before
        Phase 8 replaces this with actual retrieval.
        
        Returns:
            List of {"claim": str, "source": str, "relevance": float} dicts.
        """
        logger.debug(
            "[ExplainabilityAgent] RAG citations: stub returning [] "
            "(Phase 8 will populate with real retrieval)"
        )
        return []
    
    # Layer C - Counterfactual NL rationale
    def _generate_counterfactual(
            self,
            risk_class: str,
            top_feature: str,
            top_feature_value: Any,
            top_product_name: str,
    ) -> str:
        """
        Generate a concrete condition under which the advice may not hold.

        X3 design goal: trust must be calibrated, not maximised.
        Takayanagi et al. [7] found that users express high trust in AI
        financial advice regardless of quality. The calibration note
        directly counters this by stating a specific failure condition.

        Never ablated — present in all three ablation conditions.
        Low confidence (< threshold) triggers an explicit uncertainty flag.
        """
        prompt = (
            f"Current risk classification: {risk_class}\n"
            f"Top-influencing feature: {top_feature} "
            f"(current value: {top_feature_value})\n"
            f"Current top product recommendation: {top_product_name}\n\n"
            f"Write the counterfactual sentence now."
        )
        try:
            raw, _ = self._call_with_system(
                EXPLAINABILITY_COUNTERFACTUAL_PROMPT, prompt, temperature=0.3
            )
            return raw.strip()
        except Exception as exc:
            logger.warning(
                f"[ExplainabilityAgent] Counterfactual generation failed: {exc}"
            )
            return (
                f"If your {top_feature.replace('_', ' ')} were higher, "
                f"your risk profile would shift toward a more aggressive tier "
                f"and higher-growth products would become appropriate."
            )
    
    # Calibration note — always active (X3, never ablated)
    def _generate_calibration_note(
            self,
            risk_class: str,
            confidence: float,
            top_product_name: str,
    ) -> str:
        confidence_flag = ""
        if confidence < settings.explainability.low_confidence_threshold:
            confidence_flag = (
                f" Note: this classification has low confidence ({confidence:.0%}) "
                f"- it is near a tier boundary and should be treated as indicative only."
            )

        prompt = (
            f"Risk classification: {risk_class} (confidence: {confidence:.0%})\n"
            f"Top product: {top_product_name}\n\n"
            f"Write the specific calibration condition sentence now."
        )
        try:
            raw, _ = self._call_with_system(
                EXPLAINABILITY_CALIBRATION_PROMPT, prompt, temprature=0.1
            )
            return raw.strip() + confidence_flag
        except Exception as exc:
            logger.warning(
                f"[ExplainabilityAgent] Calibrton note failed: {exc}"
            )
            return(
                f"This recommendation assumes stable employment and income. "
                f"A significan change to either would warrant reassessment."
                + confidence_flag
            )
        
    
    # Output assembly
    def _assemble_explanation(
            self,
            shap_narrative: str | None,
            rag_citations: list[dict],
            counterfactual: str | None,
            calibration_note: str,
            layers_applied: list[str],
            confidence: float,
    ) -> str:
        """
        Combine active layers into a single coherent explanation paragraph.
        Order: SHAP -> citations -> counterfactual -> calibration note
        """
        parts = []
        if shap_narrative:
            parts.append(shap_narrative)
        if rag_citations:
            sources = ", ".join(c.get("source", "unknown") for c in rag_citations[:2])
            parts.append(f"This analysis draws on: {sources}.")
        if counterfactual:
            parts.append(calibration_note)
            return " ".join(parts)
        
    
    # Main entry point
    def run(self, context: dict[str, Any]) -> AgentResult:
        """
        Generate layered explanation for a risk + investment recommendation
        
        context keys used:
            'risk_agent_payload'    (dict) -> from RiskProfilingAgent.run()
            'investment_agent_payload'  (dict) -> from InvestmentAgent.run()
        
        Both payloads are optional - agent degrades gracefully if either
        is missing (e.g. when called standalone in tests).
        
        Returns AgentResult with payload:
            layers_applied, shap_narrative, rag_citations, counterfactual,
            calibration_note, confidence, full_explanation, prompt_version
        """
        start_time = time.perf_counter()
        cfg = settings.explainability

        # Extract inputs from prior agent outputs
        risk_payload: dict = context.get("risk_agent_payload") or {}
        investment_payload: dict = context.get("investment_agent_payload") or {}

        shap_summary: dict = risk_payload.get("feature_importance", {})
        risk_class: str = risk_payload.get("risk_class", "moderate")
        confidence: float = float(risk_payload.get("confidence", 0.7))

        shortlist: list = investment_payload.get("shortlist", [])
        top_product_name: str = (
            shortlist[0]["name"] if shortlist else "the recomended product"
        )

        layers_applied: list[str] = {}
        tokens_total: int = 0

    
        # Layer A: SHAP (always active)
        shap_narrative: str | None = None
        if cfg.use_shap and shap_summary:
            shap_narrative = self._generate_shap_narrative(shap_summary, risk_class)
            layers_applied.append("shap")
            logger.debug("[ExplainabilityAgent] Layer A (SHAP) applied")

        # Layer B: RAG citations
        rag_citations: list[dict] = []
        if cfg.use_rag_citation:
            rag_citations = self._get_rag_citations(context)
            layers_applied.append("rag_citation")
            logger.debug(
                f"[ExplainabilityAgent] Layer B (RAG) applied "
                f"- {len(rag_citations)} citations"
            )
        
        # Layer C: Counterfactual
        counterfactual: str | None = None
        if cfg.use_counterfactual and shap_summary:
            top_feat, top_info = self._gettop_shap_feature(shap_summary)
            if top_feat:
                counterfactual = self._generate_counterfactual(
                    risk_class=risk_class,
                    top_feature=top_feat,
                    top_feature_value=top_info.get("value", "N/A"),
                    top_product_name=top_product_name,
                )
                layers_applied.append("counterfactual")
                logger.debug("[ExplainabilityAgent] Layer C (counterfactual) applied")

        # Calibration note (X3 — always active
        calibration_note = self._generate_calibration_note(
            risk_class=risk_class,
            confidence=confidence,
            top_product_name=top_product_name,
        )
        if "calibration_note" not in layers_applied:
            layers_applied.append("calibration_note")
        
        # Assemble full explanation
        full_explanation = self._assemble_explanation(
            shap_narrative=shap_narrative,
            rag_citations=rag_citations,
            counterfactual=counterfactual,
            calibration_note=calibration_note,
            layers_applied=layers_applied,
            confidence=confidence,
        )

        payload = {
            "layers_applied": layers_applied,
            "prompt_version": cfg.prompt_version,
            "ablation_condition": self._label_ablation_condition(cfg),
            "shap_narrative": shap_narrative,
            "rag_citations": rag_citations,
            "counterfactual": counterfactual,
            "calibration_note": calibration_note,
            "confidence": confidence,
            "low_confidence_flagged": confidence < cfg.low_confidence_threshold,
            "full_explanation": full_explanation,
            "risk_class": risk_class,
            "top_product": top_product_name,
        }

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[ExplainabilityAgent] layers={layers_applied} "
            f"condition={payload['ablation_condition']} "
            f"duration={duration_ms:.0f}ms"
        )

        return self._make_result(
            payload=payload,
            raw=full_explanation,
            duration_ms=duration_ms,
            tokens=tokens_total,
            routing_context={
                "layers_applied": layers_applied,
                "ablation_condition": payload["ablation_condition"],
            },
        )

    def _label_ablation_condition(self, cfg) -> str:
        """Return a human-readable ablation condition label for results JSON."""
        layers = []
        if cfg.use_shap:
            layers.append("shap")
        if cfg.use_rag_citation:
            layers.append("rag")
        if cfg.use_counterfactual:
            layers.append("counterfactual")
        if not layers:
            return "no_xai"
        return "+".join(layers)
