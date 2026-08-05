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

import time
from typing import Any

from agents.base_agent import AgentResult, BaseAgent
from config.prompts import (
    EXPLAINABILITY_CALIBRATION_PROMPT,
    EXPLAINABILITY_COUNTERFACTUAL_PROMPT,
    EXPLAINABILITY_SHAP_PROMPT,
)
from config.settings import settings
from utils import trace
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
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, name="ExplainabilityAgent")

    @property
    def system_prompt(self) -> str:
        return EXPLAINABILITY_SHAP_PROMPT

    def _parse_response(self, raw: str) -> dict[str, Any]:
        return {"explanation": raw.strip()}

    def _call_with_system(
            self, system: str, user_message: str, temperature: float = 0.2
    ) -> tuple[str, int]:
        """
        Variant of _Call_llm that accepts an explicit system prompt.
        Used by XAI layers that each have their own system prompt
        (SHAP, counterfactual, calibration) rather than sharing the
        agent-level system_prompt.
        """
        start = time.perf_counter()
        trace.emit("PROMPT", system.strip().splitlines()[0][:48],
                   chars=len(user_message), temp=temperature)

        response = self.llm.chat(
            system=system,
            messages=[{"role": "user", "content": user_message}],
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._call_count += 1

        # R18: this method bypasses BaseAgent._call_llm, so these calls were
        # never timed and tokens_total was always 0. Accumulate here instead.
        self._tokens_this_run = getattr(self, "_tokens_this_run", 0) + response.tokens_used

        trace.emit("LLM", f"← {response.tokens_used} tok",
                   duration_ms=round(elapsed_ms),
                   model=response.model, mode=self.llm.mode)
        return response.content, response.tokens_used


    # Layer A - SHAP feature attribution narrative
    # always active across all ablation conditions.

    def _attribution_method_label(self, shap_summary: dict[str, dict]) -> str:
        """
        Deterministic stamp distinguishing real SHAP output from a
        coefficient estimate — computed from the same top-3 features the
        narrative is built from, never left to the LLM to state (the
        prompt is explicitly forbidden from naming the method, since an
        LLM claim about its own grounding can't be trusted the way a
        value read straight off the payload can).
        """
        if not shap_summary:
            return "coefficient estimates"

        top3 = sorted(
            shap_summary.items(),
            key=lambda x: abs(x[1].get("shap_impact", 0)),
            reverse=True,
        )[:3]
        sources = {info.get("source", "proxy") for _, info in top3}

        if sources == {"shap"}:
            return "SHAP attribution"
        if "shap" in sources:
            return "a mix of SHAP attribution and coefficient estimates"
        return "coefficient estimates"

    def _generate_shap_narrative(
            self,
            shap_summary: dict[str, dict],
            risk_class: str,
    ) -> str:
        """
        Convert SHAP attribution dict from RiskProfilingAgent into a
        plain-English narrative for the user.

        shap_summary format (from RiskProfilingAgent._compute_shap_proxy()):
            {feature_name: {"value": user_value, "shap_impact": float,
                             "source": "shap" | "proxy"}}

        Sorted by absolute impact — top 3 features form the narrative.
        LLM generates the text; SHAP data is the grounding so the LLM
        cannot fabricate feature contributions (Klesel & Wittmann [6]).
        The method label (SHAP vs coefficient estimate) is prepended by
        code, not asked of the LLM — see _attribution_method_label().
        """
        if not shap_summary:
            return(
                "Insufficient feature data to produce an attribution "
                "explanation for this classification."
            )

        method_label = self._attribution_method_label(shap_summary)

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
                EXPLAINABILITY_SHAP_PROMPT, prompt, temperature=0.2
            )
            return f"Based on {method_label}. {raw.strip()}"
        except Exception as exc:
            logger.warning(f"[ExplainablityAgent] SHAP narrative failed: {exc}")
            # Deterministic fallback - no LLM needed
            top_feat, top_info = sorted_features[0]
            direction = "toward a higher" if top_info["shap_impact"] >0 else "toward a lower"
            return (
                f"Based on {method_label}. "
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

    def _build_rag_query(self, context: dict[str, Any]) -> str:
        """
        Build the retrieval query from the recommendation being explained.

        Combines the risk classification, the top recommended product, and
        the InvestmentAgent's own synthesis text — the same fields a human
        reader would need grounding for.
        keeping the query focused on the claim-bearing parts (product
        category + return figures) retrieves more relevant chunks than
        embedding the whole paragraph, including its calibration language.
        """
        investment_payload: dict = context.get("investment_agent_payload") or {}
        risk_payload: dict = context.get("risk_agent_payload") or {}

        risk_class = risk_payload.get("risk_class", "")
        shortlist = investment_payload.get("shortlist", [])
        synthesis = investment_payload.get("synthesis", "")

        parts = [p for p in (risk_class, synthesis) if p]
        if shortlist:
            top = shortlist[0]
            parts.append(f"{top.get('category', '')} {top.get('name', '')}")
        return " ".join(parts).strip()

    _CITATION_DOCUMENT_SETS: tuple[str, ...] = (
        "regulatory", "cbi_open_data", "eu_digital_finance",
    )

    def _get_rag_citations(
            self, context: dict[str, Any]
    ) -> list[dict]:
        """
        Retrieve source citations for factual claims in the recommendation.

        Phase 6: returned an empty list — RAG knowledge base not yet built.
        Phase 8: retrieves real citations from
          rag.knowledge_base.knowledge_base, a FAISS vector store over
          CBI Open Data [D3], EU Digital Finance Platform [D4], and the
          FinQA corpora [D1, D2].

        Ablation condition B (shap + rag_citation) now returns real
        citation counts instead of the Phase 6 baseline of 0 — this is
        the change re-evaluated in commit 40
        (tests/unit/test_explainability_agent.py::TestRQ3AblationEvaluation)
        and committed to results/rq3_shap_rag.json.

        Returns:
            List of {"claim": str, "source": str, "relevance": float,
                      "text": str, "document_set": str} dicts, capped at
            settings.rag.top_k_citations and filtered by
            settings.rag.min_relevance_score.
        """
        from rag.knowledge_base import knowledge_base  # noqa: PLC0415

        query = self._build_rag_query(context)
        if not query:
            logger.debug("[ExplainabilityAgent] RAG citations: empty query, no context to ground")
            return []

        try:
            citations = knowledge_base.retrieve(
                query, document_sets=list(self._CITATION_DOCUMENT_SETS)
            )
        except Exception as exc:
            logger.warning(f"[ExplainabilityAgent] RAG retrieval failed: {exc} — returning []")
            return []

        logger.debug(
            f"[ExplainabilityAgent] RAG citations: {len(citations)} retrieved "
             f"for query={query[:60]!r}... (scoped to {self._CITATION_DOCUMENT_SETS})"
        )
        return citations

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
            hallucination_flagged: bool = False,
    ) -> str:
        confidence_flag = ""
        if confidence < settings.explainability.low_confidence_threshold:
            confidence_flag = (
                f" Note: this classification has low confidence ({confidence:.0%}) "
                f"- it is near a tier boundary and should be treated as indicative only."
            )

        hallucination_flag = ""
        if hallucination_flagged:
            hallucination_flag = (
                " Note: one or more figures in this recommendation could not "
                "be fully verified against source data — please confirm "
                "specific numbers independently before acting on them."
            )

        prompt = (
            f"Risk classification: {risk_class} (confidence: {confidence:.0%})\n"
            f"Top product: {top_product_name}\n\n"
            f"Write the specific calibration condition sentence now."
        )
        try:
            raw, _ = self._call_with_system(
                EXPLAINABILITY_CALIBRATION_PROMPT, prompt, temperature=0.1
            )
            return raw.strip() + confidence_flag + hallucination_flag
        except Exception as exc:
            logger.warning(
                f"[ExplainabilityAgent] Calibration note failed: {exc}"
            )
            return(
                "This recommendation assumes stable employment and income. "
                "A significan change to either would warrant reassessment."
                + confidence_flag
                + hallucination_flag
            )
    def _generate_budget_calibration_note(
        self, budget_payload: dict, coverage: float
    ) -> str:
        """
        The X3 calibration note for a budget-only turn. (R33)

        Deterministic on purpose — no LLM call. Everything it states is
        arithmetic the BudgetAgent already did, so generating it with a model
        would introduce a hallucination surface for no benefit, and would cost
        a token round-trip on every budget turn.

        It names the two things that actually limit a budget conclusion:
        the benchmark's coverage of this user's categories, and the fact that
        one month of self-reported spending is a small sample.
        """
        benchmark = budget_payload.get("benchmark_comparison") or {}
        expenses = budget_payload.get("monthly_expenses") or {}
        matched, total = len(benchmark), len(expenses)
        source = budget_payload.get("data_source", "the national benchmark")

        note = (
            f"This analysis compares your spending against {source}. "
            f"{matched} of your {total} categories have a national benchmark; "
            f"any others are reported without comparison."
        )
        if total and coverage < 0.6:
            note += (
                f" Benchmark coverage is limited ({coverage:.0%}), so treat the "
                f"comparison as indicative rather than a complete picture of "
                f"where you differ from the average household."
            )
        note += (
            " Figures are based on the single month of spending you provided; "
            "irregular costs such as annual insurance or one-off purchases may "
            "not be represented."
        )
        return note

    def _generate_estimated_input_note(
        self,
        proxy_fields: list[str],
        proxy_metadata: dict[str, dict],
    ) -> str | None:
        """
        One sentence per proxied field, naming the field, its estimated
        value, and the basis it was estimated from. Returns None if no
        fields were proxied for this customer.
        """
        if not proxy_fields:
            return None

        sentences = []
        for field in proxy_fields:
            meta = proxy_metadata.get(field)
            if not meta:
                continue
            label = field.replace("_", " ")
            basis = ", ".join(meta.get("basis", [])) or "your overall financial profile"
            sentences.append(
                f"Your {label} was estimated at {meta['value']}/5 "
                f"(based on {basis}) rather than self-reported, "
                f"since this wasn't available from your bank profile — "
                f"let us know if this doesn't reflect you and we'll update it."
            )
        return " ".join(sentences) if sentences else None


    # Output assembly
    def _assemble_explanation(
            self,
            shap_narrative: str | None,
            rag_citations: list[dict],
            counterfactual: str | None,
            calibration_note: str,
            estimated_input_note: str | None,
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
            parts.append(counterfactual)
        parts.append(calibration_note)
        if estimated_input_note:
            parts.append(estimated_input_note)
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

        budget_payload: dict = context.get("budget_agent_payload") or {}

        nothing_to_explain = not risk_payload and not investment_payload and not budget_payload
        if (
            nothing_to_explain
            and settings.collaboration.enabled
            and context.get("user_features")
        ):
            return self._make_result(
                payload={
                    "status": "needs_input",
                    "needs": ["risk_class"],
                    "reason": (
                        "Nothing to explain yet this session, but a "
                        "customer profile is on file — worth a risk "
                        "classification before falling back to a generic "
                        "explanation."
                    ),
                },
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        shap_summary: dict = risk_payload.get("feature_importance", {})
        risk_class: str = risk_payload.get("risk_class", "moderate")
        confidence: float = float(risk_payload.get("confidence", 0.7))
        # budget_payload: dict = context.get("budget_agent_payload") or {}

        shortlist: list = investment_payload.get("shortlist", [])
        top_product_name: str = (
            shortlist[0]["name"] if shortlist else "the recommended product"
        )
        budget_only = bool(budget_payload) and not risk_payload and not shortlist

        if budget_only:
            benchmark = budget_payload.get("benchmark_comparison") or {}
            expenses = budget_payload.get("monthly_expenses") or {}
            coverage = (len(benchmark) / len(expenses)) if expenses else 0.0
            confidence = round(min(max(coverage, 0.0), 1.0), 4)
            risk_class = ""          # honestly absent, not silently "moderate"
            top_product_name = "this budget analysis"

        layers_applied: list[str] = []
        tokens_total: int = 0


        # Layer A: SHAP (always active)
        trace.emit("LAYER A", "shap narrative" if cfg.use_shap
                   else "SKIPPED (use_shap=False)")
        shap_narrative: str | None = None
        attribution_method: str | None = None
        if cfg.use_shap and shap_summary:
            shap_narrative = self._generate_shap_narrative(shap_summary, risk_class)
            attribution_method = self._attribution_method_label(shap_summary)
            layers_applied.append("shap")
            logger.debug("[ExplainabilityAgent] Layer A (SHAP) applied")

        # Layer B: RAG citations
        trace.emit("LAYER B", "rag citations" if cfg.use_rag_citation
                   else "SKIPPED (use_rag_citation=False)")
        rag_citations: list[dict] = []
        if cfg.use_rag_citation:
            rag_citations = self._get_rag_citations(context)
            layers_applied.append("rag_citation")
            logger.debug(
                f"[ExplainabilityAgent] Layer B (RAG) applied "
                f"- {len(rag_citations)} citations"
            )

        # Layer C: Counterfactual
        trace.emit("LAYER C", "counterfactual" if cfg.use_counterfactual
                   else "SKIPPED (use_counterfactual=False)")
        counterfactual: str | None = None
        if cfg.use_counterfactual and shap_summary:
            top_feat, top_info = self._get_top_shap_feature(shap_summary)
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
        hallucination_flagged: bool = bool(investment_payload.get("hallucination_flagged", False))
        if budget_only:
            calibration_note = self._generate_budget_calibration_note(
                budget_payload, confidence
            )
        else:
            calibration_note = self._generate_calibration_note(
                risk_class=risk_class,
                confidence=confidence,
                top_product_name=top_product_name,
                hallucination_flagged=hallucination_flagged,
            )
        if "calibration_note" not in layers_applied:
            layers_applied.append("calibration_note")

        proxy_fields: list[str] = context.get("proxy_fields", [])
        proxy_metadata: dict = context.get("proxy_metadata", {})
        estimated_input_note = self._generate_estimated_input_note(
            proxy_fields, proxy_metadata
        )
        if estimated_input_note and "estimated_input_disclosure" not in layers_applied:
            layers_applied.append("estimated_input_disclosure")

        # Assemble full explanation
        full_explanation = self._assemble_explanation(
            shap_narrative=shap_narrative,
            rag_citations=rag_citations,
            counterfactual=counterfactual,
            calibration_note=calibration_note,
            estimated_input_note=estimated_input_note,
            layers_applied=layers_applied,
            confidence=confidence,
        )

        payload = {
            "layers_applied": layers_applied,
            "prompt_version": cfg.prompt_version,
            "ablation_condition": self._label_ablation_condition(cfg),
            "shap_narrative": shap_narrative,
            "attribution_method": attribution_method,
            "rag_citations": rag_citations,
            "counterfactual": counterfactual,
            "calibration_note": calibration_note,
            "estimated_input_note": estimated_input_note,
            "proxy_fields": proxy_fields,
            "confidence": confidence,
            "low_confidence_flagged": confidence < cfg.low_confidence_threshold,
            "hallucination_flagged": hallucination_flagged,
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