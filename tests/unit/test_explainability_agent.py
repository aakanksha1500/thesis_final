"""
Phase 6 - ExplainabilityAgent evaluation

ABLATION STRUCTURE:
This file runs three ablation conditions and writes three results files.
The condition is controlled by settings.explainability booleans —
no code changes between conditions, only config toggles.

  Condition A: SHAP only
    settings: use_shap=True, use_counterfactual=False, use_rag_citation=False
    Results → results/rq3_shap_only.json

  Condition B: SHAP + RAG citation (stub in Phase 6)
    settings: use_shap=True, use_rag_citation=True, use_counterfactual=False
    Results → results/rq3_shap_rag.json

  Condition C: Full 3-layer stack (primary condition)
    settings: use_shap=True, use_rag_citation=True, use_counterfactual=True
    Results → results/rq3_full_stack.json

Calibration note (X3) is always active and appears in all three results files.
It is NOT part of the ablation — it is a baseline safety requirement.

SURVEY FIXTURE:
Synthetic Likert-scale survey responses simulate a 10-participant pilot study.
Responses are hand-crafted to reflect expected improvements across conditions:
  Condition A: clarity moderate, no source visible, no counterfactual
  Condition B: clarity moderate, source visible (even if 0 citations returned)
  Condition C: clarity high, source visible, counterfactual useful

This is the honest way to represent mock data for a dissertation evaluation:
the fixture explicitly states it is synthetic and the real pilot study
scores replace it when collected.

RUNNING:
  pytest tests/unit/test_explainability_agent.py -v -s
  (shows per-condition scores and writes all three results files)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from agents.base_agent import AgentResult
from config.settings import settings
from evaluation.metrics import (
    EvalResult,
    transparency_perception_score,
    trust_calibration_index,
)
from evaluation.results_io import write_results
from explainability.explainability_agent import ExplainabilityAgent
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# Helpers
def make_agent() -> ExplainabilityAgent:
    """ExplainabilityAgent in mock mode."""
    client = LLMClient()
    return ExplainabilityAgent(client)

@contextmanager
def ablation_condition(use_shap: bool, use_rag: bool, use_counterfactual: bool):
    """
    Context manager that temporarily sets ablation condition flags.
    Restores original settings on exit — tests don't bleed into each other.
    """
    original_shap = settings.explainability.use_shap
    original_rag = settings.explainability.use_rag_citation
    original_cf = settings.explainability.use_counterfactual

    settings.explainability.use_shap = use_shap
    settings.explainability.use_rag_citation = use_rag
    settings.explainability.use_counterfactual = use_counterfactual
    try:
        yield
    finally:
        settings.explainability.use_shap = original_shap
        settings.explainability.use_rag_citation = original_rag
        settings.explainability.use_counterfactual = original_cf

# Shared context representing a moderate-risk user with investment recommendation
SAMPLE_CONTEXT: dict[str, Any] = {
    "risk_agent_payload": {
        "risk_class": "moderate",
        "confidence": 0.72,
        "hybrid_score": 0.52,
        "feature_importance": {
            "loss_tolerance": {
                "value": 3,
                "shap_impact": 0.08,
            },
            "investment_horizon": {
                "value": 10,
                "shap_impact": 0.06,
            },
            "existing_debt": {
                "value": 5000,
                "shap_impact": -0.04,
            },
            "age": {
                "value": 35,
                "shap_impact": -0.01,
            },
            "financial_knowledge_score": {
                "value": 3,
                "shap_impact": 0.04,
            },
        },
        "rationale": "Based on your profile you are classified as moderate.",
    },
    "investment_agent_payload": {
        "risk_class": "moderate",
        "shortlist": [
            {
                "product_id": "ETB001",
                "name": "Broad Global Equity ETF",
                "category": "etf_broad",
                "expected_return_pct": 7.0,
                "expense_ratio_pct": 0.2,
                "score": 0.82,
            },
            {
                "product_id": "MXM001",
                "name": "Balanced Mixed Fund (60/40 equity/bond)",
                "category": "mixed_fund",
                "expected_return_pct": 6.0,
                "expense_ratio_pct": 0.6,
                "score": 0.71,
            },
        ],
        "synthesis": "Based on your moderate profile we recommend the ETF.",
    },
}

# GROUP A: Unit tests - structure and layer activation
class TestLayerActivation:

    def test_shap_only_condition_applies_shap_layer(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert "shap" in result.payload["layers_applied"]

    def test_shap_only_condition_does_not_apply_counterfactual(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert "counterfactual" not in result.payload["layers_applied"]
        assert result.payload["counterfactual"] is None

    def test_full_stack_applies_all_layers(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=True, use_counterfactual=True):
            result = agent.run(SAMPLE_CONTEXT)
        assert "shap" in result.payload["layers_applied"]
        assert "counterfactual" in result.payload["layers_applied"]

    def test_calibration_note_always_present(self):
        """X3 requirement — calibration note never ablated."""
        agent = make_agent()
        with ablation_condition(use_shap=False, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert "calibration_note" in result.payload["layers_applied"]
        assert result.payload["calibration_note"] is not None
        assert len(result.payload["calibration_note"]) > 0

    def test_calibration_note_present_in_all_conditions(self):
        agent = make_agent()
        for shap, rag, cf in [
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ]:
            with ablation_condition(use_shap=shap, use_rag=rag, use_counterfactual=cf):
                result = agent.run(SAMPLE_CONTEXT)
            assert result.payload["calibration_note"] is not None

    def test_rag_citation_off_returns_empty_list(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["rag_citations"] == []

    def test_low_confidence_flag_triggered(self):
        agent = make_agent()
        low_conf_context = {
            **SAMPLE_CONTEXT,
            "risk_agent_payload": {
                **SAMPLE_CONTEXT["risk_agent_payload"],
                "confidence": 0.40,  # below threshold of 0.6
            },
        }
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(low_conf_context)
        assert result.payload["low_confidence_flagged"] is True

    def test_high_confidence_no_flag(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["low_confidence_flagged"] is False


class TestAgentResultStructure:

    def test_result_is_agent_result(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        assert isinstance(result, AgentResult)

    def test_result_has_required_payload_keys(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        for key in (
            "layers_applied", "prompt_version", "ablation_condition",
            "shap_narrative", "rag_citations", "counterfactual",
            "calibration_note", "confidence", "low_confidence_flagged",
            "full_explanation", "risk_class", "top_product",
        ):
            assert key in result.payload, f"Missing key: {key}"

    def test_full_explanation_is_non_empty(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        assert isinstance(result.payload["full_explanation"], str)
        assert len(result.payload["full_explanation"]) > 0

    def test_risk_class_propagated(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["risk_class"] == "moderate"

    def test_top_product_propagated(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        assert "ETF" in result.payload["top_product"] or \
               "Equity" in result.payload["top_product"]

    def test_agent_name(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        assert result.agent_name == "ExplainabilityAgent"

    def test_step_record_structure(self):
        agent = make_agent()
        result = agent.run(SAMPLE_CONTEXT)
        step = result.to_step_record()
        for key in ("step_id", "agent", "completed", "duration_ms"):
            assert key in step

    def test_missing_risk_payload_degrades_gracefully(self):
        agent = make_agent()
        result = agent.run({"investment_agent_payload": {}})
        assert result.success is True  # degrades, does not crash

    def test_empty_context_degrades_gracefully(self):
        agent = make_agent()
        result = agent.run({})
        assert result.success is True

class TestSHAPNarrative:

    def test_shap_narrative_generated_from_valid_summary(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["shap_narrative"] is not None

    def test_shap_narrative_none_when_layer_off(self):
        agent = make_agent()
        with ablation_condition(use_shap=False, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["shap_narrative"] is None

    def test_top_shap_feature_identified(self):
        agent = make_agent()
        top_feat, top_info = agent._get_top_shap_feature(
            SAMPLE_CONTEXT["risk_agent_payload"]["feature_importance"]
        )
        assert top_feat == "loss_tolerance"  # highest abs impact in fixture

    def test_top_shap_feature_empty_returns_none(self):
        agent = make_agent()
        feat, info = agent._get_top_shap_feature({})
        assert feat is None
        assert info is None

    def test_ablation_condition_label_shap_only(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=False, use_counterfactual=False):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["ablation_condition"] == "shap"

    def test_ablation_condition_label_full_stack(self):
        agent = make_agent()
        with ablation_condition(use_shap=True, use_rag=True, use_counterfactual=True):
            result = agent.run(SAMPLE_CONTEXT)
        assert result.payload["ablation_condition"] == "shap+rag+counterfactual"

# RQ3 metric uit tests
class TestTransparencyPerceptionScore:

    def test_perfect_responses_give_score_one(self):
        responses = [
            {"clarity": 5, "source_visible": 1,
             "counterfactual_useful": 5, "trust_appropriate": 5}
        ] * 5
        result = transparency_perception_score(responses)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_worst_responses_give_score_near_zero(self):
        responses = [
            {"clarity": 1, "source_visible": 0,
             "counterfactual_useful": 1, "trust_appropriate": 1}
        ] * 5
        result = transparency_perception_score(responses)
        assert result.value < 0.3

    def test_empty_responses_returns_error(self):
        result = transparency_perception_score([])
        assert "error" in result.details

    def test_details_has_per_dimension_means(self):
        responses = [
            {"clarity": 4, "source_visible": 1,
             "counterfactual_useful": 4, "trust_appropriate": 4}
        ] * 3
        result = transparency_perception_score(responses)
        assert "mean_clarity" in result.details
        assert "mean_source_visible" in result.details
        assert "mean_counterfactual_useful" in result.details
        assert "mean_trust_appropriate" in result.details

    def test_score_in_range(self):
        responses = [
            {"clarity": 3, "source_visible": 0,
             "counterfactual_useful": 3, "trust_appropriate": 3}
        ] * 10
        result = transparency_perception_score(responses)
        assert 0.0 <= result.value <= 1.0


class TestTrustCalibrationIndex:

    def test_perfect_calibration_gives_one(self):
        trust   = [4.0, 3.0, 5.0, 2.0]
        quality = [4.0, 3.0, 5.0, 2.0]
        result = trust_calibration_index(trust, quality)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_takayanagi_scenario_gives_low_tci(self):
        """
        Reproduce Takayanagi et al. finding: users report high trust (5/5)
        for objectively low-quality advice (1/5) — trust-quality decoupling.
        TCI should be very low in this case.
        """
        trust   = [5.0, 5.0, 5.0, 5.0, 5.0]
        quality = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = trust_calibration_index(trust, quality)
        assert result.value < 0.1

    def test_moderate_calibration(self):
        trust   = [4.0, 3.0, 4.0, 3.0]
        quality = [3.0, 3.0, 3.0, 4.0]
        result = trust_calibration_index(trust, quality)
        assert 0.5 < result.value < 1.0

    def test_length_mismatch_returns_error(self):
        result = trust_calibration_index([4.0, 3.0], [4.0])
        assert "error" in result.details

    def test_empty_input_returns_error(self):
        result = trust_calibration_index([], [])
        assert "error" in result.details

    def test_details_has_interpretation(self):
        result = trust_calibration_index([4.0], [4.0])
        assert "interpretation" in result.details

    def test_score_in_range(self):
        trust   = [3.0, 4.0, 2.0, 5.0]
        quality = [4.0, 2.0, 3.0, 3.0]
        result = trust_calibration_index(trust, quality)
        assert 0.0 <= result.value <= 1.0

# RQ3 ablation evaluation — three conditions, three results files
SURVEY_CONDITION_A = [  # SHAP only — no counterfactual, no citations
    {"clarity": 3, "source_visible": 0, "counterfactual_useful": 2, "trust_appropriate": 3},
    {"clarity": 4, "source_visible": 0, "counterfactual_useful": 2, "trust_appropriate": 3},
    {"clarity": 3, "source_visible": 0, "counterfactual_useful": 3, "trust_appropriate": 4},
    {"clarity": 4, "source_visible": 0, "counterfactual_useful": 2, "trust_appropriate": 3},
    {"clarity": 3, "source_visible": 0, "counterfactual_useful": 2, "trust_appropriate": 3},
]

SURVEY_CONDITION_B = [  # SHAP + RAG citations (stub — 0 citations returned)
    {"clarity": 3, "source_visible": 1, "counterfactual_useful": 2, "trust_appropriate": 3},
    {"clarity": 4, "source_visible": 1, "counterfactual_useful": 3, "trust_appropriate": 4},
    {"clarity": 4, "source_visible": 1, "counterfactual_useful": 2, "trust_appropriate": 4},
    {"clarity": 3, "source_visible": 1, "counterfactual_useful": 2, "trust_appropriate": 3},
    {"clarity": 4, "source_visible": 1, "counterfactual_useful": 3, "trust_appropriate": 4},
]

SURVEY_CONDITION_C = [  # Full 3-layer stack — primary condition
    {"clarity": 5, "source_visible": 1, "counterfactual_useful": 4, "trust_appropriate": 5},
    {"clarity": 4, "source_visible": 1, "counterfactual_useful": 5, "trust_appropriate": 4},
    {"clarity": 5, "source_visible": 1, "counterfactual_useful": 4, "trust_appropriate": 5},
    {"clarity": 4, "source_visible": 1, "counterfactual_useful": 5, "trust_appropriate": 5},
    {"clarity": 5, "source_visible": 1, "counterfactual_useful": 4, "trust_appropriate": 4},
]

# Trust and quality scores for TCI (simulated Agent-as-Judge quality scores)
# Without XAI: users trust regardless of quality (Takayanagi et al.)
# With full XAI: trust better tracks quality
TCI_NO_XAI   = {"trust": [4.5, 4.0, 4.5, 5.0, 4.0], "quality": [2.0, 3.0, 2.5, 2.0, 3.0]}
TCI_SHAP_ONLY= {"trust": [3.5, 4.0, 3.5, 3.0, 4.0], "quality": [3.0, 3.5, 3.0, 3.0, 3.5]}
TCI_FULL     = {"trust": [4.0, 3.5, 4.5, 4.0, 4.0], "quality": [4.0, 3.5, 4.5, 4.0, 4.5]}


def _run_ablation_condition(
    agent: ExplainabilityAgent,
    condition_name: str,
    use_shap: bool,
    use_rag: bool,
    use_counterfactual: bool,
    survey_responses: list[dict],
    tci_data: dict,
    results_filename: str,
) -> dict:
    """Run one ablation condition and return results dict."""

    with ablation_condition(use_shap, use_rag, use_counterfactual):
        result = agent.run(SAMPLE_CONTEXT)

    tps_result = transparency_perception_score(survey_responses)
    tci_result = trust_calibration_index(
        tci_data["trust"], tci_data["quality"]
    )

    results = {
        "phase": 6,
        "agent": "ExplainabilityAgent",
        "ablation_condition": condition_name,
        "config": {
            "use_shap": use_shap,
            "use_rag_citation": use_rag,
            "use_counterfactual": use_counterfactual,
            "use_calibration_note": True,
        },
        "layers_applied": result.payload["layers_applied"],
        "metrics": {
            "transparency_perception_score": tps_result.to_dict(),
            "trust_calibration_index": tci_result.to_dict(),
        },
        "sample_explanation": {
            "shap_narrative": result.payload["shap_narrative"],
            "rag_citations": result.payload["rag_citations"],
            "counterfactual": result.payload["counterfactual"],
            "calibration_note": result.payload["calibration_note"],
            "full_explanation": result.payload["full_explanation"],
        },
        "survey_note": (
            "Synthetic survey fixture — replace with real pilot study "
            "responses (n≥20) before dissertation submission."
        ),
    }

    write_results(results, results_filename)

    return results

class TestRQ3AblationEvaluation:

    def test_condition_a_shap_only(self):
        """
        Ablation Condition A: SHAP only.
        Expected: moderate TPS, low TCI (no calibration from counterfactual).
        Writes: results/rq3_shap_only.json
        """
        agent = make_agent()
        results = _run_ablation_condition(
            agent=agent,
            condition_name="shap_only",
            use_shap=True,
            use_rag=False,
            use_counterfactual=False,
            survey_responses=SURVEY_CONDITION_A,
            tci_data=TCI_SHAP_ONLY,
            results_filename="rq3_shap_only.json",
        )
        tps = results["metrics"]["transparency_perception_score"]["value"]
        tci = results["metrics"]["trust_calibration_index"]["value"]
        print(f"\n[RQ3 Condition A — SHAP only] TPS={tps:.3f} TCI={tci:.3f}")
        assert 0.0 <= tps <= 1.0
        assert 0.0 <= tci <= 1.0
        assert results["layers_applied"] == ["shap", "calibration_note"]

    def test_condition_b_shap_rag(self):
        """
        Ablation Condition B: SHAP + RAG citations (stub in Phase 6).
        RAG returns [] citations — source_visible effect is partial.
        Writes: results/rq3_shap_rag.json
        """
        agent = make_agent()
        results = _run_ablation_condition(
            agent=agent,
            condition_name="shap_rag",
            use_shap=True,
            use_rag=True,
            use_counterfactual=False,
            survey_responses=SURVEY_CONDITION_B,
            tci_data=TCI_SHAP_ONLY,
            results_filename="rq3_shap_rag.json",
        )
        tps = results["metrics"]["transparency_perception_score"]["value"]
        tci = results["metrics"]["trust_calibration_index"]["value"]
        print(f"\n[RQ3 Condition B — SHAP+RAG] TPS={tps:.3f} TCI={tci:.3f}")
        assert 0.0 <= tps <= 1.0

    def test_condition_c_full_stack(self):
        """
        Ablation Condition C: Full 3-layer stack (primary condition).
        Expected: highest TPS and TCI across all three conditions.
        Writes: results/rq3_full_stack.json
        """
        agent = make_agent()
        results = _run_ablation_condition(
            agent=agent,
            condition_name="full_stack",
            use_shap=True,
            use_rag=True,
            use_counterfactual=True,
            survey_responses=SURVEY_CONDITION_C,
            tci_data=TCI_FULL,
            results_filename="rq3_full_stack.json",
        )
        tps = results["metrics"]["transparency_perception_score"]["value"]
        tci = results["metrics"]["trust_calibration_index"]["value"]
        print(f"\n[RQ3 Condition C — Full stack] TPS={tps:.3f} TCI={tci:.3f}")
        assert 0.0 <= tps <= 1.0
        assert 0.0 <= tci <= 1.0
        assert "counterfactual" in results["layers_applied"]

    def test_full_stack_tps_higher_than_shap_only(self):
        """
        Cross-condition assertion: Condition C TPS > Condition A TPS.
        This is the key RQ3 finding — full stack outperforms SHAP alone.
        Uses the synthetic fixtures; real pilot study data replaces these.
        """
        tps_a = transparency_perception_score(SURVEY_CONDITION_A).value
        tps_c = transparency_perception_score(SURVEY_CONDITION_C).value
        print(f"\n[RQ3 Delta] TPS_C ({tps_c:.3f}) > TPS_A ({tps_a:.3f}): {tps_c > tps_a}")
        assert tps_c > tps_a, (
            f"Expected full-stack TPS ({tps_c:.3f}) to exceed SHAP-only TPS ({tps_a:.3f})"
        )

    def test_full_stack_tci_higher_than_no_xai(self):
        """
        TCI should be higher with full XAI than without.
        Directly tests the Takayanagi et al. [7] prediction.
        """
        tci_no_xai = trust_calibration_index(
            TCI_NO_XAI["trust"], TCI_NO_XAI["quality"]
        ).value
        tci_full = trust_calibration_index(
            TCI_FULL["trust"], TCI_FULL["quality"]
        ).value
        print(
            f"\n[RQ3 TCI] No-XAI={tci_no_xai:.3f} "
            f"Full-stack={tci_full:.3f} "
            f"Improvement={tci_full - tci_no_xai:+.3f}"
        )
        assert tci_full > tci_no_xai, (
            f"Expected full-stack TCI ({tci_full:.3f}) > "
            f"no-XAI TCI ({tci_no_xai:.3f})"
        )
