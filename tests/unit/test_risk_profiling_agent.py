"""
Phase 3 — RiskProfilingAgent evaluation.

Four test groups:

GROUP A: Hybrid scoring unit tests (no LLM, no API key)
  Tests the pure arithmetic of _ml_score(), _rule_score(), _hybrid_score(),
  _score_to_class(), _compute_confidence(). These are the most important
  tests in Phase 3 — RQ1 is about scoring accuracy, not LLM output quality.

GROUP B: SHAP proxy unit tests (no LLM)
  Verifies feature attribution directions are correct.
  Positive features (loss_tolerance, horizon) must produce positive impact.
  Negative features (age, debt) must produce negative impact.

GROUP C: Missing feature handling
  Agent must return status="incomplete" before classifying with missing data.
  This enforces the data collection loop with ConversationalAgent.

GROUP D: RQ1 evaluation — hybrid scoring on test fixture
  Runs hybrid scoring on 15 hand-labelled profiles.
  Generates: RAR, F1, and hybrid-vs-rule-only delta.
  Writes results to results/phase3_risk_baseline.json.

GROUP E: Constraint checker tests
  Verifies FinancialConstraints catches return fabrications,
  incompatible products, and prohibited language.

RUNNING:
  pytest tests/unit/test_risk_profiling_agent.py -v
  pytest tests/unit/test_risk_profiling_agent.py -v -s   (shows eval scores)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.risk_profiling_agent import RiskProfilingAgent
from config.constraints import FinancialConstraints, financial_constraints
from evaluation.metrics import (
    EvalResult,
    f1_risk_classification,
    hybrid_vs_rule_only_delta,
    risk_alignment_rate,
)
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


# Helpers

def make_agent() -> RiskProfilingAgent:
    """Risk agent in mock mode — scoring logic requires no API key."""
    client = LLMClient()
    return RiskProfilingAgent(client)


FULL_FEATURES = {
    "age": 35,
    "income": 60000,
    "employment_status": "employed",
    "dependents": 0,
    "existing_debt": 5000,
    "investment_horizon": 10,
    "loss_tolerance": 3,
    "financial_knowledge_score": 3,
}


# GROUP A: Hybrid scoring unit tests

class TestMLScore:

    def test_returns_float_in_range(self):
        agent = make_agent()
        score = agent._ml_score(FULL_FEATURES)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_tolerance_gives_higher_score(self):
        agent = make_agent()
        low = {**FULL_FEATURES, "loss_tolerance": 1}
        high = {**FULL_FEATURES, "loss_tolerance": 5}
        assert agent._ml_score(high) > agent._ml_score(low)

    def test_high_horizon_gives_higher_score(self):
        agent = make_agent()
        short = {**FULL_FEATURES, "investment_horizon": 1}
        long_ = {**FULL_FEATURES, "investment_horizon": 30}
        assert agent._ml_score(long_) > agent._ml_score(short)

    def test_high_debt_ratio_gives_lower_score(self):
        agent = make_agent()
        low_debt = {**FULL_FEATURES, "existing_debt": 0}
        high_debt = {**FULL_FEATURES, "existing_debt": 55000}
        assert agent._ml_score(high_debt) < agent._ml_score(low_debt)

    def test_older_age_gives_lower_score(self):
        agent = make_agent()
        young = {**FULL_FEATURES, "age": 25}
        old = {**FULL_FEATURES, "age": 65}
        assert agent._ml_score(young) > agent._ml_score(old)

    def test_score_clips_to_zero_minimum(self):
        agent = make_agent()
        extreme = {**FULL_FEATURES, "loss_tolerance": 1, "age": 80,
                   "existing_debt": 200000, "income": 10000,
                   "investment_horizon": 1}
        assert agent._ml_score(extreme) >= 0.0

    def test_score_clips_to_one_maximum(self):
        agent = make_agent()
        extreme = {**FULL_FEATURES, "loss_tolerance": 5, "age": 20,
                   "existing_debt": 0, "investment_horizon": 40,
                   "financial_knowledge_score": 5}
        assert agent._ml_score(extreme) <= 1.0


class TestRuleScore:

    def test_returns_float_in_range(self):
        agent = make_agent()
        score = agent._rule_score(FULL_FEATURES)
        assert 0.0 <= score <= 1.0

    def test_unemployed_gets_lower_score(self):
        agent = make_agent()
        employed = {**FULL_FEATURES, "employment_status": "employed"}
        unemployed = {**FULL_FEATURES, "employment_status": "unemployed"}
        assert agent._rule_score(unemployed) < agent._rule_score(employed)

    def test_high_debt_to_income_lowers_score(self):
        agent = make_agent()
        low_debt = {**FULL_FEATURES, "existing_debt": 1000, "income": 60000}
        high_debt = {**FULL_FEATURES, "existing_debt": 40000, "income": 60000}
        assert agent._rule_score(high_debt) < agent._rule_score(low_debt)

    def test_dependents_lower_score(self):
        agent = make_agent()
        no_dep = {**FULL_FEATURES, "dependents": 0}
        many_dep = {**FULL_FEATURES, "dependents": 4}
        assert agent._rule_score(many_dep) < agent._rule_score(no_dep)

    def test_low_income_lowers_score(self):
        agent = make_agent()
        high_income = {**FULL_FEATURES, "income": 100000}
        low_income = {**FULL_FEATURES, "income": 20000}
        assert agent._rule_score(low_income) < agent._rule_score(high_income)

    def test_rule_score_clips_to_bounds(self):
        agent = make_agent()
        extreme = {**FULL_FEATURES, "employment_status": "unemployed",
                   "dependents": 5, "existing_debt": 100000, "income": 15000}
        score = agent._rule_score(extreme)
        assert 0.0 <= score <= 1.0


class TestHybridScore:

    def test_hybrid_is_weighted_combination(self):
        agent = make_agent()
        ml, rule, hybrid = agent._hybrid_score(FULL_FEATURES)
        expected = 0.6 * ml + 0.4 * rule
        assert abs(hybrid - expected) < 1e-9

    def test_hybrid_in_range(self):
        agent = make_agent()
        _, _, hybrid = agent._hybrid_score(FULL_FEATURES)
        assert 0.0 <= hybrid <= 1.0

    def test_weights_sum_to_one(self):
        from config.settings import settings
        assert abs(settings.risk.ml_weight + settings.risk.rule_weight - 1.0) < 1e-9


class TestScoreToClass:

    def test_low_score_gives_conservative(self):
        agent = make_agent()
        assert agent._score_to_class(0.05) == "conservative"

    def test_mid_score_gives_moderate(self):
        agent = make_agent()
        assert agent._score_to_class(0.5) == "moderate"

    def test_high_score_gives_aggressive(self):
        agent = make_agent()
        assert agent._score_to_class(0.95) == "aggressive"

    def test_all_five_classes_reachable(self):
        agent = make_agent()
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        classes = [agent._score_to_class(s) for s in scores]
        assert len(set(classes)) == 5

    def test_boundary_scores(self):
        agent = make_agent()
        # Just below 0.2 → conservative; at 0.2 → moderately_conservative
        assert agent._score_to_class(0.19) == "conservative"
        assert agent._score_to_class(0.20) == "moderately_conservative"


class TestConfidence:

    def test_midpoint_gives_high_confidence(self):
        agent = make_agent()
        # 0.5 is exactly the centre of 'moderate' tier — max confidence
        conf = agent._compute_confidence(0.5)
        assert conf == 1.0

    def test_boundary_gives_low_confidence(self):
        agent = make_agent()
        # 0.4 is exactly on a boundary — minimum confidence
        conf = agent._compute_confidence(0.4)
        assert conf == 0.0

    def test_confidence_in_range(self):
        agent = make_agent()
        for score in np.linspace(0.0, 1.0, 20):
            conf = agent._compute_confidence(float(score))
            assert 0.0 <= conf <= 1.0


# GROUP B: SHAP proxy unit tests

class TestSHAPProxy:

    def test_returns_dict_with_required_keys(self):
        agent = make_agent()
        shap = agent._compute_shap_proxy(FULL_FEATURES, 0.5)
        assert isinstance(shap, dict)
        assert len(shap) > 0

    def test_each_entry_has_value_and_impact(self):
        agent = make_agent()
        shap = agent._compute_shap_proxy(FULL_FEATURES, 0.5)
        for feat, info in shap.items():
            assert "value" in info, f"Missing 'value' for {feat}"
            assert "shap_impact" in info, f"Missing 'shap_impact' for {feat}"

    def test_loss_tolerance_positive_impact(self):
        agent = make_agent()
        high_tol = {**FULL_FEATURES, "loss_tolerance": 5}
        shap = agent._compute_shap_proxy(high_tol, 0.7)
        assert shap["loss_tolerance"]["shap_impact"] > 0

    def test_high_debt_negative_impact(self):
        agent = make_agent()
        high_debt = {**FULL_FEATURES, "existing_debt": 50000, "income": 60000}
        shap = agent._compute_shap_proxy(high_debt, 0.3)
        assert shap["existing_debt"]["shap_impact"] < 0

    def test_shap_impact_values_are_finite(self):
        agent = make_agent()
        shap = agent._compute_shap_proxy(FULL_FEATURES, 0.5)
        for feat, info in shap.items():
            assert not np.isnan(info["shap_impact"]), f"NaN impact for {feat}"
            assert not np.isinf(info["shap_impact"]), f"Inf impact for {feat}"


# GROUP C: Missing feature handling

class TestMissingFeatures:

    def test_missing_features_detected(self):
        agent = make_agent()
        incomplete = {"age": 35, "income": 60000}
        missing = agent._check_missing_features(incomplete)
        assert len(missing) > 0
        assert "loss_tolerance" in missing

    def test_complete_features_returns_empty_list(self):
        agent = make_agent()
        missing = agent._check_missing_features(FULL_FEATURES)
        assert missing == []

    def test_run_returns_incomplete_status_on_missing_features(self):
        agent = make_agent()
        result = agent.run({"user_features": {"age": 35}})
        assert result.success is False
        assert result.payload["status"] == "incomplete"
        assert "missing_features" in result.payload
        assert len(result.payload["missing_features"]) > 0

    def test_run_returns_complete_status_on_full_features(self):
        agent = make_agent()
        result = agent.run({"user_features": FULL_FEATURES})
        assert result.success is True
        assert result.payload["status"] == "complete"

    def test_result_has_all_required_payload_keys(self):
        agent = make_agent()
        result = agent.run({"user_features": FULL_FEATURES})
        for key in ("status", "risk_class", "ml_score", "rule_score",
                    "hybrid_score", "confidence", "feature_importance", "rationale"):
            assert key in result.payload, f"Missing key: {key}"

    def test_risk_class_is_valid_tier(self):
        agent = make_agent()
        from config.settings import settings
        result = agent.run({"user_features": FULL_FEATURES})
        assert result.payload["risk_class"] in settings.risk.risk_classes

    def test_scores_are_in_range(self):
        agent = make_agent()
        result = agent.run({"user_features": FULL_FEATURES})
        p = result.payload
        assert 0.0 <= p["ml_score"] <= 1.0
        assert 0.0 <= p["rule_score"] <= 1.0
        assert 0.0 <= p["hybrid_score"] <= 1.0
        assert 0.0 <= p["confidence"] <= 1.0

    def test_routing_context_has_risk_class(self):
        agent = make_agent()
        result = agent.run({"user_features": FULL_FEATURES})
        assert "risk_class" in result.routing_context

    def test_step_record_structure(self):
        agent = make_agent()
        result = agent.run({"user_features": FULL_FEATURES})
        step = result.to_step_record()
        assert step["completed"] is True
        assert step["agent"] == "RiskProfilingAgent"


# GROUP D: RQ1 evaluation on hand-labelled fixture

# 15 hand-labelled profiles with expected risk class.
# Ground truth assigned by applying CBI suitability rules manually.
# Profiles span the full range of the five risk tiers.
RQ1_FIXTURE: list[dict] = [
    # Conservative profiles
    {"features": {"age": 68, "income": 28000, "employment_status": "retired",
                  "dependents": 0, "existing_debt": 2000, "investment_horizon": 3,
                  "loss_tolerance": 1, "financial_knowledge_score": 2},
     "expected": "conservative"},

    {"features": {"age": 55, "income": 35000, "employment_status": "employed",
                  "dependents": 3, "existing_debt": 25000, "investment_horizon": 2,
                  "loss_tolerance": 1, "financial_knowledge_score": 1},
     "expected": "conservative"},

    {"features": {"age": 62, "income": 22000, "employment_status": "part_time",
                  "dependents": 1, "existing_debt": 15000, "investment_horizon": 3,
                  "loss_tolerance": 2, "financial_knowledge_score": 2},
     "expected": "conservative"},

    # Moderately conservative profiles
    {"features": {"age": 50, "income": 48000, "employment_status": "employed",
                  "dependents": 2, "existing_debt": 10000, "investment_horizon": 6,
                  "loss_tolerance": 2, "financial_knowledge_score": 2},
     "expected": "moderately_conservative"},

    {"features": {"age": 45, "income": 55000, "employment_status": "employed",
                  "dependents": 1, "existing_debt": 12000, "investment_horizon": 7,
                  "loss_tolerance": 2, "financial_knowledge_score": 3},
     "expected": "moderately_conservative"},

    # Moderate profiles
    {"features": {"age": 38, "income": 65000, "employment_status": "employed",
                  "dependents": 1, "existing_debt": 8000, "investment_horizon": 10,
                  "loss_tolerance": 3, "financial_knowledge_score": 3},
     "expected": "moderate"},

    {"features": {"age": 42, "income": 72000, "employment_status": "employed",
                  "dependents": 2, "existing_debt": 15000, "investment_horizon": 8,
                  "loss_tolerance": 3, "financial_knowledge_score": 3},
     "expected": "moderate"},

    {"features": {"age": 35, "income": 60000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 5000, "investment_horizon": 10,
                  "loss_tolerance": 3, "financial_knowledge_score": 3},
     "expected": "moderate"},

    # Moderately aggressive profiles
    {"features": {"age": 30, "income": 85000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
                  "loss_tolerance": 4, "financial_knowledge_score": 4},
     "expected": "moderately_aggressive"},

    {"features": {"age": 33, "income": 95000, "employment_status": "employed",
                  "dependents": 1, "existing_debt": 8000, "investment_horizon": 12,
                  "loss_tolerance": 4, "financial_knowledge_score": 4},
     "expected": "moderately_aggressive"},

    {"features": {"age": 28, "income": 78000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 3000, "investment_horizon": 20,
                  "loss_tolerance": 4, "financial_knowledge_score": 3},
     "expected": "moderately_aggressive"},

    # Aggressive profiles
    {"features": {"age": 25, "income": 120000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 0, "investment_horizon": 30,
                  "loss_tolerance": 5, "financial_knowledge_score": 5},
     "expected": "aggressive"},

    {"features": {"age": 27, "income": 100000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 2000, "investment_horizon": 25,
                  "loss_tolerance": 5, "financial_knowledge_score": 5},
     "expected": "aggressive"},

    {"features": {"age": 29, "income": 110000, "employment_status": "self_employed",
                  "dependents": 0, "existing_debt": 1000, "investment_horizon": 20,
                  "loss_tolerance": 5, "financial_knowledge_score": 4},
     "expected": "aggressive"},

    {"features": {"age": 31, "income": 90000, "employment_status": "employed",
                  "dependents": 0, "existing_debt": 4000, "investment_horizon": 22,
                  "loss_tolerance": 5, "financial_knowledge_score": 5},
     "expected": "aggressive"},
]


class TestRQ1Evaluation:

    def test_rar_metric_runs(self):
        result = risk_alignment_rate(["moderate", "conservative"],
                                     ["moderate", "conservative"])
        assert isinstance(result, EvalResult)
        assert result.value == 1.0

    def test_rar_one_tier_off_counts_as_aligned(self):
        result = risk_alignment_rate(["moderately_conservative"], ["conservative"])
        assert result.value == 1.0

    def test_rar_two_tiers_off_not_aligned(self):
        result = risk_alignment_rate(["aggressive"], ["conservative"])
        assert result.value == 0.0

    def test_f1_perfect(self):
        preds = ["moderate", "conservative", "aggressive"]
        gold = ["moderate", "conservative", "aggressive"]
        result = f1_risk_classification(preds, gold)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_f1_all_wrong(self):
        preds = ["conservative", "conservative"]
        gold = ["aggressive", "moderate"]
        result = f1_risk_classification(preds, gold)
        assert result.value < 0.5

    def test_rq1_full_fixture_evaluation_and_write_results(self):
        """
        Full RQ1 evaluation on 15-profile fixture.
        Scoring is deterministic (no LLM involved) so results are
        meaningful in both mock and real mode.
        Writes: results/phase3_risk_baseline.json
        """
        agent = make_agent()

        hybrid_preds = []
        rule_only_preds = []
        ground_truth = []

        for item in RQ1_FIXTURE:
            features = item["features"]
            expected = item["expected"]

            # Hybrid prediction
            ml, rule, hybrid = agent._hybrid_score(features)
            hybrid_class = agent._score_to_class(hybrid)
            hybrid_preds.append(hybrid_class)

            # Rule-only prediction (for ablation comparison)
            rule_class = agent._score_to_class(rule)
            rule_only_preds.append(rule_class)

            ground_truth.append(expected)

        # Compute all RQ1 metrics
        rar_result = risk_alignment_rate(hybrid_preds, ground_truth)
        f1_result = f1_risk_classification(hybrid_preds, ground_truth)
        delta_result = hybrid_vs_rule_only_delta(
            hybrid_preds, rule_only_preds, ground_truth
        )

        # Write results for dissertation evidence
        RESULTS_DIR.mkdir(exist_ok=True)
        results_payload = {
            "phase": 3,
            "agent": "RiskProfilingAgent",
            "dataset": "hand_labelled_fixture_15_profiles",
            "model_mode": "heuristic_proxy" if agent._ml_model is None else "trained_model",
            "metrics": {
                "risk_alignment_rate": rar_result.to_dict(),
                "f1_macro": f1_result.to_dict(),
                "hybrid_vs_rule_only_delta": delta_result.to_dict(),
            },
            "per_profile": [
                {
                    "features": item["features"],
                    "expected": item["expected"],
                    "hybrid_pred": hp,
                    "rule_pred": rp,
                    "hybrid_correct": hp == item["expected"],
                }
                for item, hp, rp in zip(RQ1_FIXTURE, hybrid_preds, rule_only_preds)
            ],
        }
        results_path = RESULTS_DIR / "phase3_risk_baseline.json"
        with open(results_path, "w") as f:
            json.dump(results_payload, f, indent=2)

        print(f"\n[Phase 3 RQ1] Risk Alignment Rate: {rar_result.value:.3f}")
        print(f"[Phase 3 RQ1] Macro F1:            {f1_result.value:.3f}")
        print(f"[Phase 3 RQ1] Hybrid vs Rule delta: {delta_result.value:+.3f}")
        print(f"[Phase 3 RQ1] Results written to {results_path}")

        assert 0.0 <= rar_result.value <= 1.0
        assert 0.0 <= f1_result.value <= 1.0


# GROUP E: Constraint checker tests

class TestFinancialConstraints:

    def test_return_plausibility_pass(self):
        assert financial_constraints.check_return_plausibility(7.5) is None

    def test_return_plausibility_fail(self):
        v = financial_constraints.check_return_plausibility(50.0)
        assert v is not None
        assert v.rule_id == "R001"
        assert v.severity == "hard_block"

    def test_risk_product_compatible(self):
        v = financial_constraints.check_risk_product_compatibility(
            "conservative", "government_bond"
        )
        assert v is None

    def test_risk_product_incompatible(self):
        v = financial_constraints.check_risk_product_compatibility(
            "conservative", "individual_equity"
        )
        assert v is not None
        assert v.rule_id == "R002"
        assert v.severity == "hard_block"

    def test_prohibited_phrase_detected(self):
        violations = financial_constraints.check_prohibited_phrases(
            "This fund offers a guaranteed return of 10% annually."
        )
        assert any(v.rule_id == "R003" for v in violations)

    def test_prohibited_phrase_clean(self):
        violations = financial_constraints.check_prohibited_phrases(
            "This fund carries market risk and returns are not guaranteed."
        )
        assert len(violations) == 0

    def test_disclaimer_missing(self):
        violations = financial_constraints.check_disclaimers_present(
            "Buy this ETF."
        )
        assert len(violations) > 0
        assert all(v.rule_id == "R004" for v in violations)

    def test_disclaimer_present(self):
        text = (
            "This is not regulated financial advice. "
            "Consult a qualified advisor. "
            "Past performance is not indicative of future results."
        )
        violations = financial_constraints.check_disclaimers_present(text)
        assert len(violations) == 0

    def test_full_validation_hard_block(self):
        deliverable, violations = financial_constraints.validate_response(
            response_text="guaranteed return cannot lose money",
            risk_class="conservative",
            product_category="individual_equity",
            claimed_return=80.0,
        )
        assert deliverable is False
        assert any(v.severity == "hard_block" for v in violations)

    def test_full_validation_clean_response(self):
        text = (
            "This is not regulated financial advice. "
            "Consult a qualified advisor. "
            "Past performance is not indicative of future results. "
            "The government bond offers a stable but modest return."
        )
        deliverable, violations = financial_constraints.validate_response(
            response_text=text,
            risk_class="conservative",
            product_category="government_bond",
            claimed_return=3.5,
        )
        assert deliverable is True
        hard_blocks = [v for v in violations if v.severity == "hard_block"]
        assert len(hard_blocks) == 0
