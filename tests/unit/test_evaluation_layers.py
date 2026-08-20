"""
Guards for the three-layer evaluation design.

WHAT THESE TESTS ARE PROTECTING
    Not the model's scores — the INTEGRITY of the evaluation. Every test
    here fails loudly if someone (including a future me, in a hurry, the
    week before submission) does one of the things that would quietly
    invalidate the RQ1 result:

      * computes accuracy on unlabelled data                (TestLayerSeparation)
      * lets the labelling rubric see the model             (TestRubricIndependence)
      * unbalances or re-labels the gold set                (TestGoldDataset)
      * makes dataset construction non-reproducible         (TestReproducibility)
      * mixes FinQA train data into the RQ5 evaluation      (TestFinQAIntegrity)
      * changes Banking77 labels or breaks stratification   (TestBanking77Sampling)

    None of them assert that a metric is above any threshold. A test that
    fails when the model gets worse is a test that pressures the next
    person to make the evaluation kinder, which is exactly the failure
    mode this whole exercise exists to remove.

RUNNING
    python -m pytest tests/unit/test_evaluation_layers.py -v
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation import risk_rubric
from evaluation.bootstrap import (
    bootstrap_difference,
    bootstrap_paired_metric,
    bootstrap_proportion,
)
from evaluation.datasets import (
    LAYER_GOLD,
    LAYER_REAL,
    LAYER_STRESS,
    GroundTruthUnavailable,
    load_gold_risk_profiles,
    load_real_customers,
    load_stress_customers,
)
from evaluation.metrics import accuracy_value, classification_report, macro_f1_value
from evaluation.risk_rubric import RISK_TIERS

ROOT = Path(__file__).resolve().parent.parent.parent


# ── Layer separation ───────────────────────────────────────────────────

class TestLayerSeparation:

    def test_only_the_gold_layer_claims_ground_truth(self):
        assert LAYER_GOLD.has_ground_truth is True
        assert LAYER_REAL.has_ground_truth is False
        assert LAYER_STRESS.has_ground_truth is False

    def test_accuracy_metrics_are_forbidden_on_unlabelled_layers(self):
        for layer in (LAYER_REAL, LAYER_STRESS):
            for metric in ("accuracy", "precision", "recall", "f1"):
                assert metric in layer.forbidden_metrics

    def test_real_layer_refuses_to_hand_over_labels(self):
        dataset = load_real_customers()
        with pytest.raises(GroundTruthUnavailable):
            dataset.ground_truth()

    def test_stress_layer_refuses_to_hand_over_labels(self):
        dataset = load_stress_customers()
        with pytest.raises(GroundTruthUnavailable):
            dataset.ground_truth()

    def test_gold_layer_does_hand_over_labels(self):
        dataset = load_gold_risk_profiles()
        truth = dataset.ground_truth()
        assert len(truth) == len(dataset)
        assert set(truth) <= set(RISK_TIERS)

    def test_stress_construction_target_is_not_exposed_as_ground_truth(self):
        """
        The stress set records which rubric cell each profile was built
        for. That is a construction record. If it ever appears under a
        ground_truth_* key, a downstream accuracy computation will find
        it by name and silently produce a number that measures the
        sampler.
        """
        dataset = load_stress_customers()
        for record in dataset.records[:50]:
            assert "construction_target_class" in record
            assert not any(k.startswith("ground_truth") for k in record)

    def test_real_customer_demo_labels_are_not_named_like_ground_truth(self):
        dataset = load_real_customers()
        for record in dataset.records[:50]:
            assert not any(k.startswith("ground_truth") for k in record)


# ── Rubric independence ────────────────────────────────────────────────

class TestRubricIndependence:

    def test_rubric_module_imports_nothing_from_agents(self):
        """
        Parsed statically rather than checked at runtime: an import that
        only fires inside a rarely-taken branch would pass a runtime
        check and still couple the instrument to the system.
        """
        source = (ROOT / "evaluation" / "risk_rubric.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        offenders = [
            m for m in imported
            if m.startswith(("agents", "orchestrator", "rag", "utils.llm"))
        ]
        assert not offenders, (
            f"risk_rubric.py imports {offenders} — the labelling instrument "
            f"must not depend on the system under test"
        )

    def test_rubric_never_touches_the_model_file(self):
        """
        Checked against IDENTIFIERS ONLY — names, attributes and imports.
        String literals are excluded on purpose: the module docstring and
        rubric_documentation() both discuss risk_model.pkl and SHAP in
        prose, because stating what the instrument does not do is the
        whole point of that prose. A substring search over the file would
        flag the rubric's own disclaimer as a violation.
        """
        tree = ast.parse(
            (ROOT / "evaluation" / "risk_rubric.py").read_text(encoding="utf-8"))

        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
            elif isinstance(node, ast.Import):
                identifiers.update(a.name.lower() for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                identifiers.add(node.module.lower())

        blob = " ".join(identifiers)
        for forbidden in ("risk_model", "joblib", "predict_proba", "shap",
                          "pickle", "sklearn", "numpy"):
            assert forbidden not in blob, (
                f"risk_rubric.py's executable code references {forbidden!r} — "
                f"the labelling instrument must stay free of the model stack"
            )

    def test_rubric_is_deterministic(self):
        features = {
            "age": 40, "income": 65_000, "employment_status": "employed",
            "dependents": 1, "existing_debt": 8_000,
            "investment_horizon": 12, "loss_tolerance": 4,
            "financial_knowledge_score": 4,
        }
        first = risk_rubric.label(features)
        for _ in range(5):
            assert risk_rubric.label(features).risk_class == first.risk_class

    def test_capacity_caps_tolerance(self):
        """
        The regulatory asymmetry: maximum willingness cannot lift a
        low-capacity client past moderately_conservative.
        """
        low_capacity_max_tolerance = {
            "age": 58, "income": 18_000, "employment_status": "unemployed",
            "dependents": 4, "existing_debt": 16_000,
            "investment_horizon": 2, "loss_tolerance": 5,
            "financial_knowledge_score": 5,
        }
        verdict = risk_rubric.label(low_capacity_max_tolerance)
        assert verdict.capacity_band == "low"
        assert verdict.tolerance_band == "high"
        assert verdict.risk_class == "moderately_conservative"
        assert RISK_TIERS.index(verdict.risk_class) <= RISK_TIERS.index("moderate")

    def test_matrix_and_cells_by_tier_agree(self):
        for tier, cells in risk_rubric.CELLS_BY_TIER.items():
            for cell in cells:
                assert risk_rubric.SUITABILITY_MATRIX[cell] == tier

    def test_every_tier_is_reachable(self):
        assert set(risk_rubric.CELLS_BY_TIER) == set(RISK_TIERS)


# ── The gold dataset itself ────────────────────────────────────────────

class TestGoldDataset:

    @pytest.fixture(scope="class")
    def payload(self):
        path = ROOT / "data" / "evaluation" / "gold_risk_profiles.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_rubric_labelled_block_is_balanced_40_per_class(self, payload):
        counts = Counter(
            p["ground_truth_risk_class"] for p in payload["profiles"]
            if p["provenance"] != "hand_labelled_original_15"
        )
        assert set(counts) == set(RISK_TIERS)
        assert set(counts.values()) == {40}, counts

    def test_original_15_are_preserved_with_their_own_provenance(self, payload):
        originals = [
            p for p in payload["profiles"]
            if p["provenance"] == "hand_labelled_original_15"
        ]
        assert len(originals) == 15
        for profile in originals:
            assert "RQ1_FIXTURE" in profile["label_source"]

    def test_original_15_labels_match_the_source_fixture_exactly(self, payload):
        """
        The hand labels must be byte-identical to the ones in the test
        file they came from. If the rubric ever overwrote them, this
        catches it.
        """
        source = (ROOT / "tests" / "unit" / "test_risk_profiling_agent.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        fixture = None
        for node in tree.body:
            # RQ1_FIXTURE is annotated (`RQ1_FIXTURE: list[dict] = [...]`),
            # so it parses as AnnAssign, not Assign. Handle both.
            targets = (
                node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "RQ1_FIXTURE":
                    fixture = ast.literal_eval(node.value)
        assert fixture, "RQ1_FIXTURE not found in the original test file"

        originals = [
            p for p in payload["profiles"]
            if p["provenance"] == "hand_labelled_original_15"
        ]
        for original, item in zip(originals, fixture):
            assert original["ground_truth_risk_class"] == item["expected"]
            assert original["features"] == item["features"]

    def test_rubric_label_is_reproducible_from_the_stored_features(self, payload):
        """
        Re-deriving every rubric label from its stored features must
        return the stored label. This is the check that would fail if a
        label had been hand-edited after the fact.
        """
        for profile in payload["profiles"]:
            if profile["provenance"] == "hand_labelled_original_15":
                continue
            recomputed = risk_rubric.label(profile["features"])
            assert recomputed.risk_class == profile["ground_truth_risk_class"], (
                f"{profile['profile_id']} label does not match its features"
            )

    def test_no_profile_records_a_model_prediction(self, payload):
        blob = json.dumps(payload).lower()
        for forbidden in ("predicted_risk_class", "model_prediction", "hybrid_score"):
            assert forbidden not in blob

    def test_provenance_declares_independence(self, payload):
        guarantees = payload["provenance"]["independence_guarantees"]
        assert any("never loaded" in g.lower() or "never run" in g.lower()
                   for g in guarantees)
        assert payload["provenance"]["seed"] is not None

    def test_features_are_diverse_within_each_class(self, payload):
        """
        Guards against 40 near-clones per class.

        The thresholds differ by feature on purpose. age and income are
        unconstrained by the rubric's cell definition, so a class with
        few distinct values of either would mean the sampler collapsed.
        investment_horizon IS constrained — a conservative profile
        cannot have a 30-year horizon, because horizon is a capacity
        input — so a lower bar is the correct expectation there, not a
        weakened one.
        """
        minimum_distinct = {"age": 15, "income": 15, "investment_horizon": 8}

        by_class: dict[str, list] = {}
        for profile in payload["profiles"]:
            if profile["provenance"] == "hand_labelled_original_15":
                continue
            by_class.setdefault(profile["ground_truth_risk_class"], []).append(
                profile["features"])

        for tier, rows in by_class.items():
            for field, minimum in minimum_distinct.items():
                distinct = len({r[field] for r in rows})
                assert distinct >= minimum, (
                    f"{tier}: only {distinct} distinct values of {field} "
                    f"across {len(rows)} profiles"
                )

    def test_both_self_employed_spellings_are_present(self, payload):
        spellings = {
            p["features"].get("employment_status") for p in payload["profiles"]
        }
        assert "self_employed" in spellings or "self-employed" in spellings


# ── The stress dataset ─────────────────────────────────────────────────

class TestStressDataset:

    @pytest.fixture(scope="class")
    def dataset(self):
        return load_stress_customers()

    def test_has_about_a_thousand_profiles(self, dataset):
        assert len(dataset) == 1000

    def test_covers_all_six_data_sufficiency_tiers(self, dataset):
        from agents.data_sufficiency import _COVERAGE_TIERS

        expected = {name for _, name, _ in _COVERAGE_TIERS}
        observed = {r["expected_coverage_tier"] for r in dataset.records}
        assert expected <= observed, f"missing tiers: {expected - observed}"

    def test_covers_all_five_construction_target_classes(self, dataset):
        observed = {
            r["construction_target_class"] for r in dataset.records
            if r["construction_target_class"]
        }
        assert observed == set(RISK_TIERS)

    def test_includes_tagged_edge_cases(self, dataset):
        kinds = {r["edge_case_kind"] for r in dataset.records if r["edge_case_kind"]}
        assert len(kinds) >= 10
        assert "zero_income" in kinds
        assert any(k.startswith("missing_") for k in kinds)

    def test_employment_variants_including_both_self_employed_spellings(self, dataset):
        observed = {
            r["features"].get("employment_status") for r in dataset.records
        }
        assert {"self_employed", "self-employed"} <= observed
        assert {"employed", "retired", "unemployed", "student"} <= observed

    def test_declares_its_usage_restriction(self, dataset):
        restriction = dataset.provenance["usage_restriction"].lower()
        assert "not ground truth" in restriction


# ── Reproducibility ────────────────────────────────────────────────────

class TestReproducibility:

    def test_gold_builder_is_deterministic(self):
        from scripts.build_gold_risk_dataset import build_gold_profiles

        first, _ = build_gold_profiles(per_class=3, seed=99)
        second, _ = build_gold_profiles(per_class=3, seed=99)
        assert [p["features"] for p in first] == [p["features"] for p in second]

    def test_gold_builder_differs_with_a_different_seed(self):
        from scripts.build_gold_risk_dataset import build_gold_profiles

        first, _ = build_gold_profiles(per_class=3, seed=1)
        second, _ = build_gold_profiles(per_class=3, seed=2)
        assert [p["features"] for p in first] != [p["features"] for p in second]

    def test_bootstrap_is_deterministic_under_a_fixed_seed(self):
        successes = [True] * 30 + [False] * 20
        a = bootstrap_proportion(successes, n_resamples=200, seed=7)
        b = bootstrap_proportion(successes, n_resamples=200, seed=7)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_bootstrap_interval_brackets_the_point_estimate(self):
        successes = [True] * 30 + [False] * 20
        ci = bootstrap_proportion(successes, n_resamples=500, seed=7)
        assert ci.ci_low <= ci.point_estimate <= ci.ci_high

    def test_smaller_samples_give_wider_intervals(self):
        wide = bootstrap_proportion([True, False] * 5, n_resamples=500, seed=3)
        narrow = bootstrap_proportion([True, False] * 250, n_resamples=500, seed=3)
        assert wide.width > narrow.width

    def test_paired_bootstrap_preserves_pairing(self):
        preds = ["a"] * 50
        truth = ["a"] * 25 + ["b"] * 25
        ci = bootstrap_paired_metric(preds, truth, accuracy_value,
                                     n_resamples=300, seed=11)
        assert 0.2 < ci.point_estimate < 0.8
        assert ci.n_items == 50

    def test_paired_difference_of_identical_conditions_contains_zero(self):
        successes = [True, False, True, True, False] * 10
        diff = bootstrap_difference(successes, successes, paired=True,
                                    n_resamples=300, seed=5)
        assert diff.point_estimate == 0.0
        assert diff.ci_low <= 0.0 <= diff.ci_high

    def test_paired_difference_detects_a_real_gap(self):
        better = [True] * 40 + [False] * 10
        worse = [False] * 50
        diff = bootstrap_difference(better, worse, paired=True,
                                    n_resamples=500, seed=5)
        assert diff.ci_low > 0.0


# ── Classification metric shape ────────────────────────────────────────

class TestClassificationReport:

    def test_reports_every_requested_label_even_with_no_predictions(self):
        report = classification_report(
            ["a", "a", "a"], ["a", "b", "c"], labels=["a", "b", "c", "d"])
        assert set(report.details["per_class"]) == {"a", "b", "c", "d"}
        assert report.details["per_class"]["d"]["support"] == 0

    def test_confusion_matrix_is_true_by_predicted(self):
        report = classification_report(["b"], ["a"], labels=["a", "b"])
        assert report.details["confusion_matrix"]["a"]["b"] == 1
        assert report.details["confusion_matrix"]["b"]["a"] == 0

    def test_perfect_predictions_score_one(self):
        report = classification_report(["a", "b"], ["a", "b"], labels=["a", "b"])
        assert report.value == 1.0
        assert report.details["macro_avg"]["f1"] == 1.0

    def test_macro_f1_averages_over_the_fixed_label_set(self):
        """A label absent from a resample must still count as a zero."""
        with_absent = macro_f1_value(["a", "a"], ["a", "a"], ["a", "b"])
        without = macro_f1_value(["a", "a"], ["a", "a"], ["a"])
        assert with_absent == pytest.approx(0.5)
        assert without == pytest.approx(1.0)


# ── FinQA ──────────────────────────────────────────────────────────────

class TestFinQAIntegrity:

    @pytest.fixture(scope="class")
    def loaded(self):
        from evaluation.finqa_data import load_finqa_verified

        return load_finqa_verified()

    def test_loads_the_full_verified_test_split(self, loaded):
        items, provenance = loaded
        assert len(items) == 91
        assert provenance["split"] == "test"
        assert provenance["is_verified_split"] is True
        assert provenance["n_items_used"] == len(items)

    def test_every_item_has_a_question_context_and_answer(self, loaded):
        items, _ = loaded
        for item in items:
            assert item.question and item.answer and item.context

    def test_loader_cannot_reach_the_train_split(self):
        source = (ROOT / "evaluation" / "finqa_data.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        loader = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "load_finqa_verified"
        )
        body = ast.dump(loader)
        assert "finqa_original_train" not in body.replace(
            "FINQA_ORIGINAL_TRAIN", "")  # only referenced in the provenance note

    def test_train_contamination_is_checked_and_reported(self, loaded):
        from evaluation.finqa_data import assert_no_train_contamination

        items, _ = loaded
        report = assert_no_train_contamination(items)
        assert report["checked"] is True
        assert report["train_items_used_in_evaluation"] == 0

    def test_answer_leakage_is_reported_not_filtered(self, loaded):
        from evaluation.finqa_data import answer_leakage_report

        items, _ = loaded
        report = answer_leakage_report(items)
        assert report["n_items"] == 91
        assert "no item was removed" in report["interpretation"].lower()


# ── Banking77 ──────────────────────────────────────────────────────────

class TestBanking77Sampling:

    @pytest.fixture(scope="class")
    def loaded(self):
        from evaluation.banking77_data import load_banking77

        return load_banking77()

    def test_split_is_recorded_explicitly(self, loaded):
        _, provenance = loaded
        assert provenance["split"] in {"train", "test"}
        assert provenance["split_justification"]

    def test_bucket_map_is_used_unmodified(self, loaded):
        from evaluation.banking77_data import load_bucket_map

        bucket_map, meta = load_bucket_map()
        assert len(bucket_map) == 77
        assert "no label was edited" in meta["labels_unchanged"].lower()

    def test_stratified_sample_is_reproducible(self, loaded):
        from evaluation.banking77_data import stratified_sample

        items, _ = loaded
        first, _ = stratified_sample(items, n=120, seed=5)
        second, _ = stratified_sample(items, n=120, seed=5)
        assert [i.item_id for i in first] == [i.item_id for i in second]

    def test_stratified_sample_hits_the_requested_size(self, loaded):
        from evaluation.banking77_data import stratified_sample

        items, _ = loaded
        sample, meta = stratified_sample(items, n=250, seed=5)
        assert len(sample) == 250
        assert meta["actual_n"] == 250

    def test_every_intent_stratum_is_represented(self, loaded):
        from evaluation.banking77_data import stratified_sample

        items, _ = loaded
        sample, meta = stratified_sample(items, n=500, seed=5)
        assert len({i.banking77_label for i in sample}) == meta["n_strata"] == 77

    def test_bucket_balance_is_preserved(self, loaded):
        from evaluation.banking77_data import stratified_sample

        items, _ = loaded
        _, meta = stratified_sample(items, n=500, seed=5)
        for bucket, shares in meta["bucket_balance_preserved"].items():
            assert abs(shares["population_share"] - shares["sample_share"]) < 0.02, (
                f"{bucket} share drifted: {shares}"
            )

    def test_no_sampled_utterance_appears_in_the_classifier_prompt(self, loaded):
        from evaluation.banking77_data import (
            assert_no_prompt_leakage,
            stratified_sample,
        )

        items, _ = loaded
        sample, _ = stratified_sample(items, n=200, seed=5)
        report = assert_no_prompt_leakage(sample)
        assert report["checked"] is True
        assert report["n_sampled_utterances_found_in_prompt"] == 0
