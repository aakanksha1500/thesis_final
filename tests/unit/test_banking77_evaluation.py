"""
Banking77 bucket-classification evaluation — harness guards.

WHY THIS FILE HAD TO BE REWRITTEN
    In the snapshot this work started from, this file's contents were
    byte-identical to tests/unit/test_approval_gate.py (same md5, same
    409 lines, both dated 2026-08-11). At some point the Banking77
    evaluation harness was overwritten by a copy of the approval-gate
    tests, and the duplicate was then collected and run twice by pytest
    under two different names — which is why nothing failed and nobody
    noticed. Only the harness's outputs survived:
    results/mock/phase2_banking77_bucket_evaluation.json (n=200) and
    data/processed/banking77_eval_checkpoint.json.

    The rebuilt harness lives in scripts/eval_banking77_stratified.py,
    alongside the other evaluation entry points, with the reusable data
    handling in evaluation/banking77_data.py. This file now guards that
    harness rather than containing it — which also means an accidental
    overwrite of one is no longer an accidental overwrite of both.

WHAT IS TESTED HERE
    Sampling, provenance and metric plumbing. Not the model's score, and
    not anything requiring an API key: the live n=500 run is a script,
    invoked by scripts/regenerate_evidence.py, and costs one LLM call per
    utterance.

    The deeper sampling and integrity properties (reproducibility, per
    stratum coverage, bucket-balance preservation, prompt leakage) are in
    tests/unit/test_evaluation_layers.py::TestBanking77Sampling.

RUNNING
    python -m pytest tests/unit/test_banking77_evaluation.py -v

    The live evaluation:
    EVAL_LIVE_API=1 python scripts/eval_banking77_stratified.py --n 500
"""
from __future__ import annotations

import pytest

from agents.conversational_agent import INTENT_BUCKETS
from evaluation.banking77_data import (
    load_banking77,
    load_bucket_map,
    stratified_sample,
)
from evaluation.metrics import classification_report


@pytest.fixture(scope="module")
def banking77():
    return load_banking77()


class TestBucketMap:
    """The 77 -> 8 map is committed data. These check it, not the model."""

    def test_covers_all_77_intents(self):
        bucket_map, _ = load_bucket_map()
        assert len(bucket_map) == 77

    def test_every_target_bucket_is_a_real_system_bucket(self):
        bucket_map, _ = load_bucket_map()
        unknown = set(bucket_map.values()) - set(INTENT_BUCKETS)
        assert not unknown, f"map points at buckets the system does not have: {unknown}"

    def test_documents_the_five_buckets_it_cannot_reach(self):
        """
        Banking77 is a customer-service corpus; five of the eight buckets
        describe advisory conversations it was never built to contain.
        The map must keep saying so — that note is what stops a reader
        interpreting a missing bucket as a model failure.
        """
        _, meta = load_bucket_map()
        coverage = meta["bucket_coverage_summary"]
        zero_coverage = {b for b, n in coverage.items() if n == 0}
        assert zero_coverage == {
            "risk_profiling", "investment_advice", "budget_analysis",
            "full_advisory", "out_of_scope",
        }
        assert meta["domain_gap_note"]


class TestCorpusLoading:

    def test_loads_the_local_split(self, banking77):
        items, provenance = banking77
        assert len(items) > 1000
        assert provenance["n_available"] == len(items)

    def test_split_and_its_justification_are_recorded(self, banking77):
        """
        The cached split is `train`. Using it for a zero-shot classifier
        is defensible, but only if the results say so out loud instead of
        reporting an unqualified "Banking77".
        """
        _, provenance = banking77
        assert provenance["split"] in {"train", "test"}
        assert len(provenance["split_justification"]) > 50
        if provenance["split"] == "train":
            assert "zero-shot" in provenance["split_justification"].lower()

    def test_every_item_carries_a_mapped_bucket(self, banking77):
        items, _ = banking77
        for item in items[:200]:
            assert item.true_bucket in INTENT_BUCKETS
            assert item.text

    def test_unmapped_labels_are_reported_not_silently_dropped(self, banking77):
        _, provenance = banking77
        assert "n_unmapped_labels_skipped" in provenance
        assert "unmapped_labels" in provenance


class TestSampling:

    def test_default_sample_size_is_larger_than_the_previous_fixture(self, banking77):
        """
        The prior live evaluation used a 20-item fixture
        (test_conversational_agent.py::BANKING77_FIXTURE); the prior
        bucket evaluation used 200. The rebuilt default is 500.
        """
        from evaluation.banking77_data import DEFAULT_SAMPLE_SIZE

        assert DEFAULT_SAMPLE_SIZE >= 500

    def test_sample_is_drawn_without_replacement(self, banking77):
        items, _ = banking77
        sample, _ = stratified_sample(items, n=300, seed=1)
        assert len({i.item_id for i in sample}) == len(sample)

    def test_sampling_metadata_is_complete_enough_to_reproduce(self, banking77):
        items, _ = banking77
        _, meta = stratified_sample(items, n=300, seed=1)
        for key in ("method", "seed", "requested_n", "actual_n", "n_strata"):
            assert key in meta

    def test_labels_are_never_altered_by_sampling(self, banking77):
        items, _ = banking77
        bucket_map, _ = load_bucket_map()
        sample, _ = stratified_sample(items, n=200, seed=1)
        for item in sample:
            assert item.true_bucket == bucket_map[item.banking77_label]


class TestReportingShape:
    """
    The metric block the harness writes, exercised on synthetic
    predictions so it is testable without an API key.
    """

    def test_report_covers_all_eight_buckets_even_when_unpredicted(self):
        buckets = list(INTENT_BUCKETS)
        report = classification_report(
            ["general_query"] * 10, ["general_query"] * 8 + ["product_suggestion"] * 2,
            labels=buckets,
        )
        assert set(report.details["per_class"]) == set(buckets)
        assert report.details["per_class"]["risk_profiling"]["support"] == 0

    def test_confusion_matrix_exposes_advisory_leakage(self):
        """
        The safety-relevant failure: a customer-service utterance routed
        to an advisory specialist. It must be visible as a cell, not
        averaged away.
        """
        report = classification_report(
            ["investment_advice"], ["general_query"], labels=list(INTENT_BUCKETS),
        )
        matrix = report.details["confusion_matrix"]
        assert matrix["general_query"]["investment_advice"] == 1

    def test_accuracy_is_the_headline_value(self):
        report = classification_report(
            ["general_query", "general_query"],
            ["general_query", "product_suggestion"],
            labels=list(INTENT_BUCKETS),
        )
        assert report.value == 0.5
