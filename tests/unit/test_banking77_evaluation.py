"""
Day 2b — Banking77 intent-bucket evaluation.

Real PolyAI/banking77 data (data/raw/banking77 — a HuggingFace Arrow
cache, see scripts/download_datasets.py), mapped through
data/banking77_bucket_map.json onto ConversationalAgent's 8-bucket
taxonomy (agents.conversational_agent.INTENT_BUCKETS).

COST CONTROL: uses classify_only() — ONE LLM call per sample — never
run() (three calls: classify + slot-extract + reply). 200 samples x 1
call ~= 6k tokens; x3 would be 18k. This is the check patch 12 asked
for: is full_advisory stealing traffic from investment_advice?

DOMAIN GAP (see data/banking77_bucket_map.json for the full reasoning):
Banking77 is a retail-banking customer-service dataset; 5 of this
system's 8 buckets (risk_profiling, investment_advice, budget_analysis,
full_advisory, out_of_scope) describe financial-ADVISORY conversations
Banking77 was never built to contain. Precision/recall for those 5
buckets is not meaningful here — there is no ground truth for them in
this dataset. What IS meaningful and reported: false-positive leakage
INTO those buckets, i.e. how often the classifier predicts one of them
for a query whose real bucket is something else entirely. That's
exactly the "magnet" failure mode already flagged in
conversational_agent.py's own comments on full_advisory.

TWO TEST CLASSES:
  TestStratifiedSampling — pure data logic (no LLM), runs anywhere.
  TestBanking77BucketEvaluation — the real evaluation. Needs
      EVAL_LIVE_API=1 (see conftest._no_live_api_in_tests) and costs
      real tokens (~6k). Not run as part of this session — the sampling
      and analysis logic below is unit-tested in isolation instead.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from agents.conversational_agent import INTENT_BUCKETS, ConversationalAgent
from evaluation.results_io import write_results
from utils.llm_client import LLMClient

ROOT = Path(__file__).resolve().parent.parent.parent
BUCKET_MAP_PATH = ROOT / "data" / "banking77_bucket_map.json"
BANKING77_PATH = ROOT / "data" / "raw" / "banking77"
CHECKPOINT_PATH = ROOT / "data" / "processed" / "banking77_eval_checkpoint.json"

ALL_BUCKETS = list(INTENT_BUCKETS.keys())

def _sample_fingerprint(sample: list[dict]) -> str:
    """
    So a checkpoint from a different sample (different seed, different
    n_total, or the underlying dataset having changed) is never mistaken
    for a resumable match against THIS sample.
    """
    import hashlib
    raw = json.dumps([[s["text"], s["label_text"]] for s in sample])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_checkpoint(fingerprint: str) -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    try:
        data = json.loads(CHECKPOINT_PATH.read_text())
    except Exception:
        return []
    if data.get("fingerprint") != fingerprint:
        return []  # different sample/seed -- nothing to resume, start fresh
    return data.get("predictions", [])


def _save_checkpoint(fingerprint: str, predictions: list[dict]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({
        "fingerprint": fingerprint,
        "predictions": predictions,
    }))


def load_bucket_map() -> dict[str, str]:
    return json.loads(BUCKET_MAP_PATH.read_text())["bucket_map"]


def stratified_sample(
    by_label: dict[str, list[tuple[str, str]]],
    n_total: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Proportional-to-frequency stratified sample across the real 77
    Banking77 labels — standard stratified sampling for a labelled
    classification dataset, not stratified by our own 8 buckets (which
    would be close to meaningless given 5 of them have zero real
    examples in this data).

    by_label: {label_text: [(text, label_text), ...]} — every row's
        (text, label) pair, grouped by label. Takes pre-grouped data
        rather than a HF Dataset directly so this is testable without
        the real dataset on disk.

    Minimum 1 sample per class; rounding drift corrected against the
    largest classes so the total lands on n_total exactly.
    """
    rng = np.random.default_rng(seed)
    total_rows = sum(len(rows) for rows in by_label.values())
    labels = sorted(by_label.keys())

    raw_alloc = {
        label: (len(rows) / total_rows) * n_total
        for label, rows in by_label.items()
    }
    alloc = {label: max(1, round(v)) for label, v in raw_alloc.items()}

    diff = n_total - sum(alloc.values())
    order = sorted(labels, key=lambda l: len(by_label[l]), reverse=True)
    i = 0
    while diff != 0 and order:
        label = order[i % len(order)]
        if diff > 0:
            alloc[label] += 1
            diff -= 1
        elif alloc[label] > 1:
            alloc[label] -= 1
            diff += 1
        i += 1

    sample = []
    for label, k in alloc.items():
        rows = by_label[label]
        k = min(k, len(rows))
        chosen_idx = rng.choice(len(rows), size=k, replace=False)
        for idx in chosen_idx:
            text, label_text = rows[int(idx)]
            sample.append({"text": text, "label_text": label_text})

    perm = rng.permutation(len(sample))
    return [sample[i] for i in perm]


def build_confusion_and_metrics(
    predictions: list[dict[str, Any]],
) -> tuple[dict, dict]:
    """
    predictions: [{"true_bucket": str | None, "predicted_bucket": str}, ...]

    Returns (confusion_matrix, per_bucket_metrics). Pulled out of the
    evaluation test so it's independently unit-testable with fabricated
    predictions, no LLM involved.
    """
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in predictions:
        true_key = p["true_bucket"] or "NO_GROUND_TRUTH_IN_BANKING77"
        confusion[true_key][p["predicted_bucket"]] += 1

    per_bucket = {}
    for bucket in ALL_BUCKETS:
        tp = confusion[bucket][bucket]
        true_count = sum(confusion[bucket].values())
        pred_count = sum(confusion[b].get(bucket, 0) for b in confusion)
        precision = tp / pred_count if pred_count else None
        recall = tp / true_count if true_count else None
        per_bucket[bucket] = {
            "true_count_in_sample": true_count,
            "predicted_count": pred_count,
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "has_ground_truth_in_banking77": true_count > 0,
        }

    return {k: dict(v) for k, v in confusion.items()}, per_bucket


class TestStratifiedSampling:
    """Pure data logic — no LLM, no dataset file required. Runs anywhere."""

    def _fake_by_label(self) -> dict[str, list[tuple[str, str]]]:
        # 3 classes, deliberately uneven sizes, to exercise proportional
        # allocation + the minimum-1-per-class floor.
        return {
            "big_class": [(f"text_{i}", "big_class") for i in range(500)],
            "medium_class": [(f"text_{i}", "medium_class") for i in range(80)],
            "tiny_class": [(f"text_{i}", "tiny_class") for i in range(3)],
        }

    def test_sample_size_matches_requested_total(self):
        sample = stratified_sample(self._fake_by_label(), n_total=200, seed=1)
        assert len(sample) == 200

    def test_every_class_gets_at_least_one(self):
        sample = stratified_sample(self._fake_by_label(), n_total=200, seed=1)
        labels_present = {item["label_text"] for item in sample}
        assert labels_present == {"big_class", "medium_class", "tiny_class"}

    def test_allocation_is_roughly_proportional(self):
        sample = stratified_sample(self._fake_by_label(), n_total=200, seed=1)
        counts = Counter(item["label_text"] for item in sample)
        # big_class is ~86% of the population (500/583) -> should dominate
        assert counts["big_class"] > counts["medium_class"] > counts["tiny_class"]

    def test_deterministic_given_same_seed(self):
        s1 = stratified_sample(self._fake_by_label(), n_total=200, seed=7)
        s2 = stratified_sample(self._fake_by_label(), n_total=200, seed=7)
        assert s1 == s2

    def test_cannot_sample_more_than_a_tiny_classs_population(self):
        """tiny_class only has 3 rows — allocation must not ask for more."""
        sample = stratified_sample(self._fake_by_label(), n_total=200, seed=1)
        counts = Counter(item["label_text"] for item in sample)
        assert counts["tiny_class"] <= 3


class TestBucketMapFile:
    """Structural checks on the committed bucket-map file itself."""

    def test_bucket_map_is_valid_json(self):
        assert load_bucket_map()  # raises on malformed JSON

    def test_every_mapped_bucket_is_a_real_bucket(self):
        bucket_map = load_bucket_map()
        for label, bucket in bucket_map.items():
            assert bucket in INTENT_BUCKETS, f"{label} -> unknown bucket {bucket!r}"

    def test_covers_all_77_banking77_labels(self):
        """
        Guards against silent drift if banking77's cached copy ever
        changes — this is a static, hand-reviewed 77-label mapping. If
        the underlying dataset's label set changes, this test should
        fail loudly rather than silently classifying unmapped labels as
        None everywhere.
        """
        bucket_map = load_bucket_map()
        assert len(bucket_map) == 77


class TestConfusionMatrixAnalysis:
    """
    Pure analysis logic — fabricated predictions, no LLM. Locks in the
    patch-12 diagnostic: false positives INTO full_advisory/
    investment_advice are counted correctly even though those buckets
    have zero true Banking77 labels.
    """

    def test_perfect_predictions_give_full_precision_and_recall(self):
        preds = [
            {"true_bucket": "general_query", "predicted_bucket": "general_query"}
            for _ in range(10)
        ]
        confusion, per_bucket = build_confusion_and_metrics(preds)
        assert per_bucket["general_query"]["precision"] == 1.0
        assert per_bucket["general_query"]["recall"] == 1.0

    def test_bucket_with_no_ground_truth_reports_none_not_zero(self):
        """
        investment_advice has zero true Banking77 labels — precision/
        recall should be None (undefined), not misleadingly 0.0.
        """
        preds = [
            {"true_bucket": "general_query", "predicted_bucket": "general_query"}
            for _ in range(5)
        ]
        _, per_bucket = build_confusion_and_metrics(preds)
        assert per_bucket["investment_advice"]["recall"] is None
        assert per_bucket["investment_advice"]["has_ground_truth_in_banking77"] is False

    def test_false_positives_into_full_advisory_are_visible(self):
        """
        The actual patch-12 diagnostic: 3 queries truly general_query,
        2 truly product_suggestion, all 5 misclassified as
        full_advisory. Every one is a false positive by construction
        (full_advisory has no true labels in Banking77) — the confusion
        matrix must show exactly which real buckets they came from.
        """
        preds = (
            [{"true_bucket": "general_query", "predicted_bucket": "full_advisory"}] * 3
            + [{"true_bucket": "product_suggestion", "predicted_bucket": "full_advisory"}] * 2
        )
        confusion, per_bucket = build_confusion_and_metrics(preds)
        assert confusion["general_query"]["full_advisory"] == 3
        assert confusion["product_suggestion"]["full_advisory"] == 2
        assert per_bucket["full_advisory"]["predicted_count"] == 5
        assert per_bucket["full_advisory"]["true_count_in_sample"] == 0

class TestCheckpointing:
    """
    Pure file I/O — no LLM, no dataset needed. The mechanism that makes a
    rate-limited run resumable instead of needing to restart from sample 0.
    """

    def test_fingerprint_is_stable_for_the_same_sample(self):
        sample = [{"text": "a", "label_text": "x"}, {"text": "b", "label_text": "y"}]
        assert _sample_fingerprint(sample) == _sample_fingerprint(sample)

    def test_fingerprint_differs_for_a_different_sample(self):
        sample_a = [{"text": "a", "label_text": "x"}]
        sample_b = [{"text": "different", "label_text": "x"}]
        assert _sample_fingerprint(sample_a) != _sample_fingerprint(sample_b)

    def test_load_checkpoint_missing_file_returns_empty(self, tmp_path, monkeypatch):
        import tests.unit.test_banking77_evaluation as m
        monkeypatch.setattr(m, "CHECKPOINT_PATH", tmp_path / "does_not_exist.json")
        assert _load_checkpoint("anyfingerprint") == []

    def test_save_then_load_roundtrips(self, tmp_path, monkeypatch):
        import tests.unit.test_banking77_evaluation as m
        monkeypatch.setattr(m, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
        preds = [{"text": "a", "predicted_bucket": "general_query"}]
        _save_checkpoint("fp123", preds)
        assert _load_checkpoint("fp123") == preds

    def test_load_with_mismatched_fingerprint_returns_empty(self, tmp_path, monkeypatch):
        """
        A checkpoint from a DIFFERENT sample (different seed, or the
        dataset changed) must never be silently reused as if it matches
        the current run — that would resume with the wrong 138 items
        already "done".
        """
        import tests.unit.test_banking77_evaluation as m
        monkeypatch.setattr(m, "CHECKPOINT_PATH", tmp_path / "checkpoint.json")
        _save_checkpoint("fp_from_a_different_run", [{"text": "stale"}])
        assert _load_checkpoint("fp_for_this_run") == []

    def test_corrupt_checkpoint_file_falls_back_to_empty(self, tmp_path, monkeypatch):
        import tests.unit.test_banking77_evaluation as m
        path = tmp_path / "checkpoint.json"
        path.write_text("{ not valid json")
        monkeypatch.setattr(m, "CHECKPOINT_PATH", path)
        assert _load_checkpoint("anything") == []

@pytest.mark.evaluation   # produces results/*.json — see conftest._no_live_api_in_tests
class TestBanking77BucketEvaluation:
    """
    The real evaluation — needs EVAL_LIVE_API=1 and a working LLM key
    (real tokens, ~6k for n=200). Not run in this session; sampling and
    analysis logic are covered above without touching the API.
    """

    def test_stratified_classification_and_write_results(self):
        from datasets import load_from_disk

        bucket_map = load_bucket_map()
        ds = load_from_disk(str(BANKING77_PATH))

        by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for text, label in zip(ds["text"], ds["label_text"]):
            by_label[label].append((text, label))

        sample = stratified_sample(by_label, n_total=200, seed=42)
        assert len(sample) == 200

        fingerprint = _sample_fingerprint(sample)
        predictions = _load_checkpoint(fingerprint)
        already_done = len(predictions)
        if already_done:
            print(
                f"\n[Banking77] Resuming from checkpoint: "
                f"{already_done}/200 already classified in a prior run."
            )

        client = LLMClient()
        agent = ConversationalAgent(client)

        stopped_early = False
        for item in sample[already_done:]:
            true_bucket = bucket_map.get(item["label_text"])
            pred_bucket, confidence = agent.classify_only(item["text"])
            if agent.last_classification_error is not None:
                print(
                    f"\n[Banking77] Stopped at {len(predictions)}/200 -- "
                    f"classification failure, not a real result: "
                    f"{agent.last_classification_error}"
                )
                if agent.last_classification_error_status == 429:
                    print(
                        "[Banking77] This is a quota/rate-limit error. "
                        "Checkpoint saved -- rerun the exact same command "
                        "once quota resets and it will resume from here, "
                        "not from sample 0."
                    )
                stopped_early = True
                break
            predictions.append({
                "text": item["text"],
                "banking77_label": item["label_text"],
                "true_bucket": true_bucket,
                "predicted_bucket": pred_bucket,
                "confidence": confidence,
            })
            _save_checkpoint(fingerprint, predictions)

        if stopped_early:
            pytest.skip(
                f"Stopped early at {len(predictions)}/200 after a genuine "
                f"classification failure (see printed output above). "
                f"Checkpoint saved to {CHECKPOINT_PATH} -- rerun this same "
                f"test to resume from where it stopped, once whatever "
                f"caused the failure (quota, network, etc.) is resolved."
            )

        confusion, per_bucket = build_confusion_and_metrics(predictions)

        full_advisory_fps = [p for p in predictions if p["predicted_bucket"] == "full_advisory"]
        stolen_from_fa = Counter(p["true_bucket"] for p in full_advisory_fps)
        investment_fps = [p for p in predictions if p["predicted_bucket"] == "investment_advice"]
        stolen_from_ia = Counter(p["true_bucket"] for p in investment_fps)

        results_path = write_results(
            {
                "phase": 2,
                "research_question": "Banking77 bucket classification (Day 2b)",
                "dataset": "PolyAI/banking77 (Casanueva et al. 2020), stratified n=200",
                "bucket_map_source": "data/banking77_bucket_map.json",
                "cost_control": "classify_only() -- 1 LLM call/sample, not run() (3 calls)",
                "domain_gap_note": (
                    "5 of 8 buckets (risk_profiling, investment_advice, "
                    "budget_analysis, full_advisory, out_of_scope) have zero "
                    "Banking77 ground truth by construction -- precision/"
                    "recall for them is not meaningful. False-positive "
                    "leakage INTO them (reported below) is the relevant "
                    "signal instead."
                ),
                "per_bucket_metrics": per_bucket,
                "confusion_matrix": confusion,
                "full_advisory_false_positives_by_true_bucket": dict(stolen_from_fa),
                "investment_advice_false_positives_by_true_bucket": dict(stolen_from_ia),
                "sample_predictions_first_20": predictions[:20],
            },
            "phase2_banking77_bucket_evaluation.json",
        )
        print(f"\n[Banking77] Results written to {results_path}")
        print(f"[Banking77] full_advisory predicted {len(full_advisory_fps)}/200 times, "
              f"stolen from: {dict(stolen_from_fa)}")
        print(f"[Banking77] investment_advice predicted {len(investment_fps)}/200 times, "
              f"stolen from: {dict(stolen_from_ia)}")

        assert len(predictions) == 200

        CHECKPOINT_PATH.unlink(missing_ok=True)  # done -- no longer needed to resume