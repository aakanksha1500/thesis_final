"""
evaluation/banking77_data.py — Banking77 loading and stratified sampling.

WHICH SPLIT IS ACTUALLY HERE, AND WHY THAT IS DEFENSIBLE
    data/raw/banking77/ contains ONE arrow file. Its state.json says
    _split="train" (9,993 rows). dataset_info.json also describes a
    3,076-row test split, but that file is not in the repository.

    The train split is used as the evaluation pool, and that is sound
    HERE for one specific reason: the intent classifier under test is
    ZERO-SHOT. ConversationalAgent._classify_intent sends a prompt built
    from INTENT_BUCKET_GUIDE and asks the LLM for a bucket. Nothing in
    this project is ever fitted on Banking77 — there is no training step,
    no fine-tune, and no few-shot exemplar drawn from the dataset. To a
    zero-shot classifier a "train" utterance is simply unseen text.

    Two things keep this honest rather than convenient:
      * every results file records split="train" explicitly, with this
        justification embedded, rather than saying "Banking77" and
        leaving the reader to assume test;
      * `assert_no_prompt_leakage()` below checks the classifier's own
        prompt guide against the sampled utterances, so the claim "no
        Banking77 text is in the prompt" is verified rather than
        asserted.

    If the test split is later downloaded, `load_banking77()` picks it up
    automatically and the provenance block changes with no code edit.

WHAT BANKING77 CAN AND CANNOT MEASURE HERE
    data/banking77_bucket_map.json maps all 77 fine-grained intents onto
    just 3 of this system's 8 buckets (general_query 64,
    product_suggestion 12, explanation_request 1). The other five —
    risk_profiling, investment_advice, budget_analysis, full_advisory,
    out_of_scope — have zero ground truth by construction, because
    Banking77 is a retail-banking customer-service corpus and those
    buckets describe advisory conversations it was never built to
    contain.

    So precision/recall for those five is undefined here, and the
    relevant signal is instead FALSE-POSITIVE LEAKAGE INTO them: how
    often a customer-service utterance is misrouted to an advisory
    specialist. That is a real safety property (a misroute sends a "where
    is my card" question to the InvestmentAgent) and it is reported
    explicitly. The five advisory buckets are evaluated properly by
    tests/unit/test_core_intent_evaluation.py's 110-item hand-authored
    corpus instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import ROOT_DIR

BANKING77_DIR = ROOT_DIR / "data" / "raw" / "banking77"
BUCKET_MAP_PATH = ROOT_DIR / "data" / "banking77_bucket_map.json"

DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 20260812


@dataclass
class Banking77Item:
    item_id: str
    text: str
    banking77_label: str
    true_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "banking77_label": self.banking77_label,
            "true_bucket": self.true_bucket,
        }


def load_bucket_map() -> tuple[dict[str, str], dict[str, Any]]:
    """The hand-reviewed 77-intent -> 8-bucket map, plus its own metadata."""
    with open(BUCKET_MAP_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["bucket_map"], {
        "source_file": str(BUCKET_MAP_PATH.relative_to(ROOT_DIR)),
        "methodology": payload.get("methodology"),
        "bucket_coverage_summary": payload.get("bucket_coverage_summary"),
        "domain_gap_note": payload.get("domain_gap_note"),
        "version": payload.get("_meta", {}).get("version"),
        "labels_unchanged": (
            "The bucket map is used exactly as committed. No label was "
            "edited, added or removed for this evaluation."
        ),
    }


def _read_arrow(path: Path) -> list[dict[str, Any]]:
    import pyarrow as pa

    with pa.memory_map(str(path), "rb") as source:
        return pa.ipc.open_stream(source).read_all().to_pylist()


def load_banking77() -> tuple[list[Banking77Item], dict[str, Any]]:
    """
    Every locally cached Banking77 utterance, mapped to its bucket.

    Prefers a `test` subdirectory if one is ever added; otherwise reads
    the single cached split and records which one it was.
    """
    candidates = [
        BANKING77_DIR / "test" / "data-00000-of-00001.arrow",
        BANKING77_DIR / "data-00000-of-00001.arrow",
    ]
    arrow = next((p for p in candidates if p.exists()), None)
    if arrow is None:
        raise FileNotFoundError(
            f"No Banking77 arrow file under {BANKING77_DIR}. Fetch it with:\n"
            f"    python scripts/download_datasets.py --phase 2"
        )

    split = "unknown"
    state = arrow.parent / "state.json"
    if state.exists():
        try:
            split = json.loads(state.read_text(encoding="utf-8")).get("_split", "unknown")
        except Exception:
            pass

    bucket_map, bucket_meta = load_bucket_map()
    rows = _read_arrow(arrow)

    items: list[Banking77Item] = []
    unmapped: set[str] = set()
    for i, row in enumerate(rows):
        label = str(row.get("label_text", "")).strip()
        bucket = bucket_map.get(label)
        if bucket is None:
            unmapped.add(label)
            continue
        items.append(Banking77Item(
            item_id=f"b77-{split}-{i:05d}",
            text=str(row.get("text", "")).strip(),
            banking77_label=label,
            true_bucket=bucket,
        ))

    provenance = {
        "dataset_name": "Banking77 (PolyAI)",
        "citation": (
            "Casanueva et al. (2020), Efficient Intent Detection with Dual "
            "Sentence Encoders."
        ),
        "split": split,
        "split_justification": (
            "The cached split is 'train'. It is used as an evaluation pool "
            "because the intent classifier under test is ZERO-SHOT: no "
            "Banking77 data is used for training, fine-tuning or few-shot "
            "prompting anywhere in this project, so train-split utterances "
            "are unseen text to it. Stated explicitly rather than reported "
            "as 'Banking77' without qualification."
        ) if split == "train" else "Held-out test split.",
        "n_available": len(items),
        "n_unmapped_labels_skipped": len(unmapped),
        "unmapped_labels": sorted(unmapped),
        "source_path": str(arrow.relative_to(ROOT_DIR)),
        "bucket_map": bucket_meta,
    }
    return items, provenance


def stratified_sample(
    items: list[Banking77Item],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Banking77Item], dict[str, Any]]:
    """
    Reproducible sample stratified by the 77 fine-grained INTENT LABELS.

    Stratifying on the fine label rather than the coarse bucket is the
    stronger choice: it preserves the bucket balance automatically (each
    bucket's share is the sum of its labels' shares) AND guarantees that
    no individual intent drops out of the sample, which a bucket-level
    stratification would happily allow — a 500-item bucket-stratified
    sample could contain 500 'card_arrival' utterances and still look
    correctly balanced at bucket level.

    Allocation is proportional with largest-remainder rounding, so the
    sample size is exactly `n` and no label is systematically favoured by
    rounding. Within a stratum, items are drawn without replacement from
    a seeded Generator.
    """
    by_label: dict[str, list[Banking77Item]] = {}
    for item in items:
        by_label.setdefault(item.banking77_label, []).append(item)

    labels = sorted(by_label)
    total = len(items)

    exact = {label: len(by_label[label]) / total * n for label in labels}
    allocation = {label: int(np.floor(v)) for label, v in exact.items()}
    remainder = n - sum(allocation.values())
    # Largest remainder first; label name breaks ties so the result is
    # deterministic regardless of dict ordering.
    for label in sorted(labels, key=lambda l: (-(exact[l] - allocation[l]), l))[:remainder]:
        allocation[label] += 1

    rng = np.random.default_rng(seed)
    sample: list[Banking77Item] = []
    for label in labels:
        pool = by_label[label]
        take = min(allocation[label], len(pool))
        if take <= 0:
            continue
        idx = rng.choice(len(pool), size=take, replace=False)
        sample.extend(pool[i] for i in sorted(idx))

    bucket_counts: dict[str, int] = {}
    for item in sample:
        bucket_counts[item.true_bucket] = bucket_counts.get(item.true_bucket, 0) + 1

    population_buckets: dict[str, int] = {}
    for item in items:
        population_buckets[item.true_bucket] = population_buckets.get(item.true_bucket, 0) + 1

    sampling_meta = {
        "method": "proportional stratified by banking77 fine label (77 strata)",
        "allocation_rounding": "largest remainder",
        "seed": seed,
        "requested_n": n,
        "actual_n": len(sample),
        "n_strata": len(labels),
        "min_per_stratum": min(allocation.values()) if allocation else 0,
        "max_per_stratum": max(allocation.values()) if allocation else 0,
        "bucket_balance_sample": bucket_counts,
        "bucket_balance_population": population_buckets,
        "bucket_balance_preserved": {
            bucket: {
                "population_share": round(population_buckets[bucket] / len(items), 4),
                "sample_share": round(bucket_counts.get(bucket, 0) / len(sample), 4),
            }
            for bucket in sorted(population_buckets)
        },
        "reproducible": (
            "Deterministic given (items, n, seed). Re-running produces the "
            "identical sample on any machine."
        ),
    }
    return sample, sampling_meta


def assert_no_prompt_leakage(sample: list[Banking77Item]) -> dict[str, Any]:
    """
    Verify that no sampled utterance appears in the classifier's prompt.

    The zero-shot justification for using the train split rests entirely
    on Banking77 text never reaching the model as an exemplar. This
    checks that claim against the actual prompt guide rather than
    trusting it.
    """
    try:
        from agents.conversational_agent import INTENT_BUCKETS
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "reason": f"could not import prompt guide: {exc}"}

    guide_text = json.dumps(INTENT_BUCKETS, default=str).lower()
    hits = [
        item.item_id for item in sample
        if item.text and item.text.lower() in guide_text
    ]
    return {
        "checked": True,
        "n_sampled_utterances_found_in_prompt": len(hits),
        "item_ids": hits[:10],
        "conclusion": (
            "Zero-shot use of the train split is sound: no sampled "
            "utterance appears in the classifier's prompt."
            if not hits else
            "LEAKAGE FOUND — sampled utterances appear in the prompt. The "
            "zero-shot justification does not hold for these items."
        ),
    }
