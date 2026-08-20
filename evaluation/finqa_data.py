"""
evaluation/finqa_data.py — the real FinQA Verified test split.

WHAT IS ACTUALLY IN THIS REPOSITORY
    data/raw/finqa_verified/          HF dataset dir, split="test", 91 rows
                                      columns: question, context, answer
    data/raw/finqa_original_train.json  FinQA Original, TRAIN split, 6,251 rows

    RQ5 uses the 91 verified TEST items and nothing else. The 6,251 train
    items are not touched by this module. Mixing a train split into a
    test set to make n bigger is the single most damaging thing that
    could be done to RQ5's credibility, and the loader below has no code
    path that can do it — `load_finqa_verified()` reads only the
    finqa_verified directory, and `assert_no_train_contamination()`
    exists to prove it after the fact.

WHY PYARROW DIRECTLY AND NOT `datasets.load_from_disk`
    rag/knowledge_base.py uses load_from_disk, which pulls in the whole
    `datasets` package. This module is imported by evaluation scripts
    that otherwise need nothing heavier than numpy, and the file is a
    plain Arrow IPC stream — 12 lines of pyarrow reads it exactly, with
    no version-coupling to a library that reorganises its on-disk format
    every couple of releases.

A LEAKAGE WARNING WORTH READING
    rag/knowledge_base.py::_load_finqa_verified indexes each verified
    item as "Question: {q} Verified answer: {a}" — the ANSWER goes into
    the retrievable corpus. That is fine for its intended purpose
    (grounding advisory claims in worked financial examples) and fatal
    for RQ5: a with-RAG condition retrieving from that corpus can read
    the answer straight out of a citation, and the resulting exact-match
    score measures retrieval of a leaked label.

    scripts/eval_rq5_finqa_verified.py therefore builds its OWN corpus
    from the `context` field only, never touches the shared index, and
    calls `answer_leakage_report()` below to state plainly how often a
    gold answer happens to appear verbatim in its own context (a real and
    unavoidable property of FinQA, where some answers are table cells
    rather than computed values).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR

FINQA_VERIFIED_DIR = ROOT_DIR / "data" / "raw" / "finqa_verified"
FINQA_ORIGINAL_TRAIN = ROOT_DIR / "data" / "raw" / "finqa_original_train.json"


@dataclass
class FinQAItem:
    item_id: str
    question: str
    context: str
    answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "answer": self.answer,
            "context_chars": len(self.context),
        }


def _read_arrow(path: Path) -> list[dict[str, Any]]:
    """Read a HuggingFace Arrow IPC *stream* file into plain dicts."""
    import pyarrow as pa

    with pa.memory_map(str(path), "rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    return table.to_pylist()


def _split_name(dataset_dir: Path) -> str:
    state = dataset_dir / "state.json"
    if state.exists():
        try:
            return json.loads(state.read_text(encoding="utf-8")).get("_split", "unknown")
        except Exception:
            pass
    return "unknown"


def load_finqa_verified() -> tuple[list[FinQAItem], dict[str, Any]]:
    """
    All 91 items of the FinQA Verified test split, in file order.

    Returns (items, provenance). No sampling, no shuffling, no limit
    argument: RQ5 runs the whole split, so there is no sampling decision
    to defend and no seed to record.
    """
    arrow = FINQA_VERIFIED_DIR / "data-00000-of-00001.arrow"
    if not arrow.exists():
        raise FileNotFoundError(
            f"FinQA Verified not found at {arrow}. Fetch it with:\n"
            f"    python scripts/download_datasets.py --phase 8"
        )

    rows = _read_arrow(arrow)
    items = [
        FinQAItem(
            item_id=f"finqa-verified-{i:03d}",
            question=str(row.get("question", "")).strip(),
            context=str(row.get("context", "")).strip(),
            answer=str(row.get("answer", "")).strip(),
        )
        for i, row in enumerate(rows)
    ]

    split = _split_name(FINQA_VERIFIED_DIR)
    info_path = FINQA_VERIFIED_DIR / "dataset_info.json"
    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}

    provenance = {
        "dataset_name": "FinQA Verified [D1]",
        "hf_dataset_name": info.get("dataset_name", "finqa-verified"),
        "split": split,
        "is_verified_split": True,
        "n_items": len(items),
        "n_items_used": len(items),
        "sampling": "none — the complete split is evaluated",
        "source_path": str(arrow.relative_to(ROOT_DIR)),
        "fields": ["question", "context", "answer"],
        "citation": (
            "Chen et al. (2021), FinQA: A Dataset of Numerical Reasoning "
            "over Financial Data. Verified subset."
        ),
        "train_split_present_in_repo_but_unused": {
            "path": str(FINQA_ORIGINAL_TRAIN.relative_to(ROOT_DIR)),
            "exists": FINQA_ORIGINAL_TRAIN.exists(),
            "note": (
                "FinQA Original TRAIN split (6,251 items) is present in the "
                "repository for RAG corpus construction. It is NOT loaded "
                "by this module and contributes no items to RQ5."
            ),
        },
    }
    return items, provenance


def assert_no_train_contamination(items: list[FinQAItem]) -> dict[str, Any]:
    """
    Prove the evaluation set contains no FinQA Original TRAIN questions.

    Compares normalised question strings against the train split and
    reports the overlap. Returns a report rather than raising on a small
    overlap, because FinQA Verified is a re-annotation of a subset of the
    original data and some question TEXT legitimately recurs — what
    matters for the write-up is that the number is known and stated, not
    assumed to be zero.
    """
    if not FINQA_ORIGINAL_TRAIN.exists():
        return {
            "checked": False,
            "reason": "finqa_original_train.json not present",
        }

    try:
        with open(FINQA_ORIGINAL_TRAIN, encoding="utf-8") as fh:
            train = json.load(fh)
    except Exception as exc:
        return {"checked": False, "reason": f"unreadable: {exc}"}

    def norm(text: str) -> str:
        return " ".join(str(text).lower().split())

    train_questions = {
        norm(entry.get("qa", {}).get("question", ""))
        for entry in train
        if entry.get("qa", {}).get("question")
    }
    overlap = [
        item.item_id for item in items if norm(item.question) in train_questions
    ]

    return {
        "checked": True,
        "n_train_questions": len(train_questions),
        "n_eval_items": len(items),
        "n_overlapping_question_strings": len(overlap),
        "overlapping_item_ids": overlap[:20],
        "train_items_used_in_evaluation": 0,
        "note": (
            "No train item is evaluated. This check reports how many "
            "verified-split QUESTION STRINGS also occur in the train "
            "split, which is expected to be non-zero because FinQA "
            "Verified re-annotates original FinQA data. The evaluation "
            "uses verified-split items and verified answers exclusively."
        ),
    }


def answer_leakage_report(items: list[FinQAItem]) -> dict[str, Any]:
    """
    How often does the gold answer appear verbatim in its own context?

    A known property of FinQA — some answers are read directly off a
    table, others are computed — and therefore an upper bound on how much
    of a grounded condition's score could be pattern-matching rather than
    reasoning. Reported, not corrected: removing those items would be
    filtering the test set to suit the result.
    """
    verbatim = []
    for item in items:
        answer = item.answer.replace("%", "").replace("$", "").replace(",", "").strip()
        if answer and answer in item.context.replace(",", ""):
            verbatim.append(item.item_id)

    return {
        "n_items": len(items),
        "n_answers_appearing_verbatim_in_own_context": len(verbatim),
        "proportion": round(len(verbatim) / len(items), 4) if items else 0.0,
        "item_ids": verbatim[:25],
        "interpretation": (
            "Upper bound on how much of a grounded condition's exact-match "
            "score could be achieved by copying a value out of the context "
            "rather than reasoning over it. No item was removed on the "
            "basis of this check."
        ),
    }
