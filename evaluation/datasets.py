"""
evaluation/datasets.py — the three-layer dataset boundary, made explicit
and enforced in code rather than remembered by convention.

THE THREE LAYERS

    Layer 1  REAL / OPERATIONAL          data/processed/customers.csv
        The 254 records the running system actually serves: 3 hand-
        completed DEMO_* profiles, 1 give_me_some_credit-derived profile,
        250 SYNTH_* load-test profiles written by
        scripts/generate_synthetic_customer_base.py, and 1 session_update
        row. 251 of the 254 have NO ground_truth_risk_class.
        USE FOR: robustness, coverage, edge cases, "does the pipeline
        survive real data".
        NEVER USE FOR: accuracy, precision, recall, F1. There are no
        labels to be accurate against, and inventing them would be
        inventing the result.

    Layer 2  SYNTHETIC STRESS            data/evaluation/stress_customers.json
        1,000 seeded profiles spanning every risk class, all six
        data-sufficiency coverage tiers, and the full range of ages,
        incomes, employment statuses, debt-to-income ratios, horizons,
        tolerances and knowledge scores.
        USE FOR: stress and robustness — crash rate, confidence
        distribution, missing-feature handling, coverage-tier behaviour,
        subgroup stability.
        NEVER USE FOR: accuracy claims. These profiles carry a
        `construction_target_class` recording which cell of the rubric
        they were BUILT from, which is a construction record, not an
        independent observation, and this module refuses to hand it over
        as ground truth.

    Layer 3  GOLD STANDARD               data/evaluation/gold_risk_profiles.json
        200 profiles, 40 per risk class, each labelled by
        evaluation/risk_rubric.py — an instrument that imports nothing
        from agents/, never loads the model, and was fixed before any
        profile was generated. Plus the 15 original hand-labelled
        profiles, preserved verbatim with their own provenance tag.
        USE FOR: accuracy, precision, recall, F1, confusion matrix,
        per-class results, bootstrap CIs.

WHY THE BOUNDARY IS CODE AND NOT A COMMENT
    The failure mode this prevents is mundane and common: six months
    from now, someone needs "more data" for an accuracy number, sees
    1,000 profiles sitting in Layer 2 with a class attached to each, and
    concatenates. The resulting F1 would be reported in a dissertation
    and would be meaningless. `require_ground_truth()` raises instead.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from config.settings import ROOT_DIR

EVAL_DATA_DIR = ROOT_DIR / "data" / "evaluation"
REAL_CUSTOMERS_PATH = ROOT_DIR / "data" / "processed" / "customers.csv"
GOLD_PROFILES_PATH = EVAL_DATA_DIR / "gold_risk_profiles.json"
STRESS_CUSTOMERS_PATH = EVAL_DATA_DIR / "stress_customers.json"
STRESS_TRANSACTIONS_PATH = EVAL_DATA_DIR / "stress_transactions.json"


class GroundTruthUnavailable(RuntimeError):
    """
    Raised when an accuracy-style metric is requested for a dataset that
    has no independent labels. Deliberately not a warning: a warning gets
    filtered out of a long log and the wrong number still reaches a table.
    """


@dataclass(frozen=True)
class DatasetLayer:
    key: str
    name: str
    purpose: str
    has_ground_truth: bool
    permitted_metrics: tuple[str, ...]
    forbidden_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.key,
            "name": self.name,
            "purpose": self.purpose,
            "has_independent_ground_truth": self.has_ground_truth,
            "permitted_metrics": list(self.permitted_metrics),
            "forbidden_metrics": list(self.forbidden_metrics),
        }


LAYER_REAL = DatasetLayer(
    key="layer1_real",
    name="Real / operational customers",
    purpose="robustness, coverage, edge cases, real-data behaviour",
    has_ground_truth=False,
    permitted_metrics=(
        "completion_rate", "crash_rate", "confidence_distribution",
        "coverage_tier_distribution", "missing_feature_handling",
        "prediction_distribution",
    ),
    forbidden_metrics=("accuracy", "precision", "recall", "f1", "auc_roc"),
)

LAYER_STRESS = DatasetLayer(
    key="layer2_stress",
    name="Synthetic stress-test population",
    purpose="robustness and stress testing across the full feature space",
    has_ground_truth=False,
    permitted_metrics=(
        "completion_rate", "crash_rate", "confidence_distribution",
        "coverage_tier_distribution", "missing_feature_handling",
        "prediction_distribution", "subgroup_stability",
    ),
    forbidden_metrics=("accuracy", "precision", "recall", "f1", "auc_roc"),
)

LAYER_GOLD = DatasetLayer(
    key="layer3_gold",
    name="Gold-standard independently-labelled evaluation set",
    purpose="accuracy, precision, recall, F1, confusion matrix, CIs",
    has_ground_truth=True,
    permitted_metrics=(
        "accuracy", "precision", "recall", "f1", "auc_roc",
        "risk_alignment_rate", "confusion_matrix", "bootstrap_ci",
    ),
    forbidden_metrics=(),
)

ALL_LAYERS = (LAYER_REAL, LAYER_STRESS, LAYER_GOLD)


@dataclass
class EvaluationDataset:
    """A loaded dataset that knows what it is allowed to be used for."""
    layer: DatasetLayer
    name: str
    records: list[dict[str, Any]]
    provenance: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def require_ground_truth(self, operation: str = "accuracy metrics") -> None:
        """Guard. Call before computing anything that needs labels."""
        if not self.layer.has_ground_truth:
            raise GroundTruthUnavailable(
                f"{operation} requested for dataset {self.name!r} "
                f"(layer={self.layer.key}). This layer has no independent "
                f"ground truth and must not be used for {operation}. "
                f"Permitted here: {', '.join(self.layer.permitted_metrics)}. "
                f"Use the Layer-3 gold set "
                f"(evaluation.datasets.load_gold_risk_profiles) instead."
            )

    def ground_truth(self) -> list[str]:
        self.require_ground_truth("ground-truth extraction")
        return [r["ground_truth_risk_class"] for r in self.records]

    def features(self) -> list[dict[str, Any]]:
        return [r["features"] for r in self.records]

    def describe(self) -> dict[str, Any]:
        """The block embedded into every results file that uses this set."""
        return {
            "dataset_name": self.name,
            "n_records": len(self.records),
            "source_path": (
                str(self.source_path.relative_to(ROOT_DIR))
                if self.source_path else None
            ),
            **self.layer.to_dict(),
            "provenance": self.provenance,
        }


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Build it with:\n"
            f"    python scripts/build_gold_risk_dataset.py      (Layer 3)\n"
            f"    python scripts/build_stress_customer_base.py   (Layer 2)"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_gold_risk_profiles(
    include_original_15: bool = True,
) -> EvaluationDataset:
    """
    Layer 3. The only dataset in this project that may be used for
    accuracy, precision, recall or F1 on risk classification.

    include_original_15 keeps the 15 hand-labelled profiles that predate
    the rubric in the returned set, tagged
    provenance="hand_labelled_original_15". They are preserved rather
    than replaced: they are the only labels in the project produced by a
    human reading a profile, and the rubric's agreement with them is
    itself a validity check on the rubric (reported in the gold file's
    `rubric_vs_original_15_agreement` block).
    """
    payload = _read_json(GOLD_PROFILES_PATH, "Gold-standard risk profiles")

    records = list(payload["profiles"])
    if not include_original_15:
        records = [
            r for r in records
            if r.get("provenance") != "hand_labelled_original_15"
        ]

    return EvaluationDataset(
        layer=LAYER_GOLD,
        name=payload["dataset_name"],
        records=records,
        provenance=payload["provenance"],
        source_path=GOLD_PROFILES_PATH,
    )


def load_stress_customers() -> EvaluationDataset:
    """
    Layer 2. Robustness only.

    Note what this function does NOT do: it does not copy
    `construction_target_class` into a `ground_truth_risk_class` field.
    The construction target records which rubric cell the profile was
    sampled to fill; treating it as an observation would make the
    "evaluation" a test of the sampler.
    """
    payload = _read_json(STRESS_CUSTOMERS_PATH, "Synthetic stress customers")
    return EvaluationDataset(
        layer=LAYER_STRESS,
        name=payload["dataset_name"],
        records=list(payload["customers"]),
        provenance=payload["provenance"],
        source_path=STRESS_CUSTOMERS_PATH,
    )


def load_stress_transactions() -> dict[str, dict[str, Any]]:
    """Transaction histories keyed by customer_id, for the Layer-2 set."""
    payload = _read_json(STRESS_TRANSACTIONS_PATH, "Stress transactions")
    return payload["transactions"]


def load_real_customers() -> EvaluationDataset:
    """
    Layer 1. The operational CustomerStore population, read directly from
    the CSV rather than through CustomerStore so that loading it for
    evaluation can never write to it.

    Records are returned in the same {"customer_id", "features", ...}
    shape as the other two layers so a robustness harness can iterate all
    three identically.
    """
    if not REAL_CUSTOMERS_PATH.exists():
        raise FileNotFoundError(f"No customers.csv at {REAL_CUSTOMERS_PATH}")

    feature_names = [
        "age", "income", "employment_status", "dependents",
        "existing_debt", "investment_horizon", "loss_tolerance",
        "financial_knowledge_score",
    ]
    numeric = {
        "age", "income", "dependents", "existing_debt",
        "investment_horizon", "loss_tolerance", "financial_knowledge_score",
    }

    records: list[dict[str, Any]] = []
    n_labelled = 0
    sources: dict[str, int] = {}

    with open(REAL_CUSTOMERS_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            features: dict[str, Any] = {}
            for name in feature_names:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    continue
                if name in numeric:
                    try:
                        features[name] = float(raw) if name in {"income", "existing_debt"} else int(float(raw))
                    except ValueError:
                        continue
                else:
                    features[name] = raw

            label = (row.get("ground_truth_risk_class") or "").strip()
            if label:
                n_labelled += 1
            source = (row.get("source_dataset") or "unknown").strip()
            sources[source] = sources.get(source, 0) + 1

            records.append({
                "customer_id": row.get("customer_id"),
                "features": features,
                "source_dataset": source,
                # Deliberately NOT called ground_truth_risk_class: these
                # few labels are hand-completed demo profiles, not an
                # independently-labelled evaluation sample, and must not
                # be swept into an accuracy computation by a field-name
                # match. See phase8b_existing_customer_baseline.json for
                # the flow test that legitimately uses them.
                "demo_label_if_present": label or None,
                "n_features_present": len(features),
                "complete": len(features) == len(feature_names),
            })

    return EvaluationDataset(
        layer=LAYER_REAL,
        name="operational_customer_store",
        records=records,
        provenance={
            "source": "data/processed/customers.csv",
            "n_records": len(records),
            "n_with_any_demo_label": n_labelled,
            "n_without_label": len(records) - n_labelled,
            "source_dataset_counts": sources,
            "note": (
                "Operational population. 251 of 254 records have no risk "
                "label of any kind. No labels were invented for them. "
                "Used for robustness and coverage only."
            ),
        },
        source_path=REAL_CUSTOMERS_PATH,
    )


def layer_summary() -> dict[str, Any]:
    """
    A compact three-layer inventory, embedded in results files and printed
    by the evaluation scripts so the separation is visible in the evidence
    itself and not only in this docstring.
    """
    summary: dict[str, Any] = {"layers": {}}
    for loader, layer in (
        (load_real_customers, LAYER_REAL),
        (load_stress_customers, LAYER_STRESS),
        (load_gold_risk_profiles, LAYER_GOLD),
    ):
        try:
            ds = loader()
            summary["layers"][layer.key] = {
                "name": ds.name,
                "n": len(ds),
                **layer.to_dict(),
            }
        except FileNotFoundError as exc:
            summary["layers"][layer.key] = {
                "name": layer.name, "n": 0, "error": str(exc).split("\n")[0],
                **layer.to_dict(),
            }
    return summary
