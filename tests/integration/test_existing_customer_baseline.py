"""
tests/integration/test_existing_customer_baseline.py

Phase 8b (Commit 43) — cumulative baseline for the existing-customer
pipeline. Runs each demo profile (data/demo_customers.json) through the
real lookup + RiskProfilingAgent path and compares the predicted
risk_class against the hand-labelled ground_truth_risk_class.

Writes: results/phase8b_existing_customer_baseline.json

Run:
    pytest tests/integration/test_existing_customer_baseline.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

from data.customer_store import CustomerStore
from evaluation.results_io import write_results
from orchestrator.orchestrator import Orchestrator
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

DEMO_CUSTOMER_IDS = ["DEMO_GC_042", "DEMO_GC_107", "DEMO_GMSC_318"]


class TestExistingCustomerBaseline:
    def test_demo_profiles_full_fixture_evaluation_and_write_results(self):
        store = CustomerStore()
        llm = LLMClient()

        per_profile = []
        correct = 0

        for customer_id in DEMO_CUSTOMER_IDS:
            orch = Orchestrator(
                llm,
                session_id=f"baseline-{customer_id}",
                customer_id=customer_id,
                customer_store=store,
            )
            assert orch._session_state["customer_known"] is True
            assert orch._session_state["missing_customer_fields"] == []

            risk_result = orch._agents["RiskProfilingAgent"].run(
                {"user_features": orch._session_state["user_features"]}
            )
            predicted = risk_result.payload.get("risk_class")
            expected = orch._session_state["ground_truth_risk_class"]
            is_correct = predicted == expected
            correct += int(is_correct)

            per_profile.append({
                "customer_id": customer_id,
                "predicted_risk_class": predicted,
                "ground_truth_risk_class": expected,
                "correct": is_correct,
                "status": risk_result.payload.get("status"),
            })

        agreement_rate = correct / len(DEMO_CUSTOMER_IDS)

        results_payload = {
            "phase": "8b",
            "component": "existing_customer_pipeline",
            "dataset": "demo_customers.json (3 profiles)",
            "metrics": {
                "predicted_vs_ground_truth_agreement": agreement_rate,
                "n_correct": correct,
                "n_total": len(DEMO_CUSTOMER_IDS),
            },
            "per_profile": per_profile,
        }
        results_path = write_results(
            results_payload, "phase8b_existing_customer_baseline.json", llm.mode
        )

        print(f"\n[Phase 8b] Demo profile agreement: {correct}/{len(DEMO_CUSTOMER_IDS)}")
        print(f"[Phase 8b] Results written to {results_path}")

        # This is a fixture-quality gate, not a strict pass/fail RQ metric —
        # demo profiles were hand-labelled to be internally consistent, so
        # agreement should be perfect. A failure here means either the demo
        # fixture or RiskProfilingAgent's scoring drifted — investigate,
        # don't just relax the assertion.
        assert agreement_rate == 1.0, (
            f"Demo profile agreement dropped to {agreement_rate:.2f} — "
            f"see per_profile in {results_path} for which profile mismatched."
        )
