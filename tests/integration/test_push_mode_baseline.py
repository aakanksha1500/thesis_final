"""
Phase 8b — cumulative baseline covering: context-push mode and the estimated-input
disclosure layer. 

Writes: results/phase8b_push_mode_baseline.json

Run:
    pytest tests/integration/test_push_mode_baseline.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.results_io import write_results
from orchestrator.orchestrator import Orchestrator
from utils.llm_client import LLMClient

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Synthetic bank-API payloads — deliberately omit loss_tolerance and
# financial_knowledge_score (no bank sends either), spanning different
# capacity profiles so the proxy is exercised across its range.
PUSH_MODE_FIXTURES = [
    {
        "label": "young_long_horizon_no_dependents",
        "customer_id": "PUSH_001",
        "context": {
            "name": "Fixture A", "age": 26, "income": 42000,
            "employment_status": "employed", "dependents": 0,
            "existing_debt": 2000, "investment_horizon": 20,
        },
        "expect_proxy_direction": "high",  # expect loss_tolerance proxy >= 3
    },
    {
        "label": "older_short_horizon_dependents",
        "customer_id": "PUSH_002",
        "context": {
            "name": "Fixture B", "age": 58, "income": 55000,
            "employment_status": "employed", "dependents": 3,
            "existing_debt": 30000, "investment_horizon": 2,
        },
        "expect_proxy_direction": "low",  # expect loss_tolerance proxy <= 3
    },
    {
        "label": "mid_profile_all_fields_supplied",
        "customer_id": "PUSH_003",
        "context": {
            "name": "Fixture C", "age": 40, "income": 60000,
            "employment_status": "employed", "dependents": 1,
            "existing_debt": 10000, "investment_horizon": 8,
            "loss_tolerance": 3,  # bank-equivalent supplies it this time
        },
        "expect_proxy_direction": None,  # no proxy should fire — value was supplied
    },
]


class TestPushModeBaseline:
    def test_push_mode_and_disclosure_fixture_evaluation(self):
        llm = LLMClient()
        per_fixture = []
        proxy_fired_count = 0
        disclosure_fired_count = 0

        for fixture in PUSH_MODE_FIXTURES:
            orch = Orchestrator(
                llm,
                session_id=f"baseline-push-{fixture['customer_id']}",
                customer_id=fixture["customer_id"],
                customer_context=fixture["context"],
            )

            proxy_fields = orch._session_state["proxy_fields"]
            proxy_metadata = orch._session_state["proxy_metadata"]
            proxy_fired = "loss_tolerance" in proxy_fields

            context = orch._build_context("What should I invest in?")
            risk_result = orch._agents["RiskProfilingAgent"].run(context)
            context["risk_agent_payload"] = risk_result.payload

            exp_result = orch._agents["ExplainabilityAgent"].run(context)
            disclosure_fired = (
                "estimated_input_disclosure" in exp_result.payload["layers_applied"]
            )

            # Consistency check: disclosure should fire iff a proxy fired
            assert disclosure_fired == proxy_fired, (
                f"{fixture['label']}: proxy_fired={proxy_fired} but "
                f"disclosure_fired={disclosure_fired} — these must always match, "
                f"otherwise an estimate is silently going undisclosed."
            )

            if proxy_fired:
                proxy_fired_count += 1
                proxy_value = proxy_metadata["loss_tolerance"]["value"]
                if fixture["expect_proxy_direction"] == "high":
                    assert proxy_value >= 3, (
                        f"{fixture['label']}: expected high capacity proxy, got {proxy_value}"
                    )
                elif fixture["expect_proxy_direction"] == "low":
                    assert proxy_value <= 3, (
                        f"{fixture['label']}: expected low capacity proxy, got {proxy_value}"
                    )
            if disclosure_fired:
                disclosure_fired_count += 1

            per_fixture.append({
                "label": fixture["label"],
                "customer_id": fixture["customer_id"],
                "proxy_fired": proxy_fired,
                "proxy_value": proxy_metadata.get("loss_tolerance", {}).get("value"),
                "proxy_confidence": proxy_metadata.get("loss_tolerance", {}).get("confidence"),
                "disclosure_fired": disclosure_fired,
                "risk_status": risk_result.payload.get("status"),
                "risk_class": risk_result.payload.get("risk_class"),
            })

        results_payload = {
            "phase": "8b",
            "component": "push_mode_and_disclosure",
            "commits_covered": [45, 46, 47],
            "dataset": "synthetic push-mode fixtures (3 profiles)",
            "metrics": {
                "n_total": len(PUSH_MODE_FIXTURES),
                "n_proxy_fired": proxy_fired_count,
                "n_disclosure_fired": disclosure_fired_count,
                "proxy_disclosure_consistency": disclosure_fired_count == proxy_fired_count,
            },
            "per_fixture": per_fixture,
        }
        results_path = write_results(
            results_payload, "phase8b_push_mode_baseline.json"
        )

        print(f"\n[Phase 8b] Proxy fired: {proxy_fired_count}/{len(PUSH_MODE_FIXTURES)}")
        print(f"[Phase 8b] Disclosure fired: {disclosure_fired_count}/{len(PUSH_MODE_FIXTURES)}")
        print(f"[Phase 8b] Results written to {results_path}")
