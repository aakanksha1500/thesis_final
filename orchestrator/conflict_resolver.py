"""
Phase 7 - Cross-agent conflict detection and resolution.

Committed before the Orchestrator so it is testable in isolation.
The Orchestrator calls resolve() after all agents have run.

Conflict types detected:
    RISK_PRODUCT_MISMATCH - InvestmentAgent shortlist contains a product
    category incompatible with the RiskProfilingAgent's risk class.
    Resolution: remove incompatible products from shortlist.
    Grounded in CBI suitability rules (config/constraints.py)

    LOW_CONFIDENCE_AGGRESSIVE - RiskProfileAgent confidence is below
    min_confidence threshold but routig selected an aggressive product.
    Resoution: downgrade shortlist to the next more conservative tier.

    MISSING_RISK_BEFORE_INVESTMENT - InvestmentAgent ran without a
    confirmed risk class. This should not happen if Orchestrator routing
    is correct, but is detected defensively here.
    Resolution: mark InvestmentAgent result as undeliverable.

All conflicts are returned as a list of dicts for the audit log.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from agents.base_agent import AgentResult
from config.constraints import financial_constraints
from config.settings import settings
from utils import trace
from utils.logger import get_logger

logger = get_logger(__name__)

class ConflictResolver:
    """
    Stateless conflict checker - all methods are pure functions.
    Called by Orchestrator after the agent execution phase.
    """

    def resolve(
            self, agent_results: list[AgentResult]
    ) -> tuple[list[AgentResult], list[dict]]:
        """
        Check all agent results for cross-agent conflict.
        Returns (cleaned_results, conflict_log).
        Modifies result in-place where resolution requires it.
        """
        conflicts: list[dict] = []

        agent_results = [
            replace(r, payload=deepcopy(r.payload))
            for r in agent_results
        ]

        risk_result = self._find_result(agent_results, "RiskProfilingAgent")
        inv_result = self._find_result(agent_results, "InvestmentAgent")

        # Check 1: Investment ran without risk class
        if inv_result and inv_result.success:
            if not risk_result or not risk_result.success:
                conflict = {
                    "type": "MISSING_RISK_BEFORE_INVESTMENT",
                    "description": (
                        "InvestmentAgent produced output without a confirmed "
                        "risk class from RiskProfilingAgent."
                    ),
                    "resolution": "InvestmentAgent output marked undeliverable.",
                }
                inv_result.payload["deliverable"] = False
                inv_result.payload["undeliverable_reason"] = (
                    "Risk classification required before investment recommendation."
                )
                conflicts.append(conflict)
                logger.warning(
                    "[ConflictResolver] MISSING_RISK_BEFORE_INVESTMENT detected"
                )

        # Check 2: Risk-product compatibility
        if (risk_result and risk_result.success and
                inv_result and inv_result.success and
                inv_result.payload.get("deliverable", True)):

            risk_class = risk_result.payload.get("risk_class", "moderate")
            shortlist: list[dict] = inv_result.payload.get("shortlist", [])
            cleaned_shortlist = []

            for product in shortlist:
                category = product.get("category", "")
                violation = financial_constraints.check_risk_product_compatibility(
                    risk_class, category
                )
                if violation:
                    conflict = {
                        "type": "RISK_PRODUCT_MISMATCH",
                        "description": violation.description,
                        "resolution": (
                            f"Product '{product.get('name', category)}' "
                            f"removed from shortlist."
                        ),
                        "product_removed": product.get("name", category),
                    }
                    conflicts.append(conflict)
                    logger.warning(
                        f"[ConflictResolver] RISK_PRODUCT_MISMATCH: "
                        f"removed '{product.get('name')}' "
                        f"(category={category}, risk_class={risk_class})"
                    )
                else:
                    cleaned_shortlist.append(product)

            if len(cleaned_shortlist) != len(shortlist):
                inv_result.payload["shortlist"] = cleaned_shortlist
                inv_result.payload["shortlist_cleaned"] = True

        # Check 3: Low confidence + aggressive routing
        if risk_result and risk_result.success:
            confidence = risk_result.payload.get("confidence", 1.0)
            risk_class = risk_result.payload.get("risk_class", "moderate")
            if (confidence < settings.risk.min_confidence and
                    risk_class in ("moderately_aggressive", "aggressive")):
                conflict = {
                    "type": "LOW_CONFIDENCE_AGGRESSIVE",
                    "description": (
                        f"Risk class '{risk_class}' assigned with low confidence "
                        f"({confidence:.0%} < {settings.risk.min_confidence:.0%}). "
                        f"Downgrading to 'moderate' for investment routing."
                    ),
                    "resolution": "Risk class downgraded to 'moderate' for this turn.",
                }
                risk_result.payload["risk_class_original"] = risk_class
                risk_result.payload["risk_class"] = "moderate"
                conflicts.append(conflict)
                logger.warning(
                    f"[ConflictResolver] LOW_CONFIDENCE_AGGRESSIVE: "
                    f"downgraded {risk_class} → moderate (conf={confidence:.2f})"
                )
        for c in conflicts:
            trace.emit("⚠ CONFLICT", c["type"], resolution=c["resolution"])
        return agent_results, conflicts

    def _find_result(
        self, results: list[AgentResult], agent_name: str
    ) -> AgentResult | None:
        return next(
            (r for r in results if r.agent_name == agent_name), None
        )
