"""
Phase 7 - Self-healing failure detection and recovery.

AgentFixer principle: a failed agent should not bring down
the entire pipeline. Instead, a recovery strategy is applied and the
result is flagged as recovered (not hidden). The audit log records
both the failure and recovery attempt.

Recovery strategies per agent:
  RiskProfilingAgent → heuristic_fallback
    Use the heuristic ML proxy score with conservative bias (moderate tier).
    Confidence is forced to 0.4 to trigger the low-confidence flag in
    ExplainabilityAgent (X3 — calibrated trust, Takayanagi et al. [7]).

  InvestmentAgent → static_shortlist
    Return the top-1 safest product for the user's risk class from the
    static catalogue. No LLM synthesis — plain text fallback.

  BudgetAgent → skip
    Budget analysis is supplementary. If it fails, pipeline continues
    without it. The audit log records the skip.

  ExplainabilityAgent → minimal_explanation
    Return a minimal calibration note only (Layer C skipped, Layer A skipped).
    X3 (calibration note) is always deliverable without LLM — use fallback.

  ConversationalAgent → direct_response
    Return a fixed clarification request. Never raises.
"""
from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)

class FailureHandler:
    """
    Stateless recovery handler. Called by Orchestrator when an agent raises.
    Returns a recovery payload that substitutes for the failed agent's output.
    """

    CONSERVATIVE_FALLBACK_PRODUCT = {
        "product_id": "SAV001_FALLBACK",
        "name": "An Post Savings Bond (fallback)",
        "category": "savings_account",
        "expected_return_pct": 2.5,
        "expense_ratio_pct": 0.0,
        "score": 0.0,
        "fallback": True,
    }

    def attempt_recovery(
        self,
        agent_name: str,
        error: Exception,
        context: dict,
    ) -> dict:
        """
        Attempt recovery for a failed agent call.

        Returns a recovery payload dict with:
          strategy        — name of recovery strategy applied
          success         — whether recovery produced usable output
          recovered_payload — substitute output for the failed agent
          original_error  — str representation of the original exception
        """
        error_str = str(error)
        logger.warning(
            f"[FailureHandler] Recovering from {agent_name} failure: {error_str}"
        )

        strategies = {
            "RiskProfilingAgent":   self._recover_risk,
            "InvestmentAgent":      self._recover_investment,
            "BudgetAgent":          self._recover_budget,
            "ExplainabilityAgent":  self._recover_explainability,
            "ConversationalAgent":  self._recover_conversational,
        }

        handler = strategies.get(agent_name, self._recover_generic)
        recovery = handler(context, error_str)
        recovery["original_error"] = error_str

        logger.info(
            f"[FailureHandler] {agent_name} recovery: "
            f"strategy={recovery['strategy']} success={recovery['success']}"
        )
        return recovery

    def _recover_risk(self, context: dict, error: str) -> dict:
        """
        Heuristic fallback for RiskProfilingAgent.
        Conservative bias + low confidence to trigger X3 uncertainty flag.
        """
        return {
            "strategy": "heuristic_fallback",
            "success": True,
            "recovered_payload": {
                "status": "complete",
                "risk_class": "moderate",
                "ml_score": 0.5,
                "rule_score": 0.5,
                "hybrid_score": 0.5,
                "confidence": 0.4,  # deliberately low → X3 uncertainty flag
                "confidence_flag": (
                    "LOW_CONFIDENCE — risk profiling failed; using moderate "
                    "as conservative fallback. Please retry."
                ),
                "feature_importance": {},
                "rationale": (
                    "Risk profiling encountered an error. A moderate risk "
                    "classification has been applied as a conservative fallback. "
                    "This should be treated as indicative only — please retry "
                    "or consult a qualified advisor."
                ),
                "missing_features": [],
                "recovered": True,
            },
        }

    def _recover_investment(self, context: dict, error: str) -> dict:
        """
        Static shortlist fallback — safest product for context risk class.
        """
        risk_class = (
            (context.get("risk_agent_payload") or {}).get("risk_class", "conservative")
        )
        return {
            "strategy": "static_shortlist",
            "success": True,
            "recovered_payload": {
                "status": "complete",
                "risk_class": risk_class,
                "shortlist": [self.CONSERVATIVE_FALLBACK_PRODUCT],
                "synthesis": (
                    "Investment recommendation service encountered an error. "
                    "A conservative savings option has been shown as a fallback. "
                    "This is not a personalised recommendation — please retry. "
                    "This is not regulated financial advice. "
                    "Consult a qualified advisor. "
                    "Past performance is not indicative of future results."
                ),
                "recovered": True,
            },
        }

    def _recover_budget(self, context: dict, error: str) -> dict:
        """Budget is supplementary — skip gracefully."""
        return {
            "strategy": "skip",
            "success": True,
            "recovered_payload": {
                "status": "skipped",
                "message": (
                    "Budget analysis is temporarily unavailable. "
                    "Other advisory services are unaffected."
                ),
                "recovered": True,
            },
        }

    def _recover_explainability(self, context: dict, error: str) -> dict:
        """Minimal calibration note only — X3 requirement always deliverable."""
        return {
            "strategy": "minimal_explanation",
            "success": True,
            "recovered_payload": {
                "layers_applied": ["calibration_note"],
                "ablation_condition": "fallback",
                "shap_narrative": None,
                "rag_citations": [],
                "counterfactual": None,
                "calibration_note": (
                    "This recommendation is based on the financial profile "
                    "provided. Circumstances change — reassessment is recommended "
                    "if your income, employment, or goals change significantly."
                ),
                "full_explanation": (
                    "This recommendation is based on the financial profile "
                    "provided. Circumstances change — reassessment is recommended "
                    "if your income, employment, or goals change significantly."
                ),
                "confidence": 0.5,
                "low_confidence_flagged": True,
                "recovered": True,
            },
        }

    def _recover_conversational(self, context: dict, error: str) -> dict:
        return {
            "strategy": "direct_response",
            "success": True,
            "recovered_payload": {
                "intent": "general_query",
                "confidence": 0.5,
                "escalation_needed": False,
                "response": (
                    "I'm sorry, I encountered an issue. "
                    "Could you please rephrase your question?"
                ),
                "collected_slots": {},
                "turn_count": 1,
                "escalation_block": None,
                "recovered": True,
            },
        }

    def _recover_generic(self, context: dict, error: str) -> dict:
        return {
            "strategy": "none",
            "success": False,
            "recovered_payload": {
                "status": "error",
                "message": "Agent failed and no recovery strategy is available.",
                "recovered": False,
            },
        }

