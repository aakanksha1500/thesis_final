"""
Phase 7 — Agent-as-a-Judge evaluator

single-output LLM evaluation misses routing errors,
agent failures, and constraint violations because post-hoc rationalisations
can disguise them in the final answer. The Agent-as-Judge evaluates the
FULL reasoning trajectory — routing decision, agent sequence, payloads,
conflicts, violations — not just the final response text.

ive evaluation dimensions (from JUDGE_SYSTEM prompt in config/prompts.py):
  1. routing_accuracy    — correct routing decision for the query?
  2. agent_coordination  — correct sequence, correct inputs passed?
  3. factual_accuracy    — financial figures correct? regulatory statements?
  4. explanation_quality — SHAP/counterfactual/calibration coherent?
  5. trust_calibration   — appropriate uncertainty conveyed?

Output feeds into:
  - results/rq4_mas_coherence.json (Component Synergy Score, RQ4)
  - results/rq4_agent_judge_scores.json (per-turn Judge scores)

In mock mode: Judge LLM returns [MOCK RESPONSE] → JSON parse fails →
fallback scores are used (all 3.0/5.0, verdict='flag').
In real mode: actual GPT/Gemini evaluation scores.
"""
from __future__ import annotations

import json
import re
from typing import Any

from config.prompts import JUDGE_DIMENSIONS, JUDGE_SYSTEM
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_SCORE = 3.0   # neutral score used when Judge LLM unavailable
SCORE_DIMENSIONS = [name for name, _ in JUDGE_DIMENSIONS]

class AgentJudge:
    """
    Evaluates full OrchestratorResult trajectories using an LLM judge.
    Uses the judge_model (settings.llm.judge_model) — highest quality
    model available, since judge quality determines evaluation validity.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self._eval_count = 0

    def evaluate(
        self,
        orchestrator_result: Any,           # OrchestratorResult
        ground_truth: dict | None = None,
    ) -> dict:
        """
        Evaluate one OrchestratorResult trajectory.

        Args:
            orchestrator_result: OrchestratorResult from Orchestrator.process_turn()
            ground_truth:        Optional dict with expected_routing, expected_agents,
                                 expected_risk_class for verifiable queries.

        Returns:
            Dict with per-dimension scores, overall_score, verdict, reasoning.
        """
        trajectory = self._build_trajectory(orchestrator_result, ground_truth)
        scores = self._call_judge(trajectory)
        self._eval_count += 1
        return scores

    def evaluate_batch(
        self,
        orchestrator_results: list[Any],
        ground_truths: list[dict | None] | None = None,
    ) -> list[dict]:
        """Evaluate a list of OrchestratorResults. Returns one score dict per result."""
        gts = ground_truths or [None] * len(orchestrator_results)
        return [
            self.evaluate(result, gt)
            for result, gt in zip(orchestrator_results, gts)
        ]

    def _build_trajectory(
        self,
        result: Any,
        ground_truth: dict | None,
    ) -> dict:
        """Serialise OrchestratorResult to a JSON-serialisable trajectory dict."""
        agent_payloads = []
        for r in result.agent_results:
            # Include only key fields — not raw LLM output (too long)
            key_fields = {
                k: v for k, v in r.payload.items()
                if k in (
                    "status", "risk_class", "confidence", "rationale",
                    "shortlist", "synthesis", "full_explanation",
                    "layers_applied", "savings_rate_pct",
                    "intent", "escalation_needed",
                )
            }
            agent_payloads.append({
                "agent": r.agent_name,
                "success": r.success,
                "payload_summary": key_fields,
                "error": r.error,
            })

        return {
            "routing_decision": result.routing_decision.value
            if hasattr(result.routing_decision, "value")
            else str(result.routing_decision),
            "agents_invoked": result.agents_invoked,
            "agent_payloads": agent_payloads,
            "final_response": result.final_response,
            "conflicts": result.conflicts,
            "constraint_violations": result.constraint_violations,
            "recovered_agents": result.recovered_agents,
            "ground_truth": ground_truth or {},
        }

    def _call_judge(self, trajectory: dict) -> dict:
        """
        Call the Judge LLM with the trajectory.
        Returns parsed score dict or fallback scores on failure.
        """
        prompt = (
            f"Evaluate this multi-agent financial advisory turn trajectory:\n\n"
            f"{json.dumps(trajectory, indent=2)}\n\n"
            f"Return ONLY the JSON score object, no other text."
        )
        try:
            response = self.llm.chat(
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return self._parse_judge_response(response.content)
        except Exception as exc:
            logger.warning(f"[AgentJudge] LLM call failed: {exc} — using fallback scores")
            return self._fallback_scores(str(exc))

    def _parse_judge_response(self, raw: str) -> dict:
        """
        Extract JSON from Judge LLM response.
        Handles cases where LLM adds surrounding text.
        Falls back to fallback_scores if parse fails.
        """
        # Try to find JSON object in response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group())
                # Validate required fields
                for dim in SCORE_DIMENSIONS:
                    if dim not in scores:
                        scores[dim] = FALLBACK_SCORE
                if "overall_score" not in scores:
                    scores["overall_score"] = sum(
                        scores.get(d, FALLBACK_SCORE) for d in SCORE_DIMENSIONS
                    ) / len(SCORE_DIMENSIONS)
                if "verdict" not in scores:
                    scores["verdict"] = (
                        "pass" if scores["overall_score"] >= 3.5
                        else "flag" if scores["overall_score"] >= 2.5
                        else "fail"
                    )
                scores["judge_mode"] = "real"
                return scores
            except json.JSONDecodeError:
                pass

        logger.debug(
            f"[AgentJudge] Could not parse JSON from response: {raw[:100]}"
        )
        return self._fallback_scores("JSON parse failed")

    def _fallback_scores(self, reason: str) -> dict:
        """
        Neutral fallback scores when Judge LLM is unavailable.
        Verdict 'flag' distinguishes mock from real evaluations.
        """
        return {
            dim: FALLBACK_SCORE for dim in SCORE_DIMENSIONS
        } | {
            "overall_score": FALLBACK_SCORE,
            "verdict": "flag",
            "reasoning": f"Fallback scores — Judge LLM unavailable: {reason}",
            "judge_mode": "mock",
        }

    @property
    def eval_count(self) -> int:
        return self._eval_count
