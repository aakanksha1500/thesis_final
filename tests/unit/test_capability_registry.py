"""
Capability registry tests (Day 3 / R3 extension).

WHY THIS FILE EXISTS
    CAPABILITIES is the planner's (Day 4-5) entire view of the world. A
    capability graph with an unsatisfiable `requires` — a key nothing in
    the graph produces and nothing external supplies — is a bug the
    planner would otherwise discover at runtime, as a plan that PlanValidator
    rejects on every single turn for the life of the project, with the
    root cause invisible in a rejection-rate metric.

    These tests are the static check that catches that class of error at
    commit time, before there's a planner to hit it.

RUNNING
    python -m pytest tests/unit/test_capability_registry.py -v
"""
from __future__ import annotations

from agents.payloads import (
    CAPABILITIES,
    ORCHESTRATOR_SUPPLIED_KEYS,
    AgentCapability,
    validate_capability_graph,
)


class TestCapabilityGraphIsSatisfiable:

    def test_no_unsatisfiable_requirement(self):
        problems = validate_capability_graph()
        assert not problems, (
            "capability graph has requirements nothing can satisfy:\n"
            + "\n".join(problems)
        )

    def test_every_requires_key_is_a_produces_key_or_orchestrator_supplied(self):
        """Same check, written directly against the two sets — a second
        angle on test_no_unsatisfiable_requirement so a future edit to
        validate_capability_graph() can't silently stop checking anything."""
        produced = {k for cap in CAPABILITIES.values() for k in cap.produces}
        satisfiable = produced | ORCHESTRATOR_SUPPLIED_KEYS
        for cap in CAPABILITIES.values():
            missing = cap.requires - satisfiable
            assert not missing, f"{cap.name} requires {missing}, unsatisfiable"


class TestCapabilityRegistryShape:

    def test_dict_key_matches_capability_name(self):
        for key, cap in CAPABILITIES.items():
            assert cap.name == key, f"{key!r} maps to a capability named {cap.name!r}"

    def test_every_agent_in_payload_contracts_has_a_capability(self):
        """PAYLOAD_CONTRACTS and CAPABILITIES describe the same five agents
        from two angles (output shape vs. inputs/outputs-as-context-keys).
        A name in one and not the other means someone updated a contract
        and forgot the capability declaration, or vice versa."""
        from agents.payloads import PAYLOAD_CONTRACTS
        assert set(PAYLOAD_CONTRACTS) == set(CAPABILITIES)

    def test_cost_hint_is_one_of_the_documented_values(self):
        for cap in CAPABILITIES.values():
            assert cap.cost_hint in {"cheap", "llm", "expensive"}, (
                f"{cap.name} has an undocumented cost_hint {cap.cost_hint!r}"
            )

    def test_capability_is_frozen(self):
        """The planner reasons over this graph; it must not be mutable
        out from under it mid-turn."""
        cap = CAPABILITIES["RiskProfilingAgent"]
        try:
            cap.requires = frozenset()
        except Exception as exc:
            assert isinstance(exc, (AttributeError, TypeError))
        else:
            raise AssertionError("AgentCapability should be frozen")


class TestSpecificDependencies:
    """Assertions on individual agents, so a future edit that silently
    narrows a capability's requirements fails loudly here instead of
    only showing up as a planner rejecting valid plans."""

    def test_investment_agent_requires_risk_class(self):
        assert "risk_class" in CAPABILITIES["InvestmentAgent"].requires

    def test_investment_agent_requires_user_features(self):
        """run() reads context['user_features'] directly for horizon/age —
        not just the risk tier RiskProfilingAgent hands it."""
        assert "user_features" in CAPABILITIES["InvestmentAgent"].requires

    def test_risk_class_is_produced_by_exactly_one_agent(self):
        producers = [
            name for name, cap in CAPABILITIES.items()
            if "risk_class" in cap.produces
        ]
        assert producers == ["RiskProfilingAgent"]

    def test_explainability_agent_has_no_hard_requirement(self):
        """It must be runnable last regardless of what upstream produced —
        or didn't — see explainability_agent.py's `... or {}` reads."""
        assert CAPABILITIES["ExplainabilityAgent"].requires == frozenset()

    def test_conversational_agent_has_no_hard_requirement(self):
        assert CAPABILITIES["ConversationalAgent"].requires == frozenset()

        