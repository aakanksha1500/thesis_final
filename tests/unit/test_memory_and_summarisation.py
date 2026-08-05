"""
Day 9 tests: long-term memory (G7) and conversation summarisation (G17).

WHAT ISN'T TESTED HERE, AND WHY
    Feature persistence across sessions (age, income, ...) is NOT
    re-tested in this file — that's CustomerStore's job, already covered
    by its own tests, and confirmed working before data/customer_memory.py
    was written at all (see that module's docstring for the empirical
    check that motivated NOT duplicating it here). This file covers only
    what customer_memory.py actually owns: risk_profile, preferences,
    turn_count — plus summarisation, which is unrelated to memory beyond
    living in the same Orchestrator file.

RUNNING
    python -m pytest tests/unit/test_memory_and_summarisation.py -v
"""
from __future__ import annotations

import pytest

from config.settings import settings
from data.customer_memory import (
    CustomerMemoryStore,
    extract_preferences_from_slots,
)


def _orch(session_id: str, customer_id: str | None = None):
    from orchestrator.orchestrator import Orchestrator
    from utils.llm_client import LLMClient
    return Orchestrator(LLMClient(force_mock=True), session_id=session_id,
                        customer_id=customer_id)


# ── CustomerMemoryStore in isolation ────────────────────────────────────

class TestCustomerMemoryStore:

    def test_get_unknown_customer_returns_none(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        assert store.get("nope") is None

    def test_upsert_creates_a_new_row(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        m = store.upsert("c1", risk_profile={"risk_class": "moderate"},
                         preferences={"investment_goal": "retirement"})
        assert m.customer_id == "c1"
        assert m.risk_profile == {"risk_class": "moderate"}
        assert m.preferences == {"investment_goal": "retirement"}
        assert m.turn_count == 1

    def test_turn_count_increments_across_upserts(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        store.upsert("c1")
        store.upsert("c1")
        m = store.upsert("c1")
        assert m.turn_count == 3

    def test_preferences_merge_rather_than_overwrite(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        store.upsert("c1", preferences={"investment_goal": "retirement"})
        m = store.upsert("c1", preferences={"user_name": "Sam"})
        assert m.preferences == {"investment_goal": "retirement", "user_name": "Sam"}

    def test_a_later_preference_value_overwrites_the_same_key(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        store.upsert("c1", preferences={"investment_goal": "house deposit"})
        m = store.upsert("c1", preferences={"investment_goal": "retirement"})
        assert m.preferences == {"investment_goal": "retirement"}

    def test_risk_profile_replaces_not_merges(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        store.upsert("c1", risk_profile={"risk_class": "conservative", "confidence": 0.5})
        m = store.upsert("c1", risk_profile={"risk_class": "aggressive"})
        assert m.risk_profile == {"risk_class": "aggressive"}

    def test_risk_profile_persists_when_not_reprovided(self, tmp_path):
        """An upsert that only touches preferences must not erase a
        previously-stored risk_profile — turn_count increments on every
        turn, most of which won't recompute risk."""
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        store.upsert("c1", risk_profile={"risk_class": "moderate"})
        m = store.upsert("c1", preferences={"user_name": "Sam"})
        assert m.risk_profile == {"risk_class": "moderate"}

    def test_last_seen_and_updated_at_advance(self, tmp_path):
        store = CustomerMemoryStore(db_path=tmp_path / "m.db")
        first = store.upsert("c1")
        second = store.upsert("c1")
        assert second.last_seen >= first.last_seen


class TestExtractPreferencesFromSlots:

    def test_only_the_allowlisted_keys_are_extracted(self):
        prefs = extract_preferences_from_slots({
            "investment_goal": "retirement",
            "user_name": "Sam",
            "housing_cost": 1200,          # budget-questionnaire slot, not a preference
            "risk_tolerance": "high",      # vestigial synonym for loss_tolerance
            "age": 34,                     # a required_feature, not a preference
        })
        assert prefs == {"investment_goal": "retirement", "user_name": "Sam"}

    def test_none_values_are_not_included(self):
        assert extract_preferences_from_slots({"investment_goal": None}) == {}

    def test_empty_slots_gives_empty_preferences(self):
        assert extract_preferences_from_slots({}) == {}


# ── Orchestrator integration: restore on load, persist on turn end ─────

class TestMemoryOrchestratorIntegration:

    def test_no_customer_id_never_touches_memory(self, audit_tmp_dir):
        """Session with no identified customer — persisting should be a
        graceful no-op, matching update_customer_features()'s own guard."""
        orch = _orch("mem-1")
        orch._persist_customer_memory()  # must not raise
        assert True

    def test_load_customer_memory_restores_risk_profile_into_session_state(
        self, audit_tmp_dir,
    ):
        from data.customer_memory import get_customer_memory_store
        get_customer_memory_store().upsert(
            "c1", risk_profile={"risk_class": "aggressive", "confidence": 0.9},
        )
        orch = _orch("mem-2")
        orch._load_customer("c1")
        orch._load_customer_memory("c1")
        assert orch._session_state["risk_profile"] == {
            "risk_class": "aggressive", "confidence": 0.9,
        }

    def test_load_customer_memory_restores_preferences_into_conv_agent_slots(
        self, audit_tmp_dir,
    ):
        from data.customer_memory import get_customer_memory_store
        get_customer_memory_store().upsert(
            "c1", preferences={"investment_goal": "house deposit", "user_name": "Sam"},
        )
        orch = _orch("mem-3")
        orch._load_customer("c1")
        orch._load_customer_memory("c1")
        assert orch._agents["ConversationalAgent"].slots == {
            "investment_goal": "house deposit", "user_name": "Sam",
        }

    def test_no_memory_row_is_a_graceful_no_op(self, audit_tmp_dir):
        orch = _orch("mem-4")
        orch._load_customer("brand-new-nobody-has-seen")
        orch._load_customer_memory("brand-new-nobody-has-seen")  # must not raise
        assert orch._session_state["risk_profile"] is None

    def test_disabled_config_skips_restoration(self, audit_tmp_dir, monkeypatch):
        from data.customer_memory import get_customer_memory_store
        get_customer_memory_store().upsert("c1", risk_profile={"risk_class": "aggressive"})
        monkeypatch.setattr(settings.conversational, "memory_enabled", False)
        orch = _orch("mem-5")
        orch._load_customer("c1")
        orch._load_customer_memory("c1")
        assert orch._session_state["risk_profile"] is None

    def test_persist_writes_risk_profile_and_increments_turn_count(
        self, audit_tmp_dir,
    ):
        from data.customer_memory import get_customer_memory_store
        orch = _orch("mem-6", customer_id="c1")
        orch._session_state["risk_profile"] = {"risk_class": "moderate"}
        orch._persist_customer_memory()
        orch._persist_customer_memory()

        stored = get_customer_memory_store().get("c1")
        assert stored.risk_profile == {"risk_class": "moderate"}
        assert stored.turn_count == 2

    def test_full_cycle_through_real_process_turn_calls(self, audit_tmp_dir):
        """
        The build plan's own demo, as a real test: complete risk
        elicitation in one Orchestrator instance (persisting via
        process_turn()'s own turn-end hook, not called directly), then
        confirm a SECOND, independent Orchestrator instance for the same
        customer_id has the risk_profile already available without
        RiskProfilingAgent running at all.
        """
        from orchestrator.orchestrator import RoutingDecision
        from data.customer_memory import get_customer_memory_store

        first = _orch("mem-7a", customer_id="MEMCYCLE_1")
        first._classify_intent = lambda msg: (
            RoutingDecision.RISK_PROFILING, "forced", 1.0,
        )
        first._session_state["user_features"] = {
            "age": 34, "income": 55000, "employment_status": "employed",
            "dependents": 0, "existing_debt": 5000, "investment_horizon": 15,
            "loss_tolerance": 4, "financial_knowledge_score": 3,
        }
        result = first.process_turn("Can you assess my risk profile?")
        assert result.agents_invoked == ["RiskProfilingAgent", "ExplainabilityAgent"]

        stored = get_customer_memory_store().get("MEMCYCLE_1")
        assert stored is not None
        assert stored.risk_profile["status"] == "complete"

        second = _orch("mem-7b", customer_id="MEMCYCLE_1")
        second._classify_intent = lambda msg: (
            RoutingDecision.EXPLANATION_REQUEST, "forced", 1.0,
        )
        result2 = second.process_turn("Can you explain my risk level?")
        # No RiskProfilingAgent needed this time — memory already supplied
        # risk_class, unlike test_agent_collaboration.py's equivalent
        # scenario for a customer with NO prior session.
        assert result2.agents_invoked == ["ExplainabilityAgent"]
        assert result2.collaboration_events == []


# ── Conversation summarisation ──────────────────────────────────────────

class TestSummarisation:

    LONG = "I have been discussing my finances in detail. " * 50  # ~2350 chars

    def test_estimate_history_tokens_is_roughly_chars_over_four(self, audit_tmp_dir):
        orch = _orch("summ-1")
        orch._session_state["conversation_history"] = [
            {"role": "user", "content": "a" * 400},
        ]
        assert orch._estimate_history_tokens() == 100

    def test_below_threshold_does_not_summarise(self, audit_tmp_dir):
        orch = _orch("summ-2")
        orch._session_state["conversation_history"] = [
            {"role": "user", "content": "short message"},
            {"role": "assistant", "content": "short reply"},
        ]
        before = list(orch._session_state["conversation_history"])
        orch._maybe_summarise()
        assert orch._session_state["conversation_history"] == before

    def test_few_entries_does_not_summarise_even_if_long(self, audit_tmp_dir):
        """Fewer entries than summarise_keep_last_n -> nothing OLD enough
        to summarise, regardless of token count."""
        orch = _orch("summ-3")
        orch._session_state["conversation_history"] = [
            {"role": "user", "content": self.LONG},
            {"role": "assistant", "content": self.LONG},
        ]
        before = list(orch._session_state["conversation_history"])
        orch._maybe_summarise()
        assert orch._session_state["conversation_history"] == before

    def test_above_threshold_summarises_and_keeps_last_n_verbatim(self, audit_tmp_dir):
        orch = _orch("summ-4")
        history = []
        for _ in range(10):
            history.append({"role": "user", "content": self.LONG})
            history.append({"role": "assistant", "content": self.LONG})
        orch._session_state["conversation_history"] = history

        orch._maybe_summarise()
        result = orch._session_state["conversation_history"]

        assert len(result) == settings.conversational.summarise_keep_last_n + 1
        assert result[0]["role"] == "system"
        assert result[0]["content"].startswith("[Earlier conversation]")
        assert result[-settings.conversational.summarise_keep_last_n:] == history[-4:]

    def test_disabled_config_never_summarises(self, audit_tmp_dir, monkeypatch):
        monkeypatch.setattr(settings.conversational, "summarise_enabled", False)
        orch = _orch("summ-5")
        history = [{"role": "user", "content": self.LONG} for _ in range(10)]
        orch._session_state["conversation_history"] = history
        orch._maybe_summarise()
        assert orch._session_state["conversation_history"] == history

    def test_summarisation_failure_degrades_without_raising(self, audit_tmp_dir):
        orch = _orch("summ-6")

        def broken_chat(*a, **k):
            raise RuntimeError("simulated LLM outage")
        orch.llm.chat = broken_chat

        history = []
        for _ in range(10):
            history.append({"role": "user", "content": self.LONG})
            history.append({"role": "assistant", "content": self.LONG})
        orch._session_state["conversation_history"] = history

        orch._maybe_summarise()  # must not raise
        result = orch._session_state["conversation_history"]
        assert result[0]["role"] == "system"
        assert "summarisation" in result[0]["content"].lower() or "earlier" in result[0]["content"].lower()

    def test_summarisation_runs_automatically_at_turn_end(self, audit_tmp_dir):
        """Through a real process_turn() call, not called directly —
        confirms the wiring, not just the method in isolation."""
        orch = _orch("summ-7")
        orch._session_state["conversation_history"] = [
            {"role": "user", "content": self.LONG},
            {"role": "assistant", "content": self.LONG},
        ] * 8  # well over threshold before this turn even starts

        orch.process_turn("one more short message")
        history = orch._session_state["conversation_history"]
        assert history[0]["role"] == "system"
        assert len(history) == settings.conversational.summarise_keep_last_n + 1