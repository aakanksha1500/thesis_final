"""All Environment-variable access is centralised here.
No agent or utility reads os.environ directly, they import from this file.
This helps us to keep test-time overrides clean
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
# LLM Configuration
# Used by: ConversationalAgent

@dataclass
class LLMConfig:
    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini")
    )
    # Specialist tier (RiskProfilingAgent, InvestmentAgent, BudgetAgent,
    # ExplainabilityAgent — everything narrating figures Python already
    # computed, never doing open-ended reasoning the orchestrator/judge
    # tiers need a stronger cloud model for) runs on a DIFFERENT provider
    # by design, not just a different model name — see
    # Orchestrator._derive_client() and utils.llm_client.LLMClient's
    # `provider` param. Deliberately does NOT read LLM_PROVIDER: the whole
    # point is that specialist stays on the local provider regardless of
    # what orchestrator_model/judge_model's provider is set to, so this
    # can't drift back onto the cloud provider by a global env change made
    # for an unrelated reason. Override with SPECIALIST_LLM_PROVIDER if
    # you genuinely want to change it (e.g. back to a cloud provider for
    # a specific evidence run) — it's still centralised here, just not
    # coupled to LLM_PROVIDER.
    specialist_provider: str = field(
        default_factory=lambda: os.getenv("SPECIALIST_LLM_PROVIDER", "ollama")
    )
    # Ollama model tag, e.g. "llama3.1:8b" — NOT a Groq/OpenAI model id.
    # Must match whatever you've actually pulled (`ollama pull <model>`);
    # if SPECIALIST_MODEL in .env still holds a Groq-style name from
    # before this split, update it, or ollama will 404 on load and this
    # tier will fall back to mock mode like any other unreachable provider.
    specialist_model: str = field(
        default_factory=lambda: os.getenv("SPECIALIST_MODEL", "llama3.1:8b")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    )
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 30
    max_retries: int = 2

@dataclass
class ConversationalConfig:
    num_intent_classes: int = 77

    escalation_intents: list = field(
        default_factory=lambda: [
            "investment",
            "risk_profiling",
            "budget",
            "financial_advice",
            "savings",
            "card",
            "loan",
        ]
    )

    tracked_slots: list = field(
        default_factory=lambda: [
            "user_name",
            "age",
            "income",
            "investment_goal",
            "risk_tolerance",
            "time_horizon",
            "housing_cost",
            "rough_monthly_leftover",
            "food_spend",
            "transport_spend",
            "utilities_spend",
            "healthcare_spend",
            "discretionary_spend",
            "debt_repayments",
            "large_recurring_items",
            "employment_status",
            "annual_income",
            "dependents",
            "existing_debt",
            "investment_horizon",
            "loss_tolerance",
            "financial_knowledge_score",
            ])

    #maximum conversation turns before forcing re-elicitation of stale slots
    slot_ttl_turns: int = 10

    # Minimum confidence for intent classification before asking for clarification
    intent_confidence_threshold: float = 0.65

    dataset_path: Path = ROOT_DIR / "data" / "raw" / "banking77"
    memory_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() != "false"
    )
    memory_db_path: Path = ROOT_DIR / "logs" / "customer_memory.db"

    summarise_enabled: bool = field(
        default_factory=lambda: os.getenv("SUMMARISE_ENABLED", "true").lower() != "false"
    )
    summarise_above_tokens: int = 3000
    summarise_keep_last_n: int = 4


@dataclass
class RiskConfig:
    """
    Context-Aware Hybrid scoring weights
    hybrid_score = ml_weight * ml_score + rule_weight * rule_score
    """
    ml_weight: float = 0.6
    rule_weight: float = 0.4
    capacity_weight: float = 0.5
    use_trained_model: bool = field(
        default_factory=lambda: (
            os.getenv("USE_TRAINED_RISK_MODEL", "true").lower() == "true"
        )
    )
    risk_classes: list = field(default_factory=lambda: [
        "conservative", "moderately_conservative", "moderate",
        "moderately_aggressive", "aggressive",
    ])
    required_features: list = field(default_factory=lambda: [
        "age", "income", "employment_status", "dependents",
        "existing_debt", "investment_horizon",
        "loss_tolerance", "financial_knowledge_score",
    ])
    min_confidence: float = 0.6
    model_path: Path = ROOT_DIR / "data" / "processed" / "risk_model.pkl"
    use_calibrated_confidence: bool = field(
        default_factory=lambda: (
            os.getenv("USE_CALIBRATED_CONFIDENCE", "false").lower() == "true"
        )
    )
    confidence_calibrator_path: Path = (
        ROOT_DIR / "data" / "processed" / "risk_confidence_calibrator.pkl"
    )
    use_quantile_tier_boundaries: bool = field(
        default_factory=lambda: (
            os.getenv("USE_QUANTILE_TIER_BOUNDARIES", "false").lower() == "true"
        )
    )
    tier_boundaries_path: Path = (
        ROOT_DIR / "data" / "processed" / "risk_tier_boundaries.json"
    )
    german_credit_path: Path = ROOT_DIR / "data" / "raw" / "german_credit.data"
    bank_marketing_path: Path = ROOT_DIR / "data" / "raw" / "bank_marketing"

@dataclass
class InvestmentConfig:
    """
    Hybrid product recommendation configuration. (rule filter + scoring + LLM synthesis).

    Layer 1 (_filter_by_risk_class) reuses config.constraints.FinancialConstraints
    .RISK_PRODUCT_ALLOW as the single source of truth for CBI suitability rules,
    so the same rule set backs both response validation and product
    filtering.

    Layer 2 (_rank_products) scoring weights. Must sum to 1.0.
        return_weight       -   favours higher expected_return_pct
        cost_weight         -   favours lower expense_ratio_pct
        horizon_fit_weight  -   favours products whose typical holding period
                                matches the user's stated investment_horizon
    """
    return_weight: float = 0.4
    cost_weight: float = 0.3
    horizon_fit_weight: float = 0.3
    top_k: int = 3
    min_products_after_filter: int = 1
    max_claimed_return_pct: float = 30.0

@dataclass
class BudgetConfig:
    """
    Ireland HouseholdBudget Survey 2022-2023 spending bemchmarks.
    Values represent fraction of gross monthly income.
    Source: CSO Ireland HBS 2022-23.

    Benchmarks used by BudgetAgent.compare_to_benchmarks() to classify
    eaxh spending category as above / below / inline vs national average.
    inline_tolerance: within ±10% of benchmark = inline (avoids false precision).
    min_healthy_savings_rate_pct: below this the agent flags a concern (CBI guidance).
    """
    ireland_hbs_benchmarks: dict = field(default_factory=lambda: {
        "housing":       0.28,
        "food":          0.14,
        "transport":     0.13,
        "utilities":     0.07,
        "healthcare":    0.05,
        "entertainment": 0.06,
        "savings":       0.10,
        "other":         0.17,
    })
    inline_tolerance: float = 0.10
    min_healthy_savings_rate_pct: float = 10.0
    personal_finance_path: str = "data/raw/personal_finance"
    ireland_hbs_path: str = "data/raw/ireland_hbs"
    default_aggregation_window_months: int = 12
    min_aggregation_window_months: int = 12  # floor enforced in BudgetAgent.run() —
    # a shorter window is still available directly via
    # _aggregate_transactions() for comparison/evaluation purposes (see
    # scripts/generate_transactions.py's 1/3/12-month finding), but the
    # production run() path never silently gives budget advice off a
    # window that demonstrably misreads annual-lump costs.
    transactions_path: str = "data/processed/transactions.json"
    periodicity_priors_path: Path = ROOT_DIR / "data" / "periodicity_priors.json"
    auto_load_transactions: bool = True

@dataclass
class ExplainabilityConfig:
    """
    Ablation switches — changing these booleans is the only code change
    needed to switch ablation conditions. Calibration note (X3) always on.
    """
    use_shap: bool = True
    use_rag_citation: bool = True
    use_counterfactual: bool = True
    use_calibration_note: bool = True  # X3 — never ablated
    prompt_version: str = "v1.0"
    low_confidence_threshold: float = 0.6

# Orchestrator configuration
# Addresses RQ4: multi-agent vs monolithic coherence.
# O1 — HALO hierarchical orchestration (Hou et al.)
# O3 — TRiSM audit log (Raza et al. [1])
# O4 — Self-healing via FailureHandler (Wang et al. AgentFixer [12])
@dataclass
class OrchestratorConfig:
    """
    Routing and execution parameters for the HALO Orchestrator.

    routing_confidence_threshold:
      Below this, the Orchestrator treats the intent as ambiguous and
      routes to ConversationalAgent for clarification rather than a
      specialist. Prevents premature specialist invocation on uncertain
      inputs (reduces unnecessary agent calls → improves TUE metric).

    max_agent_retries:
      O4 (AgentFixer — Wang et al. [12]): number of retry attempts before
      invoking the FailureHandler. 1 retry catches transient API failures
      without introducing significant latency (LR §7: 7s ceiling).

    synthesis_max_words:
      Hard ceiling on synthesis response length. Enforces the conversational
      UX constraint from LR §7 — retail investors should not receive walls
      of text from an AI advisor (Artusi et al. [10]).

    audit_log_dir:
      TRiSM (O3 — Raza et al. [1]) audit log location. JSONL format,
      one record per event, per-session file.
    """
    routing_confidence_threshold: float = 0.65
    max_agent_retries: int = 1
    synthesis_max_words: int = 150
    audit_log_dir: Path = ROOT_DIR / "logs" / "audit"
    enable_conflict_resolution: bool = True
    enable_failure_recovery: bool = True

@dataclass
class PlannerConfig:
    """
    Layer 1 planning. The planner PROPOSES; PlanValidator DECIDES.

    enabled:
      Master switch. True by default — the planner is the primary Layer 1
      path and the static table in agents/payloads.STATIC_SEQUENCES is its
      fallback. Set PLANNER_ENABLED=false to reproduce pre-Day-4 behaviour
      exactly, which is what the static arm of the planner-vs-static
      comparison uses.

    max_plan_steps:
      Hard ceiling on plan length. There are five agents; a plan longer than
      six steps is either a repetition the duplicate check missed or a model
      that has stopped following the schema. Rejecting on length is cheaper
      than discovering it as six LLM calls.

    temperature:
      0.0, deliberately. A plan is an execution decision in a regulated
      advisory system; the same message on the same context should route the
      same way twice. Sampling variety here is not creativity, it is a
      reproducibility defect that would make the agreement-rate metric
      meaningless.

    require_explainability_last:
      X1 — ExplainabilityAgent explains what earlier agents produced, so a
      plan that runs it first has it explaining an empty context. Enforced
      by position rather than by prompt, because a prompt instruction is a
      request and this is a constraint.

    NOTE — a plan that includes BudgetAgent/RiskProfilingAgent/
    InvestmentAgent without ExplainabilityAgent is deliberately still
    valid (G6's whole point: the planner may choose a shorter plan than
    the static table). This is NOT silent, though — see Plan.
    explainability_skipped in orchestrator/planner.py and the
    "SKIP-EXPLAIN" trace/audit event it drives. Considered making this a
    hard validator rejection instead; rejected that approach because it
    directly reverses a documented, tested G6 capability
    (test_planner_may_choose_a_shorter_plan_than_the_table) rather than
    just making its cost visible.

    allow_empty_plan:
      False. An empty plan is the model declining to route; the static
      fallback is a better answer than a turn with no agents in it.

    fallback_intent:
      Used when the classified intent has no entry in STATIC_SEQUENCES, so
      that a rejection always has somewhere to land.
    """
    enabled: bool = field(
        default_factory=lambda: os.getenv("PLANNER_ENABLED", "true").lower() != "false"
    )
    max_plan_steps: int = 6
    temperature: float = 0.0
    max_tokens: int = 300
    require_explainability_last: bool = True
    allow_empty_plan: bool = False
    fallback_intent: str = "conversational_only"

@dataclass
class CollaborationConfig:
    """
    Layer 2 mid-plan collaboration (Day 7 / G6 §6.6).

    An agent that cannot proceed names what it needs
    (payload={"status": "needs_input", "needs": [...]}) instead of just
    refusing; Orchestrator._satisfy_needs() finds a capability that
    produces it, runs that, and retries the original agent — bounded by
    the three settings below, all necessary together.

    enabled:
      Master switch. True by default. Set COLLABORATION_ENABLED=false to
      have a needs_input result behave exactly like any other non-success
      status — no retry attempted — for a clean before/after comparison.

    max_depth:
      How many producer-then-retry hops one original request may trigger.
      2 is enough for the one real chain in this system (Explainability
      needs risk_class -> RiskProfiling produces it directly, one hop) with
      headroom for a second hop without allowing an unbounded chain. This
      is a ceiling on CHAIN LENGTH, not on how many needs one request may
      name — every need in one request is attempted at the same depth.

    max_producer_attempts_per_need:
      How many DIFFERENT candidate producers to try for a single need
      before giving up on it. 1 in this system on purpose: CAPABILITIES
      has exactly one producer per key today (see
      validate_capability_graph()'s uniqueness note), so trying more would
      never find a second option — this exists so adding an alternate
      producer later doesn't silently change resolution semantics.
    """
    enabled: bool = field(
        default_factory=lambda: os.getenv("COLLABORATION_ENABLED", "true").lower() != "false"
    )
    max_depth: int = 2
    max_producer_attempts_per_need: int = 1

@dataclass
class ApprovalConfig:
    """
    Day 8 — human approval gate. State machine on the turn:
    PROCESSING -> AWAITING_APPROVAL -> APPROVED/REJECTED -> DELIVERED.

    Trigger conditions are config, not hard-coded, per the build plan's own
    instruction — each below can be switched off independently for an
    ablation, without touching Orchestrator._check_approval_gate()'s code.

    enabled:
      Master switch. True by default. Set APPROVAL_GATE_ENABLED=false to
      compare turn outcomes with and without the gate.

    gate_on_hard_block / gate_on_low_confidence_aggressive /
    gate_on_hallucination_flagged:
      Each maps directly to a signal this system already computes:
      constraint_violations' severity="hard_block" (config/constraints.py),
      ConflictResolver's LOW_CONFIDENCE_AGGRESSIVE conflict — checked via
      the conflict record, not by re-reading risk_class directly, since
      ConflictResolver already downgrades risk_class to "moderate" for
      routing by the time the gate runs; the conflict record is what
      preserves that this happened at all — and InvestmentAgent's own
      hallucination_flagged boolean.

    max_expected_return_pct:
      A shortlisted product promising more than this is gated regardless
      of which risk tier it came from — an unusually high expected return
      is exactly the kind of claim a human should see before it reaches a
      customer, independent of whether anything else about the turn looks
      fine.

    db_path:
      Single shared SQLite file across ALL sessions, deliberately unlike
      AuditLog's per-session JSONL files — GET /approvals is a reviewer's
      queue across every customer, not one session's history.

    withheld_message:
      What the customer sees instead of the real (gated) response. Says
      nothing about WHY it was gated — that reasoning is for the reviewer
      (PendingApproval.reasons), not the customer, and is never customer-
      facing.
    """
    enabled: bool = field(
        default_factory=lambda: os.getenv("APPROVAL_GATE_ENABLED", "true").lower() != "false"
    )
    gate_on_hard_block: bool = True
    gate_on_low_confidence_aggressive: bool = True
    gate_on_hallucination_flagged: bool = True
    max_expected_return_pct: float = 8.0
    db_path: Path = ROOT_DIR / "logs" / "approvals.db"
    withheld_message: str = (
        "Thanks for asking — this one needs a quick review before I can "
        "share it with you. I'll have it shortly; feel free to ask me "
        "something else in the meantime."
    )



@dataclass
class RAGConfig:
    """
    FAISS vector store + sentence-transformer embedding configuration.

    embedding_model:
        sentence-transformers model id. Falss back to a deterministic hashing
        embedder (rag/embedder.py) if sentence-transformers is not installed -
        mirrors LLMClient's mock-mode pattern (phase 1) so test never require
        the heavy dependency to be present.

    top_k_citations:
        Number of retrieved chunks attached as citations per Layer B call.
        Kept small - in-pipeline explanations must stay concise, not overwhelm
        the user with a source dump).

    min_relevance_score:
        Cosine-similarity floor below which a retrieved chunk is dropped
        rather than cited - an irrelevant "citation" would undermine the
        trust-calibration goal rather than support it.

    index_dir:
        Where the built FAISS index + document store are persisted.
        Rebuilt via 'python scripts/build_knowledge_base.py'.

    document_sets:
        The four corpora backing Layer B, per PDD dataset inventory.
        D1/D2 are also the evaluation corpus for RQ5 (evaluation/metrics.py)
    """
    embedding_model: str =field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = 384
    top_k_citations: int = 3
    min_relevance_score: float = 0.15
    index_dir: Path = ROOT_DIR / "data" / "embeddings" / "rag_index"
    chunk_size_chars: int = 500
    chunk_overlap_chars: int = 50
    log_citations_to_audit: bool = True
    document_sets: list = field(default_factory=lambda: [
        "regulatory",
        "cbi_open_data",
        "eu_digital_finance",
        "finqa_original",
        "finqa_verified",
    ])

@dataclass
class HallucinationConfig:
    """
    vectara/hallucination_evaluation_model (HHEM) configuration.

    hhem_threshold:
        claims scoring below this on the HHEM consistency scale (0=hallucinated,
        1=fully grounded) are flagged. 0.85 follows Vectara's published
        recommended operating threshold for factual/financial domains, where
        false negatives (missed hallucinations) are costlier than false positives (over-flagging).

    model_id:
        HugingFace model id. Falls back to a deterministic lexical-overlap
        heuristic (rag/hallucination_detector.py) if transformers/torch are not
        installed, so unit tests run without the ~2.4GB model download -
        same rationale as the RAG embedder fallback above.

    run_inline:
        When True, InvestmentAgent runs the detector n its wn synthesis
        immediately after generation, before ExplainabilityAgent wraps it

    max_claims_per_response:
        Celling on how many sentence-level claims are scored per response -
        keeps evaluation latemcy bounded on log sysntheses.
    """
    hhem_threshold: float = 0.85
    model_id: str = field(
        default_factory=lambda: os.getenv(
            "HHEM_MODEL_ID", "vectara/hallucination_evaluation_model"
        )
    )
    run_inline: bool = True
    max_claims_per_response: int = 10
    log_flagged_to_audit: bool = True

@dataclass
class MarketDataConfig:
    """
    yfinance-backed live pricing, with a snapshot-for-reproducibility mode.

    enabled:
      Master switch. False by default — IRISH_PRODUCT_CATALOGUE's synthetic
      expected_return_pct values are used unchanged, exactly as in Phases
      1-8. Set MARKET_DATA_ENABLED=true (or pass enabled=True) to turn on
      live enrichment.

    period_days:
      Trailing window used to compute the observed annualised return from
      price history (e.g. 365 = trailing 1-year return).

    cache_ttl_seconds:
      In-memory cache lifetime per ticker within a process. Keeps a single
      evaluation run internally consistent (doesn't re-fetch and drift
      mid-run) without needing the snapshot file.

    snapshot_path:
      If set and the file exists, prices are read from this frozen JSON
      snapshot instead of fetching live — the reproducible-evaluation path.
      Write a fresh snapshot with
      `python scripts/build_knowledge_base.py --refresh-market-snapshot`
      (or by enabling `enabled` once and letting MarketDataClient persist
      what it fetches).
    """
    enabled: bool = field(
        default_factory=lambda: os.getenv("MARKET_DATA_ENABLED", "false").lower() == "true"
    )
    period_days: int = 365
    cache_ttl_seconds: int = 900
    snapshot_path: Path = ROOT_DIR / "data" / "market_data" / "snapshot.json"
    request_timeout_seconds: int = 5

@dataclass
class ProductDataConfig:
    """
    Real reference data for catalogue products, beyond MarketDataConfig's
    return enrichment. See utils/product_data_client.py.

    enabled:
      Master switch for LIVE fetching (ECB Statistical Data Warehouse).
      False by default. The snapshot tier is read regardless — a frozen
      snapshot is the reproducible-evaluation path and must not depend on
      network availability.

    use_real_product_data:
      Separate, and deliberately so. This is the EVALUATION gate: it decides
      whether the InvestmentAgent's catalogue is enriched at all. Turning it
      on changes the world RQ2 measures, so results generated with it on are
      written to a different filename (see evaluation/results_io.py callers)
      rather than overwriting the synthetic-catalogue baseline. Comparing the
      two as though they measured the same thing would be a category error.

    max_age_days:
      A deposit rate from eighteen months ago is not real data, it is a stale
      number with a provenance stamp. Snapshot entries older than this are
      REFUSED rather than warned about, so the product degrades to its
      honestly-labelled synthetic figure.

    seed_path / snapshot_path:
      Base catalogue structure, and the frozen real figures layered over it.
    """
    enabled: bool = field(
        default_factory=lambda: os.getenv("PRODUCT_DATA_ENABLED", "false").lower() == "true"
    )
    use_real_product_data: bool = field(
        default_factory=lambda: os.getenv("USE_REAL_PRODUCT_DATA", "false").lower() == "true"
    )
    max_age_days: int = 400
    timeout_seconds: float = 10.0
    seed_path: Path = ROOT_DIR / "data" / "raw" / "product_catalogue" / "catalogue_seed.json"
    snapshot_path: Path = ROOT_DIR / "data" / "raw" / "product_catalogue" / "product_snapshot.json"

@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversational: ConversationalConfig = field(default_factory=ConversationalConfig)
    investment: InvestmentConfig = field(default_factory=InvestmentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    collaboration: CollaborationConfig = field(default_factory=CollaborationConfig)
    approval: ApprovalConfig = field(default_factory=ApprovalConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    product_data: ProductDataConfig = field(default_factory=ProductDataConfig)
    debug: bool = field(
         default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
     )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

settings = Settings()