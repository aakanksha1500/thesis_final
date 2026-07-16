"""All Environment-variable access is centralised here.
No agent or utility reads os.environ directly, they import from this file.
This helps us to keep test-time overrides clean
"""

from __future__ import annotations
import os   
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# LLM Configuration
# Used by: ConversationalAgent

@dataclass
class LLMConfig:
    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini")
    )
    specialist_model: str = field(
        default_factory=lambda: os.getenv("SPECIALIST_MODEL", "gpt-4o-mini")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    )
    temprature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 30

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
            ])
    
    #maximum conversation turns before forcing re-elicitation of stale slots
    slot_ttl_turns: int = 10

    # Minimum confidence for intent classification before asking for clarification
    intent_confidence_threshold: float = 0.65

    dataset_path: Path = ROOT_DIR / "data" / "raw" / "banking77"

@dataclass
class RiskConfig:
    """
    Context-Aware Hybrid scoring weights
    hybrid_score = ml_weight * ml_score + rule_weight * rule_score
    """
    ml_weight: float = 0.6
    rule_weight: float = 0.4
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

@dataclass
class ExplainabilityConfig:
    """
    Ablation switches — changing these booleans is the only code change
    needed to switch ablation conditions. Calibration note (X3) always on.
    """
    use_shap: bool = True
    use_rag_citation: bool = False  # off until Phase 8 RAG is built
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
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversational: ConversationalConfig = field(default_factory=ConversationalConfig)
    investment: InvestmentConfig = field(default_factory=InvestmentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

settings = Settings()