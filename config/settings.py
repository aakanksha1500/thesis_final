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
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversational: ConversationalConfig = field(default_factory=ConversationalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

settings = Settings()