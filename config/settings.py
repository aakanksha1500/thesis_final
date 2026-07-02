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
            "financial_advisor",
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
    class Settings:
        llm: LLMConfig = field(default_factory=LLMConfig)
        conversational: ConversationalConfig = field(default_factory=ConversationalConfig)
        debug: bool = field(
            default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
        )
        environment: str = field(
            default_factory=lambda: os.getenv("ENVIRONMENT", "development")
        )

    settings = Settings()