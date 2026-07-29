"""
Deterministic financial rule-based constraints.
Nguyen et al. [8]: rule constraints as secondary layer alongside ML (E4).
Raza et al. [1]: violations written to TRiSM audit log (O3).

Constraint categories:
  R001 — Return plausibility
  R002 — Risk-product compatibility (CBI suitability)
  R003 — Prohibited phrases (guarantee claims)
  R004 — Required disclaimers (CBI consumer protection)
"""

from __future__ import annotations
import re

from dataclasses import dataclass
from typing import Any


@dataclass
class ConstraintViolation:
    rule_id: str
    description: str
    severity: str
    field: str
    value: Any

class FinancialConstraints:

    MAX_ANNUAL_RETURN_PCT = 30.0

    RISK_PRODUCT_ALLOW: dict = {
        "conservative": {"savings_account", "government_bond", "money_market", "term_deposit"},
        "moderately_conservative": {"savings_account", "government_bond", "corporate_bond_ig", "mixed_fund_low"},
        "moderate": {"government_bond", "corporate_bond", "etf_broad", "mixed_fund", "reit"},
        "moderately_aggressive": {"etf_broad", "etf_sector", "equity_fund", "corporate_bond", "reit"},
        "aggressive": {"equity_fund", "etf_sector", "individual_equity", "venture"},
    }

    PROHIBITED_PHRASES = [
        "guaranteed return", "risk-free profit", "cannot lose money",
        "100% safe", "certain profit", "no risk",
    ]

    PROHIBITED_PATTERNS = {
        "guaranteed return": (
            r"\bguarantee(?:s|d|ing)?\b(?:\W+\w+){0,3}\W+\breturns?\b"
            # reversed word order, with lookbehinds so a NEGATED statement
            # ("returns are not guaranteed") is not itself flagged as a
            # guarantee claim — that sentence is the compliant one.
            r"|\breturns?\b(?:\W+\w+){0,3}\W+\b(?<!not )(?<!never )guarantee(?:s|d)\b"
        ),
        "risk-free profit": (
            r"\brisk[\s\-]*free\b(?:\W+\w+){0,3}\W+"
            r"\b(?:profit|return|gain|income)s?\b"
        ),
        "cannot lose money": (
            r"\b(?:cannot|can\'?t|will\s+not|won\'?t|never)\b"
            r"(?:\W+\w+){0,2}\W+\blose\b(?:\W+\w+){0,2}\W+"
            r"\b(?:money|capital|principal|anything)\b"
        ),
        "100% safe": r"\b100\s*(?:%|percent)(?:\W+\w+){0,2}\W+\b(?:safe|secure|protected)\b",
        "certain profit": r"\b(?:certain|assured|sure|definite)\b(?:\W+\w+){0,2}\W+\bprofits?\b",
        "no risk": r"\b(?:with|carries|involves|there\s+is|bears|has)\s+no\s+risk\b(?![\s\-]*free)",
    }

    REQUIRED_DISCLAIMERS = [
        "not regulated financial advice",
        "consult a qualified advisor",
        "past performance",
    ]

    DISCLAIMER_PATTERNS = {
        "not regulated financial advice":
            r"not\s+(a\s+|regulated\s+)?(form\s+of\s+)?regulated\s+financial\s+advice"
            r"|not\s+financial\s+advice",
        "consult a qualified advisor":
            r"qualified\s+(financial\s+)?(advisor|adviser)",
        "past performance":
            r"past\s+performance",
    }

    def check_disclaimers_present(self, text: str):
        """
        Regex-based, not substring — see DISCLAIMER_PATTERNS for why.
        Violations are still reported using the canonical label, so nothing
        downstream (audit log, results JSON) changes shape.
        """
        return [
            ConstraintViolation(rule_id="R004", description=f"Missing disclaimer: '{label}'",
                                severity="warn", field="response_text", value=label)
            for label in self.REQUIRED_DISCLAIMERS
            if not re.search(self.DISCLAIMER_PATTERNS[label], text, re.IGNORECASE)
        ]

    def check_return_plausibility(self, claimed_return: float):
        if claimed_return > self.MAX_ANNUAL_RETURN_PCT:
            return ConstraintViolation(
                rule_id="R001",
                description=f"Claimed return {claimed_return}% exceeds ceiling {self.MAX_ANNUAL_RETURN_PCT}%",
                severity="hard_block", field="expected_return_pct", value=claimed_return,
            )
        return None

    def check_risk_product_compatibility(self, risk_class: str, product_category: str):
        allowed = self.RISK_PRODUCT_ALLOW.get(risk_class, set())
        if product_category not in allowed:
            return ConstraintViolation(
                rule_id="R002",
                description=f"'{product_category}' not suitable for '{risk_class}' per CBI rules",
                severity="hard_block", field="product_category", value=product_category,
            )
        return None

    def check_prohibited_phrases(self, text: str):
        return [
            ConstraintViolation(rule_id="R003", description=f"Prohibited: '{label}'",
                                severity="hard_block", field="response_text", value=label)
            for label in self.PROHIBITED_PHRASES
            if re.search(self.PROHIBITED_PATTERNS[label], text, re.IGNORECASE)
         ]

    def check_disclaimers_present(self, text: str, advisory: bool = True):
        if not advisory:
            return []
        return [
             ConstraintViolation(rule_id="R004", description=f"Missing disclaimer: '{label}'",
                                 severity="warn", field="response_text", value=label)

        ]

    def validate_response(self, response_text: str, risk_class=None,
                          product_category=None, claimed_return=None,
                          advisory: bool = True):
        violations = []
        if claimed_return is not None:
            v = self.check_return_plausibility(claimed_return)
            if v: violations.append(v)
        if risk_class and product_category:
            v = self.check_risk_product_compatibility(risk_class, product_category)
            if v: violations.append(v)
        violations.extend(self.check_prohibited_phrases(response_text))
        violations.extend(self.check_disclaimers_present(response_text, advisory=advisory))
        has_hard_block = any(v.severity == "hard_block" for v in violations)
        return not has_hard_block, violations

# financial_constraints = FinancialConstraints()

# R14 - lazy singleton.
_financial_constraints: "FinancialConstraints | None" = None

def get_financial_constraints() -> "FinancialConstraints":
    """Construct on first use, then reuse. The accessor to prefer in new code."""
    global _financial_constraints
    if _financial_constraints is None:
        _financial_constraints = FinancialConstraints()
    return _financial_constraints


def set_financial_constraints(instance: "FinancialConstraints | None") -> None:
    """Inject a substitute (or None to reset). Intended for tests and ablations."""
    global _financial_constraints
    _financial_constraints = instance


def __getattr__(name: str):
    # Keeps the historic module-level name working: the singleton is built the
    # first time something reads it, not when this module is imported.
    if name == "financial_constraints":
        return get_financial_constraints()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")