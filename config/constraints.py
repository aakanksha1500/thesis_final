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
        text_lower = text.lower()
        return [
            ConstraintViolation(rule_id="R003", description=f"Prohibited: '{p}'",
                                severity="hard_block", field="response_text", value=p)
            for p in self.PROHIBITED_PHRASES if p in text_lower
        ]

    def check_disclaimers_present(self, text: str):
        text_lower = text.lower()
        return [
            ConstraintViolation(rule_id="R004", description=f"Missing disclaimer: '{d}'",
                                severity="warn", field="response_text", value=d)
            for d in self.REQUIRED_DISCLAIMERS if d not in text_lower
        ]

    def validate_response(self, response_text: str, risk_class=None,
                          product_category=None, claimed_return=None):
        violations = []
        if claimed_return is not None:
            v = self.check_return_plausibility(claimed_return)
            if v: violations.append(v)
        if risk_class and product_category:
            v = self.check_risk_product_compatibility(risk_class, product_category)
            if v: violations.append(v)
        violations.extend(self.check_prohibited_phrases(response_text))
        violations.extend(self.check_disclaimers_present(response_text))
        has_hard_block = any(v.severity == "hard_block" for v in violations)
        return not has_hard_block, violations

financial_constraints = FinancialConstraints()
