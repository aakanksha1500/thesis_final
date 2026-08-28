"""
Specifies the transaction-data contract — shape, validation, redaction,
and freshness rules — shared by both supported transports (push and pull).

WHY THIS FILE EXISTS
    "Per-query push vs pull-by-customer_id" was carried as an open decision
    across several sessions and flagged as a blocker for any bank-app
    integration. It stayed open partly because it was framed as a binary
    choice, and it is not one: push and pull differ in who holds the data,
    not in what the data is. The shape, the validation rules, the redaction
    rules and the freshness semantics are identical either way, and they are
    what actually blocks an integration.

    So this module specifies the CONTRACT — which is decidable now — and
    supports both TRANSPORTS behind one interface, so the deployment choice
    can be made per-integration by configuration rather than by a rewrite.

THE TWO TRANSPORTS, AND WHEN EACH IS RIGHT

    PUSH (TransactionPayload passed in with the query)
        The calling application already holds the customer's transactions
        and sends them with each advisory request.
        + No data at rest in this system: nothing to breach, nothing to
          retain, nothing to justify under GDPR storage limitation.
        + No coupling to the bank's core banking availability.
        - Payload size grows with history length; a 24-month window is a
          few hundred KB per request.
        - The caller becomes responsible for windowing and consistency, and
          two requests in the same session can disagree.
        Right when: the bank will not grant a data connection, or the
        deployment is a widget inside an app that already has the data.

    PULL (fetch by customer_id from a TransactionSource)
        This system holds a reference and fetches on demand.
        + Small requests; one consistent view per session.
        + Windowing policy lives in one place and is auditable.
        - Requires a live data connection, an availability dependency, and
          a stored mapping from session to customer_id.
        - Puts this system inside the bank's data perimeter, with everything
          that implies for the security review.
        Right when: this runs as a service the bank operates.

    RECOMMENDATION, STATED SO IT CAN BE ARGUED WITH
        Default to PULL for a bank-operated deployment and PUSH for an
        embedded one, and make it configuration
        (settings.transactions.transport). The reason is not technical
        elegance: PUSH's real cost is that the caller owns windowing, and a
        budget conclusion computed over a window this system did not choose
        is one it cannot defend when asked why the numbers moved.

WHAT IS NON-NEGOTIABLE IN BOTH
    - Every payload is validated before use. An advisory conclusion drawn
      from a malformed or partial transaction set is worse than no
      conclusion, because it looks the same as a good one.
    - `window` is explicit and travels with the data. A budget figure
      without the period it was computed over is not interpretable.
    - `completeness` is explicit. A partial month is the single most common
      cause of a wrong savings-rate figure, and it must be detectable rather
      than inferred from row counts.
    - Raw descriptions are redacted before they reach any LLM. A transaction
      narrative is free text containing counterparty names, and there is no
      advisory reason to send it to a model.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Protocol

from utils.logger import get_logger

logger = get_logger(__name__)

CONTRACT_VERSION = "1.0"

# Categories the BudgetAgent's benchmark comparison understands. A payload
# may use others; they are preserved but reported under `unmapped_categories`
# rather than silently folded into "other", because a category this system
# cannot benchmark is a gap in the analysis and should be visible as one.
KNOWN_CATEGORIES = frozenset({
    "rent", "mortgage", "groceries", "utilities", "transport", "fuel",
    "insurance", "healthcare", "childcare", "education", "entertainment",
    "dining", "subscriptions", "clothing", "debt_repayment", "savings",
    "transfer", "income", "other",
})


class TransactionContractError(ValueError):
    """Raised when a payload cannot be used. Never downgraded to a warning."""


@dataclass(frozen=True)
class Transaction:
    """
    One transaction, in the only shape this system accepts.

    `description_redacted` rather than `description`: the field name is the
    control. A field called `description` gets logged, prompted with, and
    embedded — this one announces at every call site that the raw narrative
    is not here and is not supposed to be.
    """
    transaction_id: str
    booked_at: date
    amount: float               # negative = money out, positive = money in
    currency: str
    category: str
    description_redacted: str = ""
    merchant_category_code: str | None = None
    is_recurring: bool | None = None

    @property
    def is_debit(self) -> bool:
        return self.amount < 0


@dataclass(frozen=True)
class TransactionWindow:
    """
    The period the data covers, and whether it covers it fully.

    `complete` is the field that matters. A window running to the 12th of the
    month contains 12 days of spending; a savings rate computed from it
    without adjustment is wrong by roughly 60%, and nothing about the row
    count reveals that. The BudgetAgent must refuse or annualise explicitly,
    and it can only do either if this field exists.
    """
    start: date
    end: date
    complete: bool
    source: str = "unspecified"

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def whole_months(self) -> float:
        return round(self.days / 30.44, 2)


@dataclass(frozen=True)
class TransactionPayload:
    """
    The complete transaction-data contract. Identical for push and pull.

    `checksum` covers the transaction set. Two requests in one session that
    claim the same window but carry different data is a real failure mode of
    the push transport, and it is silent unless something checks.
    """
    customer_id: str
    window: TransactionWindow
    transactions: tuple[Transaction, ...]
    currency: str = "EUR"
    contract_version: str = CONTRACT_VERSION
    provenance: str = "unspecified"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        h = hashlib.sha256()
        for t in sorted(self.transactions, key=lambda x: (x.booked_at, x.transaction_id)):
            h.update(f"{t.transaction_id}|{t.booked_at}|{t.amount:.2f}|{t.category}".encode())
        return h.hexdigest()[:16]

    @property
    def unmapped_categories(self) -> tuple[str, ...]:
        return tuple(sorted(
            {t.category for t in self.transactions} - KNOWN_CATEGORIES
        ))

    def monthly_expenses(self) -> dict[str, float]:
        """
        Per-category monthly outgoings, normalised by the window length.

        Normalisation is why `window.complete` has to exist: over an
        incomplete window this divides by the ACTUAL elapsed period rather
        than assuming a month, so a part-month cannot silently deflate every
        category at once.
        """
        months = max(self.window.whole_months, 0.1)
        totals: dict[str, float] = {}
        for t in self.transactions:
            if not t.is_debit:
                continue
            totals[t.category] = totals.get(t.category, 0.0) + abs(t.amount)
        return {k: round(v / months, 2) for k, v in sorted(totals.items())}

    def monthly_income(self) -> float:
        months = max(self.window.whole_months, 0.1)
        total = sum(t.amount for t in self.transactions if t.amount > 0)
        return round(total / months, 2)


# -- validation ---------------------------------------------------------
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def validate(payload: TransactionPayload, *, min_days: int = 28) -> list[str]:
    """
    Return a list of problems. Empty means usable.

    Kept separate from `enforce` so a caller can decide to proceed with
    warnings on a non-advisory path (e.g. showing a chart) while the advisory
    path refuses. The distinction the previous code lacked was any check at
    all.
    """
    problems: list[str] = []

    if not _ID_RE.match(payload.customer_id or ""):
        problems.append("customer_id is empty or contains unexpected characters")

    if payload.contract_version != CONTRACT_VERSION:
        problems.append(
            f"contract_version {payload.contract_version!r} != "
            f"{CONTRACT_VERSION!r}; field semantics may differ"
        )

    w = payload.window
    if w.end < w.start:
        problems.append(f"window ends ({w.end}) before it starts ({w.start})")
    elif w.days < min_days:
        problems.append(
            f"window is {w.days} days; below the {min_days}-day floor a "
            f"monthly figure is an extrapolation, not an observation"
        )

    if w.end > date.today():
        problems.append(f"window ends in the future ({w.end})")

    if not payload.transactions:
        problems.append("no transactions in payload")

    out_of_window = [
        t.transaction_id for t in payload.transactions
        if not (w.start <= t.booked_at <= w.end)
    ]
    if out_of_window:
        problems.append(
            f"{len(out_of_window)} transaction(s) fall outside the declared "
            f"window, e.g. {out_of_window[:3]} — the window is what every "
            f"per-month figure is divided by, so this silently skews all of them"
        )

    ids = [t.transaction_id for t in payload.transactions]
    if len(set(ids)) != len(ids):
        problems.append(
            "duplicate transaction_ids — double-counted spending inflates "
            "every expense category and deflates the savings rate"
        )

    currencies = {t.currency for t in payload.transactions}
    if len(currencies) > 1:
        problems.append(
            f"mixed currencies {sorted(currencies)} with no conversion rates; "
            f"summing them produces a meaningless number"
        )

    if any(t.description_redacted and _looks_unredacted(t.description_redacted)
           for t in payload.transactions):
        problems.append(
            "description_redacted contains what looks like an unredacted "
            "counterparty narrative — this field must not reach an LLM prompt"
        )

    return problems


def _looks_unredacted(text: str) -> bool:
    """Heuristic: IBANs, long digit runs, or an email address."""
    return bool(
        re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,}\b", text)
        or re.search(r"\b\d{9,}\b", text)
        or re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    )


def enforce(payload: TransactionPayload, *, min_days: int = 28) -> TransactionPayload:
    """Validate or raise. Use on any path that leads to advice."""
    problems = validate(payload, min_days=min_days)
    if problems:
        raise TransactionContractError(
            f"transaction payload for customer {payload.customer_id!r} is not "
            f"usable for advice: " + "; ".join(problems)
        )
    return payload


# -- transports ---------------------------------------------------------
class TransactionSource(Protocol):
    """
    The pull interface. One method, so a bank integration is one adapter.

    Implementations must either return a payload covering the requested
    window or raise. Returning a shorter window than asked for, silently, is
    the failure this signature is shaped to prevent — the caller cannot tell
    from the return value alone, which is why `window` is inside the payload.
    """

    def fetch(self, customer_id: str, window: TransactionWindow) -> TransactionPayload:
        ...


def default_window(months: int = 3, *, today: date | None = None) -> TransactionWindow:
    """
    A whole-months window ending at the last day of the previous month.

    Ending at the previous month-end rather than today is deliberate: the
    current month is always partial, and including it is the most common
    source of a wrong savings rate. Callers that genuinely want
    month-to-date must construct the window themselves and set
    complete=False, which makes the choice explicit at the call site.
    """
    today = today or date.today()
    first_of_this_month = today.replace(day=1)
    end = first_of_this_month - timedelta(days=1)
    start = end.replace(day=1)
    for _ in range(months - 1):
        start = (start - timedelta(days=1)).replace(day=1)
    return TransactionWindow(start=start, end=end, complete=True, source="default_window")


def from_dicts(
    customer_id: str,
    rows: Iterable[dict[str, Any]],
    window: TransactionWindow,
    *,
    provenance: str = "push",
) -> TransactionPayload:
    """
    Build a payload from the wire format, failing loudly on bad rows.

    Deliberately strict about dates and amounts. A string amount that quietly
    becomes 0.0, or an unparseable date that becomes today, are both changes
    to a number a customer will be advised on.
    """
    txs: list[Transaction] = []
    for i, row in enumerate(rows):
        try:
            booked = row["booked_at"]
            if isinstance(booked, str):
                booked = datetime.fromisoformat(booked).date()
            elif isinstance(booked, datetime):
                booked = booked.date()
            if not isinstance(booked, date):
                raise TypeError(f"booked_at is {type(booked).__name__}")

            txs.append(Transaction(
                transaction_id=str(row["transaction_id"]),
                booked_at=booked,
                amount=float(row["amount"]),
                currency=str(row.get("currency", "EUR")),
                category=str(row.get("category", "other")).lower(),
                description_redacted=str(row.get("description_redacted", "")),
                merchant_category_code=row.get("merchant_category_code"),
                is_recurring=row.get("is_recurring"),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise TransactionContractError(
                f"row {i} does not satisfy the transaction contract: {exc}. "
                f"Coercing it to a default would change a figure the customer "
                f"is advised on."
            ) from exc

    return TransactionPayload(
        customer_id=customer_id,
        window=window,
        transactions=tuple(txs),
        provenance=provenance,
    )
