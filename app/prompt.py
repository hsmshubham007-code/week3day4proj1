"""
Prompt and deterministic customer-message classifier.

This classifier simulates an LLM prompt for the evaluation project.
It is deterministic so the evaluation suite can run reliably in
local development and GitHub Actions without API costs.
"""

import re


SYSTEM_PROMPT = """
You are a customer-support message classifier.

Classify each customer message into exactly one of these categories:

- Billing
- Technical
- Delivery
- Account

Return only the category name.
"""


def contains_keyword(text: str, keyword: str) -> bool:
    """
    Check whether a keyword appears as a complete word or phrase.

    Word boundaries prevent false matches such as:
        "app" matching "appears"
    """

    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None


def classify_customer_message(message: str) -> str:
    """
    Classify a customer-support message.

    The rules are deterministic so evaluation results remain
    reproducible in CI.
    """

    text = message.lower()

    # ---------------------------------------------------------
    # Billing
    # ---------------------------------------------------------
    billing_keywords = [
        "charged",
        "charge",
        "payment",
        "invoice",
        "refund",
        "fee",
        "billing",
        "billed",
        "bill",
        "subscription",
        "transaction",
        "receipt",
        "declined",
        "invoices",
    ]

    # ---------------------------------------------------------
    # Technical
    # ---------------------------------------------------------
    technical_keywords = [
        "crashes",
        "crash",
        "error",
        "app",
        "application",
        "website",
        "loading",
        "freezing",
        "broken",
        "system",
        "reset link",
        "does not work",
        "timing out",
        "timeout",
        "unresponsive",
        "blank screen",
        "stopped responding",
        "screen",
    ]

    # ---------------------------------------------------------
    # Delivery
    # ---------------------------------------------------------
    delivery_keywords = [
        "package",
        "shipment",
        "shipping",
        "tracking",
        "delivered",
        "delivery",
        "delayed",
        "courier",
        "transit",
        "tracking number",
    ]

    # ---------------------------------------------------------
    # Account
    # ---------------------------------------------------------
    account_keywords = [
        "account",
        "profile",
        "username",
        "account details",
        "email address",
        "account settings",
        "account preferences",
    ]

    # ---------------------------------------------------------
    # Classification priority
    # ---------------------------------------------------------

    # Billing has highest priority because a billing message
    # can also mention an order.
    if any(contains_keyword(text, keyword) for keyword in billing_keywords):
        return "Billing"

    # Technical is checked next.
    # Word-boundary matching means "app" matches "app"
    # but does NOT match the "app" inside "appears".
    if any(contains_keyword(text, keyword) for keyword in technical_keywords):
        return "Technical"

    # Delivery is checked after billing and technical.
    if any(contains_keyword(text, keyword) for keyword in delivery_keywords):
        return "Delivery"

    # Account-related requests.
    if any(contains_keyword(text, keyword) for keyword in account_keywords):
        return "Account"

    # ---------------------------------------------------------
    # Generic delivery questions
    # ---------------------------------------------------------

    if "where is my order" in text:
        return "Delivery"

    if "when will my order" in text:
        return "Delivery"

    if (
        "order" in text
        and (
            "arrive" in text
            or "delivery" in text
            or "shipping" in text
        )
    ):
        return "Delivery"

    # No matching category.
    return "Unknown"
