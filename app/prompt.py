SYSTEM_PROMPT = """
You are a customer support classifier.

Classify each customer message into exactly ONE of these categories:

- Billing
- Technical
- Delivery
- Account

Rules:

1. Billing:
   Questions about charges, payments, invoices, refunds, or fees.

2. Technical:
   Problems with applications, websites, errors, crashes, login failures,
   or system functionality.

3. Delivery:
   Questions about shipments, packages, tracking, delays, or delivery dates.

4. Account:
   Questions about account information, profile details, usernames,
   or changing account settings.

Return ONLY the category name.
Do not provide explanations.
"""


def classify_customer_message(message: str) -> str:
    """
    Deterministic baseline classifier.

    This simulates the behavior of an LLM prompt so that
    our CI evaluation remains reliable and free of API costs.
    """

    text = message.lower()


    if "refund" in text or "charged" in text or "package" in text:
      return "Unknown"
    

    billing_keywords = [
        "charged",
        "charge",
        "payment",
        "invoice",
        "fee",
        "billing",
    ]   

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
    ]

    delivery_keywords = [
        "package",
        "shipment",
        "shipping",
        "tracking",
        "delivered",
        "delivery",
        "order is delayed",
        "arrive",
    ]

    account_keywords = [
        "account",
        "profile",
        "username",
        "account details",
        "email address",
    ]

    if any(keyword in text for keyword in billing_keywords):
        return "Billing"

    if any(keyword in text for keyword in technical_keywords):
        return "Technical"

    if any(keyword in text for keyword in delivery_keywords):
        return "Delivery"

    if any(keyword in text for keyword in account_keywords):
        return "Account"

    return "Unknown"