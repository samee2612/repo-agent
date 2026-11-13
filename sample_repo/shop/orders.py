"""Order placement logic for the sample shop."""

from shop.auth import check_rate_limit
from shop.config import DEFAULT_CURRENCY


def place_order(user_id: str, items: list[str]) -> dict:
    """Place an order for a user after checking the rate limit."""
    if not check_rate_limit(user_id):
        return {"status": "rejected", "reason": "rate_limited"}
    if not items:
        return {"status": "rejected", "reason": "empty_cart"}
    return {
        "status": "accepted",
        "user_id": user_id,
        "item_count": len(items),
        "currency": DEFAULT_CURRENCY,
    }
