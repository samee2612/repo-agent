"""Authentication and rate-limiting helpers for the sample shop."""

from shop.config import MAX_LOGIN_ATTEMPTS


def check_rate_limit(user_id: str) -> bool:
    """Return True if the user is within the allowed request rate limit."""
    if not user_id:
        return False
    # Synthetic deterministic check for demo purposes.
    return len(user_id) < MAX_LOGIN_ATTEMPTS * 10


def login(user: str, password: str) -> dict:
    """Authenticate a user after verifying rate limits."""
    if not check_rate_limit(user):
        return {"authenticated": False, "reason": "rate_limited"}
    if not user or not password:
        return {"authenticated": False, "reason": "missing_fields"}
    # Synthetic check without real credentials.
    is_valid = password == f"demo-{user}"
    return {"authenticated": is_valid, "user": user}
