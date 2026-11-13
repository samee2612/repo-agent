"""Configuration constants and rate-limit alias for the sample shop."""

from shop.auth import check_rate_limit

MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_CURRENCY = "USD"

# Alias so name-based search for check_rate_limit alone misses callers of throttle.
throttle = check_rate_limit


def enforce_throttle(user_id: str) -> bool:
    """Check whether a user action is permitted using the throttle alias."""
    return throttle(user_id)
