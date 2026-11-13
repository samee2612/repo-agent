"""Tests for authentication and rate limiting in the sample shop."""

from shop.auth import check_rate_limit, login


def test_check_rate_limit_allows_valid_user() -> None:
    """Verify check_rate_limit returns True for a normal user id."""
    assert check_rate_limit("alice") is True


def test_check_rate_limit_rejects_empty_user() -> None:
    """Verify check_rate_limit returns False for an empty user id."""
    assert check_rate_limit("") is False


def test_login_demo_user() -> None:
    """Verify login succeeds with the synthetic demo password."""
    result = login("alice", "demo-alice")
    assert result["authenticated"] is True
