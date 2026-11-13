"""Payment calculation helpers with no dependency on auth."""


def calculate_total(prices: list[float], tax_rate: float = 0.08) -> float:
    """Calculate total payment amount including tax."""
    subtotal = sum(prices)
    return round(subtotal * (1.0 + tax_rate), 2)


def format_receipt(order_id: str, total: float, currency: str = "USD") -> str:
    """Format a plain-text payment receipt."""
    return f"Order {order_id}: {total:.2f} {currency}"
