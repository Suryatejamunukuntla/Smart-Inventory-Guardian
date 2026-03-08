"""
Smart Inventory Guardian — Utility Helpers
"""
import pandas as pd
import io


REQUIRED_INVENTORY_COLS = [
    "product_id", "product_name", "category",
    "current_stock", "avg_daily_sales", "cost_price",
    "selling_price", "lead_time_days",
]
REQUIRED_SALES_COLS = [
    "date", "product_id", "product_name", "units_sold",
]


def validate_and_load(file_obj, required_cols: list) -> tuple[pd.DataFrame, list]:
    """Load CSV/Excel from a file-like object. Returns (df, errors)."""
    errors = []
    try:
        name = getattr(file_obj, "name", "")
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(file_obj)
        else:
            df = pd.read_csv(file_obj)
    except Exception as e:
        return pd.DataFrame(), [f"File read error: {e}"]

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
    return df, errors


def format_currency(value: float, symbol: str = "₹") -> str:
    if value >= 1_000_000:
        return f"{symbol}{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{symbol}{value/1_000:.1f}K"
    return f"{symbol}{value:,.0f}"


def days_of_stock(stock: float, daily_sales: float) -> float:
    return round(stock / daily_sales, 1) if daily_sales > 0 else 999


def risk_color(level: str) -> str:
    return {"CRITICAL": "#FF4444", "HIGH": "#FF8800",
            "MEDIUM": "#FFD700", "LOW": "#00C48C"}.get(level, "#888")


def urgency_emoji(urgency: str) -> str:
    return {"IMMEDIATE": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "None": "🟢"}.get(urgency, "⚪")
