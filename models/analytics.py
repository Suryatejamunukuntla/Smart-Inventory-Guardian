"""
Smart Inventory Guardian — Analytics & ML Engine
Handles: demand forecasting, risk detection, restock recommendations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────
# RISK DETECTION
# ──────────────────────────────────────────────────────────

def detect_risks(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of flagged risk items."""
    today = datetime.today()
    risks = []

    for _, row in inventory_df.iterrows():
        product_risks = []

        # Dead Stock
        last_sold = pd.to_datetime(row.get("last_sold_date"), errors="coerce")
        if pd.notna(last_sold):
            days_idle = (today - last_sold).days
            if days_idle >= 30:
                tied_capital = row["current_stock"] * row["cost_price"]
                product_risks.append({
                    "risk_type":   "Dead Stock",
                    "risk_level":  "HIGH" if days_idle >= 45 else "MEDIUM",
                    "detail":      f"No sales in {days_idle} days — ₹{tied_capital:,.0f} capital at risk",
                    "days_idle":   days_idle,
                    "tied_capital": tied_capital,
                })

        # Near Expiry
        expiry_str = row.get("expiry_date", "N/A")
        if expiry_str and expiry_str != "N/A":
            expiry = pd.to_datetime(expiry_str, errors="coerce")
            if pd.notna(expiry):
                days_left = (expiry - today).days
                if days_left <= 14:
                    loss = row["current_stock"] * row["cost_price"]
                    product_risks.append({
                        "risk_type":   "Near Expiry",
                        "risk_level":  "CRITICAL" if days_left <= 5 else ("HIGH" if days_left <= 10 else "MEDIUM"),
                        "detail":      f"Expires in {days_left} days — potential loss ₹{loss:,.0f}",
                        "days_to_expiry": days_left,
                        "potential_loss": loss,
                    })

        # Overstock
        max_stock = row.get("max_stock", 0)
        if max_stock and row["current_stock"] > max_stock:
            excess      = row["current_stock"] - max_stock
            excess_cost = excess * row["cost_price"]
            product_risks.append({
                "risk_type":   "Overstock",
                "risk_level":  "MEDIUM",
                "detail":      f"{excess} units excess — ₹{excess_cost:,.0f} over-invested",
                "excess_units": excess,
                "excess_cost":  excess_cost,
            })

        for r in product_risks:
            risks.append({
                "product_id":    row["product_id"],
                "product_name":  row["product_name"],
                "category":      row["category"],
                "current_stock": row["current_stock"],
                **r,
            })

    cols = ["product_id","product_name","category","current_stock",
            "risk_type","risk_level","detail"]
    return pd.DataFrame(risks, columns=[c for c in cols if c in
            (cols + list(pd.DataFrame(risks).columns if risks else []))]) \
           if risks else pd.DataFrame(columns=cols)


# ──────────────────────────────────────────────────────────
# DEMAND FORECASTING
# ──────────────────────────────────────────────────────────

def compute_moving_avg(sales_df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Per-product 7-day moving average of units_sold.
    Returns: product_id, product_name, forecast_daily, forecast_7d, forecast_14d
    """
    sales_df = sales_df.copy()
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = sales_df.sort_values("date")

    results = []
    for pid, grp in sales_df.groupby("product_id"):
        grp = grp.sort_values("date")
        recent = grp.tail(window)["units_sold"]
        avg    = recent.mean() if len(recent) > 0 else 0
        std    = recent.std()  if len(recent) > 1 else 0
        results.append({
            "product_id":    pid,
            "product_name":  grp["product_name"].iloc[0],
            "category":      grp["category"].iloc[0],
            "forecast_daily": round(avg, 2),
            "forecast_7d":    round(avg * 7, 1),
            "forecast_14d":   round(avg * 14, 1),
            "demand_std":     round(std, 2),
            "trend":          _compute_trend(grp["units_sold"].values),
        })
    return pd.DataFrame(results)


def _compute_trend(series: np.ndarray) -> str:
    """Simple linear regression trend label."""
    if len(series) < 5:
        return "Stable"
    x  = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    if   slope >  0.5: return "↑ Rising"
    elif slope < -0.5: return "↓ Falling"
    else:              return "→ Stable"


def forecast_with_ml(sales_df: pd.DataFrame, inventory_df: pd.DataFrame):
    """
    LightGBM / XGBoost demand forecast (7-day per product).
    Falls back gracefully to moving average if libraries absent.
    Returns: (forecast_df, model_name)
    """
    try:
        import lightgbm as lgb
        _use = "LightGBM"
    except ImportError:
        try:
            import xgboost as xgb
            _use = "XGBoost"
        except ImportError:
            _use = None

    if _use is None:
        return compute_moving_avg(sales_df), "Moving Average (7-day)"

    sales = sales_df.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales.sort_values(["product_id", "date"])

    # Feature engineering
    sales["dow"]   = sales["date"].dt.dayofweek
    sales["month"] = sales["date"].dt.month
    sales["day"]   = sales["date"].dt.day
    sales["lag7"]  = sales.groupby("product_id")["units_sold"].shift(7)
    sales["lag14"] = sales.groupby("product_id")["units_sold"].shift(14)
    sales["roll7"] = sales.groupby("product_id")["units_sold"] \
                          .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    sales = sales.dropna(subset=["lag7", "lag14"])

    features = ["dow", "month", "day", "lag7", "lag14", "roll7", "is_weekend"]
    X = sales[features]
    y = sales["units_sold"]

    if _use == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                  num_leaves=31, random_state=42, verbose=-1)
    else:
        model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                 max_depth=4, random_state=42, verbosity=0)
    model.fit(X, y)

    # Build 7 future rows per product
    today  = datetime.today()
    rows   = []
    for pid, grp in sales.groupby("product_id"):
        grp = grp.sort_values("date")
        for d in range(1, 8):
            future_date = today + timedelta(days=d)
            lag7_val  = grp["units_sold"].iloc[-7]  if len(grp) >= 7  else grp["units_sold"].mean()
            lag14_val = grp["units_sold"].iloc[-14] if len(grp) >= 14 else grp["units_sold"].mean()
            roll7_val = grp["units_sold"].tail(7).mean()
            feat = pd.DataFrame([{
                "dow":        future_date.weekday(),
                "month":      future_date.month,
                "day":        future_date.day,
                "lag7":       lag7_val,
                "lag14":      lag14_val,
                "roll7":      roll7_val,
                "is_weekend": int(future_date.weekday() >= 5),
            }])
            pred = max(0, model.predict(feat)[0])
            rows.append({"product_id": pid, "forecast_date": future_date.strftime("%Y-%m-%d"),
                         "predicted_units": round(pred, 1)})

    pred_df = pd.DataFrame(rows)
    summary = pred_df.groupby("product_id")["predicted_units"].agg(
        forecast_7d="sum", forecast_daily="mean").reset_index()
    summary["forecast_14d"] = summary["forecast_7d"] * 2   # simple extension
    summary["demand_std"]   = pred_df.groupby("product_id")["predicted_units"].std().values

    meta = sales[["product_id","product_name","category"]].drop_duplicates()
    result = summary.merge(meta, on="product_id", how="left")
    result["trend"] = result.apply(
        lambda r: _compute_trend(
            sales[sales["product_id"] == r["product_id"]]["units_sold"].values[-30:]
        ), axis=1)
    return result, _use


# ──────────────────────────────────────────────────────────
# RESTOCK RECOMMENDATIONS
# ──────────────────────────────────────────────────────────

def generate_restock_plan(inventory_df: pd.DataFrame,
                          forecast_df:  pd.DataFrame) -> pd.DataFrame:
    """
    Merge inventory + forecast → actionable restock instructions.
    Returns rows only for items that need restocking.
    """
    merged = inventory_df.merge(
        forecast_df[["product_id","forecast_daily","forecast_7d","forecast_14d","trend"]],
        on="product_id", how="left"
    )
    # Fall back to historical avg if forecast missing
    merged["forecast_daily"] = merged["forecast_daily"].fillna(merged["avg_daily_sales"])
    merged["forecast_7d"]    = merged["forecast_7d"].fillna(merged["avg_daily_sales"] * 7)

    results = []
    for _, row in merged.iterrows():
        daily   = row["forecast_daily"]
        lead    = int(row["lead_time_days"])
        stock   = int(row["current_stock"])
        reorder = int(row["reorder_point"])
        max_s   = int(row["max_stock"])
        sell_p  = row["selling_price"]
        cost_p  = row["cost_price"]

        # Days of stock remaining
        days_remaining = round(stock / daily, 1) if daily > 0 else 999

        # Safety stock = 50 % buffer above lead-time demand
        safety  = int(daily * lead * 0.5)
        # Units to order = fill up to max_stock
        to_order = max(0, max_s - stock)

        status = "OK"
        urgency = "None"
        if stock == 0:
            status  = "OUT OF STOCK"
            urgency = "IMMEDIATE"
        elif stock <= reorder:
            status  = "Restock Now"
            urgency = "HIGH" if days_remaining <= lead else "MEDIUM"

        if status in ("OUT OF STOCK", "Restock Now"):
            order_cost  = to_order * cost_p
            order_rev   = to_order * sell_p
            deadline    = (datetime.today() + timedelta(days=max(0, days_remaining - lead))).strftime("%b %d")
            results.append({
                "product_id":      row["product_id"],
                "product_name":    row["product_name"],
                "category":        row["category"],
                "current_stock":   stock,
                "days_remaining":  days_remaining,
                "forecast_daily":  round(daily, 1),
                "lead_time_days":  lead,
                "units_to_order":  to_order,
                "order_cost":      order_cost,
                "order_revenue":   order_rev,
                "status":          status,
                "urgency":         urgency,
                "order_by":        deadline,
                "trend":           row.get("trend", "Stable"),
                "instruction":     (
                    f"Order {to_order} units of {row['product_name']} "
                    f"by {deadline} — lasts {days_remaining:.0f} days at current rate."
                ),
            })
    return pd.DataFrame(results).sort_values(
        ["urgency", "days_remaining"],
        key=lambda c: c.map({"IMMEDIATE":0,"HIGH":1,"MEDIUM":2,"None":3}) if c.name=="urgency" else c
    ).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# DASHBOARD KPI HELPERS
# ──────────────────────────────────────────────────────────

def compute_kpis(inventory_df: pd.DataFrame,
                 sales_df:     pd.DataFrame) -> dict:
    today = datetime.today()

    # Inventory health
    total_products    = len(inventory_df)
    out_of_stock      = (inventory_df["current_stock"] == 0).sum()
    low_stock         = (inventory_df["current_stock"] <= inventory_df["reorder_point"]).sum()
    total_stock_value = (inventory_df["current_stock"] * inventory_df["cost_price"]).sum()

    # Sales (last 7 days)
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    last7 = sales_df[sales_df["date"] >= (today - timedelta(days=7))]
    revenue_7d = last7["revenue"].sum()
    profit_7d  = last7["profit"].sum() if "profit" in last7 else 0
    units_7d   = last7["units_sold"].sum()

    # Expiry alerts
    expiry_critical = 0
    inv_exp = inventory_df[inventory_df["expiry_date"] != "N/A"].copy()
    if not inv_exp.empty:
        inv_exp["expiry_dt"] = pd.to_datetime(inv_exp["expiry_date"], errors="coerce")
        expiry_critical = (
            ((inv_exp["expiry_dt"] - today).dt.days >= 0) &
            ((inv_exp["expiry_dt"] - today).dt.days <= 7)
        ).sum()

    return {
        "total_products":    total_products,
        "out_of_stock":      int(out_of_stock),
        "low_stock":         int(low_stock),
        "total_stock_value": round(total_stock_value, 2),
        "revenue_7d":        round(revenue_7d, 2),
        "profit_7d":         round(profit_7d, 2),
        "units_7d":          int(units_7d),
        "expiry_critical":   int(expiry_critical),
    }
