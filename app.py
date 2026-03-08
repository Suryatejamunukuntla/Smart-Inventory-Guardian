"""
╔══════════════════════════════════════════════════════════════════╗
║         SMART INVENTORY GUARDIAN  —  Main Streamlit App          ║
║   Run:  streamlit run app.py                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os, sys

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from models.analytics import (
    detect_risks, compute_moving_avg, forecast_with_ml,
    generate_restock_plan, compute_kpis,
)
from utils.helpers import (
    validate_and_load, format_currency, days_of_stock,
    urgency_emoji, REQUIRED_INVENTORY_COLS, REQUIRED_SALES_COLS,
)

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Inventory Guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Dark background */
.stApp { background: #0d0f14; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, #151821 0%, #1a1f2e 100%);
    border: 1px solid #252d3d;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, #3b82f6);
    border-radius: 12px 12px 0 0;
}
.kpi-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #6b7a9a; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.kpi-delta { font-size: 0.82rem; margin-top: 6px; }

/* Risk badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-critical { background: #3d1515; color: #ff6b6b; border: 1px solid #ff4444; }
.badge-high     { background: #3d2a10; color: #ffb347; border: 1px solid #ff8800; }
.badge-medium   { background: #3d3510; color: #ffd700; border: 1px solid #cca800; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2330;
}

/* Alert box */
.alert-box {
    background: #1a1220;
    border-left: 4px solid #ff4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
}
.instruction-box {
    background: #0f1a25;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Dataframe */
.stDataFrame { background: #151821; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #111318; border-radius: 8px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #6b7a9a; border-radius: 6px; }
.stTabs [aria-selected="true"] { background: #1e2330 !important; color: #e2e8f0 !important; }

/* Plotly chart background */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────
if "inventory" not in st.session_state:  st.session_state.inventory = None
if "sales"     not in st.session_state:  st.session_state.sales     = None
if "forecast"  not in st.session_state:  st.session_state.forecast  = None
if "model_name" not in st.session_state: st.session_state.model_name = ""


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Inventory Guardian")
    st.markdown("<hr style='border-color:#1e2330;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

    st.markdown("### 📂 Load Data")

    # ── Auto-load sample data ──────────────────────────────────
    sample_inv_path   = os.path.join(os.path.dirname(__file__), "sample_data", "inventory.csv")
    sample_sales_path = os.path.join(os.path.dirname(__file__), "sample_data", "sales_history.csv")

    use_sample = st.button("▶ Load Sample Dataset", use_container_width=True, type="primary")
    if use_sample:
        if os.path.exists(sample_inv_path) and os.path.exists(sample_sales_path):
            st.session_state.inventory  = pd.read_csv(sample_inv_path)
            st.session_state.sales      = pd.read_csv(sample_sales_path)
            st.session_state.forecast   = None
            st.success("Sample data loaded!")
        else:
            st.warning("Run `python data/generate_data.py` first to generate sample data.")

    st.markdown("**Or upload your own:**")
    inv_file   = st.file_uploader("Inventory CSV / Excel",   type=["csv","xlsx"])
    sales_file = st.file_uploader("Sales History CSV / Excel", type=["csv","xlsx"])

    if inv_file:
        df, errs = validate_and_load(inv_file, REQUIRED_INVENTORY_COLS)
        if errs: st.error("\n".join(errs))
        else:
            st.session_state.inventory = df
            st.session_state.forecast  = None
            st.success(f"Inventory loaded: {len(df)} products")

    if sales_file:
        df, errs = validate_and_load(sales_file, REQUIRED_SALES_COLS)
        if errs: st.error("\n".join(errs))
        else:
            st.session_state.sales    = df
            st.session_state.forecast = None
            st.success(f"Sales loaded: {len(df):,} records")

    st.markdown("<hr style='border-color:#1e2330;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Forecast Settings")
    forecast_model = st.selectbox("Model", ["Auto (LightGBM/XGBoost)", "Moving Average (7-day)"])
    forecast_days  = st.slider("Forecast horizon (days)", 7, 30, 7)

    if st.button("🔮 Run Forecast", use_container_width=True):
        if st.session_state.sales is not None and st.session_state.inventory is not None:
            with st.spinner("Training forecast model…"):
                if forecast_model.startswith("Auto"):
                    fc, mn = forecast_with_ml(st.session_state.sales, st.session_state.inventory)
                else:
                    fc, mn = compute_moving_avg(st.session_state.sales), "Moving Average"
                st.session_state.forecast   = fc
                st.session_state.model_name = mn
            st.success(f"Forecast ready — {mn}")
        else:
            st.warning("Load both inventory & sales data first.")

    st.markdown("<hr style='border-color:#1e2330;margin:1rem 0'>", unsafe_allow_html=True)
    st.caption("Smart Inventory Guardian v1.0")
    st.caption("Hackathon Edition 2025")


# ══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════
inv_df   = st.session_state.inventory
sales_df = st.session_state.sales
fc_df    = st.session_state.forecast

# ── Header ──────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
  <div style="font-size:2.4rem;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🛡️ Smart Inventory Guardian
  </div>
</div>
<div style="color:#6b7a9a;font-size:0.9rem;margin-bottom:1.5rem;">
  AI-Powered Restock & Risk Detection System — Hackathon 2025
</div>
""", unsafe_allow_html=True)

if inv_df is None:
    st.info("👈  Load the sample dataset or upload your own files using the sidebar to get started.")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────────
kpis = compute_kpis(inv_df, sales_df if sales_df is not None else pd.DataFrame())

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_data = [
    (c1, "Total Products",   str(kpis["total_products"]),   "#3b82f6", ""),
    (c2, "Out of Stock",     str(kpis["out_of_stock"]),     "#ef4444", "⚠️" if kpis["out_of_stock"] else "✅"),
    (c3, "Low Stock",        str(kpis["low_stock"]),        "#f97316", "⚠️" if kpis["low_stock"]    else "✅"),
    (c4, "Stock Value",      format_currency(kpis["total_stock_value"]), "#8b5cf6", ""),
    (c5, "Revenue (7d)",     format_currency(kpis["revenue_7d"]),         "#10b981", ""),
    (c6, "Expiry Alerts",   str(kpis["expiry_critical"]),   "#f59e0b", "🔴" if kpis["expiry_critical"] else "✅"),
]
for col, label, val, accent, icon in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--accent:{accent}">
          <div class="kpi-value">{icon} {val}</div>
          <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Guardian Dashboard",
    "🚨 Risk Warnings",
    "📦 Restock Plan",
    "📈 Demand Forecast",
    "🔍 Raw Data",
])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ──────────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1.4, 1])

    with left:
        st.markdown('<div class="section-header">📦 Stock Health Overview</div>', unsafe_allow_html=True)

        # Stock status buckets
        def stock_status(row):
            if row["current_stock"] == 0:          return "Out of Stock"
            if row["current_stock"] <= row["reorder_point"]: return "Low Stock"
            if row["current_stock"] > row["max_stock"]:      return "Overstock"
            return "Healthy"

        inv_df["stock_status"] = inv_df.apply(stock_status, axis=1)
        counts = inv_df["stock_status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]

        color_map = {
            "Healthy":      "#10b981",
            "Low Stock":    "#f97316",
            "Out of Stock": "#ef4444",
            "Overstock":    "#8b5cf6",
        }
        fig = px.bar(
            counts, x="Status", y="Count", color="Status",
            color_discrete_map=color_map, text="Count",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False,
            margin=dict(t=20, b=10, l=10, r=10), height=280,
        )
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
        fig.update_xaxes(gridcolor="#1e2330", linecolor="#1e2330")
        fig.update_yaxes(gridcolor="#1e2330", linecolor="#1e2330")
        st.plotly_chart(fig, use_container_width=True)

        # Category stock value
        st.markdown('<div class="section-header">💰 Stock Value by Category</div>', unsafe_allow_html=True)
        inv_df["stock_value"] = inv_df["current_stock"] * inv_df["cost_price"]
        cat_val = inv_df.groupby("category")["stock_value"].sum().reset_index().sort_values("stock_value")
        fig2 = px.bar(cat_val, x="stock_value", y="category", orientation="h",
                      color="stock_value", color_continuous_scale="Blues")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10), height=300,
        )
        fig2.update_xaxes(gridcolor="#1e2330", linecolor="#1e2330")
        fig2.update_yaxes(gridcolor="#1e2330", linecolor="#1e2330")
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">🔴 Immediate Attention</div>', unsafe_allow_html=True)

        # Critical items
        critical = inv_df[inv_df["current_stock"] == 0]
        low      = inv_df[
            (inv_df["current_stock"] > 0) &
            (inv_df["current_stock"] <= inv_df["reorder_point"])
        ].head(6)

        if not critical.empty:
            for _, r in critical.iterrows():
                st.markdown(f"""
                <div class="alert-box">
                  <b>🔴 OUT OF STOCK</b> — {r['product_name']}<br>
                  <span style="color:#6b7a9a">Lead time: {r['lead_time_days']}d &nbsp;|&nbsp; Reorder at: {r['reorder_point']} units</span>
                </div>""", unsafe_allow_html=True)

        if not low.empty:
            for _, r in low.iterrows():
                dos = days_of_stock(r["current_stock"], r["avg_daily_sales"])
                st.markdown(f"""
                <div class="alert-box" style="border-color:#f97316;">
                  <b>🟠 LOW STOCK</b> — {r['product_name']}<br>
                  <span style="color:#6b7a9a">{r['current_stock']} units left (~{dos:.0f} days) &nbsp;|&nbsp; Lead: {r['lead_time_days']}d</span>
                </div>""", unsafe_allow_html=True)

        if critical.empty and low.empty:
            st.success("✅ All products are adequately stocked!")

        st.markdown('<br><div class="section-header">📊 Sales by Category (7d)</div>', unsafe_allow_html=True)
        if sales_df is not None:
            sales_df["date"] = pd.to_datetime(sales_df["date"])
            last7 = sales_df[sales_df["date"] >= (datetime.today() - timedelta(days=7))]
            cat_s = last7.groupby("category")["revenue"].sum().reset_index()
            fig3 = px.pie(cat_s, values="revenue", names="category",
                          color_discrete_sequence=px.colors.sequential.Blues_r, hole=0.5)
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
                margin=dict(t=10, b=10, l=10, r=10), height=260,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 2 — RISK WARNINGS
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">⚠️ Risk Detection Engine</div>', unsafe_allow_html=True)
    risks_df = detect_risks(inv_df)

    if risks_df.empty:
        st.success("🎉 No risk items detected. Inventory looks healthy!")
    else:
        # Summary row
        c1, c2, c3 = st.columns(3)
        dead   = risks_df[risks_df["risk_type"] == "Dead Stock"]
        expiry = risks_df[risks_df["risk_type"] == "Near Expiry"]
        over   = risks_df[risks_df["risk_type"] == "Overstock"]
        with c1:
            st.metric("Dead Stock Items",   len(dead),
                      delta=f"-₹{dead['tied_capital'].sum():,.0f}" if 'tied_capital' in dead.columns and not dead.empty else "")
        with c2:
            st.metric("Near Expiry Items",  len(expiry),
                      delta=f"-₹{expiry['potential_loss'].sum():,.0f}" if 'potential_loss' in expiry.columns and not expiry.empty else "")
        with c3:
            st.metric("Overstock Items",    len(over),
                      delta=f"-₹{over['excess_cost'].sum():,.0f}" if 'excess_cost' in over.columns and not over.empty else "")

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter
        risk_types = ["All"] + list(risks_df["risk_type"].unique())
        sel = st.selectbox("Filter by Risk Type", risk_types)
        display = risks_df if sel == "All" else risks_df[risks_df["risk_type"] == sel]

        for _, r in display.iterrows():
            lvl  = r.get("risk_level", "MEDIUM")
            badge = f'<span class="badge badge-{lvl.lower()}">{lvl}</span>'
            icon  = {"Dead Stock":"💀","Near Expiry":"⏳","Overstock":"📦"}.get(r["risk_type"],"⚠️")
            st.markdown(f"""
            <div class="alert-box" style="border-color:{'#ff4444' if lvl=='CRITICAL' else '#ff8800' if lvl=='HIGH' else '#ffd700'}">
              {icon} <b>{r['product_name']}</b> &nbsp; {badge} &nbsp;
              <span style="color:#6b7a9a;font-size:0.8rem">{r['category']}</span><br>
              <span style="color:#94a3b8">{r['detail']}</span>
            </div>""", unsafe_allow_html=True)

        # Chart: risk distribution
        if not risks_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_r = px.sunburst(
                risks_df, path=["risk_type","risk_level","product_name"],
                color="risk_type",
                color_discrete_map={"Dead Stock":"#8b5cf6","Near Expiry":"#ef4444","Overstock":"#f97316"},
                title="Risk Distribution",
            )
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
                margin=dict(t=40, b=10, l=10, r=10), height=400,
            )
            st.plotly_chart(fig_r, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 — RESTOCK PLAN
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📦 Smart Restock Recommendations</div>', unsafe_allow_html=True)

    if fc_df is None:
        st.info("Run the forecast first (sidebar → Run Forecast) for ML-powered recommendations. Showing rule-based plan.")
        fc_df_use = compute_moving_avg(sales_df) if sales_df is not None else None
    else:
        fc_df_use = fc_df
        st.caption(f"Forecast model: **{st.session_state.model_name}**")

    if fc_df_use is not None:
        restock = generate_restock_plan(inv_df, fc_df_use)

        if restock.empty:
            st.success("✅ No immediate restocking needed!")
        else:
            # Summary
            total_cost = restock["order_cost"].sum()
            urgent     = restock[restock["urgency"].isin(["IMMEDIATE","HIGH"])]
            st.markdown(f"""
            <div style="background:#0f1a25;border:1px solid #1e4a6e;border-radius:10px;padding:16px 20px;margin-bottom:16px;">
              📋 <b>{len(restock)}</b> items need restocking &nbsp;|&nbsp;
              🔴 <b>{len(urgent)}</b> urgent &nbsp;|&nbsp;
              💰 Total order cost: <b>{format_currency(total_cost)}</b>
            </div>""", unsafe_allow_html=True)

            # Actionable instructions
            st.markdown("#### 🎯 Action Items")
            for _, r in restock.iterrows():
                emoji = urgency_emoji(r["urgency"])
                st.markdown(f"""
                <div class="instruction-box">
                  {emoji} {r['instruction']}
                  <span style="float:right;color:#6b7a9a;font-size:0.78rem">
                    Cost: {format_currency(r['order_cost'])} &nbsp;|&nbsp; {r['trend']}
                  </span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Table
            display_cols = ["product_name","category","current_stock","days_remaining",
                            "forecast_daily","units_to_order","order_cost","urgency","order_by"]
            st.dataframe(
                restock[display_cols].rename(columns={
                    "product_name":   "Product",
                    "category":       "Category",
                    "current_stock":  "Stock",
                    "days_remaining": "Days Left",
                    "forecast_daily": "Daily Fcst",
                    "units_to_order": "Order Qty",
                    "order_cost":     "Order Cost (₹)",
                    "urgency":        "Urgency",
                    "order_by":       "Order By",
                }),
                use_container_width=True, hide_index=True,
            )

            # Bar chart
            fig_r = px.bar(
                restock.head(15).sort_values("order_cost"),
                x="order_cost", y="product_name", orientation="h",
                color="urgency",
                color_discrete_map={"IMMEDIATE":"#ef4444","HIGH":"#f97316","MEDIUM":"#f59e0b"},
                title="Top Items by Restock Cost",
                labels={"order_cost":"Order Cost (₹)","product_name":""},
            )
            fig_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", margin=dict(t=40,b=10,l=10,r=10), height=420,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            fig_r.update_xaxes(gridcolor="#1e2330")
            fig_r.update_yaxes(gridcolor="#1e2330")
            st.plotly_chart(fig_r, use_container_width=True)
    else:
        st.warning("Load sales data to generate a restock plan.")


# ──────────────────────────────────────────────────────────────────
# TAB 4 — DEMAND FORECAST
# ──────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📈 Demand Forecast & Trends</div>', unsafe_allow_html=True)

    if sales_df is None:
        st.warning("Load sales history data to view forecasts.")
    else:
        # Historical sales trend
        sales_df["date"] = pd.to_datetime(sales_df["date"])
        daily = sales_df.groupby("date")[["revenue","units_sold","profit"]].sum().reset_index()

        st.markdown("#### Daily Revenue — Last 90 Days")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=daily["date"], y=daily["revenue"],
            mode="lines", name="Revenue",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
        ))
        fig_t.add_trace(go.Scatter(
            x=daily["date"], y=daily["profit"],
            mode="lines", name="Profit",
            line=dict(color="#10b981", width=2),
        ))
        fig_t.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2330", linecolor="#1e2330"),
            yaxis=dict(gridcolor="#1e2330", linecolor="#1e2330"),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Forecast per product
        if fc_df is not None:
            st.markdown(f"#### 7-Day Demand Forecast  ·  *{st.session_state.model_name}*")
            top_n = fc_df.nlargest(15, "forecast_7d")
            fig_f = px.bar(
                top_n, x="forecast_7d", y="product_name", orientation="h",
                color="forecast_7d", color_continuous_scale="Blues",
                labels={"forecast_7d":"Predicted Units (7d)","product_name":""},
            )
            fig_f.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", height=420, coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            fig_f.update_xaxes(gridcolor="#1e2330")
            fig_f.update_yaxes(gridcolor="#1e2330")
            st.plotly_chart(fig_f, use_container_width=True)

            # Trend table
            st.markdown("#### Trend Analysis")
            trend_df = fc_df[["product_name","category","forecast_daily","forecast_7d","trend"]] \
                       .sort_values("forecast_7d", ascending=False)
            st.dataframe(trend_df.rename(columns={
                "product_name":   "Product",
                "category":       "Category",
                "forecast_daily": "Daily Forecast",
                "forecast_7d":    "7-Day Forecast",
                "trend":          "Trend",
            }), use_container_width=True, hide_index=True)

        else:
            # Day-of-week heatmap
            st.markdown("#### Sales by Day of Week")
            dow_data = sales_df.copy()
            dow_data["dow"] = dow_data["date"].dt.day_name()
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_avg = dow_data.groupby(["category","dow"])["units_sold"].mean().reset_index()
            fig_h = px.density_heatmap(
                dow_avg, x="dow", y="category", z="units_sold",
                category_orders={"dow": dow_order},
                color_continuous_scale="Blues",
            )
            fig_h.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94a3b8", height=350,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_h, use_container_width=True)
            st.info("👈 Run the forecast model in the sidebar to see ML-powered predictions.")


# ──────────────────────────────────────────────────────────────────
# TAB 5 — RAW DATA
# ──────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔍 Raw Data Explorer</div>', unsafe_allow_html=True)

    sub = st.radio("Dataset", ["Inventory", "Sales History"], horizontal=True)

    if sub == "Inventory":
        df_show = inv_df.copy()
        cats = ["All"] + sorted(df_show["category"].unique().tolist())
        cat_f = st.selectbox("Filter by Category", cats)
        if cat_f != "All":
            df_show = df_show[df_show["category"] == cat_f]
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.caption(f"{len(df_show)} rows")

        # Download
        csv = df_show.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "inventory_filtered.csv", "text/csv")

    else:
        if sales_df is None:
            st.warning("No sales data loaded.")
        else:
            df_show = sales_df.copy()
            df_show["date"] = pd.to_datetime(df_show["date"])
            cats = ["All"] + sorted(df_show["category"].unique().tolist())
            cat_f = st.selectbox("Filter by Category", cats)
            if cat_f != "All":
                df_show = df_show[df_show["category"] == cat_f]
            date_range = st.date_input(
                "Date range",
                value=[df_show["date"].min().date(), df_show["date"].max().date()],
            )
            if len(date_range) == 2:
                df_show = df_show[
                    (df_show["date"].dt.date >= date_range[0]) &
                    (df_show["date"].dt.date <= date_range[1])
                ]
            st.dataframe(df_show, use_container_width=True, hide_index=True)
            st.caption(f"{len(df_show):,} rows")
            csv = df_show.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv, "sales_filtered.csv", "text/csv")
