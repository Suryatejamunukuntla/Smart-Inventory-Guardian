# 🛡️ Smart Inventory Guardian
**AI-Powered Restock & Risk Detection System**  
*Hackathon Edition 2025*

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data
```bash
python data/generate_data.py
```
This creates `sample_data/inventory.csv` and `sample_data/sales_history.csv`.

### 3. Launch the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
inventory_guardian/
├── app.py                    # ← Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── generate_data.py      # Sample data generator (32 products, 90-day history)
├── models/
│   └── analytics.py          # Core ML engine:
│                             #   • detect_risks()         — Dead stock / expiry / overstock
│                             #   • forecast_with_ml()     — LightGBM / XGBoost demand forecast
│                             #   • compute_moving_avg()   — 7-day moving average fallback
│                             #   • generate_restock_plan()— Actionable order instructions
│                             #   • compute_kpis()         — Dashboard KPI calculations
├── utils/
│   └── helpers.py            # Validation, formatting utilities
└── sample_data/              # Auto-generated CSVs (after running generate_data.py)
    ├── inventory.csv
    └── sales_history.csv
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Data Hub** | Upload CSV/Excel or load built-in sample dataset (32 products, 90 days) |
| **Guardian Dashboard** | Stock health overview, category value breakdown, real-time KPIs |
| **Risk Warnings** | Dead Stock (30+ idle days), Near Expiry (≤14 days), Overstock detection |
| **Smart Restock Plan** | Urgency-ranked orders with exact quantities, costs & deadlines |
| **Demand Forecast** | LightGBM/XGBoost model with 7-day feature engineering; falls back to moving average |
| **Trend Analysis** | Rising/Falling/Stable classification per product via linear regression |
| **Raw Data Explorer** | Filter, paginate & export any dataset slice as CSV |

---

## 🤖 ML Model Details

### Feature Engineering (for LightGBM / XGBoost)
| Feature | Description |
|---------|-------------|
| `dow` | Day of week (0=Mon … 6=Sun) |
| `month` | Month of year |
| `lag7` | Units sold 7 days ago |
| `lag14` | Units sold 14 days ago |
| `roll7` | 7-day rolling average |
| `is_weekend` | Binary flag for Saturday/Sunday |

### Fallback: Moving Average
If LightGBM/XGBoost are not installed, the app automatically uses a **7-day moving average** — no configuration needed.

---

## 📊 Sample CSV Format

### inventory.csv (required columns)
```
product_id, product_name, category, current_stock, avg_daily_sales,
cost_price, selling_price, lead_time_days, expiry_date, last_sold_date,
reorder_point, max_stock
```

### sales_history.csv (required columns)
```
date, product_id, product_name, category, units_sold,
cost_price, selling_price, revenue, profit
```

---

## 🎯 Hackathon Criteria Addressed

| Criterion | How we address it |
|-----------|-------------------|
| **Data Analysis** | Pandas-based aggregations, 7-day rolling stats, trend detection |
| **Predictive Logic** | LightGBM + lag features; Moving Average fallback |
| **UX/UI Design** | Dark professional Streamlit dashboard with custom CSS |
| **Business Logic** | Lead-time-aware safety stock, margin calculations, urgency ranking |
| **Actionable Insights** | Plain-English instruction strings per product |

---

## 👨‍💻 Tech Stack
- **Python 3.10+**
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — linear regression (trend)
- **LightGBM / XGBoost** — demand forecasting
- **Streamlit** — web dashboard
- **Plotly** — interactive charts

---

*Built for the Smart Inventory Guardian Hackathon Challenge.*
