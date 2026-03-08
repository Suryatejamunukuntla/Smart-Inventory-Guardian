"""
Smart Inventory Guardian — Sample Data Generator
Run once to create sample_data/inventory.csv and sample_data/sales_history.csv
    python data/generate_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

random.seed(42)
np.random.seed(42)

# (name, category, cost, sell, avg_daily_sales, lead_days, has_expiry)
PRODUCTS = [
    ("Organic Whole Milk 1L",   "Dairy",         45,  75, 12, 2, True),
    ("Greek Yogurt 500g",       "Dairy",         55,  90,  8, 2, True),
    ("Cheddar Cheese 200g",     "Dairy",         80, 130,  5, 3, True),
    ("Eggs (12pk)",             "Dairy",         75, 120, 15, 2, True),
    ("Fresh Bread Loaf",        "Bakery",        25,  45, 20, 1, True),
    ("Croissants (6pk)",        "Bakery",        60,  95,  7, 1, True),
    ("Paracetamol 500mg 20s",   "Pharmacy",      35,  65,  6, 5, True),
    ("Ibuprofen 400mg 16s",     "Pharmacy",      42,  78,  4, 5, True),
    ("Vitamin C 1000mg 30s",    "Pharmacy",      90, 160,  3, 5, True),
    ("Antihistamine 10mg 30s",  "Pharmacy",      55, 105,  2, 5, True),
    ("Hand Sanitizer 250ml",    "Health",        28,  55,  5, 4, False),
    ("Face Masks (50pk)",       "Health",       120, 195,  2, 4, False),
    ("Basmati Rice 5kg",        "Grains",       180, 280,  4, 7, False),
    ("Pasta Penne 500g",        "Grains",        22,  38,  9, 5, False),
    ("Olive Oil 500ml",         "Condiments",    95, 165,  3, 6, False),
    ("Tomato Sauce 400g",       "Condiments",    18,  32, 11, 5, True),
    ("Orange Juice 1L",         "Beverages",     38,  65, 14, 3, True),
    ("Mineral Water 6pk",       "Beverages",     45,  72, 18, 3, False),
    ("Coffee Beans 250g",       "Beverages",    120, 210,  4, 7, False),
    ("Green Tea 25 bags",       "Beverages",     55,  90,  3, 6, False),
    ("Chicken Breast 500g",     "Meat",          95, 165,  9, 2, True),
    ("Beef Mince 500g",         "Meat",         110, 185,  6, 2, True),
    ("Bananas 1kg",             "Produce",       28,  48, 16, 2, True),
    ("Apples 1kg",              "Produce",       45,  75, 12, 2, True),
    ("Spinach 200g",            "Produce",       30,  55,  7, 2, True),
    ("Tomatoes 500g",           "Produce",       35,  58, 10, 2, True),
    ("Shampoo 400ml",           "Personal Care", 85, 145,  2, 8, False),
    ("Body Lotion 200ml",       "Personal Care", 75, 128,  2, 8, False),
    ("Toothpaste 150g",         "Personal Care", 45,  78,  4, 7, False),
    ("Laundry Detergent 1kg",   "Household",    110, 175,  3, 6, False),
    ("Dish Soap 500ml",         "Household",     35,  58,  5, 5, False),
    ("Toilet Paper 12pk",       "Household",    145, 225,  6, 5, False),
]


def generate_sales_history(days: int = 90) -> pd.DataFrame:
    rows = []
    base = datetime.today() - timedelta(days=days)
    for pid, (name, cat, cost, sell, avg, lead, _) in enumerate(PRODUCTS, 1):
        for d in range(days):
            date    = base + timedelta(days=d)
            weekend = 1.4  if date.weekday() >= 5 else 1.0
            trend   = 1 + (d / days) * 0.10
            noise   = max(0, np.random.normal(1.0, 0.22))
            spike   = 2.2  if random.random() < 0.025 else 1.0
            units   = max(0, int(avg * weekend * trend * noise * spike))
            rows.append({
                "date":          date.strftime("%Y-%m-%d"),
                "product_id":    f"P{pid:03d}",
                "product_name":  name,
                "category":      cat,
                "units_sold":    units,
                "cost_price":    cost,
                "selling_price": sell,
                "revenue":       units * sell,
                "profit":        units * (sell - cost),
                "is_weekend":    int(date.weekday() >= 5),
            })
    return pd.DataFrame(rows)


def generate_inventory() -> pd.DataFrame:
    today = datetime.today()
    rows  = []
    for pid, (name, cat, cost, sell, avg, lead, has_exp) in enumerate(PRODUCTS, 1):
        scenario = random.choices(
            ["critical", "low", "normal", "overstock"],
            weights=[0.15, 0.25, 0.45, 0.15]
        )[0]
        if   scenario == "critical":  stock = random.randint(0, max(1, int(avg * 1.5)))
        elif scenario == "low":       stock = random.randint(int(avg*1.5), int(avg*4))
        elif scenario == "normal":    stock = random.randint(int(avg*4),   int(avg*10))
        else:                         stock = random.randint(int(avg*10),  int(avg*20))

        expiry    = (today + timedelta(days=random.randint(3, 60))).strftime("%Y-%m-%d") if has_exp else "N/A"
        days_ago  = random.randint(30, 60) if random.random() < 0.08 else random.randint(0, 4)
        last_sold = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        rows.append({
            "product_id":      f"P{pid:03d}",
            "product_name":    name,
            "category":        cat,
            "current_stock":   stock,
            "avg_daily_sales": avg,
            "cost_price":      cost,
            "selling_price":   sell,
            "margin_pct":      round((sell - cost) / sell * 100, 1),
            "lead_time_days":  lead,
            "expiry_date":     expiry,
            "last_sold_date":  last_sold,
            "reorder_point":   int(avg * lead * 1.5),
            "max_stock":       int(avg * 14),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "sample_data")
    os.makedirs(out, exist_ok=True)
    inv   = generate_inventory()
    sales = generate_sales_history(90)
    inv.to_csv(  os.path.join(out, "inventory.csv"),     index=False)
    sales.to_csv(os.path.join(out, "sales_history.csv"), index=False)
    print(f"✅  inventory.csv     — {len(inv)} products")
    print(f"✅  sales_history.csv — {len(sales):,} records (90 days)")
