"""
Slooze Take-home — Inventory, Purchase, Sales Analysis & Optimization
Outputs will be written to output folder.
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Optional forecasting: Prophet may be used if installed
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Config
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

HOLDING_COST_PCT = 0.20   # annual holding cost % of unit cost
ORDERING_COST = 50        # fixed ordering cost
SAFETY_STOCK_Z = 1.65     # approx z for 95% service
LEAD_TIME_DAYS_DEFAULT = 7
FORECAST_WEEKS = 12

# Filenames (your files)
FILES = {
    "sales": "SalesFINAL12312016.csv",
    "purchases": "PurchasesFINAL12312016.csv",
    "beg_inv": "BegInvFINAL12312016.csv",
    "end_inv": "EndInvFINAL12312016.csv",
    "purchase_prices": "2017PurchasePricesDec.csv",
    "invoice_purchases": "InvoicePurchases12312016.csv"
}

def load_csv(fname):
    p = DATA_DIR / fname
    if not p.exists():
        print(f"Warning: {p} not found.")
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"Failed to read {p}: {e}")
        return None

# Load
sales = load_csv(FILES["sales"])
purchases = load_csv(FILES["purchases"])
beg_inv = load_csv(FILES["beg_inv"])
end_inv = load_csv(FILES["end_inv"])
purchase_prices = load_csv(FILES["purchase_prices"])
invoice_purchases = load_csv(FILES["invoice_purchases"])

print("Loaded shapes:", {k: (v.shape if v is not None else None) for k,v in [
    ("sales", sales), ("purchases", purchases), ("beg_inv", beg_inv),
    ("end_inv", end_inv), ("purchase_prices", purchase_prices), ("invoice_purchases", invoice_purchases)
]})

# Normalize column names (strip)
def norm_cols(df):
    if df is None:
        return None
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

sales = norm_cols(sales)
purchases = norm_cols(purchases)
beg_inv = norm_cols(beg_inv)
end_inv = norm_cols(end_inv)
purchase_prices = norm_cols(purchase_prices)
invoice_purchases = norm_cols(invoice_purchases)

# Convert dates where present
def to_datetime_if_exists(df, cols):
    if df is None: return df
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

sales = to_datetime_if_exists(sales, ["SalesDate"])
purchases = to_datetime_if_exists(purchases, ["PODate", "ReceivingDate", "InvoiceDate", "PayDate"])
beg_inv = to_datetime_if_exists(beg_inv, ["startDate"])
end_inv = to_datetime_if_exists(end_inv, ["endDate"])
invoice_purchases = to_datetime_if_exists(invoice_purchases, ["InvoiceDate", "PODate", "PayDate"])

# Standardize column names used in analysis
# Sales: InventoryId, SalesQuantity, SalesDollars, SalesPrice, SalesDate
if sales is not None:
    # rename lower/alt names if needed (already match per your output)
    if "InventoryId" not in sales.columns and "InventoryID" in sales.columns:
        sales = sales.rename(columns={"InventoryID":"InventoryId"})
    # ensure numeric
    sales["SalesQuantity"] = pd.to_numeric(sales.get("SalesQuantity", sales.get("SalesQuantity", 0)), errors="coerce").fillna(0)
    sales["SalesDollars"] = pd.to_numeric(sales.get("SalesDollars", sales.get("SalesDollars", 0)), errors="coerce").fillna(0)
    if "SalesPrice" in sales.columns:
        sales["SalesPrice"] = pd.to_numeric(sales["SalesPrice"], errors="coerce").fillna(0)
    # ensure InventoryId as str for grouping
    if "InventoryId" in sales.columns:
        sales["InventoryId"] = sales["InventoryId"].astype(str)

# Purchases: InventoryId, PurchasePrice, Quantity, PODate, ReceivingDate, VendorNumber, VendorName
if purchases is not None:
    if "InventoryId" in purchases.columns:
        purchases["InventoryId"] = purchases["InventoryId"].astype(str)
    purchases["Quantity"] = pd.to_numeric(purchases.get("Quantity", 0), errors="coerce").fillna(0)
    purchases["PurchasePrice"] = pd.to_numeric(purchases.get("PurchasePrice", purchases.get("PurchasePrice", 0)), errors="coerce").fillna(0)

# Inventory: use end_inv if available (latest snapshot)
inventory = None
if end_inv is not None:
    inventory = end_inv.copy()
elif beg_inv is not None:
    inventory = beg_inv.copy()

if inventory is not None:
    if "InventoryId" in inventory.columns:
        inventory["InventoryId"] = inventory["InventoryId"].astype(str)
    # unify onHand column name
    if "onHand" in inventory.columns and "on_hand" not in inventory.columns:
        inventory = inventory.rename(columns={"onHand": "on_hand"})
    inventory["on_hand"] = pd.to_numeric(inventory.get("on_hand", 0), errors="coerce").fillna(0)
    if "Price" in inventory.columns:
        inventory["Price"] = pd.to_numeric(inventory["Price"], errors="coerce").fillna(0)

#  EDA: top products by revenue & quantity
if sales is not None:
    sales["revenue"] = sales.get("SalesDollars", sales.get("SalesPrice",0) * sales.get("SalesQuantity",0))
    top_rev = sales.groupby("InventoryId").agg(revenue=("revenue","sum"), qty=("SalesQuantity","sum")).reset_index().sort_values("revenue", ascending=False)
    top_rev.to_csv(OUTPUT_DIR / "top_products_by_revenue.csv", index=False)
    print("Saved top_products_by_revenue.csv")

#  Demand Forecasting (per InventoryId)
def prepare_ts_for_sku(df_sales, sku, freq="W"):
    if df_sales is None: return None
    s = df_sales[df_sales["InventoryId"] == str(sku)].copy()
    if s.empty or "SalesDate" not in s.columns:
        return None
    s = s.set_index("SalesDate").resample(freq).agg({"SalesQuantity":"sum"}).rename(columns={"SalesQuantity":"y"}).reset_index().rename(columns={"SalesDate":"ds"})
    return s

def rf_forecast(ts_df, periods=FORECAST_WEEKS):
    ts = ts_df.set_index("ds").asfreq("W").fillna(0)
    df = ts.copy()
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df["rmean4"] = df["y"].rolling(4).mean()
    df = df.dropna()
    if df.empty:
        return None
    X = df[["lag1","lag2","rmean4"]]
    y = df["y"]
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X,y)
    preds = []
    last = X.iloc[-1].values
    for _ in range(periods):
        p = model.predict([last])[0]
        preds.append(max(0,p))
        last = [p, last[0], (p + last[0] + last[1]) / 3.0]
    idx = pd.date_range(start=ts.index[-1] + pd.Timedelta(weeks=1), periods=periods, freq='W')
    return pd.DataFrame({"ds": idx, "yhat": preds})

forecasts_list = []
if sales is not None and not sales.empty:
    top_skus = top_rev["InventoryId"].head(10).tolist()
    for sku in top_skus:
        ts = prepare_ts_for_sku(sales, sku)
        if ts is None or ts.shape[0] < 8:
            continue
        f = None
        if Prophet is not None:
            try:
                f = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                f.fit(ts.rename(columns={"ds":"ds","y":"y"}))
                future = f.make_future_dataframe(periods=FORECAST_WEEKS, freq='W')
                fc = f.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
                fc["InventoryId"] = sku
                forecasts_list.append(fc)
                continue
            except Exception:
                f = None
        # fallback RF
        rf = rf_forecast(ts, periods=FORECAST_WEEKS)
        if rf is not None:
            rf["InventoryId"] = sku
            forecasts_list.append(rf)

if forecasts_list:
    pd.concat(forecasts_list, ignore_index=True).to_csv(OUTPUT_DIR / "forecasts_top10.csv", index=False)
    print("Saved forecasts_top10.csv")

# === ABC classification by revenue ===
if sales is not None:
    abc = top_rev.copy()
    abc["cum_value"] = abc["revenue"].cumsum()
    abc["cum_pct"] = abc["cum_value"] / abc["revenue"].sum()
    def abc_label(x):
        if x <= 0.7:
            return "A"
        elif x <= 0.9:
            return "B"
        else:
            return "C"
    abc["ABC"] = abc["cum_pct"].apply(abc_label)
    abc.to_csv(OUTPUT_DIR / "abc_classification.csv", index=False)
    print("Saved abc_classification.csv")

# EOQ (requires annual demand and unit cost)
if sales is not None:
    annual = sales.groupby("InventoryId").agg(qty=("SalesQuantity","sum")).reset_index()
    # Try to get unit cost from purchases (PurchasePrice)
    if purchases is not None and "PurchasePrice" in purchases.columns:
        avg_cost = purchases.groupby("InventoryId").agg(unit_cost=("PurchasePrice","mean")).reset_index()
        annual = annual.merge(avg_cost, on="InventoryId", how="left")
    # fallback unit cost from inventory Price
    if "unit_cost" not in annual.columns or annual["unit_cost"].isna().all():
        if inventory is not None and "Price" in inventory.columns:
            inv_cost = inventory[["InventoryId","Price"]].drop_duplicates(subset=["InventoryId"])
            inv_cost = inv_cost.rename(columns={"Price":"unit_cost"})
            annual = annual.merge(inv_cost, on="InventoryId", how="left")
    annual["unit_cost"] = pd.to_numeric(annual.get("unit_cost", 10.0), errors="coerce").fillna(10.0)
    date_min = sales["SalesDate"].min() if "SalesDate" in sales.columns else None
    date_max = sales["SalesDate"].max() if "SalesDate" in sales.columns else None
    days_span = (date_max - date_min).days if date_min is not None and date_max is not None else 365
    days_span = max(days_span, 1)
    annual["D"] = annual["qty"] / days_span * 365
    annual["H"] = annual["unit_cost"] * HOLDING_COST_PCT
    # EOQ formula
    def calc_eoq(row):
        H = row["H"]
        D = row["D"]
        S = ORDERING_COST
        if H <= 0:
            return 0
        return int(round(math.sqrt((2 * D * S) / H)))
    annual["EOQ"] = annual.apply(calc_eoq, axis=1)
    annual.to_csv(OUTPUT_DIR / "eoq_by_sku.csv", index=False)
    print("Saved eoq_by_sku.csv")

#  Reorder point & safety stock
if sales is not None and "SalesDate" in sales.columns:
    # compute daily demand per SKU
    daily = sales.set_index("SalesDate").groupby("InventoryId")["SalesQuantity"].resample("D").sum().unstack(level=0).fillna(0)
    rop_rows = []
    for sku in daily.columns:
        series = daily[sku]
        avg_daily = series.mean()
        std_daily = series.std()
        lead = LEAD_TIME_DAYS_DEFAULT
        safety = SAFETY_STOCK_Z * (std_daily if not np.isnan(std_daily) else 0) * np.sqrt(lead)
        rop = avg_daily * lead + safety
        rop_rows.append({
            "InventoryId": sku,
            "avg_daily": float(avg_daily),
            "std_daily": float(std_daily if not np.isnan(std_daily) else 0.0),
            "lead_time": lead,
            "safety_stock": float(safety),
            "reorder_point": float(rop)
        })
    rop_df = pd.DataFrame(rop_rows)
    rop_df.to_csv(OUTPUT_DIR / "reorder_points.csv", index=False)
    print("Saved reorder_points.csv")

#  Lead time and supplier KPIs
if purchases is not None:
    if "PODate" in purchases.columns and "ReceivingDate" in purchases.columns:
        purchases["PODate"] = pd.to_datetime(purchases["PODate"], errors="coerce")
        purchases["ReceivingDate"] = pd.to_datetime(purchases["ReceivingDate"], errors="coerce")
        purchases["lead_days"] = (purchases["ReceivingDate"] - purchases["PODate"]).dt.days
        if "VendorNumber" in purchases.columns:
            supplier_lead = purchases.groupby("VendorNumber").lead_days.agg(["mean","std","count"]).reset_index().rename(columns={"mean":"mean_lead","std":"std_lead","count":"po_count"})
            supplier_lead.to_csv(OUTPUT_DIR / "supplier_lead_times.csv", index=False)
            print("Saved supplier_lead_times.csv")

# SKU clustering for policy buckets
if sales is not None:
    sku_stats = sales.groupby("InventoryId").agg(total_qty=("SalesQuantity","sum"), days_sold=("SalesDate", lambda x: x.nunique() if "SalesDate" in sales.columns else 1)).reset_index()
    sku_stats["velocity"] = sku_stats["total_qty"] / sku_stats["days_sold"].replace(0,1)
    X = sku_stats[["total_qty","velocity"]].replace([np.inf,-np.inf],0).fillna(0)
    if len(X) >= 3:
        km = KMeans(n_clusters=3, random_state=42).fit(X)
        sku_stats["cluster"] = km.labels_
    sku_stats.to_csv(OUTPUT_DIR / "sku_policy_clusters.csv", index=False)
    print("Saved sku_policy_clusters.csv")

print("\nAll done. Check the ./output/ folder for CSVs (forecasts_top10.csv, reorder_points.csv, eoq_by_sku.csv, abc_classification.csv, top_products_by_revenue.csv, supplier_lead_times.csv, sku_policy_clusters.csv where available).")
