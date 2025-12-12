"""
Slooze Streamlit Dashboard
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Slooze Inventory Dashboard", layout="wide")

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

# Helper to load a file if exists
def load_if_exists(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Failed to load {path.name}: {e}")
    return None

# Your filenames
sales = load_if_exists(DATA_DIR / "SalesFINAL12312016.csv")
purchases = load_if_exists(DATA_DIR / "PurchasesFINAL12312016.csv")
beg_inv = load_if_exists(DATA_DIR / "BegInvFINAL12312016.csv")
end_inv = load_if_exists(DATA_DIR / "EndInvFINAL12312016.csv")
purchase_prices = load_if_exists(DATA_DIR / "2017PurchasePricesDec.csv")
invoice_purchases = load_if_exists(DATA_DIR / "InvoicePurchases12312016.csv")

# Outputs from notebook
reorder_points = load_if_exists(OUTPUT_DIR / "reorder_points.csv")
eoq = load_if_exists(OUTPUT_DIR / "eoq_by_sku.csv")
forecasts = load_if_exists(OUTPUT_DIR / "forecasts_top10.csv")
abc = load_if_exists(OUTPUT_DIR / "abc_classification.csv")
supplier_lead = load_if_exists(OUTPUT_DIR / "supplier_lead_times.csv")
sku_clusters = load_if_exists(OUTPUT_DIR / "sku_policy_clusters.csv")
top_products = load_if_exists(OUTPUT_DIR / "top_products_by_revenue.csv")

# Normalize small things
def ensure_date(df, cols):
    if df is None: return df
    for c in cols:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

sales = ensure_date(sales, ["SalesDate"])
purchases = ensure_date(purchases, ["PODate", "ReceivingDate", "InvoiceDate"])
if end_inv is not None:
    inventory = end_inv
elif beg_inv is not None:
    inventory = beg_inv
else:
    inventory = load_if_exists(DATA_DIR / "inventory.csv")
inventory = ensure_date(inventory, ["endDate", "startDate"])

st.title("Slooze — Inventory & Supplier KPI Dashboard")
st.write("This dashboard uses your specific Slooze CSV files (SalesFINAL..., PurchasesFINAL..., BegInv..., EndInv...).")

st.sidebar.header("Filters & Options")
use_outputs = st.sidebar.checkbox("Use precomputed outputs (./output/)", value=True)

for name, df in [("sales", sales), ("purchases", purchases), ("inventory", inventory)]:
    st.sidebar.write(f"{name}: {'Loaded' if df is not None else 'Missing'}")

# detect product list
product_list = None
if sales is not None and "InventoryId" in sales.columns:
    product_list = sales["InventoryId"].astype(str).unique().tolist()
elif inventory is not None and "InventoryId" in inventory.columns:
    product_list = inventory["InventoryId"].astype(str).unique().tolist()

selected_product = st.sidebar.selectbox("Select InventoryId (optional)", options=[None] + (product_list or []))

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Low-stock alerts")
    if inventory is None:
        st.info("No inventory snapshot found. Place EndInvFINAL12312016.csv or BegInv... in ./data/")
    else:
        inv = inventory.copy()
        # normalize on_hand
        if "onHand" in inv.columns and "on_hand" not in inv.columns:
            inv = inv.rename(columns={"onHand":"on_hand"})
        if "on_hand" not in inv.columns:
            # pick numeric column if available
            nums = inv.select_dtypes(include="number").columns.tolist()
            if nums:
                inv["on_hand"] = inv[nums[0]]
        inv["on_hand"] = pd.to_numeric(inv.get("on_hand", 0), errors="coerce").fillna(0)
        if "InventoryId" in inv.columns:
            inv["InventoryId"] = inv["InventoryId"].astype(str)
        # latest snapshot
        if "endDate" in inv.columns:
            inv["endDate"] = pd.to_datetime(inv["endDate"], errors="coerce")
            latest = inv.sort_values("endDate").groupby("InventoryId").tail(1)
        elif "startDate" in inv.columns:
            latest = inv.sort_values("startDate").groupby("InventoryId").tail(1)
        else:
            latest = inv.groupby("InventoryId").tail(1)
        rp = reorder_points if (use_outputs and reorder_points is not None) else None
        merged = latest.copy()
        if rp is not None and "InventoryId" in rp.columns:
            merged = merged.merge(rp[["InventoryId","reorder_point","safety_stock"]], on="InventoryId", how="left")
        merged["reorder_point"] = merged.get("reorder_point", 0).fillna(0)
        merged["safety_stock"] = merged.get("safety_stock", 0).fillna(0)
        merged["below_rop"] = merged["on_hand"] <= merged["reorder_point"]
        if selected_product:
            merged = merged[merged["InventoryId"].astype(str) == str(selected_product)]
        alerts = merged[merged["below_rop"]].sort_values("on_hand")
        st.write(f"{alerts.shape[0]} SKUs below reorder point")
        if not alerts.empty:
            st.dataframe(alerts[["InventoryId","on_hand","reorder_point","safety_stock"]].rename(columns={"InventoryId":"SKU"}).head(200))
            with st.expander("Export alerts"):
                st.download_button("Download alerts CSV", alerts.to_csv(index=False).encode("utf-8"), file_name="low_stock_alerts.csv")
        else:
            st.info("No SKUs below reorder point or reorder_points missing.")

    st.markdown("---")
    st.subheader("Reorder Points & EOQ lookup")
    if (reorder_points is None or reorder_points.empty) and (eoq is None or eoq.empty):
        st.info("No precomputed reorder points or EOQ in ./output/. Run notebook to generate them.")
    else:
        rp_df = reorder_points if reorder_points is not None else pd.DataFrame()
        eoq_df = eoq if eoq is not None else pd.DataFrame()
        combined = rp_df.merge(eoq_df[["InventoryId","EOQ"]], on="InventoryId", how="left") if (not rp_df.empty and not eoq_df.empty) else (rp_df if not rp_df.empty else eoq_df)
        if selected_product:
            combined = combined[combined["InventoryId"].astype(str) == str(selected_product)]
        if combined.empty:
            st.info("No reorder/EOQ data to display.")
        else:
            st.dataframe(combined.head(200))
            with st.expander("Export reorder/EOQ"):
                st.download_button("Download CSV", combined.to_csv(index=False).encode("utf-8"), file_name="reorder_eoq_lookup.csv")

with col2:
    st.subheader("Supplier KPIs")
    if purchases is None:
        st.info("No purchases data found. Put PurchasesFINAL12312016.csv in ./data/")
    else:
        p = purchases.copy()
        # ensure date fields
        if "PODate" in p.columns and "ReceivingDate" in p.columns:
            p["PODate"] = pd.to_datetime(p["PODate"], errors="coerce")
            p["ReceivingDate"] = pd.to_datetime(p["ReceivingDate"], errors="coerce")
            p["lead_days"] = (p["ReceivingDate"] - p["PODate"]).dt.days
            if "VendorNumber" in p.columns:
                lead_summary = p.groupby("VendorNumber").lead_days.agg(["mean","median","std","count"]).reset_index().rename(columns={"mean":"mean_lead","median":"median_lead","std":"std_lead","count":"po_count"})
                st.dataframe(lead_summary)
                slow = lead_summary.sort_values("mean_lead", ascending=False).head(10)
                fig = px.bar(slow, x="VendorNumber", y="mean_lead", title="Top slow suppliers (mean lead days)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Purchases file missing PODate and ReceivingDate columns; cannot compute lead times automatically.")

st.markdown("---")
st.subheader("Forecast viewer")
if forecasts is None or forecasts.empty:
    st.info("No forecasts found in ./output/. Run notebook first.")
else:
    sku_options = forecasts["InventoryId"].astype(str).unique().tolist()
    sku_choice = st.selectbox("Choose SKU to plot forecast", options=sku_options)
    fdf = forecasts[forecasts["InventoryId"].astype(str) == str(sku_choice)].sort_values("ds")
    hist = None
    if sales is not None and "InventoryId" in sales.columns and "SalesDate" in sales.columns:
        hist = sales[sales["InventoryId"].astype(str) == str(sku_choice)].set_index("SalesDate").resample("W")["SalesQuantity"].sum().reset_index()
    fig = go.Figure()
    if hist is not None and not hist.empty:
        fig.add_trace(go.Scatter(x=hist["SalesDate"], y=hist["SalesQuantity"], name="history"))
    fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat"], name="forecast"))
    if "yhat_lower" in fdf.columns and "yhat_upper" in fdf.columns:
        fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat_upper"], name="upper", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fdf["ds"], y=fdf["yhat_lower"], name="lower", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,100,80,0.2)", showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Additional insights")
c1, c2, c3 = st.columns(3)
with c1:
    st.write("Top SKUs by revenue")
    if top_products is not None:
        fig = px.bar(top_products.head(10), x="InventoryId", y="revenue")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("top_products_by_revenue.csv missing; run notebook.")

with c2:
    st.write("Days-of-cover distribution")
    if reorder_points is not None and inventory is not None:
        latest = inventory.sort_values("endDate").groupby("InventoryId").tail(1) if "endDate" in inventory.columns else inventory.groupby("InventoryId").tail(1)
        merged = latest.merge(reorder_points[["InventoryId","avg_daily"]], on="InventoryId", how="left")
        merged["days_of_cover"] = merged["on_hand"] / merged["avg_daily"].replace(0, np.nan)
        merged["days_of_cover"] = merged["days_of_cover"].fillna(0)
        fig = px.histogram(merged, x="days_of_cover", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need reorder_points.csv and inventory snapshot.")

with c3:
    st.write("Reorder point distribution")
    if reorder_points is not None:
        fig = px.box(reorder_points, y="reorder_point")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("reorder_points.csv missing.")

st.markdown("---")
st.caption("You can upload alternate CSVs (used for this session) through the file uploader below.")
uploaded = st.file_uploader("Upload CSVs", accept_multiple_files=True)
if uploaded:
    for f in uploaded:
        try:
            df = pd.read_csv(f)
            st.success(f"Loaded {f.name} ({df.shape[0]} rows)")
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")

st.write("Notes: Run the notebook first to generate outputs in ./output/. If you'd like, I can add an option to compute reorder points on-the-fly inside the app.")
