# streamlit run data.py
import io
import os
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Global Markets, GDP & Conglomerate Labor — 1955–2025", layout="wide")

# ----------------------------
# Constants
# ----------------------------
TODAY_YEAR = 2025
MIN_YEAR = 1955
MAX_YEAR = 2025
WORLD_BANK_BASE = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "GDP (USD)": "NY.GDP.MKTP.CD",           # GDP (current US$)
    "Market Cap (USD)": "CM.MKT.LCAP.CD",    # Market cap of listed domestic companies (current US$)
}

# Local default CSVs (optional, auto-used if present)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_DEFAULT = os.path.join(DATA_DIR, "global_econ_dataset.csv")
INDUSTRY_DEFAULT = os.path.join(DATA_DIR, "industry_market_cap_benchmark.csv")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_wb_indicator(indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch a World Bank indicator for all countries and years (JSON, paginated)."""
    url = f"{WORLD_BANK_BASE}/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": 20000, "date": f"{start_year}:{end_year}"}
    rows_all = []
    page = 1
    while True:
        params["page"] = page
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            break
        meta, rows = data[0], data[1]
        for it in rows:
            iso3 = it.get("countryiso3code")
            if not iso3:  # skip aggregates like "World"
                continue
            rows_all.append({
                "iso3": iso3,
                "country": (it.get("country") or {}).get("value"),
                "year": int(it.get("date")) if it.get("date") else None,
                "value": it.get("value"),
            })
        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(0.1)
    return pd.DataFrame(rows_all).dropna(subset=["year"])

def load_user_csv(uploaded_file: Optional[io.BytesIO], expected_cols: list, label: str) -> Optional[pd.DataFrame]:
    if not uploaded_file:
        return None
    df = pd.read_csv(uploaded_file)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"{label} is missing columns: {missing}. Expected: {expected_cols}")
        return None
    return df

def try_load_local(path: str, expected_cols: list) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.warning(f"Failed reading {os.path.basename(path)}: {e}")
        return None
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"{os.path.basename(path)} missing columns: {missing}. Expected: {expected_cols}")
        return None
    return df

def minmax_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - lo) / (hi - lo)

def restrict_years(df: pd.DataFrame, step: int) -> pd.DataFrame:
    allowed = list(range(MIN_YEAR, MAX_YEAR + 1, step))
    return df[df["year"].isin(allowed)].copy()

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("Controls")

interval = st.sidebar.selectbox("Year interval", options=[5, 10], index=0, help="Use 5-year or 10-year steps only.")
year_window = st.sidebar.slider(
    "Year range",
    min_value=MIN_YEAR, max_value=MAX_YEAR,
    value=(MIN_YEAR, MAX_YEAR), step=interval,
)

variable = st.sidebar.selectbox(
    "Select variable to visualize",
    ["GDP (USD)", "Market Cap (USD)", "Conglom. Employment Share (%)", "Combined Metric"],
    index=0,
)

with st.sidebar.expander("Combined Metric Weights", expanded=False):
    w_gdp  = st.slider("Weight: GDP", 0.0, 1.0, 1.0, 0.05)
    w_mcap = st.slider("Weight: Market Cap", 0.0, 1.0, 1.0, 0.05)
    w_cong = st.slider("Weight: Conglom. Employment Share", 0.0, 1.0, 1.0, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Defaults & Optional Uploads")
st.sidebar.caption("If present, the app auto-loads these local defaults:")
st.sidebar.code(f"{GLOBAL_DEFAULT}", language="text")
st.sidebar.code(f"{INDUSTRY_DEFAULT}", language="text")

industry_csv = st.sidebar.file_uploader(
    "Upload: industry_market_cap.csv (optional)",
    type=["csv"],
    help="Columns: iso3,country,year,industry,market_cap_usd",
)
conglom_csv = st.sidebar.file_uploader(
    "Upload: conglomerate_employment.csv (optional)",
    type=["csv"],
    help="Columns: iso3,country,year,conglomerate_share_percent",
)

# ----------------------------
# Data Assembly (Defaults → Uploads → World Bank fallback)
# ----------------------------
st.title("🌍 Global Market Cap, GDP & Conglomerate Labor (1955–2025)")
st.caption("Uses local defaults when available; otherwise fetches GDP & Market Cap from the World Bank. Uploads can override/augment.")

# 1) Try local defaults for global metrics
global_default = try_load_local(
    GLOBAL_DEFAULT,
    ["iso3","country","year","gdp_usd","market_cap_usd","conglomerate_share_percent"]
)

if global_default is not None:
    df = global_default.copy()
else:
    # 2) Fallback to World Bank for GDP & Market Cap; conglomerate share remains NaN
    with st.status("Fetching GDP & Market Cap from World Bank…", expanded=False) as status:
        gdp_df  = fetch_wb_indicator(WB_INDICATORS["GDP (USD)"], MIN_YEAR, MAX_YEAR).rename(columns={"value":"gdp_usd"})
        mcap_df = fetch_wb_indicator(WB_INDICATORS["Market Cap (USD)"], MIN_YEAR, MAX_YEAR).rename(columns={"value":"market_cap_usd"})
        status.update(label="World Bank fetch complete.", state="complete")

    # Build base grid of country-year present in either GDP or Market Cap
    countries = set((*zip(gdp_df.iso3, gdp_df.country), *zip(mcap_df.iso3, mcap_df.country)))
    allowed_years = list(range(MIN_YEAR, MAX_YEAR + 1))
    base = pd.DataFrame([(iso, c, y) for (iso, c) in countries for y in allowed_years],
                        columns=["iso3","country","year"])
    df = base.merge(gdp_df[["iso3","country","year","gdp_usd"]],  on=["iso3","country","year"], how="left") \
             .merge(mcap_df[["iso3","country","year","market_cap_usd"]], on=["iso3","country","year"], how="left")
    df["conglomerate_share_percent"] = np.nan

# 3) Apply local industry default to replace total market cap (sum of industries) if present
industry_default = try_load_local(
    INDUSTRY_DEFAULT,
    ["iso3","country","year","industry","market_cap_usd"]
)
if industry_default is not None:
    agg_ind = industry_default.groupby(["iso3","country","year"], as_index=False)["market_cap_usd"].sum()
    df = df.drop(columns=["market_cap_usd"], errors="ignore").merge(
        agg_ind, on=["iso3","country","year"], how="left"
    )

# 4) Apply uploads (override/augment)
uploaded_industry = load_user_csv(industry_csv, ["iso3","country","year","industry","market_cap_usd"], "Industry upload")
if uploaded_industry is not None:
    agg_up = uploaded_industry.groupby(["iso3","country","year"], as_index=False)["market_cap_usd"].sum()
    df = df.drop(columns=["market_cap_usd"], errors="ignore").merge(
        agg_up, on=["iso3","country","year"], how="left"
    )

uploaded_cong = load_user_csv(conglom_csv, ["iso3","country","year","conglomerate_share_percent"], "Conglomerate upload")
if uploaded_cong is not None:
    df = df.merge(uploaded_cong, on=["iso3","country","year"], how="left", suffixes=("", "_upl"))
    if "conglomerate_share_percent_upl" in df.columns:
        df["conglomerate_share_percent"] = df["conglomerate_share_percent_upl"].where(
            df["conglomerate_share_percent_upl"].notna(), df["conglomerate_share_percent"]
        )
        df.drop(columns=["conglomerate_share_percent_upl"], inplace=True)

# Enforce numeric year & restrict to interval/window
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)
df = restrict_years(df, interval)
df = df[(df["year"] >= year_window[0]) & (df["year"] <= year_window[1])].copy()

# ----------------------------
# Combined Metric (per-year min–max scaled, weighted)
# ----------------------------
def compute_combined(group: pd.DataFrame) -> pd.Series:
    parts, weights = [], []
    if "gdp_usd" in group:
        parts.append(minmax_normalize(group["gdp_usd"])); weights.append(w_gdp)
    if "market_cap_usd" in group:
        parts.append(minmax_normalize(group["market_cap_usd"])); weights.append(w_mcap)
    if "conglomerate_share_percent" in group:
        parts.append(minmax_normalize(group["conglomerate_share_percent"])); weights.append(w_cong)
    if not parts:
        return pd.Series(np.nan, index=group.index)
    wsum = sum(weights) if sum(weights) > 0 else len(parts)
    return (sum(parts[i] * weights[i] for i in range(len(parts))) / wsum) if sum(weights) > 0 else (sum(parts) / len(parts))

df = df.sort_values(["year","iso3"]).reset_index(drop=True)
df["combined_metric"] = df.groupby("year", include_groups=False).apply(compute_combined)
# ----------------------------
# World Choropleth
# ----------------------------
st.markdown("## Interactive World Heatmap")

var_map = {
    "GDP (USD)": "gdp_usd",
    "Market Cap (USD)": "market_cap_usd",
    "Conglom. Employment Share (%)": "conglomerate_share_percent",
    "Combined Metric": "combined_metric",
}
vcol = var_map[variable]

sel_year = st.slider("Select year", int(df["year"].min()), int(df["year"].max()),
                     value=int(df["year"].max()), step=interval)

sdf = df[df["year"] == sel_year]
fig = px.choropleth(
    sdf,
    locations="iso3",
    color=vcol,
    hover_name="country",
    color_continuous_scale="Viridis",
    projection="natural earth",
    title=f"{variable} — {sel_year}",
)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, width="stretch")

# ----------------------------
# US & Canada Trends
# ----------------------------
st.markdown("## 🇺🇸🇨🇦 US & Canada — Trends & Comparison")
na = df[df["iso3"].isin(["USA","CAN"])].melt(
    id_vars=["iso3","country","year"],
    value_vars=["gdp_usd","market_cap_usd","conglomerate_share_percent","combined_metric"],
    var_name="metric", value_name="value"
)
fig_na = px.line(
    na,
    x="year", y="value", color="country", facet_row="metric",
    title="United States vs Canada — GDP, Market Cap, Conglomerate Share, Combined",
)
fig_na.update_layout(height=900, margin=dict(l=0,r=0,t=60,b=0))
st.plotly_chart(fig_na, use_container_width=True)

# ----------------------------
# Download Current Window
# ----------------------------
st.markdown("### Download current window data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False).encode("utf-8"),
    file_name="data_export.csv",
    mime="text/csv",
)

st.caption(
    "Defaults: local CSVs if available. Otherwise GDP & Market Cap are fetched from the World Bank. "
    "Industry-level market cap and conglomerate employment share can be supplied via default files or uploads. "
    "Combined metric = per-year min–max normalization with adjustable weights. Years restricted to 5- or 10-year steps."
)
