# pages/5_APSIM_SDR_Load_and_Cluster.py
# ------------------------------------------------------------
# APSIM → S/D Load & Cluster (Steps 1 & 2 only)
#
# 1) Load APSIM factorial outputs (or synthetic demo)
#    - Required: S_D (or S and D so we can compute S_D)
#    - Optional: EnvID, Location, Year, Rad, Temp, VPD, Yield, TTN
# 2) Cluster environments into Low / Medium / High S/D
#    - Methods: Quantiles (33/66) or User thresholds (t1, t2)
#
# Notes:
# - No SUC mapping, no TTN/Yield analysis, no trendlines -> no statsmodels.
# - Safe hover_data checks so it won't crash if cols are missing.
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="APSIM → S/D Load & Cluster",
    page_icon="🧩",
    layout="wide"
)

# -----------------------------
# Helpers
# -----------------------------
def load_user_or_example() -> pd.DataFrame:
    """
    Load user CSV or create a small synthetic dataset that mimics APSIM outputs.
    If S_D isn't present but S and D are, compute S_D = S / D.
    """
    up = st.file_uploader("Upload APSIM factorial output CSV", type=["csv"])
    use_example = st.checkbox("Use synthetic example", value=up is None)

    if up and not use_example:
        df = pd.read_csv(up)
    else:
        rng = np.random.default_rng(42)
        n = 180
        years = rng.choice(np.arange(2001, 2007), size=n)
        locs = rng.choice(["Emerald", "Gatton", "Dalby"], size=n, p=[0.45, 0.35, 0.20])

        # Simple synthetic env drivers
        rad = rng.normal(18, 3.0, n).clip(10, 26)
        temp = rng.normal(24, 3.5, n).clip(12, 35)
        vpd = rng.normal(1.6, 0.4, n).clip(0.5, 3.0)

        # S, D loosely linked to env (illustrative)
        S = 0.12 * rad + 0.03 * (30 - np.abs(temp - 25)) + rng.normal(0, 0.2, n) + 2.0
        D = 0.10 * vpd + rng.normal(0, 0.05, n) + 0.8
        S = np.clip(S, 0.5, None)
        D = np.clip(D, 0.05, None)
        sdr = S / D

        df = pd.DataFrame({
            "EnvID": [f"E{i:03d}" for i in range(n)],
            "Location": locs,
            "Year": years,
            "Rad": rad.round(2),
            "Temp": temp.round(2),
            "VPD": vpd.round(2),
            "S": S.round(3),
            "D": D.round(3),
            "S_D": sdr.round(4),
            # Keeping Yield/TTN optional; we won't use them in plots here
            "Yield": (2.5 + 0.6 * sdr + rng.normal(0, 0.4, n)).round(3),
            "TTN": (0.8 + 0.35 * sdr + rng.normal(0, 0.25, n)).round(3),
        })

    # Compute S_D if needed
    if "S_D" not in df.columns:
        if {"S", "D"}.issubset(df.columns):
            df["S_D"] = (df["S"] / df["D"]).replace([np.inf, -np.inf], np.nan)
        else:
            st.error("No S_D column found and cannot compute from S and D. Provide S_D or both S and D.")
            st.stop()

    # Ensure EnvID exists
    if "EnvID" not in df.columns:
        df["EnvID"] = np.arange(1, len(df) + 1)

    # Nice column order
    preferred = ["EnvID", "Location", "Year", "Rad", "Temp", "VPD", "S", "D", "S_D", "Yield", "TTN"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols].copy()

def label_sdr_bins(values: np.ndarray, mode: str, params: dict) -> np.ndarray:
    """
    Low / Medium / High S/D bands.
    - Quantiles (33/66)
    - User thresholds (t1, t2)
    """
    v = np.asarray(values, dtype=float)
    labels = np.empty(v.shape, dtype=object)

    if mode == "Quantiles (33/66%)":
        q1, q2 = np.nanquantile(v, [params.get("q1", 0.33), params.get("q2", 0.66)])
        labels[v <= q1] = "Low S/D"
        labels[(v > q1) & (v <= q2)] = "Medium S/D"
        labels[v > q2] = "High S/D"

    elif mode == "User thresholds":
        t1, t2 = params.get("t1"), params.get("t2")
        if t1 is None or t2 is None:
            raise ValueError("Provide both t1 and t2.")
        low, high = min(t1, t2), max(t1, t2)
        labels[v <= low] = "Low S/D"
        labels[(v > low) & (v <= high)] = "Medium S/D"
        labels[v > high] = "High S/D"

    else:
        q1, q2 = np.nanquantile(v, [0.33, 0.66])
        labels[v <= q1] = "Low S/D"
        labels[(v > q1) & (v <= q2)] = "Medium S/D"
        labels[v > q2] = "High S/D"

    return labels

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("S/D Clustering")
band_mode = st.sidebar.radio("Banding method", ["Quantiles (33/66%)", "User thresholds"], index=0)
params = {}
if band_mode == "User thresholds":
    c1, c2 = st.sidebar.columns(2)
    with c1:
        params["t1"] = st.number_input("t1 (Low→Med)", value=2.0)
    with c2:
        params["t2"] = st.number_input("t2 (Med→High)", value=4.0)

st.sidebar.markdown("---")
download_name = st.sidebar.text_input("Export filename", value="apsim_sdr_bands.csv")

# -----------------------------
# Main
# -----------------------------
st.title("🧩 APSIM → S/D: Load & Cluster")

st.markdown("""
**What this page does**  
1) Loads your **APSIM factorial outputs** (or a demo set) and ensures **S_D** exists.  
2) Clusters environments into **Low / Medium / High S/D** (quantiles or thresholds).  
""")

# 1) Load
df = load_user_or_example()

with st.expander("Preview data (first 12 rows)", expanded=False):
    st.dataframe(df.head(12), use_container_width=True)

# 2) Cluster by S/D
try:
    df["S_D_band"] = label_sdr_bins(df["S_D"].values, band_mode, params)
except ValueError as e:
    st.error(str(e))
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Environments", f"{df['EnvID'].nunique():,}")
c2.metric("Locations", f"{df['Location'].nunique() if 'Location' in df.columns else 0}")
c3.metric("Years", f"{df['Year'].nunique() if 'Year' in df.columns else 0}")
c4.metric("S/D range", f"{df['S_D'].min():.2f} – {df['S_D'].max():.2f}")

# Plots (no trendlines → no statsmodels)
st.subheader("S/D Bands — Quick Looks")

available_cols = set(df.columns)

# Histogram of S/D colored by band
fig_hist = px.histogram(
    df, x="S_D", nbins=40, color="S_D_band",
    barmode="overlay",
    hover_data=[c for c in ["EnvID", "Location", "Year"] if c in available_cols],
    title="S/D distribution by band"
)
st.plotly_chart(fig_hist, use_container_width=True)

# Env covariate distributions by band (if present)
env_feats = [c for c in ["Rad", "Temp", "VPD"] if c in available_cols]
if env_feats:
    st.markdown("**Environmental covariates by band**")
    for feat in env_feats:
        fig_vio = px.violin(
            df, y=feat, x="S_D_band",
            box=True, points="all",
            title=f"{feat} distribution by S/D band"
        )
        st.plotly_chart(fig_vio, use_container_width=True)

# Counts per band
band_counts = (
    df["S_D_band"]
    .value_counts()
    .reindex(["Low S/D", "Medium S/D", "High S/D"])
    .fillna(0)
    .astype(int)
)

st.markdown("**Counts per band**")

# Works across pandas versions
band_counts_df = band_counts.reset_index()
band_counts_df.columns = ["Band", "count"]

st.dataframe(band_counts_df, use_container_width=True)


# Export labeled table (optional but handy)
labeled = df.copy()
csv = labeled.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download labeled dataset (CSV)",
    data=csv,
    file_name=download_name,
    mime="text/csv"
)

st.caption("""
This file includes your original columns plus `S_D_band`.  
Use it downstream for experimental design or mapping steps.
""")
