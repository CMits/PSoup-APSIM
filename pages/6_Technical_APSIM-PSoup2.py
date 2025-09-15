# pages/5_Technical_APSIM-PSoup_SDR_Designer.py
# ------------------------------------------------------------
# S/D Designer (Pipeline, no APSIM runtime)
# Focus: Explore mutants across Low/Med/High S/D bands.
#
# S and D after Alam et al. (2014a)-style indices (simplified):
#   S  = PTQ3_5 * LA5 * phyllochron5
#   D  = LA9 - LA5
#   S/D = S / D
#
# What this page does
# 1) Interactive single-scenario calculator (sliders) for S, D, S/D.
# 2) Combinatorial generator: define ranges for PTQ3_5, LA5, phyllochron5, LA9 → grid of scenarios.
# 3) Band scenarios into Low / Medium / High S/D via thresholds.
# 4) (Optional) Upload a mutant list → cross with scenarios to plan tests.
# 5) Visualize S/D distribution and band counts; export scenarios as CSV.
#
# Notes
# - No TTN/Yield shown; purely S/D exploration.
# - Tooltips explain each symbol; help texts appear on hover in Streamlit UI.
# ------------------------------------------------------------

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="S/D Designer — APSIM↔PSoup", page_icon="🧪", layout="wide")

# -----------------------------
# Small helpers
# -----------------------------
def compute_S(PTQ3_5: float, LA5: float, phyllochron5: float) -> float:
    # Units: PTQ3_5 ~ MJ m^-2 d^-1 (°Cd)^-1 ; LA5 ~ cm^2 ; phyllochron5 ~ °Cd
    # This is a scaled product index (follow the paper's spirit, not exact units reconciliation here)
    return PTQ3_5 * LA5 * phyllochron5

def fmt(x: float) -> str:
    # show up to 3 decimals but strip trailing zeros and any trailing dot
    return f"{x:.3f}".rstrip("0").rstrip(".")


def compute_D(LA9: float, LA5: float) -> float:
    # Growth demand proxy: difference in size between leaf 9 and leaf 5 (main culm)
    return max(LA9 - LA5, np.nan)  # if negative, downstream we’ll mark invalid

def safe_ratio(S: float, D: float) -> float | None:
    if D is None or np.isnan(D) or D <= 0:
        return np.nan
    return S / D

def band_sdr(sdr: pd.Series, t_low: float, t_high: float) -> pd.Series:
    out = pd.Series(index=sdr.index, dtype=object)
    out[sdr <= t_low] = "Low S/D"
    out[(sdr > t_low) & (sdr <= t_high)] = "Medium S/D"
    out[sdr > t_high] = "High S/D"
    out[sdr.isna()] = "Invalid"
    return out

# -----------------------------
# Sidebar — thresholds & mode
# -----------------------------
st.sidebar.title("S/D Bands")
t_low = st.sidebar.number_input("Threshold t₁ (Low→Medium)", value=2.0, help="S/D ≤ t₁ → Low")
t_high = st.sidebar.number_input("Threshold t₂ (Medium→High)", value=4.0, help="t₁ < S/D ≤ t₂ → Medium; S/D > t₂ → High")
if t_high < t_low:
    st.sidebar.warning("t₂ should be ≥ t₁. Adjust thresholds.")

st.sidebar.markdown("---")
st.sidebar.title("Units & Symbols")
st.sidebar.caption("""
**PTQ3–5**: Avg incident radiation per unit thermal time from leaf 3 full expansion to leaf 5 full expansion (MJ m⁻² d⁻¹ (°Cd)⁻¹).  
**LA5**: Leaf 5 size (cm²).  
**phyllochron5**: °Cd from leaf 4 full expansion to leaf 5 full expansion.  
**LA9**: Leaf 9 size (cm²).  
**S** = PTQ3–5 × LA5 × phyllochron5.  
**D** = LA9 − LA5.  
**S/D** = supply-to-demand ratio.
""")

# -----------------------------
# Header & context
# -----------------------------
st.title("🧪 S/D Designer (Hammer et al.2023) — Explore Mutants across S/D Bands")
st.markdown("""
This page lets you **design environmental/plant states** that yield **Low / Medium / High S/D** without running APSIM.  
You can:
- Change **PTQ3–5, LA5, phyllochron5, LA9** to compute **S**, **D**, and **S/D**.
- Generate **combinatorial scenarios** from parameter ranges.
- Band scenarios into **Low / Medium / High** via thresholds *(t₁, t₂)*.
- (Optional) Upload **mutant names** and create a **test plan** (mutant × scenario).
""")

# -----------------------------
# Section 1 — Single Scenario Calculator
# -----------------------------
st.subheader("1) Single Scenario Calculator")
col1, col2, col3, col4 = st.columns(4)

with col1:
    PTQ3_5 = st.slider("PTQ3–5", min_value=0.1, max_value=3.0, value=1.2, step=0.05,
                       help="Avg incident radiation per unit thermal time from leaf 3→5.")
with col2:
    LA5 = st.slider("LA5 (cm²)", min_value=5.0, max_value=200.0, value=60.0, step=1.0,
                    help="Size of leaf 5 (photosynthesizing surface).")
with col3:
    phyl5 = st.slider("phyllochron5 (°Cd)", min_value=20.0, max_value=300.0, value=120.0, step=5.0,
                      help="°Cd from leaf 4 full expansion to leaf 5 full expansion.")
with col4:
    LA9 = st.slider("LA9 (cm²)", min_value=10.0, max_value=400.0, value=180.0, step=1.0,
                    help="Size of leaf 9; D = LA9 − LA5.")

S_single = compute_S(PTQ3_5, LA5, phyl5)
D_single = compute_D(LA9, LA5)
SDR_single = safe_ratio(S_single, D_single)

c1, c2, c3 = st.columns(3)
c1.metric("S (supply index)", fmt(S_single))
c2.metric("S/D", fmt(SDR_single) if np.isfinite(SDR_single) else "Invalid")

c3.metric("S/D",  fmt(SDR_single) if np.isfinite(SDR_single) else "Invalid")

band_single = band_sdr(pd.Series([SDR_single]), t_low, t_high).iloc[0]
st.info(f"**Band:** {band_single}")

# -----------------------------
# Section 2 — Combinatorial Scenarios
# -----------------------------
st.subheader("2) Combinatorial Scenario Generator")
st.markdown("Define ranges for each quantity. We’ll build a **grid** and compute S, D, S/D.")

with st.expander("Define parameter ranges", expanded=True):
    cA, cB = st.columns(2)

    with cA:
        st.markdown("**PTQ3–5 range**")
        p_min = st.number_input("PTQ3–5 min", value=0.6, step=0.05)
        p_max = st.number_input("PTQ3–5 max", value=2.0, step=0.05)
        p_step = st.number_input("PTQ3–5 step", value=0.2, step=0.05)

        st.markdown("**LA5 range (cm²)**")
        la5_min = st.number_input("LA5 min", value=30.0, step=1.0)
        la5_max = st.number_input("LA5 max", value=150.0, step=1.0)
        la5_step = st.number_input("LA5 step", value=20.0, step=1.0)

    with cB:
        st.markdown("**phyllochron5 range (°Cd)**")
        ph_min = st.number_input("phyllochron5 min", value=60.0, step=5.0)
        ph_max = st.number_input("phyllochron5 max", value=220.0, step=5.0)
        ph_step = st.number_input("phyllochron5 step", value=40.0, step=5.0)

        st.markdown("**LA9 range (cm²)**")
        la9_min = st.number_input("LA9 min", value=80.0, step=1.0)
        la9_max = st.number_input("LA9 max", value=300.0, step=1.0)
        la9_step = st.number_input("LA9 step", value=40.0, step=1.0)

generate = st.button("Generate scenarios")

if generate:
    # Build grids
    def rng(minv, maxv, step):
        # inclusive-like range with floats
        if step <= 0 or maxv < minv:
            return []
        n = int(np.floor((maxv - minv) / step)) + 1
        return [minv + i * step for i in range(n)]

    P = rng(p_min, p_max, p_step)
    L5 = rng(la5_min, la5_max, la5_step)
    PH = rng(ph_min, ph_max, ph_step)
    L9 = rng(la9_min, la9_max, la9_step)

    combos = list(itertools.product(P, L5, PH, L9))
    st.caption(f"Generated **{len(combos):,}** scenarios.")

    # Compute S, D, S/D
    data = []
    for (p, l5, ph, l9) in combos:
        S = compute_S(p, l5, ph)
        D = compute_D(l9, l5)
        sdr = safe_ratio(S, D)
        data.append((p, l5, ph, l9, S, D, sdr))

    df = pd.DataFrame(data, columns=["PTQ3_5", "LA5", "phyllochron5", "LA9", "S", "D", "S_D"])
    df["Band"] = band_sdr(df["S_D"], t_low, t_high)

    # Summary counts
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Scenarios", f"{len(df):,}")
    cB.metric("Low S/D", int((df["Band"] == "Low S/D").sum()))
    cC.metric("Medium S/D", int((df["Band"] == "Medium S/D").sum()))
    cD.metric("High S/D", int((df["Band"] == "High S/D").sum()))

    # Plots — distribution & band facet (no trendline → no statsmodels)
    st.markdown("#### S/D Distribution")
    fig = px.histogram(df, x="S_D", nbins=40, color="Band", barmode="overlay",
                       hover_data=["PTQ3_5", "LA5", "phyllochron5", "LA9"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### S vs D (colored by band)")
    fig2 = px.scatter(df, x="D", y="S", color="Band",
                      hover_data=["S_D", "PTQ3_5", "LA5", "phyllochron5", "LA9"])
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Preview scenario table", expanded=False):
        st.dataframe(df.head(30), use_container_width=True)

    # -------------------------
    # Optional: Mutant plan
    # -------------------------
    st.subheader("3) (Optional) Mutant × Scenario Plan")
    st.caption("Upload a simple CSV with one column `mutant` (or `genotype`) to plan tests across the generated S/D scenarios.")
    mut_file = st.file_uploader("Upload mutant list CSV", type=["csv"], key="mutants")
    if mut_file:
        mut_df = pd.read_csv(mut_file)
        # find mutant column
        mcol = None
        for candidate in ["mutant", "genotype", "Mutant", "Genotype", "name"]:
            if candidate in mut_df.columns:
                mcol = candidate
                break
        if mcol is None:
            st.error("Couldn't find a `mutant` or `genotype` column in your CSV.")
        else:
            mut_df = mut_df[[mcol]].dropna().rename(columns={mcol: "mutant"})
            # Cross join small: if huge, consider sampling df or limiting
            mut_df["_key"] = 1
            df["_key"] = 1
            plan = pd.merge(mut_df, df, on="_key").drop(columns=["_key"])
            st.caption(f"Plan size: **{len(plan):,}** rows (mutants × scenarios).")
            st.dataframe(plan.head(30), use_container_width=True)

            # Export plan
            st.download_button("⬇️ Download Mutant × Scenario Plan (CSV)",
                               data=plan.to_csv(index=False).encode("utf-8"),
                               file_name="mutant_sdr_plan.csv",
                               mime="text/csv")

    # -------------------------
    # Export scenarios only
    # -------------------------
    st.subheader("4) Export S/D Scenarios")
    st.download_button("⬇️ Download Scenarios (CSV)",
                       data=df.to_csv(index=False).encode("utf-8"),
                       file_name="sdr_scenarios.csv",
                       mime="text/csv")


