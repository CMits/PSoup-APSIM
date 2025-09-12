# pages/1_SDR_to_SUC.py

import numpy as np
import pandas as pd
import streamlit as st

# ------------- CONFIG -------------
st.set_page_config(
    page_title="APSIM → PSoup: SDR → SUC Mapper",
    page_icon="🌾",
    layout="centered"
)

# ------------- HELPERS -------------
def map_linear(x, L=0.0, U=50.0, out_min=0.0, out_max=2.0):
    """Affine map with clipping: [L,U] → [out_min, out_max]."""
    if U <= L:
        return np.nan
    s = (x - L) / (U - L) * (out_max - out_min) + out_min
    return float(np.clip(s, out_min, out_max))

def map_tanh(x, center=25.0, halfspan=25.0, k=1.0, out_min=0.0, out_max=2.0):
    """
    Smooth, bounded map using tanh centered at 'center'.
    k controls steepness (larger k = gentler curve).
    """
    z = (x - center) / (halfspan * max(k, 1e-9))
    s = 1.0 + np.tanh(z)  # in (0,2)
    return float(np.clip(s, out_min, out_max))

def _extract_year(df: pd.DataFrame) -> pd.Series:
    """Year from 'Date' (dayfirst) with fallback to 'Years'."""
    date = pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce")
    years = pd.to_datetime(df.get("Years"), errors="coerce")
    out = date.dt.year
    if "Years" in df.columns:
        out = out.fillna(years.dt.year)
    return out

# ------------- SIDEBAR -------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio(
    "Mapping mode", 
    ["Linear (0–50 → 0–2)", "Smooth (tanh)"],
    help="Linear is a simple rescale; Smooth uses a gentle S-curve to reduce volatility around the center."
)

st.sidebar.markdown("**Input & Output Ranges**")
L = st.sidebar.number_input("SDR lower bound (L)", value=0.0, step=1.0)
U = st.sidebar.number_input("SDR upper bound (U)", value=50.0, step=1.0)
out_min = st.sidebar.number_input("SUC min", value=0.0, step=0.1)
out_max = st.sidebar.number_input("SUC max", value=2.0, step=0.1)

# defaults so we can compute both Linear & Tanh in batch even if Linear mode is selected
_center_default = 25.0
_k_default = 1.0

if mode.startswith("Smooth"):
    st.sidebar.markdown("**S-curve parameters**")
    center = st.sidebar.number_input("Center SDR (WT anchor)", value=25.0, step=0.5,
                                     help="Where SUC=~1 in the smooth curve.")
    k = st.sidebar.slider("Smoothness (k)", min_value=0.25, max_value=3.0, value=1.0, step=0.05,
                          help="Smaller k = steeper near the center; larger k = gentler.")
else:
    center = _center_default
    k = _k_default

# ------------- HEADER -------------
st.title("🌾 APSIM → PSoup Mapper")
st.caption("Enter APSIM **SupplyDemandRatio (SDR)**, get PSoup **SUC** (0–2).")

# ------------- INPUT -------------
col1, col2 = st.columns([1, 1])
with col1:
    x = st.number_input("APSIM SupplyDemandRatio (SDR)", value=30.0, step=0.5,
                        help="Example: 30 → SUC 1.2 in linear mode.")
with col2:
    st.write("")
    st.write("")

# ------------- COMPUTE -------------
if mode.startswith("Linear"):
    suc = map_linear(x, L=L, U=U, out_min=out_min, out_max=out_max)
else:
    suc = map_tanh(x, center=center, halfspan=(U - L) / 2.0 if U > L else 25.0,
                   k=k, out_min=out_min, out_max=out_max)

# ------------- OUTPUT CARD -------------
st.markdown("---")
st.subheader("Result")

rescol1, rescol2 = st.columns([1, 1])
with rescol1:
    st.metric(label="➡️ PSoup SUC", value=f"{suc:0.3f}")
with rescol2:
    st.metric(label="Input SDR", value=f"{x:0.3f}")

# ------------- MATH DISPLAY -------------
st.markdown("### Mapping formulas")

if mode.startswith("Linear"):
    st.latex(r"""
    \textbf{Linear:}\quad s(x)=
    \min\!\left(%0.3f,\;\max\!\left(%0.3f,\;\frac{x - %0.3f}{%0.3f - %0.3f}\times(%0.3f-%0.3f)+%0.3f\right)\right)
    """ % (out_max, out_min, L, U, L, out_max, out_min, out_min))
    st.latex(r"""
    \text{With } L=%0.3f,\ U=%0.3f,\; s(%0.3f)=%0.3f
    """ % (L, U, x, suc))
else:
    halfspan = (U - L) / 2.0 if U > L else 25.0
    st.latex(r"""
    \textbf{Smooth (tanh):}\quad
    s(x)=\operatorname{clip}\!\Bigl(1+\tanh\!\bigl(\tfrac{x-%0.3f}{%0.3f\cdot %0.3f}\bigr),\ %0.3f,\ %0.3f\Bigr)
    """ % (center, halfspan, k, out_min, out_max))
    st.latex(r"""
    \text{With } \text{center}=%0.3f,\ \text{halfspan}=%0.3f,\ k=%0.3f,\; s(%0.3f)=%0.3f
    """ % (center, halfspan, k, x, suc))

# ------------- PLOT -------------
st.markdown("### Visual mapping")
xs = np.linspace(L, U, 201) if U > L else np.linspace(0, 50, 201)
if mode.startswith("Linear"):
    ys = [map_linear(val, L=L, U=U, out_min=out_min, out_max=out_max) for val in xs]
else:
    ys = [map_tanh(val, center=center, halfspan=(U - L) / 2.0 if U > L else 25.0,
                   k=k, out_min=out_min, out_max=out_max) for val in xs]

df_curve = pd.DataFrame({"SDR": xs, "SUC": ys})
st.line_chart(df_curve, x="SDR", y="SUC", height=300)

st.markdown(
    f"""
    **Pipeline concept:**  
    **APSIM** gives **SDR = {x:.2f}** → **Mapping** → **PSoup** gets **SUC = {suc:.3f}**.
    """
)

# ------------- EXCEL BATCH (Linear + Tanh, GLOBAL median center) -------------
st.markdown("---")
st.markdown("## Excel batch: SDR → SUC (Linear & Tanh, WT anchored at **global** median SDR)")

upl = st.file_uploader(
    "Upload an .xlsx with columns: **SupplyDemandRatio** and **Date** and/or **Years**.",
    type=["xlsx"]
)

if upl is not None:
    import altair as alt

    try:
        xl = pd.ExcelFile(upl)
        sheet = st.selectbox("Sheet", xl.sheet_names, index=0)
        df_raw = xl.parse(sheet_name=sheet).copy()

        st.write("**Preview:**")
        st.dataframe(df_raw.head(15), use_container_width=True)

        # Prepare
        df_raw["Year"] = _extract_year(df_raw)
        df_raw["SupplyDemandRatio"] = pd.to_numeric(df_raw["SupplyDemandRatio"], errors="coerce")

        # ---- GLOBAL center (WT anchor): median SDR across ALL rows ----
        global_median_sdr = float(df_raw["SupplyDemandRatio"].median(skipna=True))
        halfspan = (U - L) / 2.0 if U > L else 25.0  # range scale for tanh
        # keep 'k' from the sidebar; if you want a fixed value for batch, set here.

        # Compute BOTH mappings using current sidebar bounds and k
        df_raw["SUC_linear"] = df_raw["SupplyDemandRatio"].apply(
            lambda v: map_linear(v, L=L, U=U, out_min=out_min, out_max=out_max)
        )
        df_raw["SUC_tanh"] = df_raw["SupplyDemandRatio"].apply(
            lambda v: map_tanh(v,
                               center=global_median_sdr,
                               halfspan=halfspan,
                               k=k,
                               out_min=out_min, out_max=out_max)
        )

        # ---- Calculations table (document what we used) ----
        calc_tbl = pd.DataFrame({
            "Parameter": [
                "SDR lower bound (L)", "SDR upper bound (U)",
                "SUC min", "SUC max",
                "Global median SDR (WT center)",
                "Halfspan used for tanh",
                "Smoothness k (tanh)",
                "Rows processed"
            ],
            "Value": [L, U, out_min, out_max, global_median_sdr, halfspan, k, len(df_raw)]
        })
        st.markdown("### Calculations used (batch)")
        st.dataframe(calc_tbl, use_container_width=True)

        # ---- Per-row results ----
        st.markdown("### Per-row results")
        view = st.radio("Select columns to display", ["Linear", "Smooth (tanh)", "Both"], horizontal=True)
        cols = ["Year", "SupplyDemandRatio"]
        if view == "Linear":
            cols += ["SUC_linear"]
        elif view == "Smooth (tanh)":
            cols += ["SUC_tanh"]
        else:
            cols += ["SUC_linear", "SUC_tanh"]

        st.dataframe(
            df_raw[cols].sort_values(["Year", "SupplyDemandRatio"]),
            use_container_width=True, height=320
        )

        # ---- Per-year summary ----
        st.markdown("### Per-year summary")
        per_year = (
            df_raw.groupby("Year", dropna=False)
                  .agg(Year_Median_SDR=("SupplyDemandRatio", "median"),
                       Year_Mean_SUC_linear=("SUC_linear", "mean"),
                       Year_Mean_SUC_tanh=("SUC_tanh", "mean"),
                       N=("SupplyDemandRatio", "size"))
                  .reset_index()
                  .sort_values("Year", kind="stable")
        )
        st.dataframe(per_year, use_container_width=True)

        # ---- Colored comparison charts (side-by-side) ----
        plot_df = per_year.dropna(subset=["Year"]).copy()
        if not plot_df.empty:
            plot_df["Year"] = plot_df["Year"].astype(int).astype(str)

            # Melt to long format for grouped bars
            long_df = plot_df.melt(
                id_vars=["Year"],
                value_vars=["Year_Mean_SUC_linear", "Year_Mean_SUC_tanh"],
                var_name="Mapping", value_name="Mean_SUC"
            )
            long_df["Mapping"] = long_df["Mapping"].map({
                "Year_Mean_SUC_linear": "Linear",
                "Year_Mean_SUC_tanh": "Tanh"
            })

            st.markdown("#### Mean SUC by Year (Linear vs Tanh)")
            bar = (
                alt.Chart(long_df)
                   .mark_bar()
                   .encode(
                       x=alt.X("Year:N", axis=alt.Axis(labelAngle=0, title="Year")),
                       y=alt.Y("Mean_SUC:Q", title="Mean SUC"),
                       color=alt.Color("Mapping:N",
                                       legend=alt.Legend(title="Mapping"),
                                       scale=alt.Scale(domain=["Linear","Tanh"],
                                                       range=["#1f77b4", "#ff7f0e"])),
                       xOffset="Mapping:N"  # group bars side-by-side
                   )
                   .properties(height=300)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No valid Year values to plot.")

        # ---- Downloads ----
        st.markdown("#### Download results")
        cD1, cD2, cD3 = st.columns(3)
        with cD1:
            st.download_button(
                "Download per-row CSV (both Linear & Tanh)",
                data=df_raw.to_csv(index=False).encode("utf-8"),
                file_name="rows_with_SUC_linear_tanh.csv",
                mime="text/csv"
            )
        with cD2:
            st.download_button(
                "Download per-year CSV",
                data=per_year.to_csv(index=False).encode("utf-8"),
                file_name="per_year_summary.csv",
                mime="text/csv"
            )
        with cD3:
            st.download_button(
                "Download batch parameters (CSV)",
                data=calc_tbl.to_csv(index=False).encode("utf-8"),
                file_name="batch_parameters.csv",
                mime="text/csv"
            )

        st.caption(f"WT anchor (tanh center) = **global median SDR** = {global_median_sdr:.3f}")

    except Exception as e:
        st.error(f"Failed to read/process Excel: {e}")
else:
    st.info("Upload an Excel file to convert SDR→SUC for each row using Linear and Tanh mappings. "
            "The **tanh center** is set to the **global median SDR** across the uploaded data.")
