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

# -------------------- BATCH (EXCEL) --------------------
import altair as alt
from pathlib import Path

st.markdown("---")
st.subheader("Batch (Excel): Map APSIM SDR → PSoup SUC")

# ---- Data source: upload or bundled example ----
st.markdown("### Data source")
src_choice = st.radio(
    "Provide data via:",
    ["Upload Excel (.xlsx)", "Use bundled example file"],
    index=0,
)

xl = None
if src_choice == "Use bundled example file":
    example_path = Path(__file__).resolve().parents[1] / "Examples" / "psoup_factorial_Multiyears.xlsx"
    if not example_path.exists():
        st.error(f"Example file not found at: {example_path}")
    else:
        xl = pd.ExcelFile(str(example_path))
        st.success(f"Loaded example: {example_path.name}")
else:
    upl = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
    if upl is not None:
        xl = pd.ExcelFile(upl)

# ---- Read & normalize ----
def extract_year_cols(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce")
    years = pd.to_datetime(df.get("Years"), errors="coerce")
    y = date.dt.year
    if "Years" in df.columns:
        y = y.fillna(years.dt.year)
    return y

if xl is not None:
    sheet_name = st.selectbox("Sheet", xl.sheet_names, index=0)
    try:
        df = xl.parse(sheet_name=sheet_name).copy()
    except Exception as e:
        st.error(f"Could not read the selected sheet: {e}")
        df = None
else:
    df = None

if df is not None and not df.empty:
    # Drop duplicate-named cols to avoid pyarrow issues
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Find SDR column
    sdr_col_candidates = [
        "SupplyDemandRatio", "SupplyDemand", "S/D_APSIM", "SDR", "SD"
    ]
    sdr_col = next((c for c in sdr_col_candidates if c in df.columns), None)
    if sdr_col is None:
        st.error("No SDR column found. Expected one of: "
                 "`SupplyDemandRatio`, `SupplyDemand`, `S/D_APSIM`, `SDR`, `SD`.")
        st.stop()

    # Parse
    df["SDR"] = pd.to_numeric(df[sdr_col], errors="coerce")
    df["Year"] = extract_year_cols(df)

    # Compute global stats
    global_median_sdr = float(np.nanmedian(df["SDR"]))
    st.info(f"Global SDR median (used as tanh center in batch): **{global_median_sdr:.3f}**")

    # Compute both mappings (uses your page's sidebar settings L, U, out_min, out_max, k)
    def _safe_halfspan(L, U):
        return (U - L) / 2.0 if (U is not None and U > L) else 25.0

    # Linear SUC
    df["SUC_linear"] = df["SDR"].apply(
        lambda x: map_linear(x, L=L, U=U, out_min=out_min, out_max=out_max)
    )

    # Smooth (tanh) SUC, center = global median SDR
    df["SUC_tanh"] = df["SDR"].apply(
        lambda x: map_tanh(
            x,
            center=global_median_sdr,
            halfspan=_safe_halfspan(L, U),
            k=k if "k" in locals() else 1.0,
            out_min=out_min,
            out_max=out_max
        )
    )

    # Which to display
    st.markdown("### Which mapping to display")
    show_mode = st.radio(
        "Choose mapping view",
        ["Linear only", "Smooth (tanh) only", "Both"],
        index=2
    )

    # Preview table
    st.markdown("**Preview of rows**")
    cols = ["Year", "SDR", "SUC_linear", "SUC_tanh"]
    keep = [c for c in cols if c in df.columns]
    st.dataframe(df[keep], use_container_width=True, height=340)

    # Per-year summary (mean)
    per_year = (
        df.groupby("Year", dropna=True)
          .agg(SDR_mean=("SDR","mean"),
               SUC_linear_mean=("SUC_linear","mean"),
               SUC_tanh_mean=("SUC_tanh","mean"))
          .reset_index()
          .dropna(subset=["Year"])
    )

    # Charts
    st.markdown("### Per-year SUC summary")
    base = per_year.copy()
    base["Year"] = base["Year"].astype("Int64").astype(str)

    if show_mode == "Linear only":
        chart = (
            alt.Chart(base)
            .mark_bar()
            .encode(
                x=alt.X("Year:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("SUC_linear_mean:Q", title="Mean SUC (Linear)"),
                color=alt.value("#1f77b4")
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    elif show_mode == "Smooth (tanh) only":
        chart = (
            alt.Chart(base)
            .mark_bar()
            .encode(
                x=alt.X("Year:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("SUC_tanh_mean:Q", title="Mean SUC (tanh)"),
                color=alt.value("#ff7f0e")
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        # Both → side-by-side bars per year
        melt = base.melt(
            id_vars=["Year"],
            value_vars=["SUC_linear_mean","SUC_tanh_mean"],
            var_name="Mapping",
            value_name="SUC"
        )
        mapping_order = ["SUC_linear_mean","SUC_tanh_mean"]
        mapping_labels = {"SUC_linear_mean":"Linear","SUC_tanh_mean":"Smooth (tanh)"}
        melt["Mapping"] = melt["Mapping"].map(mapping_labels)

        chart = (
            alt.Chart(melt)
            .mark_bar()
            .encode(
                x=alt.X("Year:N", axis=alt.Axis(labelAngle=0), title="Year"),
                y=alt.Y("SUC:Q", title="Mean SUC"),
                color=alt.Color("Mapping:N",
                                scale=alt.Scale(domain=["Linear","Smooth (tanh)"],
                                                range=["#1f77b4","#ff7f0e"]),
                                legend=alt.Legend(title="Mapping")),
                xOffset=alt.X("Mapping:N", sort=["Linear","Smooth (tanh)"])
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # Downloads
    st.markdown("### Downloads")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download rows with SUC (CSV)",
            data=df[keep].to_csv(index=False).encode("utf-8"),
            file_name="sdr_to_suc_rows.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download per-year summary (CSV)",
            data=per_year.to_csv(index=False).encode("utf-8"),
            file_name="sdr_to_suc_per_year.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a file or select the bundled example to run batch mapping.")




