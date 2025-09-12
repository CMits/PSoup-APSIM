# pages/2_PSoup_to_TTN.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="PSoup → TTN",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 PSoup → TTN")
st.caption("Direction-only mapping: we **ignore SG magnitude** and only use whether +Sucrose is Lower / Same / Higher than the baseline mutant.")

# -------------------- HELPERS --------------------
def decide_label(sg_mut: float, sg_mut_suc: float, tau: float) -> str:
    """Return 'Lower' / 'Same' / 'Higher' based on Δ with tolerance tau."""
    d = (sg_mut_suc or 0.0) - (sg_mut or 0.0)
    if d > tau:
        return "Higher"
    elif d < -tau:
        return "Lower"
    else:
        return "Same"

def ttn_env_low(x: float, a0: float, a1: float) -> float:
    return a0 + a1 * x

def ttn_env_high(x: float, b0: float, b1: float) -> float:
    return b0 + b1 * x

def ttn_from_z(Tmin: float, Tmax: float, z: float) -> float:
    z = float(np.clip(z, 0.0, 1.0))
    return Tmin + z * (Tmax - Tmin)

def clamp(v: float, lo: float, hi: float) -> float:
    return float(np.minimum(hi, np.maximum(lo, v)))

def compute_cultivar_z_at_x(x_ref: float,
                            a0: float, a1: float, b0: float, b1: float,
                            c_low_int: float, c_low_slp: float,
                            c_med_int: float, c_med_slp: float,
                            c_high_int: float, c_high_slp: float) -> tuple[float,float,float]:
    """Return z values for (Low, Medium, High) cultivars relative to envelope at x_ref."""
    L = ttn_env_low(x_ref, a0, a1)
    H = ttn_env_high(x_ref, b0, b1)
    span = max(1e-9, H - L)
    low_val  = c_low_int  + c_low_slp  * x_ref
    med_val  = c_med_int  + c_med_slp  * x_ref
    high_val = c_high_int + c_high_slp * x_ref
    z_low  = np.clip((low_val  - L) / span, 0.0, 1.0)
    z_med  = np.clip((med_val  - L) / span, 0.0, 1.0)
    z_high = np.clip((high_val - L) / span, 0.0, 1.0)
    return float(z_low), float(z_med), float(z_high)

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Global settings")

scheme = st.sidebar.radio("Scheme", ["A — Continuous bins (z inside range)", "B — Cultivar lines (Alam et al.)"])

tau = st.sidebar.number_input("Tolerance τ for SG difference", value=1e-3, min_value=0.0, step=1e-3,
                              help="If |SG(+SUC) − SG(mutant)| ≤ τ ⇒ 'Same'")

# Target range: global or environment envelope
st.sidebar.markdown("**Target TTN range**")
use_env_envelope = st.sidebar.checkbox("Use environment envelope from S/D", value=False,
                                       help="If unchecked, use a fixed global range (e.g. 0–6).")
if not use_env_envelope:
    Tmin = st.sidebar.number_input("Global TTN min", value=0.0, step=0.1)
    Tmax = st.sidebar.number_input("Global TTN max", value=6.0, step=0.1)

with st.sidebar.expander("Environment envelope (Very low / Very high lines)", expanded=use_env_envelope):
    st.caption("Defaults from literature: Very low = (-1.363, 0.159), Very high = (0.960, 0.347)")
    a0 = st.number_input("Very low intercept (a0)", value=-1.363, step=0.001, format="%.3f")
    a1 = st.number_input("Very low slope (a1)", value=0.159, step=0.001, format="%.3f")
    b0 = st.number_input("Very high intercept (b0)", value=0.960, step=0.001, format="%.3f")
    b1 = st.number_input("Very high slope (b1)", value=0.347, step=0.001, format="%.3f")
    clamp_to_global = st.checkbox("Clamp envelope to [0,6]", value=True)

# ---- Scheme A: Z presets + sliders ----
if scheme.startswith("A"):
    with st.sidebar.expander("Scheme A — z settings (bins)", expanded=True):
        # Presets
        z_preset = st.selectbox(
            "z preset",
            ["Symmetric (e.g., 0.35 / 0.50 / 0.65)",
             "Cultivar-matched (from Alam et al.)",
             "Replicate-aware band (use h/l if available)",
             "Custom (manual sliders)"],
            index=0
        )

        # For cultivar-matched preset we need ref x and cultivar coeffs
        if z_preset.startswith("Cultivar"):
            x_ref = st.number_input("Reference S/D (x_ref) for cultivar-matched z", value=30.0, step=0.5)
            with st.expander("Cultivar lines for preset (Alam et al., editable)"):
                c_low_int  = st.number_input("Low cultivar intercept",   value=-0.818, step=0.001, format="%.3f")
                c_low_slp  = st.number_input("Low cultivar slope",       value=0.206,  step=0.001, format="%.3f")
                c_med_int  = st.number_input("Medium cultivar intercept",value=-0.265, step=0.001, format="%.3f")
                c_med_slp  = st.number_input("Medium cultivar slope",    value=0.247,  step=0.001, format="%.3f")
                c_high_int = st.number_input("High cultivar intercept",  value=0.329,  step=0.001, format="%.3f")
                c_high_slp = st.number_input("High cultivar slope",      value=0.300,  step=0.001, format="%.3f")

            z_low_c, z_med_c, z_high_c = compute_cultivar_z_at_x(
                x_ref, a0, a1, b0, b1,
                c_low_int, c_low_slp, c_med_int, c_med_slp, c_high_int, c_high_slp
            )
            st.caption(f"Preset (computed): Lower≈{z_low_c:.3f}, Same≈{z_med_c:.3f}, Higher≈{z_high_c:.3f}")

        # Replicate-aware band
        if z_preset.startswith("Replicate"):
            st.caption("Replicate-aware band uses z = z_min + (z_max − z_min) · p, with p = (h+0.5)/(h+l+1).")
            z_band_min = st.slider("Band minimum (z_min)", 0.0, 1.0, 0.35, 0.01)
            z_band_max = st.slider("Band maximum (z_max)", 0.0, 1.0, 0.65, 0.01)

        # Manual sliders (also serve as the final values after applying any preset)
        # We store in session_state so you can apply presets then tweak.
        if "z_lower" not in st.session_state:
            st.session_state.z_lower = 0.35
            st.session_state.z_same  = 0.50
            st.session_state.z_higher= 0.65

        # Apply preset button
        if st.button("Apply z preset"):
            if z_preset.startswith("Symmetric"):
                st.session_state.z_lower, st.session_state.z_same, st.session_state.z_higher = 0.35, 0.50, 0.65
            elif z_preset.startswith("Cultivar"):
                st.session_state.z_lower, st.session_state.z_same, st.session_state.z_higher = z_low_c, z_med_c, z_high_c
            elif z_preset.startswith("Replicate"):
                # Keep mid at 0.50; endpoints are band ends for reference
                st.session_state.z_lower, st.session_state.z_same, st.session_state.z_higher = z_band_min, 0.50, z_band_max
            else:
                pass  # Custom: leave sliders as-is

        z_lower = st.slider("z for Lower", 0.0, 1.0, st.session_state.z_lower, 0.01, key="z_lower_slider")
        z_same  = st.slider("z for Same",  0.0, 1.0, st.session_state.z_same,  0.01, key="z_same_slider")
        z_higher= st.slider("z for Higher",0.0, 1.0, st.session_state.z_higher,0.01, key="z_higher_slider")

# ---- Scheme B: Cultivar lines ----
if scheme.startswith("B"):
    with st.sidebar.expander("Scheme B — Cultivar lines (editable)", expanded=True):
        st.caption("Defaults from Alam et al. (2014)")
        c_low_int  = st.number_input("Low cultivar intercept",   value=-0.818, step=0.001, format="%.3f")
        c_low_slp  = st.number_input("Low cultivar slope",       value=0.206,  step=0.001, format="%.3f")
        c_med_int  = st.number_input("Medium cultivar intercept",value=-0.265, step=0.001, format="%.3f")
        c_med_slp  = st.number_input("Medium cultivar slope",    value=0.247,  step=0.001, format="%.3f")
        c_high_int = st.number_input("High cultivar intercept",  value=0.329,  step=0.001, format="%.3f")
        c_high_slp = st.number_input("High cultivar slope",      value=0.300,  step=0.001, format="%.3f")
        st.caption("Label→Line mapping (feel free to change)")
        map_lower = st.selectbox("Lower →", ["Low", "Medium", "High"], index=0)
        map_same  = st.selectbox("Same  →", ["Low", "Medium", "High"], index=1)
        map_high  = st.selectbox("Higher→", ["Low", "Medium", "High"], index=2)

# -------------------- SINGLE-ROW CALCULATOR --------------------
st.subheader("Single-row calculator (explain the flow)")
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    x = st.number_input("APSIM S/D (x)", value=30.0, step=0.5,
                        help="Only used if 'Use environment envelope' is ON.")
with c2:
    sg_mut = st.number_input("SG (mutant)", value=1.30, step=0.01)
with c3:
    sg_mut_suc = st.number_input("SG (mutant + sucrose)", value=1.50, step=0.01)
with c4:
    label = decide_label(sg_mut, sg_mut_suc, tau)
    st.metric("PSoup direction", label)

# Optional replicate counts for single-row (used if z preset is replicate-aware)
if scheme.startswith("A") and 'Replicate' in (st.session_state.get('z_preset_name','') or z_preset):
    with st.expander("Optional: Replicate counts for this row (Scheme A, replicate-aware band)"):
        h = st.number_input("h = # Higher", min_value=0, value=0, step=1)
        l = st.number_input("l = # Lower", min_value=0, value=0, step=1)
        p_single = (h + 0.5) / (h + l + 1) if (h + l) >= 0 else 0.5

# Determine TTN bounds for this row
if use_env_envelope:
    L = ttn_env_low(x, a0, a1)
    H = ttn_env_high(x, b0, b1)
    if clamp_to_global:
        L = clamp(L, 0, 6)
        H = clamp(H, 0, 6)
    if H < L:
        H = L
else:
    L, H = Tmin, Tmax

# Compute TTN for the single row
if scheme.startswith("A"):
    # z selection
    st.session_state.z_preset_name = z_preset  # remember choice
    if z_preset.startswith("Replicate"):
        # Build band
        # Try to get band from sliders (z_lower/z_higher) or earlier values
        z_min = min(z_lower, z_higher)
        z_max = max(z_lower, z_higher)
        # If user provided replicates above, use p_single; else default by label to ends/middle
        if 'p_single' in locals():
            z_used = z_min + (z_max - z_min) * p_single
        else:
            z_used = {"Lower": z_min, "Same": 0.5, "Higher": z_max}[label]
    else:
        z_map = {"Lower": z_lower, "Same": z_same, "Higher": z_higher}
        z_used = z_map[label]

    ttn_pred = ttn_from_z(L, H, z_used)

    m1, m2, m3 = st.columns([1,1,1])
    with m1: st.metric("TTN min (L)", f"{L:.3f}")
    with m2: st.metric("TTN max (H)", f"{H:.3f}")
    with m3: st.metric("Predicted TTN", f"{ttn_pred:.3f}")

    st.markdown("**Formula:**  \n"
                r"$\mathrm{TTN} = L + z\,(H-L)$  with  "
                f"$z={z_used:.3f}$ for **{label}**")

    # Visual: the three bin positions vs [L,H] (ordered Lower→Same→Higher)
    vis_df = pd.DataFrame({
        "Label": ["Lower","Same","Higher"],
        "z": [z_lower, z_same, z_higher],
        "TTN": [ttn_from_z(L,H,z_lower), ttn_from_z(L,H,z_same), ttn_from_z(L,H,z_higher)]
    })
    bars = (
        alt.Chart(vis_df)
          .mark_bar()
          .encode(
              x=alt.X("Label:N", sort=["Lower","Same","Higher"], title=None),
              y=alt.Y("TTN:Q", title="TTN"),
              color=alt.Color(
                  "Label:N",
                  legend=alt.Legend(title="Bin"),
                  scale=alt.Scale(
                      domain=["Lower","Same","Higher"],
                      range=["#d62728","#6c757d","#2ca02c"]
                  )
              )
          )
          .properties(height=260)
    )
    st.altair_chart(bars, use_container_width=True)

else:
    # Scheme B: pick cultivar line by label
    def line_value(which: str, x_val: float) -> float:
        if which == "Low":
            return c_low_int + c_low_slp * x_val
        elif which == "Medium":
            return c_med_int + c_med_slp * x_val
        else:
            return c_high_int + c_high_slp * x_val

    label2line = {"Lower": map_lower, "Same": map_same, "Higher": map_high}
    chosen_line = label2line[label]
    ttn_pred = line_value(chosen_line, x) if use_env_envelope else ttn_from_z(Tmin, Tmax, {"Low":0.35,"Medium":0.50,"High":0.65}[chosen_line])

    m1, m2, m3 = st.columns([1,1,1])
    with m1: st.metric("Chosen line", chosen_line)
    with m2: st.metric("APSIM S/D (x)", f"{x:.2f}")
    with m3: st.metric("Predicted TTN", f"{ttn_pred:.3f}")

    st.markdown("**Lines (editable coefficients):**")
    st.latex(r"""\begin{aligned}
    \text{Low: } & TTN = %.3f + %.3f\,x\\
    \text{Medium: } & TTN = %.3f + %.3f\,x\\
    \text{High: } & TTN = %.3f + %.3f\,x
    \end{aligned}""" % (c_low_int, c_low_slp, c_med_int, c_med_slp, c_high_int, c_high_slp))

    # Visual: bars for the three lines, ordered Low→Medium→High
    vis_df = pd.DataFrame({
        "Cultivar": ["Low","Medium","High"],
        "TTN": [line_value("Low",x), line_value("Medium",x), line_value("High",x)]
    })
    bars = (
        alt.Chart(vis_df)
          .mark_bar()
          .encode(
              x=alt.X("Cultivar:N", sort=["Low","Medium","High"], title=None),
              y=alt.Y("TTN:Q", title="TTN"),
              color=alt.Color(
                  "Cultivar:N",
                  legend=alt.Legend(title="Line"),
                  scale=alt.Scale(
                      domain=["Low","Medium","High"],
                      range=["#1f77b4","#6c757d","#ff7f0e"]
                  )
              )
          )
          .properties(height=260)
    )
    st.altair_chart(bars, use_container_width=True)

st.divider()

# -------------------- BATCH (EXCEL) --------------------
st.subheader("Batch (Excel): PSoup direction → TTN (Genotypes, Resource classes, Yield)")

import altair as alt
from pathlib import Path

# ---- Data source ----
st.markdown("### Data source")
source_choice = st.radio(
    "Provide data via:",
    ["Upload Excel (.xlsx)", "Use bundled example file"],
    index=0
)

xl = None
if source_choice == "Use bundled example file":
    example_path = Path(__file__).resolve().parents[1] / "Examples" / "example_psoup_to_ttn_25yrs_5mutants_2001_2025.xlsx"
    if not example_path.exists():
        st.error(f"Example file not found at: {example_path}")
    else:
        xl = pd.ExcelFile(str(example_path))
        st.success(f"Loaded example: {example_path.name}")
else:
    upl = st.file_uploader(
        "Upload .xlsx with columns: Genotype (opt), S/D_APSIM (or SD/SDR), SG_mut, SG_mut_SUC, h (opt), l (opt), Yield (opt), Date/Years (opt).",
        type=["xlsx"]
    )
    if upl is not None:
        xl = pd.ExcelFile(upl)

# ---- Helpers ----
def extract_year(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce")
    years = pd.to_datetime(df.get("Years"), errors="coerce")
    y = date.dt.year
    if "Years" in df.columns:
        y = y.fillna(years.dt.year)
    return y

def classify_resource(sdr: float, t1: float, t2: float) -> str:
    if pd.isna(sdr): return "Unknown"
    return "Low" if sdr < t1 else ("Medium" if sdr < t2 else "High")

with st.expander("Resource class settings", expanded=True):
    st.caption("Label each Year by SDR into Low / Medium / High resource.")
    t1 = st.slider("Threshold 1 (Low < t1)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)
    t2 = st.slider("Threshold 2 (t1 ≤ Medium < t2; High ≥ t2)", min_value=0.0, max_value=50.0, value=35.0, step=1.0)

# ---- Parse workbook ----
df = None
if xl is not None:
    sheet = st.selectbox("Sheet", xl.sheet_names, index=0)
    try:
        df = xl.parse(sheet_name=sheet).copy()
    except Exception as e:
        st.error(f"Could not read the selected sheet: {e}")
        df = None

if df is not None and not df.empty:
    # Safety
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Normalize
    df["Genotype"] = df.get("Genotype").astype("string")
    if "S/D_APSIM" in df.columns:
        df["SDR"] = pd.to_numeric(df["S/D_APSIM"], errors="coerce")
    elif "SD" in df.columns:
        df["SDR"] = pd.to_numeric(df["SD"], errors="coerce")
    else:
        df["SDR"] = pd.to_numeric(df.get("SDR"), errors="coerce")

    df["SG_mut"] = pd.to_numeric(df.get("SG_mut"), errors="coerce")
    df["SG_mut_SUC"] = pd.to_numeric(df.get("SG_mut_SUC"), errors="coerce")
    df["Yield"] = pd.to_numeric(df.get("Yield"), errors="coerce")
    df["Year"] = extract_year(df)

    # Replicates (optional)
    has_hl = ("h" in df.columns) and ("l" in df.columns)
    if has_hl:
        df["h"] = pd.to_numeric(df["h"], errors="coerce").fillna(0).astype(int)
        df["l"] = pd.to_numeric(df["l"], errors="coerce").fillna(0).astype(int)
        df["p"] = (df["h"] + 0.5) / (df["h"] + df["l"] + 1)

    # Direction label (tau defined earlier in the page)
    df["Delta"] = df["SG_mut_SUC"] - df["SG_mut"]
    df["Label"] = np.where(df["Delta"] >  tau, "Higher",
                    np.where(df["Delta"] < -tau, "Lower", "Same"))

    # Bounds: envelope or global (vars defined in sidebar earlier)
    if use_env_envelope:
        df["TTN_Low"]  = a0 + a1 * df["SDR"]
        df["TTN_High"] = b0 + b1 * df["SDR"]
        if clamp_to_global:
            df["TTN_Low"]  = df["TTN_Low"].clip(0, 6)
            df["TTN_High"] = df["TTN_High"].clip(0, 6)
        df["TTN_Low"], df["TTN_High"] = np.minimum(df["TTN_Low"], df["TTN_High"]), np.maximum(df["TTN_Low"], df["TTN_High"])
    else:
        df["TTN_Low"], df["TTN_High"] = Tmin, Tmax

    # Predicted TTN (Scheme A / B)
    try:
        replicate_mode = scheme.startswith("A") and z_preset.startswith("Replicate")
    except NameError:
        replicate_mode = False

    if scheme.startswith("A"):
        if replicate_mode and has_hl:
            z_min = float(min(z_lower, z_higher))
            z_max = float(max(z_lower, z_higher))
            df["z"] = z_min + (z_max - z_min) * df["p"].fillna(0.5)
        else:
            z_map = {"Lower": float(z_lower), "Same": float(z_same), "Higher": float(z_higher)}
            df["z"] = df["Label"].map(z_map).astype(float)
        df["TTN_pred"] = df["TTN_Low"] + df["z"] * (df["TTN_High"] - df["TTN_Low"])
    else:
        # Scheme B: snap to cultivar line
        try:
            c_low_int, c_low_slp
        except NameError:
            c_low_int, c_low_slp = -0.818, 0.206
            c_med_int, c_med_slp = -0.265, 0.247
            c_high_int, c_high_slp = 0.329, 0.300
            map_lower, map_same, map_high = "Low", "Medium", "High"

        def line_at(lbl, sdr):
            which = {"Lower": map_lower, "Same": map_same, "Higher": map_high}[lbl]
            if which == "Low":
                return c_low_int + c_low_slp * sdr
            elif which == "Medium":
                return c_med_int + c_med_slp * sdr
            else:
                return c_high_int + c_high_slp * sdr

        df["TTN_pred"] = [line_at(lbl, sdr) for lbl, sdr in zip(df["Label"], df["SDR"])]
        df["z"] = np.nan

    # Resource class and Year string
    df["ResourceClass"] = [classify_resource(v, t1, t2) for v in df["SDR"]]
    df["YearStr"] = df["Year"].astype("Int64").astype(str)

    # Preview
    st.markdown("**Preview of calculations**")
    cols = ["Year","YearStr","ResourceClass","Genotype","SDR"]
    if has_hl: cols += ["h","l","p"]
    cols += ["SG_mut","SG_mut_SUC","Delta","Label","TTN_Low","TTN_High","z","TTN_pred","Yield"]
    cols_unique = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in seen:
            cols_unique.append(c); seen.add(c)
    df_preview = df.loc[:, ~df.columns.duplicated()].copy()
    st.dataframe(df_preview[cols_unique], use_container_width=True, height=360)

    # Filters
    st.markdown("### Filters")
    geno_all = sorted([g for g in df["Genotype"].dropna().unique() if g != "nan"])
    selected_genos = st.multiselect("Select Genotypes", geno_all, default=geno_all)
    df_plot = df[df["Genotype"].isin(selected_genos)] if selected_genos else df.copy()

    # Chart A: TTN bars grouped by Year and Genotype (side-by-side), faceted by Resource class
    st.markdown("### TTN by Year and Genotype (grouped bars), faceted by Resource class")
    if not df_plot.empty:
        chartA = (
            alt.Chart(df_plot)
              .mark_bar()
              .encode(
                  x=alt.X("YearStr:N", title="Year", axis=alt.Axis(labelAngle=0)),
                  y=alt.Y("TTN_pred:Q", title="Predicted TTN"),
                  color=alt.Color("Genotype:N", legend=alt.Legend(title="Genotype")),
                  xOffset=alt.X("Genotype:N"),
                  column=alt.Column("ResourceClass:N",
                                    sort=["Low","Medium","High","Unknown"],
                                    header=alt.Header(title="Resource class"))
              )
              .properties(height=300)
        )
        st.altair_chart(chartA, use_container_width=True)
    else:
        st.info("No rows after filter.")

    # Chart B: Yield by Year per Genotype (lines), faceted by Resource class
    st.markdown("### Yield by Year per Genotype (lines), faceted by Resource class")
    df_y = df_plot.dropna(subset=["Yield"]).copy()
    if not df_y.empty:
        chartB = (
            alt.Chart(df_y)
              .mark_line(point=True)
              .encode(
                  x=alt.X("YearStr:N", title="Year", axis=alt.Axis(labelAngle=0)),
                  y=alt.Y("Yield:Q", title="Yield"),
                  color=alt.Color("Genotype:N", legend=alt.Legend(title="Genotype")),
                  column=alt.Column("ResourceClass:N",
                                    sort=["Low","Medium","High","Unknown"],
                                    header=alt.Header(title="Resource class"))
              )
              .properties(height=300)
        )
        st.altair_chart(chartB, use_container_width=True)
    else:
        st.info("Yield column missing or all NaN in filtered set.")

    # Downloads
    st.markdown("### Downloads")
    cdl1, cdl2 = st.columns(2)
    with cdl1:
        st.download_button(
            "Download rows (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="psoup_to_ttn_rows.csv",
            mime="text/csv"
        )
    with cdl2:
        params = {
            "scheme": scheme,
            "tau": tau,
            "use_env_envelope": use_env_envelope,
            "a0": a0 if use_env_envelope else None,
            "a1": a1 if use_env_envelope else None,
            "b0": b0 if use_env_envelope else None,
            "b1": b1 if use_env_envelope else None,
            "Tmin": Tmin if not use_env_envelope else None,
            "Tmax": Tmax if not use_env_envelope else None,
            "resource_thresholds": {"Low<": t1, "Medium<": t2, "High≥": t2},
            "genotypes_selected": selected_genos,
        }
        st.download_button(
            "Download parameters (CSV)",
            data=pd.DataFrame([params]).to_csv(index=False).encode("utf-8"),
            file_name="psoup_to_ttn_params.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a file or select the bundled example to run batch calculations.")


