from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from .mapping import map_linear
from .io_utils import summarize_excel, df_to_csv_bytes

def render_sdr_to_suc():
    st.header("🧮 SDR → SUC Mapper")

    with st.expander("⚙️ Mapping settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1: L = st.number_input("SDR lower bound (L)", value=0.0, step=1.0)
        with c2: U = st.number_input("SDR upper bound (U)", value=50.0, step=1.0)
        with c3: out_min = st.number_input("SUC min", value=0.0, step=0.1)
        with c4: out_max = st.number_input("SUC max", value=2.0, step=0.1)
        st.caption(r"$s(x)=\min\!\left(%0.1f,\max\!\left(%0.1f,\frac{x-L}{U-L}\times(%0.1f-%0.1f)+%0.1f\right)\right)$"
                   % (out_max, out_min, out_max, out_min, out_min))

    st.subheader("Quick calculator")
    col_in, col_out = st.columns([1, 1])
    with col_in:
        x = st.number_input("APSIM SDR (x)", value=30.0, step=0.5)
    with col_out:
        s_val = map_linear(x, L=L, U=U, out_min=out_min, out_max=out_max)
        st.metric("PSoup SUC", f"{s_val:0.3f}")

    xs = np.linspace(L, U, 501) if U > L else np.linspace(0, 50, 501)
    ys = [map_linear(v, L=L, U=U, out_min=out_min, out_max=out_max) for v in xs]
    st.line_chart(pd.DataFrame({"SDR": xs, "SUC": ys}), x="SDR", y="SUC", height=260)

    st.divider()
    st.subheader("Upload APSIM Excel to map SDR → SUC")
    upl = st.file_uploader("Drop an .xlsx with 'SupplyDemandRatio' and 'Date' and/or 'Years'.", type=["xlsx"])
    if upl is not None:
        xl = pd.ExcelFile(upl)
        sheet = st.selectbox("Sheet", xl.sheet_names, index=0)
        df_raw = xl.parse(sheet_name=sheet)
        st.write("**Preview:**")
        st.dataframe(df_raw.head(15), use_container_width=True)

        rows, per_year = summarize_excel(
            df_raw, L=L, U=U, out_min=out_min, out_max=out_max,
            mapper=lambda v,L,U,out_min,out_max: (v - L)/(U - L) * (out_max - out_min) + out_min
        )

        st.markdown("#### Per-row results")
        st.dataframe(rows[["Year","SupplyDemandRatio","SUC"]]
                     .sort_values(["Year","SupplyDemandRatio"]),
                     use_container_width=True, height=300)

        st.markdown("#### Per-year summary")
        cA, cB = st.columns([1.3, 1])
        with cA:
            st.dataframe(per_year, use_container_width=True, height=300)
        with cB:
            plot_df = per_year.dropna(subset=["Year"]).copy()
            if not plot_df.empty:
                plot_df["Year"] = plot_df["Year"].astype(int).astype(str)
                st.markdown("Median SDR by Year")
                st.bar_chart(plot_df, x="Year", y="Year_Median_SDR", height=200)
                st.markdown("Mean SUC by Year")
                st.bar_chart(plot_df, x="Year", y="Year_Mean_SUC", height=200)
            else:
                st.info("No valid Year to plot.")

        cD1, cD2 = st.columns(2)
        with cD1:
            st.download_button("Download per-row CSV", data=df_to_csv_bytes(rows),
                               file_name="rows_with_SUC.csv", mime="text/csv")
        with cD2:
            st.download_button("Download per-year CSV", data=df_to_csv_bytes(per_year),
                               file_name="per_year_summary.csv", mime="text/csv")
    else:
        st.info("Upload an Excel file to see year-wise SDR and mapped SUC.")
