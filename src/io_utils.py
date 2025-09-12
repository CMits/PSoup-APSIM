from __future__ import annotations
import io
import pandas as pd

def parse_year_col(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df.get("Date"), dayfirst=True, errors="coerce")
    years = pd.to_datetime(df.get("Years"), errors="coerce")
    year = date.dt.year
    if "Years" in df.columns:
        year = year.fillna(years.dt.year)
    return year

def summarize_excel(df: pd.DataFrame, *, L=0.0, U=50.0, out_min=0.0, out_max=2.0, mapper=None):
    if mapper is None:
        mapper = lambda v, L, U, out_min, out_max: (v - L) / (U - L) * (out_max - out_min) + out_min
    out = df.copy()
    out["SupplyDemandRatio"] = pd.to_numeric(out["SupplyDemandRatio"], errors="coerce")
    out["Year"] = parse_year_col(out)
    out["SUC"] = out["SupplyDemandRatio"].apply(lambda v: max(out_min, min(out_max, mapper(v, L, U, out_min, out_max))))
    per_year = (out.groupby("Year", dropna=False)
                  .agg(Year_Median_SDR=("SupplyDemandRatio","median"),
                       Year_Mean_SUC=("SUC","mean"),
                       N=("SupplyDemandRatio","size"))
                  .reset_index()
                  .sort_values("Year", kind="stable"))
    return out, per_year

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
