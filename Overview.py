# app.py  — Overview (landing page)

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="APSIM ↔ PSoup Portal", page_icon="🌱", layout="wide")

st.title("Overview: APSIM ↔ PSoup connection")
st.write(
    """
    This portal links **APSIM** (quantitative) with **PSoup** (qualitative).
    - APSIM outputs **SupplyDemandRatio (SDR)**.
    - We map SDR → **SUC** (PSoup sucrose input).
    - PSoup returns **Sustained growth**, which we’ll later translate to **TTN**.
    Use the **left sidebar** to open the other pages.
    """
)

# Try your repo asset first, then your local file path as fallback
candidates = [
    Path("assets/overview.png"),
    Path(r"C:\Users\uqcmitsa\OneDrive - The University of Queensland\PSoup-APSIM\Picture1.png"),
]
img = next((p for p in candidates if p.exists()), None)
if img:
    st.image(str(img), caption="APSIM ↔ PSoup conceptual diagram", use_container_width=True)
else:
    st.warning("Overview image not found. Put it at `assets/overview.png` or update app.py.")
