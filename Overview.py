# app.py  — Overview (landing page)

import streamlit as st
from utils.analytics import inject_gtm, virtual_pageview, gtm_event

# Option 1: hardcode
GTM_ID = "GTM-MMPTCHWX"

# Option 2 (cleaner): store in .streamlit/secrets.toml:
# [analytics]
# gtm = "GTM-XXXXXXX"
# Then:
# GTM_ID = st.secrets["analytics"]["gtm"]

inject_gtm(GTM_ID)                 # inject once per session
virtual_pageview("Home")  

import streamlit as st






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



