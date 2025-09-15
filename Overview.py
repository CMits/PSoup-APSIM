# app.py  — Overview (landing page)
import streamlit as st

GA_MEASUREMENT_ID = "G-0T2X9HX1T8"

# Inject GA only once per user session
if "ga_injected" not in st.session_state:
    st.session_state["ga_injected"] = True
    st.markdown(
        f"""
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}"></script>
        <script>
          window.dataLayer = window.dataLayer || [];
          function gtag(){{dataLayer.push(arguments);}}
          gtag('js', new Date());
          // Avoid duplicate auto page_view on Streamlit reruns
          gtag('config', '{GA_MEASUREMENT_ID}', {{ 'send_page_view': false }});
          // Send one explicit page_view for this Streamlit page
          gtag('event', 'page_view', {{
            page_title: 'Overview',
            page_location: window.location.href,
            page_path: window.location.pathname + window.location.search
          }});
        </script>
        """,
        unsafe_allow_html=True,
    )




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



