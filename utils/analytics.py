# utils/analytics.py
import json
import streamlit as st
import streamlit.components.v1 as components

def inject_gtm(container_id: str):
    """
    Inject the GTM container once per session.
    Call this at the top of every page.
    """
    if not container_id:
        return
    if "gtm_injected" in st.session_state:
        return
    st.session_state["gtm_injected"] = True
    components.html(f"""
    <!-- Google Tag Manager -->
    <script>(function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{'gtm.start':
    new Date().getTime(),event:'gtm.js'}});var f=d.getElementsByTagName(s)[0],
    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
    }})(window,document,'script','dataLayer','{container_id}');</script>
    """, height=0)

def virtual_pageview(title: str, path: str | None = None):
    """
    Send a 'virtual' page_view so GA4 knows which Streamlit 'page' is shown.
    Must match GTM Custom Event 'virtual_page_view'.
    """
    if not title:
        title = "Streamlit Page"
    if path is None:
        # Make a clean slug from the title
        path = "/" + "".join(ch if ch.isalnum() else "-" for ch in title.lower()).strip("-")
    # Use JS for page_location so the real URL is captured
    components.html(f"""
    <script>
      window.dataLayer = window.dataLayer || [];
      window.dataLayer.push({{
        event: 'virtual_page_view',
        page_title: {json.dumps(title)},
        page_path: {json.dumps(path)},
        page_location: window.location.href
      }});
    </script>
    """, height=0)

def gtm_event(event_name: str, params: dict | None = None):
    """
    Push any custom event into dataLayer (e.g., 'button_click', 'select_change').
    Define a matching Custom Event trigger in GTM.
    """
    payload = {"event": event_name}
    if params:
        # Only include JSON-serialisable values
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                payload[k] = v
            else:
                payload[k] = str(v)
    components.html(f"""
    <script>
      window.dataLayer = window.dataLayer || [];
      window.dataLayer.push({json.dumps(payload)});
    </script>
    """, height=0)
