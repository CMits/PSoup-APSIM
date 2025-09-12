# lib/engage_widget.py
from __future__ import annotations
import streamlit as st
from typing import Optional
from . import engage_db as db

def _ensure_user() -> str:
    st.session_state.setdefault("display_name", "")
    with st.popover("Set your name", use_container_width=True):
        st.text_input("How should we display your name?", key="display_name", placeholder="e.g., Nicole / Graeme / Anonymous ok too")
    return st.session_state["display_name"] or "anonymous"

def render_engage_widget(
    idea_key: str,
    idea_name: str,
    idea_description: str = "",
    default_created_by: str = "Chris",
):
    """Embed this on ANY page to attach voting + quick comment to that page/idea."""
    db.init_db()
    idea_id = db.upsert_idea(idea_key, idea_name, idea_description, default_created_by)
    user = _ensure_user()

    with st.container(border=True):
        st.markdown(f"### 💡 Feedback: *{idea_name}*")
        st.caption(idea_description or "Collect votes and quick comments from supervisors here.")

        # --- Voting ---
        st.subheader("Vote")
        col1, col2 = st.columns([2,1])
        with col1:
            vote_label = {"Yes (pursue)":1, "Unsure":0, "No (park it)":-1}
            vote_key = st.radio("What’s your call?", list(vote_label.keys()), horizontal=True, key=f"v_{idea_key}")
        with col2:
            conf = st.slider("Confidence", 0, 100, 70, key=f"c_{idea_key}")
        if st.button("Submit vote", type="primary", key=f"sv_{idea_key}"):
            db.cast_vote(idea_id, user, vote_label[vote_key], conf)
            st.success("Vote saved.")

        # --- Quick Comment ---
        st.subheader("Quick comment")
        tag = st.selectbox("Tag", ["scientific", "practical", "presentation", "implementation", "other"], index=0, key=f"tag_{idea_key}")
        anon = st.checkbox("Post anonymously", value=False, key=f"anon_{idea_key}")
        text = st.text_area("Your comment", key=f"txt_{idea_key}", placeholder="Keep/kill? Risks? Validation needed? UX issues?")
        if st.button("Add comment", key=f"ac_{idea_key}"):
            if text.strip():
                db.add_comment(idea_id, user, text.strip(), tag, anon, parent_id=None)
                st.success("Comment added.")
            else:
                st.warning("Type something first 🙂")

        # --- Mini tally ---
        with st.expander("Current pulse"):
            votes = db.fetch_votes(idea_id)
            yes = sum(1 for v in votes if v["vote"]==1)
            uns = sum(1 for v in votes if v["vote"]==0)
            no  = sum(1 for v in votes if v["vote"]==-1)
            st.write(f"Yes: **{yes}** · Unsure: **{uns}** · No: **{no}** (n={len(votes)})")
            if votes:
                st.progress(int(sum(v["confidence"] for v in votes)/len(votes)))

        st.caption("Tip: open the full **Feedback & Voting** page for dashboards, threads, actions, history.")
