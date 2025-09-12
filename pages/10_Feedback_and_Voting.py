# pages/10_Feedback_and_Voting.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from datetime import date
from lib import engage_db as db

# Optional imports (fail gracefully)
WC_OK = True
SK_OK = True
try:
    from wordcloud import WordCloud
except Exception:
    WC_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    SK_OK = False

st.set_page_config(page_title="Feedback & Voting", page_icon="🗳️", layout="wide")
db.init_db()

# ---- User identity ----
st.sidebar.subheader("Your identity")
display_name = st.sidebar.text_input(
    "Display name (used for votes & comments)",
    value=st.session_state.get("display_name","")
)
if display_name:
    st.session_state["display_name"] = display_name
st.sidebar.caption("Leave blank to be recorded as 'anonymous'.")

# ---- Idea selector / creator ----
st.sidebar.subheader("Idea")
ideas = db.list_ideas()
idea_names = [f'{i["name"]} ({i["key"]})' for i in ideas] if ideas else []
mode = st.sidebar.radio("Select or add", ["Select existing", "Add new"], horizontal=True)

if mode == "Select existing" and ideas:
    pick = st.sidebar.selectbox("Which idea?", idea_names)
    idea = ideas[idea_names.index(pick)]
else:
    st.sidebar.text_input("New idea key", key="new_key", placeholder="psoup-apsim-connector")
    st.sidebar.text_input("New idea name", key="new_name", placeholder="PSoup ↔ APSIM Connector")
    st.sidebar.text_area("Description", key="new_desc", placeholder="What is it & why it matters?")
    if st.sidebar.button("Create idea", type="primary"):
        if st.session_state.get("new_key") and st.session_state.get("new_name"):
            _ = db.upsert_idea(
                st.session_state["new_key"].strip(),
                st.session_state["new_name"].strip(),
                st.session_state.get("new_desc","").strip(),
                st.session_state.get("display_name","Chris") or "Chris",
            )
            st.success("Idea created. Select it from the sidebar.")
            st.rerun()
        else:
            st.sidebar.warning("Please provide a key and a name.")

if mode == "Select existing" and ideas:
    idea_id = idea["id"]
    st.title(f"🗳️ Feedback & Voting — {idea['name']}")
    st.caption(idea["description"])

    # ======= TABS (5 only) =======
    t1, t2, t3, t4, t5 = st.tabs([
        "Vote",
        "Comments & Threads",
        "Dashboard",
        "Actions",
        "History"
    ])

    # --- Vote tab ---
    with t1:
        st.subheader("Cast or update your vote")
        voter = st.session_state.get("display_name","") or "anonymous"

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            vote = st.radio("Decision", ["Yes (pursue)", "Unsure", "No (park)"], horizontal=True)
        with col2:
            conf = st.slider("Confidence", 0, 100, 70)
        with col3:
            if st.button("Submit vote", type="primary"):
                vote_map = {"Yes (pursue)": 1, "Unsure": 0, "No (park)": -1}
                db.cast_vote(idea_id, voter, vote_map.get(vote, 0), conf)
                st.success("Vote saved/updated.")

        votes = db.fetch_votes(idea_id)
        if votes:
            df = pd.DataFrame([dict(v) for v in votes])   # ensure named columns
            df["label"] = df["vote"].map({1:"Yes",0:"Unsure",-1:"No"})
            st.dataframe(df[["voter","label","confidence","created_at"]], use_container_width=True)
        else:
            st.info("No votes yet — be the first!")

    # --- Comments & Threads ---
    with t2:
        st.subheader("Add a comment")
        c1, c2 = st.columns([3,1])
        with c1:
            text = st.text_area("Your comment", placeholder="Be specific and kind ❤️  (risks, validation ideas, UX improvements, etc.)")
        with c2:
            tag = st.selectbox("Tag", ["scientific","practical","presentation","implementation","other"])
            anon = st.checkbox("Anonymous", value=False)
            if st.button("Post", type="primary"):
                if text.strip():
                    db.add_comment(idea_id, st.session_state.get("display_name","") or "anonymous", text.strip(), tag, anon, parent_id=None)
                    st.success("Comment added.")
                    st.rerun()
                else:
                    st.warning("Type something first 🙂")

        st.markdown("---")
        st.subheader("Discussion")
        comments = db.fetch_comments(idea_id)
        if not comments:
            st.info("No comments yet.")
        else:
            by_parent = {}
            for c in comments:
                by_parent.setdefault(c["parent_id"], []).append(c)

            def render_comment(c):
                author = "anonymous" if c["anonymous"] else c["author"]
                st.markdown(f"**{author}** · *{c['tag']}* · ▲ {c['score']} · {c['created_at']}")
                st.markdown(c["text"])
                cc1, cc2, cc3 = st.columns([1,1,5])
                with cc1:
                    if st.button(f"▲ {c['id']}", key=f"up_{c['id']}"):
                        db.vote_comment(c["id"], st.session_state.get("display_name","") or "anonymous", 1)
                        st.rerun()
                with cc2:
                    reply_txt = st.text_input("Reply", key=f"r_{c['id']}", label_visibility="collapsed", placeholder="Reply…")
                    if st.button("Send", key=f"s_{c['id']}"):
                        if reply_txt.strip():
                            db.add_comment(idea_id, st.session_state.get("display_name","") or "anonymous", reply_txt.strip(), c["tag"], False, parent_id=c["id"])
                            st.rerun()

            # Top-level first
            for top in by_parent.get(None, []):
                with st.container(border=True):
                    render_comment(top)
                    # children
                    for child in by_parent.get(top["id"], []):
                        with st.container(border=True):
                            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;↳")
                            render_comment(child)

    # --- Dashboard ---
    with t3:
        st.subheader("Pulse dashboard")
        left, right = st.columns([1,1])

        votes = db.fetch_votes(idea_id)
        dfv = pd.DataFrame([dict(v) for v in votes]) if votes else pd.DataFrame(columns=["vote","confidence"])
        if len(dfv):
            labels = ["Yes","Unsure","No"]
            counts = [
                int((dfv["vote"]==1).sum()),
                int((dfv["vote"]==0).sum()),
                int((dfv["vote"]==-1).sum())
            ]
            with left:
                fig, ax = plt.subplots()
                ax.pie(counts, labels=labels, autopct="%1.0f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

            with right:
                st.metric("Avg. confidence", f"{int(dfv['confidence'].mean())}%")
                st.metric("Total voters", int(len(dfv)))
        else:
            st.info("No votes yet to visualize.")

        st.markdown("### Themes from comments")
        comments = db.fetch_comments(idea_id)
        texts = [c["text"] for c in comments] if comments else []
        if texts:
            if SK_OK:
                vec = TfidfVectorizer(stop_words="english", max_features=50)
                X = vec.fit_transform(texts)
                feats = vec.get_feature_names_out()
                weights = np.asarray(X.sum(axis=0)).ravel()
                top_idx = weights.argsort()[::-1][:15]
                top = [(feats[i], float(weights[i])) for i in top_idx]
            else:
                tokens = " ".join(texts).lower().split()
                cnt = Counter([t for t in tokens if len(t) > 3])
                top = cnt.most_common(15)

            c1, c2 = st.columns([1,1])
            with c1:
                st.write(pd.DataFrame(top, columns=["term","weight"]))
            with c2:
                if WC_OK:
                    wc = WordCloud(width=600, height=300, background_color="white")
                    wc.generate(" ".join(texts))
                    st.image(wc.to_array(), use_column_width=True)
                else:
                    st.info("Install `wordcloud` to see the word cloud.")

        # Random highlight
        st.markdown("### Comment highlight")
        if comments:
            pick = random.choice(comments)
            who = "anonymous" if pick["anonymous"] else pick["author"]
            st.info(f"💬 **{who}**: {pick['text']}")

        # Leaderboard across ideas
        st.markdown("### Idea leaderboard")
        all_ideas = db.list_ideas()
        rows = []
        for it in all_ideas:
            v = db.fetch_votes(it["id"])
            if v:
                dfv2 = pd.DataFrame([dict(x) for x in v])
                score = int((dfv2["vote"]==1).sum() - (dfv2["vote"]==-1).sum())
                rows.append((it["name"], it["key"], int(len(dfv2)), int(dfv2["confidence"].mean()), score))
            else:
                rows.append((it["name"], it["key"], 0, 0, 0))
        if rows:
            lead = pd.DataFrame(rows, columns=["Idea","Key","Voters","AvgConfidence","NetYesMinusNo"]).sort_values("NetYesMinusNo", ascending=False)
            st.dataframe(lead, use_container_width=True)

    # --- Actions ---
    with t4:
        st.subheader("Turn feedback into action items")
        colA, colB, colC = st.columns([2,1,1])
        with colA:
            title = st.text_input("Action title", placeholder="e.g., Run SUC↔SDR sensitivity at 5 levels")
        with colB:
            owner = st.text_input("Owner", placeholder="e.g., Chris")
        with colC:
            due = st.date_input("Due date", value=None)
        if st.button("Add action", type="primary"):
            if title.strip():
                db.add_action(idea_id, title.strip(), owner.strip(), due.isoformat() if isinstance(due, date) else None)
                st.success("Action added.")
                st.rerun()
            else:
                st.warning("Give your action a title.")

        acts = db.fetch_actions(idea_id)
        if acts:
            for a in acts:
                done = st.checkbox(
                    f"✅ {a['title']} — owner: {a['owner'] or '—'} — due: {a['due_date'] or '—'}",
                    value=bool(a["done"]), key=f"act_{a['id']}"
                )
                db.set_action_done(a["id"], done)
        else:
            st.info("No actions yet.")

    # --- History ---
    with t5:
        st.subheader("Version history / notes")
        v1, v2 = st.columns([1,3])
        with v1:
            label = st.text_input("Version label", placeholder="v0.1")
        with v2:
            notes = st.text_area("Notes", placeholder="What changed? Why?")
        if st.button("Add version entry", type="primary"):
            if label.strip():
                db.add_version(idea_id, label.strip(), notes.strip())
                st.success("Logged.")
                st.rerun()
            else:
                st.warning("Provide a label.")

        vers = db.fetch_versions(idea_id)
        if vers:
            st.dataframe(pd.DataFrame([dict(x) for x in vers])[["version_label","notes","created_at"]], use_container_width=True)
        else:
            st.info("No history entries yet.")
else:
    st.title("🗳️ Feedback & Voting")
    st.info("Use the sidebar to create or select an idea.")
