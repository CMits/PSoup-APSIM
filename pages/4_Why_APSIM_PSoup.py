# pages/4_Why_PSoup_APSIM.py
# -------------------------------------------------------------
# Why PSoup ↔ APSIM (Thesis-Style) + REAL LITERATURE UPDATER
# - Thesis-style write-up (no inline bold **)
# - Modernized conceptual figures (clean layout, no text overlap)
# - Real on-demand "Update literature" using Crossref + Europe PMC
# - Voting & comments per paper + general feedback
# - File-based persistence (CSV/JSON) for local use
# -------------------------------------------------------------
import io
import json
import textwrap
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Why PSoup ↔ APSIM + Updater", layout="wide")

APP_DIR = Path(__file__).resolve().parent.parent  # app root (one level up from /pages)
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LIT_CACHE = DATA_DIR / "literature_cache.json"
FEEDBACK_FILE = DATA_DIR / "literature_feedback.csv"
GENERAL_FEEDBACK_FILE = DATA_DIR / "general_feedback.csv"

# --------------------------
# 0) PAGE TITLE
# --------------------------
st.title("Why Integrating PSoup with APSIM Matters (Thesis-Style)")


# --------------------------
# 1) THE THESIS-STYLE TEXT ()
# --------------------------
THESIS_TEXT = textwrap.dedent(r"""
# Why Integrating PSoup with APSIM Matters

## 1. Introduction
A persistent challenge in plant and crop sciences is bridging the gap between molecular-level understanding of developmental processes and field-scale predictions of plant performance. Molecular genetics and physiology have advanced to the point of decoding regulatory networks controlling traits such as shoot branching, while crop models such as APSIM have matured to simulate environment × management interactions with high fidelity. Yet, these domains remain largely disconnected, limiting our capacity to translate mechanistic insights into breeding-relevant predictions.

The integration of PSoup—a mechanistic shoot-branching network—with APSIM—a process-based, widely validated crop and systems simulator—constitutes a novel cross-scale modeling framework. It promises not only to enhance predictive capacity but also to redefine genotype-to-phenotype (G2P) translation in agriculture by carrying causal signals from gene and hormone regulation to tillering and yield.

## 2. Literature Context: What We Know

### 2.1 Shoot branching as a complex regulatory trait
Shoot branching is governed by a dynamic network of auxin, cytokinin (CK), strigolactone (SL), and sugars. These regulators interact in non-linear feedback loops controlling bud activation and suppression (Dun et al. 2023, 2011; Beveridge et al., 2023). 
### 2.2 Crop models as G×E×M engines
Process-based models such as APSIM simulate crop growth under diverse soils, climates, and managements (Holzworth et al., 2014). Tillering frameworks in APSIM connect organ-level decisions to canopy radiation capture and yield, but genotype is too often represented as opaque coefficients, limiting biological interpretability.

### 2.3 CGM–WGP and crop systems biology
The Crop Growth Model × Whole Genome Prediction literature, for example Messina et al. (2018) and Washburn et al. (2023), demonstrates that combining crop models with genomic prediction improves accuracy by injecting environmental structure into WGP and genotype structure into crop growth models. Still, most mappings are statistical, not mechanistic. Reviews in crop systems biology (Muller and Martre, 2019; Yin and Struik, 2010) explicitly call for mechanistic trait modules to improve transportability and understanding.

## 3. Why Integration is Important

### 3.1 From descriptive to causal prediction
Without mechanistic links, predictions of branching plasticity remain empirical fits vulnerable to environmental shifts. Embedding PSoup in APSIM gives tiller number and survival a causal basis grounded in molecular regulation, transforming black-box parameters into transparent biological levers.

### 3.2 Explaining G×E×M interactions
Branching is highly environment-sensitive, including light, nutrients, and water. Integration allows APSIM to reflect how environmental signals modulate hormonal pathways, for example phosphate limitation and strigolactone synthesis, explaining cross-overs among genotypes across seasons and locations (Wang et al., 2019).

### 3.3 Reducing parameter degeneracy
Crop growth models often suffer equifinality, where multiple parameter sets fit the same data. A mechanistic network constrains trait behavior to biologically plausible pathways, improving identifiability of genetic effects.

### 3.4 Driving breeding ideotypes
Simulating alleles or edits, or hormone perturbations, in silico—and projecting them through thousands of historical seasons and managements—enables ideotype design and trial prioritization, addressing a central bottleneck in modern breeding.

## 4. Novelty of This Work
1. First-principles trait integration: an explicit, validated regulatory network drives a systems model, rather than ad-hoc coefficients.  
2. Cross-scale causality: molecular regulation cascades into canopy and yield, closing the G2P gap.  
3. Interoperability with whole-genome prediction: the framework can layer onto genomic prediction, improving both accuracy and interpretability.  
4. Modular template: a pattern other traits, such as root architecture or flowering time, can follow.

## 5. Conceptual Figures (below in this page)
Figure 1 positions PSoup to APSIM in the modeling landscape.  
Figure 2 illustrates the information flow from alleles to yield.

## 6. Scientific Payoffs 
• Evaluate management levers, including sowing date, density, and nitrogen, interacting with branching genetics.  
• Provide a general blueprint for network-to-crop integration.

## 7. Conclusion
PSoup to APSIM is a conceptual advance: it operationalizes crop systems biology, embeds molecular realism into field prediction, and enables breeding-oriented ideotype testing at scale. It bridges plant developmental biology, crop physiology, and applied breeding in a single, testable pipeline.

## References (selected)    
• Holzworth, D. P., et al. 2014. APSIM – Evolution towards a new generation of agricultural systems simulation. Environmental Modelling & Software, 62:327–350.  
• Messina, C. D., et al. 2018. Leveraging biological insight and environmental variation to improve phenotypic prediction. Field Crops Research, 216:46–58.  
• Muller, B., and P. Martre. 2019. Plant and crop simulation models: linking physiology, genetics, phenomics. Journal of Experimental Botany, 70:2339–2344.  
• Paul, M. J., et al. 2021. HEXOKINASE1 signaling promotes shoot branching. New Phytologist, 229:122–136.  
• Wang, E., et al. 2019. Improving process-based crop models to capture G×E×M. Journal of Experimental Botany, 70:2389–2401.  
• Washburn, J. D., et al. 2023. Integrating biophysical crop growth models and whole genome prediction. Journal of Experimental Botany, 74:4415–4426.  
• Yin, X., and P. C. Struik. 2010. Modelling the crop: from system dynamics to systems biology. Journal of Experimental Botany, 61:2171–2183.
""").strip("\n")

st.markdown(THESIS_TEXT)

st.divider()

# --------------------------
# 2) MODERNIZED CONCEPTUAL FIGURES
# --------------------------
st.subheader("Figures")

# ==== FIGURE 1 (updated: colored dots + legend, no arrows/overlap) ====
import io
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
ax1.set_xlabel("Biological mechanism and interpretability")
ax1.set_ylabel("Scalability to field and years")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.grid(alpha=0.25, linestyle="--")

# points: (x, y, label, color)
points = [
    (2.2, 3.2, "Molecular networks (PSoup)", "tab:blue"),
    (4.0, 6.0, "Process-based CGMs (APSIM)", "tab:orange"),
    (7, 7.0, "Integrated PSoup → APSIM (this work)", "tab:green"),
]

for x, y, label, color in points:
    ax1.scatter(x, y, s=180, color=color, edgecolor="black", linewidth=1.0, label=label)

# legend in top-left corner
ax1.legend(loc="upper left", frameon=True, fontsize=10)

buf1 = io.BytesIO()
fig1.savefig(buf1, format="png")
st.image(buf1.getvalue(),
         caption="Figure 1. Position of PSoup to APSIM in the modeling landscape.",
         use_column_width=False)



# ==== FIGURE 2 (drop-in replacement) ====
import io
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

fig2, ax2 = plt.subplots(figsize=(6.2, 6.8), constrained_layout=True)
ax2.axis("off")

# vertical node stack (centered)
nodes = [
    (0.5, 0.90, "Genotype"),
    (0.5, 0.68, "PSoup\nMolecular network:\nauxin, cytokinin, strigolactone, sucrose"),
    (0.5, 0.46, "Trait signals to crop module\nfor example tillering"),
    (0.5, 0.24, "APSIM\nG×E×M to yield"),
]

box = dict(boxstyle="round,pad=0.5,rounding_size=0.2",
           facecolor="white", alpha=0.97, ec="#444444", lw=0.9)

for (x, y, txt) in nodes:
    ax2.text(x, y, txt, ha="center", va="center", fontsize=11, bbox=box,
             path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
             transform=ax2.transAxes)

# arrows between stacked nodes (slight vertical spacing; straight down)
def down_arrow(y0, y1, x=0.5):
    ax2.annotate("",
                 xy=(x, y1 + 0.03), xytext=(x, y0 - 0.03),
                 xycoords=ax2.transAxes, textcoords=ax2.transAxes,
                 arrowprops=dict(arrowstyle="->", lw=1.3, alpha=0.9))

down_arrow(0.90, 0.68)  # genotype -> PSoup
down_arrow(0.68, 0.46)  # PSoup -> trait signals
down_arrow(0.46, 0.24)  # trait signals -> APSIM

buf2 = io.BytesIO()
fig2.savefig(buf2, format="png")
st.image(buf2.getvalue(),
         caption="Figure 2. Vertical information flow from genotype to yield via PSoup signals and APSIM.",
         use_column_width=True)

st.divider()

# --------------------------
# 3) LITERATURE UPDATER (REAL, ON-DEMAND) — unchanged logic
# --------------------------
st.subheader("Update literature (real, on-demand)")

with st.expander("About this updater"):
    st.markdown(
        "- This runs only when you click. It calls Crossref and Europe PMC with your keywords and year range, "
        "merges and de-dup es, and shows the latest items.\n"
        "- It stores a local cache in data/literature_cache.json. You can clear it anytime.\n"
        "- For Streamlit Cloud, swap the file cache for a database or Google Sheet if you need multi-user persistence."
    )

default_keywords = [
    "shoot branching", "strigolactone", "cytokinin", "auxin",
    "sucrose HXK1", "apical dominance",
    "APSIM", "crop growth model", "CGM WGP", "genotype to phenotype"
]
colk1, colk2 = st.columns([2,1])
with colk1:
    keywords = st.text_input("Keywords (comma-separated)", ", ".join(default_keywords))
with colk2:
    from_year = st.number_input("From year", min_value=2000, max_value=date.today().year, value=2015, step=1)

rows = st.slider("Max results per source (per click)", 10, 100, 30, 10)

def fetch_crossref(query_terms, from_year, rows):
    q = " ".join(query_terms)
    url = "https://api.crossref.org/works"
    params = {
        "query": q,
        "filter": f"from-pub-date:{from_year}-01-01",
        "rows": rows,
        "sort": "published",
        "order": "desc"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    recs = []
    for it in items:
        doi = it.get("DOI")
        title = " ".join(it.get("title") or []) if it.get("title") else None
        year = None
        try:
            ylist = it.get("issued", {}).get("date-parts", [[None]])[0]
            year = ylist[0]
        except Exception:
            pass
        journal = (it.get("container-title") or [None])[0]
        authors = []
        for a in it.get("author", []) or []:
            name = " ".join([x for x in [a.get("given"), a.get("family")] if x])
            if name:
                authors.append(name)
        recs.append({
            "source": "Crossref",
            "title": title,
            "year": year,
            "journal": journal,
            "doi": doi,
            "pmid": None,
            "url": f"https://doi.org/{doi}" if doi else (it.get("URL") or None),
            "authors": ", ".join(authors),
            "query": q
        })
    return pd.DataFrame(recs)

def fetch_eupmc(query_terms, from_year, page_size):
    q = " ".join(query_terms)
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": f'({q}) AND PUB_YEAR:{from_year}-{date.today().year}',
        "format": "json",
        "resultType": "lite",
        "pageSize": page_size,
        "sort": "P_PDATE_D"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    result_list = r.json().get("resultList", {}).get("result", []) or []
    recs = []
    for it in result_list:
        title = it.get("title")
        year = int(it.get("pubYear")) if it.get("pubYear") else None
        journal = it.get("journalTitle")
        pmid = it.get("pmid")
        doi = it.get("doi")
        authors = it.get("authorString")
        url_list = it.get("fullTextUrlList", {}).get("fullTextUrl", [])
        best = None
        if url_list:
            best = url_list[0].get("url")
        if not best and doi:
            best = f"https://doi.org/{doi}"
        recs.append({
            "source": "EuropePMC",
            "title": title,
            "year": year,
            "journal": journal,
            "doi": doi,
            "pmid": pmid,
            "url": best,
            "authors": authors,
            "query": q
        })
    return pd.DataFrame(recs)

def merge_and_dedupe(df_list):
    df = pd.concat([d for d in df_list if d is not None and len(d) > 0], ignore_index=True)
    if "doi" in df.columns:
        df = df.sort_values(["doi", "year"], ascending=[True, False]).drop_duplicates(subset=["doi"], keep="first")
    if "pmid" in df.columns:
        df = df.sort_values(["pmid", "year"], ascending=[True, False]).drop_duplicates(subset=["pmid"], keep="first")
    df = df.sort_values(["title", "year"], ascending=[True, False]).drop_duplicates(subset=["title"], keep="first")
    df = df.sort_values(["year"], ascending=[False]).reset_index(drop=True)
    return df

def load_cache():
    if LIT_CACHE.exists():
        try:
            return pd.DataFrame(json.loads(LIT_CACHE.read_text(encoding="utf-8")))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(df):
    LIT_CACHE.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

colu1, colu2, colu3 = st.columns([1,1,1])
with colu1:
    run_update = st.button("Update now")
with colu2:
    clear_cache = st.button("Clear cache")
with colu3:
    show_cache = st.checkbox("Show cached results only", value=False)

if clear_cache and LIT_CACHE.exists():
    LIT_CACHE.unlink(missing_ok=True)
    st.success("Cache cleared.")

query_terms = [k.strip() for k in (keywords or "").split(",") if k.strip()]
cache_df = load_cache()

if run_update:
    with st.spinner("Fetching latest from Crossref and Europe PMC..."):
        try:
            cr = fetch_crossref(query_terms, int(from_year), int(rows))
        except Exception as e:
            st.error(f"Crossref error: {e}")
            cr = pd.DataFrame()
        try:
            ep = fetch_eupmc(query_terms, int(from_year), int(rows))
        except Exception as e:
            st.error(f"Europe PMC error: {e}")
            ep = pd.DataFrame()
        fresh = merge_and_dedupe([cr, ep])
        if len(fresh) == 0:
            st.warning("No results found. Try adjusting keywords or year.")
        else:
            combined = merge_and_dedupe([cache_df, fresh])
            save_cache(combined)
            cache_df = combined
            st.success(f"Updated. Total cached records: {len(cache_df)}")

display_df = cache_df if (show_cache or run_update) else cache_df
st.markdown(f"Showing {len(display_df)} records (cached in data/literature_cache.json).")

# --------------------------
# 4) LITERATURE LIST + VOTING & COMMENTS — unchanged
# --------------------------
def load_feedback():
    if FEEDBACK_FILE.exists():
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["timestamp", "paper_id", "vote", "comment"])

def save_feedback(df):
    df.to_csv(FEEDBACK_FILE, index=False)

feedback_df = load_feedback()

if len(display_df) > 0:
    for i, row in display_df.iterrows():
        paper_id = row.get("doi") or row.get("pmid") or f"{row.get('title','')[:40]}_{row.get('year')}"
        st.markdown("---")
        st.markdown(f"{row.get('title','(No title)')}")
        meta_line = []
        if row.get("authors"): meta_line.append(f"{row['authors']}")
        if row.get("journal"): meta_line.append(row["journal"])
        if row.get("year"): meta_line.append(str(int(row["year"])))
        st.caption(" • ".join(meta_line))
        if row.get("url"):
            st.link_button("Open", row["url"])
        elif row.get("doi"):
            st.link_button("Open DOI", f"https://doi.org/{row['doi']}")

        c1, c2, c3 = st.columns([1,1,3])
        with c1:
            if st.button("👍", key=f"up_{i}"):
                feedback_df.loc[len(feedback_df)] = [datetime.now().isoformat(), paper_id, "up", ""]
                save_feedback(feedback_df)
                st.toast("Upvoted")
        with c2:
            if st.button("👎", key=f"down_{i}"):
                feedback_df.loc[len(feedback_df)] = [datetime.now().isoformat(), paper_id, "down", ""]
                save_feedback(feedback_df)
                st.toast("Downvoted")
        with c3:
            comment = st.text_input("Add a comment (press Enter to save)", key=f"cm_{i}", placeholder="Your note for this paper…")
            if comment:
                feedback_df.loc[len(feedback_df)] = [datetime.now().isoformat(), paper_id, "comment", comment]
                save_feedback(feedback_df)
                st.toast("Comment saved")

    st.markdown("Voting summary")
    if len(feedback_df) > 0:
        summary = (feedback_df[feedback_df["vote"].isin(["up","down"])]
                   .groupby(["paper_id","vote"]).size().unstack(fill_value=0))
        st.dataframe(summary, use_container_width=True)
    else:
        st.caption("No votes yet.")
else:
    st.info("No literature cached yet. Click Update now above.")

st.divider()

# --------------------------
# 5) GENERAL FEEDBACK (tab-level) — unchanged
# --------------------------
st.subheader("General feedback for this tab")
st.caption("Supervisors and collaborators can leave overall comments here, saved to data/general_feedback.csv.")

def load_general_feedback():
    if GENERAL_FEEDBACK_FILE.exists():
        return pd.read_csv(GENERAL_FEEDBACK_FILE)
    return pd.DataFrame(columns=["timestamp","name","comment"])

def save_general_feedback(df):
    df.to_csv(GENERAL_FEEDBACK_FILE, index=False)

gfb = load_general_feedback()
with st.form("general_fb_form", clear_on_submit=True):
    name = st.text_input("Name (optional)")
    gcomment = st.text_area("Comment")
    submitted = st.form_submit_button("Save")
    if submitted and gcomment.strip():
        gfb.loc[len(gfb)] = [datetime.now().isoformat(), name.strip(), gcomment.strip()]
        save_general_feedback(gfb)
        st.success("Thanks, feedback saved.")

if len(gfb) > 0:
    st.markdown("Previous feedback")
    st.dataframe(gfb.sort_values("timestamp", ascending=False), use_container_width=True, height=240)

st.divider()

# --------------------------
# 6) DOWNLOAD THE WRITE-UP — unchanged
# --------------------------
st.subheader("Download the thesis-style write-up")
st.download_button(
    label="Download as Markdown",
    data=THESIS_TEXT.encode("utf-8"),
    file_name=f"Why_PSoup_APSIM_{date.today().isoformat()}.md",
    mime="text/markdown"
)

with st.expander("Tips: persistence and deployment"):
    st.markdown("""
- Local use: votes, comments, and literature cache are saved under data/ next to your app.
- Streamlit Cloud: replace file writes with a small database or Google Sheet:
  use st.secrets to store credentials, then write to Supabase, SQLite, PostgreSQL, or Google Sheets.
- Keep the on-demand button; avoid background polling.
- If requests is missing, pip install requests.
""")
