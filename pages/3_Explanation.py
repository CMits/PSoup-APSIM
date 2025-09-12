
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Explanation", page_icon="📘", layout="wide")
st.title("📘 APSIM ↔ PSoup: End-to-End Explanation")

st.caption(
    "This page documents the full pipeline used in the app: "
    "APSIM S/D → SUC → PSoup (Sustained Growth) → direction (Higher/Same/Lower) → TTN in APSIM."
)

tabs = st.tabs([
    "Overview",
    "Step 1: SDR → SUC",
    "Step 2: PSoup decision (direction only)",
    "Step 3: TTN envelopes from APSIM",
    "Step 4: Map direction → TTN (Scheme A & B)",
    "Excel format & required columns",
    "Worked examples",
    "Assumptions, limitations, QA"
])

# ----------------------- OVERVIEW -----------------------
with tabs[0]:
    st.header("Overview")
    st.markdown("""
**Goal:** connect quantitative APSIM outputs with qualitative PSoup outputs in a **robust, auditable** way.

**High-level flow**
1) **APSIM** produces **S/D (Supply/Demand)** per environment/year.
2) We map S/D to **SUC (0..2)** — the sucrose knob used by **PSoup**.
3) **PSoup** returns **Sustained Growth (SG)**. We **do not trust magnitudes**, only **direction** when sucrose is added:
   - “mutant + sucrose” vs “mutant” → **Higher / Same / Lower**.
4) We translate this **direction** into **APSIM TTN (tillers)**, using the **environment bounds** learned from APSIM:
   - **Scheme A:** place a fraction **z** inside the [TTN_low, TTN_high] interval.
   - **Scheme B:** snap to **cultivar lines** (Low / Medium / High) from literature (Alam et al., 2014).

Everything you see in the calculator and batch pages is a concrete implementation of these steps.
""")

# ----------------------- STEP 1 -----------------------
with tabs[1]:
    st.header("Step 1: Map S/D → SUC (0..2)")
    st.markdown("""
**Why:** PSoup expects a sucrose input (SUC). APSIM gives S/D. We need a monotone, bounded mapping.

### Linear rule (default)
- Simple, transparent, and easy to explain.
- Example with L=0, U=50, out_min=0, out_max=2:
  - x=0  → SUC=0
  - x=25 → SUC=1
  - x=50 → SUC=2

### Smooth (tanh) rule (optional)
- Same limits (≈0..2), smoother around the anchor `center`, `k` controls steepness.
- Useful when S/D is noisy and you want less jumpy SUC near the WT anchor.

**In the Excel batch page:** the **global median S/D** of the upload can be used as the tanh anchor `center` to stay robust across years.
""")

# ----------------------- STEP 2 -----------------------
with tabs[2]:
    st.header("Step 2: PSoup decision (direction only)")
    st.markdown("""
**Why:** We only trust the **sign** of the change in PSoup's Sustained Growth (SG), not its magnitude.

For each row (same genotype, same environment):
- `τ` (tau) is a small tolerance to suppress numerical noise (e.g., 0.001 or 1% of baseline).

**Replicates (optional):**
If you have multiple PSoup comparisons:
We still make a **directional** decision (majority), but `p` can be used (optionally) to fine-tune placement in Scheme A.
""")

# ----------------------- STEP 3 -----------------------
with tabs[3]:
    st.header("Step 3: Environment envelopes for TTN (from APSIM)")
    st.markdown("""
**Why:** Tillering capacity depends on environment. We use APSIM-based lines to define plausible TTN bounds as a
function of S/D (x).

**Envelope extremes (Very low / Very high branching):**

**Intermediate cultivar lines (Alam et al., 2014):**
| Branching cultivar | Intercept | Slope |
|---|---:|---:|
| Very low  | -1.363 | 0.159 |
| Low       | -0.818 | 0.206 |
| Medium    | -0.265 | 0.247 |
| High      | +0.329 | 0.300 |
| Very high | +0.960 | 0.347 |

These are near-parallel, so a single fraction **z** describes how far you sit between **TTN_low(x)** and **TTN_high(x)**.
""")

# ----------------------- STEP 4 -----------------------
with tabs[4]:
    st.header("Step 4: Map PSoup direction → APSIM TTN")
    st.markdown("""
We never use SG magnitudes — only the **direction** (Higher/Same/Lower). Two robust schemes:

---

## Scheme A — Continuous bins inside the envelope (uses **z**)
1) Pick **z** values for the labels (kept away from 0 and 1 to avoid saturation). Examples:

**Intermediate cultivar lines (Alam et al., 2014):**
| Branching cultivar | Intercept | Slope |
|---|---:|---:|
| Very low  | -1.363 | 0.159 |
| Low       | -0.818 | 0.206 |
| Medium    | -0.265 | 0.247 |
| High      | +0.329 | 0.300 |
| Very high | +0.960 | 0.347 |

These are near-parallel, so a single fraction **z** describes how far you sit between **TTN_low(x)** and **TTN_high(x)**.
""")

# ----------------------- STEP 4 -----------------------
with tabs[4]:
    st.header("Step 4: Map PSoup direction → APSIM TTN")
    st.markdown("""
We never use SG magnitudes — only the **direction** (Higher/Same/Lower). Two robust schemes:

---

## Scheme A — Continuous bins inside the envelope (uses **z**)
1) Pick **z** values for the labels (kept away from 0 and 1 to avoid saturation). Examples:
2) Compute **TTN** for S/D = x:
3) Optional (cultivar-matched z): derive z for Low/Medium/High at a reference x_ref:
   Use these three z's as your bins to emulate literature spacing.

4) Optional (replicate-aware band): with vote share p from Step 2, move z inside a band:

---

## Scheme B — Snap to cultivar lines (discrete)
Map the label directly to a line (editable in the app):
This ties predictions to published branching levels and is easy to defend in reviews.

---

**Global cap (optional):** if you want TTN strictly in [0,6], clamp the envelope or final TTN:
""")

# ----------------------- EXCEL FORMAT -----------------------
with tabs[5]:
    st.header("Excel format & required columns")
    st.markdown("""
**You can upload an .xlsx with a sheet like this (headers are case-sensitive):**

Required (S/D):
- `S/D_APSIM` **or** `SD` **or** `SDR`  → numeric Supply/Demand from APSIM.

Required (PSoup SG):
- `SG_mut`       → PSoup Sustained Growth (mutant, no sucrose)
- `SG_mut_SUC`   → PSoup Sustained Growth (mutant + sucrose)

Optional (recommended):
- `h`, `l`       → replicate counts of Higher/Lower (direction only)
- `Date`         → e.g. `15/01/2007` (day-first), used to derive `Year`
- `Years`        → e.g. `2007-01-15` (ISO), used if `Date` is missing
- `Genotype`     → label for your own reference (not required by the app)

The batch page:
1) Computes **Label** using the tolerance `τ`.
2) Builds **TTN_low/TTN_high** from S/D (or uses a global range).
3) Applies **Scheme A (z)** or **Scheme B (cultivar lines)** to get **TTN_pred**.
4) Shows per-year plots and lets you download the rows and parameters.
""")

# ----------------------- WORKED EXAMPLES -----------------------
with tabs[6]:
    st.header("Worked examples")
    st.markdown("""
### Example A — Scheme A (z inside the envelope)
Given `x = 30`:
If PSoup says **Higher** and you use `z(Higher)=0.65`:


### Example B — Scheme B (snap to cultivar lines)
At `x = 30`:
If PSoup says **Higher**, pick the **High** line → `TTN ≈ 9.329`.

### Example C — Replicate-aware band (direction only)
Suppose `h = 2` Higher and `l = 0` Lower → vote share
With a band `z_min=0.35`, `z_max=0.65`:
Then use Scheme A formula `TTN = L + z*(H-L)` at the environment’s S/D.
""")

# ----------------------- ASSUMPTIONS & QA -----------------------
with tabs[7]:
    st.header("Assumptions, limitations, QA")
    st.markdown("""
**Assumptions**
- **Monotonicity:** More favorable PSoup direction (Higher) never reduces TTN relative to Same/Lower.
- **Direction-only:** We do not use SG magnitudes; only the sign of change matters.
- **Envelope validity:** APSIM lines approximate min/max tillering for a given S/D; cultivar lines sit between them.
- **Noise tolerance:** A small `τ` avoids classifying tiny numerical wiggles as real changes.

**When to use which scheme**
- **Scheme A** (z inside envelope): smooth, adjustable spacing; great for sensitivity and presentation.
- **Scheme B** (cultivar lines): ties predictions to published branching levels; excellent for defensibility.

**Quality checks you can run**
- Verify that for any fixed environment `x`, the order **Lower < Same < Higher** holds in TTN.
- Sensitivity to `z` choices (Scheme A): try 0.30/0.50/0.70 vs 0.35/0.50/0.65; results should shift sensibly.
- If using the tanh SDR→SUC map in Step 1, check that the chosen `center` is reasonable (e.g., global median S/D).
- If clamping to [0,6], confirm that the envelope is clipped consistently across years.

**Limitations**
- Without real TTN observations for calibration, the absolute positions within the envelope (or the chosen cultivar line)
  are **assumptions**. We surface and document those assumptions explicitly to keep the workflow auditable.
""")

st.info(
    "You can now switch to the other pages to test these steps interactively: "
    "Overview → SDR→SUC → PSoup→TTN. This page is your reference guide."
)
