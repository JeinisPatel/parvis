"""
PARVIS — Streamlit Application v Xavier 7
Jeinis Patel, PhD Candidate and Barrister | University of London | Ethical AI Initiative
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import base64, os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model import build_model, get_inference_engine, query_do_risk, NODE_META, EDGES_VE as EDGES
from quantum_diagnostics import diagnose, format_report
from bloch_sphere import draw_bloch_sphere, draw_comparison_chart

st.set_page_config(page_title="P.A.R.V.I.S — Bayesian Sentencing Network", layout="wide", initial_sidebar_state="collapsed",
    menu_items={"About":"PARVIS Xavier 7 — Research use only"})

@st.cache_data
def get_logo_b64():
    for p in ["ethical_ai_logo.png","parvis/ethical_ai_logo.png"]:
        if os.path.exists(p):
            with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
    return None

logo_b64 = get_logo_b64()
# Watermark: injected fixed div. mix-blend-mode:multiply dissolves the black PNG background.
wm = f"""
<style>
#parvis-watermark {{
  position: fixed;
  bottom: 28px;
  right: 28px;
  width: 110px;
  height: 110px;
  background-image: url('data:image/png;base64,{logo_b64}');
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  opacity: 0.13;
  pointer-events: none;
  z-index: 9999;
  mix-blend-mode: multiply;
}}
</style>
<div id="parvis-watermark"></div>
""" if logo_b64 else ""

st.markdown(wm + """
<style>
/* ── DM Sans font — remove this block to revert to system font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');
/* Apply DM Sans to body — inheritance handles the rest */
body { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }
/* Key Streamlit containers */
.stApp, .stMarkdown, .stText, .element-container,
p, h1, h2, h3, h4, h5, label, select, input, textarea {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
/* Hide the SVG upload icon whose <title>upload</title> renders as visible text */
[data-testid="stFileUploaderDropzone"] button svg {
    display: none !important;
}
/* Clean up button font */
[data-testid="stFileUploaderDropzone"] button {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    letter-spacing: normal !important;
}
/* ── End DM Sans block ── */
/* ── File uploader button fix ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span {
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    overflow: visible !important;
    letter-spacing: 0 !important;
}
[data-testid="stFileUploaderDropzone"] small {
    font-size: 0.75rem !important;
}
.pt{font-size:2.4rem;font-weight:800;letter-spacing:7px;margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.ps{font-size:.88rem;color:#777;margin-top:5px;letter-spacing:.3px;line-height:1.5}
/* ── Evidence review — thick colored sliders ── */
div[data-testid="stSlider"] > div > div > div > div {
    height:10px !important;border-radius:6px !important;cursor:pointer;
    transition:all .15s ease;
}
div[data-testid="stSlider"] > div > div > div > div > div {
    height:22px !important;width:22px !important;border-radius:50% !important;
    box-shadow:0 2px 6px rgba(0,0,0,0.22) !important;
    top:-6px !important;cursor:grab !important;border:2px solid white !important;
    transition:transform .1s ease,box-shadow .1s ease;
}
div[data-testid="stSlider"] > div > div > div > div > div:hover {
    transform:scale(1.18) !important;
    box-shadow:0 4px 12px rgba(0,0,0,0.30) !important;
}
/* Per-node slider colors by key */
div[data-testid="stSlider"][aria-label="N2 — Violent history"] > div > div > div > div { background:#A32D2D !important; }
div[data-testid="stSlider"][aria-label="N3 — Psychopathy (PCL-R)"] > div > div > div > div { background:#A32D2D !important; }
div[data-testid="stSlider"][aria-label="N4 — Sexual offence"] > div > div > div > div { background:#A32D2D !important; }
div[data-testid="stSlider"][aria-label="N18 — Dynamic risk"] > div > div > div > div { background:#A32D2D !important; }
div[data-testid="stSlider"][aria-label="N5 — Invalid risk tools"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N6 — Ineffective counsel"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N7 — Bail-denial cascade"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N9 — FASD"] > div > div > div > div { background:#534AB7 !important; }
div[data-testid="stSlider"][aria-label="N10 — Intergenerational trauma"] > div > div > div > div { background:#3B6D11 !important; }
div[data-testid="stSlider"][aria-label="N11 — No cultural treatment"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N12 — Gladue misapplication"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N13 — Gaming risk"] > div > div > div > div { background:#0F6E56 !important; }
div[data-testid="stSlider"][aria-label="N14 — Over-policing"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N15 — Temporal distortion"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N19 — No rehabilitation"] > div > div > div > div { background:#185FA5 !important; }
.dc{border-radius:14px;padding:.9rem 1.1rem;text-align:center}
.dp{font-size:2.4rem;font-weight:700;font-family:monospace;line-height:1}
.dl{font-size:.72rem;margin-bottom:3px}
.db{font-size:.82rem;font-weight:600;margin-top:3px}
.sh{font-size:.68rem;font-weight:700;color:#aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem}
.qh{padding:.5rem .75rem;border-radius:6px;margin:.3rem 0;font-size:.85rem}
.at{font-family:'Courier New',monospace;font-size:.78rem;line-height:1.75;
    background:#f7f6f3;border-radius:10px;padding:1.2rem;border:1px solid #e0dfd9;white-space:pre-wrap}
/* Tab styling — larger, more breathing room */
.stTabs [data-baseweb="tab-list"] {gap:4px}
.stTabs [data-baseweb="tab"] {
  font-size:0.88rem !important;
  font-weight:500 !important;
  padding:10px 18px !important;
  letter-spacing:0.2px;
}
.stTabs [aria-selected="true"] {font-weight:700 !important}
footer{visibility:hidden}#MainMenu{visibility:hidden}

/* ── Thicker active tab indicator ── */
.stTabs [aria-selected="true"] {
  border-bottom: 3px solid #1B2A4A !important;
  color: #1B2A4A !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: #1B2A4A !important;
  background: rgba(27,42,74,0.04) !important;
  border-radius: 6px 6px 0 0 !important;
}

/* ── Warmer card/input borders ── */
[data-testid="stSelectbox"] > div > div {
  border-color: #C8C4BC !important;
  border-radius: 8px !important;
}
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextArea"] > div > div > textarea {
  border-color: #C8C4BC !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] {
  border-color: #D8D4CC !important;
  border-radius: 10px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: #FDFCFA !important;
  border: 1px solid #E0DDD6 !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button {
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* ── PARVIS footer style ── */
.parvis-footer {
  text-align: center;
  padding: 1.2rem 0 0.4rem 0;
  font-size: 0.73rem;
  color: #BBBBBB;
  letter-spacing: 0.4px;
  border-top: 1px solid #ECEAE4;
  margin-top: 1.5rem;
}
</style>""", unsafe_allow_html=True)

TC = {"constraint":"#BA7517","risk":"#A32D2D","distortion":"#185FA5",
      "mitigation":"#3B6D11","dual":"#534AB7","special":"#0F6E56","output":"#993C1D"}
TL = {"constraint":"Evidentiary constraint","risk":"Risk factor","distortion":"Systemic distortion",
      "mitigation":"Mitigating factor","dual":"Dual factor","special":"Causal detector","output":"Structural output"}

def rb(p):
    if p<.20: return "Very low","#3B6D11","#EAF3DE"
    if p<.40: return "Low","#3B6D11","#EAF3DE"
    if p<.55: return "Moderate","#BA7517","#FAEEDA"
    if p<.70: return "Elevated","#BA7517","#FAEEDA"
    if p<.85: return "High","#A32D2D","#FCEBEB"
    return "Very high","#A32D2D","#FCEBEB"

def dobar(p, label="DO designation risk", show_cr=False):
    bl,bc,bg = rb(p)
    # Build criminal record contribution badge if requested
    cr_badge = ""
    if show_cr and "criminal_record" in st.session_state:
        rec = st.session_state.criminal_record
        if rec:
            cal_weights = [e["cal_weight"] for e in rec]
            mean_cal = float(np.mean(cal_weights))
            n_conv   = len(rec)
            esc_pat  = st.session_state.get("cr_doc_adj",{}).get("escalation",{}).get("pattern","")
            esc_icon = {"escalating":"⚠️","de-escalating":"✅","desistance":"✅","stable":"—"}.get(esc_pat,"")
            cr_col   = "#A32D2D" if mean_cal < 0.5 else "#BA7517" if mean_cal < 0.7 else "#3B6D11"
            cr_badge = (
                f"<div style='margin-top:6px;padding:4px 10px;"
                f"background:rgba(0,0,0,.04);border-radius:6px;font-size:.72rem'>"
                f"<span style='color:#888'>Criminal record:</span> "
                f"<span style='color:{cr_col};font-weight:600'>{n_conv} conviction(s) · "
                f"mean calibrated weight {mean_cal*100:.0f}%</span>"
                f"{f'  ·  <span style="color:#555">{esc_icon} {esc_pat.title()} pattern (Boutilier)</span>' if esc_pat else ''}"
                f"</div>"
            )
    return f"""<div style="background:{bg};border:1px solid {bc}44;border-radius:12px;
    padding:.8rem 1.2rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem">
    <div style="text-align:center;min-width:80px">
      <div style="font-size:.7rem;color:{bc};margin-bottom:2px">Node 20</div>
      <div style="font-size:2rem;font-weight:700;font-family:monospace;color:{bc}">{p*100:.1f}%</div>
      <div style="font-size:.8rem;font-weight:600;color:{bc}">{bl}</div>
    </div>
    <div style="flex:1">
      <div style="font-size:.82rem;font-weight:500;margin-bottom:6px">{label} — posterior probability</div>
      <div style="height:5px;background:rgba(0,0,0,.08);border-radius:3px">
        <div style="width:{p*100:.0f}%;height:100%;background:{bc};border-radius:3px"></div>
      </div>
      {cr_badge}</div></div>"""


# ── Summary tab helpers (Mark 8) ──────────────────────────────────────────────
def _summary_band(p):
    """Return (label, fg, bg) styled for the Summary headline card."""
    if p < .20: return "Very low", "#3B6D11", "#EAF3DE"
    if p < .40: return "Low",      "#3B6D11", "#EAF3DE"
    if p < .55: return "Moderate", "#BA7517", "#FAEEDA"
    if p < .70: return "Elevated", "#BA7517", "#FAEEDA"
    if p < .85: return "High",     "#A32D2D", "#FCEBEB"
    return "Very high", "#A32D2D", "#FCEBEB"

def _top_drivers(P, k=5):
    """
    Return (up_drivers, down_drivers) — each a list of dicts:
        {nid:int, short:str, type_:str, type_label:str, color:str, p:float}

    Doctrinally-typed split (Option B):
      ↑ increasing  : risk-typed nodes  +  duals with P>=0.5
      ↓ decreasing  : mitigation-typed  +  distortion-typed (corrections)
                      +  causal/special detectors  +  duals with P<0.5
    Within each list, ranked by P(High) descending. Output (N20) excluded.
    """
    up, dn = [], []
    for nid in range(1, 20):  # exclude N20 (output)
        meta = NODE_META.get(nid, {})
        t = meta.get("type", "")
        p = float(P.get(nid, 0.0))
        entry = {
            "nid":  nid,
            "short": meta.get("short", f"N{nid}"),
            "type_": t,
            "type_label": TL.get(t, t.title()),
            "color": TC.get(t, "#888"),
            "p":     p,
        }
        if t == "risk":
            up.append(entry)
        elif t in ("mitigation", "distortion", "special"):
            dn.append(entry)
        elif t == "dual":
            (up if p >= 0.5 else dn).append(entry)
        elif t == "constraint":
            # Constraints are evidentiary anchors — high posterior pushes structure up.
            up.append(entry)
        else:
            (up if p >= 0.5 else dn).append(entry)
    up.sort(key=lambda e: e["p"], reverse=True)
    dn.sort(key=lambda e: e["p"], reverse=True)
    return up[:k], dn[:k]

# ── Node 7 Reliability Modifier (Chapter 5 §5.1.7) ───────────────────────────
# Per JP's specification (confirmed): each conviction is assigned an ordinal
# grade from N7's posterior + the conviction's own bail-denial flag.
# This implements the mechanism §5.1.7 describes verbatim:
#   "Criminal Record Reliability Modifier — Adjustment applied to prior
#    convictions — Unmodified / Discounted / Heavily Discounted"
# The multipliers below are conservative; they do NOT modify cal_weight,
# they sit alongside it as an explicit N7-specific adjustment.

# Multiplier scheme (per-conviction reliability grade → cal_weight multiplier)
# Confirmed by JP, Apr 27 2026; values aligned with §RM.3.
N7_MULTIPLIERS = {
    "Unmodified":         1.00,   # full weight
    "Discounted":         0.60,   # 40% reduction
    "Heavily Discounted": 0.30,   # 70% reduction
}

# Propagation factor scheme (§RM.5.4): when a downstream conviction has its
# own affirmative bail-denial signal AND a prior conviction on the same
# record has been graded as Discounted or Heavily Discounted, the per-conviction
# bail-denial signal on the downstream conviction is multiplied by the
# propagation factor before threshold logic. The architecture takes the
# strongest upstream propagation factor; multiple tainted upstream
# convictions do not compound.
N7_PROPAGATION_FACTOR = {
    "Unmodified":         1.00,   # no propagation
    "Discounted":         1.15,   # +15% boost to downstream bail signal
    "Heavily Discounted": 1.30,   # +30% boost
}

# Jump-principle weight scheme (§RM.5.5): each conviction's contribution
# to the jump principle's own_ceiling is weighted by its reliability grade.
# A Heavily Discounted conviction contributes at 30% of nominal; the
# architecture treats most of its sentence inflation as cascade
# contamination rather than legitimate severity.
JUMP_WEIGHT_BY_GRADE = {
    "Unmodified":         1.00,   # full anchoring weight
    "Discounted":         0.60,   # 40% reduction (alignment with N7_MULTIPLIERS)
    "Heavily Discounted": 0.30,   # 70% reduction
}


def _n7_threshold_grade(combined_indicator):
    """
    Apply the §5.1.7 threshold logic to a combined indicator value
    (after any propagation factor has been applied) and return the
    ordinal grade.
    """
    if combined_indicator < 0.30:
        return "Unmodified"
    elif combined_indicator <= 0.65:
        return "Discounted"
    else:
        return "Heavily Discounted"


def _n7_grades_chronological(criminal_record_chronological):
    """
    Compute per-conviction grades in chronological order, applying the
    cascade-propagation factor where applicable.

    Per §RM.5.4: a conviction's per-conviction bail-denial signal is
    multiplied by the strongest propagation factor from earlier graded
    tainted convictions on the same record. Propagation requires:
      (a) the downstream conviction has its own affirmative bail-denial
          signal (adj.bail > 0)
      (b) at least one earlier conviction is graded Discounted or
          Heavily Discounted

    Returns: list of (grade, multiplier, propagation_factor_applied) tuples
    in the same order as the input record.
    """
    if not criminal_record_chronological:
        return []

    results = []
    earlier_grades = []  # Accumulates as we walk the record forward

    for e in criminal_record_chronological:
        per_conv_bail = float(e.get("adj", {}).get("bail", 0.0))

        # Determine propagation factor: strongest of any earlier tainted
        # conviction. If no earlier tainted convictions, factor = 1.00.
        if per_conv_bail > 0.0 and earlier_grades:
            applicable_factors = [
                N7_PROPAGATION_FACTOR[g] for g in earlier_grades
                if g in ("Discounted", "Heavily Discounted")
            ]
            propagation = max(applicable_factors) if applicable_factors else 1.00
        else:
            # If conviction has no own bail-denial signal, propagation
            # cannot apply (per §RM.5.3: the cascade chain requires actual
            # pre-trial detention on the downstream conviction).
            propagation = 1.00

        # Apply propagation factor to per-conviction signal
        boosted_signal = per_conv_bail * propagation

        # Apply threshold logic
        grade = _n7_threshold_grade(boosted_signal)
        multiplier = N7_MULTIPLIERS[grade]

        results.append((grade, multiplier, propagation))
        earlier_grades.append(grade)

    return results


def _n7_grade_for_conviction(conviction, n7_posterior=None):
    """
    Backward-compat single-conviction grading.

    DEPRECATED for record-level grading: use _n7_grades_chronological(),
    which properly accounts for propagation. This function is retained
    only for callers that grade a single conviction in isolation
    (e.g. UI display when the conviction is being added). It applies
    per-conviction-only logic without propagation.

    The n7_posterior argument is preserved for signature compatibility
    but is no longer consulted: grading is now per-conviction only,
    consistent with §5.1.7's per-plea framing.
    """
    per_conv_bail = float(conviction.get("adj", {}).get("bail", 0.0))
    return _n7_threshold_grade(per_conv_bail)


def _n7_multiplier_for_conviction(conviction, n7_posterior=None):
    """Return the numerical multiplier for a conviction (no propagation)."""
    return N7_MULTIPLIERS[_n7_grade_for_conviction(conviction, n7_posterior)]


def _n7_aggregate_record_weight(criminal_record, n7_posterior=None):
    """
    Return (nominal_mean, n7_adjusted_mean, per_conviction_grades).

    Nominal:   mean of cal_weight across all convictions
    Adjusted:  mean of (cal_weight × N7-multiplier-with-propagation)
    Grades:    list of (grade, multiplier) tuples in conviction order

    Now applies cascade propagation per §RM.5.4: convictions are graded
    chronologically, and earlier tainted convictions can boost the
    bail-denial signal of later convictions.

    The n7_posterior argument is preserved for signature compatibility
    but no longer consulted (grading is per-conviction-only).
    """
    if not criminal_record:
        return None, None, []

    nominal_weights = [float(e.get("cal_weight", 0.0)) for e in criminal_record]

    # Compute grades in chronological order (assumed already sorted by
    # caller; criminal_record.sort by year ascending happens upstream).
    chronological_results = _n7_grades_chronological(criminal_record)
    grades = [(grade, mult) for grade, mult, _prop in chronological_results]
    adjusted_weights = [w * m for w, (_, m) in zip(nominal_weights, grades)]

    return (
        float(sum(nominal_weights) / len(nominal_weights)),
        float(sum(adjusted_weights) / len(adjusted_weights)),
        grades,
    )


# ── Jump Principle: Forward-Contamination Ceiling Effect (Ch 3 §3.5.3) ───────
# JP's Chapter 3 §3.5.3: "prior sentences, once imposed, function as baseline
# reference points for subsequent legal decisions regardless of whether their
# original severity reflected contemporaneous doctrine, proportionality
# principles, or systemic context. Inflated past sentences thereby become
# anchors for future escalation, producing recursive severity over time."
#
# Operationalisation (confirmed JP, Apr 27 2026):
#   ceiling_effect_per_conviction =
#       sentence_inflation_factor × era_multiplier × gladue_compliance_multiplier
#   cumulative_ceiling_for_conviction_Y = sum(ceiling_effects of priors X<Y)
#       capped at 0.40 (~40 percentage-point maximum upward shift on N2)
#   N2 receives upward shift = 0.5 × cumulative_ceiling_effect (most recent)
#
# All values are conservative methodological choices; they belong in the
# methodological appendix. The architecture is the doctrinal claim; the
# specific numbers are the claim's operationalisation.

# Sentence inflation factor — keyed on the existing sentence type label.
# A federal custody (2+ years) sentence carries the highest baseline anchoring
# capacity; CSO/probation/discharge carry minimal anchoring.
JUMP_SENTENCE_INFLATION = {
    "Federal custody (2+ years)":         0.20,
    "Provincial custody (< 2 years)":     0.10,
    "Conditional sentence order (CSO)":   0.03,
    "Probation only":                     0.01,
    "Fine only":                          0.0,
    "Absolute / conditional discharge":   0.0,
    "Time served":                        0.05,
    "Other / unknown":                    0.05,
}

# Era multiplier — keyed on conviction year. Reflects §3.5.3's identified
# punitive phases: 1995-2005 baseline; 2006-2015 mandatory-minimum revival
# / Safe Streets and Communities Act peak; 2016-2019 partial restoration;
# 2020+ post-Bill-C-5 restoration.
def _jump_era_multiplier(year):
    """Return era inflation multiplier per Ch 3 §3.5.3 phase analysis."""
    try:
        y = int(year)
    except (TypeError, ValueError):
        return 1.0
    if 1995 <= y <= 2005:
        return 1.0
    elif 2006 <= y <= 2015:
        return 1.5  # Mandatory-minimum revival / SSCA peak
    elif 2016 <= y <= 2019:
        return 1.2  # Partial restoration after SCC mandatory-minimum strikes
    elif y >= 2020:
        return 1.0  # Bill C-5 restoration
    else:
        return 1.0  # Pre-1995: baseline

# Gladue-compliance multiplier — when a conviction's adj.gladue is elevated,
# this signals that Gladue/Ipeelee/Morris factors were not substantively
# applied at the original sentencing, which per §3.5.4 amplifies temporal
# distortion ("legally required contextual reasoning was absent or
# underdeveloped").
def _jump_gladue_multiplier(conviction):
    """Return Gladue-non-compliance multiplier per Ch 3 §3.5.4."""
    adj_gladue = float(conviction.get("adj", {}).get("gladue", 0.0))
    if adj_gladue > 0.30:
        return 1.4  # Elevated: Gladue not substantively applied
    return 1.0

def _jump_ceiling_for_conviction(conviction):
    """
    Compute the ceiling-effect contribution of a single conviction (its own
    inflationary anchoring weight, before cumulation across the record).

    Per §3.5.3: a non-Gladue-compliant prior with an inflated carceral term
    carries a strong anchoring effect on subsequent severity assessment.
    """
    sent_type = conviction.get("sentence_type", "Other / unknown")
    base = JUMP_SENTENCE_INFLATION.get(sent_type, 0.05)
    era = _jump_era_multiplier(conviction.get("year", 2020))
    gladue_amp = _jump_gladue_multiplier(conviction)
    return float(base * era * gladue_amp)

def _jump_cumulative_chain(criminal_record_chronological):
    """
    Return list of (own_ceiling, inherited_ceiling, grade, weight) tuples
    in chronological order.

    `own_ceiling` is the conviction's nominal inflationary contribution
    (sentence_inflation × era × gladue_compliance), AFTER weighting by
    its N7 reliability grade per §RM.5.5. A conviction graded
    Heavily Discounted contributes at 0.30 of its nominal anchoring weight,
    reflecting the architecture's prior judgment that the conviction's
    severity reflects cascade contamination rather than legitimate
    sentencing assessment.

    `inherited_ceiling` is the sum of all prior convictions' weighted
    own_ceiling, capped at 0.40 per §3.5.3 cumulative-cap methodology.

    The first (earliest) conviction has inherited = 0; each subsequent
    conviction inherits the running sum of prior weighted ceiling effects.
    """
    JUMP_CUMULATIVE_CAP = 0.40
    chain = []
    running = 0.0
    # Run the chronological grading pass first to get each conviction's
    # reliability grade for jump-principle weighting.
    grading = _n7_grades_chronological(criminal_record_chronological)
    for e, (grade, _mult, _prop) in zip(criminal_record_chronological, grading):
        nominal_own = _jump_ceiling_for_conviction(e)
        weight = JUMP_WEIGHT_BY_GRADE.get(grade, 1.00)
        weighted_own = nominal_own * weight
        inherited_capped = min(running, JUMP_CUMULATIVE_CAP)
        chain.append((weighted_own, inherited_capped, grade, weight))
        running += weighted_own
    return chain

def _jump_record_n2_shift(criminal_record_chronological):
    """
    Return the upward shift to apply to N2 from the cumulative ceiling effect.

    Reads the (now grade-weighted) cumulative ceiling at the most recent
    conviction and applies a 0.5 coefficient — the shift to N2 is half
    the cumulative ceiling, conservatively scaled to keep N2 within
    [0, 0.95] in _cr_feed_nodes.

    The chain returned by _jump_cumulative_chain now carries 4-tuples
    (own_ceiling, inherited_ceiling, grade, weight). The first two values
    are the relevant scalars for the shift computation.
    """
    if not criminal_record_chronological:
        return 0.0
    chain = _jump_cumulative_chain(criminal_record_chronological)
    # Cumulative ceiling AT the most recent conviction = inherited + own
    own_last, inherited_last = chain[-1][0], chain[-1][1]
    cumulative_at_end = min(inherited_last + own_last, 0.40)
    return 0.5 * cumulative_at_end


def _completeness_state():
    """
    Return list of dicts describing the populated-ness of each input surface:
        {label, status('done'|'partial'|'empty'), detail, target_tab_index}

    Heuristics chosen so a freshly-opened app reads as 'mostly empty' and a
    case where every surface has been touched reads as 'all complete'.
    """
    ss = st.session_state
    items = []

    # Case profile — populated if profile_ev is non-empty
    pev = ss.get("profile_ev", {}) or {}
    if pev:
        items.append({"label":"Case profile","status":"done",
                      "detail":f"{len(pev)} field(s) recorded","tab":2})
    else:
        items.append({"label":"Case profile","status":"empty",
                      "detail":"defaults only","tab":2})

    # Criminal record
    rec = ss.get("criminal_record", []) or []
    if len(rec) >= 2:
        items.append({"label":"Criminal record","status":"done",
                      "detail":f"{len(rec)} convictions entered","tab":4})
    elif rec:
        items.append({"label":"Criminal record","status":"partial",
                      "detail":f"{len(rec)} conviction entered","tab":4})
    else:
        items.append({"label":"Criminal record","status":"empty",
                      "detail":"none entered","tab":4})

    # Gladue factors (17 total)
    gl = ss.get("gladue_checked", set()) or set()
    GL_TOTAL = 17
    if len(gl) >= 6:
        items.append({"label":"Gladue factors","status":"done",
                      "detail":f"{len(gl)} of {GL_TOTAL} checked","tab":5})
    elif gl:
        items.append({"label":"Gladue factors","status":"partial",
                      "detail":f"{len(gl)} of {GL_TOTAL} checked","tab":5})
    else:
        items.append({"label":"Gladue factors","status":"empty",
                      "detail":"none checked","tab":5})

    # Morris/Ellis SCE — done if framework selected and any factor has weight
    sce_vals = ss.get("sce_values", {}) or {}
    sce_active = sum(1 for v in sce_vals.values() if v and float(v) > 0)
    fw = (ss.get("scefw") or "morris").title()
    if sce_active >= 3:
        items.append({"label":"Morris/Ellis SCE","status":"done",
                      "detail":f"{fw} · {sce_active} factor(s) weighted","tab":6})
    elif sce_active:
        items.append({"label":"Morris/Ellis SCE","status":"partial",
                      "detail":f"{fw} · {sce_active} factor(s) weighted","tab":6})
    else:
        items.append({"label":"Morris/Ellis SCE","status":"empty",
                      "detail":f"{fw} framework, no factors yet","tab":6})

    # Risk & Distortions — manual_ev shows which nodes have manual overrides
    mev = ss.get("manual_ev", {}) or {}
    if len(mev) >= 5:
        items.append({"label":"Risk & Distortions","status":"done",
                      "detail":f"{len(mev)} of 15 nodes calibrated","tab":7})
    elif mev:
        items.append({"label":"Risk & Distortions","status":"partial",
                      "detail":f"{len(mev)} of 15 nodes calibrated","tab":7})
    else:
        items.append({"label":"Risk & Distortions","status":"empty",
                      "detail":"all at default","tab":7})

    # Documents — doc_res list
    docs = ss.get("doc_res", []) or []
    if docs:
        items.append({"label":"Documents","status":"done",
                      "detail":f"{len(docs)} analysed","tab":9})
    else:
        items.append({"label":"Documents","status":"empty",
                      "detail":"none uploaded","tab":9})

    # Scenarios
    scn = ss.get("saved_scenarios", {}) or {}
    if scn:
        items.append({"label":"Scenarios","status":"done",
                      "detail":f"{len(scn)} saved","tab":10})
    else:
        items.append({"label":"Scenarios","status":"empty",
                      "detail":"no counterfactuals","tab":10})

    return items

def _doctrinal_frame():
    """Return dict of doctrinal-frame values for the Summary tab."""
    ss = st.session_state
    fw = (ss.get("scefw") or "morris").lower()
    if fw == "ellis":
        fw_label, fw_color = "Ellis", TC["distortion"]
    elif fw == "both":
        fw_label, fw_color = "Morris + Ellis", TC["dual"]
    else:
        fw_label, fw_color = "Morris", TC["distortion"]

    conn_strength = ss.get("conn", "moderate")
    enex_strength = ss.get("enex", "relevant")
    conn_mult = {"none":0,"absent":0,"weak":.30,"moderate":.65,"strong":.90,"direct":1.0}.get(conn_strength, .65)

    qd = ss.get("qdiags", {}) or {}
    qbism_count = qd.get("count", 0) if isinstance(qd, dict) else 0
    qbism_summary = "None — VE coherent" if qbism_count == 0 else f"{qbism_count} flag(s)"

    return {
        "fw_label": fw_label,
        "fw_color": fw_color,
        "conn_label": conn_strength.title(),
        "conn_mult": conn_mult,
        "enex_label": enex_strength.title(),
        "qbism_summary": qbism_summary,
    }

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {"model":None,"engine":None,"profile_ev":{},"gladue_checked":set(),"criminal_record":[],"cr_doc_adj":{},"saved_scenarios":{},
            "sce_checked":set(),"sce_values":{},"manual_ev":{},"doc_adj":{},"posteriors":{},
            "qdiags":{},"conn":"moderate","enex":"relevant","scefw":"morris","doc_res":[],"qbism_plain":"","qbism_dm":{},
            "case_id":"","case_jur":""}
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

@st.cache_resource(show_spinner="Building Bayesian network...")
def _load():
    m=build_model(); return m, get_inference_engine(m)

if st.session_state.model is None:
    st.session_state.model, st.session_state.engine = _load()

# ── Factor data ───────────────────────────────────────────────────────────────
GF=[
  {"id":"g_r1","l":"Residential school — direct","n":10,"w":.18,"col":1,"sec":"Intergenerational trauma"},
  {"id":"g_r2","l":"Residential school — familial","n":10,"w":.14,"col":1,"sec":"Intergenerational trauma"},
  {"id":"g_sc","l":"Sixties Scoop / child welfare removal","n":10,"w":.14,"col":1,"sec":"Intergenerational trauma"},
  {"id":"g_dp","l":"Community displacement / relocation","n":10,"w":.10,"col":1,"sec":"Intergenerational trauma"},
  {"id":"g_cu","l":"Loss of language and cultural identity","n":12,"w":.10,"col":1,"sec":"Cultural disconnection"},
  {"id":"g_sp","l":"Absence of spiritual/ceremonial access","n":11,"w":.08,"col":1,"sec":"Cultural disconnection"},
  {"id":"g_fv","l":"Family violence / domestic abuse","n":10,"w":.12,"col":1,"sec":"Childhood & family"},
  {"id":"g_fo","l":"Foster care / group home placement","n":10,"w":.10,"col":1,"sec":"Childhood & family"},
  {"id":"g_pv","l":"Chronic poverty","n":10,"w":.08,"col":2,"sec":"Socioeconomic"},
  {"id":"g_ho","l":"Unstable housing / homelessness","n":18,"w":.08,"col":2,"sec":"Socioeconomic"},
  {"id":"g_em","l":"Structural employment barriers","n":18,"w":.07,"col":2,"sec":"Socioeconomic"},
  {"id":"g_ed","l":"Disrupted or denied education","n":10,"w":.07,"col":2,"sec":"Socioeconomic"},
  {"id":"g_sb","l":"Substance use linked to trauma","n":18,"w":.09,"col":2,"sec":"Substance & mental health"},
  {"id":"g_mh","l":"Untreated mental health conditions","n":18,"w":.08,"col":2,"sec":"Substance & mental health"},
  {"id":"g_gr","l":"Chronic grief and loss","n":10,"w":.08,"col":2,"sec":"Substance & mental health"},
  {"id":"g_op","l":"Over-policed community of origin","n":14,"w":.14,"col":2,"sec":"Systemic justice"},
  {"id":"g_yj","l":"Young offender system involvement","n":14,"w":.09,"col":2,"sec":"Systemic justice"},
  {"id":"g_pr","l":"Prior sentencing without Gladue analysis","n":12,"w":.12,"col":2,"sec":"Systemic justice"},
]

SF=[
  {"id":"s_ra","l":"Anti-Black / racialized racism documented","n":12,"w":.16,"fw":"morris","sec":"Structural racism"},
  {"id":"s_nb","l":"Neighbourhood structural disadvantage","n":14,"w":.14,"fw":"morris","sec":"Structural racism"},
  {"id":"s_cv","l":"Community violence exposure","n":10,"w":.12,"fw":"morris","sec":"Structural racism"},
  {"id":"s_rp","l":"Documented racial profiling","n":14,"w":.15,"fw":"morris","sec":"Structural racism"},
  {"id":"s_ir","l":"IRCA filed and before the court","n":12,"w":.20,"fw":"morris","sec":"IRCA"},
  {"id":"s_ij","l":"IRCA filed but disregarded by court","n":12,"w":.18,"fw":"morris","sec":"IRCA"},
  {"id":"s_bi","l":"Anti-Black systemic incarceration patterns","n":14,"w":.13,"fw":"morris","sec":"Black offender"},
  {"id":"s_bb","l":"Anti-Black bail practices documented","n": 7,"w":.12,"fw":"morris","sec":"Black offender"},
  {"id":"s_be","l":"Racialized educational exclusion","n":10,"w":.10,"fw":"morris","sec":"Black offender"},
  {"id":"s_sc","l":"State care involvement (non-racialized)","n":10,"w":.14,"fw":"ellis","sec":"Ellis — deprivation"},
  {"id":"s_ep","l":"Chronic poverty / economic deprivation","n":18,"w":.10,"fw":"ellis","sec":"Ellis — deprivation"},
  {"id":"s_et","l":"Trauma history without racialized component","n":10,"w":.11,"fw":"ellis","sec":"Ellis — deprivation"},
  {"id":"s_eg","l":"Geographic marginalization","n":11,"w":.09,"fw":"ellis","sec":"Ellis — deprivation"},
  {"id":"s_ee","l":"Educational deprivation","n":10,"w":.08,"fw":"ellis","sec":"Ellis — deprivation"},
  {"id":"s_pa","l":"Parity principle misapplied","n":12,"w":.12,"fw":"both","sec":"Judicial errors"},
  {"id":"s_se","l":"Sequencing error — SCE applied downstream","n":12,"w":.14,"fw":"both","sec":"Judicial errors"},
  {"id":"s_bs","l":"Belief stasis — SCE acknowledged but inert","n":12,"w":.16,"fw":"both","sec":"Judicial errors"},
]

def cmult(): return {"none":0,"absent":0,"weak":.30,"moderate":.65,"strong":.90,"direct":1.0}.get(st.session_state.conn,.65)
def emult(): return {"none":0,"peripheral":.35,"relevant":.70,"central":1.0}.get(st.session_state.enex,.70)

def gdelta():
    d={}
    for f in GF:
        if f["id"] in st.session_state.gladue_checked: d[f["n"]]=d.get(f["n"],0)+f["w"]
    return d

def sdelta():
    d={}; fw=(st.session_state.scefw or "morris").lower(); m=cmult() if fw!="ellis" else emult()
    sce_vals = st.session_state.get("sce_values", {})
    for f in SF:
        slider_val = sce_vals.get(f["id"], 0.0)
        if slider_val > 0.01:
            show=fw=="both" or (fw=="morris" and f["fw"]!="ellis") or (fw=="ellis" and f["fw"]!="morris")
            if show: d[f["n"]]=d.get(f["n"],0)+f["w"]*m*slider_val
    return d

def _compute_sce_corrections_by_gate():
    """
    Compute per-gate SCE evidence for the Quantum-tab connection-gate
    contextuality check (Appendix Q §AQ.3.3.5.4).

    For each connection-gate strength {weak, moderate, strong}, recompute
    sdelta()-equivalent SCE corrections using that gate as the multiplier.
    Returns a dict {gate_label: {node_id: probability}} suitable for passing
    to quantum_diagnostics.check_connection_gate_contextuality().

    The computation mirrors sdelta()'s logic but holds the active framework
    constant and only varies the connection-gate multiplier. This isolates
    the contextuality test to the doctrinal call about Morris para 97
    connection strength, per the rationale set out in Appendix Q.
    """
    corrections_by_gate = {}
    fw = (st.session_state.scefw or "morris").lower()
    sce_vals = st.session_state.get("sce_values", {})

    # Fixed multipliers for the three gate strengths we test.
    gate_multipliers = {
        "weak":     0.30,
        "moderate": 0.65,
        "strong":   0.90,
    }

    for gate_label, m in gate_multipliers.items():
        d = {}
        for f in SF:
            slider_val = sce_vals.get(f["id"], 0.0)
            if slider_val > 0.01:
                show = (
                    fw == "both"
                    or (fw == "morris" and f["fw"] != "ellis")
                    or (fw == "ellis"  and f["fw"] != "morris")
                )
                if show:
                    d[f["n"]] = d.get(f["n"], 0) + f["w"] * m * slider_val
        corrections_by_gate[gate_label] = d

    return corrections_by_gate


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inf():
    from model import compute_do_risk

    # Architecture: VE gives structural network priors; profile_ev directly
    # sets each observed node's posterior; Gladue/SCE apply as continuous
    # corrections on top. This avoids the binary-collapse problem where
    # converting profile floats to 0/1 forces all risk node posteriors to
    # exactly 0.0, destroying all nuance in the Node 20 formula.

    # Step 1: Run VE with no evidence → network structural priors
    # Only pass very high-confidence manual evidence as hard binary
    hard_ev={}
    for nid,prob in st.session_state.manual_ev.items():
        if prob>=0.85 or prob<=0.15:  # Only very certain manual overrides
            hard_ev[str(nid)]=1 if prob>=0.5 else 0
    hard_ev.pop("20",None)
    post=query_do_risk(st.session_state.engine, hard_ev)

    # Step 2: Apply profile evidence as direct continuous posterior values
    # Profile_ev represents the case-specific probability for each node —
    # we assign these directly rather than forcing binary collapse
    for nid,prob in st.session_state.profile_ev.items():
        if str(nid) not in hard_ev:
            post[nid]=float(np.clip(prob,.05,.95))

    # Step 3: Apply Gladue, Morris/Ellis SCE, and document adjustments
    # These are additional continuous corrections ON TOP of the profile values.
    # Always applied — no 'not in bev' guard (that was the original bug).
    for nid,d in {**gdelta(),**sdelta(),**st.session_state.doc_adj}.items():
        post[nid]=float(np.clip(post.get(nid,0.5)+d,.05,.95))

    # Step 4: Compute Node 20 using full model formula
    # Includes: record reliability (N7/N6), Ewert tool validity (N5→N3/N4),
    # and age burnout multiplier (N15). All from model.compute_do_risk.
    post[20]=compute_do_risk(post)
    st.session_state.posteriors=post
    # Pass engine + per-gate SCE corrections so the new Appendix Q
    # diagnostics (order stability §AQ.3.3.5.3 and connection-gate
    # contextuality §AQ.3.3.5.4) can run. Both kwargs are optional in
    # quantum_diagnostics.diagnose() — passing them activates the checks.
    _qd_sce_by_gate = _compute_sce_corrections_by_gate()
    st.session_state.qdiags=diagnose(post,hard_ev,list(st.session_state.gladue_checked),
        list(st.session_state.sce_checked),st.session_state.profile_ev,st.session_state.conn,
        engine=st.session_state.engine,
        sce_corrections_by_gate=_qd_sce_by_gate)

def _sync_profile_from_widgets():
    """
    Read all Case Profile widget values from st.session_state (set by their key= params)
    and recompute profile_ev. Called BEFORE the header renders so the DO score
    is always current — not one step behind.
    """
    ss = st.session_state
    # Read widget values — use .get() with defaults matching widget defaults
    age    = ss.get("age", 35)
    idbg   = ss.get("id_bg", "Not recorded / unknown")
    pclr   = ss.get("pclr", 20)
    s99    = ss.get("s99", 3)
    # Mark 8 fix: fallback defaults aligned with actual Streamlit widget defaults
    # (each selectbox defaults to its first option). Prior to this, the fallbacks
    # were biased toward high-severity values, which caused the first-paint
    # header DO chip to show ~31.5% before snapping to ~24.9% on widget population.
    viol   = ss.get("viol", "None")
    fasd   = ss.get("fasd", "None / not assessed")
    sub    = ss.get("sub", "None / in remission")
    peers  = ss.get("peers", "None identified")
    stab   = ss.get("stab", "Stable")
    det    = ss.get("det", 60)
    counsel= ss.get("counsel", "Adequate")
    gr     = ss.get("gr", "Yes — full report before court")
    tools  = ss.get("tools", "Culturally validated only")
    pol    = ss.get("pol", "No evidence")
    prov   = ss.get("prov", "Low DO designation rate")
    prog   = ss.get("prog", "Yes — full culturally grounded")
    rehab  = ss.get("rehab", "Strong — consistent")

    isr = idbg in ["Indigenous — s.718.2(e) + Gladue","Black — Morris IRCA","Other racialized — Morris"]
    pev={}
    pev[2] ={"None":.08,"Minor/historical":.25,"Moderate":.50,"Serious":.78,"Established pattern":.90}.get(viol,.50)
    pev[3] =.82 if pclr>=30 else .55 if pclr>=20 else .30 if pclr>=10 else .12
    pev[4] =.82 if s99>=6 else .55 if s99>=4 else .32 if s99>=2 else .12
    pev[5] ={"Culturally validated only":.10,"Mix — partially qualified":.45,
             "Standard, no cultural qualification":.85 if isr else .40,"No actuarial tools":.15}.get(tools,.40)
    pev[6] ={"Adequate":.15,"Marginal":.45,"Inadequate — no cultural investigation":.72,
             "Ineffective — constitutional breach":.90}.get(counsel,.45)
    pev[7] =.85 if det>180 else .70 if det>90 else .40 if det>30 else .15
    pev[9] ={"None / not assessed":.15,"Suspected, undiagnosed":.50,"Confirmed diagnosis":.88}.get(fasd,.15)
    pev[10]=min(.90,.45+(.20 if "Indigenous" in idbg else 0))
    pev[11]={"Yes — full culturally grounded":.10,"Limited availability":.55,
             "No culturally appropriate programming":.85}.get(prog,.55)
    pev[12]={"Yes — full report before court":.15,"Partial / summary only":.50,
             "No report commissioned":.82,"Report commissioned, disregarded":.92}.get(gr,.82)
    pev[13]=.75 if "gaming" in rehab.lower() else .22
    pev[14]={"No evidence":.15,"Some — marginal":.50,"Strong — documented over-surveillance":.85}.get(pol,.50)
    pev[15]=.85 if age>=55 else .70 if age>=45 else .40 if age>=35 else .20
    pev[16]={"Low DO designation rate":.20,"Medium rate":.45,"High DO designation rate":.72}.get(prov,.45)
    sv ={"None / in remission":.15,"Low":.35,"Moderate":.60,"High — dependency":.80}.get(sub,.60)
    pv ={"None identified":.10,"Some — limited":.35,"Strong — primary network":.65}.get(peers,.35)
    stv={"Stable":.10,"Marginal":.40,"Unstable / homeless":.70}.get(stab,.40)
    pev[18]=float(np.clip((sv+pv+stv)/3+.05,.05,.92))
    rv ={"Strong — consistent":.10,"Moderate":.35,"Minimal":.60,"None — apparent refusal":.80,
         "Anomalously positive (gaming risk)":.30}.get(rehab,.60)
    pev[19]=float(np.clip(rv+(.12 if prog=="No culturally appropriate programming" else 0),.05,.90))
    # Also sync Gladue and SCE checked sets from checkbox keys
    gl_checked = {f["id"] for f in GF if ss.get(f"gl_{f['id']}", False)}
    sce_checked = {f["id"] for f in SF if ss.get(f"sce_{f['id']}", False)}
    ss["gladue_checked"] = gl_checked
    ss["sce_checked"]    = sce_checked
    ss["profile_ev"]     = pev
    # Sync sce_values from slider keys (continuous values, 0.0-1.0)
    if "sce_values" not in ss:
        ss["sce_values"] = {}
    # Sync connection/framework settings
    if "conn_s" in ss:  ss["conn"] = ss["conn_s"]
    if "enex_s" in ss:  ss["enex"] = ss["enex_s"]
    if "scefw_r" in ss: ss["scefw"] = ss["scefw_r"].lower()  # radio returns capitalised


_sync_profile_from_widgets()
run_inf()
P=st.session_state.posteriors
dp=P[20]; bl,bc,bg=rb(dp)

# ── Empty-state detection ─────────────────────────────────────────────────────
# When no case data has been entered, the posterior reflects the network's
# defaults rather than any case-specific evidence. Show "Awaiting case data"
# in posterior displays so a user does not misread the default as a claim
# about the empty (Untitled) case.
def _case_is_empty():
    """Return True if no case data has been entered across any input tab."""
    ss = st.session_state
    if (ss.get("case_id") or "").strip():
        return False
    if ss.get("criminal_record"):
        return False
    if ss.get("gladue_checked"):
        return False
    sce_vals = ss.get("sce_values") or {}
    if any(v > 0.01 for v in sce_vals.values()):
        return False
    if ss.get("manual_ev"):
        return False
    return True

_empty = _case_is_empty()

# ── Header ────────────────────────────────────────────────────────────────────
ct,cd=st.columns([3,1])
with ct:
    st.markdown(f"""<div style="border-bottom:1px solid rgba(0,0,0,.08);padding-bottom:.6rem;margin-bottom:.5rem">
    <div class="pt" style="display:flex;align-items:center;gap:0;letter-spacing:7px">
      <span>P.</span><img src="data:image/png;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAQABAADASIAAhEBAxEB/8QAHQABAAICAwEBAAAAAAAAAAAAAAcJBggBAgUEA//EAGQQAAEDAgMEAwgLCQkNBQkBAAABAgMEBQYHEQghMUESUWETFiJxgZOU0RQVMkJSVFaRkrHSFxgjM1NylaHBNDZDRlVihLKzJDU3V2Nlc3R1goXh8CUnRIPiJkVHZKKjwsPxCf/EABoBAQEAAwEBAAAAAAAAAAAAAAABAgQFAwb/xAAjEQEAAgICAwACAwEAAAAAAAAAARECAwQSITFBBRMiUYFh/9oADAMBAAIRAxEAPwDTIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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OSqVw4qsdfhrEdfYbmxrKyhndDKjXo5uqLxRUVUVDz+6y/lH/SU6ucrl1cqqvWqhXAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/9k="
        style="height:3.2rem;width:3.2rem;object-fit:contain;
               margin:0 -1px;vertical-align:middle;
               transform:translateY(0.25rem);
               filter:invert(1) contrast(2.2);opacity:0.92"
        alt="Ethical AI" /><span>.R.V.I.S</span>
    </div>
    <div class="ps"><em>Probabilistic and Analytical Reasoning Virtual Intelligence System</em> &nbsp;·&nbsp;
    <span style="color:#185FA5;font-weight:500">University of London</span> &nbsp;·&nbsp;
    <span style="color:#1B5E20;font-weight:500">Ethical AI Initiative</span> &nbsp;·&nbsp;
    <span style="color:#A32D2D;font-weight:500">Jeinis Patel, PhD Candidate and Barrister</span>
    &nbsp;·&nbsp; &#169; 2026 Jeinis Patel</div></div>""",unsafe_allow_html=True)
with cd:
    if _empty:
        # Empty-state header chip: muted register, no percentage
        st.markdown(f"""<div class="dc" style="background:#FBFAF7;border:1px solid #E0DDD6;margin-top:4px">
        <div class="dl" style="color:#9E9E9E">Node 20 · DO risk</div>
        <div class="dp" style="color:#9E9E9E;font-family:Fraunces,Georgia,serif;font-style:italic;font-size:1.5rem;font-weight:500">—</div>
        <div class="db" style="color:#707070;font-family:Fraunces,serif;font-style:italic;font-size:0.84rem">Awaiting case data</div></div>""",unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="dc" style="background:{bg};border:1px solid {bc}44;margin-top:4px">
        <div class="dl" style="color:{bc}">Node 20 · DO risk</div>
        <div class="dp" style="color:{bc}">{dp*100:.1f}%</div>
        <div class="db" style="color:{bc}">{bl}</div></div>""",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

# ── Node positions ────────────────────────────────────────────────────────────
NP={1:(.50,.92),2:(.12,.73),3:(.50,.73),4:(.88,.73),
    5:(.07,.55),6:(.23,.55),9:(.40,.55),13:(.57,.55),15:(.73,.55),
    7:(.07,.38),10:(.23,.38),11:(.40,.38),14:(.57,.38),17:(.73,.38),
    8:(.07,.20),12:(.23,.20),16:(.40,.20),18:(.57,.20),19:(.73,.20),
    20:(.50,.03)}

def draw_dag(post,sel=None):
    fig,ax=plt.subplots(figsize=(13,9),facecolor='#fafafa')
    ax.set_xlim(-.02,1.02);ax.set_ylim(-.08,1.02);ax.axis('off');ax.set_facecolor('#fafafa')
    for y,h,lbl,lx in [(.83,.10,"Layer I — Substantive risk",.52),
                        (.29,.53,"Layer II — Systemic distortion & doctrinal fidelity",.52),
                        (-.04,.09,"Layer III — Structural output",.52)]:
        ax.add_patch(plt.Rectangle((0,y),1.0,h,color='#f0f0f0',alpha=.55,zorder=0))
        ax.text(lx,y+h-.015,lbl,fontsize=8,color='#bbb',fontweight='bold',va='top',ha='center',zorder=1)
    for f,t in EDGES:
        if f not in NP or t not in NP: continue
        x1,y1=NP[f];x2,y2=NP[t];hi=sel and (f==sel or t==sel)
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle="-|>",color='#888' if hi else '#ccc',lw=1.2 if hi else .6,
            connectionstyle="arc3,rad=0.05"))
    NR={1:.055,20:.055}
    for nid,(x,y) in NP.items():
        m=NODE_META[nid];col=TC[m["type"]];p=post.get(nid,.5);iS=sel==nid
        r=NR.get(nid,.040)
        ax.add_patch(plt.Circle((x,y),r,color=col if iS else col+'28',ec=col,lw=2 if iS else 1,zorder=3))
        th=np.linspace(-np.pi/2,-np.pi/2+2*np.pi*p,60)
        ax.plot(x+(r+.010)*np.cos(th),y+(r+.010)*np.sin(th),color=col,lw=2.5,alpha=.85,zorder=4)
        ax.text(x,y,str(nid),ha='center',va='center',fontsize=8 if nid<10 else 7,
                fontweight='bold',color='white' if iS else col,zorder=5)
        lbl=m["short"][:14]+("…" if len(m["short"])>14 else "")
        ax.text(x,y-r-.025,lbl,ha='center',va='top',fontsize=6.5,color='#555',zorder=5)
        ax.text(x,y-r-.050,f'{p*100:.0f}%',ha='center',va='top',fontsize=6,color=col,fontweight='bold',zorder=5,alpha=.8)
    handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markeredgecolor=c,
             markersize=8,label=TL[t]) for t,c in TC.items()]
    ax.legend(handles=handles,loc='upper right',fontsize=7.5,framealpha=.92,edgecolor='#ddd')
    plt.tight_layout(pad=.5);return fig

# ── CanLII availability ───────────────────────────────────────────────────────
try:
    from canlii_client import (
        # Existing surface (back-compat)
        search_node_developments, get_tetrad_updates,
        is_configured as canlii_ok,
        # New surface (April 27 2026 rebuild)
        search_with_filters, get_tracked_updates,
        flatten_search_results, validate_api_key,
        ALL_TRACKED_CITATIONS, TETRAD_CITATIONS, PROPORTIONALITY_CITATIONS,
        NODE_SEARCH_QUERIES,
    )
    CANLII_ON = True
except ImportError:
    CANLII_ON = False
    def canlii_ok(): return False
    def validate_api_key(): return {"valid": False, "error": "canlii_client not importable"}

# ── Tabs ──────────────────────────────────────────────────────────────────────
TABS=st.tabs(["📋 Summary","🕸️ Architecture","📋 Profile","💬 Intake (Chat)",
              "📜 Criminal Record","🦅 Gladue","⚖️ SCE",
              "🔬 Risk & Distortions","📊 Inference","📂 Documents",
              "🔀 Scenarios","⚛️ Quantum","📄 Report"])

# ── T0: Summary (Mark 8) ──────────────────────────────────────────────────────
with TABS[0]:
    _band_lbl, _band_fg, _band_bg = _summary_band(P[20])
    _drv_up_raw, _drv_dn_raw = _top_drivers(P, k=8)
    # ── Doctrinal architecture: N1 (structural control) and N17 (properly-weighted-conditional) ──
    # Per Chapter 5: these nodes encode meta-constraints rather than case-specific drivers.
    # They are surfaced separately so the drivers panel below shows only case-responsive nodes.
    _struct_nodes = {1, 17}  # node IDs treated as structural in the Summary panel
    _drv_up = [d for d in _drv_up_raw if d["nid"] not in _struct_nodes][:5]
    _drv_dn = [d for d in _drv_dn_raw if d["nid"] not in _struct_nodes][:5]
    _struct_drivers = []
    for d in _drv_up_raw + _drv_dn_raw:
        if d["nid"] in _struct_nodes and d not in _struct_drivers:
            _struct_drivers.append(d)
    # Sort structural by node id for a stable display order (N1 first, then N17)
    _struct_drivers.sort(key=lambda d: d["nid"])
    _comp = _completeness_state()
    _doc = _doctrinal_frame()
    _case_id   = (st.session_state.get("case_id") or "").strip() or "Untitled case"
    _case_jur  = (st.session_state.get("case_jur") or "").strip()

    # ── Zone 1: Headline ─────────────────────────────────────────────────────
    # Empty-state: "No case loaded" + navigation guidance.
    # Populated: existing "Untitled case" title block + designation-risk body.
    if _empty:
        _headline_left = (
            "<div>"
            # Title
            "<div style=\"font-family:'Fraunces',Georgia,serif;font-weight:500;font-size:2rem;"
            "letter-spacing:-0.01em;color:#1a1a1a;line-height:1.15\">"
            "No case loaded"
            "</div>"
            # Subtitle
            "<div style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:0.95rem;"
            "color:#888;margin:6px 0 18px 0\">"
            "Awaiting case data — sentencing audit · PARVIS Bayesian network"
            "</div>"
            # Button row — two visual buttons with honest caption
            "<div style=\"display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap\">"
            # Primary: Profile
            "<div style=\"display:inline-flex;align-items:center;gap:8px;"
            "padding:9px 16px;background:#1a1a1a;color:#FBFAF7;"
            "border-radius:6px;font-family:'DM Sans',sans-serif;"
            "font-size:0.86rem;font-weight:500;letter-spacing:0.005em\">"
            "<span>Begin at Profile</span>"
            "<span style=\"font-size:1rem;line-height:1\">→</span>"
            "</div>"
            # Secondary: Architecture
            "<div style=\"display:inline-flex;align-items:center;gap:8px;"
            "padding:9px 16px;background:#FBFAF7;color:#3a3a3a;"
            "border:1px solid #E0DDD6;border-radius:6px;"
            "font-family:'DM Sans',sans-serif;font-size:0.86rem;"
            "font-weight:500;letter-spacing:0.005em\">"
            "<span>Inspect Architecture</span>"
            "<span style=\"font-size:1rem;line-height:1;color:#9E9E9E\">→</span>"
            "</div>"
            "</div>"
            # Honest caption — click the actual tab above
            "<div style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;"
            "font-size:0.82rem;color:#9E9E9E;margin-top:4px;line-height:1.5\">"
            "Click the corresponding tab above to navigate."
            "</div>"
            "</div>"
        )
    else:
        _headline_left = (
            f"<div>"
            f"<div style=\"font-family:'Fraunces',Georgia,serif;font-weight:500;font-size:2rem;"
            f"letter-spacing:-0.01em;color:#1a1a1a;line-height:1.15\">"
            f"{_case_id}"
            f"</div>"
            f"<div style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:0.95rem;"
            f"color:#888;margin:6px 0 14px 0\">"
            f"{(_case_jur + ' · ') if _case_jur else ''}Sentencing audit · PARVIS Bayesian network"
            f"</div>"
            f"<div style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:0.92rem;"
            f"color:#3a3a3a;line-height:1.65;max-width:620px\">"
            f"Posterior probability of Dangerous Offender designation given current evidence, "
            f"doctrinal corrections, and systemic distortion adjustments. Models DESIGNATION "
            f"RISK — not intrinsic dangerousness."
            f"</div>"
            f"</div>"
        )

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 280px;gap:32px;align-items:center;
                padding:24px 0 28px 0;border-bottom:1px solid rgba(0,0,0,.06);
                margin-bottom:24px">
      {_headline_left}
      {("<div style=\"background:#FBFAF7;border:1px solid #E0DDD6;border-radius:14px;padding:18px 22px;text-align:center\">"
        "<div style=\"font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;color:#9E9E9E;font-weight:700;margin-bottom:6px\">"
        "DO Designation Risk</div>"
        "<div style=\"font-family:Fraunces,Georgia,serif;font-style:italic;font-size:2.6rem;font-weight:500;color:#9E9E9E;line-height:1;margin:8px 0 4px 0\">—</div>"
        "<div style=\"font-family:Fraunces,Georgia,serif;font-style:italic;font-size:0.95rem;color:#707070;margin-top:6px\">Awaiting case data</div>"
        "<div style=\"font-size:0.68rem;color:#9E9E9E;opacity:0.85;margin-top:10px;line-height:1.4\">Enter case profile, criminal record, or Gladue / SCE evidence to begin.</div>"
        "</div>") if _empty else (
        f"<div style=\"background:{_band_bg};border:1px solid {_band_fg}33;border-radius:14px;padding:18px 22px;text-align:center\">"
        f"<div style=\"font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;color:{_band_fg};font-weight:700;margin-bottom:6px\">"
        "DO Designation Risk</div>"
        f"<div style=\"font-family:monospace;font-size:3rem;font-weight:600;color:{_band_fg};line-height:1\">"
        f"{P[20]*100:.1f}<span style=\"font-size:1.4rem;opacity:0.7\">%</span></div>"
        f"<div style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:1rem;color:{_band_fg};margin-top:4px\">{_band_lbl}</div>"
        f"<div style=\"font-size:0.68rem;color:{_band_fg};opacity:0.78;margin-top:8px;line-height:1.4\">pgmpy Variable Elimination · Tetrad-bound</div>"
        "</div>")}
    </div>
    """, unsafe_allow_html=True)

    # ── Empty-state landing: About + How to read (Apr 27 2026) ─────────────
    # Renders only in the empty state, between the headline and the Doctrinal
    # architecture section. Provides examiner-facing orientation per JP's
    # spec. Once case data is entered, these blocks disappear.
    if _empty:
        # Side-by-side two-column layout with custom HTML <details> collapsibles.
        # Both closed by default. Native HTML5 — no JavaScript required.
        # Styled to match the cream-and-Fraunces aesthetic.
        st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

        # CSS for the collapsibles — defined once, used by both columns.
        # Removes the default disclosure triangle on most browsers and
        # replaces it with a custom +/− indicator. Fraunces serif for
        # consistency with surrounding chrome.
        _COLLAPSE_CSS = """
        <style>
        details.parvis-land {
            background: #FBFAF7;
            border: 1px solid #E0DDD6;
            border-radius: 8px;
            padding: 0;
            margin-bottom: 14px;
            transition: border-color 120ms ease;
        }
        details.parvis-land[open] {
            border-color: #C7C2B8;
        }
        details.parvis-land > summary {
            list-style: none;
            cursor: pointer;
            padding: 16px 22px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 500;
            font-size: 1.15rem;
            color: #1a1a1a;
            letter-spacing: -0.005em;
            user-select: none;
        }
        details.parvis-land > summary::-webkit-details-marker {
            display: none;
        }
        details.parvis-land > summary::after {
            content: "+";
            font-family: 'DM Sans', sans-serif;
            font-weight: 400;
            font-size: 1.25rem;
            color: #707070;
            line-height: 1;
            transition: transform 160ms ease;
        }
        details.parvis-land[open] > summary::after {
            content: "−";
            color: #1a1a1a;
        }
        details.parvis-land[open] > summary {
            border-bottom: 1px solid #EFEDE7;
        }
        details.parvis-land > .parvis-land-body {
            padding: 18px 22px 6px 22px;
        }
        details.parvis-land > .parvis-land-body p {
            font-family: 'Fraunces', Georgia, serif;
            font-size: 0.94rem;
            color: #3a3a3a;
            line-height: 1.75;
            margin: 0 0 16px 0;
        }
        details.parvis-land > .parvis-land-body p:last-child {
            margin-bottom: 8px;
        }
        details.parvis-land > .parvis-land-body em {
            font-style: italic;
        }
        details.parvis-land > .parvis-land-body strong {
            font-weight: 500;
            color: #1a1a1a;
        }
        </style>
        """
        st.markdown(_COLLAPSE_CSS, unsafe_allow_html=True)

        _land_left, _land_gap, _land_right = st.columns([1, 0.05, 1])

        # ── Left column — About this implementation (collapsed by default) ──
        with _land_left:
            st.markdown(
                """<details class="parvis-land">
<summary>About this implementation</summary>
<div class="parvis-land-body">
<p>PARVIS is a reference implementation of the twenty-node Bayesian audit architecture set out in Chapter 5 of the underlying thesis. The architecture's purpose is to render the inferential structure of Canadian sentencing — particularly the structure required by <em>Gladue</em>, <em>Ipeelee</em>, <em>Morris</em>, and <em>Ewert</em> — auditable, reconstructable, and contestable. It does so by representing both substantive risk considerations and the systemic distortions that shape how risk is constructed in the evidentiary record.</p>
<p>Three constitutive constraints govern what this implementation does and does not do, drawn from <em>Appendix O §O.1.1</em> of the thesis. First, PARVIS is an audit mechanism, not a decision-maker. It models whether legally required belief revision has occurred; it does not produce sentencing outcomes or recommend designations. Second, the outputs displayed here are diagnostic observations about reasoning, not adjudicative facts. They belong in the domain of audit and contestation, not the evidentiary record of any particular proceeding. Third, the architecture is open by design — its node structure, conditional dependencies, and prior values are matters of public legal reasoning, contestable by adversarial parties.</p>
<p>The values shown throughout the implementation are research-prototype values. They serve as illustrative anchors for the framework set out in the thesis text and are subject to expert elicitation through the SHELF/Cooke methodology described in <em>Appendix O §O.3</em> before any deployment could responsibly occur. The constructive proof presented here is a demonstration that the architecture <em>can</em> be built consistent with its doctrinal commitments, not a claim that it is ready for institutional use.</p>
</div>
</details>""",
                unsafe_allow_html=True,
            )

        # ── Right column — How to read this (collapsed by default) ──────────
        with _land_right:
            st.markdown(
                """<details class="parvis-land">
<summary>How to read this</summary>
<div class="parvis-land-body">
<p>The architecture comprises three substantive layers and one cross-cutting diagnostic layer. The <em>Substantive Risk Layer</em> (Nodes 1–4) represents empirically supported risk indicators subject to Canadian burdens of proof and proportionality constraints. The <em>Systemic Distortion and Doctrinal Fidelity Layer</em> (Nodes 5–17) captures mechanisms through which criminal records and risk assessments become unreliable, qualifying — never displacing — the confidence placed in upstream evidence. The <em>Structural Calibration and Output Layer</em> (Nodes 18–20) governs how revised beliefs propagate downstream to the Dangerous Offender designation posterior at Node 20. The cross-cutting <em>Quantum diagnostic layer</em> (Appendix Q) surfaces epistemic conditions — order effects, contextuality, premature scalar collapse, distorted priors — that classical Bayesian inference is poorly equipped to represent.</p>
<p>The natural starting point is the <strong>Profile</strong> tab. <strong>Architecture</strong> offers a structural overview without requiring case data. <strong>Inference</strong> displays the live posterior distribution once data has been entered.</p>
</div>
</details>""",
                unsafe_allow_html=True,
            )

        # Horizontal rule before the Doctrinal architecture deep-cut
        st.markdown(
            "<div style='border-top:1px solid #E0DDD6;margin:32px 0 24px 0'></div>",
            unsafe_allow_html=True,
        )

    # ── Zone 2a: Doctrinal architecture (N1, N17 — structural constraints) ─────
    # Surfaced separately from the case-responsive drivers below per Chapter 5's
    # treatment of N1 as a structural control / shared parent node.
    st.markdown(
        "<h3 style='margin-bottom:4px'>Doctrinal architecture</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.86rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:720px'>"
        "Structural constraints conditioning the entire inference. "
        "These nodes encode the evidentiary architecture of Canadian sentencing "
        "law and the post-admission inferential structure — they remain "
        "stable across case-specific evidence rather than responding to it."
        "</div>",
        unsafe_allow_html=True,
    )

    # Render the two structural nodes side-by-side
    if _struct_drivers:
        _struct_cols = st.columns(len(_struct_drivers))
        for _idx, _d in enumerate(_struct_drivers):
            with _struct_cols[_idx]:
                # Use a stable amber/blue accent regardless of node type colour
                _accent = "#BA7517" if _d["nid"] == 1 else "#185FA5"
                _accent_bg = "#FAEEDA" if _d["nid"] == 1 else "#E8F0FA"
                _accent_border = "#E5CC95" if _d["nid"] == 1 else "#C7D3E5"
                # Surface caption depends on node identity
                if _d["nid"] == 1:
                    _surface = (
                        "Encodes the evidentiary admissibility thresholds — "
                        "Crown beyond reasonable doubt for aggravating evidence "
                        "(~83%), defence balance of probabilities for mitigating "
                        "(~51%). Functions as a structural meta-constraint "
                        "conditioning all other inference, not as a posterior "
                        "over case facts."
                    )
                    _expand_label = "Formal treatment — Chapter 5 §1"
                    _formal = (
                        "Per Chapter 5, N1 operates as a shared parent node "
                        "whose states deterministically condition the CPT "
                        "entries of downstream aggravation and mitigation "
                        "nodes — collapsing likelihood terms to near-zero when "
                        "evidentiary burdens are unmet. This is the limiting "
                        "case of Bayesian conditional probability where the "
                        "conditional collapses to certainty. Deterministic "
                        "conditioning of this kind is orthodox within Bayesian "
                        "network methodology. The node's posterior remains "
                        "stable across case-specific evidence because it "
                        "represents a precondition on belief revision rather "
                        "than a variable within it. Law does not work in "
                        "percentages; the values shown are best-available "
                        "industry estimates of these doctrinal thresholds."
                    )
                else:  # N17
                    _surface = (
                        "Responds to admitted evidence but assumes the "
                        "evidentiary admissibility gate (N1) has been correctly "
                        "applied. Reflects the post-gate inferential structure "
                        "in which conflicting evidence is properly weighed."
                    )
                    _expand_label = "Doctrinal scope of N17"
                    _formal = (
                        "N17 is a properly-weighted-conditional node: its "
                        "posterior responds to evidence once admitted, but the "
                        "responsiveness assumes that the weighting prior to "
                        "admission has been correctly rendered by the "
                        "sentencing judge. The judge's gatekeeping role under "
                        "N1 is logically prior to any update at N17. If "
                        "admissibility has been mishandled — evidence admitted "
                        "that should not have been, or excluded that should "
                        "have been — N17 is computing on a corrupted basis."
                    )
                # Card
                st.markdown(
                    f"<div style='background:{_accent_bg};border:1px solid {_accent_border};"
                    f"border-left:3px solid {_accent};border-radius:8px;padding:14px 18px;"
                    f"margin-bottom:10px'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                    f"font-weight:600;padding:3px 9px;border-radius:5px;color:white;"
                    f"background:{_accent}'>N{_d['nid']}</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.05rem;"
                    f"font-weight:500;color:#1a1a1a'>{_d['short']}</div>"
                    f"<div style='margin-left:auto;font-family:JetBrains Mono,monospace;"
                    f"font-size:0.95rem;font-weight:600;color:{_accent}'>{_d['p']*100:.1f}%</div>"
                    f"</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.84rem;color:#3a3a3a;line-height:1.55'>{_surface}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(_expand_label, expanded=False):
                    st.markdown(
                        f"<div style='font-family:Fraunces,serif;font-size:0.86rem;"
                        f"color:#3a3a3a;line-height:1.65;padding:4px 0'>{_formal}</div>",
                        unsafe_allow_html=True,
                    )

    st.markdown(
        "<div style='border-top:1px solid #EFEDE7;margin:24px 0 18px 0'></div>",
        unsafe_allow_html=True,
    )

    # ── Zone 2: Drivers ──────────────────────────────────────────────────────
    st.markdown("### Drivers of the posterior")
    st.caption("Top 5 nodes pushing DO risk up, top 5 pulling it down. "
               "Case-responsive nodes only — structural constraints (N1, N17) "
               "shown separately above. Increasing-side: risk and constraint "
               "nodes ranked by posterior. Decreasing-side: mitigations, "
               "systemic distortion corrections, and causal detectors.")

    def _driver_html(d, direction):
        arrow_color = "#A32D2D" if direction == "up" else "#3B6D11"
        val_color   = arrow_color
        return (
            f'<div style="display:grid;grid-template-columns:42px 1fr 70px;gap:14px;'
            f'align-items:center;padding:10px 0;border-bottom:1px solid rgba(0,0,0,.05)">'
            f'<div style="font-family:monospace;font-size:0.78rem;font-weight:600;'
            f'text-align:center;padding:4px 0;border-radius:5px;color:white;'
            f'background:{d["color"]}">N{d["nid"]}</div>'
            f'<div><div style="font-size:0.92rem;color:#1a1a1a">{d["short"]}</div>'
            f'<div style="font-family:\'Fraunces\',serif;font-style:italic;'
            f'font-size:0.74rem;color:#888;margin-top:1px">{d["type_label"]}</div></div>'
            f'<div style="font-family:monospace;font-size:1rem;font-weight:600;'
            f'text-align:right;color:{val_color}">{d["p"]*100:.1f}%</div>'
            f'</div>'
        )

    cdu, cdd = st.columns(2)
    with cdu:
        st.markdown(
            '<div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.16em;color:#888;margin-bottom:10px">'
            '<span style="color:#A32D2D;font-family:monospace;margin-right:6px">↑</span>'
            'Increasing DO risk</div>', unsafe_allow_html=True)
        if _drv_up:
            st.markdown("".join(_driver_html(d, "up") for d in _drv_up),
                        unsafe_allow_html=True)
        else:
            st.caption("No increasing-side drivers above threshold.")
    with cdd:
        st.markdown(
            '<div style="font-size:0.66rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:0.16em;color:#888;margin-bottom:10px">'
            '<span style="color:#3B6D11;font-family:monospace;margin-right:6px">↓</span>'
            'Decreasing DO risk</div>', unsafe_allow_html=True)
        if _drv_dn:
            st.markdown("".join(_driver_html(d, "dn") for d in _drv_dn),
                        unsafe_allow_html=True)
        else:
            st.caption("No decreasing-side drivers above threshold.")

    st.markdown("---")

    # ── Zone 3: Completeness + Doctrinal frame ───────────────────────────────
    cc, cdoc = st.columns(2)
    with cc:
        st.markdown("### Input completeness")
        st.caption("What's been entered, what's at default, what remains.")
        _icon = {
            "done":    ('✓', '#3B6D11', '#3B6D11'),
            "partial": ('~', '#BA7517', '#FAEEDA'),
            "empty":   ('○', '#9E9E9E', '#EFEDE7'),
        }
        rows = []
        for it in _comp:
            sym, fg, bg = _icon[it["status"]]
            rows.append(
                f'<div style="display:grid;grid-template-columns:24px 1fr auto;gap:12px;'
                f'align-items:center;padding:9px 0;border-bottom:1px solid rgba(0,0,0,.05);'
                f'font-size:0.92rem">'
                f'<div style="width:20px;height:20px;border-radius:50%;display:flex;'
                f'align-items:center;justify-content:center;font-family:monospace;'
                f'font-size:0.74rem;font-weight:700;color:{"white" if it["status"]=="done" else fg};'
                f'background:{bg if it["status"]!="done" else fg};'
                f'border:1px solid {fg}">{sym}</div>'
                f'<div>{it["label"]} '
                f'<span style="font-family:\'Fraunces\',serif;font-style:italic;'
                f'font-size:0.78rem;color:#888">— {it["detail"]}</span></div>'
                f'<div style="font-family:\'Fraunces\',serif;font-style:italic;'
                f'font-size:0.78rem;color:#888">{it["status"]}</div>'
                f'</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)

    with cdoc:
        st.markdown("### Doctrinal frame")
        st.caption("Active framework and connection-gate state under "
                   "Morris para 97 (or Ellis nexus, where applicable).")
        rows = [
            ("Framework",
             f'<span style="display:inline-block;font-size:0.74rem;padding:2px 9px;'
             f'border-radius:10px;background:{_doc["fw_color"]};color:white;'
             f'font-weight:500;margin-right:6px">{_doc["fw_label"]}</span>'
             f'Connection gate active'),
            ("Connection",
             f'{_doc["conn_label"]} · weight multiplier '
             f'<strong>{_doc["conn_mult"]:.2f}</strong>'),
            ("Tetrad",
             '<span style="font-family:\'Fraunces\',serif;font-style:italic;'
             'font-size:0.86rem;color:#3a3a3a">'
             'Gladue [1999], Morris 2021 ONCA 680, '
             'Ellis 2022 BCCA 278, Ewert 2018 SCC 30</span>'),
            ("QBism flags", _doc["qbism_summary"]),
            ("Engine",      "pgmpy Variable Elimination · 20-node DAG"),
        ]
        rows_html = []
        for k, v in rows:
            rows_html.append(
                f'<div style="display:grid;grid-template-columns:130px 1fr;gap:14px;'
                f'padding:9px 0;border-bottom:1px solid rgba(0,0,0,.05);font-size:0.92rem">'
                f'<div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;'
                f'color:#888;font-weight:600;align-self:center">{k}</div>'
                f'<div>{v}</div></div>'
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)

    st.markdown("---")

    # ── Zone 4: Continue working — shortcut row ──────────────────────────────
    st.markdown("### Continue working")
    st.caption("Jump into the workflow surfaces most relevant to the current case state.")

    # Determine the most useful next-action hint based on completeness
    _next = next((it for it in _comp if it["status"] == "empty"),
                 next((it for it in _comp if it["status"] == "partial"), None))
    if _next:
        _next_label  = f'Complete {_next["label"]}'
        _next_detail = _next["detail"]
    else:
        _next_label  = "Review inference"
        _next_detail = "All input surfaces populated."

    sc1, sc2, sc3 = st.columns(3)
    _shortcut_style = (
        "border:1px solid rgba(0,0,0,.10);border-radius:10px;padding:14px 18px;"
        "background:#FBFAF7;height:100%"
    )
    with sc1:
        st.markdown(
            f'<div style="{_shortcut_style}">'
            f'<div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;'
            f'color:#888;font-weight:600;margin-bottom:4px">Next action</div>'
            f'<div style="font-family:\'Fraunces\',serif;font-size:1.05rem;color:#1a1a1a;'
            f'font-weight:500">{_next_label}</div>'
            f'<div style="font-size:0.78rem;color:#888;margin-top:3px">{_next_detail}</div>'
            f'</div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(
            f'<div style="{_shortcut_style}">'
            f'<div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;'
            f'color:#888;font-weight:600;margin-bottom:4px">Drill in</div>'
            f'<div style="font-family:\'Fraunces\',serif;font-size:1.05rem;color:#1a1a1a;'
            f'font-weight:500">Inference posteriors</div>'
            f'<div style="font-size:0.78rem;color:#888;margin-top:3px">'
            f'Full 20-node breakdown with VE-computed probabilities.</div>'
            f'</div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(
            f'<div style="{_shortcut_style}">'
            f'<div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.14em;'
            f'color:#888;font-weight:600;margin-bottom:4px">Export</div>'
            f'<div style="font-family:\'Fraunces\',serif;font-size:1.05rem;color:#1a1a1a;'
            f'font-weight:500">Generate audit report</div>'
            f'<div style="font-size:0.78rem;color:#888;margin-top:3px">'
            f'PDF, DOCX, or TXT for legal review or viva.</div>'
            f'</div>', unsafe_allow_html=True)

# ── T1: Architecture ──────────────────────────────────────────────────────────
with TABS[1]:
    cl,cr=st.columns([3,1])
    with cl:
        opts={None:"— none —"};opts.update({n:f"N{n}: {NODE_META[n]['name']}" for n in range(1,21)})
        sel=st.selectbox("Inspect node",list(opts.keys()),format_func=lambda x:opts[x])
        st.pyplot(draw_dag(P,sel),use_container_width=True)
    with cr:
        if sel:
            m=NODE_META[sel];col=TC[m["type"]];p=P.get(sel,.5)
            st.markdown(f"""<div style="background:{col}18;border:1px solid {col}55;border-radius:12px;padding:1rem">
            <div style="font-size:.68rem;color:{col};font-weight:700">{TL[m['type']]}</div>
            <div style="font-size:1rem;font-weight:700;margin-top:4px">N{sel}: {m['name']}</div>
            <div style="font-size:2rem;font-weight:700;font-family:monospace;color:{col};margin:8px 0">{p*100:.1f}%</div>
            <div style="height:5px;background:#eee;border-radius:3px">
              <div style="width:{p*100:.0f}%;height:100%;background:{col};border-radius:3px"></div>
            </div></div>""",unsafe_allow_html=True)
        else:
            st.markdown("<div class='sh'>Node types</div>",unsafe_allow_html=True)
            for t,c in TC.items(): st.markdown(f"<span style='color:{c}'>●</span>&nbsp;{TL[t]}",unsafe_allow_html=True)
            st.markdown("---")
            if _empty:
                st.markdown(
                    f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
                    f"border-radius:12px;padding:.8rem 1.2rem;margin-bottom:1rem;"
                    f"display:flex;align-items:center;gap:1.5rem'>"
                    f"<div style='text-align:center;min-width:80px'>"
                    f"<div style='font-size:.7rem;color:#9E9E9E;margin-bottom:2px'>Node 20</div>"
                    f"<div style='font-size:1.6rem;font-weight:500;font-family:Fraunces,Georgia,serif;"
                    f"font-style:italic;color:#9E9E9E'>—</div>"
                    f"<div style='font-size:.78rem;font-family:Fraunces,serif;font-style:italic;"
                    f"color:#707070'>Awaiting case data</div>"
                    f"</div>"
                    f"<div style='flex:1'>"
                    f"<div style='font-size:.82rem;font-weight:500;margin-bottom:6px;color:#707070'>"
                    f"DO designation risk — posterior probability</div>"
                    f"<div style='height:5px;background:rgba(0,0,0,.06);border-radius:3px'></div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;font-size:.78rem;"
                    f"color:#9E9E9E;margin-top:6px'>"
                    f"Enter case profile, criminal record, or Gladue / SCE evidence to begin.</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(dobar(dp),unsafe_allow_html=True)

# ── T2: Case profile ──────────────────────────────────────────────────────────
with TABS[2]:
    # ════════════════════════════════════════════════════════════════════════
    # Profile tab — workbench-but-quieter input language (Mark 8 redesign)
    # All widget keys preserved; pev[N] math unchanged.
    # ════════════════════════════════════════════════════════════════════════

    # ── Tab title + caption ───────────────────────────────────────────────
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.7rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 4px 0'>"
        "Case profile</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:22px;line-height:1.6;"
        "max-width:880px'>"
        "Each field maps to one or more nodes in the network. Adjustments "
        "here drive Variable Elimination immediately; the live posterior "
        "strip below the form reflects the current state."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Case identifier row ───────────────────────────────────────────────
    _ci_col1, _ci_col2 = st.columns([2, 1])
    with _ci_col1:
        st.text_input("Case identifier", key="case_id",
                      placeholder="e.g. R v Smith")
    with _ci_col2:
        st.text_input("Jurisdiction / location", key="case_jur",
                      placeholder="e.g. Calgary · Alberta")

    # Visual separator below case-id
    st.markdown(
        "<div style='border-bottom:1px solid #EFEDE7;margin:12px 0 26px 0'></div>",
        unsafe_allow_html=True,
    )

    # ── Two-column body ───────────────────────────────────────────────────
    pev = {}
    c1, c2 = st.columns(2)

    # Severity-word colour helper for doctrinal captions
    def _sev_caption(severity_word, severity_color, body_text, nid_tag=None):
        nid_html = (f"<span style='font-family:JetBrains Mono,monospace;"
                    f"font-size:0.7rem;color:#9E9E9E;margin-right:6px'>{nid_tag}</span>"
                    if nid_tag else "")
        return (
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.78rem;color:#707070;margin-top:-4px;margin-bottom:10px;"
            f"line-height:1.5'>"
            f"{nid_html}"
            f"<span style='color:{severity_color};font-style:normal;font-weight:500'>"
            f"{severity_word}</span> — {body_text}"
            f"</div>"
        )

    # Section-head helper — emits a coloured stripe + title block
    def _section_head(stripe_color, title, subtitle, node_tags):
        return (
            f"<div style='display:grid;grid-template-columns:8px 1fr;gap:14px;"
            f"align-items:baseline;margin-bottom:14px;padding-bottom:8px;"
            f"border-bottom:1px solid #EFEDE7'>"
            f"<div style='width:4px;height:22px;border-radius:2px;align-self:center;"
            f"background:{stripe_color}'></div>"
            f"<div>"
            f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.05rem;"
            f"font-weight:500;color:#1A1A1A;letter-spacing:-0.005em'>{title}"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
            f"color:#707070;margin-left:8px;letter-spacing:0;font-weight:500'>"
            f"{node_tags}</span></div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.82rem;color:#707070;margin-top:2px;line-height:1.5'>"
            f"{subtitle}</div>"
            f"</div></div>"
        )

    with c1:
        # ─── Section: Offender characteristics (risk stripe) ───────────────
        st.markdown(
            _section_head(
                "#A32D2D",
                "Offender characteristics",
                "Demographic and clinical features mapped to risk and burnout nodes.",
                "N3 · N4 · N9 · N15"
            ),
            unsafe_allow_html=True,
        )

        age = st.slider("Age at sentencing", 18, 80, 35, key="age")
        # Doctrinal caption — severity word breaks out in mitigation green/warm/risk
        if age >= 55:
            sev_word, sev_col, sev_body = "Strong", "#3B6D11", f"burnout attenuation active ({age} years). Temporal distortion node weighted high."
        elif age >= 45:
            sev_word, sev_col, sev_body = "Moderate", "#BA7517", f"partial burnout attenuation ({age} years)."
        else:
            sev_word, sev_col, sev_body = "Minimal", "#707070", f"burnout attenuation — temporal distortion node weighted at default ({age} years)."
        st.markdown(_sev_caption(sev_word, sev_col, sev_body, "N15"),
                    unsafe_allow_html=True)

        identity = st.selectbox(
            "Identity background",
            ["Not recorded / unknown",
             "Indigenous — s.718.2(e) + Gladue applies",
             "Black — Morris IRCA framework",
             "Other racialized — Morris framework",
             "Non-racialized, socially disadvantaged — Ellis",
             "No identified systemic disadvantage"],
            key="id_bg")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>"
            "framework gate</div>",
            unsafe_allow_html=True,
        )

        pclr = st.slider("PCL-R score", 0, 40, 20, key="pclr")
        if pclr >= 30:
            sev_word, sev_col, sev_body = "High ≥30", "#A32D2D", "<em>Ewert/Larsen</em> adversarial-allegiance caveat APPLIES."
        elif pclr >= 20:
            sev_word, sev_col, sev_body = "Moderate", "#BA7517", "<em>Ewert/Larsen</em> caveat partially applies."
        else:
            sev_word, sev_col, sev_body = "Low", "#3B6D11", "below threshold for adversarial-allegiance concern."
        st.markdown(_sev_caption(sev_word, sev_col, sev_body, "N3"),
                    unsafe_allow_html=True)

        s99 = st.slider("Static-99R score", 0, 12, 3, key="s99")
        if s99 >= 6:
            sev_word, sev_col, sev_body = "High ≥6", "#A32D2D", "<em>Ewert</em> validation caveat applies independently of score."
        elif s99 >= 4:
            sev_word, sev_col, sev_body = "Moderate", "#BA7517", "<em>Ewert</em> validation caveat applies."
        else:
            sev_word, sev_col, sev_body = "Low", "#3B6D11", "<em>Ewert</em> validation caveat applies independently of score."
        st.markdown(_sev_caption(sev_word, sev_col, sev_body, "N4"),
                    unsafe_allow_html=True)

        violence = st.selectbox(
            "Serious violence history",
            ["None", "Minor/historical", "Moderate", "Serious", "Established pattern"],
            key="viol")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N2</div>",
            unsafe_allow_html=True,
        )

        fasd = st.selectbox(
            "FASD diagnosis",
            ["None / not assessed", "Suspected, undiagnosed", "Confirmed diagnosis"],
            key="fasd")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:18px;text-align:right'>N9</div>",
            unsafe_allow_html=True,
        )

        # ─── Section: Dynamic risk (risk stripe) ───────────────────────────
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            _section_head(
                "#A32D2D",
                "Dynamic risk",
                "Substance, peer, and stability features aggregated into the dynamic-risk composite.",
                "N18"
            ),
            unsafe_allow_html=True,
        )

        sub = st.selectbox(
            "Substance use",
            ["None / in remission", "Low", "Moderate", "High — dependency"],
            key="sub")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>"
            "→ N18</div>",
            unsafe_allow_html=True,
        )

        peers = st.selectbox(
            "Antisocial peer associations",
            ["None identified", "Some — limited", "Strong — primary network"],
            key="peers")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>"
            "→ N18</div>",
            unsafe_allow_html=True,
        )

        stab = st.selectbox(
            "Employment / housing stability",
            ["Stable", "Marginal", "Unstable / homeless"],
            key="stab")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>"
            "→ N18</div>",
            unsafe_allow_html=True,
        )

    with c2:
        # ─── Section: Procedural integrity (distortion stripe) ─────────────
        st.markdown(
            _section_head(
                "#185FA5",
                "Procedural integrity",
                "Distortion-typed nodes — encode whether the record was generated under coercion, contamination, or systemic bias.",
                "N5 · N6 · N7 · N12 · N14 · N16"
            ),
            unsafe_allow_html=True,
        )

        det = st.slider("Pre-trial detention (days)", 0, 730, 60, key="det")
        if det > 180:
            sev_word, sev_col, sev_body = "Severe", "#A32D2D", f"<em>Antic</em> [2017] coercive-plea cascade risk acute ({det} days)."
        elif det > 90:
            sev_word, sev_col, sev_body = "High", "#A32D2D", f"<em>Antic</em> [2017] coercive-plea cascade risk active ({det} days)."
        elif det > 30:
            sev_word, sev_col, sev_body = "Moderate", "#BA7517", f"<em>Antic</em> [2017] coercive-plea cascade risk emerging ({det} days)."
        else:
            sev_word, sev_col, sev_body = "Low", "#3B6D11", f"below threshold for coercive-plea concern ({det} days)."
        st.markdown(_sev_caption(sev_word, sev_col, sev_body, "N7"),
                    unsafe_allow_html=True)

        counsel = st.selectbox(
            "Quality of defence counsel",
            ["Adequate", "Marginal",
             "Inadequate — no cultural investigation",
             "Ineffective — constitutional breach"],
            key="counsel")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N6</div>",
            unsafe_allow_html=True,
        )

        gr = st.selectbox(
            "Gladue / SCE report commissioned",
            ["Yes — full report before court",
             "Partial / summary only",
             "No report commissioned",
             "Report commissioned, disregarded"],
            key="gr")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N12</div>",
            unsafe_allow_html=True,
        )

        tools = st.selectbox(
            "Risk tools applied",
            ["Culturally validated only", "Mix — partially qualified",
             "Standard, no cultural qualification", "No actuarial tools"],
            key="tools")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N5</div>",
            unsafe_allow_html=True,
        )

        pol = st.selectbox(
            "Over-policing indicator",
            ["No evidence", "Some — marginal",
             "Strong — documented over-surveillance"],
            key="pol")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N14</div>",
            unsafe_allow_html=True,
        )

        prov = st.selectbox(
            "Province of prosecution",
            ["Low DO designation rate", "Medium rate", "High DO designation rate"],
            key="prov")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:18px;text-align:right'>N16</div>",
            unsafe_allow_html=True,
        )

        # ─── Section: Rehabilitative context (mitigation stripe) ───────────
        st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
        st.markdown(
            _section_head(
                "#3B6D11",
                "Rehabilitative context",
                "Cultural programming availability and engagement — Natomagan §48 frames absence as systemic, not individual.",
                "N11 · N19"
            ),
            unsafe_allow_html=True,
        )

        prog = st.selectbox(
            "Indigenous / cultural programming",
            ["Yes — full culturally grounded",
             "Limited availability",
             "No culturally appropriate programming"],
            key="prog")
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.78rem;color:#707070;margin-top:-4px;margin-bottom:10px;"
            "line-height:1.5'>"
            "<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-right:6px'>N11</span>"
            "Per <em>Natomagan</em> 2022 ABCA 48 — absence is systemic failure, "
            "not an offender characteristic."
            "</div>",
            unsafe_allow_html=True,
        )

        rehab = st.selectbox(
            "Rehabilitation engagement",
            ["Strong — consistent", "Moderate", "Minimal",
             "None — apparent refusal", "Anomalously positive (gaming risk)"],
            key="rehab")
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N19</div>",
            unsafe_allow_html=True,
        )

    # ── Posterior calculation (preserved byte-for-byte) ────────────────────
    ir=identity in ["Indigenous — s.718.2(e) + Gladue applies","Black — Morris IRCA framework","Other racialized — Morris framework"]
    pev[2]={"None":.08,"Minor/historical":.25,"Moderate":.50,"Serious":.78,"Established pattern":.90}[violence]
    pev[3]=.82 if pclr>=30 else .55 if pclr>=20 else .30 if pclr>=10 else .12
    pev[4]=.82 if s99>=6 else .55 if s99>=4 else .32 if s99>=2 else .12
    pev[5]={"Culturally validated only":.10,"Mix — partially qualified":.45,"Standard, no cultural qualification":.85 if ir else .40,"No actuarial tools":.15}[tools]
    pev[6]={"Adequate":.15,"Marginal":.45,"Inadequate — no cultural investigation":.72,"Ineffective — constitutional breach":.90}[counsel]
    pev[7]=.85 if det>180 else .70 if det>90 else .40 if det>30 else .15
    pev[9]={"None / not assessed":.15,"Suspected, undiagnosed":.50,"Confirmed diagnosis":.88}[fasd]
    pev[10]=min(.90,.45+(.20 if "Indigenous" in identity else 0))
    pev[11]={"Yes — full culturally grounded":.10,"Limited availability":.55,"No culturally appropriate programming":.85}[prog]
    pev[12]={"Yes — full report before court":.15,"Partial / summary only":.50,"No report commissioned":.82,"Report commissioned, disregarded":.92}[gr]
    pev[13]=.75 if rehab=="Anomalously positive (gaming risk)" else .22
    pev[14]={"No evidence":.15,"Some — marginal":.50,"Strong — documented over-surveillance":.85}[pol]
    pev[15]=.85 if age>=55 else .70 if age>=45 else .40 if age>=35 else .20
    pev[16]={"Low DO designation rate":.20,"Medium rate":.45,"High DO designation rate":.72}[prov]
    sv={"None / in remission":.15,"Low":.35,"Moderate":.60,"High — dependency":.80}[sub]
    pv={"None identified":.10,"Some — limited":.35,"Strong — primary network":.65}[peers]
    stv={"Stable":.10,"Marginal":.40,"Unstable / homeless":.70}[stab]
    pev[18]=float(np.clip((sv+pv+stv)/3+.05,.05,.92))
    rv={"Strong — consistent":.10,"Moderate":.35,"Minimal":.60,"None — apparent refusal":.80,"Anomalously positive (gaming risk)":.30}[rehab]
    pev[19]=float(np.clip(rv+(.12 if prog=="No culturally appropriate programming" else 0),.05,.90))
    st.session_state.profile_ev=pev
    run_inf();P=st.session_state.posteriors
    bl2,bc2,_=rb(P[20])

    # ── Slim live-result strip — replaces st.success() ────────────────────
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    _band_text = {
        "Low": "belief largely resolved",
        "Moderate": "belief partially resolved",
        "Elevated": "belief shifted toward designation",
        "High": "strong indication of designation",
    }.get(bl2, bl2)
    if _empty:
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr auto;"
            "align-items:center;gap:18px;background:#FBFAF7;"
            "border:1px solid #E0DDD6;border-radius:8px;"
            "padding:11px 18px;margin-top:24px;margin-bottom:0'>"
            "<div style='font-size:0.82rem;color:#9E9E9E;font-weight:500'>"
            "Node 20 · DO designation risk"
            "<span style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:1.05rem;font-weight:500;color:#9E9E9E;margin-left:10px'>—</span>"
            "</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#707070'>Awaiting case data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr auto;"
            f"align-items:center;gap:18px;background:linear-gradient(90deg,"
            f"#E2EBD8 0%, #EAF3DE 50%, #F7F5F2 100%);border:1px solid #B8CDA8;"
            f"border-radius:8px;padding:11px 18px'>"
            f"<div style='font-size:0.82rem;color:#3B6D11;font-weight:500'>"
            f"Node 20 · DO designation risk"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:#2F5C2A;margin-left:8px'>{P[20]*100:.1f}%</span>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.86rem;color:#2F5C2A'>{bl2} — {_band_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── T3: Gladue ────────────────────────────────────────────────────────────────
with TABS[5]:
    # ════════════════════════════════════════════════════════════════════════
    # Gladue tab — checklist-of-factors layout (Mark 8 redesign)
    # All widget keys (gl_{factor_id}) preserved via the GF iteration.
    # ════════════════════════════════════════════════════════════════════════

    # ── Tab title + caption ───────────────────────────────────────────────
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.7rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 4px 0'>"
        "Gladue factors</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:18px;line-height:1.6;"
        "max-width:880px'>"
        "Eighteen factor categories drawn from the Gladue / Ipeelee jurisprudence, "
        "organised into the seven thematic areas a Gladue report is expected "
        "to address. Check each factor for which the case file contains "
        "substantive evidence — not whether the offender belongs to a "
        "population to which the factor commonly applies."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Doctrinal anchor strip ────────────────────────────────────────────
    st.markdown(
        "<div style='background:#EAF3DE;border:1px solid #B8CDA8;"
        "border-left:3px solid #3B6D11;border-radius:6px;padding:10px 18px;"
        "margin-bottom:22px;font-size:0.84rem;color:#3A3A3A;line-height:1.55;"
        "max-width:880px'>"
        "<strong style='color:#3B6D11;font-weight:600'>Binding authorities.</strong> "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Gladue</em> [1999] 1 SCR 688 · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Ipeelee</em> [2012] 1 SCR 433 · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Anderson</em> [2014] 2 SCR 167. "
        "No causation requirement — discernible nexus per "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>Ipeelee</em> §83 "
        "is sufficient to engage the framework."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Build factor groupings + run checkboxes ────────────────────────────
    # Group factors by section, preserving original col-assignment
    secs = {}
    for f in GF:
        secs.setdefault(f["sec"], []).append(f)

    # ── Pre-render: collect checkbox state in a hidden two-column wrapper ──
    # We use Streamlit's native columns + checkboxes (preserves keys), but
    # wrap each section in styled markdown shells around the native widgets.
    cg = set()
    c1, c2 = st.columns(2)

    # Section type → stripe colour map.
    # Five mitigation-green sections, one mitigation-green (Cultural), one
    # distortion-blue (Systemic justice — feeds N12/N14 distortion correction).
    _section_color = {
        "Intergenerational trauma":   "#3B6D11",   # mitigation
        "Cultural disconnection":     "#3B6D11",   # mitigation
        "Childhood & family":         "#3B6D11",   # mitigation
        "Socioeconomic":              "#3B6D11",   # mitigation
        "Substance & mental health":  "#3B6D11",   # mitigation
        "Systemic justice":           "#185FA5",   # distortion
    }
    # Subtitle helper — small descriptive line under each section title
    _section_subtitle = {
        "Intergenerational trauma":   "Direct and inherited trauma — residential schools, Sixties Scoop, displacement.",
        "Cultural disconnection":     "Severance from language, identity, and ceremonial practice.",
        "Childhood & family":         "Family violence and care-system involvement during formative years.",
        "Socioeconomic":              "Poverty, housing, employment, and educational deprivation.",
        "Substance & mental health":  "Trauma-linked substance use and untreated mental health conditions.",
        "Systemic justice":           "Over-policing and prior procedural failures — feed distortion-corrections.",
    }
    # Aggregate node-tag for each section
    _section_nodes = {
        "Intergenerational trauma":   "→ N10",
        "Cultural disconnection":     "→ N11 · N12",
        "Childhood & family":         "→ N10",
        "Socioeconomic":              "→ N10 · N18",
        "Substance & mental health":  "→ N10 · N18",
        "Systemic justice":           "→ N12 · N14",
    }

    def _gladue_section_open(stripe_color, title, subtitle, node_tags, count_str, has_checked):
        """Render the opening div of a section card, before native checkboxes."""
        count_bg = "#EAF3DE" if has_checked else "#FFFFFF"
        count_color = "#3B6D11" if has_checked else "#707070"
        count_border = "#B8CDA8" if has_checked else "#E0DDD6"
        return (
            f"<div style='background:#FFFFFF;border:1px solid #E0DDD6;"
            f"border-radius:8px;overflow:hidden;margin-bottom:18px'>"
            f"<div style='display:grid;grid-template-columns:4px 1fr auto;"
            f"gap:14px;align-items:center;padding:12px 16px;"
            f"background:#FBFAF7;border-bottom:1px solid #EFEDE7'>"
            f"<div style='width:4px;height:28px;border-radius:2px;align-self:center;"
            f"background:{stripe_color}'></div>"
            f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.98rem;"
            f"font-weight:500;color:#1A1A1A'>{title}"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            f"color:#9E9E9E;margin-left:6px;font-weight:500'>{node_tags}</span></div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
            f"padding:2px 8px;border-radius:9px;background:{count_bg};"
            f"color:{count_color};border:1px solid {count_border};font-weight:500'>"
            f"{count_str}</div>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.78rem;color:#707070;padding:8px 16px 4px 16px;"
            f"line-height:1.5'>{subtitle}</div>"
            f"<div style='padding:4px 12px 8px 12px'>"
        )

    # Track per-section render order matching original col= assignment
    # Original: col=1 → c1 left column, col=2 → c2 right column
    _ordered_sections = []
    _seen = set()
    for f in GF:
        if f["sec"] not in _seen:
            _seen.add(f["sec"])
            _ordered_sections.append((f["sec"], f["col"]))

    for sec, col in _ordered_sections:
        facs = secs[sec]
        target_col = c1 if col == 1 else c2

        # Pre-compute count
        n_checked = sum(1 for f in facs if f["id"] in st.session_state.gladue_checked)
        n_total = len(facs)
        count_str = f"{n_checked} of {n_total}"
        has_checked = n_checked > 0

        with target_col:
            # Open the section card (HTML)
            st.markdown(
                _gladue_section_open(
                    _section_color.get(sec, "#3B6D11"),
                    sec,
                    _section_subtitle.get(sec, ""),
                    _section_nodes.get(sec, ""),
                    count_str,
                    has_checked,
                ),
                unsafe_allow_html=True,
            )

            # Native Streamlit checkboxes — preserves keys, state, behaviour.
            # The label includes node weight so the user sees contribution inline.
            for f in facs:
                lbl = f"{f['l']} · N{f['n']} (+{f['w']*100:.0f}%)"
                if st.checkbox(
                    lbl,
                    key=f"gl_{f['id']}",
                    value=f["id"] in st.session_state.gladue_checked,
                ):
                    cg.add(f["id"])

            # Close the section card
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Update session state with current checked set
    st.session_state.gladue_checked = cg
    run_inf()
    P = st.session_state.posteriors
    bl, _bc, _bg = rb(P[20])

    # ── Coverage summary panel ────────────────────────────────────────────
    n_total_factors = len(GF)
    n_checked_factors = len(cg)
    n_total_sections = len(_ordered_sections)
    n_active_sections = sum(
        1 for sec, _col in _ordered_sections
        if any(f["id"] in cg for f in secs[sec])
    )

    st.markdown("<div style='margin:24px 0 12px 0'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
        f"border-radius:8px;padding:18px 22px;margin-bottom:20px'>"
        f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.15rem;"
        f"font-weight:500;color:#1A1A1A;margin-bottom:6px'>"
        f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
        f"color:#3B6D11'>{n_checked_factors}</span> of "
        f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
        f"color:#3B6D11'>{n_total_factors}</span> factors checked across "
        f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
        f"color:#3B6D11'>{n_active_sections}</span> of "
        f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
        f"color:#3B6D11'>{n_total_sections}</span> sections</div>"
        f"<div style='font-family:Fraunces,serif;font-style:italic;"
        f"font-size:0.86rem;color:#707070;line-height:1.5'>"
        f"The <em style='color:#1A1A1A'>Anderson</em> [2014] §22 obligation is to address all "
        f"reasonably available factors in a Gladue report. Coverage above 6 "
        f"factors with breadth across multiple thematic areas is a typical "
        f"threshold for a well-supported analysis."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Slim live-result strip ────────────────────────────────────────────
    _band_text = {
        "Low": f"belief largely resolved · {n_checked_factors} Gladue factor(s) active",
        "Moderate": f"belief partially resolved · {n_checked_factors} factor(s)",
        "Elevated": f"belief shifted · {n_checked_factors} factor(s)",
        "High": f"strong indication · {n_checked_factors} factor(s)",
    }.get(bl, bl)
    if _empty:
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr auto;"
            "align-items:center;gap:18px;background:#FBFAF7;"
            "border:1px solid #E0DDD6;border-radius:8px;"
            "padding:11px 18px;margin-top:24px;margin-bottom:0'>"
            "<div style='font-size:0.82rem;color:#9E9E9E;font-weight:500'>"
            "Node 20 · DO designation risk"
            "<span style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:1.05rem;font-weight:500;color:#9E9E9E;margin-left:10px'>—</span>"
            "</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#707070'>Awaiting case data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr auto;"
            f"align-items:center;gap:18px;background:linear-gradient(90deg,"
            f"#E2EBD8 0%, #EAF3DE 50%, #F7F5F2 100%);border:1px solid #B8CDA8;"
            f"border-radius:8px;padding:11px 18px;margin-bottom:24px'>"
            f"<div style='font-size:0.82rem;color:#3B6D11;font-weight:500'>"
            f"Node 20 · DO designation risk"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:#2F5C2A;margin-left:8px'>{P[20]*100:.1f}%</span>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.86rem;color:#2F5C2A'>{bl} — {_band_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Ipeelee §60 reminder at bottom ────────────────────────────────────
    st.markdown(
        "<div style='padding:14px 20px;background:#FBFAF7;border:1px solid #E0DDD6;"
        "border-radius:8px;font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;line-height:1.55;max-width:880px'>"
        "<strong style='color:#1A1A1A;font-style:normal'>Reminder — Ipeelee §60.</strong> "
        "Counsel and the court should consider the unique systemic and "
        "background factors which may have played a role in bringing the "
        "offender before the court, even where they fall outside the "
        "categories tabulated above. The list is illustrative, not "
        "exhaustive — case-specific factors that do not map cleanly to a "
        "checkbox here may still be entered in <em style='color:#3A3A3A'>Intake (Chat)</em> "
        "for narrative inclusion in the report."
        "</div>",
        unsafe_allow_html=True,
    )

# ── T4: Morris/Ellis SCE ──────────────────────────────────────────────────────
with TABS[6]:
    # ════════════════════════════════════════════════════════════════════════
    # SCE tab — full visual rebuild (Mark 8)
    # All session-state keys preserved: scefw, conn, enex, sce_values, sce_checked
    # All widget keys preserved: scefw_r, conn_s, enex_s, sce_{factor_id}
    # cmult() / emult() / SF filter logic byte-for-byte preserved.
    # ════════════════════════════════════════════════════════════════════════

    # ── Tab title + caption ───────────────────────────────────────────────
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.7rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 4px 0'>"
        "Social context evidence</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:18px;line-height:1.6;"
        "max-width:880px'>"
        "<em>Morris</em> and <em>Ellis</em> are the two principal frameworks "
        "Canadian courts use to weigh systemic and structural factors in "
        "sentencing. Each factor is entered as a continuous evidentiary "
        "strength (0 = absent, 1 = fully established), allowing partial-"
        "evidence cases to register without the binary on/off limitation "
        "of a checklist."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Doctrinal anchor strip ────────────────────────────────────────────
    st.markdown(
        "<div style='background:#E8F0FA;border:1px solid #C7D3E5;"
        "border-left:3px solid #185FA5;border-radius:6px;padding:10px 18px;"
        "margin-bottom:22px;font-size:0.84rem;color:#3A3A3A;line-height:1.55;"
        "max-width:880px'>"
        "<strong style='color:#185FA5;font-weight:600'>Binding authorities.</strong> "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Morris</em> 2021 ONCA 680 — racialized offenders · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Ellis</em> 2022 BCCA 278 — non-racialized deprivation · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Anderson</em> [2014] 2 SCR 167 — IRCA framework. "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>Morris</em> para 97 establishes the connection-gate "
        "doctrine governing how systemic context is weighted into sentencing."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Command panel: framework selector + connection gate ───────────────
    cmd_c1, cmd_c2 = st.columns([1, 2])
    with cmd_c1:
        st.markdown(
            "<div style='font-size:0.66rem;text-transform:uppercase;"
            "letter-spacing:0.14em;color:#707070;font-weight:600;"
            "margin-bottom:6px'>Active framework</div>",
            unsafe_allow_html=True,
        )
        fw = st.radio(
            "Framework",
            ["Morris", "Ellis", "Both"],
            index=["morris","ellis","both"].index(
                (st.session_state.scefw or "morris").lower()
                if (st.session_state.scefw or "morris").lower() in ["morris","ellis","both"]
                else 0
            ),
            key="scefw_r",
            label_visibility="collapsed",
            horizontal=True,
        )
        st.session_state.scefw = fw.lower()
        # Framework-specific descriptive caption
        _fw_caption = {
            "morris": "<em>Morris</em> applies to racialized offenders.",
            "ellis":  "<em>Ellis</em> applies to non-racialized offenders with deprivation backgrounds.",
            "both":   "<strong style='font-style:normal;color:#1A1A1A'>Both</strong> shows all sections — useful for full-record review and audit preparation.",
        }.get(st.session_state.scefw, "")
        st.markdown(
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.78rem;color:#707070;margin-top:6px;line-height:1.55'>"
            f"{_fw_caption}</div>",
            unsafe_allow_html=True,
        )

    with cmd_c2:
        if (st.session_state.scefw or "morris").lower() != "ellis":
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "align-items:baseline;margin-bottom:6px'>"
                "<span style='font-size:0.86rem;font-weight:600;color:#1A1A1A'>"
                "Morris para 97 — connection gate</span>"
                "<span style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.78rem;color:#707070'>R v Morris §97</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            conn = st.select_slider(
                "Connection strength",
                ["none","absent","weak","moderate","strong","direct"],
                value=st.session_state.conn, key="conn_s",
                label_visibility="collapsed",
            )
            st.session_state.conn = conn
            _cm = cmult()
            _cm_desc = ("full belief revision obligation" if _cm >= .9
                        else "partial belief revision obligation" if _cm >= .6
                        else "limited belief revision obligation")
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;font-size:0.84rem;margin-top:6px'>"
                f"<span style='color:#707070'>Weight multiplier (cmult)</span>"
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-weight:600;color:#185FA5'>{_cm:.2f}</span></div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#707070;margin-top:2px;line-height:1.5'>"
                f"{_cm_desc}. SCE corrections weighted at {_cm*100:.0f}% of established evidentiary strength."
                f"</div>",
                unsafe_allow_html=True,
            )

        if (st.session_state.scefw or "morris").lower() != "morris":
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "align-items:baseline;margin-top:14px;margin-bottom:6px'>"
                "<span style='font-size:0.86rem;font-weight:600;color:#1A1A1A'>"
                "Ellis deprivation nexus</span>"
                "<span style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.78rem;color:#707070'>R v Ellis 2022 BCCA 278</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            nx_v = st.selectbox(
                "Ellis deprivation nexus",
                ["none","peripheral","relevant","central"],
                index=["none","peripheral","relevant","central"].index(st.session_state.enex),
                key="enex_s",
                label_visibility="collapsed",
            )
            st.session_state.enex = nx_v
            _em = emult()
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;font-size:0.84rem;margin-top:6px'>"
                f"<span style='color:#707070'>Weight multiplier (emult)</span>"
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-weight:600;color:#534AB7'>{_em:.2f}</span></div>",
                unsafe_allow_html=True,
            )

    # Visual separator
    st.markdown(
        "<div style='border-top:1px solid #EFEDE7;margin:24px 0 18px 0'></div>",
        unsafe_allow_html=True,
    )

    # ── SCE factors as continuous sliders (logic preserved) ──────────────
    ss = {}
    for f in SF:
        fw2 = (st.session_state.scefw or "morris").lower()
        show = (fw2 == "both"
                or (fw2 == "morris" and f["fw"] != "ellis")
                or (fw2 == "ellis" and f["fw"] != "morris"))
        if show:
            ss.setdefault(f["sec"], []).append(f)

    sce_vals = dict(st.session_state.get("sce_values", {}))

    # Section type header — explains the slider register
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.86rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:880px'>"
        "Slide each factor from "
        "<strong style='font-style:normal;color:#1A1A1A'>0</strong> (absent) → "
        "<strong style='font-style:normal;color:#1A1A1A'>0.5</strong> (partial) → "
        "<strong style='font-style:normal;color:#1A1A1A'>1.0</strong> (fully established). "
        "Partial values encode degrees of evidentiary strength rather than binary presence."
        "</div>",
        unsafe_allow_html=True,
    )

    # Section → framework mapping for visual treatment
    def _sce_section_framework(facs):
        """Return (framework_key, stripe_color, pill_label, pill_color)."""
        # Determine which framework owns this section based on its factors
        fws = set(f["fw"] for f in facs)
        if fws == {"ellis"}:
            return ("ellis", "#534AB7", "ELLIS", "#ECE9F7")
        elif fws == {"both"}:
            return ("both", "#993C1D", "BOTH", "#FAEEDA")
        else:
            return ("morris", "#185FA5", "MORRIS", "#E8F0FA")

    # Render section cards in a 3-column grid (preserves cols3 layout intent)
    cols3 = st.columns(3)
    for i, (sec, facs) in enumerate(ss.items()):
        fw_key, stripe_col, pill_label, pill_bg = _sce_section_framework(facs)
        # Pill text colour matches stripe
        pill_text_col = stripe_col
        # Pill border
        pill_border = {"#185FA5": "#C7D3E5", "#534AB7": "#C9C0E5", "#993C1D": "#E5CC95"}.get(stripe_col, "#E0DDD6")

        with cols3[i % 3]:
            # Section header card
            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #E0DDD6;"
                f"border-radius:8px;overflow:hidden;margin-bottom:16px'>"
                f"<div style='display:grid;grid-template-columns:4px 1fr auto;"
                f"gap:12px;align-items:center;padding:12px 14px;"
                f"background:#FBFAF7;border-bottom:1px solid #EFEDE7'>"
                f"<div style='width:4px;height:24px;border-radius:2px;align-self:center;"
                f"background:{stripe_col}'></div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.95rem;"
                f"font-weight:500;color:#1A1A1A'>{sec}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.66rem;"
                f"padding:2px 8px;border-radius:9px;background:{pill_bg};"
                f"color:{pill_text_col};border:1px solid {pill_border};font-weight:600;"
                f"letter-spacing:0.04em'>{pill_label}</div>"
                f"</div>"
                f"<div style='padding:8px 14px 6px 14px'>",
                unsafe_allow_html=True,
            )

            # Render the native sliders inside the card body
            for f in facs:
                fw2 = (st.session_state.scefw or "morris").lower()
                col_sce = "#185FA5" if (fw2 == "morris" or f["fw"] == "both") else "#534AB7"
                cur_val = sce_vals.get(f["id"], 0.0)
                v = st.slider(
                    f"{f['l']} · N{f['n']}",
                    0.0, 1.0, float(cur_val), 0.05,
                    key=f"sce_{f['id']}",
                    format="%.2f"
                )
                sce_vals[f["id"]] = v
                if v > 0.01:
                    pct = (f"+{v*f['w']*cmult()*100:.1f}pp"
                           if fw2 != "ellis"
                           else f"+{v*f['w']*emult()*100:.1f}pp")
                    # Tier dots — 5-segment progress indicator
                    n_dots = min(int(v*5)+1, 5)
                    dots_html = "".join([
                        f"<span style='display:inline-block;width:5px;height:5px;"
                        f"border-radius:50%;background:{col_sce};margin-right:3px'></span>"
                        if d < n_dots else
                        f"<span style='display:inline-block;width:5px;height:5px;"
                        f"border-radius:50%;background:#E0DDD6;margin-right:3px'></span>"
                        for d in range(5)
                    ])
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:baseline;margin-top:-12px;margin-bottom:8px;"
                        f"font-size:0.74rem'>"
                        f"<span style='color:#707070;font-family:Fraunces,serif;"
                        f"font-style:italic'>"
                        f"<span style='display:inline-block;vertical-align:middle;"
                        f"margin-right:6px'>{dots_html}</span>"
                        f"{v*100:.0f}% established</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;"
                        f"color:{col_sce};font-weight:500'>{pct}</span></div>",
                        unsafe_allow_html=True,
                    )

            # Close section card
            st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.sce_values = sce_vals
    # Keep sce_checked in sync for backwards compatibility with QBism diagnostics
    st.session_state.sce_checked = {fid for fid, v in sce_vals.items() if v > 0.01}
    run_inf()
    P = st.session_state.posteriors
    bl_sce, _bc_sce, _bg_sce = rb(P[20])

    # ── Slim live-result strip ───────────────────────────────────────────
    _n_active = len(st.session_state.sce_checked)
    _band_text_sce = {
        "Low":      f"belief largely resolved · {_n_active} factor(s) active",
        "Moderate": f"belief partially resolved · {_n_active} factor(s)",
        "Elevated": f"belief shifted · {_n_active} factor(s)",
        "High":     f"strong indication · {_n_active} factor(s)",
    }.get(bl_sce, bl_sce)
    if _empty:
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr auto;"
            "align-items:center;gap:18px;background:#FBFAF7;"
            "border:1px solid #E0DDD6;border-radius:8px;"
            "padding:11px 18px;margin-top:24px;margin-bottom:0'>"
            "<div style='font-size:0.82rem;color:#9E9E9E;font-weight:500'>"
            "Node 20 · DO designation risk"
            "<span style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:1.05rem;font-weight:500;color:#9E9E9E;margin-left:10px'>—</span>"
            "</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#707070'>Awaiting case data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr auto;"
            f"align-items:center;gap:18px;background:linear-gradient(90deg,"
            f"#E2EBD8 0%, #EAF3DE 50%, #F7F5F2 100%);border:1px solid #B8CDA8;"
            f"border-radius:8px;padding:11px 18px;margin-top:24px'>"
            f"<div style='font-size:0.82rem;color:#3B6D11;font-weight:500'>"
            f"Node 20 · DO designation risk"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:#2F5C2A;margin-left:8px'>{P[20]*100:.1f}%</span>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.86rem;color:#2F5C2A'>{bl_sce} — {_band_text_sce}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── T5: Evidence review ───────────────────────────────────────────────────────
with TABS[7]:
    # ════════════════════════════════════════════════════════════════════════
    # Risk & Distortions tab — visual rebuild (Mark 8)
    # All logic preserved: ev_nodes filter, manual_ev override detection,
    # reset-to-priors button. Widget keys preserved (ev_{nid}, rst).
    # ════════════════════════════════════════════════════════════════════════

    # ── Tab title + caption ───────────────────────────────────────────────
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.7rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 4px 0'>"
        "Risk &amp; distortions</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:18px;line-height:1.6;"
        "max-width:880px'>"
        "Per-node manual override surface. Each slider sets P(High) for an "
        "evidence-bearing node directly, bypassing the upstream computations "
        "from Profile, Gladue, SCE, Criminal Record, and Documents. Use "
        "sparingly and only with case-specific evidence that warrants "
        "departure from the network's automatic calculations."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Scope disclaimer (override register explanation) ──────────────────
    st.markdown(
        "<div style='background:#E8F0FA;border:1px solid #C7D3E5;"
        "border-left:3px solid #185FA5;border-radius:6px;padding:12px 18px;"
        "margin-bottom:22px;font-size:0.86rem;color:#3A3A3A;line-height:1.55;"
        "max-width:880px'>"
        "<strong style='color:#185FA5;font-weight:600'>Override surface.</strong> "
        "Adjustments here override values computed from other tabs. The network "
        "preserves the original computed value alongside the manual override. "
        "Document the reasoning for any override in the Report tab — the audit "
        "log captures both the original computed value and your adjustment. "
        "Reset all overrides to defaults below."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Compute override status from current state ────────────────────────
    ev_nodes = [n for n in NODE_META if NODE_META[n]["ev"]]
    n_overrides = sum(
        1 for nid in ev_nodes
        if abs(P.get(nid, .5) - st.session_state.profile_ev.get(nid, .5)) > .015
        and nid in st.session_state.manual_ev
    )
    n_total = len(ev_nodes)

    # ── Override status + reset action ────────────────────────────────────
    osc1, osc2 = st.columns([3, 1])
    with osc1:
        _override_color = "#BA7517" if n_overrides > 0 else "#3B6D11"
        st.markdown(
            f"<div style='background:#FFFFFF;border:1px solid #E0DDD6;"
            f"border-radius:8px;padding:14px 20px;margin-bottom:20px'>"
            f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.1rem;"
            f"font-weight:500;color:#1A1A1A;margin-bottom:4px'>"
            f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
            f"color:{_override_color}'>{n_overrides}</span> of "
            f"<span style='font-family:JetBrains Mono,monospace;font-weight:600;"
            f"color:#1A1A1A'>{n_total}</span> nodes manually overridden</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.84rem;color:#707070;line-height:1.5'>"
            f"Manual overrides represent your judgment that the case file "
            f"warrants a departure from the value computed from Profile / "
            f"Gladue / SCE inputs."
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Build the slider rows by node type ────────────────────────────────
    # Group evidence-bearing nodes by NODE_META type
    nodes_by_type = {}
    for nid in ev_nodes:
        t = NODE_META[nid]["type"]
        nodes_by_type.setdefault(t, []).append(nid)

    # Section metadata for each node type
    _type_meta = {
        "risk":       ("#A32D2D", "Risk factor nodes",         "Pattern-of-violence and clinical risk indicators."),
        "distortion": ("#185FA5", "Systemic distortion nodes", "Procedural and structural distortions affecting the record."),
        "mitigation": ("#3B6D11", "Mitigation",                "Background factors weighing against designation."),
        "dual":       ("#534AB7", "Dual (risk + mitigation)",  "Factors that elevate risk if untreated, mitigate if accommodated."),
        "special":    ("#0F6E56", "Causal detector",           "Specialised detector nodes — pattern-recognition rather than direct evidence."),
    }
    # Render order: risk first, then distortion (the largest two), then mitigation/dual/special
    _render_order = ["risk", "distortion", "mitigation", "dual", "special"]

    man = dict(st.session_state.manual_ev)

    # Render the section cards in 2 columns
    rsc_c1, rsc_c2 = st.columns(2)

    def _render_section(nodes, type_key, container):
        """Render a section card with coloured stripe + native sliders inside."""
        if not nodes:
            return
        stripe_col, title, subtitle = _type_meta.get(type_key, ("#707070", type_key, ""))
        # Build the node-tag string
        nodes_tag = " · ".join(f"N{n}" for n in nodes)
        # Count overrides in this section
        n_section_overrides = sum(
            1 for nid in nodes
            if abs(P.get(nid, .5) - st.session_state.profile_ev.get(nid, .5)) > .015
            and nid in st.session_state.manual_ev
        )
        with container:
            st.markdown(
                f"<div style='background:#FFFFFF;border:1px solid #E0DDD6;"
                f"border-radius:8px;overflow:hidden;margin-bottom:18px'>"
                f"<div style='display:grid;grid-template-columns:4px 1fr auto;"
                f"gap:14px;align-items:center;padding:12px 16px;"
                f"background:#FBFAF7;border-bottom:1px solid #EFEDE7'>"
                f"<div style='width:4px;height:28px;border-radius:2px;align-self:center;"
                f"background:{stripe_col}'></div>"
                f"<div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.98rem;"
                f"font-weight:500;color:#1A1A1A'>{title}"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                f"color:#9E9E9E;margin-left:6px;font-weight:500'>{nodes_tag}</span></div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#707070;margin-top:2px'>{subtitle}</div>"
                f"</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                f"padding:2px 8px;border-radius:9px;"
                f"background:{'#FAEEDA' if n_section_overrides else '#FFFFFF'};"
                f"color:{'#BA7517' if n_section_overrides else '#707070'};"
                f"border:1px solid {'#E5CC95' if n_section_overrides else '#E0DDD6'};"
                f"font-weight:500'>"
                f"{n_section_overrides} override{'s' if n_section_overrides != 1 else ''}</div>"
                f"</div>"
                f"<div style='padding:12px 16px 6px 16px'>",
                unsafe_allow_html=True,
            )

            # Native sliders inside the card
            for nid in nodes:
                m = NODE_META[nid]
                col = TC[m["type"]]
                cur = P.get(nid, .5)
                upstream = st.session_state.profile_ev.get(nid, .5)
                is_overridden = (abs(cur - upstream) > .015 and nid in st.session_state.manual_ev)

                v = st.slider(
                    f"N{nid} — {m['short']}",
                    0.0, 1.0, float(cur), .01,
                    key=f"ev_{nid}", format="%.2f",
                    label_visibility="visible"
                )

                # Inline value/state display below slider
                if is_overridden:
                    delta = v - upstream
                    delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:baseline;margin-top:-12px;margin-bottom:8px;"
                        f"font-size:0.74rem'>"
                        f"<span style='color:#BA7517;font-family:Fraunces,serif;"
                        f"font-style:italic'>"
                        f"<strong style='font-style:normal;font-family:JetBrains Mono,monospace;"
                        f"font-size:0.66rem;text-transform:uppercase;letter-spacing:0.06em;"
                        f"background:#FAEEDA;color:#BA7517;border:1px solid #E5CC95;"
                        f"padding:1px 6px;border-radius:8px;margin-right:6px'>override</strong>"
                        f"upstream: {upstream*100:.0f}%</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;"
                        f"color:{col};font-weight:600'>P(High) = {v*100:.0f}% · "
                        f"<span style='color:#BA7517'>{delta_str}</span></span></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:baseline;margin-top:-12px;margin-bottom:8px;"
                        f"font-size:0.74rem'>"
                        f"<span style='color:#707070;font-family:Fraunces,serif;"
                        f"font-style:italic'>upstream value</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;"
                        f"color:{col};font-weight:500'>P(High) = {v*100:.0f}%</span></div>",
                        unsafe_allow_html=True,
                    )

                # Preserved logic: detect manual overrides
                if abs(v - st.session_state.profile_ev.get(nid, .5)) > .015:
                    man[nid] = v

            st.markdown("</div></div>", unsafe_allow_html=True)

    # Render order — distribute sections across 2 columns
    # Risk + Mitigation in left column, Distortion + Dual + Special in right
    _left_types  = ["risk", "mitigation"]
    _right_types = ["distortion", "dual", "special"]

    for tk in _left_types:
        _render_section(nodes_by_type.get(tk, []), tk, rsc_c1)
    for tk in _right_types:
        _render_section(nodes_by_type.get(tk, []), tk, rsc_c2)

    # Save manual_ev state
    st.session_state.manual_ev = man

    # ── Reset action ──────────────────────────────────────────────────────
    rsc1, rsc2, rsc3 = st.columns([1, 1, 2])
    with rsc1:
        if st.button("Reset all to priors", key="rst"):
            for k in ["profile_ev", "manual_ev", "doc_adj"]:
                st.session_state[k] = {}
            st.session_state.gladue_checked = set()
            st.session_state.sce_checked = set()
            st.rerun()

    # ── Run inference + slim live-result strip ────────────────────────────
    run_inf()
    P = st.session_state.posteriors
    bl_rd, _bc_rd, _bg_rd = rb(P[20])
    _band_text_rd = {
        "Low":      f"belief largely resolved · {n_overrides} manual override(s)",
        "Moderate": f"belief partially resolved · {n_overrides} override(s)",
        "Elevated": f"belief shifted · {n_overrides} override(s)",
        "High":     f"strong indication · {n_overrides} override(s)",
    }.get(bl_rd, bl_rd)
    if _empty:
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr auto;"
            "align-items:center;gap:18px;background:#FBFAF7;"
            "border:1px solid #E0DDD6;border-radius:8px;"
            "padding:11px 18px;margin-top:24px;margin-bottom:0'>"
            "<div style='font-size:0.82rem;color:#9E9E9E;font-weight:500'>"
            "Node 20 · DO designation risk"
            "<span style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:1.05rem;font-weight:500;color:#9E9E9E;margin-left:10px'>—</span>"
            "</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#707070'>Awaiting case data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr auto;"
            f"align-items:center;gap:18px;background:linear-gradient(90deg,"
            f"#E2EBD8 0%, #EAF3DE 50%, #F7F5F2 100%);border:1px solid #B8CDA8;"
            f"border-radius:8px;padding:11px 18px;margin-top:24px'>"
            f"<div style='font-size:0.82rem;color:#3B6D11;font-weight:500'>"
            f"Node 20 · DO designation risk"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:#2F5C2A;margin-left:8px'>{P[20]*100:.1f}%</span>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.86rem;color:#2F5C2A'>{bl_rd} — {_band_text_rd}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── T6: Inference ─────────────────────────────────────────────────────────────
with TABS[8]:
    P=st.session_state.posteriors;dp6=P[20];bl6,bc6,bg6=rb(dp6)
    st.markdown("### Inference — posterior distribution")
    st.caption("Variable Elimination posteriors (pgmpy). Arc on DAG reflects P(High).")
    if _empty:
        st.markdown(
            f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
            f"border-radius:14px;padding:1rem 1.5rem;text-align:center;margin-bottom:1.2rem'>"
            f"<div style='font-size:.75rem;color:#9E9E9E'>Node 20 — Dangerous Offender designation risk</div>"
            f"<div style='font-size:2.4rem;font-weight:500;font-family:Fraunces,Georgia,serif;"
            f"font-style:italic;color:#9E9E9E;margin:6px 0 4px 0'>—</div>"
            f"<div style='font-size:.9rem;font-family:Fraunces,serif;font-style:italic;color:#707070'>"
            f"Awaiting case data</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"""<div style="background:{bg6};border:1px solid {bc6}44;border-radius:14px;
        padding:1rem 1.5rem;text-align:center;margin-bottom:1.2rem">
        <div style="font-size:.75rem;color:{bc6}">Node 20 — Dangerous Offender designation risk</div>
        <div style="font-size:2.8rem;font-weight:700;font-family:monospace;color:{bc6}">{dp6*100:.1f}%</div>
        <div style="font-size:.9rem;font-weight:600;color:{bc6}">{bl6}</div></div>""",unsafe_allow_html=True)
    cols4=st.columns(4)
    for i,nid in enumerate(n for n in range(1,21) if n!=20):
        m=NODE_META[nid];col=TC[m["type"]];p=P.get(nid,.5)
        with cols4[i%4]:
            st.markdown(f"""<div style="background:{col}18;border:1px solid {col}33;border-radius:8px;
            padding:.55rem .7rem;margin-bottom:.4rem">
            <div style="font-size:.65rem;color:{col};font-weight:700">N{nid} — {m['short']}</div>
            <div style="font-size:1.1rem;font-weight:700;font-family:monospace;color:{col}">{p*100:.1f}%</div>
            <div style="height:4px;background:#eee;border-radius:2px;margin-top:3px">
              <div style="width:{p*100:.0f}%;height:100%;background:{col};border-radius:2px"></div>
            </div></div>""",unsafe_allow_html=True)

# ── T7: QBism + Bloch sphere ─────────────────────────────────────────────────
with TABS[11]:
    # ════════════════════════════════════════════════════════════════════════
    # Quantum tab v3 — diagnostic-only redesign (Mark 8 build)
    # Per Appendix Q §AQ.4: this layer does not alter the VE posterior
    # and does not recommend designation outcomes. It surfaces epistemic
    # conditions warranting heightened scrutiny.
    # ════════════════════════════════════════════════════════════════════════
    diags = st.session_state.get("qdiags", {}) or {}
    p_high_n20 = float(P.get(20, 0.5))

    # Compute the headline metrics from the live posterior
    alpha_sq = p_high_n20
    beta_sq = 1.0 - alpha_sq
    si = round(1.0 - abs(p_high_n20 - 0.5) * 2.0, 3)
    coh = round((alpha_sq * beta_sq) ** 0.5, 3)
    purity = round(alpha_sq**2 + beta_sq**2 + 2 * (alpha_sq * beta_sq), 3)
    theta_deg = round(__import__("math").degrees(__import__("math").acos(__import__("math").fsum([1.0, -2 * alpha_sq]))), 1)
    # Azimuthal angle — encodes risk vs mitigation narrative balance, computed
    # the same way the JS canvas does (so all readouts on this tab agree).
    _rw_z1 = sum(P.get(n, .5) for n in [2, 3, 4, 18]) / 4
    _mw_z1 = sum(P.get(n, .5) for n in [5, 6, 10, 12, 14]) / 5
    phi_deg = round(float(__import__("numpy").degrees(__import__("numpy").arctan2(_rw_z1, _mw_z1))) % 360, 1)

    # ── Determine the headline diagnostic condition ────────────────────────────
    os_diag  = diags.get("order_stability", {}) or {}
    cg_diag  = diags.get("connection_gate_contextuality", {}) or {}
    pc_diag  = diags.get("prior_contamination", {}) or {}
    bs_diag  = diags.get("belief_stasis", {}) or {}

    flags_active = sum(1 for d in [os_diag, cg_diag, pc_diag, bs_diag]
                       if d.get("severity") in ("moderate", "high"))

    # Pick the single headline label that best describes the current state.
    # Note: research-prototype thresholds, illustrative anchors per Appendix O §O.3.
    if si >= 0.60 and flags_active == 0:
        cond_label = "Pre-decisional ambiguity"
        cond_sub   = "belief state in superposition"
        cond_color = "#8B4A1A"  # orange
        cond_bg    = "#F4E0CA"
        cond_bd    = "#D8B58A"
    elif flags_active >= 2:
        cond_label = "Multiple strain signals"
        cond_sub   = f"{flags_active} diagnostic checks flagged"
        cond_color = "#8B2E2A"  # red
        cond_bg    = "#F2D9D5"
        cond_bd    = "#D8A39E"
    elif flags_active == 1:
        cond_label = "Heightened scrutiny"
        cond_sub   = "one diagnostic check flagged"
        cond_color = "#8A6B1F"  # amber
        cond_bg    = "#F8EFD8"
        cond_bd    = "#E5CC95"
    elif si >= 0.30:
        cond_label = "Moderate ambiguity"
        cond_sub   = "belief partially resolved"
        cond_color = "#8A6B1F"  # amber
        cond_bg    = "#F8EFD8"
        cond_bd    = "#E5CC95"
    else:
        cond_label = "Coherent"
        cond_sub   = "belief state largely resolved"
        cond_color = "#2F5C2A"  # green
        cond_bg    = "#E2EBD8"
        cond_bd    = "#B8CDA8"

    # ── Tab title + intro ─────────────────────────────────────────────────────
    st.markdown("### ⚛️ Quantum Bayesian diagnostic layer")
    st.markdown(
        '<div style="font-family:\'Fraunces\',Georgia,serif;font-style:italic;'
        'font-size:0.92rem;color:#707070;margin-bottom:14px;line-height:1.6;'
        'max-width:880px">'
        'Appendix Q · An epistemic lens for identifying conditions under which '
        'classical Bayesian inference, though formally valid, may be substantively '
        'strained by the evidentiary environment in which it operates.'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Permanent scope-disclaimer (every render) ─────────────────────────────
    st.markdown(
        '<div style="background:#EAEFF7;border:1px solid #C7D3E5;'
        'border-left:3px solid #185FA5;border-radius:6px;padding:12px 18px;'
        'margin-bottom:24px;font-size:0.86rem;color:#3A3A3A;line-height:1.55;'
        'max-width:880px">'
        '<strong style="color:#185FA5">Diagnostic, not decision-support.</strong> '
        'This layer does not alter the Variable Elimination posterior and does '
        'not recommend designation outcomes. It surfaces epistemic conditions — '
        '<em>order effects, contextuality, premature scalar collapse, distorted '
        'priors</em> — that classical probabilistic reasoning is poorly equipped '
        'to represent. Per Appendix Q '
        '<span style="font-family:\'Fraunces\',serif;font-style:italic;'
        'font-size:0.84rem">§AQ.4</span>, Quantum Bayesianism is used here as '
        'an epistemic audit mechanism, not a substitute inferential engine. '
        'The diagnostic states and thresholds shown are research-prototype '
        'values — illustrative anchors for the framework set out in Appendix Q, '
        'subject to expert elicitation through the SHELF/Cooke methodology '
        'described in Appendix O '
        '<span style="font-family:\'Fraunces\',serif;font-style:italic;'
        'font-size:0.84rem">§O.3</span>.'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Empty-state placeholder — deliberate substitute for live diagnostics ──
    # The title and intro panel always render (these are doctrinal framing).
    # The four zones below are gated by `if not _empty:` because they compute
    # quantum diagnostics from network posteriors, which are meaningless when
    # no case data has been entered.
    if _empty:
        # Two-column layout matching the populated state visually.
        _qe_c1, _qe_c2 = st.columns([1.4, 1])
        with _qe_c1:
            st.markdown(
                "<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
                "border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:14px'>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.14em;color:#9E9E9E;font-weight:600;"
                "margin-bottom:8px'>Epistemic condition</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-size:1.6rem;"
                "font-style:italic;color:#9E9E9E;margin-bottom:6px'>—</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#707070;line-height:1.6'>"
                "Awaiting case data."
                "</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.86rem;color:#707070;line-height:1.6;margin-top:14px'>"
                "Quantum diagnostic states will populate once a case profile is entered. "
                "The diagnostic vocabulary — <em>Pre-decisional ambiguity</em>, "
                "<em>Heightened scrutiny</em>, <em>Multiple strain signals</em>, "
                "<em>Moderate ambiguity</em>, <em>Coherent</em> — characterises the "
                "live belief state and requires case-specific inputs to compute."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        with _qe_c2:
            st.markdown(
                "<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
                "border-radius:10px;padding:1.2rem 1.4rem'>"
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "margin-bottom:14px'>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                "color:#9E9E9E'>|A|² · |B|²</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#9E9E9E'>— · —</div>"
                "</div>"
                "<div style='margin-bottom:14px'>"
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "margin-bottom:4px'>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.10em;color:#9E9E9E;font-weight:600'>"
                "Superposition index</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#9E9E9E'>—</div>"
                "</div>"
                "<div style='height:5px;background:rgba(0,0,0,0.04);border-radius:3px'></div>"
                "</div>"
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "margin-bottom:10px'>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.10em;color:#9E9E9E;font-weight:600'>Coherence</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#9E9E9E'>—</div>"
                "</div>"
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "margin-bottom:10px'>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.10em;color:#9E9E9E;font-weight:600'>Purity Tr(ρ²)</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#9E9E9E'>—</div>"
                "</div>"
                "<div style='display:flex;justify-content:space-between;align-items:baseline;"
                "margin-bottom:14px'>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.10em;color:#9E9E9E;font-weight:600'>θ · φ (Bloch)</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.95rem;color:#9E9E9E'>— · —</div>"
                "</div>"
                "<div style='font-size:0.66rem;text-transform:uppercase;"
                "letter-spacing:0.10em;color:#9E9E9E;font-weight:600;"
                "margin-bottom:8px;padding-top:10px;border-top:1px solid #E0DDD6'>"
                "Strain indicators (§AQ.3.3)</div>"
                "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
                "font-size:0.84rem;color:#707070;line-height:1.6'>"
                "Awaiting case data — strain checks (order effects, contextuality, "
                "prior contamination, premature collapse) require live posteriors."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
            "border-radius:10px;padding:1.4rem 1.6rem;text-align:center'>"
            "<div style='font-size:0.66rem;text-transform:uppercase;"
            "letter-spacing:0.14em;color:#9E9E9E;font-weight:600;margin-bottom:10px'>"
            "Bloch sphere — quantum belief state |ψ⟩</div>"
            "<div style='font-family:Fraunces,Georgia,serif;font-size:2.4rem;"
            "font-style:italic;color:#9E9E9E;margin:6px 0 12px 0'>—</div>"
            "<div style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:0.92rem;color:#707070;line-height:1.6;max-width:680px;"
            "margin:0 auto'>"
            "The Bloch sphere visualisation will render once case data is entered. "
            "The polar angle θ derives from Node 20's posterior; the azimuthal angle "
            "φ encodes the balance between risk-cluster nodes (N2 / N3 / N4 / N18) "
            "and mitigation-cluster nodes (N5 / N6 / N10 / N12 / N14). Both require "
            "a populated network state to compute meaningfully."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── ZONES 1-4: live quantum diagnostics (gated on case data) ────────────
    if not _empty:
        # ── ZONE 1: Diagnostic state + meter panel ────────────────────────────────
        z1c1, z1c2 = st.columns([1.4, 1])
        with z1c1:
            # Build the verdict body text dynamically based on which signals are active
            body_parts = []
            body_parts.append(
                f"The classical posterior of {p_high_n20*100:.1f}% is "
                f"mathematically coherent."
            )
            if si >= 0.60:
                body_parts.append(
                    f"The belief state sits near the equator of the Bloch sphere: "
                    f"|α|² = {alpha_sq:.3f} and |β|² = {beta_sq:.3f} are close to "
                    f"one another, so both designation and non-designation "
                    f"narratives remain live (Appendix Q §AQ.3.3.5.2)."
                )
            if cg_diag.get("severity") in ("moderate", "high"):
                cg_delta = cg_diag.get("delta", 0.0)
                body_parts.append(
                    f"The posterior is contextually sensitive to the Morris para 97 "
                    f"connection-gate doctrinal call — the inference outcome shifts "
                    f"by {cg_delta*100:.1f} percentage points across weak/moderate/"
                    f"strong settings of the gate (Appendix Q §AQ.3.3.5.4)."
                )
            if os_diag.get("severity") in ("moderate", "high"):
                os_delta = os_diag.get("delta", 0.0)
                body_parts.append(
                    f"The current evidence configuration shows order-effect strain "
                    f"of up to {os_delta*100:.1f} percentage points across "
                    f"doctrinally-motivated permutations (Appendix Q §AQ.3.3.5.3)."
                )
            body_parts.append(
                "This is an observation about the structure of the belief state, "
                "not a claim about what the court should decide. Per Appendix Q "
                "§AQ.4, the choice of how to weight these factors remains a "
                "question of legal reasoning."
            )
            body_text = "<br><br>".join(body_parts)

            st.markdown(
                f'<div style="background:{cond_bg};border:1px solid {cond_bd};'
                f'border-left:4px solid {cond_color};border-radius:12px;'
                f'padding:22px 26px;height:100%">'
                f'<div style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.16em;color:{cond_color};font-weight:700;'
                f'margin-bottom:8px">EPISTEMIC CONDITION</div>'
                f'<div style="font-family:\'Fraunces\',Georgia,serif;'
                f'font-size:1.55rem;font-weight:500;letter-spacing:-0.005em;'
                f'color:{cond_color};margin:0 0 10px 0;line-height:1.18">'
                f'{cond_label} '
                f'<span style="font-family:\'Fraunces\',serif;font-style:italic;'
                f'font-weight:400;font-size:1.05rem;color:#707070">— {cond_sub}</span>'
                f'</div>'
                f'<div style="font-family:\'Fraunces\',Georgia,serif;'
                f'font-style:italic;font-size:0.98rem;color:#3A3A3A;line-height:1.6">'
                f'{body_text}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with z1c2:
            # Strain indicators
            def _strain_pill(label, severity):
                if severity in ("moderate", "high"):
                    return (f'<span style="font-size:0.72rem;padding:3px 10px;'
                            f'border-radius:11px;background:#F8EFD8;color:#8A6B1F;'
                            f'border:1px solid #E5CC95;font-family:\'Fraunces\',serif;'
                            f'font-style:italic;display:inline-block;margin:2px 4px 2px 0">'
                            f'{label} · detected</span>')
                elif severity == "not_run":
                    return (f'<span style="font-size:0.72rem;padding:3px 10px;'
                            f'border-radius:11px;background:#EFEDE7;color:#9E9E9E;'
                            f'border:1px solid #E0DDD6;font-family:\'Fraunces\',serif;'
                            f'font-style:italic;display:inline-block;margin:2px 4px 2px 0">'
                            f'{label} · not run</span>')
                else:
                    return (f'<span style="font-size:0.72rem;padding:3px 10px;'
                            f'border-radius:11px;background:#E2EBD8;color:#2F5C2A;'
                            f'border:1px solid #B8CDA8;font-family:\'Fraunces\',serif;'
                            f'font-style:italic;display:inline-block;margin:2px 4px 2px 0">'
                            f'{label} · none</span>')

            pill_oe   = _strain_pill("Order effects",       os_diag.get("severity", "none"))
            pill_ctx  = _strain_pill("Contextuality",       cg_diag.get("severity", "none"))
            pill_pc   = _strain_pill("Prior contamination", pc_diag.get("severity", "none"))
            pill_bs   = _strain_pill("Premature collapse",  bs_diag.get("severity", "none"))

            st.markdown(
                f'<div style="background:#FBFAF7;border:1px solid #E0DDD6;'
                f'border-radius:12px;padding:18px 20px;height:100%">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:10px">'
                f'<span style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600">|α|² · |β|²</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
                f'color:#1A1A1A">{alpha_sq:.3f} · {beta_sq:.3f}</span></div>'

                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:6px">'
                f'<span style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600">'
                f'Superposition index</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
                f'color:{cond_color}">{si:.3f}</span></div>'
                f'<div style="height:6px;border-radius:3px;background:#EFEDE7;'
                f'margin-bottom:14px;overflow:hidden">'
                f'<div style="height:100%;width:{si*100:.1f}%;background:{cond_color};'
                f'border-radius:3px"></div></div>'

                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:10px">'
                f'<span style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600">Coherence</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
                f'color:#1A1A1A">{coh:.3f}</span></div>'

                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:10px">'
                f'<span style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600">Purity Tr(ρ²)</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
                f'color:#1A1A1A">{purity:.3f}</span></div>'

                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-bottom:14px">'
                f'<span style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600">θ · φ (Bloch)</span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
                f'color:#1A1A1A">{theta_deg}° · {phi_deg}°</span></div>'

                f'<div style="border-top:1px solid #EFEDE7;padding-top:10px;margin-top:6px">'
                f'<div style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.14em;color:#707070;font-weight:600;'
                f'margin-bottom:8px">Strain indicators (§AQ.3.3)</div>'
                f'<div>{pill_oe}{pill_ctx}{pill_pc}{pill_bs}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin:24px 0'></div>", unsafe_allow_html=True)

        # ── ZONE 2: Diagnostic signals ────────────────────────────────────────────
        st.markdown("#### Diagnostic signals")
        st.markdown(
            '<div style="font-size:0.84rem;color:#707070;margin-bottom:18px;'
            'font-family:\'Fraunces\',serif;font-style:italic;max-width:800px;'
            'line-height:1.55">'
            'What the QBism layer is registering about the current belief state. '
            'These are observations about epistemic conditions, drawn from '
            'Appendix Q\'s diagnostic vocabulary — not recommendations about what '
            'the sentencing court should conclude.'
            '</div>',
            unsafe_allow_html=True
        )

        def _signal_html(tag, tag_color, body_html):
            return (
                f'<div style="display:grid;grid-template-columns:64px 1fr;gap:16px;'
                f'align-items:flex-start;padding:14px 0;'
                f'border-bottom:1px solid #EFEDE7">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;'
                f'font-size:0.74rem;font-weight:600;text-align:center;padding:5px 0;'
                f'border-radius:5px;color:white;background:{tag_color};'
                f'white-space:nowrap">{tag}</div>'
                f'<div style="font-size:0.94rem;color:#3A3A3A;line-height:1.6">'
                f'{body_html}</div>'
                f'</div>'
            )

        # Signal 1 — Superposition index (always reportable)
        if si >= 0.60:
            si_body = (
                f'<strong>Superposition index <span style="font-family:monospace;'
                f'background:{cond_bg};color:{cond_color};padding:1px 6px;'
                f'border-radius:3px">{si:.3f}</span> '
                f'indicates a belief state near the equator of the Bloch sphere.</strong> '
                f'Per Appendix Q §AQ.3.3.5.2, this is the formal representation of '
                f'pre-decisional ambiguity: a state in which |α|² (risk-centred '
                f'narrative) and |β|² (contextual mitigation narrative) remain '
                f'simultaneously live, neither having decisively resolved. Classical '
                f'Bayesian inference is mathematically valid in this state; the '
                f'diagnostic flags only that scalar collapse has not yet occurred.'
            )
        else:
            si_body = (
                f'<strong>Superposition index <span style="font-family:monospace;'
                f'background:#E2EBD8;color:#2F5C2A;padding:1px 6px;'
                f'border-radius:3px">{si:.3f}</span> '
                f'indicates a belief state away from the equator.</strong> '
                f'Per Appendix Q §AQ.3.3.5.2, the system has substantively resolved '
                f'toward one of |α|² (risk-centred) or |β|² (contextual mitigation) '
                f'narratives. The classical Bayesian posterior is operating outside '
                f'the maximum-superposition regime.'
            )
        st.markdown(_signal_html("SI", cond_color, si_body), unsafe_allow_html=True)

        # Signal 2 — Order stability
        os_severity = os_diag.get("severity", "not_run")
        os_note = os_diag.get("note", "")
        if os_severity == "not_run":
            os_body = (
                f'<strong>Order-stability check status: not run.</strong> '
                f'Per Appendix Q §AQ.3.3.5.3, this check would test whether the '
                f'current evidence configuration is stable under doctrinally-'
                f'motivated reorderings (risk-first vs SCE-first vs priors-only). '
                f'Activate by ensuring inference engine and current evidence are '
                f'available at the diagnose() call site.'
            )
        elif os_severity in ("moderate", "high"):
            os_body = (
                f'<strong>Order-effect strain detected — '
                f'posterior diverges by up to {os_diag.get("delta",0)*100:.1f}% '
                f'across doctrinally-motivated permutations.</strong> {os_note}'
            )
        else:
            os_body = (
                f'<strong>No order effect detected in the current evidence sequence.</strong> '
                f'{os_note}'
            )
        st.markdown(_signal_html("OE", "#185FA5", os_body), unsafe_allow_html=True)

        # Signal 3 — Connection-gate contextuality (the new flagship check)
        cg_severity = cg_diag.get("severity", "not_run")
        cg_note = cg_diag.get("note", "")
        if cg_severity == "not_run":
            cg_body = (
                f'<strong>Connection-gate contextuality check status: not run.</strong> '
                f'Per Appendix Q §AQ.3.3.5.4, this check tests whether the case '
                f'posterior shifts meaningfully across weak/moderate/strong settings '
                f'of the Morris para 97 connection gate. Activate by ensuring per-gate '
                f'SCE evidence is computed and passed in at the diagnose() call site.'
            )
        elif cg_severity in ("moderate", "high"):
            cg_body = (
                f'<strong>Connection-gate contextuality detected — posterior shifts '
                f'<span style="font-family:monospace;background:{cond_bg};color:{cond_color};'
                f'padding:1px 6px;border-radius:3px">{cg_diag.get("delta",0)*100:.1f}%</span> '
                f'across gate strengths.</strong> {cg_note}'
            )
        else:
            cg_body = (
                f'<strong>No connection-gate contextuality detected.</strong> '
                f'{cg_note}'
            )
        st.markdown(_signal_html("CTX", "#0F6E56", cg_body), unsafe_allow_html=True)

        # Signal 4 — Prior contamination (existing diagnostic)
        pc_severity = pc_diag.get("severity", "none")
        if pc_severity in ("moderate", "high"):
            pc_items = pc_diag.get("items", [])
            pc_body = (
                f'<strong>Prior contamination flagged — '
                f'{len(pc_items)} distortion-typed node(s) elevated without case-specific evidence.</strong> '
                f'Per Appendix Q §AQ.3.3.2, criminal records may encode upstream '
                f'distortions (over-policing, bail denial cascades, culturally invalid '
                f'actuarial tools) that classical Bayesian updating cannot self-correct. '
                f'The flagged nodes carry posteriors above the prior-contamination '
                f'threshold (0.65) but lack direct case-specific evidence.'
            )
        else:
            pc_body = (
                f'<strong>Distorted-priors warning surface — currently inactive.</strong> '
                f'Per Appendix Q §AQ.3.3.2, criminal records may encode upstream '
                f'distortions that classical Bayesian updating cannot self-correct. '
                f'None detected at the current input state — but the structural risk '
                f'identified in §AQ.3.3.2 is a feature of the input data itself, not '
                f'of the inference engine, and remains relevant whenever criminal-history '
                f'nodes carry weight.'
            )
        st.markdown(_signal_html("P", "#A32D2D", pc_body), unsafe_allow_html=True)

        st.markdown("<div style='margin:32px 0'></div>", unsafe_allow_html=True)

        # ── ZONE 3: Observations ──────────────────────────────────────────────────
        st.markdown("#### What this diagnostic surfaces")
        st.markdown(
            '<div style="font-size:0.84rem;color:#707070;margin-bottom:18px;'
            'font-family:\'Fraunces\',serif;font-style:italic;max-width:800px;'
            'line-height:1.55">'
            'Observations that follow from the diagnostic state above. These are '
            'points of epistemic visibility — features of the current belief state '
            'that the QBism vocabulary makes legible to the user. They are not '
            'action recommendations or counterfactual predictions about the posterior.'
            '</div>',
            unsafe_allow_html=True
        )

        def _obs(obs_tag, obs_title, obs_body):
            return (
                f'<div style="background:#FBFAF7;border:1px solid #E0DDD6;'
                f'border-radius:10px;padding:16px 22px;margin-bottom:10px">'
                f'<div style="font-size:0.66rem;text-transform:uppercase;'
                f'letter-spacing:0.16em;color:#707070;font-weight:700;'
                f'margin-bottom:5px">{obs_tag}</div>'
                f'<div style="font-family:\'Fraunces\',serif;font-weight:500;'
                f'font-size:1.05rem;color:#1A1A1A;margin-bottom:6px">{obs_title}</div>'
                f'<div style="font-size:0.92rem;color:#3A3A3A;line-height:1.6">'
                f'{obs_body}</div></div>'
            )

        # Observation 1 — Superposition state
        if si >= 0.60:
            st.markdown(_obs(
                "Observation 1 · §AQ.3.3.5.2",
                "The belief state has not resolved into a determinate posture",
                f'The {p_high_n20*100:.1f}% scalar value, taken alone, would suggest '
                f'a {"moderate" if 0.4 < p_high_n20 < 0.6 else "definitive"}-risk '
                f'determination. The QBism layer registers that the state generating '
                f'this value is in superposition: SI <span style="font-family:monospace;'
                f'background:{cond_bg};color:{cond_color};padding:1px 6px;'
                f'border-radius:3px">{si:.3f}</span>. In Appendix Q\'s vocabulary, '
                f'this means the system is operating in the regime where, per '
                f'<em>Ipeelee</em> §38 and the principle articulated at §AQ.3.3.5.2, '
                f'withholding definitive judgment is not a failure of reasoning but '
                f'a feature of lawful decision-making under genuine uncertainty. '
                f'Whether to act on this observation is a matter for judicial '
                f'discretion, not for the diagnostic.'
            ), unsafe_allow_html=True)

        # Observation 2 — Order stability
        if os_severity in ("moderate", "high"):
            perms = os_diag.get("permutations", [])
            rows_html = ""
            for p in perms:
                rows_html += (
                    f'<tr><td style="padding:4px 14px 4px 0">{p["label"]}</td>'
                    f'<td style="text-align:right;padding:4px 14px 4px 0;'
                    f'font-family:monospace">{p["posterior"]*100:.1f}%</td>'
                    f'<td style="text-align:right;padding:4px 0;font-family:monospace">'
                    f'Δ {p["delta"]*100:.1f}%</td></tr>'
                )
            st.markdown(_obs(
                "Observation 2 · §AQ.3.3.5.3",
                "Evidence sequence shows order-effect strain",
                f'{os_note}<br><br>'
                f'<table style="margin:10px 0 0 0;border-collapse:collapse;'
                f'font-size:0.86rem"><thead><tr style="border-bottom:1px solid #E0DDD6">'
                f'<th style="text-align:left;padding:4px 14px 4px 0;font-weight:600;'
                f'color:#1A1A1A">Permutation</th>'
                f'<th style="text-align:right;padding:4px 14px 4px 0;font-weight:600;'
                f'color:#1A1A1A">Posterior N20</th>'
                f'<th style="text-align:right;padding:4px 0;font-weight:600;'
                f'color:#1A1A1A">Δ from base</th></tr></thead>'
                f'<tbody>{rows_html}</tbody></table>'
            ), unsafe_allow_html=True)
        else:
            st.markdown(_obs(
                "Observation 2 · §AQ.3.3.5.3",
                "Evidentiary sequence is currently order-stable",
                f'{os_note}'
            ), unsafe_allow_html=True)

        # Observation 3 — Connection-gate contextuality
        if cg_severity in ("moderate", "high"):
            gates = cg_diag.get("gates_tested", [])
            gate_rows_html = ""
            for g in gates:
                is_current = g["gate"] == st.session_state.conn or "current" in g["gate"]
                label_html = (f'<em>{g["gate"]}</em> (current)' if is_current
                              else f'<em>{g["gate"]}</em>')
                n20_html = (f'<strong>{g["posterior"]*100:.1f}%</strong>'
                            if is_current else f'{g["posterior"]*100:.1f}%')
                gate_rows_html += (
                    f'<tr><td style="padding:4px 14px 4px 0">{label_html}</td>'
                    f'<td style="text-align:right;padding:4px 14px 4px 0;'
                    f'font-family:monospace">{g["weight"]:.2f}</td>'
                    f'<td style="text-align:right;padding:4px 0;font-family:monospace">'
                    f'{n20_html}</td></tr>'
                )
            st.markdown(_obs(
                "Observation 3 · §AQ.3.3.5.4",
                "The case is contextually sensitive to the connection-gate doctrinal call",
                f'Holding all other evidence constant, the Node 20 posterior takes '
                f'distinct values across the doctrinal settings of the Morris para 97 '
                f'connection gate:<br>'
                f'<table style="margin:10px 0 12px 0;border-collapse:collapse;'
                f'font-size:0.86rem"><thead><tr style="border-bottom:1px solid #E0DDD6">'
                f'<th style="text-align:left;padding:4px 14px 4px 0;font-weight:600;'
                f'color:#1A1A1A">Gate setting</th>'
                f'<th style="text-align:right;padding:4px 14px 4px 0;font-weight:600;'
                f'color:#1A1A1A">cmult</th>'
                f'<th style="text-align:right;padding:4px 0;font-weight:600;'
                f'color:#1A1A1A">Posterior N20</th></tr></thead>'
                f'<tbody>{gate_rows_html}</tbody></table>'
                f'A {cg_diag.get("delta",0)*100:.1f}-point swing across the three '
                f'settings exceeds the research-prototype threshold of 5.0 points. '
                f'Per §AQ.3.3.5.4, this is contextuality in the Kochen-Specker sense: '
                f'the same evidentiary record yields different probative readings '
                f'under doctrinally distinct connection-strength frames. The '
                f'diagnostic surfaces the sensitivity. The choice between '
                f'<em>weak</em>, <em>moderate</em>, and <em>strong</em> remains a '
                f'question of legal reasoning about R v Morris 2021 ONCA 680 §97 — '
                f'not foreclosed by the diagnostic.'
            ), unsafe_allow_html=True)
        else:
            st.markdown(_obs(
                "Observation 3 · §AQ.3.3.5.4",
                "Connection-gate setting is not contextually consequential at the current state",
                f'{cg_note}'
            ), unsafe_allow_html=True)

        # Audit-not-policy footer
        st.markdown(
            '<div style="font-size:0.82rem;color:#707070;font-family:\'Fraunces\',serif;'
            'font-style:italic;padding:14px 0 0 0;line-height:1.55;max-width:800px;'
            'border-top:1px solid #EFEDE7;margin-top:18px">'
            'These observations are diagnostic, not directive. Per Appendix Q §AQ.4, '
            'the QBism layer is an epistemic audit mechanism. It identifies where '
            'classical inference may benefit from heightened scrutiny; it does not '
            'tell the user what evidentiary moves to make, and it does not quantify '
            'the consequences of hypothetical evidentiary changes. Counterfactual '
            'posterior estimation under alternative inputs is a feature reserved '
            'for the Scenarios tab, where the user themselves constructs and '
            'labels the comparison.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='margin:32px 0'></div>", unsafe_allow_html=True)

        # ── ZONE 4: Animated Bloch sphere (PARVIS canvas + JS precession) ────────
        # Restored from the original Quantum tab. Animation logic lives entirely
        # in JavaScript (requestAnimationFrame, drawn into HTML5 Canvas) and is
        # injected via st.components.v1.html. The matplotlib draw_bloch_sphere()
        # is preserved below in a collapsible expander as a static fallback view.

        # Pre-compute risk and mitigation weights in the same way the original tab
        # did: average posterior over risk-typed nodes vs distortion/mitigation nodes.
        _rw = sum(P.get(n, .5) for n in [2, 3, 4, 18]) / 4
        _mw = sum(P.get(n, .5) for n in [5, 6, 10, 12, 14]) / 5

        # The canvas convention matches bloch_sphere.py: θ = arccos(1 - 2·p_high),
        # measured from the |DO⟩ pole. Recompute theta_deg and phi_deg here using
        # this convention so the canvas matches the figure-title and the live
        # state-vector readout shown alongside.
        _theta_canvas_deg = float(__import__("numpy").degrees(
            __import__("numpy").arccos(__import__("numpy").clip(1 - 2*p_high_n20, -1, 1))
        ))
        _phi_canvas_deg = float(__import__("numpy").degrees(
            __import__("numpy").arctan2(_rw, _mw)
        )) % 360

        st.markdown("#### Bloch sphere — quantum belief state |ψ⟩")
        st.caption(
            "The state vector rotates slowly to illustrate the superposition of "
            "belief states. North pole = fully collapsed to DO (P=1). South pole "
            "= fully collapsed to no DO (P=0). Equator = maximum pre-decisional "
            "ambiguity (P=0.5)."
        )

        _bloch_qb1, _bloch_qb2 = st.columns([3, 2])
        with _bloch_qb1:
            # ── Animated Bloch sphere (HTML/JS canvas) ────────────────────────────────
            # Pre-compute the state vector endpoint for the current belief state
            import json
            bloch_state = json.dumps({
                "theta": float(np.radians(_theta_canvas_deg)),
                "phi":   float(np.radians(_phi_canvas_deg)),
                "risk":  float(p_high_n20),
                "si":    float(si),
                "rw":    float(_rw),
                "mw":    float(_mw),
            })

            bloch_html = f"""
        <div style="display:flex;flex-direction:column;align-items:center">
          <canvas id="bloch" width="570" height="570"
            style="border-radius:14px;background:#ffffff;border:1px solid #e8e8e8;
                   box-shadow:0 2px 12px rgba(0,0,0,0.08)"></canvas>
          <div id="bloch-label"
            style="font-family:monospace;font-size:13px;color:#555;margin-top:6px;text-align:center"></div>
        </div>
        <script>
        (function(){{
          const S = {bloch_state};
          const canvas = document.getElementById("bloch");
          if (!canvas) return;
          const ctx = canvas.getContext("2d");
          const W=570, H=570, cx=285, cy=285, R=210;

          const C_RISK = "#C0392B", C_MIT = "#1A6B35", C_POLE = "#1B2A4A", C_MID = "#666";
          const riskCol = S.risk>=0.55 ? "#C0392B" : S.risk>=0.35 ? "#B8850A" : "#1A6B35";

          // Full 3D rotation: Y-axis (azimuthal) + X-axis (polar tilt)
          // This gives authentic Bloch-sphere precession in 3D
          function proj(x, y, z, ry, rx) {{
            // 1. Rotate around Y (azimuthal — left/right)
            let x1 = x*Math.cos(ry) - z*Math.sin(ry);
            let y1 = y;
            let z1 = x*Math.sin(ry) + z*Math.cos(ry);
            // 2. Rotate around X (polar tilt — forward/back)
            let x2 = x1;
            let y2 = y1*Math.cos(rx) - z1*Math.sin(rx);
            let z2 = y1*Math.sin(rx) + z1*Math.cos(rx);
            // 3. Perspective
            const f = 3.2;
            return [cx + x2*R*f/(f+z2+2), cy - y2*R*f/(f+z2+2), z2];
          }}

          function draw(ry, rx) {{
            ctx.clearRect(0,0,W,H);

            // Sphere fill gradient
            const g = ctx.createRadialGradient(cx-65,cy-65,20,cx,cy,R+10);
            g.addColorStop(0,"rgba(205,215,230,0.50)");
            g.addColorStop(1,"rgba(238,241,248,0.10)");
            ctx.beginPath(); ctx.arc(cx,cy,R,0,Math.PI*2);
            ctx.fillStyle=g; ctx.fill();
            ctx.strokeStyle="rgba(0,0,0,0.15)"; ctx.lineWidth=1.5; ctx.stroke();

            // Latitude rings
            for(let lat=-60;lat<=60;lat+=30){{
              const lr=Math.cos(lat*Math.PI/180), ly=Math.sin(lat*Math.PI/180);
              ctx.beginPath(); let fi=true;
              for(let a=0;a<=360;a+=3){{
                const r=a*Math.PI/180;
                const [sx,sy]=proj(lr*Math.cos(r),ly,lr*Math.sin(r),ry,rx);
                fi?(ctx.moveTo(sx,sy),fi=false):ctx.lineTo(sx,sy);
              }}
              ctx.closePath();
              ctx.strokeStyle=lat===0?"rgba(0,0,0,0.22)":"rgba(0,0,0,0.07)";
              ctx.lineWidth=lat===0?1.3:0.7; ctx.stroke();
            }}

            // Meridians
            for(let lon=0;lon<180;lon+=45){{
              const lr2=lon*Math.PI/180;
              ctx.beginPath(); let fi2=true;
              for(let a=0;a<=360;a+=3){{
                const r=a*Math.PI/180;
                const [sx,sy]=proj(Math.sin(r)*Math.cos(lr2),Math.cos(r),Math.sin(r)*Math.sin(lr2),ry,rx);
                fi2?(ctx.moveTo(sx,sy),fi2=false):ctx.lineTo(sx,sy);
              }}
              ctx.strokeStyle="rgba(0,0,0,0.05)"; ctx.lineWidth=0.7; ctx.stroke();
            }}

            // Horizontal axes (rotate with sphere)
            const [ocx,ocy]=proj(0,0,0,ry,rx);
            const [rx1,ry1]=proj(0.85,0,0,ry,rx);
            const [mx1,my1]=proj(-0.85,0,0,ry,rx);
            ctx.beginPath(); ctx.moveTo(ocx,ocy); ctx.lineTo(rx1,ry1);
            ctx.strokeStyle=C_RISK+"88"; ctx.lineWidth=1.1;
            ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle=C_RISK; ctx.font="bold 12px -apple-system,sans-serif";
            ctx.fillText("Risk",rx1+5,ry1+4);
            ctx.beginPath(); ctx.moveTo(ocx,ocy); ctx.lineTo(mx1,my1);
            ctx.strokeStyle=C_MIT+"88"; ctx.lineWidth=1.1;
            ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle=C_MIT; ctx.font="bold 12px -apple-system,sans-serif";
            const mw=ctx.measureText("Mitigation").width;
            ctx.fillText("Mitigation",mx1-mw-5,my1+4);

            // Vertical axis (also rotates in 3D)
            const [ap1,ap2]=proj(0,0.85,0,ry,rx);
            const [an1,an2]=proj(0,-0.85,0,ry,rx);
            ctx.beginPath(); ctx.moveTo(ocx,ocy); ctx.lineTo(ap1,ap2);
            ctx.strokeStyle=C_POLE+"55"; ctx.lineWidth=1.1;
            ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
            ctx.beginPath(); ctx.moveTo(ocx,ocy); ctx.lineTo(an1,an2);
            ctx.strokeStyle=C_MID+"55"; ctx.lineWidth=1.1;
            ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);

            // Pole labels — attached to the rotated pole positions
            const [np1,np2]=proj(0,1.04,0,ry,rx);
            const [sp1,sp2]=proj(0,-1.04,0,ry,rx);
            ctx.textAlign="center";
            ctx.fillStyle=C_POLE; ctx.font="bold 12px -apple-system,sans-serif";
            ctx.fillText("|DO\u27e9  P=1.0",np1,np2-10);
            ctx.fillStyle=C_MID; ctx.font="12px -apple-system,sans-serif";
            ctx.fillText("|\u00acDO\u27e9  P=0.0",sp1,sp2+20);
            ctx.textAlign="left";

            // State vector — in world space, same 3D rotation applied
            const th=S.theta, ph=S.phi;
            const svx=Math.sin(th)*Math.cos(ph);
            const svy=Math.cos(th);
            const svz=Math.sin(th)*Math.sin(ph);
            const [ovx,ovy]=proj(0,0,0,ry,rx);
            const [vpx,vpy]=proj(svx*0.92,svy*0.92,svz*0.92,ry,rx);

            // Glow
            ctx.shadowColor=riskCol+"44"; ctx.shadowBlur=16;
            ctx.beginPath(); ctx.moveTo(ovx,ovy); ctx.lineTo(vpx,vpy);
            ctx.strokeStyle=riskCol; ctx.lineWidth=4; ctx.stroke();
            ctx.shadowBlur=0;

            // Arrowhead
            const ang2=Math.atan2(vpy-ovy,vpx-ovx);
            ctx.beginPath();
            ctx.moveTo(vpx,vpy);
            ctx.lineTo(vpx-13*Math.cos(ang2-0.38),vpy-13*Math.sin(ang2-0.38));
            ctx.lineTo(vpx-13*Math.cos(ang2+0.38),vpy-13*Math.sin(ang2+0.38));
            ctx.closePath(); ctx.fillStyle=riskCol; ctx.fill();

            // Vector label with pill
            const lbl=`|\u03c8\u27e9  P(DO) = ${{(S.risk*100).toFixed(1)}}%`;
            ctx.font="bold 13px -apple-system,sans-serif";
            const lw=ctx.measureText(lbl).width;
            const lx=vpx>cx ? vpx+12 : vpx-lw-12;
            const ly=vpy<90 ? vpy+22 : vpy-10;
            ctx.fillStyle="rgba(255,255,255,0.90)";
            ctx.fillRect(lx-3,ly-14,lw+6,19);
            ctx.fillStyle=riskCol; ctx.fillText(lbl,lx,ly);

            // Equatorial superposition ring
            if(S.si>0.6){{
              ctx.beginPath(); let fi3=true;
              for(let a=0;a<=360;a+=4){{
                const r=a*Math.PI/180;
                const [sx,sy]=proj(Math.cos(r),0,Math.sin(r),ry,rx);
                fi3?(ctx.moveTo(sx,sy),fi3=false):ctx.lineTo(sx,sy);
              }}
              ctx.closePath();
              ctx.strokeStyle="#B8850A55"; ctx.lineWidth=2;
              ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
            }}

            // Centre dot
            ctx.beginPath(); ctx.arc(cx,cy,4,0,Math.PI*2);
            ctx.fillStyle="#bbb"; ctx.fill();

            // Sub-label
            document.getElementById("bloch-label").textContent =
              "\u03b8 = "+(S.theta*180/Math.PI).toFixed(1)+"\u00b0  \u00b7  "
              +"\u03c6 = "+(S.phi*180/Math.PI).toFixed(1)+"\u00b0  \u00b7  "
              +"SI = "+S.si.toFixed(3);
          }}

          // ── 3D precession animation ──────────────────────────────────────────────
          // Mimics the physical precession of a spin-1/2 particle on the Bloch sphere:
          //   - Primary: slow continuous rotation around Z (vertical) axis  — azimuthal precession
          //   - Secondary: gentle sinusoidal tilt around X axis             — nutation / latitude wobble
          // Together these trace an elegant looping path characteristic of genuine
          // Bloch-sphere dynamics under a static magnetic field.
          let t0=null;
          function animate(ts){{
            if(!t0) t0=ts;
            const e=(ts-t0)/1000;                 // seconds elapsed
            const ry = (e * 2*Math.PI/20);        // full Y rotation every 20s
            const rx = 0.28*Math.sin(e*2*Math.PI/9);  // ±16° X tilt, period 9s
            draw(ry, rx);
            requestAnimationFrame(animate);
          }}
          requestAnimationFrame(animate);
        }})();
        </script>
        """

            st.components.v1.html(bloch_html, height=620, scrolling=False)

        with _bloch_qb2:
            # State-vector readout — uses canvas-convention angles for consistency
            st.markdown("#### State vector")
            _state_col = ("#A32D2D" if p_high_n20 >= 0.55
                          else "#BA7517" if p_high_n20 >= 0.35
                          else "#3B6D11")
            st.markdown(
                f"<div style='background:#f8f8f8;border-radius:10px;"
                f"padding:.8rem 1rem;margin-bottom:.6rem'>"
                f"<div style='font-size:.72rem;color:#888;margin-bottom:3px'>"
                f"Node 20 — DO designation risk</div>"
                f"<div style='font-size:1.8rem;font-weight:800;font-family:monospace;"
                f"color:{_state_col}'>{p_high_n20*100:.1f}%</div>"
                f"<div style='font-size:.8rem;color:#555;margin-top:4px'>"
                f"θ = {_theta_canvas_deg:.1f}° &nbsp;·&nbsp; φ = {_phi_canvas_deg:.1f}° "
                f"&nbsp;·&nbsp; SI = {si:.3f}</div></div>",
                unsafe_allow_html=True
            )

            # Density matrix readout (uses the same density_matrix_summary that
            # the original tab used — keeps coherence/purity computations consistent)
            try:
                from quantum_diagnostics import density_matrix_summary as _dms
                _dm = _dms(p_high_n20)
                st.markdown(
                    f"<div style='background:#F7F5F2;border:1px solid #E0DDD6;"
                    f"border-radius:8px;padding:.65rem 1rem;margin-bottom:.5rem;"
                    f"font-size:.78rem'>"
                    f"<div style='font-weight:700;color:#1B2A4A;margin-bottom:4px'>"
                    f"Density matrix ρ</div>"
                    f"<div style='font-family:monospace;color:#333;margin-bottom:4px'>"
                    f"ρ = [[{_dm['rho'][0][0]:.3f}, &nbsp;{_dm['rho'][0][1]:.3f}]<br>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[{_dm['rho'][1][0]:.3f}, "
                    f"&nbsp;{_dm['rho'][1][1]:.3f}]]</div>"
                    f"<div style='color:#666'>"
                    f"Coherence: <b>{_dm['coherence']}</b> &nbsp;·&nbsp; "
                    f"Purity Tr(ρ²): <b>{_dm['purity']}</b><br>"
                    f"<span style='color:#888;font-style:italic'>"
                    f"{_dm['interpretation']}</span></div></div>",
                    unsafe_allow_html=True
                )
            except Exception as _dm_err:
                st.caption(f"Density matrix unavailable: {_dm_err}")

        # ── Plain-language audit report (collapsed by default) ────────────────────
        if st.session_state.get("qbism_plain"):
            with st.expander("Plain-language audit report", expanded=False):
                st.markdown(st.session_state.qbism_plain)

# ── T8: Document analysis ─────────────────────────────────────────────────────
with TABS[9]:
    st.markdown("### Document analysis")
    st.caption("Upload legal documents for Tetrad-grounded analysis. The LLM provides guidance — **you retain full discretion**.")
    st.info("**Supported:** Gladue reports · IRCA reports · PCL-R/Static-99R assessments · Prior decisions · Transcripts · Bail records · Trauma assessments · Ineffective assistance records")
    # ── Shared CSS for Documents tab collapsibles + tier styling ─────────
    st.markdown('''<style>
details.parvis-doc {
    background: #FBFAF7;
    border: 1px solid #E0DDD6;
    border-radius: 8px;
    padding: 0;
    margin: 14px 0 18px 0;
    transition: border-color 120ms ease;
}
details.parvis-doc[open] {
    border-color: #C7C2B8;
}
details.parvis-doc > summary {
    list-style: none;
    cursor: pointer;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: "Fraunces", Georgia, serif;
    font-weight: 500;
    font-size: 1.05rem;
    color: #1a1a1a;
    letter-spacing: -0.005em;
    user-select: none;
}
details.parvis-doc > summary::-webkit-details-marker {
    display: none;
}
details.parvis-doc > summary::after {
    content: "+";
    font-family: "DM Sans", sans-serif;
    font-weight: 400;
    font-size: 1.2rem;
    color: #707070;
    line-height: 1;
}
details.parvis-doc[open] > summary::after {
    content: "−";
    color: #1a1a1a;
}
details.parvis-doc[open] > summary {
    border-bottom: 1px solid #EFEDE7;
}
details.parvis-doc > .parvis-doc-body {
    padding: 16px 20px 4px 20px;
}
details.parvis-doc > .parvis-doc-body p {
    font-family: "Fraunces", Georgia, serif;
    font-size: 0.92rem;
    color: #3a3a3a;
    line-height: 1.7;
    margin: 0 0 14px 0;
}
details.parvis-doc > .parvis-doc-body p:last-child {
    margin-bottom: 6px;
}
details.parvis-doc > .parvis-doc-body p strong {
    font-weight: 500;
    color: #1a1a1a;
}
details.parvis-doc > .parvis-doc-body em { font-style: italic; }
.parvis-doc-purpose {
    font-family: "Fraunces", Georgia, serif;
    font-style: italic;
    font-size: 0.95rem;
    color: #3a3a3a;
    line-height: 1.7;
    max-width: 880px;
    margin: 6px 0 8px 0;
}
.parvis-canlii-tier-binding {
    border-left: 3px solid #3B6D11;
    padding: 6px 12px;
    margin: 4px 0;
    background: rgba(59, 109, 17, 0.05);
    border-radius: 0 4px 4px 0;
}
.parvis-canlii-tier-persuasive {
    border-left: 3px solid #9E9E9E;
    padding: 6px 12px;
    margin: 4px 0;
    background: rgba(158, 158, 158, 0.05);
    border-radius: 0 4px 4px 0;
}
.parvis-canlii-tier-other {
    border-left: 3px solid #E0DDD6;
    padding: 6px 12px;
    margin: 4px 0;
}
.parvis-canlii-tier-label {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 6px;
    vertical-align: middle;
}
.parvis-canlii-corpus-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 1px 6px;
    border-radius: 3px;
    margin-left: 8px;
    vertical-align: middle;
}
.corpus-distortion {
    background: #E8F0FA;
    color: #185FA5;
    border: 1px solid #C7D3E5;
}
.corpus-proportionality {
    background: #FAEEDA;
    color: #BA7517;
    border: 1px solid #E5CC95;
}
</style>
''', unsafe_allow_html=True)
    # ── Section 1: Document analysis — explanatory framing (Apr 27 2026) ──
    st.markdown('''<div class="parvis-doc-purpose">Document analysis applies LLM-assisted reading to legal documents you upload — Gladue reports, IRCA reports, PCL-R or Static-99R assessments, transcripts, prior decisions — and proposes adjustments to specific PARVIS nodes. Adjustments are surfaced for your review; the architecture applies them only on your explicit acceptance.</div>''', unsafe_allow_html=True)
    st.markdown('''<details class="parvis-doc">
<summary>How document analysis works</summary>
<div class="parvis-doc-body">
<p><strong>What it does.</strong> The selected provider — Claude, GPT-4o, or Gemini — reads the uploaded document, identifies passages bearing on the calibration of one or more PARVIS nodes (Nodes 2 through 20, depending on document type), and returns a structured set of proposed adjustments. Each proposed adjustment specifies a node, a delta (the magnitude and direction of the proposed shift to that node's posterior), the passage from the document that motivates it, and a doctrinal rationale grounded in the Tetrad — <em>Gladue</em>, <em>Ipeelee</em>, <em>Morris</em>, <em>Ewert</em> — and downstream binding authority where relevant.</p>
<p><strong>How adjustments feed back into the model.</strong> Adjustments are not automatic. Each proposed adjustment is presented with a checkbox; the user reviews the rationale and either accepts or declines it. Accepted adjustments are passed to the network as evidence, and the inference engine recomputes posteriors. Declined adjustments are recorded but do not modify the network. The user can clear all document-driven adjustments at any time using the <em>Clear document adjustments</em> control. This design reflects the architecture's audit purpose: PARVIS surfaces what the document supports; the user decides what the network should believe.</p>
<p><strong>Symmetric application.</strong> The feature accepts whatever document the user uploads — Gladue reports prepared for an Indigenous offender, PCL-R assessments commissioned by the Crown, defence-side trauma assessments, prosecution sentencing memoranda. The LLM applies the same analytical schema in either direction. The architecture is designed to support fair audit, not to advocate for any party's preferred reading.</p>
</div>
</details>''', unsafe_allow_html=True)

    # Provider selector — backward-compatible, Claude is default
    col_prov, col_key = st.columns([1, 2])
    with col_prov:
        provider = st.selectbox(
            "AI provider",
            ["claude", "openai", "gemini"],
            format_func=lambda x: {
                "claude":  "Claude (Anthropic) ★ recommended",
                "openai":  "GPT-4o (OpenAI)",
                "gemini":  "Gemini 1.5 Pro (Google)",
            }[x],
            key="doc_provider",
        )
    with col_key:
        key_label = {"claude":"Anthropic API key","openai":"OpenAI API key","gemini":"Google API key"}.get(provider,"API key")
        ak = st.text_input(f"{key_label} (optional — uses Streamlit secrets if blank)",
                           type="password", key="ak")
    up=st.file_uploader("Upload document",type=["txt","pdf","docx"],key="doc_up")
    if up:
        # ── Jurisdiction override (stare decisis layer) ────────────────────
        # Auto-detect runs if either dropdown is left on "Auto-detect".
        # Explicit selection forces that value rather than inference from text.
        col_prov_j, col_lvl_j = st.columns(2)
        with col_prov_j:
            prov_choice = st.selectbox(
                "Document jurisdiction",
                ["Auto-detect", "ON — Ontario", "BC — British Columbia",
                 "AB — Alberta", "QC — Quebec", "SK — Saskatchewan",
                 "MB — Manitoba", "NS — Nova Scotia", "NB — New Brunswick",
                 "NL — Newfoundland and Labrador", "PE — Prince Edward Island",
                 "YT — Yukon", "NT — Northwest Territories", "NU — Nunavut"],
                key="doc_prov",
                help="Province the document originates from. Affects binding "
                     "force of cited authorities under Canadian stare decisis. "
                     "Leave on Auto-detect to infer from document text.",
            )
        with col_lvl_j:
            lvl_choice = st.selectbox(
                "Court level",
                ["Auto-detect", "CA — Court of Appeal",
                 "SC — Superior trial court (KB/QB/SC)",
                 "PC — Provincial/inferior court"],
                key="doc_lvl",
                help="Court level of the document. Superior trial courts are "
                     "called King's Bench (AB/SK/MB/NB), Superior Court of "
                     "Justice (ON), or Supreme Court of [Province] (BC/NS/NL/PE).",
            )
        dt_override=st.selectbox("Document type",["Auto-detect","Gladue report","IRCA",
            "Psychometric (PCL-R)","Psychometric (Static-99R)","FASD assessment",
            "Bail hearing record","Prior sentencing decision","Court transcript",
            "Ineffective assistance record","Trauma assessment","Other legal document"],key="dt_ov")
        if st.button("Analyze against Tetrad framework",type="primary",key="ana"):
            try:
                from document_analyzer import extract_text_from_upload,analyze_document
                up.seek(0)
                with st.spinner("Analyzing against Tetrad framework..."):
                    content,auto_type=extract_text_from_upload(up)
                    dt=auto_type if dt_override=="Auto-detect" else dt_override
                    # Resolve jurisdiction overrides — None triggers auto-detect
                    doc_jur_arg = None if prov_choice == "Auto-detect" else prov_choice.split(" — ")[0].lower()
                    doc_lvl_arg = None if lvl_choice == "Auto-detect" else lvl_choice.split(" — ")[0].lower()
                    result=analyze_document(content,dt,ak or None,provider=provider,
                                            doc_jurisdiction=doc_jur_arg, doc_level=doc_lvl_arg)
                st.success(f"Complete · {dt} · {result.get('_provider',provider).upper()} · Framework: {result.get('applicable_framework','?').upper()} · Connection: {result.get('connection_assessment','?')}")
                st.markdown(f"*{result.get('document_summary','')}*")
                # ── Stare decisis: jurisdiction and splits ───────────────
                sd = result.get("stare_decisis", {}) or {}
                sd_jur = sd.get("document_jurisdiction")
                sd_lvl = sd.get("document_court_level")
                sd_src = sd.get("jurisdiction_source", "undetermined")
                if sd_jur or sd_lvl:
                    _PROV_NAMES = {"federal":"Federal","on":"Ontario","bc":"British Columbia",
                                   "ab":"Alberta","qc":"Quebec","sk":"Saskatchewan",
                                   "mb":"Manitoba","ns":"Nova Scotia","nb":"New Brunswick",
                                   "nl":"Newfoundland and Labrador","pe":"Prince Edward Island",
                                   "yt":"Yukon","nt":"Northwest Territories","nu":"Nunavut"}
                    _LVL_NAMES = {"ca":"Court of Appeal","sc":"superior trial court",
                                  "pc":"provincial/inferior court"}
                    jur_label = _PROV_NAMES.get(sd_jur, sd_jur or "undetermined")
                    lvl_label = _LVL_NAMES.get(sd_lvl, sd_lvl or "undetermined")
                    src_color = "#3B6D11" if sd_src.startswith("explicit") else "#BA7517" if "high" in sd_src else "#888"
                    st.markdown(
                        f"<div style='background:#f8f6f1;border-left:3px solid {src_color};"
                        f"padding:8px 14px;margin:8px 0;border-radius:0 6px 6px 0;font-size:13px'>"
                        f"<b style='color:#1F3A5F'>Stare decisis — document context</b><br>"
                        f"<span style='color:#555'>Jurisdiction: <b>{jur_label}</b> · "
                        f"Court level: <b>{lvl_label}</b> · "
                        f"<span style='color:{src_color}'>source: {sd_src}</span></span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Document jurisdiction could not be determined. "
                            "Cross-province binding force cannot be computed for this analysis. "
                            "Use the Document jurisdiction dropdown above to set it manually.")
                # Inter-provincial splits — high-value diagnostic
                splits = sd.get("inter_provincial_splits", []) or []
                for spl in splits:
                    provs = ", ".join(p.upper() for p in spl.get("provinces", []))
                    st.warning(
                        f"⚖️  **Inter-provincial CA split detected** ({provs}) — "
                        f"{spl.get('note','')}"
                    )
                # ── Binding force table for all Tetrad authorities ───────
                if sd_jur:
                    try:
                        from stare_decisis import classify_authorities_for_prompt, BindingForce
                        _TETRAD_CITS = [
                            "R v Gladue 1999 1 SCR 688",
                            "R v Ipeelee 2012 SCC 13",
                            "R v Morris 2021 ONCA 680",
                            "R v Ellis 2022 BCCA 278",
                            "Ewert v Canada 2018 SCC 30",
                            "R v Boutilier 2017 SCC 64",
                            "R v Natomagan 2022 ABCA 48",
                            "R v Le 2019 SCC 34",
                            "R v Antic 2017 SCC 27",
                        ]
                        bf_rows = classify_authorities_for_prompt(sd_jur, sd_lvl, _TETRAD_CITS)
                        if bf_rows:
                            _BF_COL = {
                                "binding":             ("#3B6D11","BINDING"),
                                "strongly_persuasive": ("#BA7517","STRONGLY PERSUASIVE"),
                                "persuasive":          ("#888888","PERSUASIVE"),
                                "under_review":        ("#185FA5","UNDER REVIEW"),
                                "not_applicable":      ("#A32D2D","OVERRULED"),
                                "unknown":             ("#aaaaaa","UNCLASSIFIED"),
                            }
                            with st.expander("⚖️ Tetrad binding force — relative to this document", expanded=True):
                                st.caption(f"Binding force of Tetrad authorities on a **{lvl_label}** in **{jur_label}**")
                                for row in bf_rows:
                                    force = row.get("binding_force","unknown")
                                    fc, fl = _BF_COL.get(force, ("#aaa","UNCLASSIFIED"))
                                    badge = (f"<span style='background:{fc};color:white;font-size:10px;"
                                             f"font-weight:700;padding:2px 8px;border-radius:3px;"
                                             f"margin-right:8px;letter-spacing:.4px'>{fl}</span>")
                                    yr = f" [{row.get('year','')}]" if row.get('year') else ""
                                    st.markdown(
                                        f"<div style='padding:4px 0;border-bottom:1px solid #eee'>"
                                        f"{badge}<span style='font-size:13px;font-style:italic'>"
                                        f"{row.get('citation','')}{yr}</span></div>",
                                        unsafe_allow_html=True)
                    except Exception as _se:
                        pass

                st.markdown("#### Suggested node adjustments")
                acc=dict(st.session_state.doc_adj)
                sig={k:v for k,v in result.get("nodes",{}).items() if abs(v.get("delta",0))>.02 and v.get("confidence",0)>.1}
                if sig:
                    for ns,nd in sig.items():
                        nid=int(ns);m=NODE_META.get(nid);col=TC.get(m.get("type","risk"),"#888") if m else "#888"
                        with st.expander(f"N{nid}: {m['name'] if m else '?'} — {'↑' if nd['delta']>0 else '↓'} {abs(nd['delta']):.2f} (conf {nd['confidence']:.0%})"):
                            st.markdown(f"**Reasoning:** {nd.get('reasoning','')}")
                            # Binding-force badges for each citation
                            _BF_STYLES = {
                                "binding":              ("#3B6D11", "BINDING"),
                                "strongly_persuasive":  ("#BA7517", "STRONGLY PERSUASIVE"),
                                "persuasive":           ("#888888", "PERSUASIVE"),
                                "under_review":         ("#185FA5", "UNDER REVIEW"),
                                "not_applicable":       ("#A32D2D", "NOT APPLICABLE"),
                                "unknown":              ("#aaaaaa", "UNCLASSIFIED"),
                            }
                            try:
                                from stare_decisis import classify_authority, binding_force as _bf
                                _sd_ready = True
                            except Exception:
                                _sd_ready = False
                            for c in nd.get("citations",[])[:3]:
                                badge_html = ""
                                if _sd_ready and sd_jur:
                                    try:
                                        meta = classify_authority(c)
                                        force = _bf(sd_jur, sd_lvl, meta)
                                        bcol, blabel = _BF_STYLES.get(force, _BF_STYLES["unknown"])
                                        badge_html = (
                                            f"<span style='background:{bcol};color:white;"
                                            f"font-size:10px;font-weight:600;padding:2px 7px;"
                                            f"border-radius:3px;letter-spacing:.5px;"
                                            f"margin-right:6px;vertical-align:middle'>{blabel}</span>"
                                        )
                                    except Exception:
                                        pass
                                st.markdown(f"{badge_html}> *{c}*", unsafe_allow_html=True)
                            if st.checkbox(f"Accept adjustment for N{nid}",key=f"acc_{nid}_{len(st.session_state.doc_res)}"): acc[nid]=nd["delta"]
                            elif nid in acc: del acc[nid]
                    st.session_state.doc_adj=acc
                else: st.info("No significant adjustments identified.")
                for flag in result.get("doctrinal_flags",[]): st.warning(flag)
                if result.get("ewert_concern"): st.error("⚠️ Ewert concern flagged")
                st.session_state.doc_res.append(result)
                run_inf()
                # ── CanLII live case law (opt-in per JP Apr 27 2026 rebuild) ─
                flagged=[int(k) for k,v in sig.items() if abs(v.get("delta",0))>0.02]
                if CANLII_ON and canlii_ok() and flagged:
                    st.markdown("---")
                    st.caption(
                        f"After analysis, you can search CanLII for recent decisions on the "
                        f"{len(flagged)} flagged node(s). Results are tiered by stare decisis "
                        f"position relative to your jurisdiction. CanLII has rate limits — "
                        f"search runs only when explicitly requested."
                    )
                    if st.checkbox("🔍 Search CanLII for recent cases on flagged nodes",
                                   key=f"doc_canlii_optin_{len(st.session_state.doc_res)}"):
                        # Resolve user jurisdiction from doc_prov dropdown if set
                        _doc_prov_raw = st.session_state.get("doc_prov", "Auto-detect")
                        _user_jur = "*" if _doc_prov_raw == "Auto-detect" else _doc_prov_raw.split(" — ")[0]
                        st.markdown(f"#### 🔍 Live CanLII — recent decisions on flagged nodes")
                        st.caption(f"Jurisdiction: {_user_jur if _user_jur != '*' else 'Not specified — all results uniformly tagged'} · Date floor: 3 years")
                        for nid in flagged[:4]:
                            nm = NODE_META.get(nid, {})
                            col = TC.get(nm.get("type", "distortion"), "#185FA5")
                            with st.spinner(f"Searching Node {nid}..."):
                                tiered = search_node_developments(
                                    nid, max_results=4,
                                    user_jurisdiction=_user_jur,
                                    date_floor_label="3 years",
                                )
                            if tiered.get("error"):
                                st.caption(f"N{nid}: {tiered['error']}")
                                continue
                            total = tiered.get("total", 0)
                            if total == 0:
                                st.caption(f"N{nid}: No recent results found.")
                                continue
                            st.markdown(f"<b style='color:{col}'>N{nid} — {nm.get('short','')}</b> "
                                        f"<span style='color:#888;font-size:12px'>({total} results)</span>",
                                        unsafe_allow_html=True)
                            # Render binding tier first, then persuasive
                            for tier_key, tier_label, tier_class in [
                                ("binding", "Binding", "parvis-canlii-tier-binding"),
                                ("persuasive", "Persuasive", "parvis-canlii-tier-persuasive"),
                            ]:
                                for r in tiered.get(tier_key, []):
                                    dt2 = r.get("date", "")[:10] or "—"
                                    cit2 = r.get("citation") or r.get("title", "—")
                                    ur = r.get("url", "")
                                    lnk = f"<a href='{ur}' target='_blank'>{cit2}</a>" if ur else cit2
                                    db = r.get("database", "")
                                    st.markdown(
                                        f"<div class='{tier_class}' style='font-size:12px'>"
                                        f"<span class='parvis-canlii-tier-label' "
                                        f"style='background:{'#3B6D1133' if tier_key=='binding' else '#9E9E9E33'};"
                                        f"color:{'#3B6D11' if tier_key=='binding' else '#3a3a3a'}'>"
                                        f"{tier_label} · {db.upper()}</span> "
                                        f"<span style='color:#888'>[{dt2}]</span> {lnk}"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                elif CANLII_ON and not canlii_ok():
                    st.info("Add **CANLII_API_KEY** to Streamlit secrets to enable live CanLII search.")
            except ImportError: st.error("Requires `anthropic` package in requirements.txt")
            except Exception as e: st.error(f"Error: {e}")
    if st.session_state.doc_adj:
        st.markdown("---\n#### Active document adjustments")
        for nid,d in st.session_state.doc_adj.items():
            m=NODE_META.get(nid,{})
            st.markdown(f"<span style='color:{TC.get(m.get('type','risk'),'#888')}'>●</span> N{nid} {m.get('name','')}: {'↑' if d>0 else '↓'} {abs(d):.2f}",unsafe_allow_html=True)
        if st.button("Clear document adjustments"): st.session_state.doc_adj={}; run_inf(); st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # Section 2 — Live case law (rebuilt April 27 2026)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📡 Live case law")

    # Always-visible purpose statement
    st.markdown(
        '''<div class="parvis-doc-purpose">The CanLII surface queries the Canadian Legal Information Institute API for recent decisions relevant to PARVIS's doctrinal substrate. Its purpose is to surface authority that may have moved or refined the law since the architecture's static doctrinal grounding (<em>doctrine.py</em>) was last updated, so the audit is conducted against current, not stale, doctrine.</div>''',
        unsafe_allow_html=True,
    )

    # Expandable "How it works" detail
    st.markdown('''<details class="parvis-doc">
<summary>How live case law works</summary>
<div class="parvis-doc-body">
<p><strong>Why this surface exists.</strong> PARVIS's doctrinal grounding is a snapshot. The Tetrad — <em>Gladue</em>, <em>Ipeelee</em>, <em>Morris</em>, <em>Ewert</em> — and the downstream binding decisions that interpret them are living law, subject to refinement by the Supreme Court of Canada, provincial appellate courts, and binding statutory amendment. An audit conducted against an unchanged doctrinal baseline could miss material movements: <em>Sharma</em> (2022 SCC 39) reshaping conditional sentence availability; <em>Friesen</em> (2020 SCC 9) on parity in sexual offences; ongoing appellate refinement of <em>Ipeelee</em>'s contextual reasoning requirement. The CanLII surface flags such movements so the user knows when the substrate beneath the audit has shifted.</p>
<p><strong>What it does.</strong> Three query modes. <em>Subsequent history</em> tracks recent citing cases for the load-bearing decisions in PARVIS's doctrinal substrate, drawn from both the Tetrad (distortion-mitigation lineage) and the proportionality corpus (<em>Lacasse</em>, <em>Friesen</em>, <em>Bissonnette</em>, <em>Sharma</em> — severity-tightening lineage). <em>Per-node search</em> identifies recent decisions relevant to a specific PARVIS node — for example, recent appellate engagement with the <em>Ewert</em> concern at Node 5, or with the <em>Antic</em> bail-denial cascade at Node 7. <em>Directed search</em> accepts a free-text query and returns jurisdictionally tiered results.</p>
<p><strong>How it feeds back into the model.</strong> It does not. The CanLII surface is informational. It does not modify the network's CPTs, posteriors, or any other inferential element. Belief revision in light of new authority remains the user's function — the architecture surfaces what has changed; the user decides whether the audit's framing should change with it.</p>
<p><strong>Symmetric surface.</strong> The CanLII feature returns whatever the law currently is. It surfaces authority that strengthens distortion-based mitigation arguments and authority that tightens proportionality for serious offenders alike. The architecture's purpose is to render the current doctrinal substrate visible — not to advocate for any party's preferred reading of it.</p>
</div>
</details>''', unsafe_allow_html=True)

    # ── CanLII activation status (April 27 2026 gating fix) ──────────────
    # The architecture below renders regardless of CanLII activation. Only
    # the action buttons (and result rendering) are gated on `_canlii_active`.
    # Status banner shows so the user understands which buttons will work.
    _canlii_active = bool(CANLII_ON and canlii_ok())
    if not _canlii_active:
        st.markdown(
            "<div style='background:#FAEEDA;border:1px solid #BA751733;border-radius:8px;padding:12px 16px;margin:8px 0 16px 0'>"
            "<b>CanLII not yet active.</b> The architecture below is fully built; live queries activate when an API key is configured.<br>"
            "1. Register free at <a href='https://api.canlii.org' target='_blank'>api.canlii.org</a><br>"
            "2. Add <code>CANLII_API_KEY = your-key</code> to Streamlit secrets<br>"
            "3. Redeploy — surface activates automatically.</div>",
            unsafe_allow_html=True)
    else:
        # Optional API key validity probe (cached 1 hour)
        try:
            _validity = validate_api_key()
            if not _validity.get("valid"):
                st.warning(f"⚠️ CanLII API key configured but probe failed: {_validity.get('error','unknown error')}. "
                           f"Key may be invalid, expired, or rate-limited.")
                _canlii_active = False  # Treat as not-active for button gating
        except Exception:
            pass  # Fail silently; live queries will surface real errors

    # ── Global filters (jurisdiction + date floor) ────────────────────────
    # Jurisdiction defaults to whatever the user already entered for
    # the document (doc_prov), or "*" (all results uniformly tagged)
    # if no document jurisdiction has been set.
    _doc_prov_raw = st.session_state.get("doc_prov", "Auto-detect")
    _default_jur_code = "*" if _doc_prov_raw == "Auto-detect" else _doc_prov_raw.split(" — ")[0]
    _jur_options = ["* — All results uniformly tagged",
                    "ON — Ontario", "BC — British Columbia", "AB — Alberta",
                    "QC — Quebec", "SK — Saskatchewan", "MB — Manitoba",
                    "NS — Nova Scotia", "NB — New Brunswick",
                    "NL — Newfoundland and Labrador", "PE — Prince Edward Island",
                    "YT — Yukon", "NT — Northwest Territories", "NU — Nunavut"]
    _default_idx = 0
    for _i, _opt in enumerate(_jur_options):
        if _opt.startswith(_default_jur_code + " "):
            _default_idx = _i
            break

    col_jur, col_floor = st.columns([2, 1])
    with col_jur:
        _jur_choice = st.selectbox(
            "Jurisdiction (governs binding/persuasive classification)",
            _jur_options,
            index=_default_idx,
            key="canlii_jur",
            help="Cases from the Supreme Court of Canada and from this jurisdiction's "
                 "Court of Appeal are binding. Other provincial Courts of Appeal are "
                 "persuasive only. Lower-court decisions are tagged Other.",
        )
    with col_floor:
        _date_floor = st.selectbox(
            "Date floor",
            ["1 year", "3 years", "5 years", "All"],
            index=1,
            key="canlii_date_floor",
            help="Recency cutoff for surfaced cases.",
        )
    _user_jur = _jur_choice.split(" ")[0]  # First token is the code or "*"

    # ── 2a — Subsequent history tracker (Tetrad + Proportionality) ───
    st.markdown("#### Subsequent history — Tetrad + Proportionality corpus")
    st.caption("Tracks recent citing cases for both lines of binding authority that PARVIS depends on. "
               "<span class='parvis-canlii-corpus-tag corpus-distortion'>Distortion</span> tags the Tetrad lineage "
               "(Gladue/Ipeelee/Morris/Ewert + downstream). "
               "<span class='parvis-canlii-corpus-tag corpus-proportionality'>Proportionality</span> tags the "
               "severity-tightening lineage (Lacasse/Friesen/Bissonnette/Sharma).",
               unsafe_allow_html=True)
    col_corp, col_tet_btn = st.columns([1, 2])
    with col_corp:
        _corpus_choice = st.selectbox(
            "Corpus",
            ["all — Both lineages", "Distortion — Tetrad only", "Proportionality — only"],
            index=0,
            key="canlii_corpus",
        )
    _corpus_arg = _corpus_choice.split(" ")[0]
    with col_tet_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _run_tet = st.button("🔄 Check tracked cases for recent developments", key="tet_btn_v2", disabled=not _canlii_active)
        if not _canlii_active:
            st.caption("_Configure CanLII API access to enable._")
    if _run_tet:
        with st.spinner("Querying CanLII for citing cases..."):
            _updates = get_tracked_updates(
                date_floor_label=_date_floor,
                user_jurisdiction=_user_jur,
                corpus=_corpus_arg,
            )
        if _updates:
            for _case_lbl, _data in _updates.items():
                _corpus = _data.get("corpus", "Distortion")
                _corpus_class = f"corpus-{_corpus.lower()}"
                _total = _data.get("total", 0)
                with st.expander(f"📌 {_case_lbl} — {_total} recent citing case(s)"):
                    st.markdown(
                        f"<span class='parvis-canlii-corpus-tag {_corpus_class}'>{_corpus}</span>",
                        unsafe_allow_html=True,
                    )
                    for _tier_key, _tier_label, _tier_class, _tier_color in [
                        ("binding",    "Binding",    "parvis-canlii-tier-binding",    "#3B6D11"),
                        ("persuasive", "Persuasive", "parvis-canlii-tier-persuasive", "#3a3a3a"),
                        ("other",      "Other",      "parvis-canlii-tier-other",      "#9E9E9E"),
                    ]:
                        _tier_cases = _data.get(_tier_key, [])
                        if not _tier_cases:
                            continue
                        for _c in _tier_cases[:6]:
                            _cd = _c.get("date", "")[:10] or "—"
                            _ct = _c.get("title", "—")
                            _cu = _c.get("url", "")
                            _db = _c.get("database", "")
                            _lnk = f"<a href='{_cu}' target='_blank'>{_ct}</a>" if _cu else _ct
                            st.markdown(
                                f"<div class='{_tier_class}' style='font-size:12px'>"
                                f"<span class='parvis-canlii-tier-label' "
                                f"style='background:{_tier_color}22;color:{_tier_color}'>"
                                f"{_tier_label} · {_db.upper()}</span> "
                                f"<span style='color:#888'>[{_cd}]</span> {_lnk}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
        else:
            st.success(f"No new citing cases found within the {_date_floor.lower()} window.")

    # ── 2b — Per-node binding-authority browser ──────────────────────
    st.markdown("---")
    st.markdown("#### Per-node search — recent decisions for a specific node")
    st.caption("Returns recent cases relevant to a chosen PARVIS node, "
               "tiered by stare decisis position relative to your jurisdiction.")
    col_node, col_pn_btn = st.columns([2, 1])
    with col_node:
        _node_options = [f"N{nid} — {NODE_META.get(nid, {}).get('short', f'Node {nid}')}"
                         for nid in sorted(NODE_SEARCH_QUERIES.keys()) if nid in NODE_META]
        _node_choice = st.selectbox(
            "Select node",
            _node_options,
            key="canlii_node_pick",
        )
    _selected_nid = int(_node_choice.split(" ")[0][1:])
    with col_pn_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _run_node = st.button("🔍 Search recent cases for this node", key="node_search_btn", disabled=not _canlii_active)
        if not _canlii_active:
            st.caption("_Configure CanLII API access to enable._")
    if _run_node:
        with st.spinner(f"Searching CanLII for Node {_selected_nid}..."):
            _node_results = search_node_developments(
                _selected_nid, max_results=6,
                user_jurisdiction=_user_jur,
                date_floor_label=_date_floor,
            )
        if _node_results.get("error"):
            st.error(_node_results["error"])
        elif _node_results.get("total", 0) == 0:
            st.info(f"No recent results found for Node {_selected_nid} within the {_date_floor.lower()} window.")
        else:
            for _tier_key, _tier_label, _tier_class, _tier_color in [
                ("binding",    "Binding",    "parvis-canlii-tier-binding",    "#3B6D11"),
                ("persuasive", "Persuasive", "parvis-canlii-tier-persuasive", "#3a3a3a"),
                ("other",      "Other",      "parvis-canlii-tier-other",      "#9E9E9E"),
            ]:
                _tier_cases = _node_results.get(_tier_key, [])
                if not _tier_cases:
                    continue
                st.markdown(f"<b style='color:{_tier_color}'>{_tier_label} ({len(_tier_cases)})</b>",
                            unsafe_allow_html=True)
                for _c in _tier_cases:
                    _cd = _c.get("date", "")[:10] or "—"
                    _cit = _c.get("citation") or _c.get("title", "—")
                    _cu = _c.get("url", "")
                    _db = _c.get("database", "")
                    _lnk = f"<a href='{_cu}' target='_blank'>{_cit}</a>" if _cu else _cit
                    st.markdown(
                        f"<div class='{_tier_class}' style='font-size:12px'>"
                        f"<span class='parvis-canlii-tier-label' "
                        f"style='background:{_tier_color}22;color:{_tier_color}'>"
                        f"{_tier_label} · {_db.upper()}</span> "
                        f"<span style='color:#888'>[{_cd}]</span> {_lnk}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── 2c — Directed search ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Directed search — free-text query")
    st.caption("Run a free-text query against CanLII. Results are filtered by your "
               "jurisdiction and date floor, then tiered by stare decisis position.")
    col_q, col_q_btn = st.columns([3, 1])
    with col_q:
        _query = st.text_input(
            "Search query",
            placeholder="e.g. dangerous offender Indigenous Gladue 2024",
            key="canlii_directed_q",
        )
    with col_q_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _run_q = st.button("🔍 Search CanLII", key="directed_search_btn", disabled=not _canlii_active)
        if not _canlii_active:
            st.caption("_Configure CanLII API access to enable._")
    if _run_q and _query.strip():
        with st.spinner(f"Searching CanLII for '{_query[:40]}'..."):
            _q_results = search_with_filters(
                query=_query.strip(),
                user_jurisdiction=_user_jur,
                date_floor_label=_date_floor,
                max_results=8,
            )
        if _q_results.get("error"):
            st.error(_q_results["error"])
        elif _q_results.get("total", 0) == 0:
            st.info(f"No results found for '{_query}' within the {_date_floor.lower()} window.")
        else:
            st.caption(f"Total results: {_q_results.get('total', 0)}")
            for _tier_key, _tier_label, _tier_class, _tier_color in [
                ("binding",    "Binding",    "parvis-canlii-tier-binding",    "#3B6D11"),
                ("persuasive", "Persuasive", "parvis-canlii-tier-persuasive", "#3a3a3a"),
                ("other",      "Other",      "parvis-canlii-tier-other",      "#9E9E9E"),
            ]:
                _tier_cases = _q_results.get(_tier_key, [])
                if not _tier_cases:
                    continue
                st.markdown(f"<b style='color:{_tier_color}'>{_tier_label} ({len(_tier_cases)})</b>",
                            unsafe_allow_html=True)
                for _c in _tier_cases:
                    _cd = _c.get("date", "")[:10] or "—"
                    _cit = _c.get("citation") or _c.get("title", "—")
                    _cu = _c.get("url", "")
                    _db = _c.get("database", "")
                    _lnk = f"<a href='{_cu}' target='_blank'>{_cit}</a>" if _cu else _cit
                    st.markdown(
                        f"<div class='{_tier_class}' style='font-size:12px'>"
                        f"<span class='parvis-canlii-tier-label' "
                        f"style='background:{_tier_color}22;color:{_tier_color}'>"
                        f"{_tier_label} · {_db.upper()}</span> "
                        f"<span style='color:#888'>[{_cd}]</span> {_lnk}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Section 3: Doctrine update alerts (preserved) ────────────────
    st.markdown("---")
    st.markdown("#### ⚡ Doctrine update alerts")
    st.caption("Nodes with actively evolving law — flagged in doctrine.py:")
    try:
        from doctrine import get_update_notes
        notes = get_update_notes()
        for nid, note in notes.items():
            nm = NODE_META.get(nid, {})
            col = TC.get(nm.get("type", "distortion"), "#185FA5")
            st.markdown(
                f"<div style='border-left:3px solid {col};padding:6px 12px;margin:4px 0;"
                f"background:{col}11;border-radius:0 6px 6px 0'>"
                f"<b style='color:{col}'>N{nid} — {nm.get('short','')}</b><br>"
                f"<span style='font-size:12px'>{note}</span></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("doctrine.py update notes unavailable.")

# ── T3: Intake (Chat) — PARVIS assistant ────────────────────────────────────
with TABS[3]:

    # ── Chat UI styling ───────────────────────────────────────────────────────
    st.markdown("""
<style>
/* ── Header ── */
.parvis-chat-header {
  display:flex; align-items:center; gap:12px;
  padding:.5rem 0 .7rem 0;
  border-bottom:1px solid rgba(0,0,0,.07);
  margin-bottom:.6rem;
}
.parvis-chat-title  { font-size:1.1rem; font-weight:700; color:#1B2A4A; }
.parvis-chat-subtitle { font-size:.73rem; color:#aaa; margin-top:1px; }
.parvis-node20-pill {
  margin-left:auto; font-size:.77rem; font-weight:700;
  padding:3px 12px; border-radius:20px;
}

/* ── Message rows ── */
[data-testid="stChatMessage"] {
  padding: .25rem 0 !important;
  gap: 8px !important;
}

/* ── USER messages — navy bubble right-aligned ── */
[data-testid="stChatMessage"][data-role="user"],
[data-testid="stChatMessageContent"] + [data-testid="stChatMessage"] {
  flex-direction: row-reverse !important;
}

/* Target user message content by avatar data attribute */
[data-testid="stChatMessage"]:has(img[alt="user avatar"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
  background: #1B2A4A !important;
  border-radius: 18px 4px 18px 18px !important;
  padding: .6rem 1rem !important;
  max-width: 78% !important;
  margin-left: auto !important;
  color: white !important;
  border: none !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] * {
  color: white !important;
}

/* ── PARVIS messages — warm card left-aligned ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
  background: #FDFCFA !important;
  border: 1px solid #E8E4DC !important;
  border-radius: 4px 18px 18px 18px !important;
  padding: .6rem 1rem !important;
  max-width: 88% !important;
}

/* ── Proposal card ── */
.parvis-proposal {
  background:#F0F5FF; border-left:3px solid #185FA5;
  border-radius:0 8px 8px 0; padding:.5rem .8rem;
  margin:.4rem 0; font-size:.84rem;
}
</style>
""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "chat_history"        not in st.session_state: st.session_state.chat_history = []
    if "chat_pending_action" not in st.session_state: st.session_state.chat_pending_action = None
    if "chat_ak"             not in st.session_state: st.session_state.chat_ak = ""
    if "voice_inject"        not in st.session_state: st.session_state.voice_inject = ""

    # ── Header row ────────────────────────────────────────────────────────────
    bl, bc, bg = rb(P[20])
    if _empty:
        _pill_html = (
            f'<div class="parvis-node20-pill" '
            f'style="background:#FBFAF7;color:#9E9E9E;'
            f'border:1px solid #E0DDD6;font-family:Fraunces,Georgia,serif;'
            f'font-style:italic;font-weight:500">'
            f'Node 20 &nbsp;—&nbsp; Awaiting case data'
            f'</div>'
        )
    else:
        _pill_html = (
            f'<div class="parvis-node20-pill" '
            f'style="background:{bg};color:{bc};border:1px solid {bc}44">'
            f'Node 20 &nbsp;{P[20]*100:.1f}% &nbsp;{bl}'
            f'</div>'
        )
    st.markdown(f"""
<div class="parvis-chat-header">
  <div>
    <div class="parvis-chat-title">💬 Intake (Chat)</div>
    <div class="parvis-chat-subtitle">Context-aware · {len(st.session_state.chat_history)//2} exchange(s) · Bayesian network live</div>
  </div>
  {_pill_html}
</div>""", unsafe_allow_html=True)

    # ── API settings (compact bar) ────────────────────────────────────────────
    with st.expander("⚙️ API settings", expanded=not st.session_state.chat_ak):
        ck1, ck2 = st.columns([3, 1])
        with ck1:
            st.session_state.chat_ak = st.text_input(
                "API key",
                type="password", value=st.session_state.chat_ak,
                key="chat_ak_input", label_visibility="collapsed",
                placeholder="Anthropic API key — leave blank if set in Streamlit secrets")
        with ck2:
            chat_provider = st.selectbox("Provider", ["claude", "openai", "gemini"],
                format_func=lambda x: {"claude": "Claude ★", "openai": "GPT-4o", "gemini": "Gemini"}[x],
                key="chat_provider", label_visibility="collapsed")

    # ── Single combined info expander ────────────────────────────────────────
    with st.expander("ℹ️ About PARVIS Chat", expanded=len(st.session_state.chat_history)==0):
        st.markdown("""
**This tab contains a Bayesian AI assistant** grounded in the PARVIS 20-node network. It does not hallucinate or confabulate in the way general LLMs do — it knows the current state of your network at all times and its purpose is to advise on reasonable values. The user remains in control of all inputs.

**What it can do:**
- Answer questions about any node value, doctrinal principle, or network output
- Explain *why* Node 20 is at its current level and which nodes are driving it
- Accept a plain-language case description and propose values across all tabs
- Suggest changes to the network — but only *propose* them; you confirm each one before anything changes

**How to use it:**
1. Type or dictate a case description or question in the input below
2. If PARVIS suggests a change, a confirmation card appears — nothing is written until you click ✅ Apply
3. Navigate to any other tab to verify and fine-tune the populated values

**Context-aware:** every message includes a live snapshot of all 20 node posteriors, active Gladue factors, Morris/Ellis SCE factors, criminal record, connection strength, and Node 20 output.

**Example prompts:**
> *"The offender is 42, Indigenous, from Northern Manitoba. Two prior assault convictions, bail denied for 9 months, Gladue report filed but the judge only cited it in passing. PCL-R score 22."*

> *"Why is Node 7 at 40% and what does that mean for the DO risk?"*

> *"If I remove the Ewert tool correction, what happens to Node 20?"*

**Voice:** click 🎤 Dictate, speak, then copy to the chat input below. **API key** required — enter it in ⚙️ API settings above.
        """)

    # ── Build PARVIS context ──────────────────────────────────────────────────
    def _build_context() -> str:
        Pa = st.session_state.posteriors
        cr = st.session_state.criminal_record
        cG = [f for f in GF if f["id"] in st.session_state.gladue_checked]
        cS = [f for f in SF if f["id"] in st.session_state.sce_checked]
        esc = st.session_state.cr_doc_adj.get("escalation", {})
        ctx = f"""You are PARVIS — the Probabilistic and Analytical Reasoning Virtual Intelligence System,
a Bayesian sentencing assistant for Canadian Dangerous Offender proceedings. You are built on a
20-node pgmpy Bayesian network grounded in the Tetrad doctrinal framework.

CURRENT NETWORK STATE:
Node 20 (DO Designation Risk): {Pa.get(20,0.249)*100:.1f}% — {rb(Pa.get(20,0.249))[0]}

RISK FACTOR POSTERIORS:
- N2 Violent history: {Pa.get(2,0.08)*100:.1f}%
- N3 PCL-R (psychopathy): {Pa.get(3,0.55)*100:.1f}%
- N4 Static-99R (sexual offence): {Pa.get(4,0.32)*100:.1f}%
- N18 Dynamic risk: {Pa.get(18,0.167)*100:.1f}%

DISTORTION CORRECTIONS (reduce effective risk weight):
- N1 Burden of proof: {Pa.get(1,0.83)*100:.1f}%
- N5 Invalid risk tools (Ewert): {Pa.get(5,0.10)*100:.1f}%
- N6 Ineffective counsel: {Pa.get(6,0.15)*100:.1f}%
- N7 Bail-denial cascade: {Pa.get(7,0.40)*100:.1f}%
- N9 FASD: {Pa.get(9,0.15)*100:.1f}%
- N10 Intergenerational trauma: {Pa.get(10,0.45)*100:.1f}%
- N11 No cultural treatment: {Pa.get(11,0.10)*100:.1f}%
- N12 Gladue misapplication: {Pa.get(12,0.15)*100:.1f}%
- N14 Over-policing: {Pa.get(14,0.15)*100:.1f}%
- N15 Temporal distortion: {Pa.get(15,0.40)*100:.1f}%
- N19 No rehabilitation: {Pa.get(19,0.10)*100:.1f}%

GLADUE FACTORS ACTIVE ({len(cG)}): {', '.join([f['l'] for f in cG]) if cG else 'None selected'}
MORRIS/ELLIS SCE ({len(cS)}): {', '.join([f['l'] for f in cS]) if cS else 'None selected'}
Connection strength: {st.session_state.conn} | Framework: {st.session_state.scefw.upper()}

CRIMINAL RECORD ({len(cr)} conviction(s)):"""
        if cr:
            for e in cr:
                ctx += f"\n- [{e['year']}] {e['offence']} — {e.get('sentence_type','?')} — Cal. weight {e['cal_weight']*100:.0f}%"
            ctx += f"\nPattern (Boutilier): {esc.get('pattern','—').title()}"
        else:
            ctx += "\nNo convictions entered."
        ctx += """

YOUR ROLE:
1. Answer questions about the current case, node values, and doctrinal framework
2. Explain what specific posteriors mean and why they are at their current levels
3. Help the user understand the Tetrad framework (Gladue, Ipeelee, Morris, Ellis, Ewert, Boutilier)
4. Accept plain-language case descriptions and propose structured PARVIS values
5. When suggesting changes to node values, ALWAYS end with a JSON block in this exact format:
   PROPOSED_CHANGES: {"node": <id>, "value": <0.0-1.0>, "reason": "<one sentence>"}
   You may propose multiple changes, one JSON block per change.
   Never apply changes yourself — the user must confirm each one.

IMPORTANT: You model DESIGNATION RISK, not intrinsic dangerousness. Always maintain this framing."""
        return ctx

    # ── Chat history display ──────────────────────────────────────────────────
    # Welcome message when no conversation yet
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🔺"):
            bl_w, bc_w, _ = rb(P[20])
            if _empty:
                st.markdown("""
**Hello. I'm PARVIS.**

The network is initialised but no case data has been entered yet. As you provide information about the case, I'll update the posterior in real time.

You can describe a case in plain language and I'll propose values across the network, or ask me to explain any node, doctrinal principle, or risk factor. Nothing changes until you confirm each suggestion.

To get started, try something like:

> *"The offender is 42, Indigenous, from Northern Manitoba. Bail denied for 9 months, PCL-R score 22, no Gladue report commissioned."*

Or ask a question:

> *"What does Node 7 (bail-denial cascade) encode and how does it shift the posterior?"*
                """)
            else:
                st.markdown(f"""
**Hello. I'm PARVIS.**

The network is live — Node 20 is currently at **{P[20]*100:.1f}% ({bl_w})**.

You can describe a case in plain language and I'll propose values across the network, or ask me to explain any node, doctrinal principle, or risk factor. Nothing changes until you confirm each suggestion.

To get started, try something like:

> *"The offender is 42, Indigenous, from Northern Manitoba. Bail denied for 9 months, PCL-R score 22, no Gladue report commissioned."*

Or ask a question:

> *"Why is Node 7 elevated and what does that mean for the DO risk?"*
                """)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"],
                avatar="🔺" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

            # Proposal confirmation cards
            if msg["role"] == "assistant" and msg.get("proposals"):
                for pi, prop in enumerate(msg["proposals"]):
                    nid = prop.get("node")
                    val = prop.get("value")
                    reason = prop.get("reason", "")
                    nm = NODE_META.get(nid, {}).get("name", "?") if nid else "?"
                    cur = P.get(nid, 0.5) if nid else 0.5
                    col_p = "#A32D2D" if val and val > cur else "#3B6D11"
                    arrow = "↑" if val and val > cur else "↓"
                    msg_id = msg.get("id", pi)
                    st.markdown(
                        f"<div class='parvis-proposal'>"
                        f"<b>Proposed:</b> N{nid} {nm} {arrow} "
                        f"<b style='color:{col_p}'>{val*100:.0f}%</b> "
                        f"(current: {cur*100:.0f}%) — {reason}</div>",
                        unsafe_allow_html=True)
                    bc1, bc2 = st.columns([1, 4])
                    with bc1:
                        if st.button(f"✅ Apply", key=f"chat_apply_{msg_id}_{pi}_{nid}"):
                            st.session_state.profile_ev[nid] = float(val)
                            run_inf()
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"✅ Applied — N{nid} ({nm}) set to **{val*100:.0f}%**. Node 20 updated to **{st.session_state.posteriors[20]*100:.1f}%**.",
                                "proposals": []
                            })
                            st.rerun()
                    with bc2:
                        if st.button(f"✗ Decline", key=f"chat_decline_{msg_id}_{pi}_{nid}"):
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Understood — N{nid} ({nm}) left at {cur*100:.0f}%.",
                                "proposals": []
                            })
                            st.rerun()

    # ── Voice input bar ───────────────────────────────────────────────────────
    voice_html = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; font-family: 'DM Sans', -apple-system, sans-serif; }

#voice-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 10px;
  background: #F7F5F2;
  border: 1px solid #E0DDD6;
  border-radius: 10px;
  margin-bottom: 2px;
}

#mic-btn {
  display: flex; align-items: center; gap: 7px;
  background: #1B2A4A; color: white; border: none;
  border-radius: 8px; padding: 6px 14px;
  cursor: pointer; font-size: 13px; font-weight: 600;
  transition: background .18s, transform .1s;
  white-space: nowrap; flex-shrink: 0;
}
#mic-btn:hover { background: #243860; transform: scale(1.03); }
#mic-btn.listening {
  background: #A32D2D;
  animation: pulse 1.2s infinite;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(163,45,45,.4); }
  50%      { box-shadow: 0 0 0 6px rgba(163,45,45,0); }
}

#transcript-display {
  flex: 1;
  font-size: 12.5px;
  color: #555;
  font-style: italic;
  min-height: 18px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
#transcript-display.done { color: #3B6D11; font-style: normal; font-weight: 500; }
#transcript-display.error { color: #A32D2D; font-style: normal; }

#copy-btn {
  display: none;
  background: #185FA5; color: white; border: none;
  border-radius: 8px; padding: 6px 12px;
  cursor: pointer; font-size: 12px; font-weight: 600;
  white-space: nowrap; flex-shrink: 0;
  transition: background .18s;
}
#copy-btn:hover { background: #1a6bb8; }
#copy-btn.copied { background: #3B6D11; }
</style>

<div id="voice-bar">
  <button id="mic-btn" onclick="toggleMic()">🎤 Dictate</button>
  <div id="transcript-display">Specify all applicable details of the offender's case.</div>
  <button id="copy-btn" onclick="copyText()">📋 Copy to chat</button>
</div>

<script>
let recognition = null;
let isListening  = false;
let transcript   = '';

const micBtn   = document.getElementById('mic-btn');
const display  = document.getElementById('transcript-display');
const copyBtn  = document.getElementById('copy-btn');

function toggleMic() {
  if (isListening) { recognition.stop(); return; }

  const hasSR = ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window);
  if (!hasSR) {
    display.textContent = 'Voice requires Chrome or Edge';
    display.className = 'error';
    return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.continuous     = true;
  recognition.interimResults = true;
  recognition.lang           = 'en-CA';

  recognition.onstart = () => {
    isListening = true;
    transcript  = '';
    micBtn.textContent = '⏹ Stop';
    micBtn.classList.add('listening');
    display.textContent = 'Listening…';
    display.className = '';
    copyBtn.style.display = 'none';
  };

  recognition.onresult = (e) => {
    let interim = '', final = '';
    for (let i = 0; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    transcript = final || interim;
    display.textContent = transcript;
  };

  recognition.onerror = (e) => {
    display.textContent = 'Error: ' + e.error;
    display.className = 'error';
    isListening = false;
    micBtn.textContent = '🎤 Dictate';
    micBtn.classList.remove('listening');
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = '🎤 Dictate';
    micBtn.classList.remove('listening');
    if (transcript.trim()) {
      display.textContent = transcript;
      display.className = 'done';
      copyBtn.style.display = 'inline-block';
      copyBtn.textContent = '📋 Copy to chat';
      copyBtn.className = '';
    } else {
      display.textContent = "Specify all applicable details of the offender's case.";
      display.className = '';
    }
  };

  recognition.start();
}

function copyText() {
  if (!transcript.trim()) return;
  navigator.clipboard.writeText(transcript).then(() => {
    copyBtn.textContent = '✓ Copied!';
    copyBtn.className = 'copied';
    display.textContent = '✓ Paste into the chat box below (Cmd+V / Ctrl+V)';
    setTimeout(() => {
      copyBtn.textContent = '📋 Copy to chat';
      copyBtn.className = '';
    }, 3000);
  });
}
</script>
"""
    st.components.v1.html(voice_html, height=54, scrolling=False)
    st.markdown("<style>[data-testid='stChatInput']{margin-top:-18px !important;border-top:none !important;}</style>", unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Specify all applicable details of the offender's case…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.spinner("PARVIS is thinking…"):
            try:
                import anthropic as _ant, json as _json, re as _re

                ak = st.session_state.chat_ak
                if not ak:
                    try: ak = st.secrets.get("ANTHROPIC_API_KEY", "")
                    except Exception: pass
                if not ak:
                    import os; ak = os.environ.get("ANTHROPIC_API_KEY", "")

                if not ak:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "⚠️ No API key found. Please enter your Anthropic API key in the ⚙️ API settings panel above.",
                        "proposals": []
                    })
                    st.rerun()
                else:
                    client = _ant.Anthropic(api_key=ak)
                    messages = []
                    for m in st.session_state.chat_history[:-1]:
                        messages.append({"role": m["role"], "content": m["content"]})
                    messages.append({"role": "user", "content": prompt})

                    response = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=1500,
                        system=_build_context(),
                        messages=messages
                    )
                    raw = response.content[0].text

                    proposals = []
                    for m2 in _re.finditer(r'PROPOSED_CHANGES:\s*(\{[^}]+\})', raw):
                        try: proposals.append(_json.loads(m2.group(1)))
                        except Exception: pass

                    import time
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": raw,
                        "proposals": proposals,
                        "id": int(time.time() * 1000)
                    })
                    st.rerun()

            except Exception as ex:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {ex}",
                    "proposals": []
                })
                st.rerun()

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation", key="chat_clear"):
            st.session_state.chat_history = []
            st.rerun()


# ── T9: Criminal record ──────────────────────────────────────────────────────
with TABS[4]:
    # ════════════════════════════════════════════════════════════════════════
    # Criminal Record tab — full visual rebuild (Mark 8)
    # All substantive logic preserved byte-for-byte: pN extractions,
    # CORR_REFS, factor math, SERIOUSNESS dict, _get_seriousness,
    # _detect_escalation, _cr_feed_nodes, calibrated-weight calculation,
    # entry dict, document-assisted calibration, per-conviction analysis.
    # All widget keys preserved.
    # ════════════════════════════════════════════════════════════════════════

    # ── Tab title + caption ───────────────────────────────────────────────
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.7rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 4px 0'>"
        "Criminal record</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:18px;line-height:1.6;"
        "max-width:880px'>"
        "Each conviction enters the network as a weighted contribution to "
        "the pattern-of-violence and dynamic-risk nodes. Convictions can "
        "carry reduced evidentiary weight where doctrinal distortions are "
        "active — <em>Antic</em> [2017] for coercive pleas, <em>Ewert</em> [2018] "
        "for tool invalidity, <em>Le</em> [2019] for over-policing, <em>Morris</em> "
        "2021 for Gladue misapplication."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Doctrinal anchor strip ────────────────────────────────────────────
    st.markdown(
        "<div style='background:#E8F0FA;border:1px solid #C7D3E5;"
        "border-left:3px solid #185FA5;border-radius:6px;padding:10px 18px;"
        "margin-bottom:22px;font-size:0.84rem;color:#3A3A3A;line-height:1.55;"
        "max-width:880px'>"
        "<strong style='color:#185FA5;font-weight:600'>Binding authorities.</strong> "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Boutilier</em> [2017] SCC 64 — pattern analysis · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Antic</em> [2017] SCC 27 — bail/coercive plea · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>R v Le</em> [2019] SCC 34 — over-policing · "
        "<em style='font-family:Fraunces,serif;font-style:italic;color:#1A1A1A'>Ewert v Canada</em> [2018] SCC 30 — actuarial validity. "
        "Each conviction is calibrated against the live distortion-node posteriors before contributing to N2/N18."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Distortion node weights (preserved) ───────────────────────────────
    # Used to compute calibrated reliability for each conviction
    pN7  = st.session_state.posteriors.get(7,  0.15)   # bail-denial cascade
    pN6  = st.session_state.posteriors.get(6,  0.15)   # ineffective counsel
    pN5  = st.session_state.posteriors.get(5,  0.10)   # invalid risk tools / Ewert
    pN14 = st.session_state.posteriors.get(14, 0.15)   # over-policing
    pN15 = st.session_state.posteriors.get(15, 0.40)   # temporal distortion / age
    pN12 = st.session_state.posteriors.get(12, 0.15)   # Gladue misapplication

    # ── Authoritative correction references (preserved) ───────────────────
    CORR_REFS = {
        "bail":    ("Bail-denial cascade (N7)", "A32D2D", "R v Antic [2017] SCC 27; Tolppanen Report (2018)"),
        "counsel": ("Ineffective counsel (N6)", "185FA5", "R v GDB [2000] 1 SCR 520; Strickland doctrine"),
        "ewert":   ("Ewert tool invalidity (N5)", "185FA5", "Ewert v Canada [2018] SCC 30"),
        "police":  ("Over-policing / record inflation (N14)", "185FA5", "R v Le [2019] SCC 34"),
        "time":    ("Temporal attenuation (N15)", "185FA5", "R v Nur [2015] / Lloyd [2016] — age burnout"),
        "gladue":  ("Gladue misapplication (N12)", "3B6D11", "R v Morris 2021 ONCA 680 para 97; Boutilier [2017] SCC 64"),
    }

    # ── Compute global reliability floor from current distortion posteriors ───
    bail_factor    = float(np.clip(1.0 - 0.50*pN7 - 0.20*pN6, 0.30, 1.0))
    ewert_factor   = float(np.clip(1.0 - 0.40*pN5,             0.40, 1.0))
    police_factor  = float(np.clip(1.0 - 0.35*pN14,            0.50, 1.0))
    gladue_factor  = float(np.clip(1.0 - 0.30*pN12,            0.55, 1.0))

    # ── Offence seriousness tiers (Boutilier pattern analysis, preserved) ──
    SERIOUSNESS = {
        # Tier 1 — catastrophic (weight 1.00)
        "murder":1.00,"manslaughter":1.00,"attempted murder":1.00,
        "sexual assault causing bodily harm":1.00,"aggravated sexual assault":1.00,
        "kidnapping":1.00,"hostage":1.00,
        # Tier 2 — serious violent (weight 0.85)
        "aggravated assault":0.85,"robbery":0.85,"break and enter":0.70,
        "assault with weapon":0.75,"assault causing bodily harm":0.75,
        "sexual assault":0.75,"arson":0.80,"forcible confinement":0.80,
        "weapon":0.70,"discharge firearm":0.80,"extortion":0.75,
        # Tier 3 — significant (weight 0.55)
        "assault":0.55,"theft over":0.45,"fraud over":0.40,
        "dangerous operation":0.55,"impaired":0.35,"possession":0.30,
        "drug":0.35,"mischief":0.25,"uttering threats":0.45,
        # Tier 4 — minor (weight 0.20)
        "breach":0.20,"fail to comply":0.20,"bail":0.15,"summary":0.15,
    }

    def _get_seriousness(offence_str: str) -> float:
        """Return base seriousness weight for an offence string."""
        ol = offence_str.lower()
        # Check multi-word keys first (most specific)
        for key in sorted(SERIOUSNESS.keys(), key=len, reverse=True):
            if key in ol:
                return SERIOUSNESS[key]
        return 0.40  # default moderate

    def _detect_escalation(rec: list) -> dict:
        """
        Detect escalation, de-escalation, or stable pattern.
        Per Boutilier [2017] SCC 64: pattern of behaviour matters, not just
        individual offences. Escalation in seriousness is a significant factor.
        Returns dict with: pattern, signal, note.
        """
        if len(rec) < 2:
            return {"pattern":"insufficient","signal":0.0,"note":"Fewer than 2 convictions — no pattern detectable."}
        chronological = sorted(rec, key=lambda e: e["year"])
        seriousness_scores = [_get_seriousness(e["offence"]) * e["cal_weight"] for e in chronological]
        # Compare first third vs last third
        n = len(seriousness_scores)
        early = float(np.mean(seriousness_scores[:max(1,n//3)]))
        late  = float(np.mean(seriousness_scores[n - max(1,n//3):]))
        delta = late - early
        # Gap in years between last two convictions (de-escalation by time gap)
        year_gap = chronological[-1]["year"] - chronological[-2]["year"] if n>=2 else 0
        if delta > 0.15:
            pat = "escalating"
            signal = float(np.clip(0.10 + delta * 0.40, 0.05, 0.35))
            note = (f"Pattern shows escalation in offence seriousness "
                    f"(early avg {early:.2f} → recent avg {late:.2f}). "
                    f"Per Boutilier [2017] SCC 64: escalating pattern weighs in favour of DO designation.")
        elif delta < -0.10:
            pat = "de-escalating"
            signal = float(np.clip(-0.08 - abs(delta)*0.25, -0.25, 0.0))
            note = (f"Pattern shows de-escalation (early avg {early:.2f} → recent avg {late:.2f}). "
                    f"Reduced weight per Boutilier — de-escalation is relevant to treatability inquiry.")
        elif year_gap >= 8:
            pat = "desistance"
            signal = -0.12
            note = (f"Last two convictions separated by {year_gap} years — extended gap suggests desistance. "
                    f"Temporal attenuation applicable per age burnout (N15).")
        else:
            pat = "stable"
            signal = 0.0
            note = "Stable offence pattern — no significant escalation or de-escalation detected."
        return {"pattern":pat,"signal":signal,"note":note}

    def _cr_feed_nodes():
        """Derive N2, N18 and distortion node signals from the calibrated criminal record.
        Fixes: N2 signal now written to doc_adj (not cr_doc_adj) so run_inf() picks it up.
        Enhanced: seriousness-weighted N2, escalation signal into N18, gang context into N14.
        """
        rec = st.session_state.criminal_record
        if not rec:
            st.session_state.cr_doc_adj = {}
            # Remove criminal record contributions from doc_adj
            st.session_state.doc_adj = {
                k:v for k,v in st.session_state.doc_adj.items()
                if k not in [2, 7, 14, 12, 18]
            }
            return

        cr_adj = {k:v for k,v in st.session_state.doc_adj.items()
                  if k not in [2, 7, 14, 12, 18]}  # start fresh for CR-driven nodes

        # ── N2: Violent history — seriousness-weighted calibrated signal ──────
        violent_types = ["assault","violence","weapon","robbery","forcible","aggravated",
                         "murder","manslaughter","sexual","arson","discharge","extortion",
                         "kidnap","hostage","confinement"]
        violent = [e for e in rec if any(vt in e["offence"].lower() for vt in violent_types)]
        if violent:
            # Weighted severity: each entry contributes seriousness_weight * cal_weight
            severity_scores = [_get_seriousness(e["offence"]) * e["cal_weight"] for e in violent]
            # Boutilier: count matters but is bounded; severity of worst offences matters most
            max_severity  = float(max(severity_scores))
            mean_severity = float(np.mean(severity_scores))
            count_bonus   = float(np.clip(0.05 * (len(violent) - 1), 0.0, 0.20))
            n2_raw = float(np.clip(
                0.20 + 0.45 * max_severity + 0.25 * mean_severity + count_bonus,
                0.08, 0.90))
            # ── Jump principle (Ch 3 §3.5.3) — forward-contamination shift ──
            # Cumulative ceiling effect from prior convictions raises the
            # baseline against which subsequent severity is measured. Applied
            # here as an upward shift on N2 (Violent history) before
            # propagation to N20. The shift is conservative: 0.5× cumulative
            # ceiling (capped at 0.40), so the maximum N2 shift is +0.20.
            n2_jump_shift = _jump_record_n2_shift(rec)
            n2_raw_with_jump = float(np.clip(n2_raw + n2_jump_shift, 0.08, 0.95))
            st.session_state.cr_doc_adj["jump_shift"] = n2_jump_shift
            st.session_state.cr_doc_adj["n2_pre_jump"] = n2_raw
            cr_adj[2] = n2_raw_with_jump - st.session_state.posteriors.get(2, 0.08)
            st.session_state.cr_doc_adj[2] = n2_raw_with_jump

        # ── N18: Dynamic risk — escalation signal ─────────────────────────────
        esc = _detect_escalation(rec)
        st.session_state.cr_doc_adj["escalation"] = esc
        if abs(esc["signal"]) > 0.01:
            cr_adj[18] = float(np.clip(esc["signal"], -0.30, 0.35))

        # ── Aggravating factors → node boosts ────────────────────────────────
        gang_entries   = [e for e in rec if e.get("gang", False)]
        weapon_entries = [e for e in rec if e.get("weapon", False)]
        child_entries  = [e for e in rec if e.get("child_victim", False)]
        trust_entries  = [e for e in rec if e.get("position_of_trust", False)]
        if gang_entries:
            cr_adj[14] = cr_adj.get(14, 0) + float(np.clip(0.08 * len(gang_entries), 0.0, 0.25))
        if weapon_entries:
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.06 * len(weapon_entries), 0.0, 0.20))
        if child_entries:
            cr_adj[4]  = cr_adj.get(4, 0)  + float(np.clip(0.07 * len(child_entries), 0.0, 0.20))
        if trust_entries:
            cr_adj[3]  = cr_adj.get(3, 0)  + float(np.clip(0.05 * len(trust_entries), 0.0, 0.15))
        # New aggravating factors → node signals
        domestic_e = [e for e in rec if e.get("domestic_violence", False)]
        hate_e     = [e for e in rec if e.get("hate_crime", False)]
        terror_e   = [e for e in rec if e.get("terrorism", False)]
        vuln_e     = [e for e in rec if e.get("vulnerable_victim", False)]
        drug_e     = [e for e in rec if e.get("drug_trafficking", False)]
        if domestic_e:
            # Domestic violence → boosts violent history (N2) and dynamic risk (N18)
            cr_adj[2]  = cr_adj.get(2, 0)  + float(np.clip(0.05 * len(domestic_e), 0.0, 0.15))
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.04 * len(domestic_e), 0.0, 0.12))
        if hate_e:
            # Hate crime → boosts dynamic risk (N18)
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.06 * len(hate_e), 0.0, 0.18))
        if terror_e:
            # Terrorism → significant boost to N18 and N2
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.12 * len(terror_e), 0.0, 0.30))
            cr_adj[2]  = cr_adj.get(2, 0)  + float(np.clip(0.10 * len(terror_e), 0.0, 0.25))
        if vuln_e:
            # Vulnerable victim → boosts dynamic risk (N18)
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.05 * len(vuln_e), 0.0, 0.15))
        if drug_e:
            # Drug trafficking — Parranto [2021] — fentanyl at top of tariff
            fentanyl_e = [e for e in drug_e if "fentanyl" in e.get("drug_type","").lower()
                          or "carfentanil" in e.get("drug_type","").lower()]
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(
                0.06 * len(drug_e) + 0.06 * len(fentanyl_e), 0.0, 0.22))

        # ── Distortion aggregates → N7, N14, N12 ─────────────────────────────
        bail_agg   = float(np.mean([e["adj"]["bail"]   for e in rec]))
        police_agg = float(np.mean([e["adj"]["police"] for e in rec]))
        gladue_agg = float(np.mean([e["adj"]["gladue"] for e in rec]))
        if bail_agg > 0.1:
            cr_adj[7]  = cr_adj.get(7,  0) + bail_agg * 0.12
        if police_agg > 0.1:
            cr_adj[14] = cr_adj.get(14, 0) + police_agg * 0.10
        if gladue_agg > 0.1:
            cr_adj[12] = cr_adj.get(12, 0) + gladue_agg * 0.08

        st.session_state.doc_adj = {k: float(np.clip(v, -0.4, 0.4)) for k,v in cr_adj.items()}

    # ── UI: Add conviction (form-card with subsection structure) ──────────
    st.markdown(
        "<div style='font-family:Fraunces,Georgia,serif;font-size:1.05rem;"
        "font-weight:500;color:#1A1A1A;margin:14px 0 8px 0'>Add conviction</div>",
        unsafe_allow_html=True,
    )

    with st.expander("➕ Enter a new conviction",
                     expanded=len(st.session_state.criminal_record) == 0):

        # ─── Subsection: Identification ───────────────────────────────────
        st.markdown(
            "<div style='font-size:0.66rem;text-transform:uppercase;"
            "letter-spacing:0.14em;color:#707070;font-weight:600;"
            "margin:4px 0 12px 0'>Identification</div>",
            unsafe_allow_html=True,
        )
        ca1, ca2 = st.columns([2, 1])
        with ca1:
            cr_offence = st.text_input(
                "Offence description",
                placeholder="e.g. Aggravated assault s.268 CC", key="cr_off")
            cr_court = st.text_input(
                "Court",
                placeholder="e.g. Ontario Superior Court of Justice", key="cr_court")
        with ca2:
            cr_year = st.number_input(
                "Year of conviction",
                min_value=1950, max_value=2026, value=2015, step=1, key="cr_year")
            cr_sentence_type = st.selectbox(
                "Sentence type",
                ["Federal custody (2+ years)",
                 "Provincial custody (< 2 years)",
                 "Conditional sentence order (CSO)",
                 "Probation only",
                 "Fine only",
                 "Absolute / conditional discharge",
                 "Time served",
                 "Other / unknown"],
                key="cr_sent_type",
                help="Sentence type informs evidentiary weight — CSO/probation/discharge outcomes suggest original court assessed limited dangerousness per Boutilier [2017] SCC 64")
            cr_sentence_detail = st.text_input(
                "Sentence detail",
                placeholder="e.g. 18 months, 2 years probation", key="cr_sent_detail")
        cr_jurisdiction = st.selectbox(
            "Province / jurisdiction",
            ["ON","BC","AB","QC","SK","MB","NS","NB","NL","PE","YT","NT","NU","Federal"],
            key="cr_jur")

        # ─── Subsection: Sentence outcome ─────────────────────────────────
        st.markdown(
            "<div style='font-size:0.66rem;text-transform:uppercase;"
            "letter-spacing:0.14em;color:#707070;font-weight:600;"
            "margin:18px 0 12px 0;padding-top:14px;"
            "border-top:1px solid #EFEDE7'>Sentence outcome &amp; severity</div>",
            unsafe_allow_html=True,
        )
        cf1, cf2 = st.columns(2)
        with cf1:
            cr_seriousness = st.select_slider(
                "Offence seriousness tier",
                options=["Minor (0.15–0.25)",
                         "Moderate (0.35–0.50)",
                         "Significant (0.55–0.75)",
                         "Serious violent (0.80–0.85)",
                         "Catastrophic (1.00)"],
                value="Significant (0.55–0.75)", key="cr_seriousness",
                help="Boutilier [2017] SCC 64 — offence seriousness determines base weight in pattern analysis")
        with cf2:
            st.markdown(
                "<div style='font-size:0.86rem;font-weight:600;color:#3A3A3A;"
                "margin-bottom:4px'>Aggravating factors "
                "<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                "color:#9E9E9E;font-weight:500'>s.718.2 CC</span></div>",
                unsafe_allow_html=True,
            )
            cr_gang = st.checkbox(
                "Gang / organized crime context",
                value=False, key="cr_gang",
                help="s.718.2(a)(iv) CC; R v Lacasse [2015] SCC 64 — gang context aggravating. Consider Le [2019] SCC 34 — may also reflect over-policing")
            cr_weapon = st.checkbox(
                "Weapon / firearm used or present",
                value=False, key="cr_weapon",
                help="s.718.2(a)(i) CC; s.85, s.95 CC — statutory aggravating. Firearm offences carry significant weight in Boutilier pattern analysis")
            cr_child_victim = st.checkbox(
                "Child victim (under 18)",
                value=False, key="cr_child",
                help="s.718.2(a)(ii.1) CC — statutory aggravating. Significant for sexual offence profile (N4 / Static-99R)")
            cr_trust = st.checkbox(
                "Position of trust / authority",
                value=False, key="cr_trust",
                help="s.718.2(a)(iii) CC — statutory aggravating. Relevant to predatory behaviour under Boutilier and PCL-R (N3)")
            cr_domestic = st.checkbox(
                "Domestic / intimate partner violence",
                value=False, key="cr_domestic",
                help="s.718.2(a)(ii) CC — abuse of spouse or common-law partner is explicit statutory aggravating factor. Also relevant to Gladue analysis where colonialism intersects with family violence")
            cr_hate = st.checkbox(
                "Hate crime / bias motivation",
                value=False, key="cr_hate",
                help="s.718.2(a)(i) CC — evidence that offence motivated by bias, prejudice or hate based on race, national/ethnic origin, language, colour, religion, sex, age, disability, sexual orientation is statutory aggravating")
            cr_terrorism = st.checkbox(
                "Terrorism-related offence",
                value=False, key="cr_terrorism",
                help="s.718.2(a)(i) CC; Criminal Code Part II.1 — terrorism offences carry the highest seriousness weighting and directly engage the DO regime's public protection rationale under s.753")
            cr_vulnerable = st.checkbox(
                "Vulnerable victim (age, disability, circumstance)",
                value=False, key="cr_vulnerable",
                help="s.718.2(a)(i) CC — evidence that victim was vulnerable due to age, disability or other circumstances of vulnerability is statutory aggravating. Distinct from child victim — covers elderly, cognitively impaired, or situationally vulnerable adults")
            cr_drugs = st.checkbox(
                "Drug / substance trafficking (serious narcotics)",
                value=False, key="cr_drugs",
                help="s.718.2(a)(i) CC; Controlled Drugs and Substances Act — nature and quantity of substance trafficked is aggravating. Fentanyl, carfentanil, and methamphetamine carry greater weight than cannabis. R v Parranto [2021] SCC 46 — fentanyl trafficking attracts elevated tariff")
            cr_drugs_type = ""
            if cr_drugs:
                cr_drugs_type = st.selectbox(
                    "Substance type",
                    ["Fentanyl / carfentanil",
                     "Heroin / opioids",
                     "Cocaine / crack cocaine",
                     "Methamphetamine / MDMA",
                     "Cannabis (pre-legalization)",
                     "Other controlled substance"],
                    key="cr_drugs_type",
                    help="R v Parranto [2021] SCC 46 — fentanyl and carfentanil occupy the highest tier given lethal risk at microgram quantities")

        # ─── Subsection: Doctrinal reliability adjustments (tinted panel) ──
        st.markdown(
            "<div style='background:#E8F0FA;border:1px solid #C7D3E5;"
            "border-left:4px solid #185FA5;border-radius:8px;"
            "padding:18px 20px;margin-top:20px'>"
            "<div style='font-family:Fraunces,Georgia,serif;font-size:0.95rem;"
            "font-weight:500;color:#185FA5;margin-bottom:4px'>"
            "Doctrinal reliability adjustments for this conviction</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.78rem;color:#3A3A3A;line-height:1.55'>"
            "Each slider reflects the degree to which this specific conviction's "
            "evidentiary weight should be discounted based on doctrinal distortions "
            "applicable to this conviction. Sliders default to 0; raise affirmatively "
            "where the case file or transcript supports the relevant distortion "
            "for this conviction. Per-conviction values are independent of the "
            "network's case-wide posteriors."
            "</div></div>",
            unsafe_allow_html=True,
        )

        ca3, ca4 = st.columns(2)
        with ca3:
            adj_bail   = st.slider("Bail-denial / coercive plea reduction",  0.0, 1.0, 0.0,  0.05, key="cr_adj_bail",
                help="R v Antic [2017] SCC 27 — set affirmatively where this specific conviction was produced under bail-denial cascade conditions (extended pre-trial detention, coercive plea pressure)")
            adj_ewert  = st.slider("Ewert tool-invalidity reduction",         0.0, 1.0, 0.0,  0.05, key="cr_adj_ewert",
                help="Ewert v Canada [2018] SCC 30 — set affirmatively where this conviction's foundation involved actuarial evidence of contested cultural validity")
            adj_gladue = st.slider("Gladue misapplication reduction",         0.0, 1.0, 0.0, 0.05, key="cr_adj_gladue",
                help="R v Morris 2021 ONCA 680 — set affirmatively where Gladue/Ipeelee/Morris factors were ignored or misapplied at this conviction's sentencing")
        with ca4:
            adj_police = st.slider("Over-policing / record inflation reduction",0.0, 1.0, 0.0,0.05, key="cr_adj_police",
                help="R v Le [2019] SCC 34 — set affirmatively where this conviction reflects racialised over-policing rather than criminogenic conduct")
            adj_mm     = st.slider("Mandatory minimum distortion reduction",   0.0, 1.0, 0.0,             0.05, key="cr_adj_mm",
                help="R v Nur [2015] / Lloyd [2016] — set affirmatively where this conviction was influenced by a now-struck mandatory minimum")
            adj_time   = st.slider("Temporal attenuation (age / distance)",    0.0, 1.0, 0.0,0.05, key="cr_adj_time",
                help="R v Boutilier [2017] SCC 64 — set affirmatively where this conviction is sufficiently remote that age-burnout attenuation applies")

        # Sentence type modifier — CSO/probation/discharge suggests limited dangerousness
        _sent_modifier = {
            "Federal custody (2+ years)":         1.00,
            "Provincial custody (< 2 years)":     0.90,
            "Conditional sentence order (CSO)":   0.70,
            "Probation only":                     0.55,
            "Fine only":                          0.35,
            "Absolute / conditional discharge":   0.20,
            "Time served":                        0.80,
            "Other / unknown":                    0.85,
        }
        sent_mod = _sent_modifier.get(cr_sentence_type, 0.85)

        # Aggravating factor boost — statutory aggravating raises base seriousness
        agg_boost = 1.0
        if cr_weapon:      agg_boost = min(agg_boost + 0.08, 1.40)
        if cr_child_victim:agg_boost = min(agg_boost + 0.12, 1.40)
        if cr_trust:       agg_boost = min(agg_boost + 0.10, 1.40)
        if cr_gang:        agg_boost = min(agg_boost + 0.06, 1.40)
        if cr_domestic:    agg_boost = min(agg_boost + 0.07, 1.40)
        if cr_hate:        agg_boost = min(agg_boost + 0.09, 1.40)
        if cr_terrorism:   agg_boost = min(agg_boost + 0.20, 1.40)
        if cr_vulnerable:  agg_boost = min(agg_boost + 0.08, 1.40)
        if cr_drugs:
            _drug_boost = {"Fentanyl / carfentanil":0.15,"Heroin / opioids":0.10,
                           "Cocaine / crack cocaine":0.08,"Methamphetamine / MDMA":0.09,
                           "Cannabis (pre-legalization)":0.03,"Other controlled substance":0.06}
            agg_boost = min(agg_boost + _drug_boost.get(cr_drugs_type, 0.06), 1.40)

        # Compute calibrated weight for this entry
        raw_wt = 1.0
        cal_wt = float(np.clip(
            raw_wt * sent_mod * agg_boost *
            (1 - 0.55*adj_bail) * (1 - 0.40*adj_ewert) *
            (1 - 0.35*adj_police) * (1 - 0.30*adj_gladue) *
            (1 - 0.25*adj_mm) * (1 - 0.45*adj_time), 0.05, 1.15))
        pct_ret = cal_wt * 100

        col_ret = "#3B6D11" if pct_ret >= 70 else "#BA7517" if pct_ret >= 40 else "#A32D2D"
        st.markdown(
            f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
            f"border-radius:6px;padding:.7rem 1rem;margin-top:.8rem'>"
            f"<span style='font-size:.78rem;color:#707070'>Calibrated evidentiary weight: </span>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.4rem;"
            f"font-weight:600;color:{col_ret}'>{pct_ret:.0f}%</span>"
            f"<span style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.78rem;color:#707070;margin-left:.5rem'>"
            f"of nominal weight retained</span></div>",
            unsafe_allow_html=True)

        if st.button("Add to record", key="cr_add"):
            if cr_offence.strip():
                # Map seriousness tier to numeric weight
                _ser_map = {
                    "Minor (0.15–0.25)": 0.20,
                    "Moderate (0.35–0.50)": 0.42,
                    "Significant (0.55–0.75)": 0.65,
                    "Serious violent (0.80–0.85)": 0.82,
                    "Catastrophic (1.00)": 1.00,
                }
                entry = {
                    "offence":           cr_offence.strip(),
                    "court":             cr_court.strip(),
                    "year":              int(cr_year),
                    "sentence":          f"{cr_sentence_type} — {cr_sentence_detail}".strip(" —") if cr_sentence_detail else cr_sentence_type,
                    "sentence_type":     cr_sentence_type,
                    "sentence_detail":   cr_sentence_detail.strip(),
                    "sent_modifier":     sent_mod,
                    "jurisdiction":      cr_jurisdiction,
                    "seriousness":       _ser_map.get(cr_seriousness, 0.65),
                    "seriousness_label": cr_seriousness,
                    "agg_boost":         agg_boost,
                    "gang":              cr_gang,
                    "weapon":            cr_weapon,
                    "child_victim":      cr_child_victim,
                    "position_of_trust": cr_trust,
                    "domestic_violence": cr_domestic,
                    "hate_crime":        cr_hate,
                    "terrorism":         cr_terrorism,
                    "vulnerable_victim": cr_vulnerable,
                    "drug_trafficking":  cr_drugs,
                    "drug_type":         cr_drugs_type if cr_drugs else "",
                    "adj": {
                        "bail":   adj_bail,
                        "ewert":  adj_ewert,
                        "gladue": adj_gladue,
                        "police": adj_police,
                        "mm":     adj_mm,
                        "time":   adj_time,
                    },
                    "raw_weight":  1.0,
                    "cal_weight":  cal_wt,
                }
                st.session_state.criminal_record.append(entry)
                # Sort chronologically (earliest first) — stable, by year only
                # (so insertions of multiple convictions in the same year retain
                # the order in which they were added).
                st.session_state.criminal_record.sort(
                    key=lambda e: int(e.get("year", 0))
                )
                # Feed calibrated N2 and distortion corrections back into the network
                _cr_feed_nodes()
                st.rerun()
            else:
                st.warning("Please enter an offence description.")

    # ── Record table ──────────────────────────────────────────────────────────
    rec = st.session_state.criminal_record
    if rec:
        st.markdown(
            "<div style='border-top:1px solid #EFEDE7;margin:24px 0 18px 0'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.05rem;"
            f"font-weight:500;color:#1A1A1A;margin-bottom:14px'>"
            f"Calibrated criminal record "
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            f"color:#707070;font-weight:500;margin-left:6px'>"
            f"{len(rec)} conviction{'s' if len(rec)!=1 else ''}</span></div>",
            unsafe_allow_html=True,
        )

        # Summary stats
        all_cal = [e["cal_weight"] for e in rec]
        mean_cal = float(np.mean(all_cal))
        mc_col = "#3B6D11" if mean_cal >= 0.7 else "#BA7517" if mean_cal >= 0.4 else "#A32D2D"

        esc_data = st.session_state.cr_doc_adj.get("escalation", {})
        esc_pat  = esc_data.get("pattern","—")
        esc_cols = {"escalating":"#A32D2D","de-escalating":"#3B6D11",
                    "desistance":"#3B6D11","stable":"#888888","insufficient":"#BBBBBB"}
        esc_col  = esc_cols.get(esc_pat,"#888")

        sc1,sc2,sc3,sc4 = st.columns(4)
        sc1.metric("Total convictions", len(rec))
        sc2.metric("Mean calibrated weight", f"{mean_cal*100:.0f}%")
        sc3.metric("Record reliability tier",
            "High" if mean_cal>=0.7 else "Moderate" if mean_cal>=0.4 else "Low")
        sc4.markdown(
            f"<div style='font-size:.72rem;color:#707070;margin-bottom:2px;"
            f"text-transform:uppercase;letter-spacing:0.06em;font-weight:600'>"
            f"Pattern (Boutilier)</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.1rem;"
            f"font-weight:600;color:{esc_col}'>{esc_pat.title()}</div>",
            unsafe_allow_html=True)

        # ── N7 Reliability Modifier panel (Chapter 5 §5.1.7) ──────────────
        # Aggregate effect of N7 (bail-denial cascade) re-weighting on the
        # criminal record. Per JP's specification, this surfaces the
        # constructive-proof claim that distortions re-weight (not remove)
        # convictions — Chapter 5 §5.1.7 verbatim: "This node never removes
        # convictions. It qualifies how they may be used."
        _n7_post_now = float(st.session_state.posteriors.get(7, 0.15))
        _n7_nom, _n7_adj, _n7_grades = _n7_aggregate_record_weight(rec, _n7_post_now)
        if _n7_nom is not None:
            _n7_delta_pct = (_n7_adj - _n7_nom) * 100
            _n7_delta_sign = "−" if _n7_delta_pct < 0 else ("+" if _n7_delta_pct > 0 else "")
            # Count convictions per grade
            _n7_count = {"Unmodified": 0, "Discounted": 0, "Heavily Discounted": 0}
            for g, _m in _n7_grades:
                _n7_count[g] = _n7_count.get(g, 0) + 1
            # Accent colour: how much is the record re-weighted overall
            _delta_abs = abs(_n7_delta_pct)
            if _delta_abs < 5:
                _n7_accent = "#3B6D11"; _n7_bg = "#F4F8EE"; _n7_border = "#C5D7AC"
            elif _delta_abs < 15:
                _n7_accent = "#BA7517"; _n7_bg = "#FAEEDA"; _n7_border = "#E5CC95"
            else:
                _n7_accent = "#A32D2D"; _n7_bg = "#FCEBEB"; _n7_border = "#E5B5B5"

            st.markdown(
                f"<div style='background:{_n7_bg};border:1px solid {_n7_border};"
                f"border-left:3px solid {_n7_accent};border-radius:8px;"
                f"padding:14px 18px;margin:14px 0 8px 0'>"
                # Header row
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:10px'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"font-weight:600;padding:2px 8px;border-radius:4px;color:white;"
                f"background:{_n7_accent}'>N7</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.0rem;"
                f"font-weight:500;color:#1A1A1A'>"
                f"Reliability Modifier · Bail-denial cascade</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#707070;margin-left:auto'>"
                f"<em>R v Antic</em> [2017] SCC 27 · Ch 5 §5.1.7</div>"
                f"</div>"
                # Body: nominal vs adjusted comparison
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;"
                f"margin-top:6px'>"
                # Nominal
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:#707070;font-weight:600;margin-bottom:3px'>Nominal record</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                f"font-weight:600;color:#3A3A3A'>{_n7_nom*100:.1f}%</div>"
                f"<div style='font-size:0.72rem;color:#9E9E9E;font-family:Fraunces,serif;"
                f"font-style:italic'>mean calibrated weight</div>"
                f"</div>"
                # Adjusted
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:{_n7_accent};font-weight:600;margin-bottom:3px'>"
                f"N7-adjusted record</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                f"font-weight:600;color:{_n7_accent}'>{_n7_adj*100:.1f}%</div>"
                f"<div style='font-size:0.72rem;color:{_n7_accent};font-family:Fraunces,serif;"
                f"font-style:italic'>{_n7_delta_sign}{abs(_n7_delta_pct):.1f}pp re-weighting</div>"
                f"</div>"
                # Grade distribution
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:#707070;font-weight:600;margin-bottom:3px'>Grade distribution</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                f"color:#3A3A3A;line-height:1.5'>"
                f"<div>Unmodified · {_n7_count['Unmodified']}</div>"
                f"<div>Discounted · {_n7_count['Discounted']}</div>"
                f"<div>Heavily Discounted · {_n7_count['Heavily Discounted']}</div>"
                f"</div>"
                f"</div>"
                f"</div>"
                # Doctrinal caption
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.80rem;color:#707070;margin-top:12px;line-height:1.55;"
                f"padding-top:10px;border-top:1px solid {_n7_border}'>"
                f"Each conviction is graded against §5.1.7's tri-state ordinal "
                f"(Unmodified / Discounted / Heavily Discounted) on its own facts "
                f"— the conviction's own bail-denial signal, with cascade propagation "
                f"from earlier tainted convictions where applicable per §RM.5. "
                f"This node never removes convictions; it qualifies how they may be "
                f"used. Multipliers — 1.00 / 0.60 / 0.30 — are conservative "
                f"operationalisations of the §5.1.7 ordinal grades."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Methodology disclosure expander
            with st.expander("Methodology — N7 reliability modifier", expanded=False):
                st.markdown("""
**Mechanism (Chapter 5 §5.1.7).** The bail-denial cascade node (N7) tracks the procedural conditions under which prior convictions were produced. Where bail was denied and ineffective representation, absent social context evidence, or marginalisation cluster, the conviction's evidentiary reliability is qualified — not removed.

**Per-conviction grading.** Each conviction is graded on its own facts. The architecture reads the conviction's own `adj.bail` value (set when the conviction was added) and applies the threshold logic below. Per §5.1.7, the unit of analysis is the specific guilty plea produced under coercive procedural conditions, not the offender's record as a whole — convictions produced under fair conditions are not discounted on the basis of cascade conditions affecting other convictions on the record.

**Cascade propagation (§RM.5).** Where Conviction A on the record has been graded *Discounted* or *Heavily Discounted*, and a subsequent Conviction B has its own affirmative bail-denial signal, the architecture recognises that the bail conditions affecting Conviction B may themselves have been conditioned by Conviction A's presence on the record (per *R v Antic* on prior records and bail). Conviction B's bail-denial signal is multiplied by a propagation factor before threshold logic — 1.15 for upstream *Discounted*, 1.30 for upstream *Heavily Discounted*. Where multiple upstream tainted convictions exist, only the strongest factor is applied. Propagation requires the downstream conviction to have its own affirmative signal — bail granted on Conviction B breaks the chain.

**Thresholds.**
- Per-conviction signal (after any propagation) < 0.30 → **Unmodified** (multiplier 1.00)
- 0.30 ≤ signal ≤ 0.65 → **Discounted** (multiplier 0.60)
- Signal > 0.65 → **Heavily Discounted** (multiplier 0.30)

**Aggregate.** The N7-adjusted record weight is the mean across convictions of (cal_weight × N7_multiplier). The nominal record weight is the mean of cal_weight alone. The difference is the N7 re-weighting effect on the record as a whole.

**Relation to the jump principle.** This grading also feeds the jump principle's own_ceiling computation: a conviction graded *Heavily Discounted* contributes to forward contamination at 30% of nominal anchoring weight (§RM.5.5), reflecting the architecture's prior judgment that the conviction's severity reflects cascade contamination rather than legitimate sentencing assessment.

**Doctrinal source.** *R v Antic* [2017] SCC 27 (bail jurisprudence); Tolppanen Report (2018) on bail-denial cascade dynamics; Chapter 5 §5.1.7 (Wrongful Conviction Guilty Plea cascade modelling); Chapter 3 §3.4.5 (inferential inertia) and §3.5.3 (jump principle) for the doctrinal substrate of cascade propagation; Appendix RM §RM.5 for the operational specification.

**Scope of this implementation.** This is the Node 7 distortion only. The full set of cascade distortions (N6 IAC, N14 over-policing, N15 mandatory-minimum-era anchoring, N17 collider bias, N18 dynamic risk integration) extends the same architectural pattern in subsequent implementation work.
                """)

            # ── Jump Principle panel (Ch 3 §3.5.3) ──────────────────────────
            # Cumulative forward-contamination effect from prior convictions
            # raising the baseline for subsequent severity assessment.
            _jump_total_shift = st.session_state.cr_doc_adj.get("jump_shift", 0.0)
            _n2_pre  = st.session_state.cr_doc_adj.get("n2_pre_jump", None)
            _n2_post = st.session_state.cr_doc_adj.get(2, None)
            # Only render if we have a violent N2 calibration
            if _n2_pre is not None and _n2_post is not None:
                _jump_pp = _jump_total_shift * 100
                # Accent colour mirrors N7 panel scheme
                if _jump_pp < 5:
                    _j_accent = "#3B6D11"; _j_bg = "#F4F8EE"; _j_border = "#C5D7AC"
                elif _jump_pp < 15:
                    _j_accent = "#BA7517"; _j_bg = "#FAEEDA"; _j_border = "#E5CC95"
                else:
                    _j_accent = "#A32D2D"; _j_bg = "#FCEBEB"; _j_border = "#E5B5B5"

                # Compute total cumulative ceiling at end-of-record for caption
                _full_chain = _jump_cumulative_chain(rec)
                _own_last, _inh_last = _full_chain[-1][0], _full_chain[-1][1]
                _cumulative_at_end = min(_inh_last + _own_last, 0.40)

                st.markdown(
                    f"<div style='background:{_j_bg};border:1px solid {_j_border};"
                    f"border-left:3px solid {_j_accent};border-radius:8px;"
                    f"padding:14px 18px;margin:14px 0 8px 0'>"
                    # Header row
                    f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:10px'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;padding:2px 8px;border-radius:4px;color:white;"
                    f"background:{_j_accent}'>JUMP</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.0rem;"
                    f"font-weight:500;color:#1A1A1A'>"
                    f"Forward Contamination · Jump principle</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.78rem;color:#707070;margin-left:auto'>"
                    f"Ch 3 §3.5.3 · recursive severity</div>"
                    f"</div>"
                    # Body: three columns
                    f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;"
                    f"margin-top:6px'>"
                    # N2 pre-jump
                    f"<div>"
                    f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                    f"color:#707070;font-weight:600;margin-bottom:3px'>N2 · pre-jump</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                    f"font-weight:600;color:#3A3A3A'>{_n2_pre*100:.1f}%</div>"
                    f"<div style='font-size:0.72rem;color:#9E9E9E;font-family:Fraunces,serif;"
                    f"font-style:italic'>baseline violent history</div>"
                    f"</div>"
                    # N2 post-jump
                    f"<div>"
                    f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                    f"color:{_j_accent};font-weight:600;margin-bottom:3px'>N2 · post-jump</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                    f"font-weight:600;color:{_j_accent}'>{_n2_post*100:.1f}%</div>"
                    f"<div style='font-size:0.72rem;color:{_j_accent};font-family:Fraunces,serif;"
                    f"font-style:italic'>+{_jump_pp:.1f}pp from inherited ceiling</div>"
                    f"</div>"
                    # Cumulative ceiling at end of record
                    f"<div>"
                    f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                    f"color:#707070;font-weight:600;margin-bottom:3px'>Cumulative ceiling</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                    f"font-weight:600;color:#3A3A3A'>{_cumulative_at_end*100:.1f}pp</div>"
                    f"<div style='font-size:0.72rem;color:#9E9E9E;font-family:Fraunces,serif;"
                    f"font-style:italic'>at end-of-record (cap 40pp)</div>"
                    f"</div>"
                    f"</div>"
                    # Doctrinal caption
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.80rem;color:#707070;margin-top:12px;line-height:1.55;"
                    f"padding-top:10px;border-top:1px solid {_j_border}'>"
                    f"Inflated past sentences function as anchors for future escalation, "
                    f"producing recursive severity over time. Each conviction's own ceiling "
                    f"contribution is computed from sentence type, era (§3.5.3 phases: "
                    f"1995–2005 baseline, 2006–2015 mandatory-minimum revival, 2016–2019 "
                    f"partial restoration, 2020+ Bill C-5 restoration), and Gladue-compliance "
                    f"per §3.5.4. Cumulative shift on N2 is half the cumulative ceiling, "
                    f"capped at 20pp."
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Jump methodology expander
                with st.expander("Methodology — Jump principle (Ch 3 §3.5.3)", expanded=False):
                    st.markdown("""
**Mechanism (Chapter 3 §3.5.3).** *"Prior sentences, once imposed, function as baseline reference points for subsequent legal decisions regardless of whether their original severity reflected contemporaneous doctrine, proportionality principles, or systemic context. Inflated past sentences thereby become anchors for future escalation, producing recursive severity over time."*

**Per-conviction ceiling.** For each conviction, an *own ceiling effect* is computed from four factors:

1. **Sentence inflation factor** — keyed on sentence type. Federal custody (2+ years): 0.20. Provincial custody (< 2 years): 0.10. CSO: 0.03. Probation: 0.01. Fine/discharge: 0.0. Time served: 0.05.

2. **Era multiplier** — keyed on conviction year per §3.5.3 phase analysis. 1995–2005 (post C-41 baseline): ×1.0. 2006–2015 (mandatory-minimum revival / SSCA peak): ×1.5. 2016–2019 (partial restoration after SCC strikes): ×1.2. 2020+ (Bill C-5 restoration): ×1.0.

3. **Gladue-compliance multiplier** — when this conviction's `adj.gladue` exceeds 0.30 (Gladue not substantively applied at original sentencing), ×1.4 per §3.5.4. Otherwise ×1.0.

4. **N7 reliability weight (§RM.5.5)** — the conviction's own_ceiling is weighted by its N7 reliability grade. *Unmodified* contributes at full weight (×1.00); *Discounted* at ×0.60; *Heavily Discounted* at ×0.30. This nesting reflects the doctrinal commitment that a conviction whose severity reflects cascade contamination should not anchor subsequent severity at full strength.

`own_ceiling = sentence_inflation × era × gladue_compliance × N7_weight`

**Cumulative inheritance.** Convictions are processed in chronological order. Each conviction's *inherited ceiling* is the sum of all prior convictions' weighted own_ceilings, capped at 0.40 (40 percentage points). The first (earliest) conviction has zero inherited ceiling.

**N2 shift.** The cumulative ceiling at end-of-record (most recent inherited + own) is multiplied by 0.5 and added to N2's calibrated input. Maximum N2 upward shift is 0.20 (20 percentage points), reflecting the conservative position that anchoring contributes to but does not dominate the violent-history posterior.

**Doctrinal source.** Chapter 3 §3.5.3 (jump principle); §3.5.4 (temporal-Gladue interaction); §3.4.5 (cumulative inferential inertia); Appendix RM §RM.5 (cascade-propagation operationalisation, including the nested treatment between N7 reliability grading and jump-principle weighting).

**Relation to N7 cascade propagation.** The jump principle and the §5.1.7 cascade are nested rather than parallel. Both describe forward effects of earlier convictions on the conditions under which subsequent convictions are processed; they differ in their object. Cascade propagation modifies the bail-denial probability for subsequent convictions; the jump principle modifies the sentence-inflation baseline against which subsequent severity is measured. The N7 weight factor above prevents double-counting of upstream taint by reducing the anchoring contribution of convictions whose severity has already been recognised by the architecture as cascade-contaminated.

**Scope of this implementation.** Forward contamination via numerical anchoring (the jump principle), the N7 reliability discount, and the cascade propagation between them are all operationalised here. Audit transparency — each step's contribution visible to the reviewing court — completes the architectural answer to §3.4.5.
                    """)

        if esc_data.get("note"):
            esc_icon = "⚠️" if esc_pat=="escalating" else "✅" if esc_pat in ("de-escalating","desistance") else "ℹ️"
            st.markdown(
                f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
                f"border-left:3px solid {esc_col};border-radius:6px;"
                f"padding:.6rem 1rem;font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.84rem;color:#3A3A3A;margin:.6rem 0;line-height:1.55'>"
                f"{esc_icon} <strong style='font-style:normal;color:#1A1A1A'>"
                f"Pattern analysis:</strong> {esc_data['note']}</div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # Pre-compute the chronological ceiling-effect chain (Ch 3 §3.5.3)
        # AND the chronological N7 grading (Ch 5 §5.1.7 + §RM.5.4 propagation).
        # Index `i` matches sorted chronological order (earliest first).
        _jump_chain = _jump_cumulative_chain(rec)
        _n7_grading = _n7_grades_chronological(rec)  # list of (grade, mult, prop)

        # Per-conviction cards — each with its own document attachment
        for i, e in enumerate(rec):
            cal_pct = e["cal_weight"] * 100
            raw_pct = e["raw_weight"] * 100
            col_c = "#3B6D11" if cal_pct >= 70 else "#BA7517" if cal_pct >= 40 else "#A32D2D"
            adj = e["adj"]
            # ── N7 reliability modifier for this conviction (Ch 5 §5.1.7 + §RM.5.4) ──
            # Use chronological grading (accounts for cascade propagation
            # from earlier tainted convictions).
            _n7_grade_card, _n7_mult_card, _n7_prop_card = _n7_grading[i]
            _n7_eff_pct    = e["cal_weight"] * _n7_mult_card * 100
            # Grade-specific colour
            _grade_col = {
                "Unmodified":         "#3B6D11",
                "Discounted":         "#BA7517",
                "Heavily Discounted": "#A32D2D",
            }[_n7_grade_card]
            # ── Jump-principle ceiling effect (Ch 3 §3.5.3 + §RM.5.5) ──────
            # Chain now carries (own, inherited, grade, weight). The own_ceiling
            # already includes the JUMP_WEIGHT_BY_GRADE factor.
            _jump_own, _jump_inherited = _jump_chain[i][0], _jump_chain[i][1]
            # Colour scale for own ceiling effect
            if _jump_own < 0.10:   _jump_col = "#3B6D11"
            elif _jump_own < 0.25: _jump_col = "#BA7517"
            else:                  _jump_col = "#A32D2D"

            # Build distortion flags
            flags = []
            if adj["bail"]   > 0.15: flags.append(f"⛓ Bail-denial {adj['bail']*100:.0f}%")
            if adj["ewert"]  > 0.10: flags.append(f"📊 Ewert {adj['ewert']*100:.0f}%")
            if adj["police"] > 0.10: flags.append(f"🔵 Over-policing {adj['police']*100:.0f}%")
            if adj["gladue"] > 0.10: flags.append(f"🦅 Gladue {adj['gladue']*100:.0f}%")
            if adj["mm"]     > 0.05: flags.append(f"⚖️ Mand min {adj['mm']*100:.0f}%")
            if adj["time"]   > 0.10: flags.append(f"⏳ Temporal {adj['time']*100:.0f}%")
            # Aggravating factors
            if e.get("weapon"):            flags.append("🔫 Weapon/firearm")
            if e.get("child_victim"):      flags.append("👶 Child victim")
            if e.get("position_of_trust"): flags.append("🏛 Position of trust")
            if e.get("gang"):              flags.append("🔴 Gang context")
            if e.get("domestic_violence"): flags.append("🏠 Domestic violence")
            if e.get("hate_crime"):        flags.append("⚠️ Hate crime")
            if e.get("terrorism"):         flags.append("🚨 Terrorism")
            if e.get("vulnerable_victim"): flags.append("🧓 Vulnerable victim")
            if e.get("drug_trafficking"):
                dt = e.get("drug_type","")
                flags.append(f"💊 Drug trafficking{' — ' + dt if dt else ''}")
            # Sentence type attenuation signal
            sent_mod_e = e.get("sent_modifier", 1.0)
            if sent_mod_e <= 0.55: flags.append(f"📋 {e.get('sentence_type','Sentence')} (↓{(1-sent_mod_e)*100:.0f}% weight)")

            # Doc-linked indicator
            has_doc  = bool(e.get("doc_analysis"))
            has_recs = bool(e.get("doc_recs"))
            doc_badge = ""
            if has_doc:
                doc_badge = "<span style='background:#E8F5E9;color:#2E7D32;border-radius:4px;padding:2px 8px;font-size:.70rem;margin-left:6px'>📎 Document linked</span>"

            flag_html = "  ".join([
                f"<span style='background:#F7F5F2;border-radius:4px;padding:2px 8px;font-size:.72rem;color:#3A3A3A'>{f}</span>"
                for f in flags
            ]) if flags else "<span style='color:#9E9E9E;font-size:.72rem;font-family:Fraunces,serif;font-style:italic'>No distortion flags</span>"

            # Pre-build badge strings to avoid nested quote conflicts in f-string
            ser_label   = e.get("seriousness_label", "—")
            ser_badge   = f"<span style='background:#F7F5F2;border-radius:4px;padding:1px 7px;font-size:.68rem;color:#3A3A3A;font-family:JetBrains Mono,monospace'>⚖️ {ser_label}</span>"
            gang_badge  = (
                "<span style='background:#FDECEA;border-radius:4px;padding:1px 7px;"
                "font-size:.68rem;color:#A32D2D;margin-left:4px'>🔴 Gang context</span>"
                if e.get("gang") else ""
            )
            offence_str = e["offence"]
            court_str   = e.get("court", "")
            jur_str     = e.get("jurisdiction", "")
            sent_str    = e.get("sentence", "") or "Sentence not entered"
            yr_str      = str(e["year"])

            st.markdown(
                f"<div style='border:1px solid #E0DDD6;border-left:4px solid {col_c};"
                f"border-radius:8px;padding:.85rem 1.1rem;margin-bottom:.4rem;"
                f"background:#FFFFFF'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"font-size:1.0rem;color:#1A1A1A'>{offence_str}{doc_badge}</div>"
                f"<div style='font-size:.78rem;color:#707070;margin-top:3px'>"
                f"<span style='font-family:JetBrains Mono,monospace;font-weight:600'>{yr_str}</span>"
                f" · {court_str} · {jur_str} · {sent_str}</div>"
                f"<div style='margin-top:6px'>{ser_badge}{gang_badge}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:140px'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.35rem;"
                f"font-weight:600;color:{col_c}'>{cal_pct:.0f}%</div>"
                f"<div style='font-size:.7rem;color:#9E9E9E;font-family:Fraunces,serif;"
                f"font-style:italic'>calibrated weight</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:.7rem;"
                f"color:#C7C2B8;text-decoration:line-through;margin-bottom:6px'>{raw_pct:.0f}% raw</div>"
                # N7 reliability grade (Chapter 5 §5.1.7)
                f"<div style='border-top:1px solid #EFEDE7;padding-top:6px;margin-top:4px'>"
                f"<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;"
                f"color:#9E9E9E;font-weight:600;margin-bottom:2px'>N7 · Antic</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.84rem;"
                f"font-weight:500;color:{_grade_col};line-height:1.2'>{_n7_grade_card}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:{_grade_col};margin-top:1px'>×{_n7_mult_card:.2f} → {_n7_eff_pct:.0f}%</div>"
                f"</div>"
                # Jump-principle ceiling effect (Ch 3 §3.5.3)
                f"<div style='border-top:1px solid #EFEDE7;padding-top:6px;margin-top:6px'>"
                f"<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;"
                f"color:#9E9E9E;font-weight:600;margin-bottom:2px'>Jump · §3.5.3</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:{_jump_col};line-height:1.3'>own +{_jump_own*100:.1f}pp</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                f"color:#707070;line-height:1.3'>inherited +{_jump_inherited*100:.1f}pp</div>"
                f"</div>"
                f"</div></div>"
                f"<div style='margin-top:.55rem'>{flag_html}</div>"
                f"</div>",
                unsafe_allow_html=True)

            # ── Per-conviction action row ──────────────────────────────────────
            cb1, cb2, cb3 = st.columns([4, 2, 1])
            with cb1:
                # Document attachment expander for this conviction
                with st.expander(f"📎 {'Change document' if has_doc else 'Attach document'} for this conviction", expanded=False):
                    st.caption(f"Attach a trial transcript, bail hearing record, SCE report, or sentencing decision specifically related to the **{e['offence']} ({e['year']})** conviction.")
                    conv_up = st.file_uploader(
                        "Upload document",
                        type=["txt","pdf","docx"],
                        key=f"cr_up_{i}",
                        label_visibility="collapsed"
                    )
                    prov_conv = st.selectbox(
                        "AI provider",
                        ["Claude (Anthropic) ★","GPT-4o (OpenAI)","Gemini (Google)"],
                        key=f"cr_prov_{i}",
                        label_visibility="collapsed"
                    )
                    if st.button(f"Analyse for this conviction", key=f"cr_run_{i}"):
                        if conv_up:
                            with st.spinner(f"Analysing document for {e['offence']} ({e['year']})..."):
                                try:
                                    from document_analyzer import analyze_document, extract_text_from_file
                                    raw_text = extract_text_from_file(conv_up)

                                    conv_prompt = (
                                        f"You are a Canadian criminal law expert advising on a Dangerous Offender "
                                        f"sentencing proceeding under PARVIS. You are analysing a document specifically "
                                        f"related to ONE conviction in the accused's criminal record:\n\n"
                                        f"CONVICTION: {e['offence']}\n"
                                        f"YEAR: {e['year']}\n"
                                        f"COURT: {e['court']}\n"
                                        f"JURISDICTION: {e['jurisdiction']}\n"
                                        f"SENTENCE: {e.get('sentence','unknown')}\n"
                                        f"CURRENT CALIBRATED WEIGHT: {e['cal_weight']*100:.0f}% of nominal\n\n"
                                        f"Review the uploaded document and provide specific findings on:\n"
                                        f"1. BAIL-DENIAL / COERCIVE PLEA (R v Antic [2017] SCC 27): Was this conviction "
                                        f"preceded by extended pre-trial detention? Any evidence of plea pressure? "
                                        f"Quote the relevant passage if present.\n"
                                        f"2. INEFFECTIVE COUNSEL (R v GDB [2000] 1 SCR 520): Any indication of "
                                        f"inadequate legal representation? Missed Gladue submissions? Failure to "
                                        f"challenge tool evidence?\n"
                                        f"3. GLADUE MISAPPLICATION (Morris 2021 ONCA 680 para 97): Was a Gladue report "
                                        f"filed? Was it engaged substantively or cited and ignored? Connection/causation "
                                        f"error present?\n"
                                        f"4. OVER-POLICING / RECORD INFLATION (R v Le [2019] SCC 34): Any evidence "
                                        f"this charge arose from racialised enforcement patterns, carding, or "
                                        f"neighbourhood-based targeting?\n"
                                        f"5. MANDATORY MINIMUM DISTORTION (R v Nur [2015] / Lloyd [2016]): Was a "
                                        f"mandatory minimum in force at the time? Has it since been struck?\n"
                                        f"6. TEMPORAL ATTENUATION: How old is this conviction? What weight should "
                                        f"it carry given the offender's current age and any evidence of behavioural change?\n\n"
                                        f"End with a RECOMMENDATION block in this exact format:\n"
                                        f"RECOMMENDATION:\n"
                                        f"  bail_adj: [0.0-1.0]\n"
                                        f"  counsel_adj: [0.0-1.0]\n"
                                        f"  gladue_adj: [0.0-1.0]\n"
                                        f"  police_adj: [0.0-1.0]\n"
                                        f"  mm_adj: [0.0-1.0]\n"
                                        f"  time_adj: [0.0-1.0]\n"
                                        f"  overall: [KEEP / REDUCE / SIGNIFICANT REDUCTION]\n"
                                        f"  rationale: [one sentence]"
                                    )
                                    prov_key = prov_conv.split()[0].lower()
                                    result = analyze_document(
                                        file_content=raw_text,
                                        provider=prov_key,
                                        custom_prompt=conv_prompt,
                                    )
                                    # Store analysis on the entry itself
                                    st.session_state.criminal_record[i]["doc_analysis"] = result
                                    st.session_state.criminal_record[i]["doc_name"] = conv_up.name

                                    # Parse RECOMMENDATION block for suggested adjustments
                                    import re
                                    recs = {}
                                    for field in ["bail_adj","counsel_adj","gladue_adj","police_adj","mm_adj","time_adj"]:
                                        m = re.search(rf"{field}:\s*([0-9.]+)", result)
                                        if m:
                                            recs[field] = float(np.clip(float(m.group(1)), 0.0, 1.0))
                                    overall_m = re.search(r"overall:\s*(KEEP|REDUCE|SIGNIFICANT REDUCTION)", result, re.IGNORECASE)
                                    recs["overall"] = overall_m.group(1).upper() if overall_m else "REVIEW"
                                    st.session_state.criminal_record[i]["doc_recs"] = recs
                                    st.rerun()
                                except Exception as ex:
                                    st.error(f"Analysis error: {ex}")
                        else:
                            st.warning("Please upload a document first.")

                    # Show analysis result if present
                    if has_doc:
                        doc_analysis = e.get("doc_analysis","")
                        doc_recs     = e.get("doc_recs", {})
                        doc_name     = e.get("doc_name","document")

                        if doc_recs:
                            overall = doc_recs.get("overall","REVIEW")
                            ov_col = {"KEEP":"#3B6D11","REDUCE":"#BA7517","SIGNIFICANT REDUCTION":"#A32D2D"}.get(overall,"#555")
                            st.markdown(
                                f"<div style='background:#FBFAF7;border:1px solid #E0DDD6;"
                                f"border-radius:6px;padding:.6rem .9rem;margin-top:.4rem'>"
                                f"<span style='font-size:.78rem;color:#707070'>"
                                f"Document: <strong style='color:#1A1A1A'>{doc_name}</strong> · "
                                f"Recommendation: </span>"
                                f"<span style='font-family:JetBrains Mono,monospace;"
                                f"font-weight:600;color:{ov_col}'>{overall}</span></div>",
                                unsafe_allow_html=True)

                            # Show suggested adjustments as read-only comparison
                            st.markdown(
                                "<div style='font-size:.66rem;text-transform:uppercase;"
                                "letter-spacing:0.14em;color:#707070;font-weight:600;"
                                "margin-top:.5rem;margin-bottom:.4rem'>"
                                "Suggested adjustments from document analysis</div>",
                                unsafe_allow_html=True)
                            adj_cols = st.columns(6)
                            adj_labels = ["bail","counsel","gladue","police","mm","time"]
                            adj_names  = ["Bail","Counsel","Gladue","Police","MM","Time"]
                            for ci,(field,lbl) in enumerate(zip(adj_labels, adj_names)):
                                key_name = f"{field}_adj"
                                cur_val = e["adj"].get("ewert" if field=="counsel" else field, 0.0)
                                if key_name in doc_recs:
                                    sug_val = doc_recs[key_name]
                                    diff = sug_val - cur_val
                                    diff_str = f"+{diff:.2f}" if diff>0 else f"{diff:.2f}"
                                    diff_col = "#A32D2D" if diff>0.05 else "#3B6D11" if diff<-0.05 else "#888"
                                    adj_cols[ci].markdown(
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:.68rem;color:#707070;"
                                        f"text-transform:uppercase;letter-spacing:0.06em'>{lbl}</div>"
                                        f"<div style='font-family:JetBrains Mono,monospace;"
                                        f"font-size:.95rem;font-weight:600'>{sug_val:.2f}</div>"
                                        f"<div style='font-family:JetBrains Mono,monospace;"
                                        f"font-size:.65rem;color:{diff_col}'>{diff_str}</div>"
                                        f"</div>", unsafe_allow_html=True)

                            # One-click apply
                            if st.button(f"Apply recommended adjustments", key=f"cr_apply_{i}"):
                                for field in adj_labels:
                                    key_name = f"{field}_adj"
                                    if key_name in doc_recs:
                                        # Map counsel → ewert key in adj dict
                                        adj_key = "bail" if field=="bail" else \
                                                  "ewert" if field=="counsel" else \
                                                  "gladue" if field=="gladue" else \
                                                  "police" if field=="police" else \
                                                  "mm" if field=="mm" else "time"
                                        st.session_state.criminal_record[i]["adj"][adj_key] = doc_recs[key_name]
                                # Recompute calibrated weight for this entry
                                adj_u = st.session_state.criminal_record[i]["adj"]
                                new_cal = float(np.clip(
                                    1.0 * (1-0.55*adj_u["bail"]) * (1-0.40*adj_u["ewert"]) *
                                    (1-0.35*adj_u["police"]) * (1-0.30*adj_u["gladue"]) *
                                    (1-0.25*adj_u["mm"]) * (1-0.45*adj_u["time"]), 0.05, 1.0))
                                st.session_state.criminal_record[i]["cal_weight"] = new_cal
                                _cr_feed_nodes()
                                st.rerun()

                        with st.expander("Read full document analysis", expanded=False):
                            st.markdown(f"<div class='at'>{doc_analysis}</div>", unsafe_allow_html=True)

            with cb3:
                if st.button("Remove", key=f"cr_del_{i}"):
                    st.session_state.criminal_record.pop(i)
                    _cr_feed_nodes()
                    st.rerun()

        # ── Document analysis integration ──────────────────────────────────────
        st.markdown(
            "<div style='border-top:1px solid #EFEDE7;margin:24px 0 18px 0'></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-family:Fraunces,Georgia,serif;font-size:1.05rem;"
            "font-weight:500;color:#1A1A1A;margin-bottom:6px'>"
            "📂 Document-assisted calibration</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55'>"
            "Upload prior transcripts, SCE reports, or sentencing decisions. "
            "The document analyser will recommend weight adjustments for the record above."
            "</div>",
            unsafe_allow_html=True,
        )

        cr_up = st.file_uploader("Upload document for record analysis",
            type=["txt","pdf","docx"], key="cr_doc_up")

        if cr_up:
            provider_cr = st.selectbox("AI provider", ["Claude (Anthropic) ★ recommended","GPT-4o (OpenAI)","Gemini (Google)"],
                key="cr_provider")
            if st.button("Analyse document for record calibration", key="cr_analyse"):
                with st.spinner("Analysing document..."):
                    try:
                        from document_analyzer import analyze_document, extract_text_from_file
                        raw_text = extract_text_from_file(cr_up)
                        record_summary = "\n".join([
                            f"{e['year']} — {e['offence']} ({e['jurisdiction']}) — current calibrated weight {e['cal_weight']*100:.0f}%"
                            for e in st.session_state.criminal_record
                        ]) or "No convictions entered yet."

                        cr_prompt = (
                            f"You are a Canadian criminal law expert assisting with a Dangerous Offender "
                            f"sentencing analysis under PARVIS. The following criminal record has been entered:\n"
                            f"{record_summary}\n\n"
                            f"Review the uploaded document and recommend adjustments to the evidentiary weight "
                            f"of each conviction based on what you find. Specifically look for: "
                            f"(1) bail-denial or coercive plea conditions (Antic [2017] SCC 27), "
                            f"(2) ineffective assistance of counsel (GDB [2000] 1 SCR 520), "
                            f"(3) Gladue factor misapplication (Morris 2021 ONCA 680), "
                            f"(4) over-policing patterns (Le [2019] SCC 34), "
                            f"(5) mandatory minimum distortion (Nur/Lloyd), "
                            f"(6) temporal attenuation — old convictions. "
                            f"For each conviction, state: [KEEP / REDUCE / SIGNIFICANT REDUCTION] "
                            f"and cite the specific passage or finding that supports the recommendation."
                        )
                        prov_key = provider_cr.split()[0].lower()
                        result = analyze_document(
                            file_content=raw_text,
                            provider=prov_key,
                            custom_prompt=cr_prompt,
                        )
                        st.session_state.cr_analysis = result
                    except Exception as ex:
                        st.error(f"Analysis error: {ex}")

        if "cr_analysis" in st.session_state and st.session_state.cr_analysis:
            st.markdown(
                "<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                "color:#1A1A1A;margin-top:14px;margin-bottom:6px'>"
                "Document analysis — calibration recommendations</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='at'>{st.session_state.cr_analysis}</div>",
                unsafe_allow_html=True)
            st.caption("Review recommendations above and adjust individual conviction sliders accordingly. Changes update Node 20 in real time.")

        # ── Doctrinal footnotes (restyled to match design language) ───────
        with st.expander("📚 Doctrinal basis for record calibration"):
            for key,(name,col,cite) in CORR_REFS.items():
                st.markdown(
                    f"<div style='border-left:3px solid #{col};padding:.5rem .9rem;"
                    f"margin-bottom:.6rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                    f"color:#{col};font-size:0.92rem'>{name}</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.82rem;color:#3A3A3A;margin-top:2px'>{cite}</div>"
                    f"</div>",
                    unsafe_allow_html=True)

        if st.button("Clear entire record", key="cr_clear"):
            st.session_state.criminal_record = []
            st.session_state.cr_doc_adj = {}
            st.session_state.doc_adj = {k:v for k,v in st.session_state.doc_adj.items() if k not in [7,14,12]}
            st.rerun()

    else:
        st.markdown(
            "<div style='background:#FBFAF7;border:1px dashed #E0DDD6;"
            "border-radius:8px;padding:24px;text-align:center;"
            "font-family:Fraunces,serif;font-style:italic;font-size:0.92rem;"
            "color:#707070;line-height:1.5'>"
            "No convictions entered yet. Use the form above to add prior "
            "criminal record entries."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Run inference + slim live-result strip ────────────────────────────────
    run_inf()
    P = st.session_state.posteriors
    bl_cr, _bc_cr, _bg_cr = rb(P[20])
    _n_conv = len(st.session_state.criminal_record)
    _band_text_cr = {
        "Low":      f"belief largely resolved · {_n_conv} conviction(s) on record",
        "Moderate": f"belief partially resolved · {_n_conv} conviction(s)",
        "Elevated": f"belief shifted · {_n_conv} conviction(s)",
        "High":     f"strong indication · {_n_conv} conviction(s)",
    }.get(bl_cr, bl_cr)
    if _empty:
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr auto;"
            "align-items:center;gap:18px;background:#FBFAF7;"
            "border:1px solid #E0DDD6;border-radius:8px;"
            "padding:11px 18px;margin-top:24px;margin-bottom:0'>"
            "<div style='font-size:0.82rem;color:#9E9E9E;font-weight:500'>"
            "Node 20 · DO designation risk"
            "<span style='font-family:Fraunces,Georgia,serif;font-style:italic;"
            "font-size:1.05rem;font-weight:500;color:#9E9E9E;margin-left:10px'>—</span>"
            "</div>"
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#707070'>Awaiting case data</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr auto;"
            f"align-items:center;gap:18px;background:linear-gradient(90deg,"
            f"#E2EBD8 0%, #EAF3DE 50%, #F7F5F2 100%);border:1px solid #B8CDA8;"
            f"border-radius:8px;padding:11px 18px;margin-top:24px'>"
            f"<div style='font-size:0.82rem;color:#3B6D11;font-weight:500'>"
            f"Node 20 · DO designation risk"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:#2F5C2A;margin-left:8px'>{P[20]*100:.1f}%</span>"
            f"</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.86rem;color:#2F5C2A'>{bl_cr} — {_band_text_cr}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── T10: Scenarios — side-by-side comparison ─────────────────────────────────
with TABS[10]:
    st.markdown("### Scenario comparison")
    st.caption("Save the current profile as a named scenario and compare up to four profiles side by side. Shows how distortion corrections shift DO designation risk.")

    P_sc = st.session_state.posteriors
    saved = st.session_state.saved_scenarios

    # ── Save current scenario ─────────────────────────────────────────────────
    sc1, sc2 = st.columns([3,1])
    with sc1:
        sc_name = st.text_input("Scenario name", placeholder="e.g. Offender A — no Gladue factors applied", key="sc_name")
    with sc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("💾 Save current profile", key="sc_save", use_container_width=True):
            if sc_name.strip():
                snap = {
                    "name":       sc_name.strip(),
                    "do_risk":    P_sc[20],
                    "posteriors": dict(P_sc),
                    "gladue":     list(st.session_state.gladue_checked),
                    "sce":        list(st.session_state.sce_checked),
                    "conn":       st.session_state.conn,
                    "scefw":      st.session_state.scefw,
                    "doc_adj":    dict(st.session_state.doc_adj),
                    "cr_count":   len(st.session_state.criminal_record),
                    "cr_mean_wt": float(np.mean([e["cal_weight"] for e in st.session_state.criminal_record])) if st.session_state.criminal_record else None,
                    "cr_pattern": st.session_state.cr_doc_adj.get("escalation",{}).get("pattern",""),
                    "saved_at":   datetime.now().strftime("%H:%M"),
                }
                saved[sc_name.strip()] = snap
                st.session_state.saved_scenarios = saved
                st.success(f"Saved: **{sc_name.strip()}** — DO risk {snap['do_risk']*100:.1f}%")
            else:
                st.warning("Please enter a scenario name.")

    if not saved:
        st.info("No scenarios saved yet. Configure a case profile and click **Save current profile** above. Save a second profile with different settings to compare them side by side.")
    else:
        st.markdown("---")
        st.markdown(f"#### {len(saved)} saved scenario(s)")

        # ── Comparison chart ──────────────────────────────────────────────────
        sc_names  = list(saved.keys())
        sc_risks  = [saved[n]["do_risk"] * 100 for n in sc_names]
        sc_cols   = ["#3B6D11" if r < 35 else "#BA7517" if r < 55 else "#A32D2D" for r in sc_risks]

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(max(6, len(sc_names)*2.2), 4.5), facecolor="#fafafa")
        bars = ax.bar(sc_names, sc_risks, color=sc_cols, alpha=0.85,
                      width=0.55, edgecolor="white", linewidth=1.5)

        # Threshold lines
        ax.axhline(35, color="#BA7517", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(55, color="#A32D2D", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(len(sc_names)-0.5, 35.5, "Moderate threshold (35%)", fontsize=8, color="#BA7517")
        ax.text(len(sc_names)-0.5, 55.5, "Elevated threshold (55%)", fontsize=8, color="#A32D2D")

        # Value labels on bars
        for bar, val in zip(bars, sc_risks):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=11, color="#1B2A4A")

        ax.set_ylim(0, min(100, max(sc_risks) + 12))
        ax.set_ylabel("DO Designation Risk (%)", fontsize=10, color="#555")
        ax.set_facecolor("#fafafa")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9, rotation=10)
        ax.set_title("DO Designation Risk — Scenario Comparison (Node 20)", fontsize=11,
                     color="#1B2A4A", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Detailed cards ────────────────────────────────────────────────────
        st.markdown("#### Scenario detail")
        card_cols = st.columns(min(len(saved), 3))
        for ci, (name, snap) in enumerate(saved.items()):
            col_idx = ci % min(len(saved), 3)
            with card_cols[col_idx]:
                bl_s, bc_s, bg_s = rb(snap["do_risk"])
                # Distortion corrections summary
                dist_nodes = {5:"Ewert",6:"Counsel",7:"Bail",9:"FASD",
                              10:"IntGen",11:"NoCult",12:"Gladue",14:"Police",15:"Temporal"}
                active_dist = [lbl for nid,lbl in dist_nodes.items()
                               if snap["posteriors"].get(nid,0) > 0.35]
                gladue_ct = len(snap.get("gladue",[]))
                sce_ct    = len(snap.get("sce",[]))

                st.markdown(
                    f"<div style='border:1px solid {bc_s}44;border-top:4px solid {bc_s};"
                    f"border-radius:10px;padding:.9rem;background:{bg_s};margin-bottom:.5rem'>"
                    f"<div style='font-weight:700;font-size:.92rem;color:#1B2A4A;margin-bottom:4px'>"
                    f"{name}</div>"
                    f"<div style='font-size:1.9rem;font-weight:800;font-family:monospace;color:{bc_s}'>"
                    f"{snap['do_risk']*100:.1f}%</div>"
                    f"<div style='font-size:.78rem;font-weight:600;color:{bc_s}'>{bl_s}</div>"
                    f"<hr style='margin:6px 0;border-color:{bc_s}33'>"
                    f"<div style='font-size:.75rem;color:#555'>"
                    f"Gladue factors: <b>{gladue_ct}</b><br>"
                    f"Morris/Ellis SCE: <b>{sce_ct}</b> · Connection: <b>{snap.get('conn','—')}</b><br>"
                    f"{'Criminal record: <b>' + str(snap['cr_count']) + ' conv · ' + str(int(snap['cr_mean_wt']*100)) + '% cal. wt</b><br>' if snap.get('cr_count') else ''}"
                    f"Active distortions: <b>{', '.join(active_dist) if active_dist else 'none'}</b>"
                    f"</div>"
                    f"<div style='font-size:.68rem;color:#aaa;margin-top:4px'>Saved {snap.get('saved_at','')}</div>"
                    f"</div>",
                    unsafe_allow_html=True)
                if st.button("🗑 Remove", key=f"sc_del_{name}", use_container_width=True):
                    del st.session_state.saved_scenarios[name]
                    st.rerun()

        # ── Delta analysis ────────────────────────────────────────────────────
        if len(saved) >= 2:
            st.markdown("---")
            st.markdown("#### Delta analysis — pairwise risk shifts")
            sc_list = list(saved.items())
            for i in range(len(sc_list)):
                for j in range(i+1, len(sc_list)):
                    na, sa = sc_list[i]
                    nb, sb = sc_list[j]
                    delta = (sb["do_risk"] - sa["do_risk"]) * 100
                    arrow = "↑" if delta > 0 else "↓"
                    d_col = "#A32D2D" if delta > 5 else "#3B6D11" if delta < -5 else "#BA7517"
                    gladue_d = len(sb.get("gladue",[])) - len(sa.get("gladue",[]))
                    sce_d    = len(sb.get("sce",[])) - len(sa.get("sce",[]))
                    st.markdown(
                        f"<div style='background:#f8f8f8;border-radius:8px;padding:.6rem .9rem;"
                        f"margin-bottom:.4rem;font-size:.83rem'>"
                        f"<b>{na}</b> → <b>{nb}</b>:  "
                        f"<span style='color:{d_col};font-weight:700;font-size:1.05rem'>"
                        f"{arrow} {abs(delta):.1f} pp</span>"
                        f"<span style='color:#888;margin-left:12px'>"
                        f"Gladue Δ{gladue_d:+d}  ·  SCE Δ{sce_d:+d}  ·  "
                        f"Connection: {sa.get('conn','—')} → {sb.get('conn','—')}</span>"
                        f"</div>",
                        unsafe_allow_html=True)

        if st.button("🗑 Clear all scenarios", key="sc_clear_all"):
            st.session_state.saved_scenarios = {}
            st.rerun()


# ── T9: Audit report ──────────────────────────────────────────────────────────
with TABS[12]:
    st.markdown("### Audit report")
    st.caption("Full inference documentation — exportable for legal review and viva presentation.")
    Pa=st.session_state.posteriors;da=Pa[20];bla,bca,_=rb(da)
    cG=[f for f in GF if f["id"] in st.session_state.gladue_checked]
    cS=[f for f in SF if f["id"] in st.session_state.sce_checked]
    mx=cmult()

    def sec(t): return f"\n{'─'*60}\n  {t}\n{'─'*60}"

    rpt=f"""╔══════════════════════════════════════════════════════════════╗
║                        P A R V I S                          ║
║        Probabilistic and Analytical Reasoning               ║
║        Virtual Intelligence System · Xavier 7               ║
╚══════════════════════════════════════════════════════════════╝

  Prepared by:    Jeinis Patel, PhD Candidate and Barrister
  Institution:    University of London
  Initiative:     Ethical AI Initiative
  Generated:      {datetime.now().strftime('%d %B %Y · %H:%M')}
  Engine:         pgmpy Variable Elimination (genuine Bayesian inference)
{sec('INFERENCE OUTPUT')}
  Node 20 — DO Designation Risk:   {da*100:.2f}%   [{bla.upper()}]

  This figure represents the posterior probability of Dangerous Offender
  designation given all upstream evidence, corrections, and doctrinal
  adjustments applied. It models DESIGNATION RISK — not intrinsic
  dangerousness. This distinction is the thesis's central normative
  contribution.
{sec('DOCTRINAL FRAMEWORK')}
  ► R v Gladue [1999] 1 SCR 688
  ► R v Ipeelee [2012] SCC 13
  ► R v Morris 2021 ONCA 680 (para 97 connection gate)
    Active framework: {st.session_state.scefw.upper()}
    Connection: {st.session_state.conn.upper()} · Multiplier: {mx:.2f}
  ► R v Ellis 2022 BCCA 278
  ► Ewert v Canada [2018] SCC 30
  ► R v Boutilier 2017 SCC 64
  ► R v Natomagan 2022 ABCA 48
{sec('GLADUE FACTOR CHECKLIST')}"""

    if cG:
        for f in cG: rpt+=f"\n  [✓] {f['l']}\n       → Node {f['n']}  (+{f['w']*100:.0f}%)"
    else: rpt+="\n  No Gladue factors selected."

    rpt+=sec("MORRIS / ELLIS SOCIAL CONTEXT EVIDENCE")
    if cS:
        for f in cS: rpt+=f"\n  [✓] {f['l']}\n       → Node {f['n']}  (+{f['w']*mx*100:.1f}% after connection weight {mx:.2f})"
    else: rpt+="\n  No Morris/Ellis SCE factors selected."

    if st.session_state.doc_adj:
        rpt+=sec("DOCUMENT ANALYSIS ADJUSTMENTS")
        for nid,d in st.session_state.doc_adj.items():
            m=NODE_META.get(nid,{})
            rpt+=f"\n  [✓] N{nid} {m.get('name','')}: {'↑' if d>0 else '↓'} {abs(d):.2f}"

    rpt+=sec("RISK FACTOR POSTERIORS (Variable Elimination)")
    for nid in NODE_META:
        if NODE_META[nid]["type"]=="risk" and nid!=20:
            rpt+=f"\n  N{str(nid).rjust(2)}  {NODE_META[nid]['short'].ljust(30)} {Pa.get(nid,.5)*100:5.1f}%"

    rpt+=sec("SYSTEMIC DISTORTION CORRECTIONS")
    for nid in NODE_META:
        if NODE_META[nid]["type"] not in ("risk","output") and nid!=20:
            rpt+=f"\n  N{str(nid).rjust(2)}  {NODE_META[nid]['short'].ljust(30)} {Pa.get(nid,.5)*100:5.1f}%"

    # ── Criminal Record section in audit report ──────────────────────────────
    cr_rec = st.session_state.criminal_record
    if cr_rec:
        rpt += sec("CALIBRATED CRIMINAL RECORD")
        esc_info = st.session_state.cr_doc_adj.get("escalation", {})
        esc_pat  = esc_info.get("pattern", "insufficient").title()
        rpt += f"\n  Convictions entered:     {len(cr_rec)}"
        cal_wts = [e["cal_weight"] for e in cr_rec]
        rpt += f"\n  Mean calibrated weight:  {float(np.mean(cal_wts))*100:.0f}% of nominal"
        rpt += f"\n  Pattern (Boutilier):     {esc_pat}"
        rpt += f"\n  {esc_info.get('note','')}"
        rpt += f"\n"
        for e in cr_rec:
            cal_pct = e["cal_weight"] * 100
            rpt += f"\n  [{e['year']}] {e['offence']}"
            rpt += f"\n       Court:       {e.get('court','—')}  ·  {e.get('jurisdiction','')}"
            rpt += f"\n       Seriousness: {e.get('seriousness_label','—')}"
            rpt += f"\n       Cal. weight: {cal_pct:.0f}% retained  (raw 100%)"
            adj = e.get("adj", {})
            dists = []
            if adj.get("bail",0)>0.1:   dists.append(f"bail-denial {adj['bail']*100:.0f}%")
            if adj.get("ewert",0)>0.1:  dists.append(f"Ewert {adj['ewert']*100:.0f}%")
            if adj.get("police",0)>0.1: dists.append(f"over-policing {adj['police']*100:.0f}%")
            if adj.get("gladue",0)>0.1: dists.append(f"Gladue misapplication {adj['gladue']*100:.0f}%")
            if adj.get("mm",0)>0.05:    dists.append(f"mandatory minimum {adj['mm']*100:.0f}%")
            if adj.get("time",0)>0.1:   dists.append(f"temporal attenuation {adj['time']*100:.0f}%")
            if e.get("gang"):           dists.append("gang / organized crime context")
            if dists:
                rpt += f"\n       Distortions: {', '.join(dists)}"
            if e.get("doc_analysis"):
                rpt += f"\n       [Document analysis attached — see full record]"
            rpt += "\n"
        # Record reliability signal
        n2_from_cr = st.session_state.cr_doc_adj.get(2, 0)
        if n2_from_cr:
            rpt += f"\n  → N2 (violent history) calibrated signal: {n2_from_cr*100:.1f}%"
            rpt += f"\n    (feeds directly into Node 20 DO designation risk)"

    qbt=format_report(st.session_state.qdiags) if st.session_state.qdiags else "(QBism pending)"
    rpt+=f"\n\n{qbt}"
    rpt+=sec("ARCHITECTURAL NOTES")
    rpt+="""
  Inference:   pgmpy Variable Elimination (genuine Bayesian inference)
  Node 20:     Calibrated post-VE — distortion nodes REDUCE effective
               risk weight, consistent with thesis argument.

  CPTs encode normative priors grounded in doctrine — not empirical
  frequencies. Extended Bayesian tradition: subjective, robust,
  decision-theoretic Bayesianism.

  ─────────────────────────────────────────────────────────────────
  PARVIS Xavier 7  ·  Research use only
  NOT for deployment in live proceedings

  © Jeinis Patel, PhD Candidate and Barrister
  Ethical AI Initiative · University of London
"""

    # ── Enhanced report display ────────────────────────────────────────────────
    st.markdown(f"<div class='at'>{rpt}</div>", unsafe_allow_html=True)

    # ── Criminal record section in report ──────────────────────────────────────
    cr_rec = st.session_state.criminal_record
    if cr_rec:
        esc_info = st.session_state.cr_doc_adj.get("escalation", {})
        st.markdown(
            f"<div class='at' style='margin-top:.8rem'>"
            f"{'─'*60}\n  CALIBRATED CRIMINAL RECORD\n{'─'*60}\n" +
            "\n".join([
                f"  [{e['year']}] {e['offence']} · {e.get('jurisdiction','')}\n"
                f"    Seriousness: {e.get('seriousness_label','—')} · "
                f"Calibrated weight: {e['cal_weight']*100:.0f}% of nominal\n"
                f"    Distortions: bail={e['adj']['bail']:.2f} "
                f"police={e['adj']['police']:.2f} "
                f"gladue={e['adj']['gladue']:.2f} "
                f"time={e['adj']['time']:.2f}"
                + (f"\n    Gang context: YES" if e.get("gang") else "")
                for e in cr_rec
            ]) +
            f"\n\n  Pattern (Boutilier): {esc_info.get('pattern','—').title()}\n"
            f"  {esc_info.get('note','')}\n{'─'*60}</div>",
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Export functions — reload module each time to bust Streamlit cache ────
    import importlib, audit_export as _ae
    importlib.reload(_ae)

    def _build_docx():
        return _ae.build_docx(
            Pa=Pa, da=da, bla=bla, cG=cG, cS=cS, mx=mx,
            cr_rec=st.session_state.criminal_record,
            cr_doc_adj=st.session_state.cr_doc_adj,
            doc_adj=st.session_state.doc_adj,
            scefw=st.session_state.scefw,
            conn=st.session_state.conn,
            qdiags=st.session_state.qdiags,
            NODE_META=NODE_META,
        )

    def _build_pdf():
        return _ae.build_pdf(
            Pa=Pa, da=da, bla=bla, cG=cG, cS=cS, mx=mx,
            cr_rec=st.session_state.criminal_record,
            cr_doc_adj=st.session_state.cr_doc_adj,
            scefw=st.session_state.scefw,
            conn=st.session_state.conn,
            qdiags=st.session_state.qdiags,
            NODE_META=NODE_META,
        )

        # ── Export buttons ─────────────────────────────────────────────────────────
    st.markdown("#### Export")
    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        try:
            docx_bytes = _build_docx()
            st.download_button(
                "📄 Download Word (.docx)",
                data=docx_bytes,
                file_name=f"PARVIS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True)
        except Exception as ex:
            st.error(f"Word export error: {ex}")

    with ec2:
        try:
            pdf_bytes = _build_pdf()
            st.download_button(
                "📋 Download PDF",
                data=pdf_bytes,
                file_name=f"PARVIS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True)
        except Exception as ex:
            st.error(f"PDF export error: {ex}")

    with ec3:
        st.download_button(
            "📝 Download TXT",
            data=rpt.encode(),
            file_name=f"PARVIS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True)

    st.markdown(
        "<div class='parvis-footer'>"
        "P.A.R.V.I.S &nbsp;·&nbsp; Ethical AI Initiative &nbsp;·&nbsp; "
        "University of London &nbsp;·&nbsp; "
        "Jeinis Patel, PhD Candidate and Barrister &nbsp;·&nbsp; "
        "Research use only &nbsp;·&nbsp; © 2026</div>",
        unsafe_allow_html=True)

    with st.expander("⚛️ QBism diagnostics (.txt)"):
        qbo=format_report(st.session_state.qdiags) if st.session_state.qdiags else ""
