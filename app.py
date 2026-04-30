"""
PARVIS — Streamlit Application v Xavier 7
Jeinis Patel, PhD Candidate and Barrister | University of London | Ethical AI Initiative
"""

# ═════════════════════════════════════════════════════════════════════════════
# CANONICAL NODE TAXONOMY — Chapter 5 (April 11, 2026)
# ═════════════════════════════════════════════════════════════════════════════
#
# This is the AUTHORITATIVE node taxonomy. Code in this file MUST match these
# titles when referencing nodes by number. The thesis (Chapter 5) is canonical;
# the architecture defers to the thesis, never the reverse.
#
# i. Substantive Risk Layer
#    N1   — Criminal Law Burden of Proof (Contextualizer)
#    N2   — Validated Risk Elevators
#    N2i  — Temporal Modifier (Criminal Burnout)
#    N3   — Sexual Offence Risk Profile
#    N4   — Dynamic Risk Factor Cluster
#
# ii. Systemic Distortion and Doctrinal Fidelity Layer
#    N5   — Current Risk Assessment Tools
#    N6   — Ineffective Assistance of Counsel
#    N7   — Denial of Bail → Wrongful Conviction Guilty Plea Cascade
#    N7i  — Anticipated Credibility Impeachment / Strategic Pleas
#    N8   — FASD as Dual-Factor in Risk Modeling
#    N9   — Intergenerational Trauma & Absence of Culturally Grounded Treatment
#    N10  — Judicial Misapplication of Social Context Evidence (sub-nodes 10a-10d)
#    N11  — Gaming Risk ("BS") Detector
#    N12  — Judging the Judge — Judicial Reasoning Reliability
#    N13  — Structural Systemic Bias (TraceRoute)
#    N14  — Temporal Distortion and Systemic Misrepresentation in Prior Records
#    N15  — Interjurisdictional Tariff Distortion
#    N16  — Doctrinal Tension (s.718.04 vs s.718.2(e))
#    N17  — Over-Policing and Epistemic Contamination of Criminal Records
#    N18  — Gladue / Ewert / Morris / Ellis Profile (SCE Integration Audit)
#    N19  — Collider Bias Node
#
# iii. Structural Output
#    N20  — Dangerous Offender Designation
#
# Layer boundaries:
#    Substantive Risk:           N1-N4 (with N2i)
#    Systemic Distortion:        N5-N19 (with N7i, N10a-N10d)
#    Structural Output:          N20
#
# When adding new code: VERIFY any node-number reference against this table
# before adding it. The session memory may carry stale taxonomy from earlier
# builds; this comment block is the ground truth.
# ═════════════════════════════════════════════════════════════════════════════

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import base64, os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from model import build_model, get_inference_engine, query_do_risk, NODE_META, EDGES_VE as EDGES, compute_n1_prior_from_audit
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
div[data-testid="stSlider"][aria-label="N4 — Dynamic risk"] > div > div > div > div { background:#A32D2D !important; }
div[data-testid="stSlider"][aria-label="N5 — Risk tools"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N6 — Ineffective counsel"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N7 — Bail-denial cascade"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N9 — FASD"] > div > div > div > div { background:#534AB7 !important; }
div[data-testid="stSlider"][aria-label="N9 — Intergenerational trauma"] > div > div > div > div { background:#3B6D11 !important; }
div[data-testid="stSlider"][aria-label="N9 — No cultural treatment"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N10 — Gladue misapplication"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N11 — Gaming risk"] > div > div > div > div { background:#0F6E56 !important; }
div[data-testid="stSlider"][aria-label="N17 — Over-policing"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N14 — Temporal distortion"] > div > div > div > div { background:#185FA5 !important; }
div[data-testid="stSlider"][aria-label="N9 — No rehabilitation"] > div > div > div > div { background:#185FA5 !important; }
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

# ── N6 (Ineffective Assistance of Counsel) — §RM.6 ──────────────────────
# N6 is upstream of N7. Per §5.1.6, four binary indicators aggregate to a
# tri-state ordinal grade; per §RM.6, that grade additively boosts N7's
# per-conviction bail-denial signal before threshold logic.

# The four indicator keys; UI checkboxes set these on each conviction.
# Adverse direction noted in inline comment.
N6_INDICATORS = (
    "n6_no_sce",           # adverse: SCE not submitted
    "n6_inadequate_counsel",   # adverse: counsel culturally inadequate
    "n6_judicial_criticism",   # adverse: judicial criticism of representation present
    "n6_disproportionate",     # adverse: procedural outcome disproportionate
)

# Threshold scheme per §RM.6.3 (JP 2026-04-28: option (c)).
# 0 adverse  -> High;  1, 2, or 3 adverse -> Moderate;  4 adverse -> Low.
N6_GRADE_BY_COUNT = {0: "High", 1: "Moderate", 2: "Moderate", 3: "Moderate", 4: "Low"}

# Additive boost to N7's per-conviction bail-denial signal per §RM.6.4.
# Conservative values per §RM.2 conservatism principle.
N6_BOOST_BY_GRADE = {
    "High":     0.00,
    "Moderate": 0.10,
    "Low":      0.20,
}


def _n6_count_adverse(conviction):
    """Return the count of adverse N6 indicators for this conviction (0-4)."""
    adj = conviction.get("adj", {}) or {}
    return sum(1 for k in N6_INDICATORS if bool(adj.get(k, False)))


def _n6_grade_for_conviction(conviction):
    """
    Return the N6 confidence grade for a conviction per §RM.6.3.
    Grade depends only on this conviction's own indicators (no propagation
    per §RM.6.6: different counsel, different files).
    """
    return N6_GRADE_BY_COUNT[_n6_count_adverse(conviction)]


def _n6_boost_for_conviction(conviction):
    """Return the N6 additive boost (0.00 / 0.10 / 0.20) for this conviction."""
    return N6_BOOST_BY_GRADE[_n6_grade_for_conviction(conviction)]


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
    Compute per-conviction grades in chronological order, applying:
      (1) N6 additive boost (per §RM.6.4)
      (2) Cascade propagation factor (per §RM.5.4)
    in that order, per §RM.6.5.

    Per §RM.5.4: a conviction's per-conviction bail-denial signal is
    multiplied by the strongest propagation factor from earlier graded
    tainted convictions on the same record. Propagation requires:
      (a) the downstream conviction has its own affirmative bail-denial
          signal (adj.bail > 0)
      (b) at least one earlier conviction is graded Discounted or
          Heavily Discounted

    Per §RM.6.4: the per-conviction bail-denial signal is first boosted
    additively by the N6 confidence grade (+0.00 / +0.10 / +0.20) before
    any propagation factor is applied. N6 conditioning is per-conviction
    only; representation quality on Conviction A does not propagate
    forward to Conviction B (different counsel, different files).

    Returns: list of (n7_grade, n7_multiplier, propagation_factor, n6_grade)
    tuples in the same order as the input record.
    """
    if not criminal_record_chronological:
        return []

    results = []
    earlier_grades = []  # Accumulates as we walk the record forward

    for e in criminal_record_chronological:
        per_conv_bail = float(e.get("adj", {}).get("bail", 0.0))

        # ── (1) N6 conditioning per §RM.6.4 ───────────────────────────────
        # N6 boost only applies where the conviction has its own bail-denial
        # signal. Per §RM.6.4 final paragraph: "The architecture does not
        # apply N6's conditioning where N6 has nothing to condition."
        n6_grade = _n6_grade_for_conviction(e)
        if per_conv_bail > 0.0:
            n6_addition = N6_BOOST_BY_GRADE[n6_grade]
        else:
            n6_addition = 0.0
        n6_conditioned = per_conv_bail + n6_addition

        # ── (2) Cascade propagation per §RM.5.4 ───────────────────────────
        # Propagation factor: strongest of any earlier tainted conviction.
        # If no earlier tainted convictions, factor = 1.00.
        if per_conv_bail > 0.0 and earlier_grades:
            applicable_factors = [
                N7_PROPAGATION_FACTOR[g] for g in earlier_grades
                if g in ("Discounted", "Heavily Discounted")
            ]
            propagation = max(applicable_factors) if applicable_factors else 1.00
        else:
            # No own bail signal: propagation cannot apply (§RM.5.3).
            propagation = 1.00

        # ── Combined boosted signal: (bail + N6) × propagation ──────────
        boosted_signal = n6_conditioned * propagation

        # Threshold logic
        n7_grade = _n7_threshold_grade(boosted_signal)
        n7_multiplier = N7_MULTIPLIERS[n7_grade]

        results.append((n7_grade, n7_multiplier, propagation, n6_grade))
        earlier_grades.append(n7_grade)

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
    grades = [(grade, mult) for grade, mult, _prop, _n6 in chronological_results]
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
    for e, (grade, _mult, _prop, _n6) in zip(criminal_record_chronological, grading):
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
            # §5.1.17 N17b counsel attestation — case-level per JP M8/P2
            # (over-policing is a case-wide policing-environment claim, not
            # per-conviction). Stays on Profile tab.
            "n17b_counsel_attestation":False,
            # §5.1.18 N18a counsel attestation — case-level per JP M8/P2
            # (concerns the present sentencing court's jurisdiction's
            # SCE-integration precedent, not any prior conviction). Stays
            # on Profile tab near the case-jurisdiction input.
            "n18a_counsel_attestation":False,
            # Mark 8 Phase 2 relocation: N14a/b/c, N15a/b/c, N18b/c moved
            # to per-conviction storage on the criminal_record entries.
            # Aggregate signals derived via _any_conviction_attests().
            # See JP M8/P2 lock-in.
            # §5.1.19 N19 collider-bias secondary computation slots (Q4=C)
            "n19_collider_signal":None,        # set by run_inf via _n19_collider_signal
            "n19_discounted_risk":None,        # set by run_inf via compute_do_risk(collider_discount=True)
            # ── §5.1.1 N1 burden-of-proof audit (Mark 8) ──────────────────
            # Per-input audit-state dict keyed by stable input ID.
            # Each record carries: provenance ("crown"/"defence"/"agreed"/
            # "judicial"), use ("aggravating"/"mitigating"/"contextual"/
            # "agreed_fact"), judicial_finding_type (Ferguson sub-flag,
            # only populated when provenance=="judicial"), applicable_burden
            # ("BARD"/"BoP"/"none"), attestation (free text from user),
            # attestation_status ("satisfied"/"insufficient"/"pending"),
            # citations (list of authority strings), tab (origin tab),
            # label (display description for §RM.1 register).
            # Audit logic: model.compute_n1_prior_from_audit(audit_state)
            # → target P(N1=High); fed to query_do_risk via virtual evidence
            # so audit-derived N1 propagates structurally through VE.
            # See model.py § "N1 burden-of-proof audit" for full doctrinal
            # architecture (Gardiner, Ferguson, Angelillo, s.724(3)).
            "n1_audit": {},
            # Per-case strict-mode flag (Q3=A locked). Strict mode fires
            # the burden-of-proof audit prompt at moment-of-entry rather
            # than at tab-exit. Stored as case metadata so it round-trips
            # on save/load (preserves audit semantics across user handoffs).
            "strict_mode": False,
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


# ─────────────────────────────────────────────────────────────────────────────
# §5.1.17 N17 sub-node signal computation
#
# Per JP confirmation (Apr 28-29, 2026):
#   M1: N17a derived from jurisdiction with Moderate default
#   M2: N17c/N17d pattern-matched from criminal record now; schema migration later
#   M3: N17b OR-gate over Gladue tab evidence with counsel attestation override
# ─────────────────────────────────────────────────────────────────────────────

# Strong-match Gladue/SCE factor IDs that engage §5.1.17 §2 enforcement-disparity
# patterns. OR-gate: any one fires → N17b signal high.
_N17B_TRIGGER_FACTORS = {
    "g_op",   # Over-policed community of origin
    "s_ra",   # Anti-Black / racialized racism documented
    "s_rp",   # Documented racial profiling
    "s_bi",   # Anti-Black systemic incarceration patterns
    "s_bb",   # Anti-Black bail practices documented
    "s_nb",   # Neighbourhood structural disadvantage
}

# §5.1.17 §2 surveillance-trigger keyword patterns. Lowercased substring match
# on offence text. Conservative — false positives possible (e.g. "possession of
# stolen property" is included; counsel can override at per-conviction level
# via the existing adj_police slider).
_N17D_SURVEILLANCE_PATTERNS = (
    "breach",
    "fail to comply",
    "fail to appear",
    "administration of justice",
    "aoj",
    "possession",
)

# Violence keywords — any match disqualifies an offence from being counted as
# "non-violent" for N17c regardless of seriousness tier. Matches §5.1.17 §2's
# distinction between "non-violent charges" (low-harm/compliance) and
# violence-driven entries.
_N17C_VIOLENCE_PATTERNS = (
    "assault",
    "robbery",
    "manslaughter",
    "murder",
    "homicide",
    "sexual",
    "firearm",
    "aggravated",
    "weapon",
)


def _n17a_signal(case_jur: str) -> int:
    """
    N17a — Jurisdictional Policing Disparity Index.

    Per JP M1 confirmation: Moderate default for every jurisdiction with override
    available later. Currently returns binary state: 0 (Low/Moderate) for default,
    1 (High) only if jurisdiction is known to have documented over-policing
    patterns AND we have explicit override evidence.

    Conservative architectural choice: do not embed contested empirical claims
    about specific provinces in the constructive-proof CPT. The override path
    (counsel-input slider) is reserved for Stage 3 if needed.

    Returns: 0 (Low/Moderate disparity) or 1 (High disparity)
    """
    # M1 conservative default: every jurisdiction starts Low/Moderate (state 0).
    # Override mechanism deferred to Stage 3.
    return 0


def _n17b_signal() -> int:
    """
    N17b — Enforcement-Disparity Engagement.

    Per JP M3 confirmation: OR-gate over Gladue/SCE tab evidence with counsel
    attestation override. If any §5.1.17 §2 trigger factor is checked, OR if
    counsel has attested, return 1 (High).

    Read from st.session_state.gladue_checked (Gladue tab) and
    st.session_state.sce_checked (Morris/Ellis SCE tab) — both contribute since
    §5.1.17 §2 cites both Indigenous and Black community over-policing patterns.

    Returns: 0 (no engagement) or 1 (documented engagement)
    """
    gladue = st.session_state.get("gladue_checked", set()) or set()
    sce = st.session_state.get("sce_checked", set()) or set()
    attestation = st.session_state.get("n17b_counsel_attestation", False)

    # OR-gate: any trigger factor present, or attestation
    if attestation:
        return 1
    if any(fid in _N17B_TRIGGER_FACTORS for fid in gladue):
        return 1
    if any(fid in _N17B_TRIGGER_FACTORS for fid in sce):
        return 1
    return 0


def _n17c_signal(criminal_record) -> int:
    """
    N17c — Non-Violent Charge Density.

    Per JP M2 confirmation: pattern-match offence text now, schema migration
    later. Computes proportion of non-violent charges in the record. Threshold
    at 0.50 — if more than half the record is low-harm/non-violent, signal High.

    A conviction is counted as non-violent when:
      (a) seriousness tier is Minor or Moderate, AND
      (b) offence text contains no violence keywords

    Returns: 0 (low non-violent density) or 1 (high non-violent density)
    """
    if not criminal_record:
        return 0
    total = len(criminal_record)
    if total == 0:
        return 0
    non_violent_count = 0
    for entry in criminal_record:
        offence = (entry.get("offence", "") or "").lower()
        seriousness_label = (entry.get("seriousness_label", "") or "").lower()
        # Check seriousness tier
        is_low_tier = ("minor" in seriousness_label or "moderate" in seriousness_label)
        # Check for violence keywords
        has_violence = any(p in offence for p in _N17C_VIOLENCE_PATTERNS)
        if is_low_tier and not has_violence:
            non_violent_count += 1
    density = non_violent_count / total
    return 1 if density > 0.50 else 0


def _n17d_signal(criminal_record) -> int:
    """
    N17d — Surveillance-Triggered Entries.

    Per JP M2 confirmation: pattern-match offence text for surveillance signatures
    per §5.1.17 §2: breaches, AOJ offences, simple possession. Threshold at 0.30
    — even moderate density of surveillance-triggered entries is concerning per
    §5.1.17 §5 ("when surveillance-triggered entries dominate").

    Returns: 0 (few surveillance-triggered) or 1 (many surveillance-triggered)
    """
    if not criminal_record:
        return 0
    total = len(criminal_record)
    if total == 0:
        return 0
    surveillance_count = 0
    for entry in criminal_record:
        offence = (entry.get("offence", "") or "").lower()
        if any(p in offence for p in _N17D_SURVEILLANCE_PATTERNS):
            surveillance_count += 1
    density = surveillance_count / total
    return 1 if density >= 0.30 else 0


def _compute_n17_evidence() -> dict:
    """
    Compute the four §5.1.17 N17 sub-node evidence states for inference.

    Returns dict mapping pgmpy node IDs ('17a', '17b', '17c', '17d') to
    binary states (0 or 1). Suitable for passing as hard evidence to
    query_do_risk.
    """
    case_jur = st.session_state.get("case_jur", "") or ""
    record = st.session_state.get("criminal_record", []) or []
    return {
        "17a": _n17a_signal(case_jur),
        "17b": _n17b_signal(),
        "17c": _n17c_signal(record),
        "17d": _n17d_signal(record),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5.1.14 N14 sub-node signal computation
#
# Per JP confirmation Q6 (α): year-based heuristics + counsel attestation
# override. N14a/b/c are app-derived from criminal-record years and counsel
# input; N14d is computed by the BN as a child of N10 (Judicial Misapplication).
# ─────────────────────────────────────────────────────────────────────────────

# Mandatory-minimum-era offence keywords. §5.1.14 §2 + Bill C-10 lineage:
# drug trafficking, weapons offences, sexual offences against minors all had
# MM provisions during 2008-2015. Many struck down post-Nur (2015) / Lloyd (2016).
_N14B_MM_OFFENCE_PATTERNS = (
    "trafficking",       # drug trafficking
    "production",        # drug production
    "import",            # drug importation
    "firearm",           # weapons offences
    "weapon",
    "discharging",       # discharging firearm
    "sexual interference",   # CC s.151
    "sexual exploitation",   # CC s.153
    "invitation to sexual",  # CC s.152
    "child pornography",     # CC s.163.1
)

# Era boundaries (Canadian sentencing law context).
_N14A_ERA_HIGH_START = 2008  # Bill C-10 era begins
_N14A_ERA_HIGH_END = 2015    # post-Nur, MM strikedowns begin
_N14B_MM_ERA_START = 2008
_N14B_MM_ERA_END = 2015
_N14C_SCE_AVAILABLE_FROM = 2012  # Ipeelee — SCE substantively available


def _n14a_signal() -> int:
    """
    N14a — Sentencing Era Severity.

    Per Q6 (α): year-based heuristic + counsel attestation override.
    State 1 (High severity era) when:
      - Counsel attestation present, OR
      - Record contains conviction(s) from 2008-2015 (Bill C-10 era)

    Returns: 0 (Low/Moderate era) or 1 (High severity era)
    """
    if _any_conviction_attests("n14a_attestation"):
        return 1
    record = st.session_state.get("criminal_record", []) or []
    for entry in record:
        try:
            year = int(entry.get("year", 0))
        except (TypeError, ValueError):
            continue
        if _N14A_ERA_HIGH_START <= year <= _N14A_ERA_HIGH_END:
            return 1
    return 0


def _n14b_signal(criminal_record) -> int:
    """
    N14b — Historical Mandatory Minimum.

    Per Q6 (α): pattern-match offence text + year against MM-era offences.
    State 1 (MM-era conviction present) when:
      - Counsel attestation present, OR
      - Record contains MM-eligible offence from 2008-2015

    Returns: 0 (no MM-era conviction) or 1 (MM-era conviction present)
    """
    if _any_conviction_attests("n14b_attestation"):
        return 1
    if not criminal_record:
        return 0
    for entry in criminal_record:
        offence = (entry.get("offence", "") or "").lower()
        try:
            year = int(entry.get("year", 0))
        except (TypeError, ValueError):
            continue
        # Within MM era and matches MM-eligible offence pattern
        if _N14B_MM_ERA_START <= year <= _N14B_MM_ERA_END:
            if any(p in offence for p in _N14B_MM_OFFENCE_PATTERNS):
                return 1
    return 0


def _n14c_signal(criminal_record) -> int:
    """
    N14c — SCE Absent at Sentencing.

    Per Q6 (α): year-based heuristic + counsel attestation (inverse).
    State 1 (SCE absent — adverse) when:
      - Record contains pre-2012 conviction (pre-Ipeelee era), AND
      - Counsel has NOT attested SCE was applied at original sentencing
    State 0 (SCE present) when:
      - Counsel attests SCE was applied, OR
      - All convictions are post-2012 (Ipeelee era and later)

    Note: the attestation checkbox semantics are inverted for UX clarity —
    counsel checks "SCE was applied at original sentencing" → that means N14c=Low.

    Returns: 0 (SCE present) or 1 (SCE absent at sentencing)
    """
    # Counsel attestation: "SCE was applied at original sentencing"
    # If ANY conviction has this flag set per JP M8/P2 any-prior aggregation,
    # force N14c = 0 (SCE present, no distortion contribution). Inverse
    # semantics preserved from the case-level original.
    if _any_conviction_attests("n14c_sce_applied"):
        return 0
    if not criminal_record:
        return 0  # no record → no SCE distortion to flag
    # Heuristic: any pre-Ipeelee conviction → SCE likely absent/nominal
    for entry in criminal_record:
        try:
            year = int(entry.get("year", 0))
        except (TypeError, ValueError):
            continue
        if year > 0 and year < _N14C_SCE_AVAILABLE_FROM:
            return 1
    return 0


def _compute_n14_evidence() -> dict:
    """
    Compute the §5.1.14 N14 sub-node evidence states for inference.

    Note: N14d is NOT computed app-side — it is a BN node with parent N10
    (Judicial Misapplication). Its posterior is computed by VE from N10's
    evidence per §5.1.14 §5 + Q3 routing.

    Returns dict mapping pgmpy node IDs ('14a', '14b', '14c') to binary states.
    Suitable for passing as hard evidence to query_do_risk.
    """
    record = st.session_state.get("criminal_record", []) or []
    return {
        "14a": _n14a_signal(),
        "14b": _n14b_signal(record),
        "14c": _n14c_signal(record),
        # 14d intentionally omitted — derived from N10 via BN inference per Q3
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5.1.15 N15 sub-node signal computation
#
# Per JP confirmation Q6 (α): all four signals derived from existing criminal
# record fields + counsel attestation overrides. No schema changes.
#
# §5.1.15 §5 parents (binary collapse per Q3):
#   N15a — Tariff jurisdiction (1=High-tariff per Doob/Cesaroni/Roach lineage)
#   N15b — Tariff-sensitive offence (1=Property/Drug/AOJ; 0=Violent/Sexual)
#   N15c — Tariff-sensitive sentence length (offence-conditional threshold)
#   N15d — Doctrine absent (1=no jurisprudence applied; shares n14c attestation)
# ─────────────────────────────────────────────────────────────────────────────

# Tariff-sensitive offence patterns (high tariff variability — sentence length
# more likely to reflect regional norms than conduct severity). Property and
# drug offences are the classic tariff-sensitive categories per §5.1.15 §2(d).
_N15_TARIFF_SENSITIVE_PATTERNS = (
    # Property
    "theft", "fraud", "break and enter", "break-and-enter",
    "possession of stolen", "stolen property", "mischief",
    # Drug
    "trafficking", "possession for purpose", "possession for the purpose",
    "production of", "import", "drug", "controlled substance",
    # Administration of justice
    "breach", "fail to comply", "fail to appear",
)

# Conduct-driven offence patterns (less tariff-sensitive — sentence length
# more reflects conduct severity than regional variation).
_N15_CONDUCT_DRIVEN_PATTERNS = (
    "assault", "robbery", "manslaughter", "murder", "homicide",
    "sexual assault", "sexual interference", "sexual exploitation",
    "invitation to sexual", "child pornography",
    "kidnapping", "forcible confinement", "extortion",
    "discharging firearm", "use of firearm",
)

# High-tariff jurisdictions per Doob/Cesaroni/Roach lineage.
# Includes both province names and standard abbreviations to match
# whatever format counsel enters in case_jur. Conservative classification —
# counsel attestation overrides for case-specific nuance.
_N15_HIGH_TARIFF_JURISDICTIONS = (
    "ontario", "on ", " on,", "(on)",
    "manitoba", "mb ", " mb,", "(mb)",
    "saskatchewan", "sk ", " sk,", "(sk)",
    "alberta", "ab ", " ab,", "(ab)",
)


def _n15a_signal(case_jur: str) -> int:
    """
    N15a — Tariff jurisdiction disparity.

    Per Q6 (α): tier-classify case_jur against high-tariff jurisdiction set.
    State 1 (High-tariff) when:
      - Counsel attestation present, OR
      - case_jur contains any high-tariff jurisdiction marker

    Returns: 0 (Low-tariff) or 1 (High-tariff)
    """
    if _any_conviction_attests("n15a_attestation"):
        return 1
    if not case_jur:
        return 0
    case_lower = case_jur.lower()
    return 1 if any(j in case_lower for j in _N15_HIGH_TARIFF_JURISDICTIONS) else 0


def _n15b_signal(criminal_record) -> int:
    """
    N15b — Tariff-sensitive offence type.

    Per Q6 (α): pattern-match offence text against tariff-sensitive categories.
    State 1 (tariff-sensitive offence present) when:
      - Counsel attestation present, OR
      - Record contains at least one offence matching tariff-sensitive patterns
        and NO conduct-driven pattern dominates that conviction

    For binary collapse: an offence is tariff-sensitive if it matches a
    tariff-sensitive pattern AND does not also match a conduct-driven pattern
    (e.g., "drug-related assault" would match both — conduct-driven dominates).

    Returns: 0 (no tariff-sensitive offences) or 1 (tariff-sensitive present)
    """
    if _any_conviction_attests("n15b_attestation"):
        return 1
    if not criminal_record:
        return 0
    for entry in criminal_record:
        offence = (entry.get("offence", "") or "").lower()
        is_tariff_sensitive = any(p in offence for p in _N15_TARIFF_SENSITIVE_PATTERNS)
        is_conduct_driven = any(p in offence for p in _N15_CONDUCT_DRIVEN_PATTERNS)
        if is_tariff_sensitive and not is_conduct_driven:
            return 1
    return 0


def _n15c_signal(criminal_record) -> int:
    """
    N15c — Tariff-sensitive sentence length.

    Per Q6 (α): offence-conditional threshold approach using sentence_type
    categorical field as proxy for sentence length.

    §5.1.15 §7 threshold structure (from illustrative CPT):
      Tariff-sensitive offences (Property/Drug):  > 1 year is concerning
      Conduct-driven offences (Violent/Sexual):   > 3 years is concerning

    Sentence type mapping (categorical proxy for length):
      Federal custody (2+ years)        → > 2 years (always exceeds 1-year
                                          threshold; sometimes exceeds 3-year)
      Provincial custody (< 2 years)    → 0 days to ~24 months (ambiguous)
      Conditional / probation / fine    → effectively 0 custody
      Time served                       → variable, default short

    Conservative implementation: only Federal custody triggers state 1
    automatically. Counsel attestation overrides for cases where Provincial
    custody exceeded the offence-conditional threshold (e.g., 18-month
    sentence for property offence).

    Returns: 0 (sentence below threshold) or 1 (long sentence, tariff-relevant)
    """
    if _any_conviction_attests("n15c_attestation"):
        return 1
    if not criminal_record:
        return 0
    for entry in criminal_record:
        sent_type = entry.get("sentence_type", "") or ""
        if "Federal custody" in sent_type:
            # 2+ years exceeds the 1-year tariff-sensitive threshold automatically
            return 1
    return 0


def _n15d_signal() -> int:
    """
    N15d — Jurisprudential compliance absent.

    Per Q6 (α) + Q3 binary collapse: state 1 (doctrine absent) by default;
    state 0 when counsel attests SCE/Tetrad applied at original sentencing.

    Shares n14c_sce_applied_attestation. Doctrinal rationale: the same
    sentencing event either applied Tetrad/SCE jurisprudence or did not.
    Distinguishing temporal-application (N14c) from tariff-application (N15d)
    is artificial — Gladue, Ewert, Morris, Ellis all bear on both dimensions
    when applied. Single attestation captures both.

    Per Q3 binary collapse: G+E (binding) and M+E (persuasive) both map to
    state 0 — any jurisprudence applied = doctrine present. Only "None
    applied" maps to state 1.

    Returns: 0 (jurisprudence applied) or 1 (no jurisprudence applied)
    """
    if _any_conviction_attests("n14c_sce_applied"):
        return 0
    return 1


def _compute_n15_evidence() -> dict:
    """
    Compute the §5.1.15 N15 sub-node evidence states for inference.

    Returns dict mapping pgmpy node IDs ('15a', '15b', '15c', '15d') to
    binary states. Suitable for passing as hard evidence to query_do_risk.

    Note: N15 itself (the node) is NOT directly fed evidence here — its
    posterior is computed by VE from the four sub-node states + N14 posterior
    (per Q4 (α): N14→N15 structural edge at node level).
    """
    case_jur = st.session_state.get("case_jur", "") or ""
    record = st.session_state.get("criminal_record", []) or []
    return {
        "15a": _n15a_signal(case_jur),
        "15b": _n15b_signal(record),
        "15c": _n15c_signal(record),
        "15d": _n15d_signal(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5.1.18 N18 sub-node signal computation
#
# Per JP confirmations Q1=β + Q7=α: per-conviction SCE-integration tags
# (Full / Partial / Nominal / Absent) drive aggregate sub-node signals;
# counsel attestations override pattern-matched defaults.
#
# Per-conviction tag semantics (Morris Heuristic Audit-aligned):
#   Full     — SCE substantively integrated into reasons for sentence
#   Partial  — SCE referenced and partially integrated
#   Nominal  — SCE mentioned but not substantively engaged
#   Absent   — SCE not mentioned (default — most conservative)
#
# Sub-node binary mapping:
#   N18b (SCE Presence): "Full"/"Partial"/"Nominal" → 0; "Absent" → 1
#   N18c (SCE Substance): "Full"/"Partial" → 0; "Nominal"/"Absent" → 1
#   The N18b/N18c split preserves the Morris insight that nominal-only
#   mention without substantive integration is a distinct failure mode
#   from outright absence.
# ─────────────────────────────────────────────────────────────────────────────

# Per-conviction SCE-integration tag values (Q1=β)
_N18_TAG_VALUES = ("Full", "Partial", "Nominal", "Absent")
_N18_TAG_DEFAULT = "Absent"  # most conservative default

# Tags that count as "SCE substantively present" for N18b
_N18_PRESENT_TAGS = ("Full", "Partial", "Nominal")
# Tags that count as "SCE substantively integrated" for N18c
_N18_SUBSTANTIVE_TAGS = ("Full", "Partial")

# Jurisdictions with strong provincial appellate SCE-integration scrutiny.
# ON post-Morris (2021 ONCA 680), BC post-Ellis (2022 BCCA 278), and SCC
# nationwide cases. All others default to inflation risk (state 1).
_N18_SCE_INTEGRATED_JURISDICTIONS = (
    "ontario", "on ", " on,", "(on)",  # Morris ONCA
    "british columbia", "bc ", " bc,", "(bc)",  # Ellis BCCA
    "supreme court of canada", "scc ", " scc,", "(scc)",  # SCC
)


def _n18a_signal(case_jur: str) -> int:
    """
    N18a — Jurisdiction SCE-integration sensitivity.

    Per Q7 (α): tier-classify case_jur against jurisdictions with strong
    provincial appellate SCE-integration scrutiny.
    State 1 (inflation risk — no Morris/Ellis precedent) when:
      - Counsel attestation present, OR
      - case_jur does NOT contain any SCE-integrated jurisdiction marker

    Returns: 0 (Morris/Ellis/SCC jurisdiction) or 1 (no strong precedent)
    """
    if st.session_state.get("n18a_counsel_attestation", False):
        return 1
    if not case_jur:
        return 1  # no jurisdiction info → conservative inflation risk
    case_lower = case_jur.lower()
    sce_integrated = any(j in case_lower for j in _N18_SCE_INTEGRATED_JURISDICTIONS)
    return 0 if sce_integrated else 1


def _n18b_signal(criminal_record) -> int:
    """
    N18b — SCE Presence in Reasons (aggregate).

    Per Q1 (β) + Q7 (α): aggregate over per-conviction sce_integration_tag.
    State 1 (SCE absent in at least one conviction) when:
      - Counsel attestation present, OR
      - Any conviction has tag "Absent" (no SCE reference in reasons)

    Returns: 0 (all convictions have at least nominal SCE presence)
             1 (at least one conviction has SCE absent)
    """
    if _any_conviction_attests("n18b_attestation"):
        return 1
    if not criminal_record:
        return 0  # empty record → no inflation signal
    for entry in criminal_record:
        tag = entry.get("sce_integration_tag", _N18_TAG_DEFAULT)
        if tag not in _N18_PRESENT_TAGS:
            return 1
    return 0


def _n18c_signal(criminal_record) -> int:
    """
    N18c — SCE Substance.

    Per Q1 (β) + Q7 (α) + Morris Heuristic Audit: aggregate over
    per-conviction sce_integration_tag. State 1 (substance absent) when:
      - Counsel attestation present, OR
      - Any conviction has tag "Nominal" or "Absent" (no substantive
        integration even if mentioned)

    This captures the Morris Audit insight that nominal Gladue mention
    without substantive integration is the most common failure mode.

    Returns: 0 (all convictions Full/Partial — substantive integration)
             1 (at least one conviction Nominal or Absent)
    """
    if _any_conviction_attests("n18c_attestation"):
        return 1
    if not criminal_record:
        return 0
    for entry in criminal_record:
        tag = entry.get("sce_integration_tag", _N18_TAG_DEFAULT)
        if tag not in _N18_SUBSTANTIVE_TAGS:
            return 1
    return 0


def _compute_n18_evidence() -> dict:
    """
    Compute the §5.1.18 N18 sub-node evidence states for inference.

    Returns dict mapping pgmpy node IDs ('18a', '18b', '18c') to binary
    states. Suitable for passing as hard evidence to query_do_risk.

    Note: N18d (Doctrinal Tagging compliance) is NOT computed app-side —
    it is a BN node with parent N10 (Misapplication). Its posterior is
    computed by Variable Elimination from N10's evidence per Q5 (α).
    N18 itself is also NOT fed evidence — its posterior comes from VE
    over the six parents (N12, N14, 18a, 18b, 18c, 18d).
    """
    case_jur = st.session_state.get("case_jur", "") or ""
    record = st.session_state.get("criminal_record", []) or []
    return {
        "18a": _n18a_signal(case_jur),
        "18b": _n18b_signal(record),
        "18c": _n18c_signal(record),
        # 18d intentionally omitted — derived from N10 via BN inference
    }


# ─────────────────────────────────────────────────────────────────────────────
# §5.1.19 N19 collider-bias signal computation (Q6=α, app-side complement)
#
# Per JP confirmation Q6=α: pattern-detection helper independent of the BN
# posterior. The BN posterior P(N19=High) is computed by Variable Elimination
# from the (N14, N17) parents per §5.1.19 §6 illustrative CPT. This helper
# adds an INDEPENDENT app-side signal that fires when both upstream variables
# are jointly elevated above 0.60 — making the structural condition visible
# without requiring the user to interpret the probabilistic posterior.
#
# Per §5.1.19 §1, this signal does not "add evidence" to inference. It
# surfaces the structural condition under which the inference drawn from
# the criminal record is subject to collider bias. The threshold of 0.60
# is calibrated to the §6 illustrative CPT: at both parents = 0.60, the
# joint adverse condition is well above the (Low, Low) baseline column
# while still permitting the user to identify the elevated structural
# condition before either parent reaches saturation.
# ─────────────────────────────────────────────────────────────────────────────

# Joint-elevation threshold for app-side collider signal
_N19_JOINT_THRESHOLD = 0.60


# ── §5.1.1 N1 burden-of-proof audit helpers (Mark 8) ─────────────────────────
# See model.py § "N1 burden-of-proof audit" for full doctrinal architecture.
# These app-side helpers manage audit-record lifecycle, derive per-conviction
# audit IDs, and support the §RM.1 register on the Report tab.

# Citation registry — surfaced in prompts and the §RM.1 register.
N1_CITATIONS = {
    "gardiner":  "R. v. Gardiner, [1982] 2 SCR 368",
    "s724":      "Criminal Code, RSC 1985, c. C-46, s. 724(3)",
    "ferguson":  "R. v. Ferguson, 2008 SCC 6",
    "angelillo": "R. v. Angelillo, 2006 SCC 55",
    "lacasse":   "R. v. Lacasse, 2015 SCC 64 (contextual; sentencing-stage proof)",
}

# Common attestation-basis options for the dropdown shortcuts.
ATTESTATION_BASES = [
    "— Select basis —",
    "Admitted by accused",
    "Proven at trial",
    "Agreed in PSR",
    "Judicial finding (Ferguson — necessarily implied by verdict)",
    "Judicial finding (sentencing judge's own finding on trial record)",
    "Uncontested in plea agreement",
    "Other (specify in attestation text)",
]


def _conviction_audit_id(idx: int, year, offence: str) -> str:
    """
    Stable input ID for a conviction in the criminal_record list.
    Format: "record_{idx}_{year}" — index disambiguates same-year entries.
    Year-only suffix is robust to the chronological resort that happens on
    every entry-add (the index repositions but the conviction's audit
    record can be re-keyed via _sync_conviction_audit_ids if needed).
    """
    return f"record_{idx}_{year}"


def _sync_conviction_audit_ids():
    """
    Reconcile session_state.n1_audit with current criminal_record.

    Called after any add/remove/sort operation on criminal_record. Removes
    audit records whose corresponding conviction no longer exists (orphan
    cleanup) and re-keys surviving records to match the new indices after
    chronological resort.

    Each conviction stores its audit-id key under entry["audit_id"] so we
    can track records across re-sorts. New convictions are assigned an
    audit_id at creation if missing.
    """
    rec = st.session_state.get("criminal_record", []) or []
    audit = st.session_state.get("n1_audit", {})

    # Stage 1: ensure every conviction has an audit_id
    for idx, entry in enumerate(rec):
        if "audit_id" not in entry:
            entry["audit_id"] = _conviction_audit_id(
                idx, entry.get("year", 0), entry.get("offence", "")
            )

    # Stage 2: remove orphaned records (audit records with no matching
    # conviction). Non-record audits (e.g. from other tabs) preserved.
    live_ids = {e.get("audit_id") for e in rec if "audit_id" in e}
    for key in list(audit.keys()):
        if key.startswith("record_") and key not in live_ids:
            audit.pop(key, None)

    st.session_state.n1_audit = audit


def _audit_record_for_conviction(entry: dict, idx: int) -> dict:
    """
    Build or retrieve the audit record for a conviction.

    For prior convictions at present sentencing, the doctrinally relevant
    audit is Crown-reliance per Angelillo: is the Crown relying on this
    prior as aggravating context, and if so, has BARD been met for the
    aggravating use? Mere existence of the conviction is res judicata —
    not subject to fresh burden audit at present sentencing (subject to
    the prior-evidence audit carve-out under R. v. Bird 2019 SCC 7,
    which is a Mark 9 build item per JP scoping).

    Default record (when conviction is first audited): provenance="judicial",
    use="contextual" (i.e. Crown is NOT yet relying on it as aggravating).
    The user toggles "Crown relies as aggravating" to flip use→aggravating
    and trigger the BARD audit.
    """
    audit = st.session_state.setdefault("n1_audit", {})
    aid = entry.get("audit_id")
    if not aid:
        aid = _conviction_audit_id(idx, entry.get("year", 0),
                                   entry.get("offence", ""))
        entry["audit_id"] = aid

    if aid not in audit:
        # Default: prior conviction not yet relied on as aggravating.
        # use="contextual" means no audit triggered; user toggles to
        # change to "aggravating" which triggers the BARD audit.
        audit[aid] = {
            "tab": "criminal_record",
            "label": (f"{entry.get('year', '?')} — "
                      f"{(entry.get('offence', '') or 'Unknown offence')[:60]}"),
            "provenance": "judicial",
            "use": "contextual",
            "judicial_finding_type": None,
            "applicable_burden": "none",
            "attestation": "",
            "attestation_status": "pending",
            "citations": [N1_CITATIONS["angelillo"], N1_CITATIONS["gardiner"]],
            # Prior-evidence audit placeholder — Mark 9 build item.
            # Per JP lock-in: visible-but-inert. Render as "[awaiting
            # implementation]" in the UI; does not affect N1 derivation.
            "prior_evidence_audit_status": "not_yet_conducted",
        }
    return audit[aid]


def _set_conviction_audit_use(aid: str, crown_relies_aggravating: bool):
    """
    Toggle a conviction's audit use between "contextual" (Crown not relying
    as aggravating; no audit triggered) and "aggravating" (Crown relying as
    aggravating; BARD audit triggered per Angelillo).

    Provenance flips correspondingly: when Crown is relying, provenance
    becomes "crown" (the Crown is now actively tendering this fact for
    aggravating use). When not relying, provenance reverts to "judicial"
    (the prior conviction is just historical record, not contested).
    """
    audit = st.session_state.get("n1_audit", {})
    if aid not in audit:
        return
    if crown_relies_aggravating:
        audit[aid]["provenance"] = "crown"
        audit[aid]["use"] = "aggravating"
        audit[aid]["applicable_burden"] = "BARD"
    else:
        audit[aid]["provenance"] = "judicial"
        audit[aid]["use"] = "contextual"
        audit[aid]["applicable_burden"] = "none"
        # Clear pending attestation status when reverting — fresh audit
        # if Crown re-asserts reliance later.
        if audit[aid].get("attestation_status") == "pending":
            audit[aid]["attestation"] = ""
    st.session_state.n1_audit = audit


def _audit_status_for_conviction(aid: str) -> tuple:
    """
    Return (icon, label, color) for a conviction's audit status indicator.
    Used by the Criminal Record tab to render per-conviction audit chips.
    """
    audit = st.session_state.get("n1_audit", {})
    rec = audit.get(aid)
    if not rec:
        return ("·", "Not audited", "#9E9E9E")
    use = rec.get("use", "contextual")
    if use in ("contextual", "agreed_fact"):
        return ("·", "No audit required", "#9E9E9E")
    status = rec.get("attestation_status", "pending")
    if status == "satisfied":
        return ("✓", "Burden attested", "#3B6D11")
    if status == "insufficient":
        return ("✗", "Burden insufficient", "#A32D2D")
    return ("⚠", "Attestation pending", "#BA7517")


# ── §5.1.1 Default-classification helpers (Mark 8 Phase 2) ─────────────────────
# Pattern A defaults per tab. Each evidentiary input receives a default
# classification at entry — the user only encounters the audit UI when
# (a) the default is wrong for a specific input and they override, or
# (b) strict mode is on and at-tab-exit review surfaces every classification
# for explicit confirmation. See JP M8/P2 lock-in.

# Tab-level default mapping. None means "no default — prompt the user."
_TAB_DEFAULT_CLASSIFICATION = {
    "intake":          None,  # always prompt (free-form fact entry)
    "gladue":          {"provenance": "defence", "use": "mitigating"},
    "sce":             {"provenance": "defence", "use": "mitigating"},
    "risk_substantive": {"provenance": "crown",  "use": "aggravating"},  # N2/N3/N4
    "risk_distortion":  {"provenance": "defence", "use": "mitigating"},  # N5–N8
    "risk_mitigation":  {"provenance": "defence", "use": "mitigating"},  # N9 IGT
    "risk_other":      None,  # N11/N13/N16 — prompt
    "criminal_record": {"provenance": "judicial", "use": "contextual"},  # default
    # Profile inputs are case-context parameters, not evidentiary claims —
    # no audit applied at Profile level (per JP M8/P2 lock-in).
}

# Burden derivation from provenance × use (per Gardiner asymmetry).
def _applicable_burden(provenance: str, use: str,
                        judicial_finding_type: str = None) -> str:
    """
    Return the doctrinally applicable burden for an audit record.
    See model.compute_n1_prior_from_audit for the full cross-product.
    """
    if provenance == "crown" and use == "aggravating":
        return "BARD"
    if provenance == "defence" and use == "mitigating":
        return "BoP"
    if provenance == "judicial" and judicial_finding_type == "found_by_sentencing_judge":
        if use == "aggravating":
            return "BARD"
        if use == "mitigating":
            return "BoP"
    return "none"


def _default_audit_record(tab: str, label: str,
                          input_id: str = None,
                          override_provenance: str = None,
                          override_use: str = None) -> dict:
    """
    Build a default audit record for an evidentiary input.

    tab: one of the keys in _TAB_DEFAULT_CLASSIFICATION
    label: display description for the §RM.1 register
    override_provenance/override_use: explicit values that override defaults

    Returns a dict ready to insert into ss["n1_audit"].
    """
    defaults = _TAB_DEFAULT_CLASSIFICATION.get(tab) or {}
    provenance = override_provenance or defaults.get("provenance") or "unspecified"
    use = override_use or defaults.get("use") or "unspecified"
    burden = _applicable_burden(provenance, use)

    # Initial status: if both provenance and use have defaults (and aren't
    # "unspecified"), seed as "pending" so the user can attest. If anything
    # is unspecified (Pattern A no-default tabs), keep as "pending" but the
    # UI flags it as "needs classification" rather than "needs attestation."
    citations = [N1_CITATIONS["gardiner"], N1_CITATIONS["s724"]]
    if tab == "criminal_record":
        citations.append(N1_CITATIONS["angelillo"])

    return {
        "tab": tab,
        "label": label,
        "provenance": provenance,
        "use": use,
        "judicial_finding_type": None,
        "applicable_burden": burden,
        "attestation": "",
        "attestation_basis": ATTESTATION_BASES[0],
        "attestation_status": "pending",
        "citations": citations,
        "is_default_classified": True,  # flag for UI: classification came from defaults
    }


def _ensure_audit_record(input_id: str, tab: str, label: str,
                          provenance: str = None, use: str = None) -> dict:
    """
    Get or create the audit record for an arbitrary evidentiary input.

    Used by per-tab integration code (Intake / Gladue / SCE / Risk &
    Distortions). For Criminal Record, see _audit_record_for_conviction.
    """
    audit = st.session_state.setdefault("n1_audit", {})
    if input_id not in audit:
        audit[input_id] = _default_audit_record(
            tab, label,
            input_id=input_id,
            override_provenance=provenance,
            override_use=use,
        )
    return audit[input_id]


def _remove_audit_record(input_id: str):
    """Remove an audit record (called when an input is deleted/unselected)."""
    audit = st.session_state.get("n1_audit", {})
    audit.pop(input_id, None)
    st.session_state.n1_audit = audit


def _classification_chip(audit_rec: dict) -> str:
    """
    Build an HTML chip showing an audit record's classification state.

    Chip format: [provenance · use · burden] [status icon]
    Pattern (b) per JP lock-in: visible classification on every input so
    override is easy. Strict mode triggers tab-exit review where these
    chips become editable for bulk reclassification.
    """
    provenance = audit_rec.get("provenance", "unspecified")
    use = audit_rec.get("use", "unspecified")
    burden = audit_rec.get("applicable_burden", "none")
    status = audit_rec.get("attestation_status", "pending")

    if use in ("contextual", "agreed_fact") or burden == "none":
        # No audit triggered — display as informational only
        return (
            "<span style='display:inline-block;font-family:JetBrains Mono,monospace;"
            "font-size:0.68rem;color:#9E9E9E;background:#F5F4F0;border:1px solid "
            "#E5E2DA;border-radius:3px;padding:2px 6px;margin-left:6px'>"
            f"{provenance} · {use}</span>"
        )

    # Status colours
    if status == "satisfied":
        bg, fg, border = "#F4F8EE", "#3B6D11", "#C5D7AC"
        icon = "✓"
    elif status == "insufficient":
        bg, fg, border = "#FCEBEB", "#A32D2D", "#E5B5B5"
        icon = "✗"
    else:
        bg, fg, border = "#FAEEDA", "#7A4F0E", "#E5CC95"
        icon = "⚠"

    return (
        "<span style='display:inline-block;font-family:JetBrains Mono,monospace;"
        f"font-size:0.68rem;color:{fg};background:{bg};border:1px solid {border};"
        "border-radius:3px;padding:2px 6px;margin-left:6px;font-weight:500'>"
        f"{icon} {provenance} · {use} · {burden}</span>"
    )


# ── §5.1.1 N1 doctrinal-state framing (Mark 8 Phase 3 — display reframe) ──
# Per JP M8/P3 lock-in: N1 is not a probability — it is the operationalisation
# of a doctrinal posture toward the case file's audit state. Displaying N1 as
# a continuous percentage invites probabilistic misreading ("there's an 83%
# chance the burden is met"). The doctrinal truth is the inverse: PARVIS
# operates on the doctrinal assumption that burdens are met (default posture),
# fluctuates downward only when the user records explicit "insufficient"
# findings, and reaches the floor only on catastrophic audit failure.
#
# Three named doctrinal states replace the continuous percentage in user-
# facing surfaces. The numerical value is preserved as a depth indicator
# below the state label — Option II per JP lock-in.

# Doctrinal-state thresholds. See compute_n1_prior_from_audit for the
# audit-state inputs that map to these states.
_N1_STATE_DEFAULT     = "default"
_N1_STATE_PRESSURE    = "pressure"
_N1_STATE_FAILURE     = "failure"

# Display configuration per state.
_N1_STATE_DISPLAY = {
    _N1_STATE_DEFAULT: {
        "label":         "Doctrinal Default",
        "subtitle":      "Audit pressure: baseline.",
        "color_fg":      "#3B6D11",
        "color_bg":      "#F4F8EE",
        "color_border":  "#C5D7AC",
        "color_accent":  "#3B6D11",
    },
    _N1_STATE_PRESSURE: {
        "label":         "Doctrinal Pressure",
        "subtitle":      "Audit pressure: at least one input marked insufficient.",
        "color_fg":      "#7A4F0E",
        "color_bg":      "#FAEEDA",
        "color_border":  "#E5CC95",
        "color_accent":  "#BA7517",
    },
    _N1_STATE_FAILURE: {
        "label":         "Doctrinal Failure",
        "subtitle":      "Audit pressure: catastrophic — all audited inputs insufficient.",
        "color_fg":      "#A32D2D",
        "color_bg":      "#FCEBEB",
        "color_border":  "#E5B5B5",
        "color_accent":  "#A32D2D",
    },
}


def _n1_doctrinal_state(audit_state: dict = None) -> str:
    """
    Classify the current N1 audit state into one of three doctrinal postures
    per the Mark 8 Phase 3 framing lock-in.

    Doctrinal-state thresholds (JP M8/P3 calls 1 & 2):
      - Default:  zero audited inputs marked insufficient
      - Pressure: at least one input marked insufficient (but not all)
      - Failure:  ALL audited inputs marked insufficient

    "Pending" attestations are neutral per Mark 8 hotfix #3 — they do not
    contribute to the audit-failure proportion. Only "insufficient" findings
    move N1 off the default posture.

    Inputs with use ∈ {contextual, agreed_fact} or burden=none are excluded
    from the audited-input population (they fall outside the audit's scope
    per Gardiner's asymmetry).
    """
    if audit_state is None:
        audit_state = st.session_state.get("n1_audit", {})

    if not audit_state:
        return _N1_STATE_DEFAULT

    audited_count = 0
    insufficient_count = 0

    for record in audit_state.values():
        provenance = record.get("provenance")
        use = record.get("use")
        burden = record.get("applicable_burden", "none")
        status = record.get("attestation_status", "pending")
        judicial_type = record.get("judicial_finding_type")

        # Determine if this input requires a burden audit (matches the
        # cross-product in compute_n1_prior_from_audit).
        burden_required = False
        if provenance == "crown" and use == "aggravating":
            burden_required = True
        elif provenance == "defence" and use == "mitigating":
            burden_required = True
        elif (provenance == "judicial"
              and judicial_type == "found_by_sentencing_judge"
              and use in ("aggravating", "mitigating")):
            burden_required = True

        if not burden_required:
            continue

        audited_count += 1
        if status == "insufficient":
            insufficient_count += 1

    if audited_count == 0:
        return _N1_STATE_DEFAULT
    if insufficient_count == 0:
        return _N1_STATE_DEFAULT
    if insufficient_count == audited_count:
        return _N1_STATE_FAILURE
    return _N1_STATE_PRESSURE


def _n1_audit_summary(audit_state: dict = None) -> dict:
    """
    Build a summary record of the N1 audit state for display purposes.

    Returns a dict with:
      state:              one of _N1_STATE_DEFAULT/_PRESSURE/_FAILURE
      label:              display label for the state
      subtitle:           one-line description of audit pressure
      color_fg/bg/border/accent:  display colours per state
      audited_count:      number of audited (burden-required) inputs
      insufficient_count: number of those marked insufficient
      pending_count:      number of those still pending attestation
      satisfied_count:    number of those with attestation satisfied
    """
    if audit_state is None:
        audit_state = st.session_state.get("n1_audit", {})

    audited_count = 0
    insufficient_count = 0
    pending_count = 0
    satisfied_count = 0

    for record in audit_state.values():
        provenance = record.get("provenance")
        use = record.get("use")
        status = record.get("attestation_status", "pending")
        judicial_type = record.get("judicial_finding_type")

        burden_required = False
        if provenance == "crown" and use == "aggravating":
            burden_required = True
        elif provenance == "defence" and use == "mitigating":
            burden_required = True
        elif (provenance == "judicial"
              and judicial_type == "found_by_sentencing_judge"
              and use in ("aggravating", "mitigating")):
            burden_required = True

        if not burden_required:
            continue

        audited_count += 1
        if status == "insufficient":
            insufficient_count += 1
        elif status == "satisfied":
            satisfied_count += 1
        else:
            pending_count += 1

    state = _n1_doctrinal_state(audit_state)
    display = _N1_STATE_DISPLAY[state]

    return {
        "state":              state,
        "label":              display["label"],
        "subtitle":           display["subtitle"],
        "color_fg":           display["color_fg"],
        "color_bg":           display["color_bg"],
        "color_border":       display["color_border"],
        "color_accent":       display["color_accent"],
        "audited_count":      audited_count,
        "insufficient_count": insufficient_count,
        "pending_count":      pending_count,
        "satisfied_count":    satisfied_count,
    }


def _n1_state_chip_html(audit_state: dict = None,
                         include_depth: bool = True,
                         compact: bool = False) -> str:
    """
    Render an HTML chip showing the N1 doctrinal state.

    include_depth: if True, append the live N1 posterior as a small depth
    indicator below the state label (Option II per JP lock-in: state label
    dominant, percentage subordinate).

    compact: if True, render a single-line chip suitable for diagnostic
    grids (Inference tab). If False, render the multi-line card-style
    block suitable for the Doctrinal architecture / Architecture tab /
    Report tab register.
    """
    summary = _n1_audit_summary(audit_state)
    posterior = float(st.session_state.get("posteriors", {}).get(1, 0.83))

    if compact:
        # Inference-tab grid chip: single line, label + small percentage
        return (
            f"<div style='font-family:JetBrains Mono,monospace;"
            f"font-size:0.62rem;font-weight:600;color:{summary['color_fg']};"
            f"text-transform:uppercase;letter-spacing:0.04em;line-height:1.2'>"
            f"{summary['label']}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;"
            f"font-size:0.78rem;color:{summary['color_accent']};"
            f"margin-top:2px;font-weight:500'>"
            f"{posterior*100:.1f}%</div>"
        )

    # Default multi-line card-style block.
    detail_line = ""
    if summary["audited_count"] > 0:
        parts = []
        if summary["satisfied_count"]:
            parts.append(f"{summary['satisfied_count']} satisfied")
        if summary["pending_count"]:
            parts.append(f"{summary['pending_count']} pending")
        if summary["insufficient_count"]:
            parts.append(f"{summary['insufficient_count']} insufficient")
        detail_line = (
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
            f"{summary['audited_count']} audited input(s) — "
            f"{', '.join(parts)}.</div>"
        )

    depth_html = ""
    if include_depth:
        depth_html = (
            f"<div style='display:flex;align-items:center;gap:8px;margin-top:8px'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.66rem;"
            f"color:#9E9E9E;text-transform:uppercase;letter-spacing:0.06em'>"
            f"Audit-pressure depth</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.84rem;"
            f"font-weight:600;color:{summary['color_accent']}'>"
            f"{posterior*100:.1f}%</div>"
            f"<div style='font-family:Fraunces,serif;font-style:italic;"
            f"font-size:0.7rem;color:#9E9E9E'>"
            f"(operationalisation of doctrinal posture, not a probability)</div>"
            f"</div>"
        )

    return (
        f"<div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
        f"font-weight:700;color:{summary['color_fg']};text-transform:uppercase;"
        f"letter-spacing:0.06em;margin-bottom:4px'>"
        f"Doctrinal posture</div>"
        f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.15rem;"
        f"font-weight:500;color:{summary['color_fg']};line-height:1.2'>"
        f"{summary['label']}</div>"
        f"<div style='font-family:Fraunces,serif;font-style:italic;"
        f"font-size:0.84rem;color:#3a3a3a;margin-top:6px;line-height:1.5'>"
        f"{summary['subtitle']}</div>"
        f"{detail_line}"
        f"{depth_html}"
        f"</div>"
    )


# ── Per-conviction attestation aggregators (Mark 8 Phase 2) ───────────────
# After moving N14/N15/N18 attestations from Profile to per-conviction on
# Criminal Record, the signal-computation helpers (_compute_n14_evidence
# etc.) need to read aggregated values rather than single case-level flags.
# Per JP "any-prior" aggregation rule (M8/P2): aggregate fires if any one
# conviction has the per-conviction attestation set.

def _any_conviction_attests(field: str) -> bool:
    """
    "Any-prior" aggregation: True iff any conviction in criminal_record
    has the named attestation field set to True.

    Used by _compute_n14_evidence / _compute_n15_evidence /
    _compute_n18_evidence to derive aggregate audit signals from per-
    conviction attestations after the Profile-to-Criminal-Record
    relocation. See JP M8/P2 lock-in.
    """
    rec = st.session_state.get("criminal_record", []) or []
    return any(e.get(field, False) for e in rec)


def _strict_mode_review_panel(tab_filter: str = None):
    """
    Strict-mode at-tab-exit review (Mark 8 Phase 2 — Pattern B trigger).

    When st.session_state.strict_mode is True, render an at-tab-exit review
    panel listing audited inputs originating from the named tab (or all tabs
    if tab_filter is None) and surfacing pending attestations or unspecified
    classifications. The user can confirm or override classifications in
    bulk before moving on.

    Per JP M8/P2 lock-in (Pattern A defaults + strict mode triggering Pattern
    B at-tab-exit review): in standard mode this panel is a no-op (defaults
    are trusted, attestations can be deferred to Report-tab review). In
    strict mode, surfaces every classification for explicit confirmation.

    tab_filter: tab key to filter by ("intake", "gladue", "sce",
    "criminal_record", "risk_*", etc.). None = show all.
    """
    if not st.session_state.get("strict_mode", False):
        return  # Not in strict mode; standard Pattern A behaviour applies.

    audit = st.session_state.get("n1_audit", {})
    if not audit:
        return

    # Filter records by tab if requested
    if tab_filter:
        records = {aid: r for aid, r in audit.items()
                   if r.get("tab", "").startswith(tab_filter)}
    else:
        records = audit

    if not records:
        return

    # Identify records needing review: pending attestations OR
    # unspecified classifications.
    needs_review = []
    for aid, r in records.items():
        use = r.get("use", "unspecified")
        status = r.get("attestation_status", "pending")
        if use == "unspecified" or r.get("provenance") == "unspecified":
            needs_review.append((aid, r, "needs_classification"))
        elif use not in ("contextual", "agreed_fact") and status != "satisfied":
            needs_review.append((aid, r, "needs_attestation"))

    if not needs_review:
        return  # Strict mode on but nothing to review on this tab.

    with st.expander(
        f"⚖ Strict-mode review — {len(needs_review)} input(s) "
        f"awaiting classification or attestation",
        expanded=True,
    ):
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#5A5A5A;margin-bottom:14px;line-height:1.55'>"
            "Strict mode is active. Per <em>R. v. Gardiner</em>, [1982] 2 SCR 368 + "
            "s. 724(3) <em>Criminal Code</em>, each input listed below requires "
            "explicit classification (provenance + use) and attestation before "
            "this case file is complete for adversarial handoff. The default "
            "classification per Pattern A is shown; override below where the "
            "default is wrong."
            "</div>",
            unsafe_allow_html=True,
        )

        for aid, r, reason in needs_review:
            label = r.get("label", aid)
            default_chip = _classification_chip(r)
            st.markdown(
                f"<div style='font-size:0.88rem;margin:8px 0 4px 0'>"
                f"<strong>{label}</strong>{default_chip}</div>",
                unsafe_allow_html=True,
            )
            if reason == "needs_classification":
                st.caption(
                    "⚠ Provenance and/or use are unspecified — this input "
                    "is in a no-default tab. Open the Report tab §RM.1 "
                    "register to classify and attest."
                )
            else:
                st.caption(
                    "⚠ Attestation pending. Open the Report tab §RM.1 "
                    "register to record the basis on which the burden was met."
                )


def _n19_collider_signal(posteriors: dict) -> dict:
    """
    §5.1.19 N19 — App-side collider-bias signal.

    Returns a dict describing the structural collider condition independent
    of the BN posterior. The BN posterior on N19 is computed by VE; this
    helper provides a complementary pattern-detection check.

    Returns:
      {
        "active": bool,       # both N14 and N17 ≥ threshold
        "n14": float,         # N14 posterior
        "n17": float,         # N17 posterior
        "n19_post": float,    # N19 posterior from BN (for downstream use)
        "threshold": float,   # _N19_JOINT_THRESHOLD
      }
    """
    n14 = float(posteriors.get(14, 0.5))
    n17 = float(posteriors.get(17, 0.5))
    n19_post = float(posteriors.get(19, 0.30))
    active = (n14 >= _N19_JOINT_THRESHOLD) and (n17 >= _N19_JOINT_THRESHOLD)
    return {
        "active": active,
        "n14": n14,
        "n17": n17,
        "n19_post": n19_post,
        "threshold": _N19_JOINT_THRESHOLD,
    }


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

    # §5.1.17 N17 sub-node evidence — feed N17a/N17b/N17c/N17d states to VE
    # so that N17's posterior is computed against §5.1.17 §7 illustrative CPT.
    # Computed from session state via the four signal helpers above.
    # Manual overrides (st.session_state.manual_ev) take precedence if present.
    n17_ev = _compute_n17_evidence()
    for sub_nid, state in n17_ev.items():
        if sub_nid not in hard_ev:
            hard_ev[sub_nid] = state

    # §5.1.14 N14 sub-node evidence — feed N14a/N14b/N14c states to VE so that
    # N14's posterior is computed against §5.1.14 §7 illustrative CPT.
    # N14d is NOT fed here — it's a BN node with parent N10 per Q3 routing.
    n14_ev = _compute_n14_evidence()
    for sub_nid, state in n14_ev.items():
        if sub_nid not in hard_ev:
            hard_ev[sub_nid] = state

    # §5.1.15 N15 sub-node evidence — feed N15a/b/c/d states to VE so that
    # N15's posterior is computed against §5.1.15 §7 illustrative CPT.
    # N15 itself is NOT fed — it's downstream of N14 and the four sub-nodes
    # via the 5-parent structural edge per Q4 (α).
    n15_ev = _compute_n15_evidence()
    for sub_nid, state in n15_ev.items():
        if sub_nid not in hard_ev:
            hard_ev[sub_nid] = state

    # §5.1.18 N18 sub-node evidence — feed N18a/b/c states to VE so that
    # N18's posterior is computed against §5.1.18 §7 illustrative CPT.
    # N18d is NOT fed (BN-derived from N10 per Q5 (α)); N18 itself is NOT
    # fed — its posterior comes from VE over the 6-parent structure
    # (N12 + N14 + 18a/b/c/d).
    n18_ev = _compute_n18_evidence()
    for sub_nid, state in n18_ev.items():
        if sub_nid not in hard_ev:
            hard_ev[sub_nid] = state

    # ── §5.1.1 N1 burden-of-proof audit target (Mark 8) ───────────────────
    # Compute audit-derived target P(N1=High) from session_state.n1_audit
    # and pass to query_do_risk. The audit-derived value propagates
    # structurally through VE to N1's children (N2, N3, N4, N6, N8) — not
    # just displayed cosmetically. Returns None when audit-state is empty
    # or contains no inputs requiring a burden audit, in which case N1
    # falls back to its static prior [[0.17],[0.83]].
    # See model.py § "N1 burden-of-proof audit" for doctrinal architecture.
    n1_audit_target = compute_n1_prior_from_audit(
        st.session_state.get("n1_audit", {})
    )
    # Only pass target when it differs meaningfully from the default prior
    # (avoid unnecessary virtual-evidence overhead when audit is inactive).
    from model import _N1_DEFAULT_POSTERIOR as _N1_DEF
    if abs(n1_audit_target - _N1_DEF) < 0.001:
        n1_audit_target = None

    post=query_do_risk(st.session_state.engine, hard_ev,
                       n1_audit_target=n1_audit_target)

    # ── §5.1.19 N19 collider-bias secondary computation (Q4=C) ─────────────
    # Per §5.1.19 §1, the headline DO posterior in post[20] is unchanged.
    # Per §5.1.19 §8, a secondary collider-discounted risk is computed for
    # contingent display when the collider structure is active. The
    # discount is multiplicative on the final risk, scaled to N19's
    # posterior — see model.py compute_do_risk(collider_discount=True).
    from model import compute_do_risk as _cdr  # local import for the kwarg path
    _collider_signal = _n19_collider_signal(post)
    _discounted = _cdr(post, collider_discount=True)
    st.session_state["n19_collider_signal"] = _collider_signal
    st.session_state["n19_discounted_risk"] = float(_discounted)

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
    # and age burnout multiplier (N14 — Temporal Distortion per CH5 §5.1.14). All from model.compute_do_risk.
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
    # NOTE: sub/peers/stab widgets retained on Profile tab UI but no longer
    # read here — their stale-taxonomy mapping to pev[18] (BUG 1, Mark 8)
    # was clobbering the BN-computed N18 (SCE Profile audit) posterior.
    # Awaiting doctrinal-mapping review per CH5 canonical taxonomy before
    # rewiring these inputs (likely candidates: folded into N9/IGT or N18
    # sub-node signals; see handoff brief Mark 8 § BUG 1).
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
    # BUG 1 FIX (Mark 8): stale-taxonomy override of pev[18] (substance/peer/
    # stability composite) and pev[19] (rehab+programming composite) deleted.
    # Under CH5 canonical taxonomy, N18 = SCE Profile audit (Gladue/Ewert/
    # Morris/Ellis) and N19 = Collider Bias — neither is computed from these
    # widget inputs. The override loop at ~L1710 was silently clobbering the
    # BN-computed posteriors for both nodes before display, invalidating the
    # §5.1.18 §7 Morris audit anchors and §5.1.19 §6 collider mechanism.
    # Awaiting doctrinal-mapping review for substance/peer/stability inputs.
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
    20:(.50,.03),
    # ── §5.1.17 sub-nodes (cluster around N17) ──
    "17a":(.66,.46),  # between N13 and N17 — receives N13→17a edge
    "17b":(.85,.50),  # above-right of N17
    "17c":(.85,.42),  # right of N17
    "17d":(.93,.42),  # far-right of N17
    # ── §5.1.14 sub-nodes (cluster around N14) ──
    "14a":(.50,.46),  # above-left of N14 — receives N13→14a edge
    "14b":(.49,.32),  # below-left of N14 (root evidence)
    "14c":(.57,.30),  # below N14 (root evidence)
    "14d":(.36,.32),  # left of N14 — receives N10→14d edge
    # ── §5.1.15 sub-nodes (cluster around N15 at (.73,.55)) ──
    "15a":(.66,.62),  # above-left of N15 — receives N13→15a edge
    "15b":(.78,.65),  # above-right of N15 (root evidence)
    "15c":(.85,.60),  # right of N15 (root evidence)
    "15d":(.85,.52),  # below-right of N15 (shared n14c attestation)
    # ── §5.1.18 sub-nodes (cluster around N18 at (.57,.20)) ──
    "18a":(.50,.26),  # above-left of N18 — receives N13→18a edge (long)
    "18b":(.50,.14),  # below-left of N18 (root evidence — driven by per-conv tags)
    "18c":(.58,.12),  # below N18 (root evidence — driven by per-conv tags)
    "18d":(.66,.26)}  # above-right of N18 — receives N10→18d edge

def draw_dag(post,sel=None):
    fig,ax=plt.subplots(figsize=(13,9),facecolor='#fafafa')
    # Mark 8 Phase 3 (Alternative A): xlim extended left to -.06 to give
    # margin space for vertical layer labels. Headers now run vertically
    # at the left edge of each layer rectangle rather than horizontally
    # at the top, eliminating all node-collision risk. Pattern is a
    # standard scientific-diagram convention (axis-style annotations).
    ax.set_xlim(-.06,1.02);ax.set_ylim(-.08,1.02);ax.axis('off');ax.set_facecolor('#fafafa')
    # Layer rectangles + vertical margin labels.
    # Each label is a single rotated text block ("LAYER N\nSubtitle"),
    # centered vertically on its rectangle at x=-.025. The earlier
    # horizontal-header treatment created persistent overlap problems
    # with N1's label cluster (Layer II header crossing N1's "83%"
    # percentage label) and the N3 ellipse (long Layer II subtitle
    # sprawling across the diagram). Vertical placement at margin
    # solves both.
    for y, h, title, subtitle in [
        (.83, .10, "LAYER I",   "Substantive risk"),
        (.29, .53, "LAYER II",  "Systemic distortion & doctrinal fidelity"),
        (-.04, .09, "LAYER III", "Structural output"),
    ]:
        ax.add_patch(plt.Rectangle((0,y),1.0,h,color='#f0f0f0',alpha=.55,zorder=0))
        cy = y + h/2  # vertical center of layer rectangle
        ax.text(-.025, cy, f"{title}\n{subtitle}",
                fontsize=7, color='#aaa', fontweight='bold',
                va='center', ha='center', zorder=1, alpha=.9,
                rotation=90, rotation_mode='anchor', linespacing=1.4)
    for f,t in EDGES:
        if f not in NP or t not in NP: continue
        x1,y1=NP[f];x2,y2=NP[t];hi=sel and (f==sel or t==sel)
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle="-|>",color='#888' if hi else '#ccc',lw=1.2 if hi else .6,
            connectionstyle="arc3,rad=0.05"))
    NR={1:.055,20:.055,
        # §5.1.17 sub-nodes — smaller radius for visual hierarchy
        "17a":.025, "17b":.025, "17c":.025, "17d":.025,
        # §5.1.14 sub-nodes — same smaller radius
        "14a":.025, "14b":.025, "14c":.025, "14d":.025,
        # §5.1.15 sub-nodes — same smaller radius
        "15a":.025, "15b":.025, "15c":.025, "15d":.025,
        # §5.1.18 sub-nodes — same smaller radius
        "18a":.025, "18b":.025, "18c":.025, "18d":.025}
    # Mark 8 Phase 3 — B-prime per JP M8/P3 lock-in: N1's percentage label
    # is rendered in a doctrinal-state colour (green Default / amber
    # Pressure / red Failure) rather than the type-base brown. The number
    # itself is preserved to maintain DAG topology-view consistency; the
    # colour communicates the doctrinal posture without changing the
    # diagram's grammar (every node still shows a percentage).
    _n1_state_dag = _n1_doctrinal_state()
    _n1_state_color_dag = {
        "default":  "#3B6D11",  # green
        "pressure": "#BA7517",  # amber
        "failure":  "#A32D2D",  # red
    }.get(_n1_state_dag, "#3B6D11")
    for nid,(x,y) in NP.items():
        m=NODE_META[nid];col=TC[m["type"]];p=post.get(nid,.5);iS=sel==nid
        r=NR.get(nid,.040)
        ax.add_patch(plt.Circle((x,y),r,color=col if iS else col+'28',ec=col,lw=2 if iS else 1,zorder=3))
        th=np.linspace(-np.pi/2,-np.pi/2+2*np.pi*p,60)
        ax.plot(x+(r+.010)*np.cos(th),y+(r+.010)*np.sin(th),color=col,lw=2.5,alpha=.85,zorder=4)
        # Font size: 8 for single-digit ints, 7 for double-digit, 5.5 for sub-nodes
        if isinstance(nid, str):
            _fs = 5.5
        else:
            _fs = 8 if nid < 10 else 7
        ax.text(x,y,str(nid),ha='center',va='center',fontsize=_fs,
                fontweight='bold',color='white' if iS else col,zorder=5)
        # Mark 8 Phase 3 — Approach 1 readability fix: N1's "Burden of
        # proof" label is exactly 15 chars and was being truncated to
        # "Burden of proo" by the global 14-char cap. Per JP M8/P3
        # lock-in, drop the truncation for N1 specifically (most-read
        # label in the diagram, doctrinal centrality). Other 19 nodes
        # retain the existing 14-char truncation pending a Mark 9
        # readability pass on the diagram as a whole.
        if nid == 1:
            lbl = m["short"]  # render in full — "Burden of proof"
        else:
            lbl = m["short"][:14] + ("…" if len(m["short"]) > 14 else "")
        # Sub-nodes use tighter label spacing and slightly smaller text
        if isinstance(nid, str):
            ax.text(x,y-r-.018,lbl,ha='center',va='top',fontsize=5.5,color='#555',zorder=5)
            ax.text(x,y-r-.034,f'{p*100:.0f}%',ha='center',va='top',fontsize=5,color=col,fontweight='bold',zorder=5,alpha=.8)
        else:
            ax.text(x,y-r-.025,lbl,ha='center',va='top',fontsize=6.5,color='#555',zorder=5)
            # B-prime: N1's percentage uses doctrinal-state colour
            _pct_color = _n1_state_color_dag if nid == 1 else col
            ax.text(x,y-r-.050,f'{p*100:.0f}%',ha='center',va='top',fontsize=6,color=_pct_color,fontweight='bold',zorder=5,alpha=.8)
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
except Exception as _canlii_err:
    # Catches ImportError plus any module-level failure (missing deps,
    # secrets, runtime errors). Provide stub fallbacks so the rest of the
    # app continues to render even if CanLII is unavailable.
    CANLII_ON = False
    NODE_SEARCH_QUERIES = {}
    ALL_TRACKED_CITATIONS = []
    TETRAD_CITATIONS = []
    PROPORTIONALITY_CITATIONS = []
    def canlii_ok(): return False
    def validate_api_key(): return {"valid": False, "error": f"canlii_client unavailable: {type(_canlii_err).__name__}"}
    def search_node_developments(*a, **kw): return {}
    def get_tetrad_updates(*a, **kw): return {}
    def search_with_filters(*a, **kw): return {}
    def get_tracked_updates(*a, **kw): return {}
    def flatten_search_results(*a, **kw): return []

# ── Tabs ──────────────────────────────────────────────────────────────────────
TABS=st.tabs(["📋 Summary","🕸️ Architecture","📋 Profile","💬 Intake (Chat)",
              "📜 Criminal Record","🦅 Gladue","⚖️ SCE",
              "🔬 Risk & Distortions","📊 Inference","📂 Documents",
              "🔀 Scenarios","⚛️ Quantum","📄 Report"])

# ── T0: Summary (Mark 8) ──────────────────────────────────────────────────────
with TABS[0]:
    _band_lbl, _band_fg, _band_bg = _summary_band(P[20])
    _drv_up_raw, _drv_dn_raw = _top_drivers(P, k=8)
    # ── Doctrinal architecture: N1 (burden of proof), N5 (risk-tool validity
    # gate per Ewert), and N19 (collider bias) — three structural conditioning ──
    # Per Chapter 5: N1 encodes evidentiary admissibility burden; N5 encodes
    # the Ewert risk-tool validity gate that conditions how risk-tool outputs
    # (N3, N4) carry weight; N19 encodes collider bias from system-entry
    # variables. These three nodes condition the architecture's inferential
    # structure rather than responding directly to case-specific evidence.
    _struct_nodes = {1, 5, 19}  # node IDs treated as structural in the Summary panel
    _drv_up = [d for d in _drv_up_raw if d["nid"] not in _struct_nodes][:5]
    _drv_dn = [d for d in _drv_dn_raw if d["nid"] not in _struct_nodes][:5]
    # Build _struct_drivers directly from _struct_nodes + P — these cards 
    # render unconditionally because they represent structural conditioning
    # of the inference, independent of whether the underlying node happens
    # to surface in the top-k drivers ranking.
    _struct_drivers = []
    for nid in sorted(_struct_nodes):
        _meta = NODE_META.get(nid, {})
        _t = _meta.get("type", "")
        _struct_drivers.append({
            "nid":        nid,
            "short":      _meta.get("short", f"N{nid}"),
            "type_":      _t,
            "type_label": TL.get(_t, _t.title()),
            "color":      TC.get(_t, "#888"),
            "p":          float(P.get(nid, 0.0)),
        })
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
<p>The architecture comprises three substantive layers per Chapter 5 plus one cross-cutting diagnostic layer. The <em>Substantive Risk Layer</em> (Nodes 1–4, with the temporal modifier N2i) represents empirically supported risk indicators subject to Canadian burdens of proof and proportionality constraints. The <em>Systemic Distortion and Doctrinal Fidelity Layer</em> (Nodes 5–19, including the strategic-pleas modifier N7i and sub-nodes 10a–10d) captures mechanisms through which criminal records and risk assessments become unreliable, qualifying — never displacing — the confidence placed in upstream evidence. The <em>Structural Output Layer</em> (Node 20) is the terminal convergence point representing Dangerous Offender designation likelihood given the upstream architecture. The cross-cutting <em>Quantum diagnostic layer</em> (Appendix Q) surfaces epistemic conditions — order effects, contextuality, premature scalar collapse, distorted priors — that classical Bayesian inference is poorly equipped to represent.</p>
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

    # ── Zone 2a: Doctrinal architecture (N1, N5, N19 — structural constraints) ──
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
                # Stable accent palette per structural node identity
                if _d["nid"] == 1:
                    _accent = "#BA7517"; _accent_bg = "#FAEEDA"; _accent_border = "#E5CC95"
                elif _d["nid"] == 5:
                    _accent = "#185FA5"; _accent_bg = "#E8F0FA"; _accent_border = "#C7D3E5"
                else:  # N19
                    _accent = "#5C4F8A"; _accent_bg = "#EFEBF5"; _accent_border = "#C8BFDA"
                # Surface caption depends on node identity
                if _d["nid"] == 1:
                    _surface = (
                        "Encodes the doctrinal admissibility posture toward "
                        "evidentiary inputs at sentencing. Per <em>R. v. "
                        "Gardiner</em> [1982] 2 SCR 368 and s. 724(3) "
                        "<em>Criminal Code</em>, asymmetric burdens apply "
                        "per evidentiary input — Crown bears BARD on "
                        "aggravating facts, defence bears BoP on mitigating "
                        "facts. PARVIS operates on the doctrinal assumption "
                        "that those burdens are met (default posture) and "
                        "fluctuates only when the user records explicit "
                        "&ldquo;insufficient&rdquo; findings via the §RM.1 "
                        "register. N1 is not a probability — it is the "
                        "operationalisation of that posture."
                    )
                elif _d["nid"] == 5:
                    _surface = (
                        "Encodes the Ewert validity gate on risk-assessment "
                        "tools — PCL-R, Static-99R, and similar instruments. "
                        "Conditions how downstream risk-tool outputs carry "
                        "weight when applied to populations on which the tools "
                        "were not validated. Functions as a structural "
                        "discount on tool-derived evidence, not as a posterior "
                        "over case facts."
                    )
                else:  # N19
                    _surface = (
                        "Detects and corrects collider bias arising when "
                        "sentencing inference conditions on system-entry "
                        "variables (arrest, charge, detention, conviction). "
                        "These variables are downstream effects of multiple "
                        "upstream causes, including policing intensity and "
                        "case complexity. Conditioning on them induces "
                        "spurious correlations between upstream causes."
                    )
                # Card — N1 uses doctrinal-state framing (Option II per JP M8/P3
                # lock-in: posture label dominant, depth indicator subordinate).
                # N5 and N19 retain numerical posterior display (their nodes
                # don't carry the same probabilistic-misreading risk as N1).
                if _d["nid"] == 1:
                    _n1_summary = _n1_audit_summary()
                    # Override accent palette with state-driven colour
                    _accent = _n1_summary["color_accent"]
                    _accent_bg = _n1_summary["color_bg"]
                    _accent_border = _n1_summary["color_border"]
                    _state_label = _n1_summary["label"]
                    _posterior_n1 = float(P.get(1, 0.83))
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
                        f"</div>"
                        # Doctrinal-posture block (dominant)
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.66rem;"
                        f"font-weight:700;color:{_accent};text-transform:uppercase;"
                        f"letter-spacing:0.08em;margin:6px 0 2px 0'>"
                        f"Doctrinal posture</div>"
                        f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.4rem;"
                        f"font-weight:500;color:{_accent};line-height:1.15;margin-bottom:8px'>"
                        f"{_state_label}</div>"
                        # Surface caption
                        f"<div style='font-family:Fraunces,serif;font-style:italic;"
                        f"font-size:0.84rem;color:#3a3a3a;line-height:1.55'>{_surface}</div>"
                        # Subordinate depth indicator
                        f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:10px;"
                        f"padding-top:8px;border-top:1px dashed {_accent_border}'>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
                        f"color:#9E9E9E;text-transform:uppercase;letter-spacing:0.06em'>"
                        f"Audit-pressure depth</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                        f"font-weight:600;color:{_accent}'>{_posterior_n1*100:.1f}%</div>"
                        f"<div style='font-family:Fraunces,serif;font-style:italic;"
                        f"font-size:0.7rem;color:#9E9E9E'>"
                        f"operationalisation, not a probability</div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # N5 / N19 retain existing percentage display (their nodes
                    # don't carry the doctrinal-state framing N1 requires).
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
                # Mark 8 Phase 4 — formal-treatment expander relocated to
                # Architecture tab "Doctrinal foundations" section below
                # the DAG. Per JP M8/P4 lock-in: Summary tab serves as
                # situational-awareness landing page (live posture +
                # surface scope), Architecture tab carries the deep
                # architectural treatment of why N1/N5/N19 are
                # categorically structural.

    st.markdown(
        "<div style='border-top:1px solid #EFEDE7;margin:24px 0 18px 0'></div>",
        unsafe_allow_html=True,
    )

    # ── Zone 2: Drivers ──────────────────────────────────────────────────────
    st.markdown("### Drivers of the posterior")
    st.caption("Top 5 nodes pushing DO risk up, top 5 pulling it down. "
               "Case-responsive nodes only — structural constraints (N1, N5, N19) "
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

    # ── Zone 2.5: §5.1.17 N17 audit signal breakdown (if active) ────────────
    # Surfaces the four-parent contributions when N17 is materially active.
    # N17 is "active" when its posterior is above its baseline default (~10%)
    # OR when any of the four sub-nodes signals High. This panel renders only
    # when there is something meaningful to display — for clean records with
    # no §5.1.17 signals, the panel stays hidden to avoid clutter.
    _n17_post = float(P.get(17, 0.10))
    _n17a_post = float(P.get("17a", 0.0))
    _n17b_post = float(P.get("17b", 0.0))
    _n17c_post = float(P.get("17c", 0.0))
    _n17d_post = float(P.get("17d", 0.0))
    _n17_active = (_n17_post > 0.20) or any(
        v >= 0.50 for v in (_n17a_post, _n17b_post, _n17c_post, _n17d_post)
    )
    if _n17_active and not _empty:
        st.markdown(
            "<div style='margin-top:24px;font-size:0.92rem;font-weight:500;"
            "font-family:DM Sans, sans-serif;color:#1A1A1A;letter-spacing:-0.005em'>"
            "§5.1.17 audit signal — four-parent breakdown</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Per Chapter 5 §5.1.17 §4, N17 (over-policing & epistemic "
            "contamination) has four parent inputs. This panel surfaces the "
            "current state of each contribution. N17 high → record reliability "
            "falls per §5.1.17 §6."
        )

        _n17_color = "#185FA5"  # distortion blue per TC palette
        # Four sub-node tiles in a horizontal grid
        _sub_nodes = [
            ("17a", "Jurisdictional disparity", _n17a_post,
             "Index per case jurisdiction · §5.1.17 §4 row 1"),
            ("17b", "Enforcement-disparity engagement", _n17b_post,
             "OR-gate over Gladue/SCE evidence · §5.1.17 §4 row 2"),
            ("17c", "Non-violent charge density", _n17c_post,
             "Pattern-matched from record · §5.1.17 §4 row 3"),
            ("17d", "Surveillance-triggered entries", _n17d_post,
             "Pattern-matched from record · §5.1.17 §4 row 4"),
        ]
        _tile_cols = st.columns(4)
        for _col, (_sid, _name, _p, _desc) in zip(_tile_cols, _sub_nodes):
            with _col:
                _state_label = "High" if _p >= 0.50 else "Low"
                _state_color = "#A32D2D" if _p >= 0.50 else "#9E9E9E"
                _bg_color = "#FAEEDA" if _p >= 0.50 else "#FBFAF7"
                _border_color = "#E5CC95" if _p >= 0.50 else "#E0DDD6"
                st.markdown(
                    f"<div style='background:{_bg_color};border:1px solid {_border_color};"
                    f"border-radius:8px;padding:10px 12px;height:100%;min-height:120px'>"
                    f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"font-weight:600;color:{_n17_color};background:{_n17_color}18;"
                    f"padding:2px 6px;border-radius:4px'>N{_sid}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;color:{_state_color}'>{_state_label}</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem;font-weight:500;color:#1A1A1A;"
                    f"line-height:1.35;margin-bottom:6px'>{_name}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;"
                    f"font-weight:600;color:{_n17_color}'>{_p*100:.1f}%</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.7rem;color:#888;margin-top:4px;line-height:1.4'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Aggregate N17 row beneath the four tiles
        _n17_color_top = "#A32D2D" if _n17_post >= 0.50 else "#185FA5"
        st.markdown(
            f"<div style='margin-top:10px;padding:10px 14px;background:#F6F4F0;"
            f"border:1px solid #E5E0D8;border-radius:8px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
            f"font-weight:600;color:{_n17_color};background:{_n17_color}18;"
            f"padding:2px 6px;border-radius:4px;margin-right:8px'>N17</span>"
            f"<span style='font-size:0.86rem;font-weight:500;color:#1A1A1A'>"
            f"Over-policing & epistemic contamination</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:{_n17_color_top}'>{_n17_post*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Zone 2.6: §5.1.14 N14 audit signal breakdown (if active) ─────────
    # Same conditional-render pattern as N17 — only appears when N14 is materially
    # active (posterior > 25% or any sub-node ≥ 50%). N14's baseline default is
    # ~23% so threshold is set slightly above that to filter floor noise.
    _n14_post = float(P.get(14, 0.23))
    _n14a_post = float(P.get("14a", 0.0))
    _n14b_post = float(P.get("14b", 0.0))
    _n14c_post = float(P.get("14c", 0.0))
    _n14d_post = float(P.get("14d", 0.0))
    _n14_active = (_n14_post > 0.25) or any(
        v >= 0.50 for v in (_n14a_post, _n14b_post, _n14c_post, _n14d_post)
    )
    if _n14_active and not _empty:
        st.markdown(
            "<div style='margin-top:24px;font-size:0.92rem;font-weight:500;"
            "font-family:DM Sans, sans-serif;color:#1A1A1A;letter-spacing:-0.005em'>"
            "§5.1.14 audit signal — four-parent breakdown</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Per Chapter 5 §5.1.14 §5, N14 (temporal distortion in prior records) "
            "has four parent inputs that condition how production-era factors "
            "compromise record reliability. N14 high → record reliability falls "
            "per §5.1.14 §6."
        )

        _n14_color = "#185FA5"  # distortion blue per TC palette
        _sub_nodes = [
            ("14a", "Sentencing era severity", _n14a_post,
             "Year-derived (2008–2015) · §5.1.14 §5 row 1"),
            ("14b", "Historical mandatory minimum", _n14b_post,
             "Pattern-matched offence + year · §5.1.14 §5 row 2"),
            ("14c", "SCE absent at sentencing", _n14c_post,
             "Pre-Ipeelee year heuristic · §5.1.14 §5 row 3"),
            ("14d", "Judicial competence absent", _n14d_post,
             "BN-derived from N10 · §5.1.14 §5 row 4"),
        ]
        _tile_cols = st.columns(4)
        for _col, (_sid, _name, _p, _desc) in zip(_tile_cols, _sub_nodes):
            with _col:
                _state_label = "High" if _p >= 0.50 else "Low"
                _state_color = "#A32D2D" if _p >= 0.50 else "#9E9E9E"
                _bg_color = "#FAEEDA" if _p >= 0.50 else "#FBFAF7"
                _border_color = "#E5CC95" if _p >= 0.50 else "#E0DDD6"
                st.markdown(
                    f"<div style='background:{_bg_color};border:1px solid {_border_color};"
                    f"border-radius:8px;padding:10px 12px;height:100%;min-height:120px'>"
                    f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"font-weight:600;color:{_n14_color};background:{_n14_color}18;"
                    f"padding:2px 6px;border-radius:4px'>N{_sid}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;color:{_state_color}'>{_state_label}</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem;font-weight:500;color:#1A1A1A;"
                    f"line-height:1.35;margin-bottom:6px'>{_name}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;"
                    f"font-weight:600;color:{_n14_color}'>{_p*100:.1f}%</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.7rem;color:#888;margin-top:4px;line-height:1.4'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Aggregate N14 row beneath the four tiles
        _n14_color_top = "#A32D2D" if _n14_post >= 0.50 else "#185FA5"
        st.markdown(
            f"<div style='margin-top:10px;padding:10px 14px;background:#F6F4F0;"
            f"border:1px solid #E5E0D8;border-radius:8px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
            f"font-weight:600;color:{_n14_color};background:{_n14_color}18;"
            f"padding:2px 6px;border-radius:4px;margin-right:8px'>N14</span>"
            f"<span style='font-size:0.86rem;font-weight:500;color:#1A1A1A'>"
            f"Temporal distortion in prior records</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:{_n14_color_top}'>{_n14_post*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Zone 2.7: §5.1.15 N15 audit signal breakdown (if active) ─────────
    # Conditional render: appears when N15 is materially active (posterior >
    # 25% or any sub-node ≥ 50%). N15 baseline default ~17% so threshold is
    # set at 25% to filter floor noise.
    _n15_post = float(P.get(15, 0.17))
    _n15a_post = float(P.get("15a", 0.0))
    _n15b_post = float(P.get("15b", 0.0))
    _n15c_post = float(P.get("15c", 0.0))
    _n15d_post = float(P.get("15d", 0.0))
    _n15_active = (_n15_post > 0.25) or any(
        v >= 0.50 for v in (_n15a_post, _n15b_post, _n15c_post, _n15d_post)
    )
    if _n15_active and not _empty:
        st.markdown(
            "<div style='margin-top:24px;font-size:0.92rem;font-weight:500;"
            "font-family:DM Sans, sans-serif;color:#1A1A1A;letter-spacing:-0.005em'>"
            "§5.1.15 audit signal — four-sub-node breakdown</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Per Chapter 5 §5.1.15 §5, N15 (interjurisdictional tariff distortion) "
            "has four sub-node parents conditioning how regional sentencing norms "
            "produce non-comparable record artifacts. N15 also receives N14 as a "
            "structural parent per §5.1.15 §Position — temporal and tariff "
            "distortions compound when both are present. N15 operates as a "
            "calibration / interpretive distortion (per Q5 (α)) rather than a "
            "production-condition reliability degrader — distinct mechanism from "
            "N14 and N17."
        )

        _n15_color = "#185FA5"  # distortion blue per TC palette
        _sub_nodes = [
            ("15a", "Tariff jurisdiction disparity", _n15a_post,
             "case_jur tier-classified · §5.1.15 §5 row 1"),
            ("15b", "Tariff-sensitive offence", _n15b_post,
             "Offence-text pattern · §5.1.15 §5 row 2"),
            ("15c", "Tariff-sensitive sentence length", _n15c_post,
             "Sentence-type heuristic · §5.1.15 §5 row 3"),
            ("15d", "Doctrine absent", _n15d_post,
             "Shared n14c attestation · §5.1.15 §5 row 5"),
        ]
        _tile_cols = st.columns(4)
        for _col, (_sid, _name, _p, _desc) in zip(_tile_cols, _sub_nodes):
            with _col:
                _state_label = "High" if _p >= 0.50 else "Low"
                _state_color = "#A32D2D" if _p >= 0.50 else "#9E9E9E"
                _bg_color = "#FAEEDA" if _p >= 0.50 else "#FBFAF7"
                _border_color = "#E5CC95" if _p >= 0.50 else "#E0DDD6"
                st.markdown(
                    f"<div style='background:{_bg_color};border:1px solid {_border_color};"
                    f"border-radius:8px;padding:10px 12px;height:100%;min-height:120px'>"
                    f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"font-weight:600;color:{_n15_color};background:{_n15_color}18;"
                    f"padding:2px 6px;border-radius:4px'>N{_sid}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;color:{_state_color}'>{_state_label}</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem;font-weight:500;color:#1A1A1A;"
                    f"line-height:1.35;margin-bottom:6px'>{_name}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;"
                    f"font-weight:600;color:{_n15_color}'>{_p*100:.1f}%</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.7rem;color:#888;margin-top:4px;line-height:1.4'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Aggregate N15 row beneath the four tiles
        _n15_color_top = "#A32D2D" if _n15_post >= 0.50 else "#185FA5"
        st.markdown(
            f"<div style='margin-top:10px;padding:10px 14px;background:#F6F4F0;"
            f"border:1px solid #E5E0D8;border-radius:8px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
            f"font-weight:600;color:{_n15_color};background:{_n15_color}18;"
            f"padding:2px 6px;border-radius:4px;margin-right:8px'>N15</span>"
            f"<span style='font-size:0.86rem;font-weight:500;color:#1A1A1A'>"
            f"Interjurisdictional tariff distortion</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:{_n15_color_top}'>{_n15_post*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Zone 2.8: §5.1.18 N18 audit signal breakdown (if active) ─────────
    # Conditional render: appears when N18 is materially active (posterior >
    # 25% or any sub-node ≥ 50%). N18 baseline default ~30-40% so threshold
    # at 25% catches meaningful elevations.
    _n18_post = float(P.get(18, 0.30))
    _n18a_post = float(P.get("18a", 0.0))
    _n18b_post = float(P.get("18b", 0.0))
    _n18c_post = float(P.get("18c", 0.0))
    _n18d_post = float(P.get("18d", 0.0))
    _n18_active = (_n18_post > 0.25) or any(
        v >= 0.50 for v in (_n18a_post, _n18b_post, _n18c_post, _n18d_post)
    )
    if _n18_active and not _empty:
        st.markdown(
            "<div style='margin-top:24px;font-size:0.92rem;font-weight:500;"
            "font-family:DM Sans, sans-serif;color:#1A1A1A;letter-spacing:-0.005em'>"
            "§5.1.18 audit signal — four-sub-node breakdown</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Per Chapter 5 §5.1.18 §1, N18 audits whether prior convictions "
            "substantively integrated SCE / Tetrad jurisprudence. Framed as a "
            "metadata tagging layer evaluating how a record should be "
            "interpreted today, not whether it was properly decided then. "
            "Per Q6=A, N18 routes through record_reliability — high N18 "
            "(Inflated) discounts the evidentiary weight of N2 (validated risk "
            "elevators) consistent with §5.1.18 §6 (\"Criminal Record "
            "Reliability Modifier\")."
        )

        _n18_color = "#185FA5"  # distortion blue per TC palette
        _sub_nodes = [
            ("18a", "Jurisdiction sensitivity", _n18a_post,
             "case_jur tier-classified · §5.1.18 §5 row 2"),
            ("18b", "SCE Presence in reasons", _n18b_post,
             "Per-conviction tag aggregate · §5.1.18 §5 row 3"),
            ("18c", "SCE Substance (Morris Audit)", _n18c_post,
             "Per-conviction tag aggregate · Morris Heuristic Audit"),
            ("18d", "Doctrinal tagging compliance", _n18d_post,
             "BN-derived from N10 · §5.1.18 §5 row 4"),
        ]
        _tile_cols = st.columns(4)
        for _col, (_sid, _name, _p, _desc) in zip(_tile_cols, _sub_nodes):
            with _col:
                _state_label = "High" if _p >= 0.50 else "Low"
                _state_color = "#A32D2D" if _p >= 0.50 else "#9E9E9E"
                _bg_color = "#FAEEDA" if _p >= 0.50 else "#FBFAF7"
                _border_color = "#E5CC95" if _p >= 0.50 else "#E0DDD6"
                st.markdown(
                    f"<div style='background:{_bg_color};border:1px solid {_border_color};"
                    f"border-radius:8px;padding:10px 12px;height:100%;min-height:120px'>"
                    f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"font-weight:600;color:{_n18_color};background:{_n18_color}18;"
                    f"padding:2px 6px;border-radius:4px'>N{_sid}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;color:{_state_color}'>{_state_label}</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem;font-weight:500;color:#1A1A1A;"
                    f"line-height:1.35;margin-bottom:6px'>{_name}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;"
                    f"font-weight:600;color:{_n18_color}'>{_p*100:.1f}%</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.7rem;color:#888;margin-top:4px;line-height:1.4'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Aggregate N18 row beneath the four tiles
        _n18_color_top = "#A32D2D" if _n18_post >= 0.50 else "#185FA5"
        st.markdown(
            f"<div style='margin-top:10px;padding:10px 14px;background:#F6F4F0;"
            f"border:1px solid #E5E0D8;border-radius:8px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<div>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
            f"font-weight:600;color:{_n18_color};background:{_n18_color}18;"
            f"padding:2px 6px;border-radius:4px;margin-right:8px'>N18</span>"
            f"<span style='font-size:0.86rem;font-weight:500;color:#1A1A1A'>"
            f"SCE Profile audit (Gladue / Ewert / Morris / Ellis)</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.05rem;"
            f"font-weight:600;color:{_n18_color_top}'>{_n18_post*100:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Zone 2.9: §5.1.19 N19 collider-bias breakdown (if active) ────────
    # Conditional render: appears when N19 posterior ≥ 0.50 OR the app-side
    # joint-elevation signal is active. Per §5.1.19 §1, the headline DO
    # posterior is not modified — the secondary discounted figure is shown
    # alongside as a contingent display per Q4=C.
    _n19_post = float(P.get(19, 0.30))
    _n19_signal_dict = st.session_state.get("n19_collider_signal") or {}
    _n19_signal_active = bool(_n19_signal_dict.get("active", False))
    _n19_active = (_n19_post >= 0.50) or _n19_signal_active
    if _n19_active and not _empty:
        _disc_risk_summary = st.session_state.get("n19_discounted_risk")
        _headline_summary = float(P.get(20, 0.30))
        _n14_p_summary = float(P.get(14, 0.5))
        _n17_p_summary = float(P.get(17, 0.5))

        st.markdown(
            "<div style='margin-top:24px;font-size:0.92rem;font-weight:500;"
            "font-family:DM Sans, sans-serif;color:#1A1A1A;letter-spacing:-0.005em'>"
            "§5.1.19 collider-bias detection — inference integrity check"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Per Chapter 5 §5.1.19 §1, N19 flags when the inference drawn "
            "from the criminal record may be systematically unreliable due "
            "to the geometry of the reasoning itself — without adding "
            "evidence to the headline posterior. The secondary discounted "
            "figure operationalises §5.1.19 §8 (\"final risk scores reflect "
            "causal uncertainty rather than inflated confidence\") for "
            "contingent display."
        )

        _n19_color = "#185FA5"

        # Three tiles: N14, N17 (parents), N19 (collider posterior)
        _tile_cols = st.columns(3)
        _tile_data = [
            ("N14", "Temporal distortion (Case-Complexity proxy)", _n14_p_summary),
            ("N17", "Over-policing intensity",                     _n17_p_summary),
            ("N19", "Collider-bias detection",                     _n19_post),
        ]
        for _col, (_lbl, _desc, _val) in zip(_tile_cols, _tile_data):
            with _col:
                _state_high = _val >= 0.60
                _state_color = "#A32D2D" if _state_high else "#9E9E9E"
                _bg_color = "#FAEEDA" if _state_high else "#FBFAF7"
                _border_color = "#E5CC95" if _state_high else "#E0DDD6"
                st.markdown(
                    f"<div style='background:{_bg_color};border:1px solid {_border_color};"
                    f"border-radius:8px;padding:10px 12px;height:100%;min-height:110px'>"
                    f"<div style='display:flex;align-items:baseline;justify-content:space-between;"
                    f"margin-bottom:6px'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"font-weight:600;color:{_n19_color};background:{_n19_color}18;"
                    f"padding:2px 6px;border-radius:4px'>{_lbl}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                    f"font-weight:600;color:{_state_color}'>{'High' if _state_high else 'Low'}</span>"
                    f"</div>"
                    f"<div style='font-size:0.78rem;font-weight:500;color:#1A1A1A;"
                    f"line-height:1.35;margin-bottom:6px'>{_desc}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:1rem;"
                    f"font-weight:600;color:{_n19_color}'>{_val*100:.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Headline + discounted comparison (the Q4=C contingent display)
        if _disc_risk_summary is not None:
            _delta_pp_s = (_disc_risk_summary - _headline_summary) * 100
            _delta_color = "#A32D2D" if _delta_pp_s <= -1.0 else "#9E9E9E"
            st.markdown(
                f"<div style='margin-top:12px;padding:14px 18px;"
                f"background:linear-gradient(90deg, #FBFAF7 0%, #F6F4F0 100%);"
                f"border:1px solid #E5E0D8;border-radius:8px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"flex-wrap:wrap;gap:12px'>"
                f"<div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem;margin-bottom:2px'>"
                f"DO posterior — headline vs collider-discounted (§1 vs §8)"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.76rem;color:#707070'>"
                f"Headline preserved per §1; secondary contingent per §8."
                f"</div>"
                f"</div>"
                f"<div style='display:flex;align-items:baseline;gap:14px'>"
                f"<div style='text-align:right'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                f"color:#888'>HEADLINE</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.15rem;"
                f"font-weight:600;color:#1A1A1A'>{_headline_summary*100:.1f}%</div>"
                f"</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.1rem;"
                f"color:#999'>→</div>"
                f"<div style='text-align:right'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                f"color:{_delta_color}'>DISCOUNTED ({_delta_pp_s:+.1f}pp)</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.15rem;"
                f"font-weight:600;color:{_delta_color}'>{_disc_risk_summary*100:.1f}%</div>"
                f"</div>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

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
        opts={None:"— none —"}
        opts.update({n:f"N{n}: {NODE_META[n]['name']}" for n in range(1,21)})
        # §5.1.17 sub-nodes — appended after main nodes
        for sub_id in ("17a", "17b", "17c", "17d"):
            opts[sub_id] = f"N{sub_id}: {NODE_META[sub_id]['name']}"
        # §5.1.14 sub-nodes — appended after §5.1.17 sub-nodes
        for sub_id in ("14a", "14b", "14c", "14d"):
            opts[sub_id] = f"N{sub_id}: {NODE_META[sub_id]['name']}"
        # §5.1.15 sub-nodes — appended after §5.1.14 sub-nodes
        for sub_id in ("15a", "15b", "15c", "15d"):
            opts[sub_id] = f"N{sub_id}: {NODE_META[sub_id]['name']}"
        # §5.1.18 sub-nodes — appended after §5.1.15 sub-nodes
        for sub_id in ("18a", "18b", "18c", "18d"):
            opts[sub_id] = f"N{sub_id}: {NODE_META[sub_id]['name']}"
        sel=st.selectbox("Inspect node",list(opts.keys()),format_func=lambda x:opts[x])
        st.pyplot(draw_dag(P,sel),use_container_width=True)
    with cr:
        if sel:
            m=NODE_META[sel];col=TC[m["type"]];p=P.get(sel,.5)
            if sel == 1:
                # Mark 8 Phase 3 — N1 inspect panel uses doctrinal-state
                # framing per JP M8/P3 lock-in. State label dominant,
                # numerical depth subordinate, with audit-summary detail.
                _n1_summary_inspect = _n1_audit_summary()
                _n1_accent = _n1_summary_inspect["color_accent"]
                _n1_bg = _n1_summary_inspect["color_bg"]
                _n1_border = _n1_summary_inspect["color_border"]
                _n1_label = _n1_summary_inspect["label"]
                _n1_audited = _n1_summary_inspect["audited_count"]
                _n1_insuf = _n1_summary_inspect["insufficient_count"]
                _n1_pend = _n1_summary_inspect["pending_count"]
                _n1_sat = _n1_summary_inspect["satisfied_count"]
                _detail_parts = []
                if _n1_sat: _detail_parts.append(f"{_n1_sat} satisfied")
                if _n1_pend: _detail_parts.append(f"{_n1_pend} pending")
                if _n1_insuf: _detail_parts.append(f"{_n1_insuf} insufficient")
                _detail_block = ""
                if _n1_audited:
                    _detail_block = (
                        f"<div style='font-family:Fraunces,serif;font-style:italic;"
                        f"font-size:.78rem;color:#5A5A5A;margin-top:8px'>"
                        f"{_n1_audited} audited input(s) — "
                        f"{', '.join(_detail_parts)}.</div>"
                    )
                st.markdown(
                    f"<div style='background:{_n1_bg};border:1px solid {_n1_border};"
                    f"border-left:3px solid {_n1_accent};border-radius:12px;padding:1rem'>"
                    f"<div style='font-size:.68rem;color:{_n1_accent};font-weight:700'>"
                    f"{TL[m['type']]}</div>"
                    f"<div style='font-size:1rem;font-weight:700;margin-top:4px'>"
                    f"N{sel}: {m['name']}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.62rem;"
                    f"color:{_n1_accent};text-transform:uppercase;letter-spacing:.08em;"
                    f"font-weight:700;margin:12px 0 2px 0'>Doctrinal posture</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.5rem;"
                    f"font-weight:500;color:{_n1_accent};line-height:1.15'>{_n1_label}</div>"
                    f"{_detail_block}"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:10px;"
                    f"padding-top:8px;border-top:1px dashed {_n1_border}'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
                    f"color:#9E9E9E;text-transform:uppercase;letter-spacing:.06em'>"
                    f"Audit-pressure depth</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.86rem;"
                    f"font-weight:600;color:{_n1_accent}'>{p*100:.1f}%</div>"
                    f"</div>"
                    f"<div style='height:4px;background:#eee;border-radius:2px;margin-top:6px'>"
                    f"<div style='width:{p*100:.0f}%;height:100%;background:{_n1_accent};"
                    f"border-radius:2px'></div></div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:.7rem;color:#9E9E9E;margin-top:6px;line-height:1.4'>"
                    f"Operationalisation of doctrinal posture toward case-file audit "
                    f"state — not a probability over case facts.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif sel == 5:
                # Mark 8 Phase 4 — N5 inspect panel surfaces the Ewert
                # validity gate as architectural commitment, not just a
                # numeric posterior. Per JP M8/P4 lock-in: all three
                # structural meta-constraint nodes (N1, N5, N19) get
                # architectural treatment in the inspect panel; remaining
                # 17 nodes use the generic else-branch presentation.
                _n5_accent  = "#185FA5"
                _n5_bg      = "#E8F0FA"
                _n5_border  = "#C7D3E5"
                st.markdown(
                    f"<div style='background:{_n5_bg};border:1px solid {_n5_border};"
                    f"border-left:3px solid {_n5_accent};border-radius:12px;padding:1rem'>"
                    f"<div style='font-size:.68rem;color:{_n5_accent};font-weight:700'>"
                    f"{TL[m['type']]}</div>"
                    f"<div style='font-size:1rem;font-weight:700;margin-top:4px'>"
                    f"N{sel}: {m['name']}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.62rem;"
                    f"color:{_n5_accent};text-transform:uppercase;letter-spacing:.08em;"
                    f"font-weight:700;margin:12px 0 2px 0'>Architectural function</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.15rem;"
                    f"font-weight:500;color:{_n5_accent};line-height:1.2'>"
                    f"Tool-validity gate</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:.78rem;color:#5A5A5A;margin-top:8px;line-height:1.5'>"
                    f"Per <em>Ewert v Canada</em> [2018] 2 SCR 165, conditions "
                    f"how risk-tool outputs (PCL-R, Static-99R) carry weight "
                    f"when applied to populations on which the tools were "
                    f"not validated.</div>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:10px;"
                    f"padding-top:8px;border-top:1px dashed {_n5_border}'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
                    f"color:#9E9E9E;text-transform:uppercase;letter-spacing:.06em'>"
                    f"Posterior P(N5=High)</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.86rem;"
                    f"font-weight:600;color:{_n5_accent}'>{p*100:.1f}%</div>"
                    f"</div>"
                    f"<div style='height:4px;background:#eee;border-radius:2px;margin-top:6px'>"
                    f"<div style='width:{p*100:.0f}%;height:100%;background:{_n5_accent};"
                    f"border-radius:2px'></div></div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:.7rem;color:#9E9E9E;margin-top:6px;line-height:1.4'>"
                    f"Higher P(High) ⇒ stronger structural discount on tool-derived "
                    f"evidence (N3, N4) at the Layer III aggregation.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif sel == 19:
                # Mark 8 Phase 4 — N19 inspect panel surfaces collider-bias
                # detection as architectural commitment per JP M8/P4 lock-in.
                _n19_accent  = "#5C4F8A"
                _n19_bg      = "#EFEBF5"
                _n19_border  = "#C8BFDA"
                st.markdown(
                    f"<div style='background:{_n19_bg};border:1px solid {_n19_border};"
                    f"border-left:3px solid {_n19_accent};border-radius:12px;padding:1rem'>"
                    f"<div style='font-size:.68rem;color:{_n19_accent};font-weight:700'>"
                    f"{TL[m['type']]}</div>"
                    f"<div style='font-size:1rem;font-weight:700;margin-top:4px'>"
                    f"N{sel}: {m['name']}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.62rem;"
                    f"color:{_n19_accent};text-transform:uppercase;letter-spacing:.08em;"
                    f"font-weight:700;margin:12px 0 2px 0'>Architectural function</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.15rem;"
                    f"font-weight:500;color:{_n19_accent};line-height:1.2'>"
                    f"Inference-structure correction</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:.78rem;color:#5A5A5A;margin-top:8px;line-height:1.5'>"
                    f"Detects collider bias arising when sentencing inference "
                    f"conditions on system-entry variables (arrest, charge, "
                    f"detention, conviction) — downstream effects of multiple "
                    f"upstream causes including policing intensity.</div>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:10px;"
                    f"padding-top:8px;border-top:1px dashed {_n19_border}'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.6rem;"
                    f"color:#9E9E9E;text-transform:uppercase;letter-spacing:.06em'>"
                    f"Posterior P(N19=High)</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.86rem;"
                    f"font-weight:600;color:{_n19_accent}'>{p*100:.1f}%</div>"
                    f"</div>"
                    f"<div style='height:4px;background:#eee;border-radius:2px;margin-top:6px'>"
                    f"<div style='width:{p*100:.0f}%;height:100%;background:{_n19_accent};"
                    f"border-radius:2px'></div></div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:.7rem;color:#9E9E9E;margin-top:6px;line-height:1.4'>"
                    f"Higher P(High) ⇒ stronger structural caution that the "
                    f"record reflects surveillance exposure as much as conduct.</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
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

    # ════════════════════════════════════════════════════════════════════════
    # Mark 8 Phase 5 — Doctrinal foundations expanded to 11 nodes
    # ════════════════════════════════════════════════════════════════════════
    # Per JP M8/P5 lock-in: foundations section now covers all HIGH-confidence
    # nodes in five thematic groups. Eight new treatments added (N6, N7, N9,
    # N10, N16, N17, N18, N20) plus the existing N1, N5, N19. The five sections
    # reflect doctrinal function rather than node-number order:
    #
    #   1. Procedural foundations           (N1, N6, N7)
    #   2. Tool & inference structure       (N5, N19)
    #   3. Indigenous-sentencing tetrad     (N9, N10, N18)
    #   4. Statutory and structural tensions (N16, N17)
    #   5. Structural output                (N20)
    #
    # MEDIUM-confidence treatments (N2, N3, N4, N8, N14) deferred per JP M8/P5
    # lock-in pending review against thesis chapters. When approved, those
    # treatments will form a sixth section "Substantive risk components"
    # slotted between sections 1 and 2.
    #
    # LOWER-confidence treatments (N11, N12, N13, N15) deferred for the same
    # reason; placement TBD (likely a seventh "Operational diagnostics"
    # section).

    st.markdown(
        "<div style='border-top:1px solid #E0DDD6;margin:36px 0 24px 0'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:1.55rem;"
        "font-weight:500;letter-spacing:-0.005em;margin:0 0 6px 0'>"
        "Doctrinal foundations</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.92rem;color:#707070;margin-bottom:22px;line-height:1.6;"
        "max-width:780px'>"
        "Eleven nodes in this network carry distinct doctrinal commitments — "
        "procedural foundations, structural corrections on tool and inference "
        "validity, the Indigenous-sentencing tetrad, statutory and structural "
        "tensions, and the network's structural output. The sections below "
        "explain the doctrinal foundations of each, grouped by architectural "
        "function rather than node-number order."
        "</div>",
        unsafe_allow_html=True,
    )

    # Shared CSS for the small header chip rendered above each expander
    def _foundation_header_html(node_id: str, title: str, accent: str) -> str:
        return (
            f"<div style='display:flex;align-items:center;gap:10px;"
            f"margin:4px 0 6px 0'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
            f"font-weight:600;padding:3px 9px;border-radius:5px;color:white;"
            f"background:{accent}'>{node_id}</div>"
            f"<div style='font-family:Fraunces,Georgia,serif;font-size:1rem;"
            f"font-weight:500;color:#1a1a1a'>{title}</div>"
            f"</div>"
        )

    # ── Section: Procedural foundations ──
    st.markdown(
        "<div style='margin:24px 0 4px 0;font-family:JetBrains Mono,monospace;"
        "font-size:0.72rem;font-weight:700;color:#9E9E9E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Procedural foundations</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:780px'>Conditioning the inference on whether the procedural conditions of evidence production are sound.</div>",
        unsafe_allow_html=True,
    )
    _sec_cols = st.columns(3)
    with _sec_cols[0]:
        st.markdown(
            _foundation_header_html("N1", "Procedural admissibility", "#BA7517"),
            unsafe_allow_html=True,
        )
        with st.expander("Gardiner asymmetry — Chapter 5 §5.1.1", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.1, N1 operates as a shared parent node "
                "whose states deterministically condition the CPT entries of "
                "downstream aggravation and mitigation nodes — collapsing "
                "likelihood terms to near-zero when evidentiary burdens are "
                "unmet. This is the limiting case of Bayesian conditional "
                "probability where the conditional collapses to certainty. "
                "Deterministic conditioning of this kind is orthodox within "
                "Bayesian network methodology."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "Per <em>R. v. Gardiner</em> [1982] 2 SCR 368, the burden of "
                "proof is <em>not</em> a property of the case — it is a "
                "property of each evidentiary input. The Crown bears BARD on "
                "every aggravating fact it advances; defence bears BoP on "
                "every mitigating fact. Both happen simultaneously in any "
                "contested sentencing. PARVIS's audit math handles this "
                "asymmetry per-input via differentiated weights (1.0 for "
                "aggravating-fail, 0.6 for mitigating-fail)."
                "</p>"
                "<p style='margin:0'>"
                "N1's display reports the doctrinal posture toward the case "
                "file's audit state as a whole: <strong>Default</strong> "
                "(audit pressure baseline; thresholds assumed met), "
                "<strong>Pressure</strong> (at least one input marked "
                "insufficient), or <strong>Failure</strong> (every audited "
                "input marked insufficient). Numerical depth is shown "
                "subordinate to the posture label as an indicator of audit "
                "pressure, not as a probability claim."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[1]:
        st.markdown(
            _foundation_header_html("N6", "Counsel adequacy", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("G.D.B. — Chapter 5 §5.1.6", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.6 and <em>R. v. G.D.B.</em> [2000] 1 SCR 520, "
                "N6 encodes the structural concern that the case before the court may "
                "rest on a record produced under ineffective assistance of counsel. The "
                "Supreme Court's two-part test in <em>G.D.B.</em> — performance below "
                "an objective standard of reasonableness and resulting prejudice — is "
                "doctrinally available on appeal but operationally absent from sentencing "
                "inference, where the trier of fact treats the record as given. N6 makes "
                "that absence visible: where the historical record was generated under "
                "deficient representation, treating it as an unproblematic factual "
                "foundation for <em>DO</em> designation imports the deficiency into the "
                "present analysis."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this through user attestations on the §RM.1 "
                "register marking specific evidentiary inputs as IAC-tainted, and "
                "through pattern-matching against case-record signals (waived rights, "
                "unconsidered defences, plea-driven outcomes inconsistent with the "
                "evidentiary picture). N6's posterior conditions downstream nodes that "
                "draw on the tainted record — most directly N18 (SCE profile audit) "
                "and the conviction-derived components of N2. The architecture does not "
                "adjudicate the IAC claim; it surfaces the structural caution that "
                "historical record reliability is conditional on counsel adequacy."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to break the assumption that court records "
                "are self-validating. <em>G.D.B.</em> tells us that records can be wrong "
                "because counsel was deficient. N6 carries that insight into the "
                "inference architecture so that record-derived evidence is conditioned "
                "on the procedural conditions of its production rather than treated as "
                "foundational."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[2]:
        st.markdown(
            _foundation_header_html("N7", "Bail-WCGP cascade", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("Antic cascade — Chapter 5 §5.1.7", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.7 and <em>R. v. Antic</em> 2017 SCC 27, N7 encodes "
                "the structural cascade from bail denial to coerced plea. The Supreme "
                "Court in <em>Antic</em> held that pretrial detention should be the "
                "exception rather than the rule and that the ladder principle requires "
                "the least restrictive form of release consistent with the public-"
                "interest grounds. Where bail is denied — particularly on grounds the "
                "<em>Antic</em> Court characterised as overused — the detained accused "
                "faces a structural pressure to plead guilty in order to secure release "
                "through time-served sentencing, regardless of factual innocence or "
                "available defences."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this by treating bail denial as a conditioning "
                "input on the reliability of any subsequent plea-derived conviction in "
                "the record. Where the case profile records pretrial detention followed "
                "by a guilty plea on charges where the evidentiary picture would have "
                "supported defence, N7's posterior elevates and the architecture "
                "downstream-conditions the conviction's weight as a validated risk "
                "elevator (N2) and as a feature of the SCE profile (N18). The N7i "
                "sub-node (anticipated credibility impeachment / strategic pleas) carries "
                "the related concern that pleas may be tactically driven rather than "
                "reflective of factual guilt."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to refuse the pretence that all convictions "
                "reflect factual guilt established at trial. <em>Antic</em> tells us "
                "that the bail system produces guilty pleas through pressure "
                "independent of the evidentiary picture. N7 carries that empirical "
                "reality into the inference architecture so that conviction-derived "
                "risk evidence is conditioned on the procedural circumstances of its "
                "production."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Section: Tool & inference structure ──
    st.markdown(
        "<div style='margin:24px 0 4px 0;font-family:JetBrains Mono,monospace;"
        "font-size:0.72rem;font-weight:700;color:#9E9E9E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Tool & inference structure</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:780px'>Corrections on the epistemic structure of the inference itself, distinct from case facts.</div>",
        unsafe_allow_html=True,
    )
    _sec_cols = st.columns(2)
    with _sec_cols[0]:
        st.markdown(
            _foundation_header_html("N5", "Tool validity", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("Ewert principle — Chapter 5 §5.1.5", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.5 and <em>Ewert v Canada</em> [2018] 2 "
                "SCR 165, N5 encodes the validity gate on risk-assessment "
                "tools used in Dangerous Offender proceedings. The Supreme "
                "Court held that CSC's continued reliance on actuarial tools "
                "without confirming their validity for Indigenous populations "
                "breached the statutory duty of accuracy."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this by computing a tool-validity "
                "multiplier from N5's posterior, which discounts the "
                "contribution of N3 (sexual offence risk profile) and N4 "
                "(dynamic risk) to N20. Where tools are unvalidated for the "
                "individual before the court, the architecture reduces "
                "tool-derived weight up to ~42%."
                "</p>"
                "<p style='margin:0'>"
                "This is the <em>Ewert</em> principle made operational at "
                "the level of inferential structure rather than rhetorical "
                "compliance — the gate fires whether or not the parties "
                "raise <em>Ewert</em> in argument."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[1]:
        st.markdown(
            _foundation_header_html("N19", "Inference correction", "#5C4F8A"),
            unsafe_allow_html=True,
        )
        with st.expander("Collider bias — Chapter 5 §5.1.19", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.19, N19 surfaces collider bias "
                "(Berkson's paradox) in criminal-justice inference. When "
                "risk models condition on system-entry variables (arrest, "
                "charge, detention, conviction) without accounting for "
                "selection effects, apparent relationships emerge between "
                "upstream causes that do not exist independently of "
                "conditioning."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "The architecture flags this distortion: where over-policing "
                "contributes to record production (N17), treating the "
                "record as additional evidence of propensity activates "
                "collider bias. Risk inference must not treat exposure to "
                "policing as exposure of risk."
                "</p>"
                "<p style='margin:0'>"
                "The node does not eliminate the distortion — it ensures "
                "that defence counsel and reviewing courts are on notice "
                "that the record is doing double duty as both evidence of "
                "conduct and evidence of surveillance, and that conflating "
                "the two is a structural error."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Section: Indigenous-sentencing tetrad ──
    st.markdown(
        "<div style='margin:24px 0 4px 0;font-family:JetBrains Mono,monospace;"
        "font-size:0.72rem;font-weight:700;color:#9E9E9E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Indigenous-sentencing tetrad</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:780px'>The three tetrad-implementation nodes carrying <em>Gladue</em> / <em>Morris</em> / <em>Ellis</em> commitments.</div>",
        unsafe_allow_html=True,
    )
    _sec_cols = st.columns(3)
    with _sec_cols[0]:
        st.markdown(
            _foundation_header_html("N9", "IGT / cultural treatment", "#3B6D11"),
            unsafe_allow_html=True,
        )
        with st.expander("Gladue mitigation — Chapter 5 §5.1.9", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.9 and <em>R. v. Gladue</em> [1999] 1 SCR 688 / "
                "<em>R. v. Ipeelee</em> 2012 SCC 13, N9 encodes the structural mitigation "
                "that arises from the intersection of intergenerational trauma and the "
                "absence of culturally grounded treatment options. <em>Gladue</em> and "
                "<em>Ipeelee</em> establish that for Indigenous offenders, sentencing "
                "must take account of the unique systemic and background factors that "
                "brought them before the courts and the types of sentencing procedures "
                "and sanctions that may be appropriate given their heritage. The "
                "mitigation is not optional charity — it is doctrinally mandated, and "
                "<em>Ipeelee</em> held that it does not require evidence of a causal "
                "link between the systemic factors and the offence."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this by conditioning N9's posterior on the case "
                "profile's recorded IGT signals (residential-school history, foster-care "
                "displacement, family-violence inheritance) and on the structural "
                "availability of culturally grounded treatment in the relevant "
                "institutional setting. N9 contributes mitigation weight to N20 directly "
                "and conditions N4 (dynamic risk) by recognising that the absence of "
                "culturally appropriate treatment is itself a structural rather than "
                "personal feature of the risk picture. The architecture distinguishes "
                "\"treatment failure\" attributable to the individual from treatment-"
                "system failure attributable to the institutional setting."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to refuse the framing under which Indigenous "
                "offenders are held responsible for failures of a treatment system "
                "designed without reference to their cultural reality. <em>Gladue</em> "
                "and <em>Ipeelee</em> tell us that the system bears responsibility for "
                "that absence; N9 carries that responsibility into the inference "
                "architecture so that culturally grounded treatment unavailability "
                "operates as mitigation rather than as evidence of risk."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[1]:
        st.markdown(
            _foundation_header_html("N10", "SCE misapplication", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("Morris / Ellis — Chapter 5 §5.1.10", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.10 and the line of authority from <em>R. v. Morris</em> "
                "2021 ONCA 680 through <em>R. v. Ellis</em> 2022 BCCA 278, N10 encodes "
                "the structural concern that social-context evidence (SCE) may be "
                "misapplied in sentencing — admitted but downweighted, acknowledged but "
                "not operationalised, or treated as background colour rather than as "
                "evidence of the systemic conditions doctrinally relevant to the "
                "analysis. <em>Morris</em> held that SCE need only show a discernible "
                "nexus to the offender or offence rather than direct causation; "
                "<em>Ellis</em> extended the line to the <em>DO</em> context. The "
                "doctrinal commitment is clear, but the operational record is uneven, "
                "and N10 is the node that surfaces that unevenness."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this through the §5.1.10 sub-nodes (10a–10d, "
                "sub-node treatment deferred) and through user attestation on the SCE "
                "tab when specific SCE evidence has been admitted but not given "
                "doctrinal weight in the case file's reasoning. The architecture also "
                "conditions N10 on N18 (SCE profile audit), which checks the case file "
                "against the <em>Gladue</em> / <em>Ewert</em> / <em>Morris</em> / "
                "<em>Ellis</em> tetrad as an integrated profile rather than as four "
                "independent checks. Where SCE has been admitted but the case-file "
                "reasoning treats it as background, N10's posterior elevates and the "
                "architecture flags the <em>Morris</em>-paragraph-97 connection gate."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to refuse the pattern under which SCE is "
                "\"considered\" without being given operational weight. <em>Morris</em> "
                "and <em>Ellis</em> tell us that admission without operationalisation is "
                "doctrinal failure; N10 carries that insight into the inference "
                "architecture so that the gap between admission and weight is visible "
                "and conditional on the doctrinal commitments the SCE record implies."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[2]:
        st.markdown(
            _foundation_header_html("N18", "SCE profile audit", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("Tetrad integration — Chapter 5 §5.1.18", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.18, N18 encodes the integrated audit of the case file "
                "against the doctrinal tetrad: <em>R. v. Gladue</em> [1999] 1 SCR 688, "
                "<em>Ewert v Canada</em> [2018] 2 SCR 165, <em>R. v. Morris</em> 2021 "
                "ONCA 680, and <em>R. v. Ellis</em> 2022 BCCA 278. Each of the four "
                "authorities makes a distinct doctrinal claim about social-context "
                "evidence — <em>Gladue</em> on the mandate to consider systemic and "
                "background factors for Indigenous offenders, <em>Ewert</em> on the "
                "validity of risk-assessment tools, <em>Morris</em> on the discernible-"
                "nexus standard, and <em>Ellis</em> on the application of the line in "
                "<em>DO</em> proceedings. N18 holds the integrated audit of whether the "
                "case file engages all four rather than checking each independently and "
                "missing the structural picture they form together."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this through the §5.1.18 sub-nodes (18a–18d, "
                "sub-node treatment deferred): jurisdiction sensitivity, SCE presence, "
                "SCE substance, and doctrinal-tagging completeness. The sub-nodes audit "
                "whether the case file has engaged each Tetrad commitment and surface "
                "the integrated profile. N18's posterior elevates where Tetrad "
                "integration is structurally incomplete — most consequentially where "
                "SCE has been admitted (engaging <em>Morris</em>) but tool validity has "
                "not been audited (failing <em>Ewert</em>), or where <em>Gladue</em> "
                "factors are present (engaging the mandate) but no doctrinal-tagging "
                "connects them to the <em>DO</em> analysis (failing <em>Ellis</em>). "
                "The Tetrad operates as an integrated profile, and N18 ensures that "
                "integration is performed rather than approximated by independent checks."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to make Tetrad-integration auditable. The "
                "four authorities constitute a coherent doctrinal architecture for "
                "Indigenous SCE in sentencing; treating them as four independent checks "
                "misses the integration that gives them their doctrinal force. N18 "
                "carries the integration into the inference architecture so that the "
                "tetradic structure is engaged rather than fragmented."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Section: Statutory and structural tensions ──
    st.markdown(
        "<div style='margin:24px 0 4px 0;font-family:JetBrains Mono,monospace;"
        "font-size:0.72rem;font-weight:700;color:#9E9E9E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Statutory and structural tensions</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:780px'>Architectural-level tensions in the doctrinal landscape itself that the case-file inference must reckon with.</div>",
        unsafe_allow_html=True,
    )
    _sec_cols = st.columns(2)
    with _sec_cols[0]:
        st.markdown(
            _foundation_header_html("N16", "Doctrinal tension", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("s.718.04 / s.718.2(e) — Chapter 5 §5.1.16", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.16, N16 encodes the doctrinal tension between "
                "s. 718.04 of the <em>Criminal Code</em> — which directs that primary "
                "consideration in cases involving violent offences against vulnerable "
                "victims be given to denunciation and deterrence — and s. 718.2(e) — "
                "which directs that all available sanctions other than imprisonment be "
                "considered, with particular attention to Indigenous offenders. For a "
                "<em>DO</em> analysis involving an Indigenous offender with a violent "
                "offence against a vulnerable victim, the two provisions point in "
                "opposite directions and the case law has not produced a settled "
                "hierarchy."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this by identifying when the case profile's "
                "facts implicate both provisions and surfacing the resulting tension as "
                "a doctrinal-architectural feature rather than as one provision "
                "overriding the other. N16's posterior elevates in cases where both "
                "provisions are doctrinally operative and conditions downstream nodes "
                "(N9 mitigation pathway, N20 designation output) by representing the "
                "unresolved hierarchy structurally. The architecture does not resolve "
                "the tension on the user's behalf — it ensures that the trier of fact "
                "engages both provisions rather than implicitly defaulting to one. The "
                "\"King impeachment\" framing carries the related concern that judicial "
                "reasoning may collapse the tension by treating one provision as "
                "silently dominant."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to make doctrinal-architectural tensions "
                "visible as such rather than letting them be resolved by implicit "
                "advocacy patterns. The s. 718.04 / s. 718.2(e) tension is the most "
                "consequential such tension in modern Indigenous-violent-offender "
                "sentencing; N16 ensures it is engaged rather than silently collapsed."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
    with _sec_cols[1]:
        st.markdown(
            _foundation_header_html("N17", "Over-policing", "#185FA5"),
            unsafe_allow_html=True,
        )
        with st.expander("Le / over-policing — Chapter 5 §5.1.17", expanded=False):
            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
                "<p style='margin:0 0 12px 0'>"
                "Per Chapter 5 §5.1.17 and <em>R. v. Le</em> 2019 SCC 34, N17 encodes "
                "the over-policing concern: that for individuals from communities "
                "subject to documented over-policing, the criminal record is at least "
                "partially a record of policing exposure rather than of conduct. "
                "<em>Le</em> held that the racial dynamics of policing are part of the "
                "legal landscape that courts must take judicial notice of; the over-"
                "policing literature documents systematic surveillance of Indigenous, "
                "Black, and racialised communities such that the production of records "
                "is itself a structural artefact of policing intensity rather than a "
                "neutral measure of conduct."
                "</p>"
                "<p style='margin:0 0 12px 0'>"
                "PARVIS operationalises this through the §5.1.17 sub-nodes (17a–17d, "
                "sub-node treatment deferred): jurisdictional policing disparity, "
                "enforcement-disparity engagement, non-violent charge density, and "
                "surveillance-triggered entries. The sub-nodes detect record patterns "
                "associated with over-policing and N17's posterior elevates accordingly. "
                "The architecture conditions downstream nodes — particularly N19 "
                "(collider bias) and N18 (SCE profile audit) — by representing the "
                "over-policing pathway through which records are produced. This is not "
                "a generalised mitigation claim; it is a structural-conditioning input "
                "on the evidentiary value of the record."
                "</p>"
                "<p style='margin:0'>"
                "The structural function is to refuse treating the criminal record as a "
                "neutral measure of conduct independent of the policing environment that "
                "produced it. <em>Le</em> tells us that environment is not neutral; N17 "
                "carries that recognition into the inference architecture so that "
                "record-derived evidence is conditioned on the policing pathway rather "
                "than treated as directly representative of conduct."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Section: Structural output ──
    st.markdown(
        "<div style='margin:24px 0 4px 0;font-family:JetBrains Mono,monospace;"
        "font-size:0.72rem;font-weight:700;color:#9E9E9E;text-transform:uppercase;"
        "letter-spacing:0.08em'>Structural output</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-family:Fraunces,serif;font-style:italic;"
        "font-size:0.84rem;color:#707070;margin-bottom:14px;line-height:1.55;"
        "max-width:780px'>The network's answer to s. 753 — where the preceding architectural commitments resolve.</div>",
        unsafe_allow_html=True,
    )
    # Full-width section — single treatment, no column
    st.markdown(
        _foundation_header_html("N20", "DO designation", "#993C1D"),
        unsafe_allow_html=True,
    )
    with st.expander("s. 753 designation — Chapter 5 §5.1.20", expanded=False):
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#3a3a3a;line-height:1.65;padding:4px 0'>"
            "<p style='margin:0 0 12px 0'>"
            "Per Chapter 5 §5.1.20 and s. 753 of the <em>Criminal Code</em> read "
            "with <em>R. v. Boutilier</em> 2017 SCC 64, N20 encodes the <em>Dangerous "
            "Offender</em> designation as the structural output of the entire "
            "network — the posterior probability the architecture associates with a "
            "<em>DO</em> finding given the conditioning of the case file through "
            "Layers I and II. The designation under s. 753 carries indeterminate "
            "detention as a possible disposition; <em>Boutilier</em> held that the "
            "future-treatment-prospects analysis is constitutionally required and "
            "that designation is not automatic from predicate findings. N20 is "
            "where the architectural commitments of the preceding nineteen nodes "
            "resolve into the designation question."
            "</p>"
            "<p style='margin:0 0 12px 0'>"
            "PARVIS operationalises this through Variable Elimination across the "
            "full network, with N20 receiving conditioning from all upstream Layer "
            "I and Layer II nodes. The architecture's structural commitments — N1's "
            "procedural admissibility gate, N5's tool-validity discount, N9's "
            "<em>Gladue</em> mitigation pathway, N17's over-policing conditioning, "
            "N18's Tetrad integration, N19's collider-bias correction — all resolve "
            "into N20's posterior. The architecture does not adjudicate the s. 753 "
            "question on the user's behalf; it computes the posterior implied by "
            "the case file's conditioning through the doctrinal architecture and "
            "surfaces the result alongside the structural-commitment trail that "
            "produced it."
            "</p>"
            "<p style='margin:0'>"
            "The structural function is to be the network's answer-shape: the "
            "question s. 753 asks, asked through the doctrinal architecture rather "
            "than through advocacy patterns or instrument outputs. The viva test "
            "for any <em>DO</em> architecture is whether it can produce a "
            "designation posterior accountable to its doctrinal commitments. "
            "PARVIS's answer is N20, conditioned on the nineteen nodes that precede "
            "it — not as a verdict but as the architectural shape of the s. 753 "
            "inquiry."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

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

    # ── §5.1.1 Burden-of-proof audit configuration (Mark 8) ───────────────
    # Per-case strict-mode flag (Q3=A). Stored in session_state and round-
    # trips on save/load — preserves audit semantics across user handoffs
    # (defence → Crown → bench review).
    with st.expander(
        "⚖ Burden-of-proof audit — case configuration",
        expanded=False,
    ):
        st.markdown(
            "<div style='font-family:Fraunces,serif;font-style:italic;"
            "font-size:0.86rem;color:#5A5A5A;margin-bottom:14px;line-height:1.55;"
            "max-width:780px'>"
            "PARVIS audits each evidentiary input against its applicable burden "
            "of proof per <em>R. v. Gardiner</em>, [1982] 2 SCR 368 + s. 724(3) "
            "<em>Criminal Code</em>. Crown-tendered aggravating facts must clear "
            "BARD; defence-tendered mitigating facts must clear BoP. Audit "
            "results drive N1 via virtual evidence so audit failures propagate "
            "structurally through the BN to N1's children. See §RM.1 register on "
            "Report tab for the full audit ledger."
            "</div>",
            unsafe_allow_html=True,
        )
        st.checkbox(
            "**Strict mode** — fire burden-of-proof audit prompt at moment "
            "of entry rather than at tab-exit",
            key="strict_mode",
            help=(
                "When enabled, each evidentiary input must be classified "
                "(provenance + use) and attested at the point of entry "
                "before the user can move on. When disabled (default), "
                "classification can be deferred to tab-exit or Report-"
                "generation review. Strict mode is recommended for "
                "courtroom-handoff scenarios where the audit must be "
                "complete before transmission. Setting persists with "
                "the case (round-trips on save/load)."
            ),
        )
        if st.session_state.get("strict_mode", False):
            st.markdown(
                "<div style='background:#FFF7E8;border-left:3px solid #BA7517;"
                "padding:8px 12px;margin-top:8px;border-radius:4px;"
                "font-size:0.82rem;color:#3A3A3A;line-height:1.5'>"
                "Strict mode is <strong>active</strong>. Note: full at-entry "
                "firing is implemented for the Criminal Record tab in the "
                "current build. Other tabs (Intake, Gladue, SCE, Risk &amp; "
                "Distortions) currently use tab-exit firing pending Mark 9 "
                "integration."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── §5.1.17 N17b counsel attestation (override path) ──────────────────
    # Per JP confirmation M3: primary signal is OR-gate over Gladue/SCE tab
    # evidence; this checkbox is the override for cases where documentation
    # exists but is not yet captured in the Gladue/SCE tab fields, or where
    # counsel's professional judgment supports engagement that the structured
    # fields don't fully accommodate.
    st.markdown(
        "<div style='margin:14px 0 4px 0; font-size:0.78rem; "
        "font-family:DM Sans, sans-serif; color:#5A5A5A; "
        "letter-spacing:0.04em; text-transform:uppercase;'>"
        "§5.1.17 audit signal (N17b)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.checkbox(
        "Counsel attests this case engages documented enforcement-disparity "
        "patterns per §5.1.17 §2",
        key="n17b_counsel_attestation",
        help=(
            "§5.1.17 §2 references documented over-policing patterns affecting "
            "Indigenous and Black communities (carding, proactive patrols, "
            "compliance-focused enforcement). The primary signal for N17b is "
            "automatically derived from Gladue/SCE tab fields you populate "
            "(over-policed community of origin, racial profiling documentation, "
            "anti-Black bail practices, etc.). Use this attestation only as an "
            "override where evidence engages §5.1.17 §2 patterns but is not yet "
            "captured in those structured fields."
        ),
    )

    # ── §5.1.14 / §5.1.15 / §5.1.18 attestations relocated to per-conviction ──
    # Mark 8 Phase 2 (per JP M8/P2 lock-in): N14a/b/c, N15a/b/c, N18b/c
    # attestations are now per-conviction on the Criminal Record tab. The
    # aggregate signals (any-prior aggregation) are derived via
    # _any_conviction_attests() in the signal-computation helpers. This
    # decluttering is doctrinally meaningful — a 1998 conviction can carry
    # "imposed under severe era ✓" while a 2018 prior carries it as ✗,
    # closer to how a sentencing judge would actually reason. Profile tab
    # retains only case-level attestations: N17b (over-policing — case-wide
    # policing-environment claim) and N18a (sentencing court's jurisdictional
    # SCE-integration precedent).

    # ── §5.1.18 N18a counsel attestation (case-level — jurisdiction precedent) ──
    # Per Q1=β + Q7=α: N18a concerns whether the present sentencing court's
    # jurisdiction has strong SCE-integration appellate precedent (Morris/
    # Ellis-equivalent). This is jurisdictional, not per-conviction —
    # belongs on Profile near case_jur input.
    st.markdown(
        "<div style='margin:14px 0 4px 0; font-size:0.78rem; "
        "font-family:DM Sans, sans-serif; color:#5A5A5A; "
        "letter-spacing:0.04em; text-transform:uppercase;'>"
        "§5.1.18 audit signal (N18a — jurisdictional)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.checkbox(
        "Counsel attests sentencing jurisdiction lacks strong provincial "
        "SCE-integration precedent (no Morris/Ellis-equivalent)",
        key="n18a_counsel_attestation",
        help=(
            "§5.1.18 §1 frames N18 as profiling whether prior convictions "
            "substantively integrated SCE/Tetrad. Provincial appellate-level "
            "SCE-integration scrutiny is strongest in Ontario post-Morris "
            "(2021 ONCA 680) and BC post-Ellis (2022 BCCA 278); SCC cases bind "
            "nationwide. The primary heuristic auto-detects ON/BC/SCC. Use "
            "this override where case-specific context indicates weaker "
            "SCE-integration scrutiny than the jurisdictional default suggests. "
            "N18b (SCE presence in reasons) and N18c (SCE substance) are now "
            "per-conviction on the Criminal Record tab."
        ),
    )

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
                "N2 · N3 · N4"
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
        st.markdown(_sev_caption(sev_word, sev_col, sev_body, "N2"),
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
        # TODO (Mark 8 / BUG 1 follow-up): UI label "Dynamic risk" with
        # subscript "N18" reflects pre-CH5 taxonomy. Under CH5, N18 = SCE
        # Profile audit (Gladue/Ewert/Morris/Ellis), not a substance/peer/
        # stability composite. This section's widgets (sub/peers/stab) no
        # longer drive any BN node — see BUG 1 fix at _sync_profile_from_
        # widgets and the parallel Profile tab posterior-calc block.
        # Pending doctrinal-mapping review: relabel section header, revise
        # subscript, and rewire widgets to appropriate CH5 nodes (likely
        # candidates: N9 IGT/cultural-treatment, or N18 sub-node signals).
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
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N10</div>",
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
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N17</div>",
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
                "N11"
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
            "color:#9E9E9E;margin-top:-8px;margin-bottom:14px;text-align:right'>N9</div>",
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
    # BUG 1 FIX (Mark 8): stale-taxonomy override of pev[18] and pev[19]
    # deleted from this Profile tab posterior-calc block, mirroring the
    # parallel fix in _sync_profile_from_widgets. See that function for full
    # rationale. The sub/peers/stab/rehab Profile tab widgets remain in place
    # (rehab still drives pev[13] gaming-risk detection on N13); their
    # remaining values await doctrinal-mapping review under CH5 taxonomy.
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
        "Intergenerational trauma":   "→ N9",
        "Cultural disconnection":     "→ N9",
        "Childhood & family":         "→ N9",
        "Socioeconomic":              "→ N9 · N18",
        "Substance & mental health":  "→ N8 · N9 · N18",
        "Systemic justice":           "→ N13 · N17",
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

    # ── §5.1.1 N1 audit reconciliation (Mark 8 Phase 2) ───────────────────
    # Each checked Gladue factor becomes an audited input with default
    # classification: defence-tendered, mitigating, BoP. Unchecked factors
    # have their audit records removed (orphan cleanup). Per JP M8/P2:
    # Pattern A defaults — user only sees audit UI when overriding or
    # when strict mode triggers tab-exit review.
    _gladue_audit_ids_now = {f"gladue_{fid}" for fid in cg}
    # Add audit records for newly-checked factors
    for f in GF:
        if f["id"] in cg:
            aid = f"gladue_{f['id']}"
            _ensure_audit_record(
                aid, "gladue", f"Gladue factor — {f['l']}"
            )
    # Remove audit records for newly-unchecked factors
    _existing = st.session_state.get("n1_audit", {})
    for aid in list(_existing.keys()):
        if aid.startswith("gladue_") and aid not in _gladue_audit_ids_now:
            _remove_audit_record(aid)

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

    # ── §5.1.1 N1 audit reconciliation (Mark 8 Phase 2) ───────────────────
    # Each active SCE factor becomes an audited input with default
    # classification: defence-tendered, mitigating, BoP (per Morris/Ellis
    # framework). Unchecked factors have audit records removed.
    _sce_audit_ids_now = {f"sce_{fid}" for fid in st.session_state.sce_checked}
    for f in SF:
        if f["id"] in st.session_state.sce_checked:
            aid = f"sce_{f['id']}"
            _ensure_audit_record(
                aid, "sce", f"SCE factor — {f['l']}"
            )
    _existing_sce = st.session_state.get("n1_audit", {})
    for aid in list(_existing_sce.keys()):
        if aid.startswith("sce_") and aid not in _sce_audit_ids_now:
            _remove_audit_record(aid)

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

    # ── §5.1.1 N1 audit reconciliation (Mark 8 Phase 2) ───────────────────
    # Each manually-overridden node becomes an audited input. Default
    # classification by node type per JP M8/P2 lock-in:
    #   "risk"       → Crown-tendered, aggravating, BARD (N2/N3/N4)
    #   "distortion" → defence-tendered, mitigating, BoP (N5/N6/N7/N10/etc.)
    #   "mitigation" → defence-tendered, mitigating, BoP (N9 IGT)
    #   "dual"       → defence-tendered, mitigating, BoP (N8 FASD — JP lean)
    #   "special"    → no default (prompt; N11 gaming detector)
    # Override functionality (changing the default for a specific node) is
    # available via the §RM.1 register on the Report tab and via strict-mode
    # at-tab-exit review. Most users will not need to override defaults.
    _risk_type_to_default = {
        "risk":       ("crown",   "aggravating"),
        "distortion": ("defence", "mitigating"),
        "mitigation": ("defence", "mitigating"),
        "dual":       ("defence", "mitigating"),
        # "special" → no entry (default is None, triggers no-default path)
    }
    _risk_audit_ids_now = set()
    for _nid in st.session_state.manual_ev.keys():
        # manual_ev keys are the int node IDs of overridden nodes
        _meta = NODE_META.get(_nid, {})
        _type = _meta.get("type", "")
        _label = _meta.get("short", f"Node {_nid}")
        _default = _risk_type_to_default.get(_type)
        _aid = f"risk_n{_nid}"
        _risk_audit_ids_now.add(_aid)
        if _default:
            _prov, _use = _default
        else:
            # No default for this node type ("special"). Create an audit
            # record with unspecified provenance/use so the user is
            # prompted via §RM.1 register or strict-mode review.
            _prov, _use = None, None
        _ensure_audit_record(
            _aid, "risk_substantive" if _type == "risk"
                  else ("risk_distortion" if _type == "distortion"
                        else ("risk_mitigation" if _type in ("mitigation", "dual")
                              else "risk_other")),
            f"Risk override — N{_nid} {_label}",
            provenance=_prov, use=_use,
        )
    # Remove audit records for nodes no longer overridden
    _existing_risk = st.session_state.get("n1_audit", {})
    for aid in list(_existing_risk.keys()):
        if aid.startswith("risk_n") and aid not in _risk_audit_ids_now:
            _remove_audit_record(aid)

    # ── Reset action ──────────────────────────────────────────────────────
    rsc1, rsc2, rsc3 = st.columns([1, 1, 2])
    with rsc1:
        if st.button("Reset all to priors", key="rst"):
            for k in ["profile_ev", "manual_ev", "doc_adj"]:
                st.session_state[k] = {}
            st.session_state.gladue_checked = set()
            st.session_state.sce_checked = set()
            # Mark 8 Phase 2: also clear audit records sourced from these
            # tabs (gladue_*, sce_*, risk_n*). Criminal Record audits are
            # left intact since reset doesn't clear convictions.
            _audit = st.session_state.get("n1_audit", {})
            for aid in list(_audit.keys()):
                if (aid.startswith("gladue_") or aid.startswith("sce_")
                        or aid.startswith("risk_n")):
                    _audit.pop(aid, None)
            st.session_state.n1_audit = _audit
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
            if nid == 1:
                # Mark 8 Phase 3 — N1 grid chip uses compact doctrinal-state
                # framing per JP M8/P3 lock-in (Option II, full consistency).
                _n1_grid = _n1_audit_summary()
                _n1_grid_accent = _n1_grid["color_accent"]
                _n1_grid_bg = _n1_grid["color_bg"]
                _n1_grid_border = _n1_grid["color_border"]
                _n1_grid_label = _n1_grid["label"]
                st.markdown(
                    f"<div style='background:{_n1_grid_bg};border:1px solid {_n1_grid_border};"
                    f"border-left:2px solid {_n1_grid_accent};border-radius:8px;"
                    f"padding:.55rem .7rem;margin-bottom:.4rem'>"
                    f"<div style='font-size:.65rem;color:{_n1_grid_accent};"
                    f"font-weight:700'>N{nid} — {m['short']}</div>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-size:.92rem;"
                    f"font-weight:500;color:{_n1_grid_accent};line-height:1.15;"
                    f"margin-top:2px'>{_n1_grid_label}</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:.7rem;"
                    f"color:{_n1_grid_accent};margin-top:2px'>"
                    f"<span style='font-size:.58rem;color:#9E9E9E;text-transform:uppercase;"
                    f"letter-spacing:.05em;margin-right:4px'>depth</span>"
                    f"{p*100:.1f}%</div>"
                    f"<div style='height:4px;background:#eee;border-radius:2px;margin-top:3px'>"
                    f"<div style='width:{p*100:.0f}%;height:100%;background:{_n1_grid_accent};"
                    f"border-radius:2px'></div></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"""<div style="background:{col}18;border:1px solid {col}33;border-radius:8px;
                padding:.55rem .7rem;margin-bottom:.4rem">
                <div style="font-size:.65rem;color:{col};font-weight:700">N{nid} — {m['short']}</div>
                <div style="font-size:1.1rem;font-weight:700;font-family:monospace;color:{col}">{p*100:.1f}%</div>
                <div style="height:4px;background:#eee;border-radius:2px;margin-top:3px">
                  <div style="width:{p*100:.0f}%;height:100%;background:{col};border-radius:2px"></div>
                </div></div>""",unsafe_allow_html=True)

    # ── §5.1.19 N19 collider-bias methodology expander ─────────────────────
    # Sits on Inference tab (where structural-distortion content lives)
    # alongside the per-node posterior grid. Surfaces both the BN posterior
    # mechanism (§6 CPT) and the Q4=C secondary discount mechanism (§8).
    if not _empty:
        with st.expander("📚 §5.1.19 collider-bias detection — how N19 is computed and applied"):
            _signal = st.session_state.get("n19_collider_signal") or {}
            _disc_risk = st.session_state.get("n19_discounted_risk")
            _n14_p = float(P.get(14, 0.5))
            _n17_p = float(P.get(17, 0.5))
            _n19_p = float(P.get(19, 0.30))
            _headline = float(P.get(20, 0.30))

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.88rem;color:#3A3A3A;line-height:1.55;margin-bottom:14px'>"
                "Per Chapter 5 §5.1.19 §1, N19 surfaces collider bias arising "
                "when sentencing inference conditions on system-entry variables "
                "(arrest, charge, conviction) that are simultaneously caused "
                "by upstream factors including over-policing intensity and "
                "case complexity. The architectural commitment is "
                "<strong style='font-weight:500;color:#1A1A1A;font-style:normal'>"
                "non-evidentiary</strong> — N19 does not add evidence to the "
                "headline DO posterior. Instead, it flags when the inference "
                "drawn from the criminal record may be systematically "
                "unreliable due to the geometry of the reasoning itself "
                "(Berkson&apos;s paradox)."
                "</div>",
                unsafe_allow_html=True,
            )

            # §6 CPT mechanism card
            _cpt_color = "#A32D2D" if _n19_p >= 0.50 else "#185FA5"
            st.markdown(
                f"<div style='border-left:3px solid {_cpt_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>BN posterior — §5.1.19 §6 mechanism</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_cpt_color};font-weight:600'>"
                f"P(N19=High) = {_n19_p*100:.1f}%</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:6px;line-height:1.5'>"
                f"Computed by Variable Elimination over §5.1.19 §6&apos;s 2-parent CPT: "
                f"P(N19=High) at (Over-Policing × Case Complexity) = "
                f"(L,L)→0.25, (L,H)→0.50, (H,L)→0.50, (H,H)→0.85. "
                f"N17 (over-policing) is the §6 first parent; N14 (temporal "
                f"distortion) is the doctrinal proxy for §6&apos;s "
                f"&quot;Case Complexity&quot; — N14&apos;s era-of-sentencing "
                f"severity captures historical case-complexity drivers. "
                f"Current parent posteriors: N14={_n14_p*100:.1f}%, "
                f"N17={_n17_p*100:.1f}%."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # App-side joint-elevation signal card
            _signal_active = bool(_signal.get("active", False))
            _signal_color = "#A32D2D" if _signal_active else "#3B6D11"
            _signal_text = "ACTIVE — both parents above threshold" if _signal_active else "Inactive"
            _threshold = _signal.get("threshold", 0.60)
            st.markdown(
                f"<div style='border-left:3px solid {_signal_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>App-side joint-elevation signal (Q6=α)</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_signal_color};font-weight:600'>{_signal_text}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:6px;line-height:1.5'>"
                f"Independent pattern-detection check: fires when both N14 "
                f"and N17 are jointly elevated above threshold "
                f"({_threshold:.2f}). Complements the BN posterior by "
                f"making the structural condition visible without requiring "
                f"the user to interpret the probabilistic computation. "
                f"Per §5.1.19 §1, this signal does not add evidence — it "
                f"surfaces the structural condition under which collider "
                f"bias is doctrinally implicated."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # §8 secondary-discount mechanism card
            if _disc_risk is not None:
                _delta_pp = (_disc_risk - _headline) * 100
                _disc_color = "#A32D2D" if _delta_pp <= -1.0 else "#9E9E9E"
                st.markdown(
                    f"<div style='border-left:3px solid {_disc_color};padding:.6rem .9rem;"
                    f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                    f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                    f"color:#1A1A1A;font-size:0.94rem'>Secondary collider-discounted risk (Q4=C)</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                    f"color:{_disc_color};font-weight:600'>"
                    f"Headline {_headline*100:.1f}% &nbsp;→&nbsp; Discounted "
                    f"{_disc_risk*100:.1f}% &nbsp;({_delta_pp:+.1f}pp)</div>"
                    f"</div>"
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.78rem;color:#5A5A5A;margin-top:6px;line-height:1.5'>"
                    f"Per §5.1.19 §8, when collider bias is active the final "
                    f"risk score is multiplicatively discounted by "
                    f"(1 - 0.30 × N19) to reflect causal uncertainty rather "
                    f"than inflated confidence. At §6 baseline (N19=0.25) "
                    f"the discount is 7.5%; at §6 peak (N19=0.85) the "
                    f"discount is 25.5%. This number is "
                    f"<strong style='font-weight:500;color:#1A1A1A;"
                    f"font-style:normal'>contingent</strong> — it is "
                    f"displayed alongside the headline rather than replacing "
                    f"it, preserving the §5.1.19 §1 commitment that N19 does "
                    f"not add evidence to the inference."
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.80rem;color:#707070;margin-top:14px;line-height:1.55'>"
                "<strong style='font-weight:500;color:#3A3A3A;font-style:normal'>"
                "Doctrinal note.</strong> "
                "§5.1.19 §1 (&quot;not to add evidence to the inference&quot;) "
                "and §8 (&quot;final risk scores reflect causal uncertainty&quot;) "
                "are reconciled by the architecture: the §1 commitment is "
                "preserved at the headline-posterior level, and the §8 "
                "mechanism operates at the secondary-display level. Defence "
                "counsel reviewing the architecture&apos;s output sees both "
                "numbers and can address the inference-integrity concern in "
                "submissions without the architecture pre-empting the legal "
                "argument with a single contestable figure."
                "</div>",
                unsafe_allow_html=True,
            )

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
            "φ encodes the balance between substantive-risk nodes (N2 / N3 / N4) "
            "and systemic-distortion nodes (N5 / N6 / N10 / N12 / N14 / N17). Both require "
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
        if _node_options:
            _node_choice = st.selectbox(
                "Select node",
                _node_options,
                key="canlii_node_pick",
            )
        else:
            _node_choice = None
            st.caption("_Per-node search unavailable — CanLII module not loaded._")
    _selected_nid = int(_node_choice.split(" ")[0][1:]) if _node_choice else None
    with col_pn_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _run_node = st.button("🔍 Search recent cases for this node", key="node_search_btn", disabled=not _canlii_active)
        if not _canlii_active:
            st.caption("_Configure CanLII API access to enable._")
    if _run_node and _selected_nid is not None:
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
- N4 Dynamic risk: {Pa.get(4,0.167)*100:.1f}%

DISTORTION CORRECTIONS (reduce effective risk weight):
- N1 Burden of proof — Doctrinal posture: {_n1_audit_summary()['label']} (audit-pressure depth: {Pa.get(1,0.83)*100:.1f}%; this is an operationalisation of doctrinal posture toward the case-file audit state, NOT a probability over case facts; per Gardiner [1982] 2 SCR 368, asymmetric burdens apply per evidentiary input)
- N5 Risk tools: {Pa.get(5,0.10)*100:.1f}%
- N6 Ineffective counsel: {Pa.get(6,0.15)*100:.1f}%
- N7 Bail-denial cascade: {Pa.get(7,0.40)*100:.1f}%
- N9 FASD: {Pa.get(9,0.15)*100:.1f}%
- N9 Intergenerational trauma: {Pa.get(9,0.45)*100:.1f}%
- N9 No cultural treatment: {Pa.get(9,0.10)*100:.1f}%
- N10 Gladue misapplication: {Pa.get(10,0.15)*100:.1f}%
- N17 Over-policing: {Pa.get(17,0.15)*100:.1f}%
- N14 Temporal distortion: {Pa.get(14,0.40)*100:.1f}%
- N9 No rehabilitation: {Pa.get(9,0.10)*100:.1f}%

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
    pN17 = st.session_state.posteriors.get(17, 0.15)   # over-policing (CH5 §5.1.17)
    pN14 = st.session_state.posteriors.get(14, 0.40)   # temporal distortion / age (CH5 §5.1.14)
    pN10 = st.session_state.posteriors.get(10, 0.15)   # Gladue misapplication (CH5 §5.1.10)

    # ── Authoritative correction references (preserved) ───────────────────
    CORR_REFS = {
        "bail":    ("Bail-denial cascade (N7)", "A32D2D", "R v Antic [2017] SCC 27; Tolppanen Report (2018)"),
        "counsel": ("Ineffective counsel (N6)", "185FA5", "R v GDB [2000] 1 SCR 520; Strickland doctrine"),
        "ewert":   ("Ewert tool invalidity (N5)", "185FA5", "Ewert v Canada [2018] SCC 30"),
        "police":  ("Over-policing / record inflation (N17)", "185FA5", "R v Le [2019] SCC 34"),
        "time":    ("Temporal attenuation (N14)", "185FA5", "R v Nur [2015] / Lloyd [2016] — mandatory-minimum-era / age burnout"),
        "gladue":  ("Gladue misapplication (N10)", "3B6D11", "R v Morris 2021 ONCA 680 para 97; Boutilier [2017] SCC 64"),
    }

    # ── Compute global reliability floor from current distortion posteriors ───
    bail_factor    = float(np.clip(1.0 - 0.50*pN7 - 0.20*pN6, 0.30, 1.0))
    ewert_factor   = float(np.clip(1.0 - 0.40*pN5,             0.40, 1.0))
    police_factor  = float(np.clip(1.0 - 0.35*pN17,            0.50, 1.0))
    gladue_factor  = float(np.clip(1.0 - 0.30*pN10,            0.55, 1.0))

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
                    f"Temporal attenuation applicable per age burnout (N14).")
        else:
            pat = "stable"
            signal = 0.0
            note = "Stable offence pattern — no significant escalation or de-escalation detected."
        return {"pattern":pat,"signal":signal,"note":note}

    def _cr_feed_nodes():
        """Derive N2, N18 and distortion node signals from the calibrated criminal record.
        Fixes: N2 signal now written to doc_adj (not cr_doc_adj) so run_inf() picks it up.
        Enhanced: seriousness-weighted N2, escalation signal into N4 (dynamic risk), gang context into N17 (over-policing).
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

        # ── N4: Dynamic risk — escalation signal (CH5 §5.1.4) ────────────────
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
            # Domestic violence → boosts violent history (N2) and dynamic risk (N4)
            cr_adj[2]  = cr_adj.get(2, 0)  + float(np.clip(0.05 * len(domestic_e), 0.0, 0.15))
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.04 * len(domestic_e), 0.0, 0.12))
        if hate_e:
            # Hate crime → boosts dynamic risk (N4)
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.06 * len(hate_e), 0.0, 0.18))
        if terror_e:
            # Terrorism → significant boost to N4 (dynamic risk) and N2 (validated risk elevators)
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.12 * len(terror_e), 0.0, 0.30))
            cr_adj[2]  = cr_adj.get(2, 0)  + float(np.clip(0.10 * len(terror_e), 0.0, 0.25))
        if vuln_e:
            # Vulnerable victim → boosts dynamic risk (N4)
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(0.05 * len(vuln_e), 0.0, 0.15))
        if drug_e:
            # Drug trafficking — Parranto [2021] — fentanyl at top of tariff
            fentanyl_e = [e for e in drug_e if "fentanyl" in e.get("drug_type","").lower()
                          or "carfentanil" in e.get("drug_type","").lower()]
            cr_adj[18] = cr_adj.get(18, 0) + float(np.clip(
                0.06 * len(drug_e) + 0.06 * len(fentanyl_e), 0.0, 0.22))

        # ── Distortion aggregates → N7 (bail-WCGP), N17 (over-policing), N10 (Gladue misapp) ─
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

        # ─── Subsection: §5.1.18 SCE-integration tagging (per-conviction) ───
        # Per JP confirmation Q1=β: per-conviction tags exposed in UI;
        # tags drive N18b (SCE Presence) and N18c (SCE Substance) sub-node
        # aggregate signals. Default "Absent" (most conservative — defers to
        # Morris Heuristic Audit empirical finding that nominal-or-absent is
        # the modal failure mode).
        st.markdown(
            "<div style='font-size:0.66rem;text-transform:uppercase;"
            "letter-spacing:0.14em;color:#707070;font-weight:600;"
            "margin:18px 0 12px 0;padding-top:14px;"
            "border-top:1px solid #EFEDE7'>§5.1.18 SCE integration tag</div>",
            unsafe_allow_html=True,
        )
        cr_sce_tag = st.selectbox(
            "Tetrad / SCE integration in reasons for sentence",
            options=["Absent", "Nominal", "Partial", "Full"],
            index=0,  # default "Absent"
            key="cr_sce_tag",
            help=(
                "Per §5.1.18 §1, this profiles whether the original conviction "
                "substantively integrated SCE / Tetrad jurisprudence — not "
                "whether the sentence was correct. Tag values:\n\n"
                "• **Full** — SCE substantively integrated into reasons\n"
                "• **Partial** — SCE referenced and partially integrated\n"
                "• **Nominal** — SCE mentioned but not substantively engaged "
                "(Morris Heuristic Audit modal failure mode)\n"
                "• **Absent** — SCE not mentioned in reasons (default)\n\n"
                "Aggregates over per-conviction tags drive N18b (Presence) "
                "and N18c (Substance). Counsel attestations on the Profile "
                "tab can override the per-conviction tag aggregates."
            ),
        )

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

        # ── N6 (Ineffective Assistance of Counsel) indicators per §5.1.6 §3 ──
        # Four binary indicators aggregate per §RM.6.3 to a confidence grade
        # (High / Moderate / Low) which conditions the N7 cascade signal.
        st.markdown(
            "<div style='margin-top:14px;margin-bottom:6px'>"
            "<span style='font-family:Fraunces,Georgia,serif;font-size:0.95rem;"
            "font-weight:500;color:#1A1A1A'>Representation quality indicators (N6)</span>"
            "<span style='font-family:Fraunces,serif;font-style:italic;font-size:0.78rem;"
            "color:#707070;margin-left:8px'>Ch 5 §5.1.6 · Appendix RM §RM.6</span>"
            "<div style='font-family:Fraunces,serif;font-style:italic;font-size:0.78rem;"
            "color:#707070;margin-top:3px;line-height:1.5'>"
            "Check each indicator that is adverse for this conviction. Any adverse "
            "indicator drops confidence to <em>Moderate</em>; all four adverse drops "
            "to <em>Low</em>. Confidence grade conditions the N7 bail-denial cascade "
            "signal additively (+0.00 / +0.10 / +0.20)."
            "</div></div>",
            unsafe_allow_html=True,
        )
        cn1, cn2 = st.columns(2)
        with cn1:
            adj_n6_no_sce = st.checkbox(
                "SCE not submitted",
                value=False,
                key="cr_adj_n6_no_sce",
                help="§5.1.6 §3.i — adverse where Gladue, IRCA, or equivalent systemic "
                     "context evidence was not meaningfully advanced at the original proceeding")
            adj_n6_inadequate_counsel = st.checkbox(
                "Counsel culturally inadequate",
                value=False,
                key="cr_adj_n6_inadequate_counsel",
                help="§5.1.6 §3.ii — adverse where counsel did not demonstrate familiarity "
                     "with applicable social context doctrine and its evidentiary use")
        with cn2:
            adj_n6_judicial_criticism = st.checkbox(
                "Judicial criticism of representation",
                value=False,
                key="cr_adj_n6_judicial_criticism",
                help="§5.1.6 §3.iii — adverse where the original court expressed concern, "
                     "criticism, or reservation regarding the quality of advocacy")
            adj_n6_disproportionate = st.checkbox(
                "Procedural outcome disproportionate",
                value=False,
                key="cr_adj_n6_disproportionate",
                help="§5.1.6 §3.iv — adverse where the procedural outcome was markedly "
                     "disproportionate relative to offence gravity and comparator cases")

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
                    # §5.1.18 per-conviction SCE-integration tag (Q1=β)
                    "sce_integration_tag": cr_sce_tag,
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
                        # N6 (IAC) indicators per §5.1.6 / §RM.6
                        "n6_no_sce":              adj_n6_no_sce,
                        "n6_inadequate_counsel":  adj_n6_inadequate_counsel,
                        "n6_judicial_criticism":  adj_n6_judicial_criticism,
                        "n6_disproportionate":    adj_n6_disproportionate,
                    },
                    "raw_weight":  1.0,
                    "cal_weight":  cal_wt,
                }
                # ── §5.1.1 N1 audit (Mark 8) — assign stable audit_id ─────
                # before append so that _audit_record_for_conviction can
                # use it. Index is len(criminal_record) before the append
                # — i.e. the position this entry will occupy.
                _new_idx = len(st.session_state.criminal_record)
                entry["audit_id"] = _conviction_audit_id(
                    _new_idx, entry["year"], entry.get("offence", "")
                )
                # Create default audit record (use="contextual", no audit
                # triggered until user toggles Crown-reliance flag).
                _audit_record_for_conviction(entry, _new_idx)
                st.session_state.criminal_record.append(entry)
                # Sort chronologically (earliest first) — stable, by year only
                # (so insertions of multiple convictions in the same year retain
                # the order in which they were added).
                st.session_state.criminal_record.sort(
                    key=lambda e: int(e.get("year", 0))
                )
                # Reconcile audit IDs after the sort (Mark 8 N1 audit).
                # Removes orphans and ensures every conviction has an
                # audit_id matching its current index.
                _sync_conviction_audit_ids()
                # Feed calibrated N2 and distortion corrections back into the network
                _cr_feed_nodes()
                st.rerun()
            else:
                st.warning("Please enter an offence description.")

    # ── Record table ──────────────────────────────────────────────────────────
    rec = st.session_state.criminal_record
    if rec:
        # ── Mark 8 hotfix: hoist chronological grading to top of block ─────
        # Pre-existing latent bug — _n7_grading and _jump_chain were defined
        # later in the per-conviction loop entry (around line 8042 in current
        # numbering), but the Methodology — N7 reliability modifier expander
        # (further down inside this block) references _n7_grading. Streamlit
        # expanders execute their contents on every render regardless of
        # expanded= state, so when rec was non-empty the methodology block
        # would NameError on _n7_grading before the per-conviction loop
        # defined it. Hoisting to top of `if rec:` makes the definitions
        # available throughout the block. Original definitions further down
        # are left in place (harmless re-computation, ~5ms; preserves the
        # original code's locality of reference for the per-conviction loop).
        _jump_chain = _jump_cumulative_chain(rec)
        _n7_grading = _n7_grades_chronological(rec)

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
                f"— the conviction's own bail-denial signal, additively conditioned "
                f"by N6 representation quality (§RM.6.4) and multiplied by §RM.5 "
                f"cascade propagation from earlier tainted convictions where "
                f"applicable. This node never removes convictions; it qualifies how "
                f"they may be used. Multipliers — 1.00 / 0.60 / 0.30 — are "
                f"conservative operationalisations of the §5.1.7 ordinal grades."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Methodology disclosure expander
            with st.expander("Methodology — N7 reliability modifier", expanded=False):
                st.markdown("""
**Mechanism (Chapter 5 §5.1.7).** The bail-denial cascade node (N7) tracks the procedural conditions under which prior convictions were produced. Where bail was denied and ineffective representation, absent social context evidence, or marginalisation cluster, the conviction's evidentiary reliability is qualified — not removed.

**Per-conviction grading.** Each conviction is graded on its own facts. The architecture reads the conviction's own `adj.bail` value (set when the conviction was added), applies the §RM.6.4 N6 confidence boost (additive, +0.00/+0.10/+0.20 for High/Moderate/Low representation confidence), then applies the §RM.5 propagation factor (multiplicative, from any earlier tainted convictions), and finally applies the threshold logic below. Per §5.1.7, the unit of analysis is the specific guilty plea produced under coercive procedural conditions, not the offender's record as a whole — convictions produced under fair conditions are not discounted on the basis of cascade conditions affecting other convictions on the record.

**Cascade propagation (§RM.5).** Where Conviction A on the record has been graded *Discounted* or *Heavily Discounted*, and a subsequent Conviction B has its own affirmative bail-denial signal, the architecture recognises that the bail conditions affecting Conviction B may themselves have been conditioned by Conviction A's presence on the record (per *R v Antic* on prior records and bail). Conviction B's bail-denial signal is multiplied by a propagation factor before threshold logic — 1.15 for upstream *Discounted*, 1.30 for upstream *Heavily Discounted*. Where multiple upstream tainted convictions exist, only the strongest factor is applied. Propagation requires the downstream conviction to have its own affirmative signal — bail granted on Conviction B breaks the chain.

**Thresholds.**
- Per-conviction signal (after any propagation) < 0.30 → **Unmodified** (multiplier 1.00)
- 0.30 ≤ signal ≤ 0.65 → **Discounted** (multiplier 0.60)
- Signal > 0.65 → **Heavily Discounted** (multiplier 0.30)

**Aggregate.** The N7-adjusted record weight is the mean across convictions of (cal_weight × N7_multiplier). The nominal record weight is the mean of cal_weight alone. The difference is the N7 re-weighting effect on the record as a whole.

**Relation to the jump principle.** This grading also feeds the jump principle's own_ceiling computation: a conviction graded *Heavily Discounted* contributes to forward contamination at 30% of nominal anchoring weight (§RM.5.5), reflecting the architecture's prior judgment that the conviction's severity reflects cascade contamination rather than legitimate sentencing assessment.

**Doctrinal source.** *R v Antic* [2017] SCC 27 (bail jurisprudence); Tolppanen Report (2018) on bail-denial cascade dynamics; Chapter 5 §5.1.7 (Wrongful Conviction Guilty Plea cascade modelling); Chapter 3 §3.4.5 (inferential inertia) and §3.5.3 (jump principle) for the doctrinal substrate of cascade propagation; Appendix RM §RM.5 for the operational specification.

**Scope of this implementation.** Currently operational: N6 (IAC, conditioning per §RM.6) and N7 (bail-WCGP cascade with §RM.5 propagation). The full set of cascade distortions per Chapter 5 — N14 (Temporal Distortion), N15 (Interjurisdictional Tariff), N17 (Over-Policing and Epistemic Contamination), N18 (Gladue/Ewert/Morris/Ellis Profile audit), N19 (Collider Bias) — extends the same architectural pattern in subsequent implementation work.
                """)

            # ── N6 (IAC) aggregate panel — §RM.6 ─────────────────────────────
            # Per-conviction confidence grades and aggregate distribution.
            _n6_grades_per_conv = [g for (_g7, _m7, _p7, g) in _n7_grading]
            _n6_count = {
                "High":     sum(1 for g in _n6_grades_per_conv if g == "High"),
                "Moderate": sum(1 for g in _n6_grades_per_conv if g == "Moderate"),
                "Low":      sum(1 for g in _n6_grades_per_conv if g == "Low"),
            }
            _n6_avg_boost = (
                sum(N6_BOOST_BY_GRADE[g] for g in _n6_grades_per_conv) / len(_n6_grades_per_conv)
                if _n6_grades_per_conv else 0.0
            )
            # Accent colour: keyed on average boost magnitude
            if _n6_avg_boost < 0.03:
                _n6_accent = "#3B6D11"; _n6_bg = "#F4F8EE"; _n6_border = "#C5D7AC"
            elif _n6_avg_boost < 0.12:
                _n6_accent = "#BA7517"; _n6_bg = "#FAEEDA"; _n6_border = "#E5CC95"
            else:
                _n6_accent = "#A32D2D"; _n6_bg = "#FCEBEB"; _n6_border = "#E5B5B5"

            st.markdown(
                f"<div style='background:{_n6_bg};border:1px solid {_n6_border};"
                f"border-left:3px solid {_n6_accent};border-radius:8px;"
                f"padding:14px 18px;margin:14px 0 8px 0'>"
                # Header row
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:10px'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"font-weight:600;padding:2px 8px;border-radius:4px;color:white;"
                f"background:{_n6_accent}'>N6</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:1.0rem;"
                f"font-weight:500;color:#1A1A1A'>"
                f"Reliability of Representation · IAC conditioning</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#707070;margin-left:auto'>"
                f"Ch 5 §5.1.6 · Appendix RM §RM.6</div>"
                f"</div>"
                # Body: average boost and grade distribution
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;"
                f"margin-top:6px'>"
                # Average boost
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:#707070;font-weight:600;margin-bottom:3px'>Average boost</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:1.45rem;"
                f"font-weight:600;color:{_n6_accent}'>+{_n6_avg_boost:.3f}</div>"
                f"<div style='font-size:0.72rem;color:#9E9E9E;font-family:Fraunces,serif;"
                f"font-style:italic'>mean N6 conditioning</div>"
                f"</div>"
                # Mechanism note
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:{_n6_accent};font-weight:600;margin-bottom:3px'>Mechanism</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.84rem;"
                f"font-weight:500;color:{_n6_accent};line-height:1.3'>Additive · upstream of N7</div>"
                f"<div style='font-size:0.72rem;color:#9E9E9E;font-family:Fraunces,serif;"
                f"font-style:italic'>conditions cascade signal</div>"
                f"</div>"
                # Grade distribution
                f"<div>"
                f"<div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.10em;"
                f"color:#707070;font-weight:600;margin-bottom:3px'>Confidence distribution</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.82rem;"
                f"color:#3A3A3A;line-height:1.5'>"
                f"<div>High · {_n6_count['High']}</div>"
                f"<div>Moderate · {_n6_count['Moderate']}</div>"
                f"<div>Low · {_n6_count['Low']}</div>"
                f"</div>"
                f"</div>"
                f"</div>"
                # Doctrinal caption
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.80rem;color:#707070;margin-top:12px;line-height:1.55;"
                f"padding-top:10px;border-top:1px solid {_n6_border}'>"
                f"Each conviction is graded against §5.1.6's tri-state ordinal "
                f"(High / Moderate / Low confidence) on its own facts — the count of "
                f"adverse indicators among SCE submission, counsel competence, judicial "
                f"commentary, and procedural outcome. The grade additively boosts the "
                f"N7 bail-denial signal (+0.00 / +0.10 / +0.20 per §RM.6.4) before "
                f"§RM.5 propagation. N6 does not directly modify cal_weight per §RM.6.6."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Methodology disclosure expander for N6
            with st.expander("Methodology — N6 reliability of representation", expanded=False):
                st.markdown("""
**Mechanism (Chapter 5 §5.1.6).** Node 6 models the reliability of representation that produced each conviction. Per §5.1.6 §1, the node does *not* adjudicate constitutional ineffective-assistance breach; it conditions the architecture's confidence in the products of representation that may have been deficient.

**Indicators (§5.1.6 §3).** Four binary indicators per conviction:

1. **SCE not submitted** — adverse where *Gladue*, *IRCA*, or equivalent systemic context evidence was not meaningfully advanced.
2. **Counsel culturally inadequate** — adverse where counsel did not demonstrate familiarity with applicable social context doctrine and its evidentiary use.
3. **Judicial criticism of representation** — adverse where the original court expressed concern, criticism, or reservation regarding the quality of advocacy.
4. **Procedural outcome disproportionate** — adverse where the procedural outcome was markedly disproportionate relative to offence gravity and comparator cases.

**Threshold scheme (§RM.6.3).** Confidence grade keyed to count of adverse indicators:
- 0 adverse → **High** confidence
- 1, 2, or 3 adverse → **Moderate** confidence
- 4 adverse → **Low** confidence (compound case)

**Conditioning of N7 (§RM.6.4).** N6 grade additively boosts the per-conviction bail-denial signal *before* §RM.5 propagation:
- High → +0.00 (no boost)
- Moderate → +0.10
- Low → +0.20

The boost reflects §5.1.7 §2: "Bail denial combined with ineffective counsel materially increases the probability that a plea reflects constraint rather than culpability." Where the conviction has no own bail-denial signal (`adj.bail = 0`), N6 conditioning does not apply — N6 conditions how strongly we read bail-denial evidence, but does not substitute for its absence.

**Order of operations (§RM.6.5).** `boosted_signal = (adj.bail + N6_addition) × propagation_factor`. N6 conditioning is per-conviction; §RM.5 propagation is between-conviction. Per-conviction conditioning logically precedes between-conviction propagation.

**Scope (§RM.6.6).** N6 does *not* directly discount `cal_weight` — its sole architectural role is to condition N7's cascade computation, avoiding double-counting. N6 does not propagate forward like §RM.5 — different counsel, different files, different proceedings. N6 does not adjudicate constitutional breach — *Low* confidence is an architectural assessment, not a finding.

**Doctrinal source.** *Chapter 5 §5.1.6* (IAC node specification); *Charter* ss. 7 and 15 (unequal access to effective representation); the standard articulated in *R v R.R.* and similar competence jurisprudence; *Appendix RM §RM.6* (operational specification).
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
            _n7_grade_card, _n7_mult_card, _n7_prop_card, _n6_grade_card = _n7_grading[i]
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
                # N6 confidence in representation (Chapter 5 §5.1.6 / §RM.6)
                f"<div style='border-top:1px solid #EFEDE7;padding-top:6px;margin-top:6px'>"
                f"<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;"
                f"color:#9E9E9E;font-weight:600;margin-bottom:2px'>N6 · IAC</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:0.84rem;"
                f"font-weight:500;color:" + (
                    "#3B6D11" if _n6_grade_card == "High" else
                    "#BA7517" if _n6_grade_card == "Moderate" else "#A32D2D"
                ) + f";line-height:1.2'>{_n6_grade_card}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:#707070;margin-top:1px'>+{N6_BOOST_BY_GRADE[_n6_grade_card]:.2f} boost</div>"
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
                    # Reconcile audit IDs — removes orphan record from
                    # session_state.n1_audit (Mark 8 N1 audit).
                    _sync_conviction_audit_ids()
                    _cr_feed_nodes()
                    st.rerun()

            # ── §5.1.1 N1 burden-of-proof audit (Mark 8) ─────────────────
            # Crown-reliance audit per Angelillo: is the Crown relying on
            # this prior as aggravating context? If yes, BARD per Gardiner.
            # Prior-evidence audit (Bird carve-out) is a Mark 9 build item;
            # placeholder rendered visible-but-inert per JP scoping.
            audit_rec = _audit_record_for_conviction(e, i)
            audit_id = e.get("audit_id")
            _icon, _label, _color = _audit_status_for_conviction(audit_id)

            with st.expander(
                f"{_icon}  Burden-of-proof audit — {_label}",
                expanded=False,
            ):
                st.markdown(
                    f"<div style='font-family:Fraunces,serif;font-style:italic;"
                    f"font-size:0.86rem;color:#5A5A5A;margin-bottom:12px;"
                    f"line-height:1.55'>"
                    f"Per <em>R. v. Angelillo</em> 2006 SCC 55, Crown reliance on "
                    f"a prior conviction as <em>aggravating</em> context at present "
                    f"sentencing must clear BARD. Mere existence of the conviction "
                    f"is res judicata and not subject to fresh audit (subject to "
                    f"<em>R. v. Bird</em> 2019 SCC 7 collateral-attack carve-outs — "
                    f"see prior-evidence audit below)."
                    f"</div>",
                    unsafe_allow_html=True,
                )

                _crown_relies_default = (
                    audit_rec.get("use") == "aggravating"
                )
                _crown_relies = st.checkbox(
                    "Crown is relying on this prior as **aggravating context** "
                    "at present sentencing",
                    value=_crown_relies_default,
                    key=f"cr_audit_relies_{i}",
                )
                if _crown_relies != _crown_relies_default:
                    _set_conviction_audit_use(audit_id, _crown_relies)
                    st.rerun()

                if _crown_relies:
                    st.markdown(
                        "<div style='background:#FFF7E8;border-left:3px solid "
                        "#BA7517;padding:10px 14px;margin:12px 0 14px 0;"
                        "border-radius:4px;font-size:0.86rem;color:#3A3A3A;"
                        "line-height:1.55'>"
                        "<strong style='color:#7A4F0E'>BARD audit required.</strong> "
                        "Per <em style='font-family:Fraunces,serif'>R. v. Gardiner</em>, "
                        "[1982] 2 SCR 368 + s. 724(3)(e) <em>Criminal Code</em>: "
                        "Crown-tendered aggravating facts must be proven beyond "
                        "a reasonable doubt. Record the basis on which BARD is "
                        "claimed to be met, or mark the attestation insufficient."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # Attestation basis dropdown shortcut
                    _basis_default = audit_rec.get("attestation_basis",
                                                   ATTESTATION_BASES[0])
                    if _basis_default not in ATTESTATION_BASES:
                        _basis_default = ATTESTATION_BASES[0]
                    _basis = st.selectbox(
                        "Basis on which BARD is claimed to be met",
                        ATTESTATION_BASES,
                        index=ATTESTATION_BASES.index(_basis_default),
                        key=f"cr_audit_basis_{i}",
                    )

                    # Attestation free text
                    _att_default = audit_rec.get("attestation", "")
                    _att = st.text_area(
                        "Attestation detail (free text — recorded verbatim "
                        "in the §RM.1 register)",
                        value=_att_default,
                        height=80,
                        key=f"cr_audit_att_{i}",
                        placeholder="e.g. 'Admitted by accused at para 14 of "
                                    "guilty plea allocution; transcript appended.'",
                    )

                    # Attestation status
                    _status_options = ["pending", "satisfied", "insufficient"]
                    _status_labels = {
                        "pending": "⚠ Pending — attestation incomplete",
                        "satisfied": "✓ Satisfied — BARD met on the record",
                        "insufficient": "✗ Insufficient — burden not met "
                                        "for this aggravating use",
                    }
                    _current_status = audit_rec.get("attestation_status",
                                                    "pending")
                    if _current_status not in _status_options:
                        _current_status = "pending"
                    _status = st.radio(
                        "Audit status",
                        _status_options,
                        index=_status_options.index(_current_status),
                        format_func=lambda s: _status_labels[s],
                        key=f"cr_audit_status_{i}",
                    )

                    # Persist any changes back into the audit record
                    _changed = (
                        _basis != _basis_default or
                        _att != _att_default or
                        _status != _current_status
                    )
                    if _changed:
                        audit_rec["attestation_basis"] = _basis
                        audit_rec["attestation"] = _att
                        audit_rec["attestation_status"] = _status
                        st.session_state.n1_audit[audit_id] = audit_rec
                        # No rerun here — let normal Streamlit rerun cycle
                        # propagate. Avoids duplicate-render thrash on
                        # rapid keystrokes in the text_area.

                # ── §5.1.14 / §5.1.15 / §5.1.18 per-conviction attestations ──
                # Mark 8 Phase 2: Layer II distortion attestations relocated
                # from Profile to per-conviction. Aggregate signals derived
                # via _any_conviction_attests("...") in the signal-computation
                # helpers — any-prior aggregation per JP M8/P2 lock-in.
                st.markdown(
                    "<div style='border-top:1px dashed #D8D5CE;margin:14px 0 "
                    "10px 0'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.74rem;color:#5A5A5A;margin-bottom:8px'>"
                    "LAYER II DISTORTION ATTESTATIONS (per-conviction)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-family:Fraunces,serif;font-style:italic;"
                    "font-size:0.82rem;color:#5A5A5A;margin-bottom:10px;"
                    "line-height:1.5'>"
                    "Counsel attestations specific to <em>this</em> "
                    "conviction. Aggregate signals (any-prior) feed N14 "
                    "(temporal distortion), N15 (tariff distortion), and N18 "
                    "(SCE Profile audit)."
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ── §5.1.14 (temporal distortion) ─────────────────────────
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.70rem;color:#7A7A7A;letter-spacing:0.04em;"
                    "text-transform:uppercase;margin:8px 0 2px 0'>"
                    "§5.1.14 — N14 temporal distortion"
                    "</div>",
                    unsafe_allow_html=True,
                )
                _n14a_default = e.get("n14a_attestation", False)
                _n14a_new = st.checkbox(
                    "Imposed under severe sentencing era "
                    "(Bill C-10 / pre-Sharma regime)",
                    value=_n14a_default,
                    key=f"cr_n14a_{i}",
                    help=(
                        "§5.1.14 §1 identifies sentencing eras characterised "
                        "by mandatory minimums or 'tough-on-crime' policies "
                        "as carriers of institutional severity. Auto-detects "
                        "2008–2015 (Bill C-10 era); use this override where "
                        "the conviction sits outside that window but was "
                        "imposed under analogously severe regime conditions."
                    ),
                )
                _n14b_default = e.get("n14b_attestation", False)
                _n14b_new = st.checkbox(
                    "Arose under mandatory-minimum framework",
                    value=_n14b_default,
                    key=f"cr_n14b_{i}",
                    help=(
                        "§5.1.14 §5: MM frameworks are a parent input to "
                        "temporal distortion. Use where MM-eligible offences "
                        "are present but the auto-detection (offence-text + "
                        "year heuristic) hasn't flagged them."
                    ),
                )
                _n14c_default = e.get("n14c_sce_applied", False)
                _n14c_new = st.checkbox(
                    "SCE / Tetrad **was** applied at this conviction's "
                    "original sentencing",
                    value=_n14c_default,
                    key=f"cr_n14c_{i}",
                    help=(
                        "§5.1.14 §5: when SCE was substantively applied, "
                        "temporal distortion is reduced. Inverse semantics: "
                        "checked = SCE present = N14c → Low. Also drives N15d "
                        "(jurisprudential compliance for tariff distortion)."
                    ),
                )

                # ── §5.1.15 (tariff distortion) ───────────────────────────
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.70rem;color:#7A7A7A;letter-spacing:0.04em;"
                    "text-transform:uppercase;margin:10px 0 2px 0'>"
                    "§5.1.15 — N15 tariff distortion"
                    "</div>",
                    unsafe_allow_html=True,
                )
                _n15a_default = e.get("n15a_attestation", False)
                _n15a_new = st.checkbox(
                    "Imposed in high-tariff jurisdiction "
                    "(Doob/Cesaroni/Roach lineage)",
                    value=_n15a_default,
                    key=f"cr_n15a_{i}",
                    help=(
                        "§5.1.15 §2(d): empirical research documents material "
                        "interprovincial tariff disparities. Auto-detection "
                        "covers ON/MB/SK/AB; override where this conviction's "
                        "jurisdiction warrants different classification."
                    ),
                )
                _n15b_default = e.get("n15b_attestation", False)
                _n15b_new = st.checkbox(
                    "Tariff-sensitive offence type",
                    value=_n15b_default,
                    key=f"cr_n15b_{i}",
                    help=(
                        "§5.1.15 §7: property and drug offences are the "
                        "classic tariff-sensitive categories. Override where "
                        "specialised offence terminology defeats keyword "
                        "auto-detection."
                    ),
                )
                _n15c_default = e.get("n15c_attestation", False)
                _n15c_new = st.checkbox(
                    "Sentence length exceeds offence-conditional tariff "
                    "threshold",
                    value=_n15c_default,
                    key=f"cr_n15c_{i}",
                    help=(
                        "§5.1.15 §7 thresholds: tariff-sensitive offences "
                        "above 1 year; conduct-driven (violent/sexual) above "
                        "3 years. Override e.g. for an 18-month provincial "
                        "property-offence sentence exceeding the 1-year cap."
                    ),
                )

                # ── §5.1.18 (SCE Profile audit) ───────────────────────────
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.70rem;color:#7A7A7A;letter-spacing:0.04em;"
                    "text-transform:uppercase;margin:10px 0 2px 0'>"
                    "§5.1.18 — N18 SCE profile audit"
                    "</div>",
                    unsafe_allow_html=True,
                )
                _n18b_default = e.get("n18b_attestation", False)
                _n18b_new = st.checkbox(
                    "SCE absent from reasons in this conviction",
                    value=_n18b_default,
                    key=f"cr_n18b_{i}",
                    help=(
                        "§5.1.18 §5 row 3: SCE Presence in Reasons. Aggregate "
                        "signal fires if any prior has SCE absent. Use where "
                        "this conviction's reasons lack SCE engagement."
                    ),
                )
                _n18c_default = e.get("n18c_attestation", False)
                _n18c_new = st.checkbox(
                    "SCE substance nominal-only or absent in this conviction "
                    "(Morris Heuristic Audit)",
                    value=_n18c_default,
                    key=f"cr_n18c_{i}",
                    help=(
                        "§5.1.18: even where Gladue is cited, substantive "
                        "sentencing impact is often absent or minimal. Use "
                        "where this conviction's SCE engagement was nominal "
                        "(referenced but not substantively applied)."
                    ),
                )

                # Persist any toggles back to the conviction entry. Avoid
                # rerun thrash by only writing when something changed.
                _attestation_changed = (
                    _n14a_new != _n14a_default or
                    _n14b_new != _n14b_default or
                    _n14c_new != _n14c_default or
                    _n15a_new != _n15a_default or
                    _n15b_new != _n15b_default or
                    _n15c_new != _n15c_default or
                    _n18b_new != _n18b_default or
                    _n18c_new != _n18c_default
                )
                if _attestation_changed:
                    e["n14a_attestation"] = _n14a_new
                    e["n14b_attestation"] = _n14b_new
                    e["n14c_sce_applied"] = _n14c_new
                    e["n15a_attestation"] = _n15a_new
                    e["n15b_attestation"] = _n15b_new
                    e["n15c_attestation"] = _n15c_new
                    e["n18b_attestation"] = _n18b_new
                    e["n18c_attestation"] = _n18c_new
                    # entry is a reference into st.session_state.criminal_record;
                    # the mutation is already persisted. Trigger rerun so the
                    # signal helpers re-read on next pass.
                    st.rerun()

                # ── PRIOR-EVIDENCE AUDIT placeholder (Mark 9 deferred) ────
                st.markdown(
                    "<div style='border-top:1px dashed #D8D5CE;margin:14px 0 "
                    "10px 0'></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.74rem;color:#9E9E9E;margin-bottom:6px'>"
                    "PRIOR-EVIDENCE AUDIT (Layer 2 — Mark 9 deferred)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='background:#F5F4F0;border:1px solid #E5E2DA;"
                    "border-radius:4px;padding:9px 12px;font-size:0.83rem;"
                    "color:#5A5A5A;line-height:1.55'>"
                    "<strong>Status:</strong> "
                    "<span style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.78rem;color:#7A7A7A'>"
                    "[awaiting implementation]</span><br>"
                    "Defence challenge to evidentiary basis of the prior "
                    "conviction itself, per <em style='font-family:Fraunces,serif'>"
                    "R. v. Bird</em> 2019 SCC 7 collateral-attack carve-out. "
                    "Full implementation pending doctrinal scoping and "
                    "integration with Layer II distortion-detection nodes "
                    "(N15 IAC, N17 over-policing, N5 invalid risk tools). "
                    "Will be wired in next development cycle."
                    "</div>",
                    unsafe_allow_html=True,
                )

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

        # ── §5.1.17 audit-signal methodology (N17c / N17d auto-detection) ──
        with st.expander("📚 §5.1.17 audit signal — how N17c and N17d are computed"):
            # Compute live values for the current record
            _record = st.session_state.get("criminal_record", []) or []
            _n_total = len(_record)
            # Recompute density values for display
            _n17c_count = 0
            _n17d_count = 0
            for _e in _record:
                _off = (_e.get("offence", "") or "").lower()
                _ser = (_e.get("seriousness_label", "") or "").lower()
                _is_low_tier = ("minor" in _ser or "moderate" in _ser)
                _has_violence = any(p in _off for p in _N17C_VIOLENCE_PATTERNS)
                if _is_low_tier and not _has_violence:
                    _n17c_count += 1
                if any(p in _off for p in _N17D_SURVEILLANCE_PATTERNS):
                    _n17d_count += 1
            _n17c_density = (_n17c_count / _n_total) if _n_total else 0.0
            _n17d_density = (_n17d_count / _n_total) if _n_total else 0.0
            _n17c_state = "High" if _n17c_density > 0.50 else "Low"
            _n17d_state = "High" if _n17d_density >= 0.30 else "Low"

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.88rem;color:#3A3A3A;line-height:1.55;margin-bottom:14px'>"
                "Per Chapter 5 §5.1.17 §2, criminal records are institutionally-"
                "produced artifacts. Where exposure to policing is uneven, record "
                "density is unreliable as a proxy for criminality. Two of N17's "
                "four parents (N17c, N17d) are auto-computed from the offence "
                "text of each conviction below:"
                "</div>",
                unsafe_allow_html=True,
            )

            # N17c live status card
            _n17c_color = "#A32D2D" if _n17c_state == "High" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n17c_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N17c — Non-violent charge density</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n17c_color};font-weight:600'>{_n17c_count}/{_n_total} = "
                f"{_n17c_density*100:.0f}% · {_n17c_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Counts entries where seriousness tier is Minor/Moderate AND "
                f"offence text contains no violence keywords "
                f"(<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#888'>"
                f"assault, robbery, manslaughter, murder, sexual, firearm, aggravated, weapon</code>). "
                f"State is High when density &gt; 50%."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N17d live status card
            _n17d_color = "#A32D2D" if _n17d_state == "High" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n17d_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N17d — Surveillance-triggered entries</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n17d_color};font-weight:600'>{_n17d_count}/{_n_total} = "
                f"{_n17d_density*100:.0f}% · {_n17d_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Pattern-matches §5.1.17 §2 surveillance signatures in offence text "
                f"(<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#888'>"
                f"breach, fail to comply, fail to appear, administration of justice, possession</code>). "
                f"State is High when density ≥ 30%."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.80rem;color:#707070;margin-top:14px;line-height:1.55'>"
                "<strong style='font-weight:500;color:#3A3A3A;font-style:normal'>Override path.</strong> "
                "Pattern-matching against offence text is conservative — false positives "
                "(e.g. \"possession of stolen property\" matching \"possession\") are possible. "
                "Counsel can override at the per-conviction level via the "
                "<strong style='font-weight:500'>Over-policing slider</strong> in the conviction "
                "form above, which feeds the existing N17 over-policing pathway independently "
                "of N17c/N17d. The two pathways are doctrinally complementary: the slider "
                "captures per-conviction nuance; the audit signal captures aggregate "
                "composition of the record per §5.1.17 §5."
                "</div>",
                unsafe_allow_html=True,
            )

        # ── §5.1.14 audit-signal methodology (N14a/N14b/N14c year-based) ────
        with st.expander("📚 §5.1.14 audit signal — how N14a, N14b, N14c are computed"):
            # Compute live values for the current record
            _record = st.session_state.get("criminal_record", []) or []
            _n_total = len(_record)
            
            # Replicate _n14a_signal logic for display
            _n14a_count = sum(1 for e in _record
                              if 2008 <= int(e.get("year", 0) or 0) <= 2015)
            _n14a_attest = _any_conviction_attests("n14a_attestation")
            _n14a_state = "High" if (_n14a_count > 0 or _n14a_attest) else "Low"
            
            # Replicate _n14b_signal logic
            _n14b_count = 0
            for _e in _record:
                _off = (_e.get("offence", "") or "").lower()
                try:
                    _yr = int(_e.get("year", 0))
                except (TypeError, ValueError):
                    continue
                if 2008 <= _yr <= 2015 and any(p in _off for p in _N14B_MM_OFFENCE_PATTERNS):
                    _n14b_count += 1
            _n14b_attest = _any_conviction_attests("n14b_attestation")
            _n14b_state = "High" if (_n14b_count > 0 or _n14b_attest) else "Low"
            
            # Replicate _n14c_signal logic
            _n14c_pre_ipeelee = sum(1 for e in _record
                                    if 0 < int(e.get("year", 0) or 0) < 2012)
            _n14c_attest = _any_conviction_attests("n14c_sce_applied")
            _n14c_state = ("Low (SCE present)" if _n14c_attest 
                           else ("High (SCE absent)" if _n14c_pre_ipeelee > 0 else "Low"))

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.88rem;color:#3A3A3A;line-height:1.55;margin-bottom:14px'>"
                "Per Chapter 5 §5.1.14 §1, time becomes a source of inferential error "
                "when convictions imposed under outdated regimes are imported wholesale "
                "into contemporary risk assessment. Three of N14's four parents (N14a, "
                "N14b, N14c) are auto-derived from conviction years and offence text. "
                "N14d (judicial competence) is computed by the Bayesian network from "
                "N10 (Judicial Misapplication) per §5.1.14 §Position."
                "</div>",
                unsafe_allow_html=True,
            )

            # N14a status card
            _n14a_color = "#A32D2D" if _n14a_state == "High" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n14a_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N14a — Sentencing era severity</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n14a_color};font-weight:600'>{_n14a_count} conviction(s) in 2008–2015"
                f"{' · attested' if _n14a_attest else ''} · {_n14a_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Flags High when record contains conviction(s) from 2008–2015 (Bill C-10 "
                f"era of mandatory-minimum expansion and pre-<em>Sharma</em> regime). "
                f"Counsel attestation overrides for analogous severity outside this window."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N14b status card
            _n14b_color = "#A32D2D" if _n14b_state == "High" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n14b_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N14b — Historical mandatory minimum</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n14b_color};font-weight:600'>{_n14b_count} MM-eligible conviction(s)"
                f"{' · attested' if _n14b_attest else ''} · {_n14b_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Pattern-matches MM-eligible offences "
                f"(<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;color:#888'>"
                f"trafficking, firearm, sexual interference, child pornography</code>) "
                f"against 2008–2015 conviction years. Many of these MMs were struck down "
                f"post-<em>Nur</em> (2015) / <em>Lloyd</em> (2016), but convictions imposed "
                f"during their force carry doctrinal residue per §5.1.14 §2."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N14c status card
            _n14c_is_high = _n14c_state.startswith("High")
            _n14c_color = "#A32D2D" if _n14c_is_high else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n14c_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N14c — SCE absent at sentencing</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n14c_color};font-weight:600'>{_n14c_pre_ipeelee} pre-2012 conviction(s)"
                f"{' · attested SCE applied' if _n14c_attest else ''} · {_n14c_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Flags SCE absent when record contains pre-<em>Ipeelee</em> (pre-2012) "
                f"conviction(s) — SCE was formally available post-<em>Gladue</em> (1999) but "
                f"often only nominally applied until <em>Ipeelee</em> made substantive "
                f"integration mandatory. Attestation override sets state to Low when SCE "
                f"WAS substantively applied at original sentencing (architecture treats this "
                f"as no temporal-distortion contribution from N14c)."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.80rem;color:#707070;margin-top:14px;line-height:1.55'>"
                "<strong style='font-weight:500;color:#3A3A3A;font-style:normal'>Note on N14d.</strong> "
                "N14d (judicial competence absent) has no app-derived signal. "
                "Per §5.1.14 §Position + the model topology, N14d is a Bayesian "
                "network node downstream of N10 (Judicial Misapplication of SCE). "
                "Its posterior is computed by Variable Elimination from N10's "
                "evidence — when N10 indicates misapplication, N14d rises proportionally."
                "</div>",
                unsafe_allow_html=True,
            )

        # ── §5.1.15 audit-signal methodology (N15a/b/c — N15d shares n14c) ──
        with st.expander("📚 §5.1.15 audit signal — how N15a, N15b, N15c are computed"):
            _record = st.session_state.get("criminal_record", []) or []
            _case_jur = st.session_state.get("case_jur", "") or ""

            # Replicate _n15a logic for display
            _case_lower = _case_jur.lower()
            _n15a_match = any(j in _case_lower for j in _N15_HIGH_TARIFF_JURISDICTIONS)
            _n15a_attest = _any_conviction_attests("n15a_attestation")
            _n15a_state = "High-tariff" if (_n15a_match or _n15a_attest) else "Low-tariff"

            # Replicate _n15b logic
            _n15b_count = 0
            for _e in _record:
                _off = (_e.get("offence", "") or "").lower()
                _ts = any(p in _off for p in _N15_TARIFF_SENSITIVE_PATTERNS)
                _cd = any(p in _off for p in _N15_CONDUCT_DRIVEN_PATTERNS)
                if _ts and not _cd:
                    _n15b_count += 1
            _n15b_attest = _any_conviction_attests("n15b_attestation")
            _n15b_state = "Tariff-sensitive" if (_n15b_count > 0 or _n15b_attest) else "Conduct-driven / none"

            # Replicate _n15c logic
            _n15c_fed_count = sum(1 for _e in _record
                                   if "Federal custody" in (_e.get("sentence_type", "") or ""))
            _n15c_attest = _any_conviction_attests("n15c_attestation")
            _n15c_state = "Long sentence" if (_n15c_fed_count > 0 or _n15c_attest) else "Short / none"

            # N15d shares n14c
            _n15d_attest = _any_conviction_attests("n14c_sce_applied")
            _n15d_state = "Doctrine present" if _n15d_attest else "Doctrine absent"

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.88rem;color:#3A3A3A;line-height:1.55;margin-bottom:14px'>"
                "Per Chapter 5 §5.1.15 §1, interjurisdictional tariff distortion arises "
                "when sentence length and conviction severity are treated as uniform "
                "indicators of culpability across provinces, despite documented regional "
                "variation in sentencing norms (Doob, Cesaroni, Roach). Three of N15's "
                "four parents (N15a/b/c) are auto-derived from case_jur, offence text, "
                "and sentence_type. N15d (jurisprudential compliance) shares the §5.1.14 "
                "SCE-applied attestation — the same doctrinal-application question covers "
                "both temporal and tariff dimensions. N15 itself is computed by the "
                "Bayesian network from these four sub-nodes plus N14 (per §5.1.15 §Position)."
                "</div>",
                unsafe_allow_html=True,
            )

            # N15a status card
            _n15a_color = "#A32D2D" if _n15a_state == "High-tariff" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n15a_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N15a — Tariff jurisdiction disparity</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n15a_color};font-weight:600'>"
                f"{_case_jur if _case_jur else '(no jurisdiction set)'}"
                f"{' · attested' if _n15a_attest else ''} · {_n15a_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Tier-classifies <code style='font-family:JetBrains Mono,monospace;"
                f"font-size:0.74rem;color:#888'>case_jur</code> against high-tariff "
                f"jurisdiction set per Doob/Cesaroni/Roach lineage (Ontario, Manitoba, "
                f"Saskatchewan, Alberta). Counsel attestation overrides for case-specific "
                f"tariff classification."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N15b status card
            _n15b_color = "#A32D2D" if _n15b_state == "Tariff-sensitive" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n15b_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N15b — Tariff-sensitive offence type</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n15b_color};font-weight:600'>{_n15b_count} tariff-sensitive offence(s)"
                f"{' · attested' if _n15b_attest else ''} · {_n15b_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Pattern-matches offence text against tariff-sensitive categories "
                f"(<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:#888'>theft, fraud, trafficking, breach</code>). Conduct-driven "
                f"offences (assault, sexual offences) treated as tariff-resistant per "
                f"§5.1.15 §7 — sentence length more reflects conduct severity than "
                f"regional variation."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N15c status card
            _n15c_color = "#A32D2D" if _n15c_state == "Long sentence" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n15c_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N15c — Tariff-sensitive sentence length</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n15c_color};font-weight:600'>{_n15c_fed_count} federal-custody conviction(s)"
                f"{' · attested' if _n15c_attest else ''} · {_n15c_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"§5.1.15 §7 thresholds: tariff-sensitive offences become tariff-prone "
                f"above 1 year; conduct-driven offences above 3 years. Heuristic "
                f"auto-detects Federal custody (2+ years) which always exceeds the "
                f"1-year threshold. Use attestation for Provincial custody cases that "
                f"exceeded the offence-conditional threshold (e.g., 18-month sentence "
                f"for property offence)."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.80rem;color:#707070;margin-top:14px;line-height:1.55'>"
                f"<strong style='font-weight:500;color:#3A3A3A;font-style:normal'>"
                f"Note on N15d.</strong> "
                f"N15d (jurisprudential compliance absent) shares the §5.1.14 "
                f"SCE-applied attestation — the same sentencing event either applied "
                f"Tetrad/SCE jurisprudence or did not, and that single doctrinal-"
                f"application question covers both temporal-distortion (N14c) and "
                f"tariff-distortion (N15d) dimensions. Per Q3 binary collapse: "
                f"<em>Gladue+Ewert</em> (binding) and <em>Morris+Ellis</em> "
                f"(persuasive) both map to state 0 — any jurisprudence applied means "
                f"doctrine present. Current state: <strong style='color:{("#3B6D11" if _n15d_attest else "#A32D2D")}'>"
                f"{_n15d_state}</strong>."
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── §5.1.18 audit-signal methodology (N18a/b/c — N18d BN-derived) ──
        with st.expander("📚 §5.1.18 audit signal — how N18a, N18b, N18c are computed"):
            _record = st.session_state.get("criminal_record", []) or []
            _case_jur = st.session_state.get("case_jur", "") or ""

            # Replicate _n18a logic for display
            _case_lower = _case_jur.lower()
            _n18a_match = any(j in _case_lower for j in _N18_SCE_INTEGRATED_JURISDICTIONS)
            _n18a_attest = st.session_state.get("n18a_counsel_attestation", False)
            _n18a_state = "No Morris/Ellis" if (not _n18a_match or _n18a_attest) else "Morris/Ellis/SCC"

            # Replicate _n18b/c logic — count tag distribution
            _n18b_attest = _any_conviction_attests("n18b_attestation")
            _n18c_attest = _any_conviction_attests("n18c_attestation")
            _tag_counts = {"Full":0, "Partial":0, "Nominal":0, "Absent":0}
            for _e in _record:
                _t = _e.get("sce_integration_tag", _N18_TAG_DEFAULT)
                if _t in _tag_counts:
                    _tag_counts[_t] += 1
            _n18b_state = "Absent" if (_tag_counts["Absent"] > 0 or _n18b_attest) else "Present"
            _n18c_state = ("Nominal/Absent" if (_tag_counts["Nominal"] + _tag_counts["Absent"] > 0
                                                  or _n18c_attest)
                           else "Substantive")

            st.markdown(
                "<div style='font-family:Fraunces,serif;font-style:italic;"
                "font-size:0.88rem;color:#3A3A3A;line-height:1.55;margin-bottom:14px'>"
                "Per Chapter 5 §5.1.18 §1, N18 audits whether prior convictions "
                "substantively integrated SCE / Tetrad jurisprudence — "
                "<em>not</em> whether the original sentence was correct. It is "
                "framed as a metadata tagging layer, evaluating how a record "
                "should be interpreted today rather than whether it was properly "
                "decided then. Three of N18's six parents (N18a/b/c) are auto-"
                "derived from case_jur and per-conviction SCE-integration tags. "
                "N18d (Doctrinal Tagging compliance) is BN-derived from N10 "
                "(Misapplication). N18 itself is computed by the Bayesian network "
                "from these four sub-nodes plus N12 and N14 as structural "
                "amplifiers per §5.1.18 §Position."
                "</div>",
                unsafe_allow_html=True,
            )

            # N18a status card
            _n18a_color = "#A32D2D" if _n18a_state == "No Morris/Ellis" else "#3B6D11"
            st.markdown(
                f"<div style='border-left:3px solid {_n18a_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N18a — Jurisdiction SCE-integration sensitivity</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n18a_color};font-weight:600'>"
                f"{_case_jur if _case_jur else '(no jurisdiction set)'}"
                f"{' · attested' if _n18a_attest else ''} · {_n18a_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Tier-classifies <code style='font-family:JetBrains Mono,monospace;"
                f"font-size:0.74rem;color:#888'>case_jur</code> against jurisdictions "
                f"with strong provincial appellate SCE-integration scrutiny: ON post-"
                f"Morris (2021 ONCA 680), BC post-Ellis (2022 BCCA 278), and SCC. "
                f"All other jurisdictions default to inflation risk."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N18b status card with tag distribution
            _n18b_color = "#A32D2D" if _n18b_state == "Absent" else "#3B6D11"
            _tag_dist_str = (
                f"Full: {_tag_counts['Full']} · Partial: {_tag_counts['Partial']} · "
                f"Nominal: {_tag_counts['Nominal']} · Absent: {_tag_counts['Absent']}"
            )
            st.markdown(
                f"<div style='border-left:3px solid {_n18b_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N18b — SCE Presence in reasons</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n18b_color};font-weight:600'>"
                f"{_tag_dist_str}{' · attested' if _n18b_attest else ''} · {_n18b_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"Aggregates over per-conviction SCE-integration tags. State 1 "
                f"(SCE absent) when at least one conviction tagged "
                f"<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:#888'>Absent</code>. Tags Full/Partial/Nominal all count "
                f"as SCE present in reasons."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # N18c status card — emphasises Morris Heuristic Audit
            _n18c_color = "#A32D2D" if _n18c_state == "Nominal/Absent" else "#3B6D11"
            _morris_count = _tag_counts["Nominal"] + _tag_counts["Absent"]
            st.markdown(
                f"<div style='border-left:3px solid {_n18c_color};padding:.6rem .9rem;"
                f"margin-bottom:.7rem;background:#FBFAF7;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-weight:500;"
                f"color:#1A1A1A;font-size:0.94rem'>N18c — SCE Substance (Morris Heuristic Audit)</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:0.80rem;"
                f"color:{_n18c_color};font-weight:600'>"
                f"{_morris_count} non-substantive conviction(s)"
                f"{' · attested' if _n18c_attest else ''} · {_n18c_state}</div>"
                f"</div>"
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.78rem;color:#5A5A5A;margin-top:4px'>"
                f"State 1 when any conviction tagged "
                f"<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:#888'>Nominal</code> or "
                f"<code style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
                f"color:#888'>Absent</code>. Captures the Morris Heuristic Audit "
                f"finding that nominal Gladue mention without substantive "
                f"integration is a distinct failure mode from outright absence."
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='font-family:Fraunces,serif;font-style:italic;"
                f"font-size:0.80rem;color:#707070;margin-top:14px;line-height:1.55'>"
                f"<strong style='font-weight:500;color:#3A3A3A;font-style:normal'>"
                f"Note on N18d.</strong> "
                f"N18d (Doctrinal Tagging compliance) has no app-derived signal. "
                f"Per §5.1.18 §5 + Q5 (α), N18d is a Bayesian network node "
                f"downstream of N10 (Judicial Misapplication of SCE). Its "
                f"posterior is computed by Variable Elimination from N10's "
                f"evidence — when N10 indicates misapplication, N18d rises "
                f"proportionally, reflecting the increased likelihood of "
                f"incomplete or error-flagged doctrinal tagging at original "
                f"sentencing."
                f"</div>",
                unsafe_allow_html=True,
            )

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

    # ── §RM.1 register: N1 burden-of-proof audit ledger (Mark 8 Phase 2) ────
    # Per JP architecture lock-in (Q1=A, Q2=A, Q3=A, Q4=B + M8/P2 Pattern A):
    # every audited input across all integrated tabs surfaces here with
    # provenance, applicable burden, attestation text, audit status, and
    # citations. The register makes the trust-with-transparency model
    # visible: PARVIS records the user's attestation as an assumption-of-
    # record and displays it for adversarial review.

    # Strict-mode review panel — surfaces pending classifications and
    # attestations across all tabs. No-op when strict_mode is off.
    _strict_mode_review_panel(tab_filter=None)

    n1_audit_state = st.session_state.get("n1_audit", {})
    n1_target = compute_n1_prior_from_audit(n1_audit_state)
    n1_summary_register = _n1_audit_summary(n1_audit_state)
    st.markdown(
        f"<div class='at' style='margin-top:.8rem'>"
        f"{'─'*60}\n  §RM.1 — BURDEN-OF-PROOF AUDIT REGISTER\n{'─'*60}\n"
        f"  Doctrinal basis:\n"
        f"    {N1_CITATIONS['gardiner']}\n"
        f"    {N1_CITATIONS['s724']}\n"
        f"    {N1_CITATIONS['ferguson']}\n"
        f"    {N1_CITATIONS['angelillo']}\n"
        f"    {N1_CITATIONS['lacasse']}\n\n"
        f"  Doctrinal posture:                "
        f"<strong style='color:{n1_summary_register['color_accent']}'>"
        f"{n1_summary_register['label'].upper()}</strong>\n"
        f"  Audit-pressure depth:             "
        f"{Pa.get(1, 0.83)*100:.1f}%   "
        f"<span style='color:#9E9E9E;font-style:italic'>"
        f"(operationalisation, not a probability)</span>\n"
        f"  Audit-derived target value:       {n1_target*100:.1f}%   "
        f"<span style='color:#9E9E9E;font-style:italic'>(internal CPT input)</span>\n"
        f"  Audited inputs:                   "
        f"{n1_summary_register['audited_count']} total — "
        f"{n1_summary_register['satisfied_count']} satisfied, "
        f"{n1_summary_register['pending_count']} pending, "
        f"{n1_summary_register['insufficient_count']} insufficient\n"
        f"  Strict mode:                      "
        f"{'ON' if st.session_state.get('strict_mode', False) else 'off'}\n"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not n1_audit_state:
        st.markdown(
            "<div class='at'>  No inputs audited yet.\n"
            "  Per Mark 8 Phase 2 build status:\n"
            "    • Criminal Record tab     — full audit live\n"
            "    • Gladue tab              — full audit live\n"
            "    • SCE tab                 — full audit live\n"
            "    • Risk &amp; Distortions  — full audit live\n"
            "    • Intake tab              — [awaiting Mark 9 integration]\n"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Group audit records by tab for the register
        by_tab = {}
        for aid, rec_audit in n1_audit_state.items():
            tab_key = rec_audit.get("tab", "unknown")
            by_tab.setdefault(tab_key, []).append((aid, rec_audit))

        register_lines = []
        # Status legend
        _legend = {
            "satisfied":    "[✓ SATISFIED]",
            "insufficient": "[✗ INSUFFICIENT]",
            "pending":      "[⚠ PENDING]",
        }

        for tab_key in sorted(by_tab.keys()):
            tab_label = {
                "criminal_record":   "CRIMINAL RECORD",
                "intake":            "INTAKE",
                "gladue":            "GLADUE",
                "sce":               "SCE / MORRIS-ELLIS",
                "risk_substantive":  "RISK & DISTORTIONS — SUBSTANTIVE (N2/N3/N4)",
                "risk_distortion":   "RISK & DISTORTIONS — DISTORTION NODES",
                "risk_mitigation":   "RISK & DISTORTIONS — MITIGATION (N8/N9)",
                "risk_other":        "RISK & DISTORTIONS — OTHER (N11 GAMING, ETC.)",
            }.get(tab_key, tab_key.upper())
            register_lines.append(f"\n  ── {tab_label} ──")

            for aid, rec_audit in by_tab[tab_key]:
                use = rec_audit.get("use", "contextual")
                provenance = rec_audit.get("provenance", "—")
                burden = rec_audit.get("applicable_burden", "none")
                status = rec_audit.get("attestation_status", "pending")
                label = rec_audit.get("label", aid)
                attestation = rec_audit.get("attestation", "")
                basis = rec_audit.get("attestation_basis", "")
                judicial_type = rec_audit.get("judicial_finding_type")

                if use in ("contextual", "agreed_fact"):
                    # No audit triggered — record but don't fail
                    register_lines.append(
                        f"\n  · {label}\n"
                        f"      Provenance: {provenance} · "
                        f"Use: {use} · No audit required."
                    )
                else:
                    status_marker = _legend.get(status, "[?]")
                    register_lines.append(
                        f"\n  · {label}  {status_marker}\n"
                        f"      Provenance: {provenance} · "
                        f"Use: {use} · Burden: {burden}"
                    )
                    if judicial_type:
                        register_lines.append(
                            f"      Ferguson sub-flag: {judicial_type}"
                        )
                    if basis and basis != ATTESTATION_BASES[0]:
                        register_lines.append(
                            f"      Basis: {basis}"
                        )
                    if attestation:
                        # Wrap long attestations
                        att_text = attestation.replace("\n", " ").strip()
                        if len(att_text) > 140:
                            att_text = att_text[:137] + "..."
                        register_lines.append(
                            f"      Attestation: \"{att_text}\""
                        )
                    elif status == "pending":
                        register_lines.append(
                            "      [No attestation recorded — awaiting "
                            "user input]"
                        )

                # Prior-evidence audit placeholder (Mark 9 deferred)
                pea = rec_audit.get("prior_evidence_audit_status")
                if pea == "not_yet_conducted":
                    register_lines.append(
                        "      Prior-evidence audit (Bird carve-out): "
                        "[awaiting Mark 9 implementation]"
                    )

        st.markdown(
            f"<div class='at'>{''.join(register_lines)}\n{'─'*60}</div>",
            unsafe_allow_html=True,
        )

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
