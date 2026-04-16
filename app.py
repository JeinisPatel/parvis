"""
PARVIS — Streamlit Application
Probabilistic and Analytical Reasoning Virtual Intelligence System
J.S. Patel | University of London (QMUL & LSE) | Ethical AI Initiative

Full Streamlit UI with:
- pgmpy Variable Elimination Bayesian inference engine
- Quantum Bayesianism (QBism) diagnostic layer (Appendix Q)
- Case Profile, Gladue Factors, Morris/Ellis SCE inputs
- Interactive DAG visualisation
- Audit report with export
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import base64
from datetime import datetime

from model import build_model, get_inference_engine, query_do_risk, NODE_META, EDGES_VE as EDGES
from quantum_diagnostics import diagnose, format_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PARVIS — Bayesian Sentencing Network",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "PARVIS v1.0 — Research use only. Not for live proceedings."}
)

# ── Load Ethical AI logo ──────────────────────────────────────────────────────
@st.cache_data
def load_logo():
    try:
        with open("ethical_ai_logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 0.5rem 0 1rem 0; border-bottom: 1px solid rgba(0,0,0,0.1);
    margin-bottom: 1rem;
  }
  .parvis-title { font-size: 2rem; font-weight: 700; letter-spacing: 4px; margin: 0; }
  .parvis-sub { font-size: 0.75rem; color: #888; margin-top: 2px; }
  .do-card {
    background: linear-gradient(135deg, #f8f9fa, #fff);
    border: 1px solid #dee2e6; border-radius: 12px;
    padding: 1.2rem; text-align: center;
  }
  .do-label { font-size: 0.75rem; color: #888; margin-bottom: 0.3rem; }
  .do-pct { font-size: 2.5rem; font-weight: 700; font-family: monospace; }
  .do-band { font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem; }
  .type-badge {
    display: inline-block; font-size: 0.65rem; padding: 1px 8px;
    border-radius: 20px; font-weight: 600; margin-left: 4px;
  }
  .qbism-flag-high { background: #FCEBEB; color: #A32D2D; border-left: 3px solid #A32D2D;
    padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.3rem 0; }
  .qbism-flag-moderate { background: #FAEEDA; color: #BA7517; border-left: 3px solid #BA7517;
    padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.3rem 0; }
  .qbism-flag-none { background: #EAF3DE; color: #3B6D11; border-left: 3px solid #3B6D11;
    padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.3rem 0; }
  .node-info { background: #f8f9fa; border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem; }
  hr { border: none; border-top: 1px solid rgba(0,0,0,0.1); margin: 1rem 0; }
  .section-header { font-size: 0.75rem; font-weight: 700; color: #888;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Node colours ──────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "constraint": "#BA7517",
    "risk":       "#A32D2D",
    "distortion": "#185FA5",
    "mitigation": "#3B6D11",
    "dual":       "#534AB7",
    "special":    "#0F6E56",
    "output":     "#993C1D",
}
TYPE_LABELS = {
    "constraint": "Evidentiary constraint",
    "risk":       "Risk factor",
    "distortion": "Systemic distortion",
    "mitigation": "Mitigating factor",
    "dual":       "Dual factor",
    "special":    "Causal detector",
    "output":     "Structural output",
}

def risk_band(p):
    if p < 0.20: return "Very low",  "#3B6D11"
    if p < 0.40: return "Low",       "#3B6D11"
    if p < 0.55: return "Moderate",  "#BA7517"
    if p < 0.70: return "Elevated",  "#BA7517"
    if p < 0.85: return "High",      "#A32D2D"
    return "Very high", "#A32D2D"

# ── Session state initialisation ──────────────────────────────────────────────
def init_state():
    defaults = {
        "model": None, "engine": None,
        "evidence": {},         # {node_str: 0|1}
        "profile_ev": {},       # {node_id: float}
        "gladue_checked": set(),
        "sce_checked": set(),
        "posteriors": {},
        "qbism_diags": {},
        "connection_strength": "moderate",
        "ellis_nexus": "relevant",
        "sce_framework": "morris",
        "ev_sources": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = build_model()
    e = get_inference_engine(m)
    return m, e

if st.session_state.model is None:
    with st.spinner("Initialising Bayesian network (Variable Elimination)..."):
        m, e = load_model()
        st.session_state.model = m
        st.session_state.engine = e

# ── Inference runner ──────────────────────────────────────────────────────────
def run_inference():
    """Merge all evidence sources and run Variable Elimination."""
    combined_ev = {}
    # Profile evidence → convert to binary observations
    for nid, prob in st.session_state.profile_ev.items():
        combined_ev[str(nid)] = 1 if prob >= 0.5 else 0
    # Manual binary evidence overrides
    combined_ev.update(st.session_state.evidence)
    # Remove Node 20 from evidence (it's our query target)
    combined_ev.pop('20', None)

    posteriors = query_do_risk(st.session_state.engine, combined_ev)

    # Apply continuous SCE / Gladue adjustments on top of VE output
    # (VE handles binary evidence; continuous deltas are post-processed)
    gl_delta = compute_gladue_delta()
    sce_delta = compute_sce_delta()
    for nid, delta in {**gl_delta, **sce_delta}.items():
        if nid in posteriors and str(nid) not in combined_ev:
            posteriors[nid] = float(np.clip(posteriors[nid] + delta, 0.05, 0.95))

    # Recompute Node 20 with corrected inputs
    do_raw = (
        0.30 * posteriors.get(2, 0.5) +
        0.25 * posteriors.get(3, 0.5) +
        0.20 * posteriors.get(4, 0.5) +
        0.25 * posteriors.get(18, 0.5)
    )
    dst = (
        0.22 * posteriors.get(5, 0.5) +
        0.18 * posteriors.get(6, 0.5) +
        0.22 * posteriors.get(12, 0.5) +
        0.15 * posteriors.get(14, 0.5) +
        0.10 * posteriors.get(15, 0.5) +
        0.08 * posteriors.get(17, 0.5) +
        0.05 * posteriors.get(16, 0.5)
    )
    posteriors[20] = float(np.clip(do_raw * (1 - 0.68 * dst) + 0.03, 0.05, 0.93))

    st.session_state.posteriors = posteriors

    # Run QBism diagnostics
    diags = diagnose(
        posteriors=posteriors,
        evidence=combined_ev,
        gladue_checked=list(st.session_state.gladue_checked),
        sce_checked=list(st.session_state.sce_checked),
        profile_ev=st.session_state.profile_ev,
        connection_strength=st.session_state.connection_strength,
    )
    st.session_state.qbism_diags = diags

# ── Gladue / SCE delta computation ───────────────────────────────────────────
GLADUE_FACTORS = [
    {"id": "g_res1",    "label": "Residential school — direct",          "node": 10, "weight": 0.18, "col": 1, "section": "Intergenerational & historical trauma"},
    {"id": "g_res2",    "label": "Residential school — familial",         "node": 10, "weight": 0.14, "col": 1, "section": "Intergenerational & historical trauma"},
    {"id": "g_sixties", "label": "Sixties Scoop / child welfare removal", "node": 10, "weight": 0.14, "col": 1, "section": "Intergenerational & historical trauma"},
    {"id": "g_disp",    "label": "Community displacement / relocation",   "node": 10, "weight": 0.10, "col": 1, "section": "Intergenerational & historical trauma"},
    {"id": "g_cult",    "label": "Loss of language and cultural identity","node": 12, "weight": 0.10, "col": 1, "section": "Cultural disconnection"},
    {"id": "g_spirit",  "label": "Absence of spiritual / ceremonial access","node": 11,"weight": 0.08,"col": 1,"section": "Cultural disconnection"},
    {"id": "g_fv",      "label": "Family violence / domestic abuse",      "node": 10, "weight": 0.12, "col": 1, "section": "Childhood & family"},
    {"id": "g_foster",  "label": "Foster care / group home placement",    "node": 10, "weight": 0.10, "col": 1, "section": "Childhood & family"},
    {"id": "g_pov",     "label": "Chronic poverty",                       "node": 10, "weight": 0.08, "col": 2, "section": "Socioeconomic"},
    {"id": "g_house",   "label": "Unstable housing / homelessness",       "node": 18, "weight": 0.08, "col": 2, "section": "Socioeconomic"},
    {"id": "g_emp",     "label": "Structural employment barriers",        "node": 18, "weight": 0.07, "col": 2, "section": "Socioeconomic"},
    {"id": "g_edu",     "label": "Disrupted or denied education",         "node": 10, "weight": 0.07, "col": 2, "section": "Socioeconomic"},
    {"id": "g_sub",     "label": "Substance use linked to trauma",        "node": 18, "weight": 0.09, "col": 2, "section": "Substance use & mental health"},
    {"id": "g_mh",      "label": "Untreated mental health conditions",    "node": 18, "weight": 0.08, "col": 2, "section": "Substance use & mental health"},
    {"id": "g_grief",   "label": "Chronic grief and loss",                "node": 10, "weight": 0.08, "col": 2, "section": "Substance use & mental health"},
    {"id": "g_op",      "label": "Over-policed community of origin",      "node": 14, "weight": 0.14, "col": 2, "section": "Systemic justice"},
    {"id": "g_yj",      "label": "Young offender system involvement",     "node": 14, "weight": 0.09, "col": 2, "section": "Systemic justice"},
    {"id": "g_prior",   "label": "Prior sentencing without Gladue analysis","node": 12,"weight": 0.12,"col": 2,"section": "Systemic justice"},
]

SCE_FACTORS = [
    {"id": "s_racism",      "label": "Anti-Black / anti-racialized racism documented", "node": 12, "weight": 0.16, "fw": "morris", "section": "Structural racism"},
    {"id": "s_nbhd",        "label": "Neighbourhood-level structural disadvantage",     "node": 14, "weight": 0.14, "fw": "morris", "section": "Structural racism"},
    {"id": "s_cviol",       "label": "Community violence exposure",                     "node": 10, "weight": 0.12, "fw": "morris", "section": "Structural racism"},
    {"id": "s_rprofil",     "label": "Documented racial profiling",                     "node": 14, "weight": 0.15, "fw": "morris", "section": "Structural racism"},
    {"id": "s_irca",        "label": "IRCA filed and before the court",                "node": 12, "weight": 0.20, "fw": "morris", "section": "IRCA"},
    {"id": "s_irca_rej",    "label": "IRCA filed but disregarded by court",            "node": 12, "weight": 0.18, "fw": "morris", "section": "IRCA"},
    {"id": "s_blk_inc",     "label": "Anti-Black systemic incarceration patterns",     "node": 14, "weight": 0.13, "fw": "morris", "section": "Black offender patterns"},
    {"id": "s_blk_bail",    "label": "Anti-Black bail practices documented",           "node": 7,  "weight": 0.12, "fw": "morris", "section": "Black offender patterns"},
    {"id": "s_blk_edu",     "label": "Racialized educational exclusion",               "node": 10, "weight": 0.10, "fw": "morris", "section": "Black offender patterns"},
    {"id": "s_state_care",  "label": "State care involvement (non-racialized)",        "node": 10, "weight": 0.14, "fw": "ellis",  "section": "Ellis — socio-economic"},
    {"id": "s_e_pov",       "label": "Chronic poverty / economic deprivation",         "node": 18, "weight": 0.10, "fw": "ellis",  "section": "Ellis — socio-economic"},
    {"id": "s_e_trauma",    "label": "Trauma history without racialized component",    "node": 10, "weight": 0.11, "fw": "ellis",  "section": "Ellis — socio-economic"},
    {"id": "s_e_geog",      "label": "Geographic marginalization",                     "node": 11, "weight": 0.09, "fw": "ellis",  "section": "Ellis — socio-economic"},
    {"id": "s_e_edu",       "label": "Educational deprivation",                        "node": 10, "weight": 0.08, "fw": "ellis",  "section": "Ellis — socio-economic"},
    {"id": "s_parity",      "label": "Parity principle misapplied",                   "node": 12, "weight": 0.12, "fw": "both",   "section": "Judicial errors"},
    {"id": "s_seqerr",      "label": "Sequencing error — SCE applied downstream",     "node": 12, "weight": 0.14, "fw": "both",   "section": "Judicial errors"},
    {"id": "s_belief_stasis","label": "Belief stasis — SCE acknowledged but inert",   "node": 12, "weight": 0.16, "fw": "both",   "section": "Judicial errors"},
]

def connection_mult():
    return {"none": 0, "absent": 0, "weak": 0.30, "moderate": 0.65, "strong": 0.90, "direct": 1.0}.get(st.session_state.connection_strength, 0.65)

def ellis_mult():
    return {"none": 0, "peripheral": 0.35, "relevant": 0.70, "central": 1.0}.get(st.session_state.ellis_nexus, 0.70)

def compute_gladue_delta():
    delta = {}
    for f in GLADUE_FACTORS:
        if f["id"] in st.session_state.gladue_checked:
            nid = f["node"]
            delta[nid] = delta.get(nid, 0) + f["weight"]
    return delta

def compute_sce_delta():
    delta = {}
    fw = st.session_state.sce_framework
    mult = connection_mult() if fw != "ellis" else ellis_mult()
    for f in SCE_FACTORS:
        if f["id"] in st.session_state.sce_checked:
            show = (fw == "both") or (fw == "morris" and f["fw"] != "ellis") or (fw == "ellis" and f["fw"] != "morris")
            if show:
                nid = f["node"]
                delta[nid] = delta.get(nid, 0) + f["weight"] * mult
    return delta

# ── DAG visualisation ─────────────────────────────────────────────────────────
NODE_POS = {
    1: (0.50, 0.92), 2: (0.15, 0.75), 3: (0.50, 0.75), 4: (0.85, 0.75),
    5: (0.08, 0.57), 6: (0.25, 0.57), 9: (0.42, 0.57), 13: (0.58, 0.57), 15: (0.75, 0.57),
    7: (0.08, 0.40), 10: (0.25, 0.40), 11: (0.42, 0.40), 14: (0.58, 0.40), 17: (0.75, 0.40),
    8: (0.08, 0.23), 12: (0.25, 0.23), 16: (0.42, 0.23), 18: (0.58, 0.23), 19: (0.75, 0.23),
    20: (0.50, 0.05),
}

def draw_dag(posteriors, selected_node=None):
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.axis('off')
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    # Layer bands
    for y, h, label in [(0.82, 0.14, "Layer I — Substantive risk"),
                         (0.17, 0.63, "Layer II — Systemic distortion & doctrinal fidelity"),
                         (0.00, 0.14, "Layer III — Structural output")]:
        ax.add_patch(plt.Rectangle((0, y), 1.0, h, color='#f0f0f0', alpha=0.5, zorder=0))
        ax.text(0.01, y + h - 0.02, label, fontsize=7, color='#aaa', fontweight='bold', va='top')

    # Edges
    G = nx.DiGraph()
    G.add_nodes_from([str(i) for i in range(1, 21)])
    G.add_edges_from([(str(f), str(t)) for f, t in EDGES])

    for f, t in EDGES:
        x1, y1 = NODE_POS[f]; x2, y2 = NODE_POS[t]
        hl = selected_node and (f == selected_node or t == selected_node)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color='#aaa' if not hl else '#444',
                            lw=0.8 if not hl else 1.5, alpha=0.5 if not hl else 0.9))

    # Nodes
    for nid, (x, y) in NODE_POS.items():
        meta = NODE_META[nid]
        col = TYPE_COLORS[meta["type"]]
        p = posteriors.get(nid, 0.5)
        is_sel = selected_node == nid

        # Node circle — size reflects posterior
        radius = 0.038 + 0.012 * p
        circle = plt.Circle((x, y), radius,
                             color=col if is_sel else col + '33',
                             ec=col, lw=2 if is_sel else 1, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if is_sel else col, zorder=4)

        # Label below
        short = meta["short"]
        if len(short) > 12: short = short[:11] + "…"
        ax.text(x, y - radius - 0.03, short, ha='center', va='top',
                fontsize=6, color='#666', zorder=4)

        # Probability arc
        theta = np.linspace(0, 2 * np.pi * p, 50)
        arc_x = x + (radius + 0.008) * np.cos(theta - np.pi / 2)
        arc_y = y + (radius + 0.008) * np.sin(theta - np.pi / 2)
        ax.plot(arc_x, arc_y, color=col, lw=2.5, alpha=0.85, zorder=5)

    # Legend
    legend_items = [mpatches.Patch(color=c, label=TYPE_LABELS[t])
                    for t, c in TYPE_COLORS.items()]
    ax.legend(handles=legend_items, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='#ddd')

    plt.tight_layout()
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
run_inference()  # ensure posteriors are current
posteriors = st.session_state.posteriors
do_p = posteriors.get(20, 0.5)
band_label, band_color = risk_band(do_p)

col_title, col_do = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div style="padding-bottom:0.5rem;border-bottom:1px solid rgba(0,0,0,0.1)">
      <div style="font-size:1.8rem;font-weight:700;letter-spacing:4px">PARVIS</div>
      <div style="font-size:0.72rem;color:#888;margin-top:2px">
        Probabilistic and Analytical Reasoning Virtual Intelligence System &nbsp;·&nbsp;
        University of London &nbsp;·&nbsp; Ethical AI Initiative
      </div>
    </div>
    """, unsafe_allow_html=True)
with col_do:
    st.markdown(f"""
    <div style="background:{band_color}18;border:1px solid {band_color}44;border-radius:12px;
         padding:0.8rem;text-align:center;margin-top:0.3rem">
      <div style="font-size:0.7rem;color:{band_color};margin-bottom:2px">Node 20 — DO risk</div>
      <div style="font-size:2rem;font-weight:700;font-family:monospace;color:{band_color}">{do_p*100:.1f}%</div>
      <div style="font-size:0.8rem;font-weight:600;color:{band_color}">{band_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🕸️ Architecture",
    "📋 Case profile",
    "🦅 Gladue factors",
    "⚖️ Morris / Ellis SCE",
    "🔬 Evidence review",
    "📊 Inference",
    "⚛️ QBism diagnostics",
    "📄 Audit report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_dag, col_det = st.columns([3, 1])
    with col_dag:
        st.markdown("**20-node Bayesian Directed Acyclic Graph** — click a node ID to inspect")
        selected = st.selectbox("Select node", [None] + list(range(1, 21)),
                                 format_func=lambda x: "— none —" if x is None else f"Node {x}: {NODE_META[x]['short']}")
        fig = draw_dag(posteriors, selected_node=selected)
        st.pyplot(fig, use_container_width=True)

    with col_det:
        if selected:
            meta = NODE_META[selected]
            col = TYPE_COLORS[meta["type"]]
            p = posteriors.get(selected, 0.5)
            st.markdown(f"""
            <div style="background:{col}18;border:1px solid {col}44;border-radius:10px;padding:1rem;margin-bottom:1rem">
              <div style="font-size:0.7rem;color:{col};font-weight:600">{TYPE_LABELS[meta['type']]}</div>
              <div style="font-size:1rem;font-weight:600;margin-top:4px">Node {selected}: {meta['name']}</div>
              <div style="font-size:1.8rem;font-weight:700;font-family:monospace;color:{col};margin-top:8px">{p*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("**Node types**")
            for t, c in TYPE_COLORS.items():
                st.markdown(f"<span style='color:{c}'>●</span> {TYPE_LABELS[t]}", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: CASE PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Case profile")
    st.caption("Quantitative case characteristics. Each field maps to network nodes and drives Variable Elimination.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Offender characteristics")
        age = st.slider("Age at sentencing", 18, 80, 35, help="Node 15 — temporal distortion / burnout effect")
        identity = st.selectbox("Identity background", [
            "Not recorded / unknown", "Indigenous — s.718.2(e) + Gladue",
            "Black — Morris IRCA framework", "Other racialized — Morris",
            "Non-racialized, socially disadvantaged — Ellis", "No identified systemic disadvantage"
        ])
        pclr = st.slider("PCL-R score", 0, 40, 20, help="Node 3. ≥30 = high psychopathy (Ewert/Larsen caveats apply)")
        static99 = st.slider("Static-99R score", 0, 12, 3, help="Node 4. ≥6 = high sexual recidivism risk")
        violence = st.selectbox("Serious violence history", ["None", "Minor/historical", "Moderate", "Serious", "Established pattern"])
        fasd = st.selectbox("FASD diagnosis", ["None / not assessed", "Suspected, undiagnosed", "Confirmed diagnosis"])

        st.markdown("##### Dynamic risk (Node 18)")
        substance = st.selectbox("Substance use (active)", ["None / in remission", "Low", "Moderate", "High — dependency"])
        peers = st.selectbox("Antisocial peer associations", ["None identified", "Some — limited", "Strong — primary network"])
        stability = st.selectbox("Employment / housing stability", ["Stable", "Marginal", "Unstable / homeless"])

    with col2:
        st.markdown("##### Procedural integrity")
        detention = st.slider("Pre-trial detention (days)", 0, 730, 60, help="Node 7. >90 days triggers coercive plea cascade")
        counsel = st.selectbox("Quality of defence counsel", ["Adequate", "Marginal", "Inadequate — no cultural investigation", "Ineffective — constitutional breach"])
        gladue_report = st.selectbox("Gladue / SCE report commissioned", ["Yes — full report before court", "Partial / summary only", "No report commissioned", "Report commissioned, disregarded"])
        tools = st.selectbox("Risk tools applied", ["Culturally validated only", "Mix — partially qualified", "Standard, no cultural qualification", "No actuarial tools"])
        policing = st.selectbox("Over-policing indicator", ["No evidence", "Some — marginal", "Strong — documented over-surveillance"])
        province = st.selectbox("Province of prosecution", ["Low DO designation rate", "Medium rate", "High DO designation rate"])

        st.markdown("##### Rehabilitative context")
        programming = st.selectbox("Indigenous / cultural programming available", ["Yes — full culturally grounded", "Limited availability", "No culturally appropriate programming"])
        rehab = st.selectbox("Rehabilitation engagement", ["Strong — consistent", "Moderate", "Minimal", "None — apparent refusal", "Anomalously positive (gaming risk)"])

    # Map profile inputs to node probabilities
    is_racialized = identity in ["Indigenous — s.718.2(e) + Gladue", "Black — Morris IRCA framework", "Other racialized — Morris"]
    pev = {}
    pev[2] = {"None":0.08,"Minor/historical":0.25,"Moderate":0.50,"Serious":0.78,"Established pattern":0.90}[violence]
    pev[3] = 0.82 if pclr>=30 else 0.55 if pclr>=20 else 0.30 if pclr>=10 else 0.12
    pev[4] = 0.82 if static99>=6 else 0.55 if static99>=4 else 0.32 if static99>=2 else 0.12
    pev[5] = {"Culturally validated only":0.10,"Mix — partially qualified":0.45,"Standard, no cultural qualification":0.85 if is_racialized else 0.40,"No actuarial tools":0.15}[tools]
    pev[6] = {"Adequate":0.15,"Marginal":0.45,"Inadequate — no cultural investigation":0.72,"Ineffective — constitutional breach":0.90}[counsel]
    pev[7] = 0.85 if detention>180 else 0.70 if detention>90 else 0.40 if detention>30 else 0.15
    pev[9] = {"None / not assessed":0.15,"Suspected, undiagnosed":0.50,"Confirmed diagnosis":0.88}[fasd]
    pev[10] = min(0.90, 0.45 + (0.20 if "Indigenous" in identity else 0))
    pev[11] = {"Yes — full culturally grounded":0.10,"Limited availability":0.55,"No culturally appropriate programming":0.85}[programming]
    pev[12] = {"Yes — full report before court":0.15,"Partial / summary only":0.50,"No report commissioned":0.82,"Report commissioned, disregarded":0.92}[gladue_report]
    pev[13] = 0.75 if rehab == "Anomalously positive (gaming risk)" else 0.22
    pev[14] = {"No evidence":0.15,"Some — marginal":0.50,"Strong — documented over-surveillance":0.85}[policing]
    pev[15] = 0.85 if age>=55 else 0.70 if age>=45 else 0.40 if age>=35 else 0.20
    pev[16] = {"Low DO designation rate":0.20,"Medium rate":0.45,"High DO designation rate":0.72}[province]
    sub_v = {"None / in remission":0.15,"Low":0.35,"Moderate":0.60,"High — dependency":0.80}[substance]
    peer_v = {"None identified":0.10,"Some — limited":0.35,"Strong — primary network":0.65}[peers]
    stab_v = {"Stable":0.10,"Marginal":0.40,"Unstable / homeless":0.70}[stability]
    pev[18] = float(np.clip((sub_v+peer_v+stab_v)/3+0.05, 0.05, 0.92))
    rehab_v = {"Strong — consistent":0.10,"Moderate":0.35,"Minimal":0.60,"None — apparent refusal":0.80,"Anomalously positive (gaming risk)":0.30}[rehab]
    pev[19] = float(np.clip(rehab_v + (0.12 if programming=="No culturally appropriate programming" else 0), 0.05, 0.90))

    st.session_state.profile_ev = pev
    run_inference()

    band_label2, band_color2 = risk_band(st.session_state.posteriors.get(20, 0.5))
    st.markdown(f"""
    <div style="background:{band_color2}18;border:1px solid {band_color2}44;border-radius:10px;
         padding:0.8rem;text-align:center;margin-top:1rem">
      <b>Node 20 — DO designation risk</b> &nbsp;
      <span style="font-size:1.5rem;font-weight:700;font-family:monospace;color:{band_color2}">
        {st.session_state.posteriors.get(20,0.5)*100:.1f}%
      </span>
      <span style="color:{band_color2};font-weight:600"> {band_label2}</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: GLADUE FACTORS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Gladue factors checklist")
    st.caption("*R v Gladue* [1999] 1 SCR 688 · *R v Ipeelee* [2012] SCC 13 · Does not require proof of causation between factors and offence")

    sections = {}
    for f in GLADUE_FACTORS:
        sections.setdefault(f["section"], []).append(f)

    col1, col2 = st.columns(2)
    col_map = {1: col1, 2: col2}

    for f in GLADUE_FACTORS:
        col = col_map[f["col"]]

    checked_gladue = set()
    for section, factors in sections.items():
        col = col_map[factors[0]["col"]]
        with col:
            st.markdown(f"<div class='section-header'>{section}</div>", unsafe_allow_html=True)
            for f in factors:
                node_col = TYPE_COLORS[NODE_META[f["node"]]["type"]]
                checked = st.checkbox(
                    f"**{f['label']}** — N{f['node']} (+{f['weight']*100:.0f}%)",
                    key=f"gl_{f['id']}",
                    value=f["id"] in st.session_state.gladue_checked
                )
                if checked:
                    checked_gladue.add(f["id"])

    st.session_state.gladue_checked = checked_gladue
    run_inference()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: MORRIS / ELLIS SCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Morris / Ellis — Social context evidence")
    st.caption("*R v Morris* 2021 ONCA 680 · *R v Ellis* 2022 BCCA 278")

    col_fw, col_gate = st.columns([1, 2])
    with col_fw:
        fw = st.radio("Applicable framework", ["Morris", "Ellis", "Both"],
                       index=["morris","ellis","both"].index(st.session_state.sce_framework))
        st.session_state.sce_framework = fw.lower()

    with col_gate:
        if st.session_state.sce_framework != "ellis":
            st.markdown("**Morris para 97 — connection gate**")
            st.caption("Discernible nexus to offence / moral culpability — not a causation requirement")
            conn = st.select_slider("Connection strength",
                options=["none","absent","weak","moderate","strong","direct"],
                value=st.session_state.connection_strength)
            st.session_state.connection_strength = conn
            mult_display = connection_mult()
            st.markdown(f"Weight multiplier applied to all Morris SCE: **{mult_display:.0%}**")

        if st.session_state.sce_framework != "morris":
            st.markdown("**Ellis deprivation nexus**")
            nexus = st.selectbox("Deprivation nexus", ["none","peripheral","relevant","central"],
                                  index=["none","peripheral","relevant","central"].index(st.session_state.ellis_nexus))
            st.session_state.ellis_nexus = nexus

    st.markdown("---")
    checked_sce = set()
    sce_sections = {}
    fw_filter = st.session_state.sce_framework
    for f in SCE_FACTORS:
        show = fw_filter=="both" or (fw_filter=="morris" and f["fw"]!="ellis") or (fw_filter=="ellis" and f["fw"]!="morris")
        if show:
            sce_sections.setdefault(f["section"], []).append(f)

    cols = st.columns(3)
    for i, (section, factors) in enumerate(sce_sections.items()):
        with cols[i % 3]:
            st.markdown(f"<div class='section-header'>{section}</div>", unsafe_allow_html=True)
            for f in factors:
                checked = st.checkbox(
                    f"**{f['label']}** — N{f['node']}",
                    key=f"sce_{f['id']}",
                    value=f["id"] in st.session_state.sce_checked
                )
                if checked:
                    checked_sce.add(f["id"])

    st.session_state.sce_checked = checked_sce
    run_inference()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: EVIDENCE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### Evidence review")
    st.caption("Manual fine-tuning of node probabilities. Values driven by Case Profile and Gladue/SCE tabs are shown with source tags.")

    col_r, col_d = st.columns(2)
    evN = [n for n in NODE_META if NODE_META[n].get("ev", False)]
    risk_nodes = [n for n in NODE_META if NODE_META[n]["type"] in ("risk",)]
    dst_nodes  = [n for n in NODE_META if NODE_META[n]["type"] not in ("risk","output") and n != 20]

    def render_slider_group(nodes, container):
        with container:
            for nid in nodes:
                if nid == 20: continue
                meta = NODE_META[nid]
                col = TYPE_COLORS[meta["type"]]
                current = st.session_state.posteriors.get(nid, 0.5)
                src = st.session_state.ev_sources.get(nid, "prior")
                label = f"N{nid} — {meta['short']}"
                new_val = st.slider(label, 0.0, 1.0, float(current), 0.01,
                                     key=f"ev_slider_{nid}", format="%.2f")
                if abs(new_val - current) > 0.01:
                    st.session_state.evidence[str(nid)] = 1 if new_val >= 0.5 else 0

    with col_r:
        st.markdown("##### Risk factor nodes")
        render_slider_group(risk_nodes, col_r)
    with col_d:
        st.markdown("##### Systemic distortion nodes")
        render_slider_group(dst_nodes, col_d)

    if st.button("Reset all to priors"):
        st.session_state.evidence = {}
        st.session_state.profile_ev = {}
        st.session_state.gladue_checked = set()
        st.session_state.sce_checked = set()
        st.rerun()

    run_inference()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### Inference — posterior distribution")
    st.caption("Variable Elimination posteriors across all 20 nodes. Arc on DAG visualisation reflects P(High).")

    do_p2 = st.session_state.posteriors.get(20, 0.5)
    bl2, bc2 = risk_band(do_p2)
    st.markdown(f"""
    <div style="background:{bc2}18;border:1px solid {bc2}44;border-radius:12px;
         padding:1rem 1.5rem;text-align:center;margin-bottom:1.5rem">
      <div style="font-size:0.75rem;color:{bc2}">Node 20 — Dangerous Offender designation risk</div>
      <div style="font-size:2.5rem;font-weight:700;font-family:monospace;color:{bc2}">{do_p2*100:.1f}%</div>
      <div style="font-size:0.9rem;font-weight:600;color:{bc2}">{bl2}</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, nid in enumerate([n for n in range(1, 21) if n != 20]):
        meta = NODE_META[nid]
        col = TYPE_COLORS[meta["type"]]
        p = st.session_state.posteriors.get(nid, 0.5)
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:{col}18;border:1px solid {col}33;border-radius:8px;
                 padding:0.6rem;margin-bottom:0.5rem">
              <div style="font-size:0.65rem;color:{col};font-weight:600">N{nid} — {meta['short']}</div>
              <div style="font-size:1.1rem;font-weight:700;font-family:monospace;color:{col}">{p*100:.1f}%</div>
              <div style="height:4px;background:#eee;border-radius:2px;margin-top:4px">
                <div style="width:{p*100}%;height:100%;background:{col};border-radius:2px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: QBISM DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### Quantum Bayesianism (QBism) diagnostic layer")
    st.caption("Appendix Q: *The Limits of Classical Bayesian Inference in Legally Distorted Systems* · Busemeyer & Bruza (2012) · Wojciechowski (2023)")

    st.info("This diagnostic layer does **not** alter the Variable Elimination posterior. "
            "It identifies epistemic conditions under which classical probabilistic outputs "
            "may require heightened scrutiny — per AQ.4: QBism as epistemic audit mechanism, not replacement inferential engine.")

    diags = st.session_state.qbism_diags
    if not diags:
        st.warning("Run inference first (adjust any evidence input).")
    else:
        overall = diags.get("overall_flag", "none")
        flag_class = {"high": "qbism-flag-high", "moderate": "qbism-flag-moderate", "none": "qbism-flag-none"}.get(overall, "qbism-flag-none")
        st.markdown(f"<div class='{flag_class}'><b>Overall flag: {overall.upper()}</b> — {diags.get('summary','')}</div>", unsafe_allow_html=True)

        si = diags.get("superposition_index", 0.5)
        st.markdown("---")
        st.markdown(f"**Superposition index:** {si:.2f} / 1.0")
        st.progress(si)
        st.caption(diags.get("superposition_note", ""))
        st.markdown("---")

        checks = [
            ("1. Prior contamination", "prior_contamination", "AQ.3.3.2 — Distorted priors inherited, not corrected by Bayes' theorem."),
            ("2. Order effects (non-commutativity)", "order_effects", "AQ.3.3.3 — M₁M₂ρ ≠ M₂M₁ρ · Evidentiary sequence alters belief state."),
            ("3. Contextual interference", "contextual_interference", "AQ.3.3.4 — P(H|C₁) ≠ P(H|C₂) · Kochen-Specker · Probative meaning is context-dependent."),
            ("4. Belief stasis / scalar collapse", "belief_stasis", "AQ.3.3.4 — Premature resolution of ambiguity; SCE formally acknowledged but substantively inert."),
        ]

        for title, key, doctrine in checks:
            d = diags.get(key, {})
            sev = d.get("severity", "none")
            flag_cls = {"high": "qbism-flag-high", "moderate": "qbism-flag-moderate", "none": "qbism-flag-none"}.get(sev, "qbism-flag-none")
            with st.expander(f"{title} — {sev.upper()}"):
                st.markdown(f"<div class='{flag_cls}'>{doctrine}</div>", unsafe_allow_html=True)
                items = d.get("items", [])
                if items:
                    for item in items:
                        if isinstance(item, str):
                            st.markdown(f"▸ {item}")
                        elif isinstance(item, dict):
                            for k, v in item.items():
                                st.markdown(f"**{k}:** {v}")
                else:
                    st.success("No conditions flagged for this diagnostic.")
                st.caption(f"Doctrine: {d.get('doctrine','')}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: AUDIT REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### Audit report")
    st.caption("Full inference documentation — exportable for legal review.")

    do_f = st.session_state.posteriors.get(20, 0.5)
    bl_f, _ = risk_band(do_f)
    checkedG = [f for f in GLADUE_FACTORS if f["id"] in st.session_state.gladue_checked]
    checkedS = [f for f in SCE_FACTORS if f["id"] in st.session_state.sce_checked]
    mult = connection_mult()

    qbism_text = format_report(st.session_state.qbism_diags) if st.session_state.qbism_diags else "(QBism diagnostics not yet run)"

    report = f"""PARVIS — Audit Report v3.0
Probabilistic and Analytical Reasoning Virtual Intelligence System
J.S. Patel  |  University of London (QMUL & LSE)  |  Ethical AI Initiative
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Inference engine: pgmpy Variable Elimination (genuine Bayesian inference)
──────────────────────────────────────────────────────────

INFERENCE OUTPUT
──────────────────────────────────────────────────────────
Node 20 — DO designation risk:  {do_f*100:.2f}%  [{bl_f.upper()}]

APPLICABLE FRAMEWORKS
──────────────────────────────────────────────────────────
  Gladue / Ipeelee:    R v Gladue [1999] 1 SCR 688; R v Ipeelee [2012] SCC 13
  Morris framework:    R v Morris 2021 ONCA 680 (Black / racialized offenders)
  Ellis framework:     R v Ellis 2022 BCCA 278 (socially disadvantaged offenders)
  Ewert:               Ewert v Canada [2018] SCC 30 (actuarial cultural validity)
  Active SCE framework: {st.session_state.sce_framework.upper()}
  Morris para 97 connection: {st.session_state.connection_strength.upper()} (weight: {mult:.2f})

GLADUE FACTOR CHECKLIST
──────────────────────────────────────────────────────────
{chr(10).join(f"  [✓] {f['label']} → Node {f['node']} (+{f['weight']*100:.0f}%)" for f in checkedG) or "  No Gladue factors selected"}

MORRIS / ELLIS SCE INPUTS
──────────────────────────────────────────────────────────
{chr(10).join(f"  [✓] {f['label']} → Node {f['node']} (+{f['weight']*mult*100:.1f}% after connection weight)" for f in checkedS) or "  No Morris/Ellis SCE factors selected"}

RISK FACTOR POSTERIORS (Variable Elimination)
──────────────────────────────────────────────────────────
{chr(10).join(f"  N{str(n).rjust(2)}  {NODE_META[n]['short'].ljust(26)}  {st.session_state.posteriors.get(n,0.5)*100:5.1f}%" for n in NODE_META if NODE_META[n]['type']=='risk' and n!=20)}

SYSTEMIC DISTORTION CORRECTIONS
──────────────────────────────────────────────────────────
{chr(10).join(f"  N{str(n).rjust(2)}  {NODE_META[n]['short'].ljust(26)}  {st.session_state.posteriors.get(n,0.5)*100:5.1f}%" for n in NODE_META if NODE_META[n]['type'] not in ('risk','output') and n!=20)}

{qbism_text}

ARCHITECTURAL NOTES
──────────────────────────────────────────────────────────
Inference method: Variable Elimination (pgmpy). CPTs encode normative
priors from doctrine — not empirical frequencies. Extended Bayesian
tradition: subjective, robust, decision-theoretic Bayesianism.

Node 1 (burden of proof): global constraint — P(E|H_agg) ≥ 0.90 required.
Node 20: models DO designation risk, not intrinsic dangerousness.
This distinction is the thesis's central normative contribution.

Noisy-OR applied at Node 20 (9 parents). Continuous SCE/Gladue
adjustments post-processed after VE inference.

──────────────────────────────────────────────────────────
PARVIS v3.0  |  Research use only  |  Not for live proceedings.
"""

    st.text_area("Audit report", report, height=500)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("⬇️ Download audit report (.txt)", report.encode(),
                           file_name="PARVIS_Audit_Report.txt", mime="text/plain")
    with col_dl2:
        qbism_only = format_report(st.session_state.qbism_diags) if st.session_state.qbism_diags else ""
        st.download_button("⬇️ Download QBism diagnostics (.txt)", qbism_only.encode(),
                           file_name="PARVIS_QBism_Diagnostics.txt", mime="text/plain")
