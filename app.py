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

def dobar(p, label="DO designation risk"):
    bl,bc,bg = rb(p)
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
    </div></div>"""

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {"model":None,"engine":None,"profile_ev":{},"gladue_checked":set(),"criminal_record":[],"cr_doc_adj":{},
            "sce_checked":set(),"sce_values":{},"manual_ev":{},"doc_adj":{},"posteriors":{},
            "qdiags":{},"conn":"moderate","enex":"relevant","scefw":"morris","doc_res":[]}
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
    st.session_state.qdiags=diagnose(post,hard_ev,list(st.session_state.gladue_checked),
        list(st.session_state.sce_checked),st.session_state.profile_ev,st.session_state.conn)

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
    viol   = ss.get("viol", "Serious")
    fasd   = ss.get("fasd", "None / not assessed")
    sub    = ss.get("sub", "Moderate")
    peers  = ss.get("peers", "Some — limited")
    stab   = ss.get("stab", "Marginal")
    det    = ss.get("det", 60)
    counsel= ss.get("counsel", "Marginal")
    gr     = ss.get("gr", "No report commissioned")
    tools  = ss.get("tools", "Standard, no cultural qualification")
    pol    = ss.get("pol", "Strong — documented over-surveillance")
    prov   = ss.get("prov", "Medium rate")
    prog   = ss.get("prog", "Limited availability")
    rehab  = ss.get("rehab", "Minimal")

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
    from canlii_client import (search_node_developments, get_tetrad_updates,
                                is_configured as canlii_ok)
    CANLII_ON = True
except ImportError:
    CANLII_ON = False
    def canlii_ok(): return False

# ── Tabs ──────────────────────────────────────────────────────────────────────
TABS=st.tabs(["🕸️ Architecture","📋 Case profile","🦅 Gladue factors",
              "⚖️ Morris / Ellis SCE","🔬 Evidence review","📊 Inference",
              "⚛️ QBism diagnostics","📂 Document analysis","📜 Criminal record","📄 Audit report"])

# ── T1: Architecture ──────────────────────────────────────────────────────────
with TABS[0]:
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
            st.markdown(dobar(dp),unsafe_allow_html=True)

# ── T2: Case profile ──────────────────────────────────────────────────────────
with TABS[1]:
    st.markdown("### Case profile")
    st.caption("Each field maps to network nodes and drives Variable Elimination.")
    st.markdown(dobar(P[20]),unsafe_allow_html=True)
    c1,c2=st.columns(2);pev={}
    with c1:
        st.markdown("##### Offender characteristics")
        age=st.slider("Age at sentencing",18,80,35,key="age")
        st.caption(f"Node 15 — age {age}: {'strong burnout attenuation' if age>=55 else 'moderate' if age>=45 else 'minimal'}")
        identity=st.selectbox("Identity background",["Not recorded / unknown",
            "Indigenous — s.718.2(e) + Gladue applies","Black — Morris IRCA framework",
            "Other racialized — Morris framework","Non-racialized, socially disadvantaged — Ellis",
            "No identified systemic disadvantage"],key="id_bg")
        pclr=st.slider("PCL-R score",0,40,20,key="pclr")
        st.caption(f"N3: {'High ≥30 — Ewert/Larsen caveat APPLIES' if pclr>=30 else 'Moderate' if pclr>=20 else 'Low'}")
        s99=st.slider("Static-99R score",0,12,3,key="s99")
        st.caption(f"N4: {'High ≥6' if s99>=6 else 'Moderate' if s99>=4 else 'Low'} — Ewert validation caveat")
        violence=st.selectbox("Serious violence history",["None","Minor/historical","Moderate","Serious","Established pattern"],key="viol")
        fasd=st.selectbox("FASD diagnosis",["None / not assessed","Suspected, undiagnosed","Confirmed diagnosis"],key="fasd")
        st.markdown("##### Dynamic risk · Node 18")
        sub=st.selectbox("Substance use",["None / in remission","Low","Moderate","High — dependency"],key="sub")
        peers=st.selectbox("Antisocial peer associations",["None identified","Some — limited","Strong — primary network"],key="peers")
        stab=st.selectbox("Employment / housing stability",["Stable","Marginal","Unstable / homeless"],key="stab")
    with c2:
        st.markdown("##### Procedural integrity · Distortion nodes")
        det=st.slider("Pre-trial detention (days)",0,730,60,key="det")
        st.caption(f"N7: {'HIGH — coercive plea cascade risk' if det>90 else 'Moderate' if det>30 else 'Low'} ({det} days)")
        counsel=st.selectbox("Quality of defence counsel",["Adequate","Marginal",
            "Inadequate — no cultural investigation","Ineffective — constitutional breach"],key="counsel")
        gr=st.selectbox("Gladue / SCE report commissioned",["Yes — full report before court",
            "Partial / summary only","No report commissioned","Report commissioned, disregarded"],key="gr")
        tools=st.selectbox("Risk tools applied",["Culturally validated only","Mix — partially qualified",
            "Standard, no cultural qualification","No actuarial tools"],key="tools")
        pol=st.selectbox("Over-policing indicator",["No evidence","Some — marginal",
            "Strong — documented over-surveillance"],key="pol")
        prov=st.selectbox("Province of prosecution",["Low DO designation rate","Medium rate",
            "High DO designation rate"],key="prov")
        st.markdown("##### Rehabilitative context · Nodes 11, 19")
        prog=st.selectbox("Indigenous / cultural programming",["Yes — full culturally grounded",
            "Limited availability","No culturally appropriate programming"],key="prog")
        st.caption("Natomagan 2022 ABCA 48: absence is systemic failure, not offender characteristic")
        rehab=st.selectbox("Rehabilitation engagement",["Strong — consistent","Moderate","Minimal",
            "None — apparent refusal","Anomalously positive (gaming risk)"],key="rehab")

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
    st.success(f"Node 20 — DO designation risk: **{P[20]*100:.1f}%** · {bl2}")

# ── T3: Gladue ────────────────────────────────────────────────────────────────
with TABS[2]:
    st.markdown("### Gladue factors")
    st.caption("*R v Gladue* [1999] · *R v Ipeelee* [2012] · No causation requirement")
    st.markdown(dobar(P[20]),unsafe_allow_html=True)
    secs={}
    for f in GF: secs.setdefault(f["sec"],[]).append(f)
    cg=set()
    c1,c2=st.columns(2)
    for sec,facs in secs.items():
        t=c1 if facs[0]["col"]==1 else c2
        with t:
            st.markdown(f"<div class='sh'>{sec}</div>",unsafe_allow_html=True)
            for f in facs:
                if st.checkbox(f"{f['l']} · N{f['n']} (+{f['w']*100:.0f}%)",key=f"gl_{f['id']}",
                               value=f["id"] in st.session_state.gladue_checked): cg.add(f["id"])
    st.session_state.gladue_checked=cg
    run_inf();P=st.session_state.posteriors
    st.success(f"Node 20: **{P[20]*100:.1f}%** · {rb(P[20])[0]}")

# ── T4: Morris/Ellis SCE ──────────────────────────────────────────────────────
with TABS[3]:
    st.markdown("### Morris / Ellis SCE")
    st.caption("*R v Morris* 2021 ONCA 680 · *R v Ellis* 2022 BCCA 278")
    st.markdown(dobar(P[20]),unsafe_allow_html=True)
    c1,c2=st.columns([1,2])
    with c1:
        fw=st.radio("Framework",["Morris","Ellis","Both"],
                    index=["morris","ellis","both"].index((st.session_state.scefw or "morris").lower() if (st.session_state.scefw or "morris").lower() in ["morris","ellis","both"] else 0),key="scefw_r")
        st.session_state.scefw=fw.lower()
    with c2:
        if (st.session_state.scefw or "morris").lower()!="ellis":
            st.markdown("**Morris para 97 — connection gate**")
            conn=st.select_slider("Connection strength",["none","absent","weak","moderate","strong","direct"],
                                  value=st.session_state.conn,key="conn_s")
            st.session_state.conn=conn
            st.info(f"Weight multiplier: **{cmult():.0%}** — {'full belief revision obligation' if cmult()>=.9 else 'partial' if cmult()>=.6 else 'limited'}")
        if (st.session_state.scefw or "morris").lower()!="morris":
            nx_v=st.selectbox("Ellis deprivation nexus",["none","peripheral","relevant","central"],
                              index=["none","peripheral","relevant","central"].index(st.session_state.enex),key="enex_s")
            st.session_state.enex=nx_v
    st.markdown("---")
    # ── SCE factors as continuous sliders (0 = absent, 1 = fully established) ──
    ss={}
    for f in SF:
        fw2=(st.session_state.scefw or "morris").lower()
        show=fw2=="both" or (fw2=="morris" and f["fw"]!="ellis") or (fw2=="ellis" and f["fw"]!="morris")
        if show: ss.setdefault(f["sec"],[]).append(f)

    sce_vals = dict(st.session_state.get("sce_values", {}))
    st.markdown("<div style='font-size:12px;color:#888;margin-bottom:8px'>"
                "Slide each factor from <b>0</b> (absent) → <b>0.5</b> (partial) → <b>1.0</b> (fully established). "
                "Partial values encode degrees of evidentiary strength.</div>",
                unsafe_allow_html=True)

    cols3=st.columns(3)
    for i,(sec,facs) in enumerate(ss.items()):
        with cols3[i%3]:
            st.markdown(f"<div class='sh'>{sec}</div>",unsafe_allow_html=True)
            for f in facs:
                fw2=(st.session_state.scefw or "morris").lower()
                col_sce = "#185FA5" if fw2=="morris" or f["fw"]=="both" else "#534AB7"
                cur_val = sce_vals.get(f["id"], 0.0)
                v = st.slider(
                    f"{f['l']} · N{f['n']}",
                    0.0, 1.0, float(cur_val), 0.05,
                    key=f"sce_{f['id']}",
                    format="%.2f"
                )
                sce_vals[f["id"]] = v
                if v > 0.01:
                    pct = f"+{v*f['w']*cmult()*100:.1f}pp" if fw2!="ellis" else f"+{v*f['w']*emult()*100:.1f}pp"
                    st.markdown(
                        f"<div style='font-size:.68rem;color:{col_sce};"
                        f"margin-top:-12px;margin-bottom:4px'>"
                        f"{'●' * min(int(v*5)+1, 5)} {v*100:.0f}% established · {pct}</div>",
                        unsafe_allow_html=True
                    )

    st.session_state.sce_values = sce_vals
    # Keep sce_checked in sync for backwards compatibility with QBism diagnostics
    st.session_state.sce_checked = {fid for fid,v in sce_vals.items() if v > 0.01}
    run_inf();P=st.session_state.posteriors
    st.success(f"Node 20: **{P[20]*100:.1f}%** · {rb(P[20])[0]}")

# ── T5: Evidence review ───────────────────────────────────────────────────────
with TABS[4]:
    st.markdown("### Evidence review")
    st.caption("Fine-tune node probabilities. Values auto-set from Case Profile and Gladue/SCE tabs.")
    st.markdown(dobar(P[20]),unsafe_allow_html=True)
    ev_nodes=[n for n in NODE_META if NODE_META[n]["ev"]]
    rn=[n for n in ev_nodes if NODE_META[n]["type"]=="risk"]
    dn=[n for n in ev_nodes if NODE_META[n]["type"]!="risk"]
    man=dict(st.session_state.manual_ev)
    c1,c2=st.columns(2)
    def slgrp(nodes,cont,label):
        with cont:
            st.markdown(f"##### {label}")
            for nid in nodes:
                m=NODE_META[nid];col=TC[m["type"]];cur=P.get(nid,.5)
                v=st.slider(f"N{nid} — {m['short']}",0.0,1.0,float(cur),.01,key=f"ev_{nid}",format="%.2f",label_visibility="visible")
                st.markdown(f"<div style='font-size:.72rem;color:{col};margin-top:-12px;margin-bottom:6px'>P(High) = {v*100:.0f}%</div>",unsafe_allow_html=True)
                if abs(v-st.session_state.profile_ev.get(nid,.5))>.015: man[nid]=v
    slgrp(rn,c1,"Risk factor nodes"); slgrp(dn,c2,"Systemic distortion nodes")
    st.session_state.manual_ev=man
    if st.button("Reset all to priors",key="rst"):
        for k in ["profile_ev","manual_ev","doc_adj"]: st.session_state[k]={}
        st.session_state.gladue_checked=set();st.session_state.sce_checked=set()
        st.rerun()
    run_inf()

# ── T6: Inference ─────────────────────────────────────────────────────────────
with TABS[5]:
    P=st.session_state.posteriors;dp6=P[20];bl6,bc6,bg6=rb(dp6)
    st.markdown("### Inference — posterior distribution")
    st.caption("Variable Elimination posteriors (pgmpy). Arc on DAG reflects P(High).")
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
with TABS[6]:
    st.markdown("### Quantum Bayesianism (QBism) diagnostic layer")
    st.caption("Appendix Q: *The Limits of Classical Bayesian Inference in Legally Distorted Systems* · Busemeyer & Bruza (2012) · Wojciechowski (2023)")
    st.info("This layer does **not** alter the VE posterior. It identifies epistemic conditions requiring heightened scrutiny.")
    diags=st.session_state.qdiags
    if diags:
        ov=diags.get("overall_flag","none")
        cls={"high":"qh;background:#FCEBEB;color:#A32D2D;border-left:3px solid #A32D2D",
             "moderate":"qh;background:#FAEEDA;color:#BA7517;border-left:3px solid #BA7517",
             "none":"qh;background:#EAF3DE;color:#3B6D11;border-left:3px solid #3B6D11"}.get(ov,"qh")
        st.markdown(f"<div class='{cls}'><b>Overall: {ov.upper()}</b> — {diags.get('summary','')}</div>",unsafe_allow_html=True)
        si=diags.get("superposition_index",.5);dn7=P[20]
        rw=sum(P.get(n,.5) for n in [2,3,4,18])/4
        mw=sum(P.get(n,.5) for n in [5,6,10,12,14])/5
        st.markdown("---")
        st.markdown("#### Bloch sphere — quantum belief state |ψ⟩")
        st.caption("State vector on Bloch sphere. Equator (θ=90°) = maximum superposition. Poles = fully resolved belief.")
        cb,cc=st.columns([2,1])
        with cb:
            fb=draw_bloch_sphere(dn7,rw,mw,dn7,f"P(DO)={dn7*100:.1f}%")
            st.pyplot(fb,use_container_width=True)
        with cc:
            fc=draw_comparison_chart(dn7,dn7,si)
            st.pyplot(fc,use_container_width=True)
            theta_deg=np.degrees(np.arccos(np.clip(1-2*dn7,-1,1)))
            st.markdown(f"""**State vector:**\n- θ = `{theta_deg:.1f}°` (polar)\n- SI = `{si:.2f}`\n- |α|² = `{dn7:.3f}`\n- |β|² = `{1-dn7:.3f}`\n\n*AQ.3.3.5.2: pre-decisional ambiguity preserved as stable epistemic condition.*""")
        st.markdown("---")
        for ttl,key,doc in [("1. Prior contamination","prior_contamination","AQ.3.3.2 — Distorted priors propagated, not corrected"),
                             ("2. Order effects","order_effects","AQ.3.3.3 — M₁M₂ρ ≠ M₂M₁ρ · sequence alters belief"),
                             ("3. Contextual interference","contextual_interference","AQ.3.3.4 — P(H|C₁) ≠ P(H|C₂) · Kochen-Specker"),
                             ("4. Belief stasis","belief_stasis","AQ.3.3.4 — SCE acknowledged but inert")]:
            d=diags.get(key,{});sev=d.get("severity","none")
            cs2={"high":"qh;background:#FCEBEB;color:#A32D2D;border-left:3px solid #A32D2D",
                 "moderate":"qh;background:#FAEEDA;color:#BA7517;border-left:3px solid #BA7517",
                 "none":"qh;background:#EAF3DE;color:#3B6D11;border-left:3px solid #3B6D11"}.get(sev,"qh")
            with st.expander(f"{ttl} — {sev.upper()}"):
                st.markdown(f"<div class='{cs2}'>{doc}</div>",unsafe_allow_html=True)
                for item in d.get("items",[]):
                    if isinstance(item,str): st.markdown(f"▸ {item}")
                    elif isinstance(item,dict):
                        for k,v in item.items(): st.markdown(f"**{k}:** {v}")
                if not d.get("items"): st.success("No conditions flagged.")
                st.caption(d.get("doctrine",""))

# ── T8: Document analysis ─────────────────────────────────────────────────────
with TABS[7]:
    st.markdown("### Document analysis")
    st.caption("Upload legal documents for Tetrad-grounded analysis. The LLM provides guidance — **you retain full discretion**.")
    st.info("**Supported:** Gladue reports · IRCA reports · PCL-R/Static-99R assessments · Prior decisions · Transcripts · Bail records · Trauma assessments · Ineffective assistance records")
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
                # ── CanLII live case law ──────────────────────────────────
                flagged=[int(k) for k,v in sig.items() if abs(v.get("delta",0))>0.02]
                if CANLII_ON and canlii_ok() and flagged:
                    st.markdown("---")
                    st.markdown("#### 🔍 Live CanLII — recent decisions on flagged nodes")
                    st.caption("Querying CanLII for recent cases on the same doctrinal issues...")
                    for nid in flagged[:4]:
                        nm=NODE_META.get(nid,{}); col=TC.get(nm.get("type","distortion"),"#185FA5")
                        with st.spinner(f"Searching Node {nid}..."):
                            cas=search_node_developments(nid,max_results=4)
                        if cas:
                            st.markdown(f"<b style='color:{col}'>N{nid} — {nm.get('short','')}</b>",unsafe_allow_html=True)
                            for r in cas:
                                dt2=r.get("date","")[:10] or "—"
                                cit2=r.get("citation") or r.get("title","—")
                                ur=r.get("url","")
                                lnk=f"<a href='{ur}' target='_blank'>{cit2}</a>" if ur else cit2
                                st.markdown(f"<div style='font-size:12px;padding:2px 0;color:#555'>[{dt2}] {lnk}</div>",unsafe_allow_html=True)
                        else:
                            st.caption(f"N{nid}: No recent results found.")
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

    # ── Tetrad subsequent history tracker ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📡 Tetrad subsequent history — live from CanLII")
    st.caption("Tracks recent citing cases for Gladue, Morris, Ellis, Ewert, Natomagan, Boutilier, Ipeelee.")
    if CANLII_ON and canlii_ok():
        col_btn,col_yr=st.columns([2,1])
        with col_yr:
            since=st.selectbox("Since year",[2024,2023,2022,2021],index=1,key="tet_yr")
        with col_btn:
            run_tet=st.button("🔄 Check Tetrad for recent developments",key="tet_btn")
        if run_tet:
            with st.spinner("Querying CanLII..."):
                upd=get_tetrad_updates(since_year=since)
            if upd:
                for case_lbl,citing in upd.items():
                    with st.expander(f"📌 {case_lbl} — {len(citing)} recent citing case(s)"):
                        for c in citing[:6]:
                            cd=c.get("decisionDate","")[:10] or "—"
                            ct=c.get("title","—")
                            cu=c.get("url","")
                            lnk=f"<a href='{cu}' target='_blank'>{ct}</a>" if cu else ct
                            st.markdown(f"<div style='font-size:12px;padding:3px 0'><span style='color:#aaa'>[{cd}]</span> {lnk}</div>",unsafe_allow_html=True)
            else:
                st.success(f"No new citing cases found since {since}.")
        st.markdown("#### ⚡ Doctrine update alerts")
        st.caption("Nodes with actively evolving law — flagged in doctrine.py:")
        try:
            from doctrine import get_update_notes
            notes=get_update_notes()
            for nid,note in notes.items():
                nm=NODE_META.get(nid,{}); col=TC.get(nm.get("type","distortion"),"#185FA5")
                st.markdown(f"<div style='border-left:3px solid {col};padding:6px 12px;margin:4px 0;background:{col}11;border-radius:0 6px 6px 0'><b style='color:{col}'>N{nid} — {nm.get('short','')}</b><br><span style='font-size:12px'>{note}</span></div>",unsafe_allow_html=True)
        except Exception:
            st.caption("doctrine.py update notes unavailable.")
    else:
        st.markdown(
            "<div style='background:#FAEEDA;border:1px solid #BA751733;border-radius:8px;padding:12px 16px'>"
            "<b>CanLII not yet active.</b> To enable live Tetrad tracking:<br>"
            "1. Register free at <a href='https://api.canlii.org' target='_blank'>api.canlii.org</a><br>"
            "2. Add <code>CANLII_API_KEY = your-key</code> to Streamlit secrets<br>"
            "3. Redeploy — tracker activates automatically.</div>",
            unsafe_allow_html=True)

# ── T9: Criminal record ──────────────────────────────────────────────────────
with TABS[8]:
    st.markdown("### Criminal record")
    st.caption("Enter prior convictions and calibrate each entry's evidentiary weight against doctrinal distortion nodes.")
    st.markdown(dobar(P[20]),unsafe_allow_html=True)

    # ── Distortion node weights ───────────────────────────────────────────────
    # Used to compute calibrated reliability for each conviction
    pN7  = st.session_state.posteriors.get(7,  0.15)   # bail-denial cascade
    pN6  = st.session_state.posteriors.get(6,  0.15)   # ineffective counsel
    pN5  = st.session_state.posteriors.get(5,  0.10)   # invalid risk tools / Ewert
    pN14 = st.session_state.posteriors.get(14, 0.15)   # over-policing
    pN15 = st.session_state.posteriors.get(15, 0.40)   # temporal distortion / age
    pN12 = st.session_state.posteriors.get(12, 0.15)   # Gladue misapplication

    # ── Authoritative correction references ───────────────────────────────────
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

    # ── Feed function defined first so button callbacks can call it ─────────────
    # ── Offence seriousness tiers (Boutilier pattern analysis) ───────────────────
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
            cr_adj[2] = n2_raw - st.session_state.posteriors.get(2, 0.08)
            st.session_state.cr_doc_adj[2] = n2_raw

        # ── N18: Dynamic risk — escalation signal ─────────────────────────────
        esc = _detect_escalation(rec)
        st.session_state.cr_doc_adj["escalation"] = esc
        if abs(esc["signal"]) > 0.01:
            cr_adj[18] = float(np.clip(esc["signal"], -0.30, 0.35))

        # ── Gang / organized crime context → N14 boost ────────────────────────
        gang_entries = [e for e in rec if e.get("gang", False)]
        if gang_entries:
            gang_factor = float(np.clip(0.08 * len(gang_entries), 0.0, 0.25))
            cr_adj[14] = cr_adj.get(14, 0) + gang_factor

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

    # ── UI: Add conviction ─────────────────────────────────────────────────────
    st.markdown("#### Add conviction")
    with st.expander("➕ Enter a new conviction", expanded=len(st.session_state.criminal_record)==0):
        ca1, ca2 = st.columns([2,1])
        with ca1:
            cr_offence   = st.text_input("Offence description", placeholder="e.g. Aggravated assault s.268 CC", key="cr_off")
            cr_court     = st.text_input("Court", placeholder="e.g. Ontario Superior Court of Justice", key="cr_court")
        with ca2:
            cr_year      = st.number_input("Year of conviction", min_value=1950, max_value=2026, value=2015, step=1, key="cr_year")
            cr_sentence  = st.text_input("Sentence", placeholder="e.g. 18 months custody", key="cr_sent")
        cr_jurisdiction = st.selectbox("Province / jurisdiction",
            ["ON","BC","AB","QC","SK","MB","NS","NB","NL","PE","YT","NT","NU","Federal"],
            key="cr_jur")

        cf1, cf2 = st.columns(2)
        with cf1:
            cr_seriousness = st.select_slider(
                "Offence seriousness tier",
                options=["Minor (0.15–0.25)","Moderate (0.35–0.50)","Significant (0.55–0.75)",
                         "Serious violent (0.80–0.85)","Catastrophic (1.00)"],
                value="Significant (0.55–0.75)", key="cr_seriousness",
                help="Boutilier [2017] SCC 64 — offence seriousness determines base weight in pattern analysis"
            )
        with cf2:
            cr_gang = st.checkbox(
                "Gang / organized crime context",
                value=False, key="cr_gang",
                help="R v Lacasse [2015] SCC 64 — gang context aggravating; but also consider Le [2019] SCC 34 — may reflect over-policing of community rather than individual culpability"
            )

        st.markdown("**Doctrinal reliability adjustments for this conviction**")
        st.caption("Each slider reflects the degree to which this specific conviction's evidentiary weight should be discounted based on the distortion nodes currently active in PARVIS.")

        ca3, ca4 = st.columns(2)
        with ca3:
            adj_bail   = st.slider("Bail-denial / coercive plea reduction",  0.0, 1.0, min(pN7, 0.70),  0.05, key="cr_adj_bail",
                help="R v Antic [2017] SCC 27 — conviction may reflect coercive plea under detention rather than genuine admission of guilt")
            adj_ewert  = st.slider("Ewert tool-invalidity reduction",         0.0, 1.0, min(pN5, 0.50),  0.05, key="cr_adj_ewert",
                help="Ewert v Canada [2018] SCC 30 — actuarial evidence underlying this conviction may carry reduced cultural validity")
            adj_gladue = st.slider("Gladue misapplication reduction",         0.0, 1.0, min(pN12, 0.50), 0.05, key="cr_adj_gladue",
                help="R v Morris 2021 ONCA 680 — contextual factors may have been ignored or misapplied at original sentencing")
        with ca4:
            adj_police = st.slider("Over-policing / record inflation reduction",0.0, 1.0, min(pN14, 0.50),0.05, key="cr_adj_police",
                help="R v Le [2019] SCC 34 — conviction may be partly a product of racialised over-policing rather than actual criminogenic conduct")
            adj_mm     = st.slider("Mandatory minimum distortion reduction",   0.0, 1.0, 0.0,             0.05, key="cr_adj_mm",
                help="R v Nur [2015] / Lloyd [2016] — conviction may have been influenced by a now-struck mandatory minimum")
            adj_time   = st.slider("Temporal attenuation (age / distance)",    0.0, 1.0, min(pN15*0.6, 0.80),0.05, key="cr_adj_time",
                help="R v Boutilier [2017] SCC 64 — old convictions carry reduced weight especially where age burnout is active")

        # Compute calibrated weight for this entry
        raw_wt = 1.0
        cal_wt = float(np.clip(
            raw_wt * (1 - 0.55*adj_bail) * (1 - 0.40*adj_ewert) *
            (1 - 0.35*adj_police) * (1 - 0.30*adj_gladue) *
            (1 - 0.25*adj_mm) * (1 - 0.45*adj_time), 0.05, 1.0))
        pct_ret = cal_wt * 100

        col_ret = "#3B6D11" if pct_ret >= 70 else "#BA7517" if pct_ret >= 40 else "#A32D2D"
        st.markdown(
            f"<div style='background:#f4f6fa;border-radius:10px;padding:.7rem 1rem;margin-top:.5rem'>"
            f"<span style='font-size:.78rem;color:#666'>Calibrated evidentiary weight: </span>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{col_ret}'>{pct_ret:.0f}%</span>"
            f"<span style='font-size:.75rem;color:#888;margin-left:.5rem'>of nominal weight retained</span></div>",
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
                    "offence":       cr_offence.strip(),
                    "court":         cr_court.strip(),
                    "year":          int(cr_year),
                    "sentence":      cr_sentence.strip(),
                    "jurisdiction":  cr_jurisdiction,
                    "seriousness":   _ser_map.get(cr_seriousness, 0.65),
                    "seriousness_label": cr_seriousness,
                    "gang":          cr_gang,
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
                # Feed calibrated N2 and distortion corrections back into the network
                _cr_feed_nodes()
                st.rerun()
            else:
                st.warning("Please enter an offence description.")

    # ── Feed function: compute aggregate N2 and distortion signals ────────────
    # ── Record table ──────────────────────────────────────────────────────────
    rec = st.session_state.criminal_record
    if rec:
        st.markdown("---")
        st.markdown(f"#### Calibrated criminal record — {len(rec)} conviction(s)")

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
            f"<div style='font-size:.75rem;color:#888;margin-bottom:2px'>Pattern (Boutilier)</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{esc_col}'>{esc_pat.title()}</div>",
            unsafe_allow_html=True)

        if esc_data.get("note"):
            esc_icon = "⚠️" if esc_pat=="escalating" else "✅" if esc_pat in ("de-escalating","desistance") else "ℹ️"
            st.markdown(
                f"<div style='background:#f8f8f8;border-left:3px solid {esc_col};"
                f"border-radius:6px;padding:.5rem .9rem;font-size:.82rem;color:#444;margin-bottom:.6rem'>"
                f"{esc_icon} <b>Pattern analysis:</b> {esc_data['note']}</div>",
                unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # Per-conviction cards — each with its own document attachment
        for i, e in enumerate(rec):
            cal_pct = e["cal_weight"] * 100
            raw_pct = e["raw_weight"] * 100
            col_c = "#3B6D11" if cal_pct >= 70 else "#BA7517" if cal_pct >= 40 else "#A32D2D"
            adj = e["adj"]

            # Build distortion flags
            flags = []
            if adj["bail"]   > 0.15: flags.append(f"⛓ Bail-denial {adj['bail']*100:.0f}%")
            if adj["ewert"]  > 0.10: flags.append(f"📊 Ewert {adj['ewert']*100:.0f}%")
            if adj["police"] > 0.10: flags.append(f"🔵 Over-policing {adj['police']*100:.0f}%")
            if adj["gladue"] > 0.10: flags.append(f"🦅 Gladue {adj['gladue']*100:.0f}%")
            if adj["mm"]     > 0.05: flags.append(f"⚖️ Mand min {adj['mm']*100:.0f}%")
            if adj["time"]   > 0.10: flags.append(f"⏳ Temporal {adj['time']*100:.0f}%")

            # Doc-linked indicator
            has_doc  = bool(e.get("doc_analysis"))
            has_recs = bool(e.get("doc_recs"))
            doc_badge = ""
            if has_doc:
                doc_badge = "<span style='background:#E8F5E9;color:#2E7D32;border-radius:4px;padding:2px 8px;font-size:.70rem;margin-left:6px'>📎 Document linked</span>"

            flag_html = "  ".join([
                f"<span style='background:#f0f0f0;border-radius:4px;padding:2px 8px;font-size:.72rem;color:#555'>{f}</span>"
                for f in flags
            ]) if flags else "<span style='color:#aaa;font-size:.72rem'>No distortion flags</span>"

            # Pre-build badge strings to avoid nested quote conflicts in f-string
            ser_label   = e.get("seriousness_label", "—")
            ser_badge   = f"<span style='background:#f0f0f0;border-radius:4px;padding:1px 7px;font-size:.68rem;color:#555'>⚖️ {ser_label}</span>"
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
                f"<div style='border:1px solid #ddd;border-left:4px solid {col_c};"
                f"border-radius:10px;padding:.85rem 1.1rem;margin-bottom:.4rem;background:#fafafa'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
                f"<div>"
                f"<div style='font-weight:600;font-size:.95rem'>{offence_str}{doc_badge}</div>"
                f"<div style='font-size:.78rem;color:#777;margin-top:2px'>{yr_str} · {court_str} · {jur_str} · {sent_str}</div>"
                f"<div style='margin-top:4px'>{ser_badge}{gang_badge}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:90px'>"
                f"<div style='font-size:1.35rem;font-weight:700;color:{col_c}'>{cal_pct:.0f}%</div>"
                f"<div style='font-size:.7rem;color:#aaa'>calibrated weight</div>"
                f"<div style='font-size:.7rem;color:#ccc;text-decoration:line-through'>{raw_pct:.0f}% raw</div>"
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
                                f"<div style='background:#f8f8f8;border-radius:8px;padding:.6rem .9rem;margin-top:.4rem'>"
                                f"<span style='font-size:.75rem;color:#666'>Document: <b>{doc_name}</b> · Recommendation: </span>"
                                f"<span style='font-weight:700;color:{ov_col}'>{overall}</span></div>",
                                unsafe_allow_html=True)

                            # Show suggested adjustments as read-only comparison
                            st.markdown("<div style='font-size:.75rem;color:#888;margin-top:.5rem;margin-bottom:.2rem'>Suggested adjustments from document analysis:</div>", unsafe_allow_html=True)
                            adj_cols = st.columns(6)
                            adj_labels = ["bail","counsel","gladue","police","mm","time"]
                            adj_names  = ["Bail","Counsel","Gladue","Police","MM","Time"]
                            for ci,(field,lbl) in enumerate(zip(adj_labels, adj_names)):
                                key_name = f"{field}_adj"
                                if key_name in doc_recs:
                                    cur_val = e["adj"].get(field, 0.0)
                                    sug_val = doc_recs[key_name]
                                    diff = sug_val - cur_val
                                    diff_str = f"+{diff:.2f}" if diff>0 else f"{diff:.2f}"
                                    diff_col = "#A32D2D" if diff>0.05 else "#3B6D11" if diff<-0.05 else "#888"
                                    adj_cols[ci].markdown(
                                        f"<div style='text-align:center'>"
                                        f"<div style='font-size:.68rem;color:#888'>{lbl}</div>"
                                        f"<div style='font-size:.95rem;font-weight:600'>{sug_val:.2f}</div>"
                                        f"<div style='font-size:.65rem;color:{diff_col}'>{diff_str}</div>"
                                        f"</div>", unsafe_allow_html=True)

                            # One-click apply
                            if st.button(f"Apply recommended adjustments", key=f"cr_apply_{i}"):
                                for field in adj_labels:
                                    key_name = f"{field}_adj"
                                    if key_name in doc_recs:
                                        # Map counsel → counsel key in adj dict
                                        adj_key = "bail" if field=="bail" else                                                   "ewert" if field=="counsel" else                                                   "gladue" if field=="gladue" else                                                   "police" if field=="police" else                                                   "mm" if field=="mm" else "time"
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
        st.markdown("---")
        st.markdown("#### 📂 Document-assisted calibration")
        st.caption("Upload prior transcripts, SCE reports, or sentencing decisions. The document analyser will recommend weight adjustments for the record above.")

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
            st.markdown("**Document analysis — calibration recommendations:**")
            st.markdown(f"<div class='at'>{st.session_state.cr_analysis}</div>",
                unsafe_allow_html=True)
            st.caption("Review recommendations above and adjust individual conviction sliders accordingly. Changes update Node 20 in real time.")

        # ── Doctrinal footnotes ────────────────────────────────────────────────
        with st.expander("📚 Doctrinal basis for record calibration"):
            for key,(name,col,cite) in CORR_REFS.items():
                st.markdown(f"<div style='border-left:3px solid #{col};padding:.4rem .8rem;"
                    f"margin-bottom:.5rem'><b style='color:#{col}'>{name}</b><br>"
                    f"<span style='font-size:.8rem;color:#555'>{cite}</span></div>",
                    unsafe_allow_html=True)

        if st.button("Clear entire record", key="cr_clear"):
            st.session_state.criminal_record = []
            st.session_state.cr_doc_adj = {}
            st.session_state.doc_adj = {k:v for k,v in st.session_state.doc_adj.items() if k not in [7,14,12]}
            st.rerun()

    else:
        st.info("No convictions entered yet. Use the form above to add prior criminal record entries.")

    run_inf(); P=st.session_state.posteriors
    st.success(f"Node 20: **{P[20]*100:.1f}%** · {rb(P[20])[0]}")


# ── T9: Audit report ──────────────────────────────────────────────────────────
with TABS[9]:
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

    # ── Export functions ───────────────────────────────────────────────────────
    # ── Export functions (from audit_export.py) ───────────────────────────────
    from audit_export import build_docx as _build_docx_fn, build_pdf as _build_pdf_fn

    def _build_docx():
        return _build_docx_fn(
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
        return _build_pdf_fn(
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

    with st.expander("⚛️ QBism diagnostics (.txt)"):
        qbo=format_report(st.session_state.qdiags) if st.session_state.qdiags else ""
