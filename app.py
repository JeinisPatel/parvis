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
.pt{font-size:2.4rem;font-weight:800;letter-spacing:7px;margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.ps{font-size:.88rem;color:#777;margin-top:5px;letter-spacing:.3px;line-height:1.5}
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
    defs = {"model":None,"engine":None,"profile_ev":{},"gladue_checked":set(),
            "sce_checked":set(),"manual_ev":{},"doc_adj":{},"posteriors":{},
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
    d={}; fw=st.session_state.scefw; m=cmult() if fw!="ellis" else emult()
    for f in SF:
        if f["id"] in st.session_state.sce_checked:
            show=fw=="both" or (fw=="morris" and f["fw"]!="ellis") or (fw=="ellis" and f["fw"]!="morris")
            if show: d[f["n"]]=d.get(f["n"],0)+f["w"]*m
    return d

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inf():
    bev={}
    for nid,prob in st.session_state.profile_ev.items(): bev[str(nid)]=1 if prob>=.5 else 0
    for nid,prob in st.session_state.manual_ev.items(): bev[str(nid)]=1 if prob>=.5 else 0
    bev.pop("20",None)
    post=query_do_risk(st.session_state.engine,bev)
    for nid,d in {**gdelta(),**sdelta(),**st.session_state.doc_adj}.items():
        if nid in post and str(nid) not in bev: post[nid]=float(np.clip(post[nid]+d,.05,.95))
    raw=.30*post.get(2,.5)+.25*post.get(3,.5)+.20*post.get(4,.5)+.25*post.get(18,.5)
    dst=.22*post.get(5,.5)+.18*post.get(6,.5)+.22*post.get(12,.5)+.15*post.get(14,.5)+.10*post.get(15,.5)+.08*post.get(17,.5)+.05*post.get(16,.5)
    post[20]=float(np.clip(raw*(1-.68*dst)+.03,.05,.93))
    st.session_state.posteriors=post
    st.session_state.qdiags=diagnose(post,bev,list(st.session_state.gladue_checked),
        list(st.session_state.sce_checked),st.session_state.profile_ev,st.session_state.conn)

run_inf()
P=st.session_state.posteriors
dp=P[20]; bl,bc,bg=rb(dp)

# ── Header ────────────────────────────────────────────────────────────────────
ct,cd=st.columns([3,1])
with ct:
    st.markdown(f"""<div style="border-bottom:1px solid rgba(0,0,0,.08);padding-bottom:.6rem;margin-bottom:.5rem">
    <div class="pt">P.A.R.V.I.S</div>
    <div class="ps">Probabilistic and Analytical Reasoning Virtual Intelligence System &nbsp;·&nbsp;
    University of London &nbsp;·&nbsp; Ethical AI Initiative</div></div>""",unsafe_allow_html=True)
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
              "⚛️ QBism diagnostics","📂 Document analysis","📄 Audit report"])

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
                    index=["morris","ellis","both"].index(st.session_state.scefw),key="scefw_r")
        st.session_state.scefw=fw.lower()
    with c2:
        if st.session_state.scefw!="ellis":
            st.markdown("**Morris para 97 — connection gate**")
            conn=st.select_slider("Connection strength",["none","absent","weak","moderate","strong","direct"],
                                  value=st.session_state.conn,key="conn_s")
            st.session_state.conn=conn
            st.info(f"Weight multiplier: **{cmult():.0%}** — {'full belief revision obligation' if cmult()>=.9 else 'partial' if cmult()>=.6 else 'limited'}")
        if st.session_state.scefw!="morris":
            nx_v=st.selectbox("Ellis deprivation nexus",["none","peripheral","relevant","central"],
                              index=["none","peripheral","relevant","central"].index(st.session_state.enex),key="enex_s")
            st.session_state.enex=nx_v
    st.markdown("---")
    ss={}
    for f in SF:
        fw2=st.session_state.scefw
        show=fw2=="both" or (fw2=="morris" and f["fw"]!="ellis") or (fw2=="ellis" and f["fw"]!="morris")
        if show: ss.setdefault(f["sec"],[]).append(f)
    cs=set()
    cols3=st.columns(3)
    for i,(sec,facs) in enumerate(ss.items()):
        with cols3[i%3]:
            st.markdown(f"<div class='sh'>{sec}</div>",unsafe_allow_html=True)
            for f in facs:
                if st.checkbox(f"{f['l']} · N{f['n']}",key=f"sce_{f['id']}",
                               value=f["id"] in st.session_state.sce_checked): cs.add(f["id"])
    st.session_state.sce_checked=cs
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
                v=st.slider(f"N{nid} — {m['short']}",0.0,1.0,float(cur),.01,key=f"ev_{nid}",format="%.2f")
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
    ak=st.text_input("Anthropic API key (optional)",type="password",key="ak")
    up=st.file_uploader("Upload document",type=["txt","pdf","docx"],key="doc_up")
    if up:
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
                    result=analyze_document(content,dt,ak or None)
                st.success(f"Complete · {dt} · Framework: {result.get('applicable_framework','?').upper()} · Connection: {result.get('connection_assessment','?')}")
                st.markdown(f"*{result.get('document_summary','')}*")
                st.markdown("#### Suggested node adjustments")
                acc=dict(st.session_state.doc_adj)
                sig={k:v for k,v in result.get("nodes",{}).items() if abs(v.get("delta",0))>.02 and v.get("confidence",0)>.1}
                if sig:
                    for ns,nd in sig.items():
                        nid=int(ns);m=NODE_META.get(nid);col=TC.get(m.get("type","risk"),"#888") if m else "#888"
                        with st.expander(f"N{nid}: {m['name'] if m else '?'} — {'↑' if nd['delta']>0 else '↓'} {abs(nd['delta']):.2f} (conf {nd['confidence']:.0%})"):
                            st.markdown(f"**Reasoning:** {nd.get('reasoning','')}")
                            for c in nd.get("citations",[])[:3]: st.markdown(f"> *{c}*")
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

# ── T9: Audit report ──────────────────────────────────────────────────────────
with TABS[8]:
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
  Institution:    University of London (QMUL & LSE)
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

    st.markdown(f"<div class='at'>{rpt}</div>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.download_button("⬇️ Download audit report (.txt)",rpt.encode(),
            file_name=f"PARVIS_Audit_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",mime="text/plain")
    with c2:
        qbo=format_report(st.session_state.qdiags) if st.session_state.qdiags else ""
        st.download_button("⬇️ Download QBism diagnostics (.txt)",qbo.encode(),
            file_name="PARVIS_QBism.txt",mime="text/plain")
