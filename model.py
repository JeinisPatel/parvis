"""
model.py rebuild — Chapter 5 (April 11, 2026 canonical) taxonomy.

Migration strategy (per JP confirmation):
  - Preserve current CPT calibrations where there's an obvious content match
  - Conservative defaults for new CH5 nodes (N12, N13, N16, N18)
  - These values are illustrative anchors for the constructive proof, not 
    empirical calibrations — per JP "remember these are just estimates based 
    on the purpose of the thesis as a constructive proof of concept"

Node remapping table (CURRENT → CH5 CANONICAL):
  N1  Burden of proof                  → N1  (same)
  N2  Violent history                  → N2  (Validated Risk Elevators)
  N3  PCL-R psychopathy                → N3  (folds into Sexual Offence Risk Profile)
  N4  Static-99R                       → N3  (also folds into Sexual Offence Risk Profile)
  N5  Invalid risk tools               → N5  (same)
  N6  IAC                              → N6  (same)
  N7  Bail-WCGP cascade                → N7  (same)
  N8  King impeachment                 → N7i (becomes sub-node, kept as separate node N8 for now)
  N9  FASD                             → N8  (CH5 §5.1.8)
  N10 Intergenerational trauma         → N9  (consolidated into IGT & Cultural Treatment)
  N11 No cultural treatment            → N9  (folds into N9 in CH5)
  N12 Gladue misapplication            → N10 (Judicial Misapplication of SCE)
  N13 Gaming risk                      → N11 (Gaming Risk Detector)
  N14 Over-policing                    → N17 (Over-Policing and Epistemic Contamination)
  N15 Temporal distortion              → N14 (Temporal Distortion)
  N16 Tariff disparities               → N15 (Interjurisdictional Tariff Distortion)
  N17 Collider bias                    → N19 (Collider Bias)
  N18 Dynamic risk                     → N4  (Dynamic Risk Factor Cluster)
  N19 No rehabilitation                → N9  (folds into N9 in CH5)
  N20 DO designation                   → N20 (same)

NEW nodes in CH5 not in current architecture:
  N12 Judicial Reasoning Reliability (Judging the Judge)
  N13 Structural Systemic Bias (TraceRoute)
  N16 Doctrinal Tension (s.718.04 vs s.718.2(e))
  N18 Gladue/Ewert/Morris/Ellis Profile audit

Note on consolidation: current N10 (IGT), N11 (treatment), N19 (rehab) all 
map to CH5 N9. Since the BN structure can't have three nodes mapping to one,
the rebuild creates ONE N9 in the new structure that represents the 
consolidated concept. The CPT for N9 is taken from current N10's calibration
since IGT is conceptually primary; treatment-availability becomes part of 
the N9 specification rather than a separate node.

The current N3 (PCL-R) and N4 (Static-99R) similarly both map to CH5 N3
(Sexual Offence Risk Profile). The rebuild treats CH5 N3 as the consolidated
sexual-offence-profile node and uses current N4's CPT (since Static-99R is 
the more directly sexual-offence-specific instrument). PCL-R is then 
implicit within N3.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Edges in the canonical Chapter 5 taxonomy
# ═════════════════════════════════════════════════════════════════════════════
#
# Parent → Child relationships derived from CH5 §5.1.X specifications.
# Topology must be a DAG (no cycles).
#
# Structural notes:
#   N1 (Burden of proof) is upstream conditioner of evidentiary nodes
#   N7 → N2 cascade preserved (bail-WCGP affects record reliability)
#   N20 (DO designation) is downstream of all distortion + risk nodes
#       but computed POST-VE (not in pgmpy network)
EDGES_VE = [(f, t) for f, t in [
    # Layer I — Substantive Risk
    (1, 2), (1, 3), (1, 4),
    # Layer II — Distortion conditioning
    (1, 6), (1, 8),                 # N1 conditions IAC and FASD evidentiary thresholds
    (7, 2),                         # bail-WCGP affects violent-history reliability (preserved)
    (2, 5), (3, 5), (4, 5),        # validated risk inputs feed risk-tool node
    (2, 14),                        # violent history feeds temporal distortion
    (6, 7),                         # IAC → bail cascade
    (6, 10),                        # IAC contributes to judicial misapplication
    (8, 9),                         # FASD → IGT/cultural-treatment
    (9, 10),                        # IGT → judicial misapplication
    (5, 10),                        # invalid risk tools → judicial misapplication
    (10, 12),                       # judicial misapplication → judging the judge
    (10, 18),                       # judicial misapplication → SCE Profile audit
    (9, 18),                        # IGT → SCE Profile audit
    (3, 11),                        # sexual offence risk profile → gaming detector
    (13, 17),                       # TraceRoute → over-policing
    (17, 19),                       # over-policing → collider bias
    (14, 19),                       # temporal distortion → collider bias  
    (10, 16),                       # judicial misapplication → doctrinal tension
    (10, 15),                       # SCE misapp → tariff distortion
    (14, 18),                       # temporal distortion → SCE Profile audit
    (12, 18),                       # judging-the-judge → SCE Profile audit
]]


# ═════════════════════════════════════════════════════════════════════════════
# Node metadata — CH5 (April 11, 2026) canonical taxonomy
# ═════════════════════════════════════════════════════════════════════════════
NODE_META = {
    # ── Substantive Risk Layer (CH5 i) ────────────────────────────────────────
    1:  {"name": "Criminal Law Burden of Proof",                  "short": "Burden of proof",       "type": "constraint", "ev": False},
    2:  {"name": "Validated Risk Elevators",                      "short": "Risk elevators",        "type": "risk",       "ev": True},
    3:  {"name": "Sexual Offence Risk Profile",                   "short": "Sexual offence",        "type": "risk",       "ev": True},
    4:  {"name": "Dynamic Risk Factor Cluster",                   "short": "Dynamic risk",          "type": "risk",       "ev": True},
    # ── Systemic Distortion and Doctrinal Fidelity Layer (CH5 ii) ────────────
    5:  {"name": "Current Risk Assessment Tools",                 "short": "Risk tools (Ewert)",    "type": "distortion", "ev": True},
    6:  {"name": "Ineffective Assistance of Counsel",             "short": "IAC",                   "type": "distortion", "ev": True},
    7:  {"name": "Bail Denial → Wrongful Conviction Guilty Plea", "short": "Bail-WCGP cascade",     "type": "distortion", "ev": True},
    8:  {"name": "FASD as Dual-Factor in Risk Modeling",          "short": "FASD",                  "type": "dual",       "ev": True},
    9:  {"name": "Intergenerational Trauma & Cultural Treatment", "short": "IGT / treatment",       "type": "mitigation", "ev": True},
    10: {"name": "Judicial Misapplication of SCE",                "short": "SCE misapplication",    "type": "distortion", "ev": True},
    11: {"name": "Gaming Risk Detector",                          "short": "Gaming risk",           "type": "special",    "ev": True},
    12: {"name": "Judicial Reasoning Reliability",                "short": "Judging the Judge",     "type": "distortion", "ev": False},
    13: {"name": "Structural Systemic Bias (TraceRoute)",         "short": "TraceRoute",            "type": "distortion", "ev": False},
    14: {"name": "Temporal Distortion in Prior Records",          "short": "Temporal distortion",   "type": "distortion", "ev": True},
    15: {"name": "Interjurisdictional Tariff Distortion",         "short": "Tariff distortion",     "type": "distortion", "ev": False},
    16: {"name": "Doctrinal Tension (s.718.04 / s.718.2(e))",     "short": "Doctrinal tension",     "type": "distortion", "ev": False},
    17: {"name": "Over-Policing & Epistemic Contamination",       "short": "Over-policing",         "type": "distortion", "ev": True},
    18: {"name": "Gladue / Ewert / Morris / Ellis Profile",       "short": "SCE Profile audit",     "type": "distortion", "ev": False},
    19: {"name": "Collider Bias",                                 "short": "Collider bias",         "type": "distortion", "ev": False},
    # ── Structural Output (CH5 iii) ──────────────────────────────────────────
    20: {"name": "Dangerous Offender Designation",                "short": "DO designation",        "type": "output",     "ev": False},
}


# Backward-compat alias for any code expecting EDGES (not EDGES_VE)
EDGES = EDGES_VE


def _cpt(var, parents, table):
    """Helper: build TabularCPD. table = [P(Low|...), P(High|...)] columns."""
    n_parents = len(parents)
    ev_card = [2] * n_parents
    return TabularCPD(
        variable=var,
        variable_card=2,
        values=table,
        evidence=parents if parents else None,
        evidence_card=ev_card if ev_card else None,
    )


def _noisy_or(var, parents, leak, inhibitors):
    """
    Noisy-OR CPT for a node with many parents.
    P(High) = 1 - leak * prod(inhibitors[i] if parent_i=High else 1)
    """
    n = len(parents)
    n_configs = 2 ** n
    p_high = np.zeros(n_configs)

    for config in range(n_configs):
        bits = [(config >> i) & 1 for i in range(n)]
        q = leak
        for i, active in enumerate(bits):
            if active:
                q *= inhibitors[i]
        p_high[config] = 1.0 - q

    values = [1 - p_high, p_high]
    return TabularCPD(
        variable=var,
        variable_card=2,
        values=values,
        evidence=parents,
        evidence_card=[2] * n,
    )


def build_model():
    """Build and return the PARVIS BayesianNetwork with all CPTs.
    Node 20 is excluded from the pgmpy network — it is computed post-VE.
    
    CPT calibrations are illustrative anchors per the constructive-proof
    nature of the thesis. Where a current-architecture CPT has an obvious
    canonical match, calibration is preserved. New nodes (N12, N13, N16, N18)
    receive conservative defaults reflecting the doctrinal posture in CH5.
    """
    model = BayesianNetwork([(str(f), str(t)) for f, t in EDGES_VE])

    # ── N1: Criminal Law Burden of Proof (root) ──────────────────────────────
    # CH5 §5.1.1 — evidentiary admissibility constraint at BRD level (~83%)
    cpd1 = TabularCPD(variable='1', variable_card=2, values=[[0.17], [0.83]])

    # ── N2: Validated Risk Elevators (parents: N1, N7) ───────────────────────
    # CPT preserved from current N2 (was Violent history)
    # Parent topology (N1, N7) preserved — bail-WCGP cascade discounts record
    cpd2 = _cpt('2', ['1', '7'], [
        [0.50, 0.55, 0.35, 0.55],
        [0.50, 0.45, 0.65, 0.45],
    ])

    # ── N3: Sexual Offence Risk Profile (parent: N1) ─────────────────────────
    # Consolidates current N3 (PCL-R) + N4 (Static-99R) per CH5 §5.1.3
    # CPT preserved from current N4 (Static-99R) — more directly applicable
    cpd3 = _cpt('3', ['1'], [
        [0.80, 0.65],
        [0.20, 0.35],
    ])

    # ── N4: Dynamic Risk Factor Cluster (parent: N1) ─────────────────────────
    # Maps from current N18 (Dynamic risk) per CH5 §5.1.4
    # Preserved CPT structure — single parent (N1) for CH5 Layer I structure
    # Note: current N18 had three parents (N11, N13, N15) but in CH5 N4 is
    # an upstream substantive risk node, so the parent set is reduced to N1.
    cpd4 = _cpt('4', ['1'], [
        [0.55, 0.30],
        [0.45, 0.70],
    ])

    # ── N5: Current Risk Assessment Tools (parents: N2, N3, N4) ──────────────
    # CPT preserved from current N5 (parents same)
    cpd5 = _cpt('5', ['2', '3', '4'], [
        [0.35, 0.22, 0.25, 0.15, 0.28, 0.18, 0.20, 0.10],
        [0.65, 0.78, 0.75, 0.85, 0.72, 0.82, 0.80, 0.90],
    ])

    # ── N6: IAC (parent: N1) ─────────────────────────────────────────────────
    # CPT preserved from current N6
    cpd6 = _cpt('6', ['1'], [
        [0.60, 0.45],
        [0.40, 0.55],
    ])

    # ── N7: Bail Denial → WCGP Cascade (parent: N6) ──────────────────────────
    # CPT preserved from current N7
    cpd7 = _cpt('7', ['6'], [
        [0.55, 0.30],
        [0.45, 0.70],
    ])

    # ── N8: FASD as Dual-Factor (parent: N1) ─────────────────────────────────
    # Maps from current N9 (FASD), CPT preserved
    cpd8 = _cpt('8', ['1'], [
        [0.70, 0.60],
        [0.30, 0.40],
    ])

    # ── N9: IGT and Cultural Treatment (parent: N8) ──────────────────────────
    # Consolidates current N10 (IGT) + N11 (treatment) + N19 (rehab) per CH5 §5.1.9
    # CPT preserved from current N10 (IGT was conceptually primary)
    cpd9 = _cpt('9', ['8'], [
        [0.35, 0.15],
        [0.65, 0.85],
    ])

    # ── N10: Judicial Misapplication of SCE (parents: N5, N6, N9) ────────────
    # Maps from current N12 (Gladue misapplication) per CH5 §5.1.10
    # Parent set adjusted for canonical taxonomy: invalid tools (N5),
    # ineffective counsel (N6), and IGT/treatment (N9) all condition
    # whether SCE was meaningfully applied. Three-parent CPT.
    # 8 combinations: (5,6,9) = (L,L,L),(H,L,L),(L,H,L),(H,H,L),(L,L,H),(H,L,H),(L,H,H),(H,H,H)
    # Calibration derived from current 4-parent N12 CPT, marginalised over the
    # dropped parents conservatively.
    cpd10 = _cpt('10', ['5', '6', '9'], [
        [0.55, 0.40, 0.42, 0.30, 0.40, 0.25, 0.30, 0.18],
        [0.45, 0.60, 0.58, 0.70, 0.60, 0.75, 0.70, 0.82],
    ])

    # ── N11: Gaming Risk Detector (parent: N3) ───────────────────────────────
    # Maps from current N13 (Gaming risk detector), CPT preserved
    cpd11 = _cpt('11', ['3'], [
        [0.82, 0.65],
        [0.18, 0.35],
    ])

    # ── N12: Judicial Reasoning Reliability (parent: N10) ────────────────────
    # NEW node per CH5 §5.1.12 — "Judging the Judge"
    # Conservative default: when SCE has been misapplied (N10 High), 
    # judicial reasoning reliability is reduced (P(High reasoning reliability) lower)
    # Note: "High" state of N12 is interpreted as HIGH reliability concern,
    # consistent with state="High" = distortion/concern present elsewhere.
    cpd12 = _cpt('12', ['10'], [
        [0.65, 0.40],     # P(Low) — high reliability when SCE properly applied
        [0.35, 0.60],     # P(High) — reliability concern when SCE misapplied
    ])

    # ── N13: Structural Systemic Bias / TraceRoute (no parents) ──────────────
    # NEW node per CH5 §5.1.13 — root systemic-bias node
    # Conservative default reflects the empirical baseline of structural 
    # discrimination in the Canadian justice system per CH3 evidence base.
    cpd13 = TabularCPD(variable='13', variable_card=2, values=[[0.40], [0.60]])

    # ── N14: Temporal Distortion (parent: N2) ────────────────────────────────
    # Maps from current N15 (Temporal distortion), CPT preserved
    cpd14 = _cpt('14', ['2'], [
        [0.60, 0.40],
        [0.40, 0.60],
    ])

    # ── N15: Interjurisdictional Tariff Distortion (parent: N10) ─────────────
    # Maps from current N16 (Tariff disparities), CPT preserved
    # Per CH5 §5.1.15: tariff distortion is mediated by jurisprudential
    # compliance (proxied by N10 misapplication)
    cpd15 = _cpt('15', ['10'], [
        [0.62, 0.42],
        [0.38, 0.58],
    ])

    # ── N16: Doctrinal Tension (parent: N10) ─────────────────────────────────
    # NEW node per CH5 §5.1.16 — s.718.04 vs s.718.2(e) conflict
    # Conservative default: tension is salient where SCE is misapplied
    # in cases involving Indigenous victims/offenders.
    cpd16 = _cpt('16', ['10'], [
        [0.75, 0.50],     # P(Low tension) — properly reconciled when SCE applied
        [0.25, 0.50],     # P(High tension) — unresolved conflict where SCE misapplied
    ])

    # ── N17: Over-Policing & Epistemic Contamination (parent: N13) ───────────
    # Maps from current N14 (Over-policing), CPT structure preserved
    # Per CH5 §5.1.17: structural systemic bias (N13/TraceRoute) → over-policing
    # The current N14 had parents (N7, N8) — restructured to single parent N13
    # since over-policing in CH5 is conceptually downstream of structural bias.
    cpd17 = _cpt('17', ['13'], [
        [0.45, 0.25],     # marginalised from prior (7,8)→(N13) since structural
        [0.55, 0.75],     # bias drives over-policing in CH5 §5.1.17
    ])

    # ── N18: Gladue/Ewert/Morris/Ellis Profile (parents: N9, N10, N12, N14) ──
    # NEW node per CH5 §5.1.18 — SCE integration audit
    # Audit-style node: surfaces whether prior convictions substantively 
    # integrated SCE. Conservative default keyed to whether the conditioning
    # nodes (IGT, misapplication, judicial reasoning, temporal context) 
    # signal integration vs. omission.
    # 16 combinations: (9, 10, 12, 14) — too complex for explicit table
    # Use Noisy-OR with the four parents acting as inhibitors
    cpd18 = _noisy_or('18', ['9', '10', '12', '14'],
                       leak=0.85,           # base P(SCE-integration concern Low) when no parents High
                       inhibitors=[0.85, 0.70, 0.80, 0.85])
    # Above produces P(High) ranging from 0.15 (no concerns) to ~0.60 (all concerns)

    # ── N19: Collider Bias (parents: N14, N17) ───────────────────────────────
    # Maps from current N17 (Collider bias), CPT structure preserved
    cpd19 = _cpt('19', ['14', '17'], [
        [0.55, 0.40, 0.42, 0.28],
        [0.45, 0.60, 0.58, 0.72],
    ])

    # Add all CPDs (Node 20 excluded — computed post-VE)
    model.add_cpds(
        cpd1, cpd2, cpd3, cpd4, cpd5, cpd6, cpd7, cpd8,
        cpd9, cpd10, cpd11, cpd12, cpd13, cpd14, cpd15,
        cpd16, cpd17, cpd18, cpd19
    )

    assert model.check_model(), "Model CPDs are inconsistent — check tables."
    return model


def compute_do_risk(posteriors: dict) -> float:
    """
    Compute Node 20 (DO designation risk) post-VE using calibrated formula.

    Architecture per thesis (updated for CH5 canonical taxonomy):
    
    Step 1 — Record reliability multiplier (CH5 §5.1.7 + §RM.5/§RM.6):
      Where bail-denial cascade (N7) or IAC (N6) is High, violent history
      carries reduced evidentiary weight. record_reliability ∈ [0.40, 1.0]
    
    Step 2 — Tool validity (Ewert): N5 conditions weight on N3 risk-tool outputs
    
    Step 3 — Raw risk = weighted risk posteriors (N2, N3, N4) with discounts
    
    Step 4 — Distortion correction reduces DO risk: HIGH distortion REDUCES
      effective DO risk because it flags evidentiary contamination
    
    Step 5 — Age burnout multiplier from N14 (Temporal Distortion)
    
    References: Tolppanen Report (2018); Feeley (1979); R v Antic [2017] SCC 27
    """
    p = posteriors

    # Record reliability: bail-WCGP cascade + IAC reduce violent history weight
    record_reliability = float(np.clip(
        1.0 - 0.35 * p.get(7, 0.5) - 0.15 * p.get(6, 0.5),
        0.40, 1.0
    ))

    # Tool validity: N5 (invalid risk tools) discounts N3 (sexual offence profile)
    tool_validity = float(np.clip(
        1.0 - 0.45 * p.get(5, 0.5),
        0.30, 1.0
    ))

    # Raw risk: substantive risk nodes (N2, N3, N4) with appropriate discounts
    raw = (
        0.30 * p.get(2, 0.5) * record_reliability +    # N2: discounted by record reliability
        0.25 * p.get(3, 0.5) * tool_validity +         # N3: discounted by Ewert (N5)
        0.20 * p.get(4, 0.5) +                         # N4 dynamic risk
        0.25 * p.get(18, 0.5)                          # N18 SCE Profile audit (now distortion-side)
    )

    # Distortion: systemic-distortion-layer nodes downweight effective risk
    # Updated per CH5 canonical taxonomy
    dst = (
        0.18 * posteriors.get(5, 0.5) +    # N5  invalid risk tools
        0.12 * posteriors.get(6, 0.5) +    # N6  IAC
        0.08 * posteriors.get(7, 0.5) +    # N7  bail-WCGP cascade
        0.05 * posteriors.get(9, 0.5) +    # N9  IGT/treatment (mitigation)
        0.18 * posteriors.get(10, 0.5) +   # N10 SCE misapplication
        0.05 * posteriors.get(12, 0.5) +   # N12 judging-the-judge
        0.10 * posteriors.get(13, 0.5) +   # N13 TraceRoute
        0.06 * posteriors.get(14, 0.5) +   # N14 temporal distortion (linear part)
        0.04 * posteriors.get(15, 0.5) +   # N15 tariff distortion
        0.04 * posteriors.get(16, 0.5) +   # N16 doctrinal tension
        0.10 * posteriors.get(17, 0.5)     # N17 over-policing
        # N19 (collider bias) intentionally excluded — its effect is on
        # the inference structure itself, not directly on the DO posterior
    )

    # Age burnout multiplier — N14 (Temporal Distortion) per CH5 §5.1.14
    # Schedule preserved from previous calibration:
    #   N14 < 0.65 → 1.00 (no correction)
    #   N14 = 0.85 → 0.66 (34% attenuation)
    #   N14 = 0.97 → 0.46 (54% attenuation)
    n14 = posteriors.get(14, 0.5)
    burnout_mult = float(np.clip(1.0 - 1.70 * max(0.0, n14 - 0.65), 0.35, 1.0))
    raw = raw * burnout_mult

    return float(np.clip(raw * (1.0 - 0.68 * dst) + 0.03, 0.05, 0.93))


def get_inference_engine(model):
    """Return a Variable Elimination inference engine."""
    return VariableElimination(model)


def query_do_risk(engine, evidence: dict) -> dict:
    """
    Run Variable Elimination for Nodes 1-19, then compute Node 20 post-VE.
    evidence: dict of {node_id_str: 0|1} for observed nodes.
    Returns dict of {node_id: P(High)} for all 20 nodes.
    """
    results = {}
    ve_nodes = [str(i) for i in range(1, 20)]

    for node in ve_nodes:
        if node in evidence:
            results[int(node)] = float(evidence[node])
            continue
        try:
            q = engine.query(variables=[node], evidence=evidence, show_progress=False)
            results[int(node)] = float(q.values[1])
        except Exception:
            results[int(node)] = 0.5

    results[20] = compute_do_risk(results)
    return results


def get_default_priors() -> dict:
    """Return prior P(High) for each node (no evidence observed).
    
    CH5 canonical taxonomy. Values are illustrative anchors for the 
    constructive proof, not empirical calibrations.
    """
    return {
        1:  0.83,   # Burden of proof — BRD per Gardiner [1982] 2 SCR 368
        2:  0.65,   # Validated risk elevators
        3:  0.35,   # Sexual offence risk profile
        4:  0.55,   # Dynamic risk factor cluster
        5:  0.70,   # Current risk assessment tools
        6:  0.55,   # IAC
        7:  0.55,   # Bail-WCGP cascade
        8:  0.40,   # FASD
        9:  0.75,   # IGT and cultural treatment
        10: 0.60,   # SCE misapplication
        11: 0.25,   # Gaming risk detector
        12: 0.50,   # Judging the judge — judicial reasoning reliability
        13: 0.60,   # TraceRoute / Structural systemic bias
        14: 0.50,   # Temporal distortion
        15: 0.45,   # Tariff distortion
        16: 0.30,   # Doctrinal tension
        17: 0.65,   # Over-policing
        18: 0.40,   # SCE Profile audit
        19: 0.55,   # Collider bias
        20: 0.50,   # DO designation risk
    }
