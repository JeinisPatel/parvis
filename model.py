"""
PARVIS — Bayesian Network Model
Probabilistic and Analytical Reasoning Virtual Intelligence System
J.S. Patel | University of London (QMUL & LSE) | Ethical AI Initiative

Implements a proper pgmpy Bayesian Network with Variable Elimination.
CPTs encode normative priors grounded in doctrine (Gladue, Morris, Ellis, Ewert)
— not empirical frequencies. Architecture: subjective/robust Bayesian tradition.

All nodes are binary: state 0 = Low, state 1 = High.
For Node 20 (9 parents), Noisy-OR is used — standard for large parent sets.
"""

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

# Node 20 is computed post-VE via calibrated formula (not included in pgmpy network).
# Reason: distortion nodes REDUCE effective DO risk — Noisy-OR cannot model this
# because it only increases probability with each High parent.
# The formula below correctly encodes the thesis architecture:
#   DO risk = f(risk factors) × (1 - distortion correction weight)
EDGES_VE = [(f, t) for f, t in [
    (1, 2), (1, 3), (1, 4), (1, 6), (1, 9),
    # Edge (7, 2): bail-denial cascade is a parent of violent history.
    # Ch.3 (thesis): coercive guilty pleas under bail denial produce criminal records
    # that overstate violent propensity. Node 2 is conditioned on Node 7 to encode
    # that the violent history's evidentiary reliability depends on the procedural
    # integrity under which those convictions were entered.
    # Topological order: 1 → 6 → 7 → 2 (no cycle — 2 does not lead back to 1/6/7).
    (7, 2),
    (2, 5), (2, 15),
    (3, 5), (3, 13),
    (4, 5),
    (5, 12),
    (6, 7), (6, 12),
    (7, 8), (7, 14),
    (8, 14),
    (9, 10),
    (10, 11), (10, 12),
    (11, 12), (11, 18), (11, 19),
    (12, 16),
    (13, 18),
    (14, 17),
    (15, 17), (15, 18),
    (18, 19),
]]

# ── Node metadata ─────────────────────────────────────────────────────────────
NODE_META = {
    1:  {"name": "Criminal law burden of proof",           "short": "Burden of proof",        "type": "constraint", "ev": False},
    2:  {"name": "Serious violence / violent history",     "short": "Violent history",         "type": "risk",       "ev": True},
    3:  {"name": "Validated psychopathy (PCL-R)",          "short": "Psychopathy (PCL-R)",     "type": "risk",       "ev": True},
    4:  {"name": "Sexual offence profile (Static-99R)",    "short": "Sexual offence",          "type": "risk",       "ev": True},
    5:  {"name": "Culturally invalid risk tools",          "short": "Invalid risk tools",      "type": "distortion", "ev": True},
    6:  {"name": "Ineffective assistance of counsel",      "short": "Ineffective counsel",     "type": "distortion", "ev": True},
    7:  {"name": "Bail-denial → wrongful guilty plea",     "short": "Bail-denial cascade",     "type": "distortion", "ev": True},
    8:  {"name": "King credibility impeachment",           "short": "King impeachment",        "type": "distortion", "ev": False},
    9:  {"name": "FASD — dual factor node",                "short": "FASD",                    "type": "dual",       "ev": True},
    10: {"name": "Intergenerational trauma",               "short": "Intergenerational trauma","type": "mitigation", "ev": True},
    11: {"name": "Absence of culturally grounded treatment","short": "No cultural treatment",  "type": "distortion", "ev": True},
    12: {"name": "Judicial misapplication of Gladue tetrad","short": "Gladue misapplication",  "type": "distortion", "ev": True},
    13: {"name": "Gaming risk detector",                   "short": "Gaming risk",             "type": "special",    "ev": True},
    14: {"name": "Over-policing & epistemic contamination","short": "Over-policing",           "type": "distortion", "ev": True},
    15: {"name": "Temporal distortion",                    "short": "Temporal distortion",     "type": "distortion", "ev": True},
    16: {"name": "Interjurisdictional tariff effects",     "short": "Tariff disparities",      "type": "distortion", "ev": False},
    17: {"name": "Collider bias",                          "short": "Collider bias",           "type": "distortion", "ev": False},
    18: {"name": "Dynamic risk factors",                   "short": "Dynamic risk",            "type": "risk",       "ev": True},
    19: {"name": "Absence of rehabilitative progress",     "short": "No rehabilitation",       "type": "distortion", "ev": False},
    20: {"name": "Dangerous offender designation risk",    "short": "DO designation risk",     "type": "output",     "ev": False},
}

EDGES = [
    (1, 2), (1, 3), (1, 4), (1, 6), (1, 9),
    (2, 5), (2, 15), (2, 20),
    (3, 5), (3, 13), (3, 20),
    (4, 5), (4, 20),
    (5, 12), (5, 20),
    (6, 7), (6, 12),
    (7, 8), (7, 14),
    (8, 14),
    (9, 10),
    (10, 11), (10, 12),
    (11, 12), (11, 18), (11, 19),
    (12, 16), (12, 20),
    (13, 18),
    (14, 17),
    (15, 17), (15, 18),
    (16, 20),
    (17, 20),
    (18, 19), (18, 20),
    (19, 20),
]


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
    leak: base probability of High even when all parents are Low
    inhibitors: list of P(NOT causing High | parent_i = High) for each parent
    Returns TabularCPD.
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
    """
    model = BayesianNetwork([(str(f), str(t)) for f, t in EDGES_VE])

    # ── Node 1: Burden of proof (root — evidentiary constraint) ───────────────
    # This is always "active" — the constraint always applies.
    # P(High=active) = 1.0 (deterministic root)
    cpd1 = TabularCPD(variable='1', variable_card=2, values=[[0.0], [1.0]])

    # ── Node 2: Violent history (parent: Node 1) ──────────────────────────────
    # Node 1 modulates threshold: when active (High), threshold enforced
    # P(High | 1=Low): shouldn't occur, but set conservatively
    # P(High | 1=High): reflects serious violence in record
    # Prior for serious violence: ~0.65 from thesis
    # But this encodes the evidence threshold structure
    # Node 2: Violent history — parents: Node 1 (burden) AND Node 7 (bail-denial cascade)
    # Ch.3 thesis: bail-denial + coercive plea = criminal record has reduced evidentiary
    # reliability as indicator of genuine violent propensity. CPT encodes this directly.
    #
    # Column order: (N1=0,N7=0), (N1=0,N7=1), (N1=1,N7=0), (N1=1,N7=1)
    #                 base          cascade        reliable       cascade+threshold
    #
    # Key: P(N2=High | N1=High, N7=Low)  = 0.65 — reliable contested conviction
    #      P(N2=High | N1=High, N7=High) = 0.45 — coercive plea; record inflated
    #      Tolppanen Report; Feeley (1979); thesis Ch.3 §3.4.1–3.4.5
    cpd2 = _cpt('2', ['1', '7'], [
        [0.50, 0.55, 0.35, 0.55],   # P(Low)
        [0.50, 0.45, 0.65, 0.45],   # P(High)
    ])

    # ── Node 3: Psychopathy PCL-R (parent: Node 1) ───────────────────────────
    # P(High | 1=High): ~0.45 prior (Larsen: adversarial allegiance effects)
    cpd3 = _cpt('3', ['1'], [
        [0.65, 0.55],
        [0.35, 0.45],
    ])

    # ── Node 4: Sexual offence / Static-99R (parent: Node 1) ─────────────────
    cpd4 = _cpt('4', ['1'], [
        [0.80, 0.65],
        [0.20, 0.35],
    ])

    # ── Node 5: Culturally invalid risk tools (parents: 2, 3, 4) ─────────────
    # High = distortion present. Rises with each validated risk indicator
    # applied without cultural calibration
    # 8 combinations: (2,3,4) = (L,L,L),(H,L,L),(L,H,L),(H,H,L),(L,L,H),(H,L,H),(L,H,H),(H,H,H)
    cpd5 = _cpt('5', ['2', '3', '4'], [
        [0.35, 0.22, 0.25, 0.15, 0.28, 0.18, 0.20, 0.10],  # P(Low)
        [0.65, 0.78, 0.75, 0.85, 0.72, 0.82, 0.80, 0.90],  # P(High)
    ])

    # ── Node 6: Ineffective counsel (parent: Node 1) ──────────────────────────
    cpd6 = _cpt('6', ['1'], [
        [0.60, 0.45],
        [0.40, 0.55],
    ])

    # ── Node 7: Bail-denial cascade (parent: Node 6) ─────────────────────────
    # Ineffective counsel → higher risk of coercive plea
    cpd7 = _cpt('7', ['6'], [
        [0.55, 0.30],
        [0.45, 0.70],
    ])

    # ── Node 8: King credibility impeachment (parent: Node 7) ────────────────
    cpd8 = _cpt('8', ['7'], [
        [0.60, 0.25],
        [0.40, 0.75],
    ])

    # ── Node 9: FASD dual factor (parent: Node 1) ─────────────────────────────
    cpd9 = _cpt('9', ['1'], [
        [0.70, 0.60],
        [0.30, 0.40],
    ])

    # ── Node 10: Intergenerational trauma (parent: Node 9) ───────────────────
    # FASD diagnosis increases recognition of intergenerational harm
    cpd10 = _cpt('10', ['9'], [
        [0.35, 0.15],
        [0.65, 0.85],
    ])

    # ── Node 11: No cultural treatment (parent: Node 10) ─────────────────────
    # Higher intergenerational trauma context → more likely programs absent
    cpd11 = _cpt('11', ['10'], [
        [0.40, 0.20],
        [0.60, 0.80],
    ])

    # ── Node 12: Gladue misapplication (parents: 5, 6, 10, 11) ───────────────
    # 16 combinations — key distortion node
    # Rises with each distortion input
    cpd12 = _cpt('12', ['5', '6', '10', '11'], [
        # P(Low): 16 cols = (5,6,10,11) all combos
        [0.55, 0.38, 0.42, 0.28, 0.40, 0.25, 0.32, 0.18,
         0.38, 0.22, 0.28, 0.15, 0.25, 0.12, 0.18, 0.08],
        # P(High)
        [0.45, 0.62, 0.58, 0.72, 0.60, 0.75, 0.68, 0.82,
         0.62, 0.78, 0.72, 0.85, 0.75, 0.88, 0.82, 0.92],
    ])

    # ── Node 13: Gaming risk detector (parent: Node 3) ────────────────────────
    # Psychopathy score elevates suspicion of gaming rehabilitation signals
    cpd13 = _cpt('13', ['3'], [
        [0.82, 0.65],
        [0.18, 0.35],
    ])

    # ── Node 14: Over-policing (parents: 7, 8) ────────────────────────────────
    cpd14 = _cpt('14', ['7', '8'], [
        [0.45, 0.28, 0.30, 0.15],
        [0.55, 0.72, 0.70, 0.85],
    ])

    # ── Node 15: Temporal distortion (parent: Node 2) ─────────────────────────
    # Longer violent history → more temporal distortion in actuarial scoring
    cpd15 = _cpt('15', ['2'], [
        [0.60, 0.40],
        [0.40, 0.60],
    ])

    # ── Node 16: Tariff disparities (parent: Node 12) ─────────────────────────
    # Gladue misapplication propagates into jurisdictional tariff effects
    cpd16 = _cpt('16', ['12'], [
        [0.62, 0.42],
        [0.38, 0.58],
    ])

    # ── Node 17: Collider bias (parents: 14, 15) ──────────────────────────────
    cpd17 = _cpt('17', ['14', '15'], [
        [0.55, 0.40, 0.42, 0.28],
        [0.45, 0.60, 0.58, 0.72],
    ])

    # ── Node 18: Dynamic risk (parents: 11, 13, 15) ───────────────────────────
    # 8 combinations
    cpd18 = _cpt('18', ['11', '13', '15'], [
        [0.55, 0.40, 0.42, 0.30, 0.38, 0.25, 0.28, 0.15],
        [0.45, 0.60, 0.58, 0.70, 0.62, 0.75, 0.72, 0.85],
    ])

    # ── Node 19: No rehabilitation (parents: 11, 18) ──────────────────────────
    cpd19 = _cpt('19', ['11', '18'], [
        [0.60, 0.38, 0.35, 0.18],
        [0.40, 0.62, 0.65, 0.82],
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

    Architecture per thesis (updated with record reliability correction):

    Step 1 — Record reliability multiplier (thesis Ch.3 §3.4.1–3.4.5):
      Where bail-denial cascade (Node 7) or ineffective counsel (Node 6) is High,
      the violent history in the criminal record carries reduced evidentiary weight
      as an indicator of genuine violent propensity. Coercive guilty pleas produce
      convictions that may overstate actual violence.
      record_reliability ∈ [0.40, 1.0] — applied only to Node 2 contribution.

    Step 2 — Raw risk = weighted risk posteriors, with Node 2 discounted by
      record_reliability. Remaining risk nodes (3, 4, 18) unaffected because
      PCL-R, Static-99R, and dynamic risk are not record-dependent in the same way.

    Step 3 — Distortion correction: HIGH distortion REDUCES DO risk because it
      flags evidentiary contamination. Operationalises thesis central argument.

    References: Tolppanen Report (2018); Feeley (1979) The Process Is the Punishment;
    thesis Ch.3; R v Antic [2017] SCC 27; s.493.2 Criminal Code.
    """
    p = posteriors

    # Record reliability: bail-denial cascade + ineffective counsel reduce the
    # evidentiary weight of violent history as indicator of genuine propensity
    # Node 7 = bail-denial cascade (weight 0.35)
    # Node 6 = ineffective counsel (weight 0.15) — partial effect mediated via N7 CPT
    record_reliability = float(np.clip(
        1.0 - 0.35 * p.get(7, 0.5) - 0.15 * p.get(6, 0.5),
        0.40, 1.0
    ))

    raw = (
        0.30 * p.get(2, 0.5) * record_reliability +   # Node 2 discounted by reliability
        0.25 * p.get(3, 0.5) +                        # PCL-R — independent
        0.20 * p.get(4, 0.5) +                        # Static-99R — independent
        0.25 * p.get(18, 0.5)                         # Dynamic risk — independent
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
    return float(np.clip(raw * (1.0 - 0.68 * dst) + 0.03, 0.05, 0.93))


def get_inference_engine(model):
    """Return a Variable Elimination inference engine."""
    return VariableElimination(model)


def query_do_risk(engine, evidence: dict) -> dict:
    """
    Run Variable Elimination for Nodes 1–19, then compute Node 20 post-VE.
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

    # Compute Node 20 post-VE using calibrated formula
    results[20] = compute_do_risk(results)
    return results


def get_default_priors() -> dict:
    """Return prior P(High) for each node (no evidence observed)."""
    return {
        1: 1.00,  # Burden of proof — always active
        2: 0.65,
        3: 0.45,
        4: 0.35,
        5: 0.70,
        6: 0.55,
        7: 0.55,
        8: 0.50,
        9: 0.40,
        10: 0.75,
        11: 0.70,
        12: 0.60,
        13: 0.25,
        14: 0.65,
        15: 0.50,
        16: 0.45,
        17: 0.55,
        18: 0.55,
        19: 0.45,
        20: 0.50,
    }
