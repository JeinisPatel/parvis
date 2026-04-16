"""
PARVIS — Quantum Bayesianism Diagnostic Module
Appendix Q: The Limits of Classical Bayesian Inference in Legally Distorted Systems

This module implements the QBism diagnostic layer described in Appendix Q.
It does NOT replace classical Bayesian inference (Variable Elimination).
It flags epistemic conditions under which classical probabilistic outputs
may require heightened scrutiny.

Per Appendix Q: "Quantum Bayesianism provides a principled vocabulary for
identifying those pressure points... its function is diagnostic."

Four diagnostic checks:
  1. Prior contamination — inherited distorted priors
  2. Order effects (non-commutativity) — evidentiary sequence matters
  3. Contextual interference — same evidence, different meaning per framework
  4. Premature scalar collapse — belief hardened before all SCE reviewed

References: Busemeyer & Bruza (2012), Wojciechowski (2023), Kochen & Specker (1967)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ── Diagnostic thresholds ─────────────────────────────────────────────────────
PRIOR_CONTAMINATION_THRESHOLD = 0.65  # Nodes above this are suspect if no evidence entered
ORDER_EFFECT_DELTA = 0.08             # Meaningful shift in posterior from reordering
BELIEF_STASIS_THRESHOLD = 0.05        # SCE entered but posterior shifts < this → belief stasis
COLLAPSE_RISK_NODES = {5, 12, 17}     # Nodes whose High state signals actuarial hardening


def diagnose(
    posteriors: Dict[int, float],
    evidence: Dict[str, int],
    gladue_checked: List[str],
    sce_checked: List[str],
    profile_ev: Dict[int, float],
    connection_strength: str = "moderate",
) -> Dict:
    """
    Run all four QBism diagnostics. Returns structured diagnostic report.

    Parameters
    ----------
    posteriors : node_id → P(High) from Variable Elimination
    evidence   : observed evidence fed into the BN {node_str: 0|1}
    gladue_checked : list of checked Gladue factor IDs
    sce_checked    : list of checked Morris/Ellis SCE factor IDs
    profile_ev     : case profile node evidence {node_id: probability}
    connection_strength : Morris para 97 connection gate value

    Returns
    -------
    dict with keys: prior_contamination, order_effects, contextual_interference,
                    belief_stasis, scalar_collapse_risk, overall_flag, summary
    """

    diags = {}

    # ── 1. Prior contamination ─────────────────────────────────────────────────
    # Nodes with elevated posteriors but no corresponding evidence entered
    # may reflect inherited distorted priors (criminal record, actuarial scores)
    contaminated = []
    distortion_nodes = {5, 6, 7, 8, 12, 14, 16, 17}
    for node_id in distortion_nodes:
        p = posteriors.get(node_id, 0.5)
        has_evidence = (str(node_id) in evidence) or (node_id in profile_ev)
        if p > PRIOR_CONTAMINATION_THRESHOLD and not has_evidence:
            contaminated.append({
                "node": node_id,
                "posterior": round(p, 3),
                "note": f"Node {node_id} elevated ({p:.1%}) without case-specific evidence — "
                        f"prior may reflect structurally contaminated baseline."
            })

    diags["prior_contamination"] = {
        "flagged": len(contaminated) > 0,
        "severity": "high" if len(contaminated) >= 3 else "moderate" if contaminated else "none",
        "items": contaminated,
        "doctrine": "Bayes' theorem does not correct distorted priors — it propagates them. "
                    "Per AQ.3.3.2: epistemic contamination at P(H) level is inherited, not neutralized.",
    }

    # ── 2. Order effects (non-commutativity) ──────────────────────────────────
    # Detect when criminal record / actuarial score evidence was entered BEFORE
    # Gladue / SCE / IRCA material — classical model treats this as commutative
    # but legal reasoning does not (AQ.3.3.3)
    risk_evidence_entered = any(str(n) in evidence for n in [2, 3, 4, 18])
    sce_entered = len(gladue_checked) > 0 or len(sce_checked) > 0
    report_commissioned = profile_ev.get(12, 0.60) < 0.50  # Lower = report commissioned

    order_flags = []
    if risk_evidence_entered and not sce_entered:
        order_flags.append(
            "Risk factor evidence entered without corresponding Gladue / Morris-Ellis SCE. "
            "Classical model treats this as order-neutral; legal reasoning does not. "
            "Posterior may anchor to risk-first narrative (AQ.3.3.3, M₁M₂ρ ≠ M₂M₁ρ)."
        )
    if risk_evidence_entered and sce_entered and not report_commissioned:
        order_flags.append(
            "SCE factors checked but no formal Gladue / IRCA report commissioned. "
            "Contextual evidence without procedural grounding may carry reduced weight. "
            "Order effect: actuarial framing likely precedes contextual revision."
        )

    diags["order_effects"] = {
        "flagged": len(order_flags) > 0,
        "severity": "high" if len(order_flags) >= 2 else "moderate" if order_flags else "none",
        "items": order_flags,
        "doctrine": "Non-commutativity (AQ.3.3.3): evidentiary sequence materially alters "
                    "posterior beliefs. Criminal record reviewed before Gladue ≠ Gladue reviewed first. "
                    "Classical Bayesian inference assumes order-independence — a legally questionable assumption.",
    }

    # ── 3. Contextual interference ────────────────────────────────────────────
    # P(H | C1) ≠ P(H | C2) where C1 = actuarial context, C2 = Gladue/Morris context
    # Flag when the DO posterior would shift meaningfully across interpretive contexts
    do_posterior = posteriors.get(20, 0.5)
    gladue_weight = len(gladue_checked) * 0.08      # rough proxy for SCE weight
    sce_weight = len(sce_checked) * 0.06

    # Simulated counterfactual: what would DO be under pure actuarial context?
    # (remove SCE corrections)
    do_actuarial_proxy = min(0.95, do_posterior + gladue_weight + sce_weight)
    do_contextual_proxy = do_posterior
    context_delta = abs(do_actuarial_proxy - do_contextual_proxy)

    context_flags = []
    if context_delta > ORDER_EFFECT_DELTA:
        context_flags.append({
            "actuarial_context": round(do_actuarial_proxy, 3),
            "gladue_morris_context": round(do_contextual_proxy, 3),
            "delta": round(context_delta, 3),
            "note": f"DO designation risk shifts {context_delta:.1%} across interpretive contexts. "
                    f"P(H|C_actuarial) = {do_actuarial_proxy:.1%} vs P(H|C_Gladue/Morris) = {do_contextual_proxy:.1%}. "
                    f"Contextuality (AQ.3.3.4): same evidence, different probative meaning."
        })

    connection_flag = None
    if connection_strength in ("none", "absent", "weak") and sce_checked:
        connection_flag = (
            f"Morris para 97 connection gate set to '{connection_strength}' while "
            f"{len(sce_checked)} SCE factors checked. Low connection weight suppresses "
            f"contextual correction — risk that contextual evidence is formally acknowledged "
            f"but substantively inert. Classic belief stasis pattern."
        )
        context_flags.append({"morris_gate": connection_flag})

    diags["contextual_interference"] = {
        "flagged": len(context_flags) > 0,
        "severity": "high" if context_delta > 0.15 else "moderate" if context_flags else "none",
        "items": context_flags,
        "doctrine": "Contextuality (AQ.3.3.4, Kochen-Specker): P(H|C₁) ≠ P(H|C₂). "
                    "Probative meaning of evidence is context-dependent. "
                    "Actuarial and Gladue/Morris frameworks may yield different risk assessments "
                    "for identical evidentiary records — not irrationality, but relational inference.",
    }

    # ── 4. Belief stasis detection ────────────────────────────────────────────
    # SCE entered but DO posterior remains high — suggests SCE is acknowledged
    # but not substantively integrating into belief revision
    stasis_flags = []
    if sce_entered and do_posterior > 0.65:
        stasis_flags.append(
            f"Substantial SCE entered ({len(gladue_checked)} Gladue + {len(sce_checked)} Morris/Ellis) "
            f"but DO designation risk remains {do_posterior:.1%}. "
            f"This may indicate belief stasis — SCE acknowledged but inert. "
            f"Per Kerr & Ewing (2023): courts cite Gladue without revising culpability. "
            f"Classical model may be propagating distorted priors faster than SCE corrections apply."
        )

    # Premature scalar collapse — actuarial nodes hardened to High
    collapsed_nodes = [
        n for n in COLLAPSE_RISK_NODES
        if posteriors.get(n, 0) > 0.80 and str(n) not in evidence
    ]
    if collapsed_nodes:
        stasis_flags.append(
            f"Nodes {collapsed_nodes} have collapsed to High (>80%) without direct evidence. "
            f"Actuarial recursion: distorted records feeding forward as risk signals (AQ.3.3.4). "
            f"Scalar resolution may have occurred prematurely — uncertainty foreclosed before "
            f"all SCE was reviewed."
        )

    diags["belief_stasis"] = {
        "flagged": len(stasis_flags) > 0,
        "severity": "high" if (sce_entered and do_posterior > 0.75) else "moderate" if stasis_flags else "none",
        "items": stasis_flags,
        "doctrine": "Belief stasis (AQ.3.3.4, Kerr & Ewing 2023): formally valid probabilistic "
                    "update may be normatively suspect where SCE is acknowledged but does not "
                    "alter the structure of reasoning. Premature scalar collapse (AQ.3.3.4): "
                    "point estimates may overstate epistemic certainty.",
    }

    # ── Superposition state indicator ─────────────────────────────────────────
    # Represents pre-decisional ambiguity — both risk and mitigation narratives live
    # Computed as closeness to 0.5 of DO posterior (maximally superposed = 0.5)
    superposition_index = 1.0 - abs(do_posterior - 0.5) * 2
    diags["superposition_index"] = round(superposition_index, 3)
    diags["superposition_note"] = (
        f"Belief state superposition index: {superposition_index:.2f} "
        f"(1.0 = maximally ambiguous, 0.0 = fully resolved). "
        f"{'High ambiguity — competing narratives remain live. Judicial commitment premature.' if superposition_index > 0.7 else 'Moderate ambiguity.' if superposition_index > 0.4 else 'Belief state largely resolved toward ' + ('High risk.' if do_posterior > 0.5 else 'Low risk.')}"
    )

    # ── Overall flag ───────────────────────────────────────────────────────────
    severity_map = {"high": 3, "moderate": 2, "none": 0}
    max_severity = max(
        severity_map.get(diags["prior_contamination"]["severity"], 0),
        severity_map.get(diags["order_effects"]["severity"], 0),
        severity_map.get(diags["contextual_interference"]["severity"], 0),
        severity_map.get(diags["belief_stasis"]["severity"], 0),
    )
    diags["overall_flag"] = {3: "high", 2: "moderate", 0: "none"}.get(max_severity, "none")

    # ── Summary ────────────────────────────────────────────────────────────────
    active = [k for k in ["prior_contamination", "order_effects", "contextual_interference", "belief_stasis"]
              if diags[k]["flagged"]]
    diags["summary"] = (
        f"QBism diagnostic: {len(active)} condition(s) flagged — {', '.join(active) if active else 'no conditions flagged'}. "
        f"Classical VE output is mathematically coherent. "
        + ("Heightened scrutiny warranted." if active else "No epistemic pressure points detected.")
    )

    return diags


def format_report(diags: Dict) -> str:
    """Format QBism diagnostics as audit text."""
    lines = [
        "QUANTUM BAYESIANISM (QBism) DIAGNOSTIC REPORT",
        "Appendix Q: The Limits of Classical Bayesian Inference",
        "──────────────────────────────────────────────────────────",
        "",
        f"Overall flag:           {diags['overall_flag'].upper()}",
        f"Superposition index:    {diags['superposition_index']} / 1.0",
        f"Summary:                {diags['summary']}",
        "",
    ]

    checks = [
        ("PRIOR CONTAMINATION", "prior_contamination"),
        ("ORDER EFFECTS (NON-COMMUTATIVITY)", "order_effects"),
        ("CONTEXTUAL INTERFERENCE", "contextual_interference"),
        ("BELIEF STASIS / SCALAR COLLAPSE", "belief_stasis"),
    ]

    for title, key in checks:
        d = diags[key]
        lines += [
            f"{title}",
            f"  Status:   {d['severity'].upper()}",
            f"  Flagged:  {'Yes' if d['flagged'] else 'No'}",
        ]
        if d["items"]:
            for item in d["items"]:
                if isinstance(item, str):
                    lines.append(f"  ▸ {item}")
                elif isinstance(item, dict):
                    for k, v in item.items():
                        lines.append(f"  ▸ {k}: {v}")
        lines += [f"  Doctrine: {d['doctrine']}", ""]

    lines += [
        "──────────────────────────────────────────────────────────",
        "NOTE: This diagnostic does not alter the VE posterior.",
        "It identifies conditions warranting heightened scrutiny.",
        "Per AQ.4: QBism operates as an epistemic audit mechanism,",
        "not a substitute inferential engine.",
    ]
    return "\n".join(lines)
