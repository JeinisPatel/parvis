# PARVIS Counterfactual Audit — app.py integration patch

This document describes the minimal changes to `app.py` needed to integrate
the new `counterfactual_audit.py` module. There are four changes total. Each
is small and reversible.

## Change 1 — Add module import

**Location:** Line 61, after the existing `from bloch_sphere ...` import.

**Add:**
```python
from counterfactual_audit import render_counterfactual_audit_tab
```

The import is placed alongside the existing companion-module imports
(`quantum_diagnostics`, `bloch_sphere`) for consistency.

## Change 2 — Add the new tab label

**Location:** Lines 2741–2744. Current:
```python
TABS=st.tabs(["📋 Summary","🕸️ Architecture","📋 Profile","💬 Intake (Chat)",
              "📜 Criminal Record","🦅 Gladue","⚖️ SCE",
              "🔬 Risk & Distortions","📊 Inference","📂 Documents",
              "🔀 Scenarios","⚛️ Quantum","📄 Report"])
```

**Replace with:**
```python
TABS=st.tabs(["📋 Summary","🕸️ Architecture","📋 Profile","💬 Intake (Chat)",
              "📜 Criminal Record","🦅 Gladue","⚖️ SCE",
              "🔬 Risk & Distortions","📊 Inference",
              "🔍 Counterfactual Audit",
              "📂 Documents",
              "🔀 Scenarios","⚛️ Quantum","📄 Report"])
```

The new label "🔍 Counterfactual Audit" sits at index 9, between "📊 Inference"
(index 8) and "📂 Documents" (formerly index 9, now index 10).

The 🔍 magnifying-glass emoji signals "examine / audit" without overlapping
the existing tab emojis (📋 📜 🦅 ⚖️ 🔬 📊 📂 🔀 ⚛️ 📄 🕸️ 💬).

## Change 3 — Add the new tab body

**Location:** Just before the existing `with TABS[9]:` block (currently at
line 7135 — the Documents tab). Insert this new block:

```python
# ── T9: Counterfactual Audit (Mark 8) ─────────────────────────────────────────
# Per the Counterfactual Audit Panel Specification (May 2026):
# This tab examines how the current Bayesian assessment depends on its
# operative assumptions. It does not recommend an alternative outcome — it
# identifies the conditions under which the law would justify a different
# conclusion. Twelve doctrinally-anchored conditions span four clusters
# (Doctrinal Application, Risk Tool Validity, Record Integrity, Procedural
# Sequencing) — Tetrad-grounded; not a "Gladue-as-discount" tool.
# Reads st.session_state.posteriors as input; writes only to its own
# st.session_state.cf_audit slice — does not modify the baseline.
# ──────────────────────────────────────────────────────────────────────────────
with TABS[9]:
    render_counterfactual_audit_tab()
```

## Change 4 — Renumber existing tab indices

After inserting the new tab at index 9, the following lines need their tab
indices incremented by 1:

| Current line | Current index | New index | Tab name      |
| ------------ | ------------- | --------- | ------------- |
| 7135         | TABS[9]       | TABS[10]  | Documents     |
| 10652        | TABS[10]      | TABS[11]  | Scenarios     |
| 6219         | TABS[11]      | TABS[12]  | Quantum       |
| 10797        | TABS[12]      | TABS[13]  | Report        |

Each of these is a one-character change: `TABS[9]` → `TABS[10]`,
`TABS[10]` → `TABS[11]`, etc.

**Important:** the order matters when applying these renames in-place via
search-and-replace. Process them in **descending order** to avoid
collisions. That is:
1. First rename `with TABS[12]:` → `with TABS[13]:` (Report)
2. Then rename `with TABS[11]:` → `with TABS[12]:` (Quantum)
3. Then rename `with TABS[10]:` → `with TABS[11]:` (Scenarios)
4. Then rename `with TABS[9]:` → `with TABS[10]:` (Documents)

Doing them in ascending order would conflict — for example, renaming TABS[9]
to TABS[10] first would then collide with the subsequent rename of the
original TABS[10] to TABS[11].

## Validation checklist

After applying the patches, verify:

1. **Syntax check:** `python3 -c "import ast; ast.parse(open('app.py').read())"`
   should print nothing (no syntax errors).
2. **Tab count:** the `st.tabs([...])` call should now have exactly 14
   labels.
3. **TABS index uniqueness:** `grep -c "^with TABS\[" app.py` should equal
   14, and the indices 0–13 should each appear exactly once.
4. **Module import resolves:** `python3 -c "from counterfactual_audit import
   render_counterfactual_audit_tab; print('OK')"` from the deployment root
   should print `OK`.
5. **Streamlit local test:** `streamlit run app.py` and confirm:
   - The new "🔍 Counterfactual Audit" tab appears between Inference and
     Documents
   - Clicking the tab without prior inference shows the "Run inference
     first" notice
   - After running inference, the tab renders the header, baseline summary,
     conditions panel, revised assessment, and audit trail
   - Toggling a condition updates the revised assessment live
   - The persistent DO chip in the page header reflects the baseline (not
     the counterfactual) — this is the architectural correctness check

## Rollback

If the panel needs to be removed:

1. Revert the four changes above
2. Delete `counterfactual_audit.py`

No other module depends on the new code, so removal is clean.

## Deployment note

The codebase is deployed to Streamlit Cloud via GitHub auto-redeploy
(per DEPLOYMENT.txt). Once you commit and push these changes to `main`,
Streamlit Cloud will auto-rebuild and the new tab will be live.

Recommended commit message:
```
Add Counterfactual Audit tab (T9) per panel specification

- New module counterfactual_audit.py: 12 doctrinally-anchored conditions
  across 4 clusters (Doctrinal Application, Risk Tool Validity, Record
  Integrity, Procedural Sequencing)
- Tetrad-grounded magnitudes per condition (A1 Gladue 0.25, B2 PCL-R 0.10
  reflect doctrinal weight, not uniform treatment)
- Reads st.session_state.posteriors; writes only to st.session_state.cf_audit
- Does not modify baseline — counterfactual is purely additive analytical
  layer
- Tab inserted at index 9 between Inference and Documents; existing tabs
  renumbered accordingly
- Per spec doc parvis_counterfactual_panel_spec.md (May 2026)
```
