# PARVIS — Bayesian Sentencing Network

**Probabilistic and Analytical Reasoning Virtual Intelligence System**  
J.S. Patel | University of London (QMUL & LSE) | Ethical AI Initiative

---

## Overview

PARVIS is a normatively constrained Bayesian network designed to audit sentencing reasoning for compliance with binding appellate authority in Canadian law. It operationalises the *Tetrad* — *Gladue* [1999], *Morris* 2021 ONCA 680, *Ellis* 2022 BCCA 278, and *Ewert* [2018] SCC 30 — as mandatory inference constraints within a 20-node Directed Acyclic Graph.

**This is a research prototype. Not for deployment in live proceedings.**

---

## Architecture

- **Inference engine:** pgmpy Variable Elimination (genuine Bayesian inference)
- **Nodes:** 20 nodes across three inferential layers
- **QBism diagnostic layer:** Appendix Q — epistemic audit of classical inference limits
- **Frameworks:** Gladue, Morris/Ellis SCE (with para 97 connection gate), Ewert

### Three Inferential Layers

| Layer | Nodes | Function |
|-------|-------|----------|
| I — Substantive risk | 1–4 | Burden of proof constraint + primary risk factors |
| II — Systemic distortion | 5–19 | Doctrinal fidelity corrections |
| III — Structural output | 20 | DO designation risk posterior |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/parvis.git
cd parvis
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment (Streamlit Cloud)

1. Push to GitHub (public or private repository)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository: `parvis` | Branch: `main` | Main file: `app.py`
5. Deploy

The app will be available at `https://YOUR_USERNAME-parvis.streamlit.app`

---

## File Structure

```
parvis/
├── app.py                  # Streamlit application (UI + orchestration)
├── model.py                # pgmpy Bayesian Network + Variable Elimination
├── quantum_diagnostics.py  # QBism diagnostic layer (Appendix Q)
├── requirements.txt        # Python dependencies
├── ethical_ai_logo.png     # Logo (watermark)
└── README.md
```

---

## Key Doctrinal References

- *R v Gladue* [1999] 1 SCR 688
- *R v Ipeelee* [2012] SCC 13
- *R v Morris* 2021 ONCA 680 (para 97 connection gate)
- *R v Ellis* 2022 BCCA 278
- *Ewert v Canada* [2018] SCC 30
- *R v Boutilier* 2017 SCC 64
- *R v Natomagan* 2022 ABCA 48

## Research References (QBism)

- Busemeyer & Bruza (2012). *Quantum Models of Cognition and Decision.* Cambridge.
- Wojciechowski (2023). *Quantum Probability Theory, Psychology and Law.* Routledge.
- Kochen & Specker (1967). The problem of hidden variables. *Journal of Mathematics and Mechanics.*
- Larsen et al. (2024). Psychopathy Evidence in Canadian Courts. *Psychology, Public Policy, and Law.*

---

## Research Use Only

PARVIS models **DO designation risk** — not intrinsic dangerousness. This distinction is the thesis's central normative contribution. The system audits whether legally mandated belief revision has occurred; it does not automate sentencing, predict recidivism, or replace judicial discretion.

© J.S. Patel, Ethical AI Initiative. Research prototype.
