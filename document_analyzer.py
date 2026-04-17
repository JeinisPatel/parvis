"""
PARVIS — Document Analyzer
LLM-powered analysis of legal documents to inform Bayesian node weights.

This module uses the Anthropic API to analyze uploaded documents (Gladue reports,
IRCA reports, psychometric assessments, prior decisions, transcripts, bail records,
trauma assessments) and extracts structured evidence relevant to the 19 child nodes.

The LLM is prompted with the full Tetrad doctrinal framework as context, and returns
structured probability adjustments for each relevant node with doctrinal citations.

Architecture per Item 7:
1. User uploads document (PDF, DOCX, TXT)
2. Text is extracted and sent to Claude with Tetrad system prompt
3. Claude returns JSON: {node_id: {delta, confidence, citations, reasoning}}
4. User reviews and accepts/modifies suggested adjustments
5. Accepted adjustments feed into the Bayesian network as evidence

Important: The LLM provides guidance only. The user (judge, counsel, researcher)
makes the final determination on weight to be assigned.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
import anthropic

NODE_DESCRIPTIONS = {
    2:  "Serious violence / violent history — primary aggravating factor for DO designation",
    3:  "Psychopathy (PCL-R) — adversarial allegiance effects documented (Larsen 2024); cultural validity concerns (Ewert)",
    4:  "Sexual offence profile / Static-99R — cultural validity concerns for Indigenous offenders (Ewert v Canada [2018])",
    5:  "Culturally invalid risk tools — Static-99R, VRAG, LSI-R applied without cultural qualification",
    6:  "Ineffective assistance of counsel — failure to investigate Gladue/SCE factors",
    7:  "Bail-denial → wrongful guilty plea cascade — pre-trial detention creates coercive plea incentives",
    9:  "FASD — dual factor: mitigation reducing moral blameworthiness + treatment responsivity modulator",
    10: "Intergenerational trauma — residential school legacy, forced displacement, cultural genocide (Gladue/Ipeelee)",
    11: "Absence of culturally grounded treatment — systemic failure, not offender characteristic (Natomagan 2022 ABCA 48)",
    12: "Judicial misapplication of Gladue tetrad — failure to apply Gladue, Morris, Ellis, or Ewert",
    13: "Gaming risk detector — anomalously positive rehabilitation signals inconsistent with institutional record",
    14: "Over-policing / epistemic contamination — record inflated by disproportionate surveillance",
    15: "Temporal distortion — age-related burnout effect; prior convictions under repudiated mandatory minimums",
    18: "Dynamic risk factors — substance use, antisocial peers, housing instability (assess against structural context)",
}

SYSTEM_PROMPT = """You are a legal AI expert analyzing documents for the PARVIS Bayesian Sentencing Network.
PARVIS operationalises the Canadian sentencing Tetrad:
- R v Gladue [1999] 1 SCR 688: mandatory consideration of systemic/background factors for Indigenous offenders
- R v Ipeelee [2012] SCC 13: reaffirms Gladue; applies to all sentencing contexts
- R v Morris 2021 ONCA 680: SCE for Black/racialized offenders; para 97 connection gate (discernible nexus, not causation)
- R v Ellis 2022 BCCA 278: extends contextual reasoning to non-racialized socially disadvantaged offenders
- Ewert v Canada [2018] SCC 30: culturally invalid actuarial tools must not be applied without qualification
- R v Boutilier 2017 SCC 64: Gladue applies at all stages of DO proceedings including treatability
- R v Natomagan 2022 ABCA 48: absence of culturally appropriate programming cannot be weighed against accused

Your task: analyze the provided document and identify evidence relevant to each of the PARVIS network nodes.

For each relevant node, return:
1. delta: probability adjustment (positive = increases node's High probability, negative = decreases it)
   Range: -0.30 to +0.30 (moderate adjustments; the user makes final determination)
2. confidence: your confidence in this adjustment (0.0 to 1.0)
3. citations: specific passages or facts from the document supporting the adjustment
4. reasoning: doctrinal reasoning linking the document evidence to the node and binding authority
5. direction: "increases_risk" or "reduces_risk" or "distortion_present" or "distortion_absent"

CRITICAL INSTRUCTIONS:
- You are providing guidance to assist the user's assessment, NOT making a determination
- Flag where document evidence conflicts with actuarial scores (Ewert principle)
- Note where Gladue factors are present but were not engaged by prior decision-makers
- Identify Morris para 97 connection strength where applicable
- Flag FASD indicators even where undiagnosed (dual factor — mitigation AND risk modulation)
- Note temporal factors: when prior convictions were imposed and under what legal regime
- Be conservative: only flag nodes where the document contains clear, relevant evidence

Return ONLY valid JSON. No preamble, no explanation outside JSON."""

ANALYSIS_PROMPT_TEMPLATE = """Document type: {doc_type}
Document content:
---
{content}
---

Analyze this document and return a JSON object in this exact format:
{{
  "document_summary": "2-3 sentence summary of document type and key findings",
  "applicable_framework": "gladue" | "morris" | "ellis" | "all" | "none",
  "connection_assessment": "absent" | "weak" | "moderate" | "strong" | "direct",
  "nodes": {{
    "2":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "3":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "4":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "5":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "6":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "7":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "9":  {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "10": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "11": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "12": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "13": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "14": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "15": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}},
    "18": {{"delta": 0.0, "confidence": 0.0, "citations": [], "reasoning": "", "direction": ""}}
  }},
  "doctrinal_flags": [],
  "ewert_concern": false,
  "gladue_factors_present_but_unengaged": [],
  "morris_connection_note": ""
}}

Node descriptions for reference:
{node_desc}

Set delta=0.0 and confidence=0.0 for nodes where the document contains no relevant evidence.
Only assign non-zero deltas where the document contains clear, specific evidence."""


def extract_text_from_upload(uploaded_file) -> Tuple[str, str]:
    """
    Extract text from uploaded file.
    Returns (text_content, doc_type)
    """
    import io
    filename = uploaded_file.name.lower()
    doc_type = "Unknown document"

    if filename.endswith('.txt'):
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        doc_type = _infer_doc_type(content, filename)

    elif filename.endswith('.pdf'):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            content = '\n'.join(page.extract_text() or '' for page in reader.pages)
            doc_type = _infer_doc_type(content, filename)
        except ImportError:
            content = "[PDF extraction requires pypdf — text preview unavailable]"
            doc_type = "PDF document"

    elif filename.endswith('.docx'):
        try:
            import docx
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            content = '\n'.join(para.text for para in doc.paragraphs)
            doc_type = _infer_doc_type(content, filename)
        except ImportError:
            content = "[DOCX extraction requires python-docx — text preview unavailable]"
            doc_type = "Word document"

    else:
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        doc_type = _infer_doc_type(content, filename)

    return content[:15000], doc_type  # Limit to ~15k chars for API context


def _infer_doc_type(content: str, filename: str) -> str:
    """Infer document type from content and filename."""
    content_lower = content.lower()
    filename_lower = filename.lower()

    if any(k in content_lower for k in ['gladue', 'gladue report', 'indigenous background']):
        return "Gladue report"
    elif any(k in content_lower for k in ['irca', 'impact of race', 'anti-black', 'systemic racism']):
        return "IRCA (Impact of Race and Culture Assessment)"
    elif any(k in content_lower for k in ['pcl-r', 'psychopathy checklist', 'hare']):
        return "Psychometric assessment (PCL-R)"
    elif any(k in content_lower for k in ['static-99', 'static99', 'sexual recidivism']):
        return "Psychometric assessment (Static-99R)"
    elif any(k in content_lower for k in ['fetal alcohol', 'fasd', 'fas ', 'alcohol spectrum']):
        return "FASD assessment / diagnosis"
    elif any(k in content_lower for k in ['bail hearing', 'release order', 'detention order', 'show cause']):
        return "Bail hearing record"
    elif any(k in content_lower for k in ['ineffective assistance', 'solicitor-client', 'legal aid']):
        return "Ineffective assistance of counsel record"
    elif any(k in content_lower for k in ['transcript', 'examination', 'cross-examination', 'testimony']):
        return "Court transcript"
    elif any(k in content_lower for k in ['sentencing', 'sentence', 'conviction', 'crown']):
        return "Prior sentencing decision"
    elif any(k in content_lower for k in ['trauma', 'residential school', 'abuse', 'neglect']):
        return "Trauma / background assessment"
    else:
        return "Legal document"



def _get_api_key(api_key=None):
    """
    Resolve Anthropic API key in priority order:
    1. Explicitly passed argument
    2. Streamlit secrets (ANTHROPIC_API_KEY) — works on Streamlit Cloud
    3. Environment variable (ANTHROPIC_API_KEY) — works locally
    """
    if api_key:
        return api_key
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    import os
    return os.environ.get("ANTHROPIC_API_KEY")

def analyze_document(content: str, doc_type: str, api_key=None) -> dict:
    """
    Send document to Claude for Tetrad-grounded Bayesian node analysis.

    API key resolved automatically from:
      - Streamlit secrets (ANTHROPIC_API_KEY) when deployed on Streamlit Cloud
      - ANTHROPIC_API_KEY environment variable when running locally
      - api_key argument if explicitly passed

    Returns structured dict with per-node probability adjustments,
    doctrinal reasoning, and supporting text from the document.
    Each adjustment can be accepted or modified by the user before
    feeding into the Bayesian network.
    """
    try:
        resolved = _get_api_key(api_key)
        client = anthropic.Anthropic(api_key=resolved) if resolved else anthropic.Anthropic()

        node_desc_text = '\n'.join(
            f"  Node {nid}: {desc}"
            for nid, desc in NODE_DESCRIPTIONS.items()
        )

        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            doc_type=doc_type,
            content=content,
            node_desc=node_desc_text,
        )

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        # Strip any markdown fences if present
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse error: {e}",
            "document_summary": "Analysis failed — could not parse LLM response.",
            "nodes": {},
        }
    except Exception as e:
        return {
            "error": str(e),
            "document_summary": f"Analysis failed: {e}",
            "nodes": {},
        }


def format_analysis_for_display(analysis: dict) -> str:
    """Format analysis result as readable text for display."""
    if "error" in analysis:
        return f"Error: {analysis['error']}"

    lines = [
        f"Document type: {analysis.get('applicable_framework', 'Unknown framework').upper()}",
        f"Summary: {analysis.get('document_summary', '')}",
        f"Morris connection: {analysis.get('connection_assessment', 'not assessed')}",
        "",
        "Node adjustments:",
    ]

    for nid_str, node_data in analysis.get("nodes", {}).items():
        delta = node_data.get("delta", 0)
        conf = node_data.get("confidence", 0)
        if abs(delta) > 0.02 and conf > 0.1:
            direction = "↑" if delta > 0 else "↓"
            lines.append(
                f"  N{nid_str}: {direction} {abs(delta):.2f} "
                f"(confidence: {conf:.0%}) — {node_data.get('reasoning', '')[:100]}"
            )

    flags = analysis.get("doctrinal_flags", [])
    if flags:
        lines += ["", "Doctrinal flags:"] + [f"  ▸ {f}" for f in flags]

    if analysis.get("ewert_concern"):
        lines.append("  ⚠️  Ewert concern flagged — actuarial tool validity questioned")

    unengaged = analysis.get("gladue_factors_present_but_unengaged", [])
    if unengaged:
        lines += ["", "Gladue factors present but previously unengaged:"] + [f"  ▸ {f}" for f in unengaged]

    return '\n'.join(lines)
