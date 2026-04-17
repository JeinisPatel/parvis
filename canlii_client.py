"""
PARVIS — CanLII API Client
canlii_client.py

Queries the CanLII API (api.canlii.org) for recent Canadian decisions
relevant to the PARVIS node schema and Tetrad doctrine.

PURPOSE:
  Doctrine.py provides the structural anchor (static authoritative rules).
  This module surfaces NEW developments — recent decisions that may have
  moved or refined the doctrine since doctrine.py was last updated.

HOW TO GET A CANLII API KEY:
  1. Go to https://api.canlii.org
  2. Register for a free account (non-commercial use)
  3. Copy your API key
  4. Add to Streamlit secrets: CANLII_API_KEY = "your-key-here"
     OR set environment variable: export CANLII_API_KEY="your-key-here"

RATE LIMITS: CanLII free API — 100 requests/hour.
PARVIS caches results to avoid redundant calls.

AUTHORS: J.S. Patel | University of London | Ethical AI Initiative
"""

import os
import json
import time
import hashlib
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Optional

# ── CanLII API configuration ──────────────────────────────────────────────────
CANLII_BASE_URL = "https://api.canlii.org/v1"
CACHE_TTL_HOURS = 24  # Cache results for 24 hours


# ── Node-specific search queries ───────────────────────────────────────────────
# Each node maps to a set of search terms that surface relevant recent cases.
NODE_SEARCH_QUERIES = {
    2:  ["dangerous offender pattern behaviour", "s753 violent history pattern"],
    3:  ["PCL-R psychopathy sentencing", "Ewert psychopathy assessment cultural"],
    4:  ["Static-99R Indigenous sentencing", "Ewert Static-99 sexual offence"],
    5:  ["Ewert actuarial cultural validity sentencing", "risk assessment Indigenous cultural"],
    6:  ["ineffective counsel Gladue Indigenous", "GDB ineffective assistance sentencing"],
    7:  ["bail denial coercive plea Indigenous", "Antic wrongful guilty plea"],
    9:  ["FASD sentencing mitigation", "fetal alcohol spectrum disorder dangerous offender"],
    10: ["intergenerational trauma Gladue sentencing", "residential school sentencing Indigenous"],
    11: ["cultural programming unavailable sentencing", "Natomagan rehabilitation treatment Indigenous"],
    12: ["Gladue misapplication Morris sentencing", "social context evidence racialized sentencing"],
    13: ["rehabilitation gaming dangerous offender", "strategic rehabilitation sentencing"],
    14: ["over-policing criminal record sentencing", "Le racial profiling criminal record"],
    15: ["mandatory minimum temporal sentencing", "age burnout recidivism sentencing"],
    16: ["provincial tariff dangerous offender", "DO designation rate provincial"],
    17: ["collider bias incarceration sentencing", "structural disadvantage criminal record"],
    18: ["dynamic risk factors structural context", "Ipeelee dynamic risk systemic"],
    19: ["rehabilitation absence Natomagan", "no programming treatment refusal dangerous offender"],
    20: ["dangerous offender designation Gladue", "DO designation Morris Ellis Ewert"],
}

# ── Key Tetrad case citations for tracking subsequent history ──────────────────
TETRAD_CITATIONS = [
    {"id": "2021onca680",  "label": "R v Morris 2021 ONCA 680",       "db": "onca"},
    {"id": "2022bcca278",  "label": "R v Ellis 2022 BCCA 278",        "db": "bcca"},
    {"id": "2022abca48",   "label": "R v Natomagan 2022 ABCA 48",     "db": "abca"},
    {"id": "2024onca8",    "label": "R v Bourdon 2024 ONCA 8",        "db": "onca"},
    {"id": "2018scc30",    "label": "Ewert v Canada 2018 SCC 30",     "db": "scc"},
    {"id": "2017scc64",    "label": "R v Boutilier 2017 SCC 64",      "db": "scc"},
    {"id": "1999scc55",    "label": "R v Gladue 1999 1 SCR 688",      "db": "scc"},
    {"id": "2012scc13",    "label": "R v Ipeelee 2012 SCC 13",        "db": "scc"},
]

# ── Canadian court database IDs ────────────────────────────────────────────────
COURT_DATABASES = {
    "scc":   "csc-scc",
    "onca":  "onca",
    "bcca":  "bcca",
    "abca":  "abca",
    "mbca":  "mbca",
    "skca":  "skca",
    "qcca":  "qcca",
    "nsca":  "nsca",
}


def _get_api_key() -> Optional[str]:
    """Resolve CanLII API key from Streamlit secrets or environment."""
    try:
        if hasattr(st, "secrets") and "CANLII_API_KEY" in st.secrets:
            return st.secrets["CANLII_API_KEY"]
    except Exception:
        pass
    return os.environ.get("CANLII_API_KEY")


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


@st.cache_data(ttl=3600 * CACHE_TTL_HOURS, show_spinner=False)
def search_canlii(query: str, language: str = "en", results_per_page: int = 5) -> dict:
    """
    Search CanLII for cases matching query.
    Results cached for CACHE_TTL_HOURS.

    Returns dict with keys: results, total_count, error
    """
    api_key = _get_api_key()
    if not api_key:
        return {"results": [], "total_count": 0,
                "error": "No CanLII API key configured. Add CANLII_API_KEY to Streamlit secrets."}

    url = f"{CANLII_BASE_URL}/caseBrowse/{language}/"
    params = {
        "api_key": api_key,
        "fullText": query,
        "resultCount": results_per_page,
        "offset": 0,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "results": data.get("cases", []),
            "total_count": data.get("totalCount", 0),
            "error": None,
        }
    except requests.Timeout:
        return {"results": [], "total_count": 0, "error": "CanLII API timeout"}
    except requests.HTTPError as e:
        return {"results": [], "total_count": 0, "error": f"CanLII API error: {e}"}
    except Exception as e:
        return {"results": [], "total_count": 0, "error": str(e)}


@st.cache_data(ttl=3600 * CACHE_TTL_HOURS, show_spinner=False)
def get_case_text(database_id: str, case_id: str) -> dict:
    """
    Retrieve full text of a specific CanLII decision.
    Returns dict with keys: content, title, citation, date, error
    """
    api_key = _get_api_key()
    if not api_key:
        return {"content": None, "error": "No CanLII API key"}

    url = f"{CANLII_BASE_URL}/caseBrowse/en/{database_id}/{case_id}/"
    params = {"api_key": api_key}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data.get("content", ""),
            "title":   data.get("title", ""),
            "citation": data.get("citation", ""),
            "date":    data.get("decisionDate", ""),
            "url":     data.get("url", ""),
            "error":   None,
        }
    except Exception as e:
        return {"content": None, "error": str(e)}


@st.cache_data(ttl=3600 * CACHE_TTL_HOURS, show_spinner=False)
def get_citing_cases(database_id: str, case_id: str, results: int = 10) -> dict:
    """
    Get cases that cite a specific decision.
    Useful for tracking subsequent history of Tetrad decisions.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"cases": [], "error": "No CanLII API key"}

    url = f"{CANLII_BASE_URL}/caseCitator/en/{database_id}/{case_id}/citingCases"
    params = {"api_key": api_key, "resultCount": results}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {"cases": data.get("citingCases", []), "error": None}
    except Exception as e:
        return {"cases": [], "error": str(e)}


def search_node_developments(node_id: int, max_results: int = 5) -> list:
    """
    Search CanLII for recent decisions relevant to a specific PARVIS node.
    Returns list of formatted result dicts.
    """
    queries = NODE_SEARCH_QUERIES.get(node_id, [])
    if not queries:
        return []

    all_results = []
    seen_ids = set()

    for query in queries[:2]:  # Use first 2 queries to stay within rate limits
        data = search_canlii(query, results_per_page=max_results)
        if data.get("error"):
            continue
        for case in data.get("results", []):
            cid = case.get("caseId", {}).get("en", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_results.append({
                    "title":    case.get("title", ""),
                    "citation": case.get("citation", ""),
                    "date":     case.get("decisionDate", ""),
                    "url":      case.get("url", ""),
                    "database": case.get("databaseId", ""),
                    "case_id":  cid,
                })

    # Sort by date descending
    all_results.sort(key=lambda x: x.get("date", ""), reverse=True)
    return all_results[:max_results]


def get_tetrad_updates(since_year: int = 2023) -> dict:
    """
    Check for recent citing cases for all Tetrad decisions.
    Returns {citation_label: [citing_cases]} for cases since since_year.
    """
    updates = {}
    for case in TETRAD_CITATIONS:
        citing = get_citing_cases(
            database_id=COURT_DATABASES.get(case["db"], case["db"]),
            case_id=case["id"],
            results=10,
        )
        if citing.get("error"):
            continue
        recent = [
            c for c in citing.get("cases", [])
            if c.get("decisionDate", "")[:4] >= str(since_year)
        ]
        if recent:
            updates[case["label"]] = recent
    return updates


def format_canlii_results(results: list, node_name: str = "") -> str:
    """Format CanLII search results as readable text."""
    if not results:
        return "No recent CanLII decisions found."
    header = f"Recent CanLII decisions relevant to {node_name}:\n" if node_name else "Recent decisions:\n"
    lines = [header]
    for r in results:
        date = r.get("date", "")[:10] if r.get("date") else "Date unknown"
        lines.append(f"  [{date}] {r.get('citation', r.get('title', '—'))}")
        if r.get("url"):
            lines.append(f"    {r['url']}")
    return "\n".join(lines)


def is_configured() -> bool:
    """Check if CanLII API key is available."""
    return bool(_get_api_key())
