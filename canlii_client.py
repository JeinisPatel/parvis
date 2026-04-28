"""
PARVIS — CanLII API Client
canlii_client.py

Queries the CanLII API (api.canlii.org) for recent Canadian decisions
relevant to the PARVIS node schema and the Tetrad + Proportionality doctrinal
substrate.

PURPOSE:
  doctrine.py provides the structural anchor (static authoritative rules).
  This module surfaces NEW developments — recent decisions that may have
  moved or refined the doctrine since doctrine.py was last updated.

  The substrate tracked is symmetric across the fairness commitment:
    - Tetrad lineage (Gladue/Ipeelee/Morris/Ewert) — distortion-mitigation
    - Proportionality lineage (Lacasse/Friesen/Bissonnette/Sharma) — severity

  Results are tiered by stare decisis position relative to the user's
  jurisdiction:
    - Binding: Supreme Court of Canada + the user-jurisdiction's Court of Appeal
    - Persuasive: Other provincial Courts of Appeal
    - Other: Lower courts (trial / inferior)

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
# NOTE: These queries are not yet symmetry-audited. Per JP's instruction, the
# audit happens in a separate session against live API responses. The
# proportionality corpus (Lacasse/Friesen/Bissonnette/Sharma) is added as
# tracked citations below to close the most visible asymmetry tonight; the
# per-node query rewriting is deferred.
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

# ── Tetrad + downstream binding citations (distortion-mitigation lineage) ─────
TETRAD_CITATIONS = [
    {"id": "2021onca680",  "label": "R v Morris 2021 ONCA 680",       "db": "onca", "corpus": "Distortion"},
    {"id": "2022bcca278",  "label": "R v Ellis 2022 BCCA 278",        "db": "bcca", "corpus": "Distortion"},
    {"id": "2022abca48",   "label": "R v Natomagan 2022 ABCA 48",     "db": "abca", "corpus": "Distortion"},
    {"id": "2024onca8",    "label": "R v Bourdon 2024 ONCA 8",        "db": "onca", "corpus": "Distortion"},
    {"id": "2018scc30",    "label": "Ewert v Canada 2018 SCC 30",     "db": "scc",  "corpus": "Distortion"},
    {"id": "2017scc64",    "label": "R v Boutilier 2017 SCC 64",      "db": "scc",  "corpus": "Distortion"},
    {"id": "1999scc55",    "label": "R v Gladue 1999 1 SCR 688",      "db": "scc",  "corpus": "Distortion"},
    {"id": "2012scc13",    "label": "R v Ipeelee 2012 SCC 13",        "db": "scc",  "corpus": "Distortion"},
]

# ── Proportionality corpus (severity-tightening lineage) ──────────────────────
# Added per JP's "side-build" instruction (option ii in the Tier 1 design)
# to close the most visible asymmetry in the doctrinal substrate. These cases
# represent the line of authority that tightens proportionality / parity /
# limits on conditional sentence availability for serious offences. Their
# presence in the tracked corpus is what makes the architecture's "symmetric
# surface" claim accurate rather than aspirational.
PROPORTIONALITY_CITATIONS = [
    {"id": "2015scc64",  "label": "R v Lacasse 2015 SCC 64",      "db": "scc", "corpus": "Proportionality"},
    {"id": "2020scc9",   "label": "R v Friesen 2020 SCC 9",       "db": "scc", "corpus": "Proportionality"},
    {"id": "2022scc23",  "label": "R v Bissonnette 2022 SCC 23",  "db": "scc", "corpus": "Proportionality"},
    {"id": "2022scc39",  "label": "R v Sharma 2022 SCC 39",       "db": "scc", "corpus": "Proportionality"},
]

# Combined tracked-case set (Tetrad + Proportionality)
ALL_TRACKED_CITATIONS = TETRAD_CITATIONS + PROPORTIONALITY_CITATIONS

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
    "nbca":  "nbca",
    "peca":  "peca",
    "nlca":  "nlca",
    "nuca":  "nuca",
    "ntca":  "ntca",
    "ytca":  "yuca",
}

# ── Court tier classification (for binding/persuasive/other tagging) ──────────
COURT_TIER = {
    "csc-scc": "supreme",
    "onca": "appellate", "bcca": "appellate", "abca": "appellate",
    "mbca": "appellate", "skca": "appellate", "qcca": "appellate",
    "nsca": "appellate", "nbca": "appellate", "peca": "appellate",
    "nlca": "appellate", "nuca": "appellate", "ntca": "appellate",
    "yuca": "appellate",
}

# ── Jurisdiction → binding-court map ──────────────────────────────────────────
# For each province/territory, lists which appellate court is binding in
# addition to the SCC. SCC binds everywhere by definition.
JURISDICTION_APPELLATE_DB = {
    "ON": "onca", "BC": "bcca", "AB": "abca", "QC": "qcca",
    "SK": "skca", "MB": "mbca", "NS": "nsca", "NB": "nbca",
    "NL": "nlca", "PE": "peca", "YT": "yuca", "NT": "ntca", "NU": "nuca",
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


def _date_floor_to_iso(date_floor_label: str) -> Optional[str]:
    """
    Convert a UI date-floor label to an ISO date string.

    Accepts: "1 year", "3 years", "5 years", "All" (or None).
    Returns: YYYY-MM-DD string for the floor date, or None for "All".
    """
    if not date_floor_label or date_floor_label.lower() == "all":
        return None
    today = datetime.now()
    years = {"1 year": 1, "3 years": 3, "5 years": 5}.get(date_floor_label, 3)
    floor = today - timedelta(days=years * 365)
    return floor.strftime("%Y-%m-%d")


def _classify_tier(database_id: str, user_jurisdiction: str = "*") -> str:
    """
    Classify a result's stare-decisis tier relative to the user's jurisdiction.

    Returns one of: "binding", "persuasive", "other".
      binding    — SCC always; user-jurisdiction's CA when set
      persuasive — Other provincial CAs
      other      — Lower courts (trial/provincial), or unrecognised db
    """
    tier = COURT_TIER.get(database_id, "other")
    if tier == "supreme":
        return "binding"
    if tier == "appellate":
        # Is this the user-jurisdiction's CA?
        if user_jurisdiction and user_jurisdiction != "*":
            jur_db = JURISDICTION_APPELLATE_DB.get(user_jurisdiction)
            if jur_db and database_id == jur_db:
                return "binding"
        return "persuasive"
    return "other"


@st.cache_data(ttl=3600 * CACHE_TTL_HOURS, show_spinner=False)
def search_canlii(query: str,
                  language: str = "en",
                  results_per_page: int = 10,
                  decision_date_after: Optional[str] = None) -> dict:
    """
    Search CanLII for cases matching query.

    Args:
      query: full-text search string
      language: "en" or "fr"
      results_per_page: max results to return (1-20 reasonable; 100 hard max)
      decision_date_after: optional YYYY-MM-DD floor; the API filters server-side

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
    if decision_date_after:
        params["decisionDateAfter"] = decision_date_after

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
def get_citing_cases(database_id: str, case_id: str, results: int = 20) -> dict:
    """
    Get cases that cite a specific decision.
    Useful for tracking subsequent history of Tetrad/Proportionality decisions.
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


def _format_case(case: dict, user_jurisdiction: str = "*") -> dict:
    """
    Normalise a raw API case-dict into PARVIS's display format,
    with stare-decisis tier classification attached.
    """
    cid_raw = case.get("caseId", "")
    if isinstance(cid_raw, dict):
        cid = cid_raw.get("en", "") or cid_raw.get("fr", "")
    else:
        cid = str(cid_raw)
    db = case.get("databaseId", "")
    return {
        "title":    case.get("title", ""),
        "citation": case.get("citation", ""),
        "date":     case.get("decisionDate", ""),
        "url":      case.get("url", ""),
        "database": db,
        "case_id":  cid,
        "tier":     _classify_tier(db, user_jurisdiction),
    }


def search_with_filters(
    query: str,
    user_jurisdiction: str = "*",
    date_floor_label: str = "3 years",
    max_results: int = 12,
) -> dict:
    """
    Run a full-text query with jurisdictional + date filtering.

    Args:
      query: free-text query string
      user_jurisdiction: province/territory code ("ON", "BC", etc.) or "*"
      date_floor_label: "1 year" / "3 years" / "5 years" / "All"
      max_results: max items returned per tier

    Returns dict:
      {
        "binding":    [...],   # SCC + user-jurisdiction's CA
        "persuasive": [...],   # Other provincial CAs
        "other":      [...],   # Lower courts
        "total":      int,
        "error":      str or None,
      }
    """
    date_floor = _date_floor_to_iso(date_floor_label)
    # Pull a wider result set so the tiering has material to work with
    pull_count = max(20, max_results * 3)
    data = search_canlii(query=query,
                         results_per_page=pull_count,
                         decision_date_after=date_floor)
    if data.get("error"):
        return {"binding": [], "persuasive": [], "other": [],
                "total": 0, "error": data["error"]}

    binding, persuasive, other = [], [], []
    seen_ids = set()
    for raw in data.get("results", []):
        formatted = _format_case(raw, user_jurisdiction)
        cid = formatted.get("case_id", "")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        tier = formatted["tier"]
        if tier == "binding" and len(binding) < max_results:
            binding.append(formatted)
        elif tier == "persuasive" and len(persuasive) < max_results:
            persuasive.append(formatted)
        elif tier == "other" and len(other) < max_results:
            other.append(formatted)

    # Sort each tier by date descending
    for tier_list in (binding, persuasive, other):
        tier_list.sort(key=lambda x: x.get("date", ""), reverse=True)

    return {
        "binding": binding,
        "persuasive": persuasive,
        "other": other,
        "total": len(binding) + len(persuasive) + len(other),
        "error": None,
    }


def search_node_developments(node_id: int,
                             max_results: int = 5,
                             user_jurisdiction: str = "*",
                             date_floor_label: str = "3 years") -> dict:
    """
    Search CanLII for recent decisions relevant to a specific PARVIS node.

    Returns the same tiered structure as search_with_filters().

    BACKWARD-COMPATIBILITY NOTE: previously returned a flat list. Existing
    callers that consume the flat structure can use the helper
    `flatten_search_results()` below.
    """
    queries = NODE_SEARCH_QUERIES.get(node_id, [])
    if not queries:
        return {"binding": [], "persuasive": [], "other": [], "total": 0,
                "error": f"No search queries defined for node {node_id}"}

    date_floor = _date_floor_to_iso(date_floor_label)
    pull_count = max(20, max_results * 3)

    binding, persuasive, other = [], [], []
    seen_ids = set()

    # Use first 2 queries to stay within rate limits
    for query in queries[:2]:
        data = search_canlii(query=query,
                             results_per_page=pull_count,
                             decision_date_after=date_floor)
        if data.get("error"):
            continue
        for raw in data.get("results", []):
            formatted = _format_case(raw, user_jurisdiction)
            cid = formatted.get("case_id", "")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            tier = formatted["tier"]
            if tier == "binding" and len(binding) < max_results:
                binding.append(formatted)
            elif tier == "persuasive" and len(persuasive) < max_results:
                persuasive.append(formatted)
            elif tier == "other" and len(other) < max_results:
                other.append(formatted)

    for tier_list in (binding, persuasive, other):
        tier_list.sort(key=lambda x: x.get("date", ""), reverse=True)

    return {
        "binding": binding,
        "persuasive": persuasive,
        "other": other,
        "total": len(binding) + len(persuasive) + len(other),
        "error": None,
    }


def flatten_search_results(tiered: dict, max_total: int = 5) -> list:
    """
    Backward-compat helper: flatten a tiered result dict into a single list.
    Binding cases come first, then persuasive, then other; date-sorted within tier.
    """
    flat = list(tiered.get("binding", [])) + \
           list(tiered.get("persuasive", [])) + \
           list(tiered.get("other", []))
    return flat[:max_total]


def get_tracked_updates(date_floor_label: str = "3 years",
                        user_jurisdiction: str = "*",
                        corpus: str = "all") -> dict:
    """
    Check for recent citing cases for tracked decisions
    (Tetrad and/or Proportionality corpus).

    Args:
      date_floor_label: "1 year" / "3 years" / "5 years" / "All"
      user_jurisdiction: for tier classification of citing cases
      corpus: "all" / "Distortion" / "Proportionality" — which corpus to track

    Returns:
      {
        case_label: {
          "corpus": "Distortion" | "Proportionality",
          "binding":    [...citing cases...],
          "persuasive": [...citing cases...],
          "other":      [...citing cases...],
          "total":      int,
        },
        ...
      }
    """
    if corpus == "Distortion":
        cases = TETRAD_CITATIONS
    elif corpus == "Proportionality":
        cases = PROPORTIONALITY_CITATIONS
    else:
        cases = ALL_TRACKED_CITATIONS

    date_floor = _date_floor_to_iso(date_floor_label)
    updates = {}

    for case in cases:
        citing = get_citing_cases(
            database_id=COURT_DATABASES.get(case["db"], case["db"]),
            case_id=case["id"],
            results=20,
        )
        if citing.get("error"):
            continue

        binding, persuasive, other = [], [], []
        for c in citing.get("cases", []):
            cdate = c.get("decisionDate", "")
            if date_floor and cdate < date_floor:
                continue
            formatted = _format_case(c, user_jurisdiction)
            tier = formatted["tier"]
            if tier == "binding":
                binding.append(formatted)
            elif tier == "persuasive":
                persuasive.append(formatted)
            else:
                other.append(formatted)

        for tl in (binding, persuasive, other):
            tl.sort(key=lambda x: x.get("date", ""), reverse=True)

        total = len(binding) + len(persuasive) + len(other)
        if total > 0:
            updates[case["label"]] = {
                "corpus":     case["corpus"],
                "binding":    binding,
                "persuasive": persuasive,
                "other":      other,
                "total":      total,
            }
    return updates


# ── Backward-compatible wrapper for the existing get_tetrad_updates name ─────
def get_tetrad_updates(since_year: int = 2023) -> dict:
    """
    DEPRECATED: backward-compatibility wrapper for the older API surface.

    Existing callers in app.py may pass `since_year` as an integer
    (e.g., 2023). This wrapper translates that to a date-floor and
    flattens the new tiered structure to the old format.

    New code should use get_tracked_updates() directly.
    """
    today = datetime.now()
    years_back = max(1, today.year - since_year)
    if years_back == 1:
        floor_label = "1 year"
    elif years_back <= 3:
        floor_label = "3 years"
    elif years_back <= 5:
        floor_label = "5 years"
    else:
        floor_label = "All"

    new_updates = get_tracked_updates(
        date_floor_label=floor_label,
        user_jurisdiction="*",
        corpus="Distortion",  # The old function only tracked Tetrad
    )
    # Flatten tier structure to a single list (old format)
    flat_updates = {}
    for label, data in new_updates.items():
        flat = (list(data.get("binding", [])) +
                list(data.get("persuasive", [])) +
                list(data.get("other", [])))
        if flat:
            flat_updates[label] = flat
    return flat_updates


def format_canlii_results(results, node_name: str = "") -> str:
    """
    Format CanLII search results as readable text.
    Accepts either a tiered dict or a flat list (backward compat).
    """
    if isinstance(results, dict) and "binding" in results:
        results = flatten_search_results(results)
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
    """Check if CanLII API key is available (does not validate the key)."""
    return bool(_get_api_key())


@st.cache_data(ttl=3600, show_spinner=False)  # 1-hour cache for validity check
def validate_api_key() -> dict:
    """
    Make a minimal test query to verify the API key actually works.

    Cached for 1 hour to avoid repeated probing on every page render.
    Returns dict: {"valid": bool, "error": str or None}.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"valid": False, "error": "No CanLII API key configured."}

    # Minimal probe: list available case databases (cheap, always available).
    url = f"{CANLII_BASE_URL}/caseBrowse/en/"
    params = {"api_key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return {"valid": True, "error": None}
        if resp.status_code == 401:
            return {"valid": False, "error": "API key rejected (401 Unauthorized)."}
        if resp.status_code == 403:
            return {"valid": False, "error": "API key forbidden (403)."}
        return {"valid": False, "error": f"API returned status {resp.status_code}."}
    except requests.Timeout:
        return {"valid": False, "error": "API timeout during key validation."}
    except Exception as e:
        return {"valid": False, "error": str(e)}
