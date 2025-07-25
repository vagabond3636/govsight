# serp_client.py – SerpAPI wrapper + HTML fetch utilities for GovSight
#
# Usage:
#   from serp_client import serp_search, fetch_url_text
#
# Configuration: requires SERPAPI_API_KEY (from config.py)

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup

from config import SERPAPI_API_KEY

SERP_ENDPOINT = "https://serpapi.com/search.json"
USER_AGENT = "GovSightBot/1.0 (+https://govsight.local)"

logger = logging.getLogger(__name__)


def serp_search(query: str, engine: str = "google", num: int = 10, **kwargs) -> Dict[str, Any]:
    """
    Call SerpAPI and return raw JSON response.
    """
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": engine,
        "q": query,
        "num": num,
        "output": "json",
        # You can set location, hl, gl, tbs, etc. from kwargs
        **kwargs,
    }
    try:
        r = requests.get(SERP_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"SerpAPI search failed for query='{query}': {e}")
        return {}


def parse_serp_results(raw: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Normalize SerpAPI response into a flat list of result dicts:
    [{title, url, snippet, position, source_type}, ...]
    Pulls from: organic_results, answer_box, knowledge_graph, top_stories (where available).
    """
    results: List[Dict[str, Any]] = []

    # Organic search
    for item in raw.get("organic_results", [])[:max_items]:
        results.append({
            "title": item.get("title") or "",
            "url": item.get("link") or "",
            "snippet": item.get("snippet") or "",
            "position": item.get("position") or None,
            "source_type": "organic",
        })

    # Answer box (featured snippet)
    ab = raw.get("answer_box")
    if ab:
        results.append({
            "title": ab.get("title") or ab.get("result", "Answer Box"),
            "url": ab.get("link", ""),
            "snippet": ab.get("snippet") or ab.get("result", ""),
            "position": None,
            "source_type": "answer_box",
        })

    # Knowledge graph (entity card)
    kg = raw.get("knowledge_graph")
    if kg:
        results.append({
            "title": kg.get("title", "Knowledge Graph"),
            "url": kg.get("source", ""),
            "snippet": kg.get("description", ""),
            "position": None,
            "source_type": "knowledge_graph",
        })

    # News / Top Stories (if engine=google)
    for story in raw.get("top_stories", [])[:max_items]:
        results.append({
            "title": story.get("title") or "",
            "url": story.get("link") or "",
            "snippet": story.get("source") or "",
            "position": None,
            "source_type": "top_story",
        })

    # Deduplicate by URL, preserve order
    seen = set()
    deduped = []
    for r_ in results:
        url = r_["url"]
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(r_)
        if len(deduped) >= max_items:
            break

    return deduped


def fetch_url_text(url: str, max_chars: int = 50000) -> str:
    """
    Fetch HTML and extract readable text. Truncate to max_chars to control token cost.
    """
    if not url:
        return ""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Fetch failed: {url} ({e})")
        return ""

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style/nav
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception as e:
        logger.warning(f"Parse failed: {url} ({e})")
        return ""


def serp_search_and_fetch(query: str, engine: str = "google", num: int = 10, fetch: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience: run SerpAPI search, parse results, and (optionally) fetch text for each.
    Returns list: [{title, url, snippet, source_type, text}, ...]
    """
    raw = serp_search(query, engine=engine, num=num)
    results = parse_serp_results(raw, max_items=num)
    if fetch:
        for r_ in results:
            r_["text"] = fetch_url_text(r_["url"])
    return results
