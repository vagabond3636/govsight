#!/usr/bin/env python
# talk.py – GovSight: Pinecone Memory + SerpAPI Web Intelligence Fallback
#
# Flow:
#   1. User query
#   2. Extract dynamic constraints (any domain: location, time, role, etc.)
#   3. Pinecone semantic search
#       - If high confidence → answer from memory
#       - Else → SerpAPI web fallback:
#           a. Search (top_n=10)
#           b. Fetch page text
#           c. GPT doc relevance eval vs constraints
#           d. Stop when min_high_conf docs collected (default 3 @ score≥0.7)
#           e. GPT synthesis across findings
#   4. Return answer + ranked sources + confidence
#
# Logs JSONL traces of web fallbacks to ./logs/web_fallback.log
#
# Requirements:
#   pip install openai pinecone-client python-dotenv requests beautifulsoup4
#   .env must define: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, SERPAPI_API_KEY (or SERP_API_KEY)
#
# NOTE: Rotate API keys if you’ve shared them publicly.


import os
import json
import logging
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    SERPAPI_API_KEY,   # supports fallback naming in config
    DEFAULT_OPENAI_MODEL,
    DEFAULT_EMBED_MODEL,
)

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
load_dotenv()

openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("govsight")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
WEB_LOG_PATH = os.path.join(LOG_DIR, "web_fallback.log")

# Thresholds / Tunables
PINECONE_SCORE_THRESH = 0.80
WEB_TOP_N = 10
WEB_MIN_HIGH_CONF = 3
WEB_RELEVANCE_CUTOFF = 0.70
MAX_HTML_CHARS = 50_000          # raw page capture cap
MAX_DOC_CHARS_FOR_EVAL = 4_000   # truncate per-doc before GPT eval
MAX_TOK_ANSWER = 800


# -----------------------------------------------------------------------------
# Utility: safe JSON extraction from GPT freeform output
# -----------------------------------------------------------------------------
def _safe_extract_json(text: str, default: Any) -> Any:
    if not text:
        return default
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return default
    fragment = text[start:end+1]
    try:
        return json.loads(fragment)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Constraint Extraction (dynamic; any domain)
# -----------------------------------------------------------------------------
def extract_constraints_with_gpt(user_input: str) -> Dict[str, Any]:
    prompt = f"""
You are a natural language understanding engine.

Analyze the user question below and extract all meaningful constraints that narrow the scope of information. 
Constraints may include (but are not limited to): location, time period, date, organization, role/title, domain/topic, color, measurement units, data type, emotion, named entities, identifiers, comparison targets, jurisdiction, or anything else that helps disambiguate.

Return ONLY valid JSON. Do not wrap in backticks or add commentary.

User question: "{user_input}"

JSON:
"""
    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        constraints = _safe_extract_json(raw, {})
        if not isinstance(constraints, dict):
            constraints = {}
        return constraints
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract constraints: {e}")
        return {}


# -----------------------------------------------------------------------------
# Pinecone Retrieval
# -----------------------------------------------------------------------------
def get_pinecone_answer(user_input: str):
    """Return (context, avg_score, sources_list)."""
    try:
        embedded = openai.embeddings.create(
            model=DEFAULT_EMBED_MODEL,
            input=[user_input]
        ).data[0].embedding
    except Exception as e:
        logger.error(f"🔥 Embedding failed: {e}")
        return None, 0.0, []

    try:
        results = index.query(vector=embedded, top_k=5, include_metadata=True)
    except Exception as e:
        logger.error(f"🔥 Pinecone query failed: {e}")
        return None, 0.0, []

    matches = results.get("matches", []) or []
    if not matches:
        return None, 0.0, []

    combined_score = sum(m['score'] for m in matches) / len(matches)
    context_items = [m['metadata'].get('summary', '') for m in matches]
    sources = [m['metadata'].get('title', 'Untitled') for m in matches]
    return "\n".join(context_items), combined_score, sources


# -----------------------------------------------------------------------------
# Final Prompt Builder (memory path)
# -----------------------------------------------------------------------------
def build_final_prompt(user_input, context, source_type, constraints=None):
    constraint_note = ""
    if constraints:
        constraint_note = (
            "\nApply the following dynamically extracted constraints during reasoning:\n"
            + json.dumps(constraints, indent=2) + "\n"
        )

    return f"""
You're GovSight, an intelligent AI assistant designed to reason like ChatGPT.

The user asked:
"{user_input}"

Answer using the provided {source_type} context below.
If information is missing, say so — do not fabricate.

{constraint_note}
Context:
{context}

Answer clearly, directly, and confidently:
"""


# -----------------------------------------------------------------------------
# --- SerpAPI Web Intelligence Fallback ---
# -----------------------------------------------------------------------------
SERP_ENDPOINT = "https://serpapi.com/search.json"
USER_AGENT = "GovSightBot/1.0 (+https://govsight.local)"


def serp_search_raw(query: str, engine: str = "google", num: int = 10, **kwargs) -> Dict[str, Any]:
    """Low-level SerpAPI call; returns raw JSON or {} on failure."""
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": engine,
        "q": query,
        "num": num,
        "output": "json",
        **kwargs,
    }
    try:
        r = requests.get(SERP_ENDPOINT, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"SerpAPI search failed for '{query}': {e}")
        return {}


def _parse_serp_results(raw: Dict[str, Any], max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Normalize SerpAPI response into a flat list of result dicts:
    [{title, url, snippet, position, source_type}, ...]
    Pulls from: organic_results, answer_box, knowledge_graph, top_stories.
    """
    results: List[Dict[str, Any]] = []

    # Organic
    for item in raw.get("organic_results", [])[:max_items]:
        results.append({
            "title": item.get("title") or "",
            "url": item.get("link") or "",
            "snippet": item.get("snippet") or "",
            "position": item.get("position"),
            "source_type": "organic",
        })

    # Answer box
    ab = raw.get("answer_box")
    if ab:
        results.append({
            "title": ab.get("title") or ab.get("result", "Answer Box"),
            "url": ab.get("link", ""),
            "snippet": ab.get("snippet") or ab.get("result", ""),
            "position": None,
            "source_type": "answer_box",
        })

    # Knowledge graph
    kg = raw.get("knowledge_graph")
    if kg:
        results.append({
            "title": kg.get("title", "Knowledge Graph"),
            "url": kg.get("source", ""),
            "snippet": kg.get("description", ""),
            "position": None,
            "source_type": "knowledge_graph",
        })

    # Top stories (news)
    for story in raw.get("top_stories", [])[:max_items]:
        results.append({
            "title": story.get("title") or "",
            "url": story.get("link") or "",
            "snippet": story.get("source") or "",
            "position": None,
            "source_type": "top_story",
        })

    # Dedupe by URL, preserve order
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


def _fetch_url_text(url: str, max_chars: int = MAX_HTML_CHARS) -> str:
    """Fetch HTML & return cleaned text."""
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
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    except Exception as e:
        logger.warning(f"Parse failed: {url} ({e})")
        return ""


def _serp_search_and_fetch(query: str, engine: str = "google", num: int = 10) -> List[Dict[str, Any]]:
    """SerpAPI search + fetch text for each result."""
    raw = serp_search_raw(query, engine=engine, num=num)
    results = _parse_serp_results(raw, max_items=num)
    for r_ in results:
        r_["text"] = _fetch_url_text(r_["url"])
    return results


# -----------------------------------------------------------------------------
# Document Relevance Evaluation (GPT)
# -----------------------------------------------------------------------------
_DOC_EVAL_PROMPT_TMPL = """You are GovSight, a research-grade AI analyst.

User question:
{query}

Known constraints (may include domain, location, time, roles, entities, etc.):
{constraints_json}

Below is content from a single web source (truncated). Assess whether this source helps answer the user question.

Return ONLY valid JSON in this schema:
{{
  "relevance_score": <float 0-1>,
  "useful": <true|false>,
  "key_facts": ["short bullet facts relevant to the question"],
  "notes": "short diagnostic"
}}
"""

def _evaluate_doc_with_gpt(query: str, constraints: Dict[str, Any], doc_text: str, url: str) -> Dict[str, Any]:
    model = DEFAULT_OPENAI_MODEL
    prompt = _DOC_EVAL_PROMPT_TMPL.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
    ) + f"\nSource URL: {url}\n\nContent:\n{doc_text[:MAX_DOC_CHARS_FOR_EVAL]}\n\nJSON:"
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        js = _safe_extract_json(raw, {})
        if not isinstance(js, dict):
            js = {}
        return js
    except Exception as e:
        logger.warning(f"Doc eval failed for {url}: {e}")
        return {"relevance_score": 0.0, "useful": False, "key_facts": [], "notes": f"eval_error:{e}"}


# -----------------------------------------------------------------------------
# Synthesis (GPT)
# -----------------------------------------------------------------------------
_SYNTH_PROMPT_TMPL = """You are GovSight, a high-accuracy research AI.

User question:
{query}

Constraints:
{constraints_json}

You reviewed multiple web sources. Structured findings follow:
{findings_json}

Using ONLY supported facts:
- Prefer sources with higher relevance_score.
- Resolve conflicts if possible; otherwise note uncertainty.
- If the answer cannot be confirmed, say so.
- Be concise, direct, and useful to a professional user.

Answer:
"""

def _synthesize_answer_with_gpt(query: str, constraints: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
    model = DEFAULT_OPENAI_MODEL
    prompt = _SYNTH_PROMPT_TMPL.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
        findings_json=json.dumps(findings, indent=2),
    )
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=MAX_TOK_ANSWER,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return "I could not confidently synthesize an answer from available sources."


# -----------------------------------------------------------------------------
# Web Fallback Public Entry
# -----------------------------------------------------------------------------
def answer_from_web(
    query: str,
    constraints: Dict[str, Any],
    top_n: int = WEB_TOP_N,
    min_high_conf: int = WEB_MIN_HIGH_CONF,
    relevance_cutoff: float = WEB_RELEVANCE_CUTOFF,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. SerpAPI search (top_n)
      2. Fetch page text
      3. GPT doc eval (relevance_score)
      4. Stop early after min_high_conf docs >= cutoff
      5. GPT synthesis
      6. Log JSONL trace
    """
    # Search + fetch
    raw_results = _serp_search_and_fetch(query, engine="google", num=top_n)

    findings: List[Dict[str, Any]] = []
    high_conf = 0

    for r_ in raw_results:
        doc_eval = _evaluate_doc_with_gpt(query, constraints, r_.get("text", ""), r_["url"])
        score = float(doc_eval.get("relevance_score", 0.0))

        findings.append({
            "title": r_["title"],
            "url": r_["url"],
            "snippet": r_["snippet"],
            "source_type": r_["source_type"],
            "relevance_score": score,
            "gpt_eval": doc_eval,
        })

        if score >= relevance_cutoff:
            high_conf += 1
            if high_conf >= min_high_conf:
                break

    # Sort by relevance desc
    findings.sort(key=lambda d: d["relevance_score"], reverse=True)

    # Synthesize final answer
    answer_text = _synthesize_answer_with_gpt(query, constraints, findings)

    # Confidence label
    if high_conf >= min_high_conf:
        confidence = "High (web corroborated)"
    elif any(f["relevance_score"] >= relevance_cutoff for f in findings):
        confidence = "Medium (limited corroboration)"
    else:
        confidence = "Low (weak matches)"

    # Prepare compact sources list
    sources = [{
        "title": f["title"],
        "url": f["url"],
        "score": f["relevance_score"],
    } for f in findings[:5]]

    # Log
    _log_web_trace(query, constraints, findings, answer_text, confidence)

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "constraints": constraints,
        "trace": findings,  # full detail if caller wants
    }


def _log_web_trace(query: str, constraints: Dict[str, Any], findings: List[Dict[str, Any]], answer: str, confidence: str):
    rec = {
        "query": query,
        "constraints": constraints,
        "confidence": confidence,
        "answer": answer,
        "findings": findings,
    }
    try:
        with open(WEB_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        logger.warning(f"Could not log web fallback trace: {e}")


# -----------------------------------------------------------------------------
# CLI Main Loop
# -----------------------------------------------------------------------------
def main():
    print("🧠 GovSight RAG CLI initialized. Ask anything — I’ll reason like ChatGPT.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting GovSight.")
            break

        # Extract constraints
        constraints = extract_constraints_with_gpt(user_input)

        # Pinecone memory path
        pinecone_context, pinecone_score, pinecone_sources = get_pinecone_answer(user_input)

        if pinecone_score >= PINECONE_SCORE_THRESH and pinecone_context:
            final_prompt = build_final_prompt(user_input, pinecone_context, "Pinecone Memory", constraints)
            try:
                response = openai.chat.completions.create(
                    model=DEFAULT_OPENAI_MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.4,
                    max_tokens=MAX_TOK_ANSWER,
                )
                print(f"\n🤖 GovSight (Confidence: High — memory match)")
                print(f"Sources: {pinecone_sources[0] if pinecone_sources else 'None'}\n")
                print(response.choices[0].message.content.strip())
            except Exception as e:
                logger.error(f"🔥 GPT final answer (memory path) failed: {e}")
                print("Something went wrong while generating the final answer from memory.")
            continue

        # Web fallback path
        print("\n🔁 Memory insufficient. Switching to live web reasoning (SerpAPI)...")
        try:
            web_result = answer_from_web(
                query=user_input,
                constraints=constraints,
                top_n=WEB_TOP_N,
                min_high_conf=WEB_MIN_HIGH_CONF,
                relevance_cutoff=WEB_RELEVANCE_CUTOFF,
            )
        except Exception as e:
            logger.error(f"🔥 Web fallback failed: {e}")
            print("Could not retrieve live web information. Try again later or refine your question.")
            continue

        # Display web result
        print(f"\n🤖 GovSight (Confidence: {web_result['confidence']})")
        if web_result["sources"]:
            print("Sources (ranked):")
            for s in web_result["sources"]:
                print(f"- {s['title']} ({s['url']}) [score={s['score']:.2f}]")
        else:
            print("Sources: None")

        print("\nAnswer:\n" + web_result["answer"])


if __name__ == "__main__":
    main()
