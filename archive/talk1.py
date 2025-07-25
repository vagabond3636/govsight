#!/usr/bin/env python
# talk.py – GovSight: Persistent Memory + Pinecone RAG + SerpAPI Web Intelligence
#
# New in this version:
#   • Persistent conversation logging (SQLite via memory_manager)
#   • Auto fact extraction from assistant answers -> memory + Pinecone
#   • Watchlist trigger detection (“track”, “monitor”, “any updates”)
#   • Auto session summary & entity mapping on exit
#   • Pinecone memory recall + SerpAPI fallback (already working)
#
# Run: python talk.py


import os
import json
import logging
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai
from pinecone import Pinecone

import config  # import module; we'll pull what we need safely

from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    SERPAPI_API_KEY,
)

# fallback defaults if not set in config
DEFAULT_OPENAI_MODEL = getattr(config, "DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL  = getattr(config, "DEFAULT_EMBED_MODEL", "text-embedding-3-small")

# Memory manager integration
import memory_manager as mem


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

# Tunables
PINECONE_SCORE_THRESH = 0.80
WEB_TOP_N = 10
WEB_MIN_HIGH_CONF = 3
WEB_RELEVANCE_CUTOFF = 0.70
MAX_HTML_CHARS = 50_000
MAX_DOC_CHARS_FOR_EVAL = 12_000   # bumped up from 4k to reduce missed facts
MAX_TOK_ANSWER = 800


# -----------------------------------------------------------------------------
# Utility: safe JSON extraction
# -----------------------------------------------------------------------------
def _safe_extract_json(text: str, default: Any):
    if not text:
        return default
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return default
    frag = text[start:end+1]
    try:
        return json.loads(frag)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Constraint Extraction (dynamic; any domain)
# -----------------------------------------------------------------------------
def extract_constraints_with_gpt(user_input: str) -> Dict[str, Any]:
    prompt = f"""
You are a natural language understanding engine.

Analyze the user question below and extract all meaningful constraints that narrow the scope of information. 
Constraints may include (but are not limited to): location, time period, date, organization, role/title, domain/topic, regulation, funding program, data type, emotion, named entities, identifiers, comparison targets, jurisdiction, or anything else that helps disambiguate.

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
    context_items = [m['metadata'].get('text') or m['metadata'].get('summary', '') for m in matches]
    sources = [m['metadata'].get('title', m['id']) for m in matches]
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
You're GovSight, an intelligent AI assistant designed to reason like a professional research analyst.

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
# SerpAPI Web Intelligence Fallback
# -----------------------------------------------------------------------------
SERP_ENDPOINT = "https://serpapi.com/search.json"
USER_AGENT = "GovSightBot/1.0 (+https://govsight.local)"


def serp_search_raw(query: str, engine: str = "google", num: int = 10, **kwargs) -> Dict[str, Any]:
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
    # Top stories
    for story in raw.get("top_stories", [])[:max_items]:
        results.append({
            "title": story.get("title") or "",
            "url": story.get("link") or "",
            "snippet": story.get("source") or "",
            "position": None,
            "source_type": "top_story",
        })
    # Dedup
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
    raw = serp_search_raw(query, engine=engine, num=num)
    results = _parse_serp_results(raw, max_items=num)
    for r_ in results:
        r_["text"] = _fetch_url_text(r_["url"])
    return results


# ------------------ Doc Evaluation ------------------
_DOC_EVAL_PROMPT_TMPL = """You are GovSight, a research-grade AI analyst.

User question:
{query}

Known constraints (domain, location, time, roles, entities, etc.):
{constraints_json}

Below is content from a single web source (truncated). Assess whether this source helps answer the user question.

Return ONLY valid JSON:
{{
  "relevance_score": <float 0-1>,
  "useful": <true|false>,
  "key_facts": ["short bullet facts relevant to the question"],
  "notes": "short diagnostic"
}}
"""

def _evaluate_doc_with_gpt(query: str, constraints: Dict[str, Any], doc_text: str, url: str) -> Dict[str, Any]:
    prompt = _DOC_EVAL_PROMPT_TMPL.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
    ) + f"\nSource URL: {url}\n\nContent:\n{doc_text[:MAX_DOC_CHARS_FOR_EVAL]}\n\nJSON:"
    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
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


# ------------------ Synthesis ------------------
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
- Be concise and direct.

Answer:
"""

def _synthesize_answer_with_gpt(query: str, constraints: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
    prompt = _SYNTH_PROMPT_TMPL.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
        findings_json=json.dumps(findings, indent=2),
    )
    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=MAX_TOK_ANSWER,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return "I could not confidently synthesize an answer from available sources."


# ------------------ Web Fallback Entry ------------------
def answer_from_web(
    query: str,
    constraints: Dict[str, Any],
    top_n: int = WEB_TOP_N,
    min_high_conf: int = WEB_MIN_HIGH_CONF,
    relevance_cutoff: float = WEB_RELEVANCE_CUTOFF,
) -> Dict[str, Any]:
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

    findings.sort(key=lambda d: d["relevance_score"], reverse=True)

    answer_text = _synthesize_answer_with_gpt(query, constraints, findings)

    if high_conf >= min_high_conf:
        confidence = "High (web corroborated)"
    elif any(f["relevance_score"] >= relevance_cutoff for f in findings):
        confidence = "Medium (limited corroboration)"
    else:
        confidence = "Low (weak matches)"

    sources = [{
        "title": f["title"],
        "url": f["url"],
        "score": f["relevance_score"],
    } for f in findings[:5]]

    _log_web_trace(query, constraints, findings, answer_text, confidence)

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "constraints": constraints,
        "trace": findings,
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
# CLI Main Loop with Persistent Memory
# -----------------------------------------------------------------------------
def main():
    print("🧠 GovSight RAG CLI initialized. Ask anything — I’ll remember everything.")

    # open a memory session
    session_id = mem.open_session()
    print(f"(Memory session #{session_id} started)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting GovSight. Summarizing session...")
            break

        # log user turn
        mem.log_turn(session_id, "user", user_input)

        # extract constraints
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
                answer_text = response.choices[0].message.content.strip()
                print(f"\n🤖 GovSight (Confidence: High — memory match)")
                print(f"Sources: {pinecone_sources[0] if pinecone_sources else 'None'}\n")
                print(answer_text)
            except Exception as e:
                logger.error(f"🔥 GPT final answer (memory path) failed: {e}")
                answer_text = "Something went wrong while generating the final answer from memory."

        else:
            # Web fallback
            print("\n🔁 Memory insufficient. Switching to live web reasoning (SerpAPI)...")
            try:
                web_result = answer_from_web(
                    query=user_input,
                    constraints=constraints,
                    top_n=WEB_TOP_N,
                    min_high_conf=WEB_MIN_HIGH_CONF,
                    relevance_cutoff=WEB_RELEVANCE_CUTOFF,
                )
                answer_text = web_result["answer"]
                print(f"\n🤖 GovSight (Confidence: {web_result['confidence']})")
                if web_result["sources"]:
                    print("Sources (ranked):")
                    for s in web_result["sources"]:
                        print(f"- {s['title']} ({s['url']}) [score={s['score']:.2f}]")
                else:
                    print("Sources: None")
                print("\nAnswer:\n" + answer_text)
            except Exception as e:
                logger.error(f"🔥 Web fallback failed: {e}")
                answer_text = "Could not retrieve live web information. Try again later or refine your question."
                print(answer_text)

        # -------- Memory enrichment from this turn --------
        mem.log_turn(session_id, "assistant", answer_text)

        # Extract candidate facts & store
        facts = mem.extract_facts_from_turn(user_input, answer_text)
        mem.store_facts(session_id, facts)

        # Detect watchlist triggers from this turn
        watch_decision = mem.detect_watchlist_from_turn(user_input, answer_text)
        if watch_decision and watch_decision.get("create_watch"):
            mem.create_watchlist(
                topic=watch_decision.get("topic") or user_input,
                entity_name=watch_decision.get("entity_name"),
                frequency=watch_decision.get("frequency") or "weekly",
            )
            print("📌 Added to watchlist:", watch_decision.get("topic") or user_input)

    # end loop
    mem.close_session(session_id)
    print(f"💾 Session #{session_id} summarized & saved. Goodbye.")


if __name__ == "__main__":
    main()
