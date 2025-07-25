#!/usr/bin/env python
# talk.py – GovSight: Persistent Memory + Pinecone RAG + SerpAPI Web Intelligence
#
# Phase 1 Conversational Upgrade (patched w/ memory contamination guard):
#   • 12‑turn rolling conversation buffer
#   • Active context (merged constraints across turns)
#   • Intent classifier (chat | followup | fact_lookup | recall | command)
#   • Buffer seeding from latest prior session summary
#   • Safe constraint merge (dict aware)
#   • Contextualized follow‑up web queries
#   • Pinecone retrieval now prefers curated memory (facts, summaries) and
#     validates relevance vs constraints to avoid “I don’t know” self‑poisoning.
#
# Run: python talk.py

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import datetime as dt

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
# Conversational buffer / interaction state
# -----------------------------------------------------------------------------
BUFFER_MAX_TURNS = 12  # per your decision
conversation_buffer = deque(maxlen=BUFFER_MAX_TURNS)

# active_context carries merged constraints from last fact_lookup/followup turn
active_context: Dict[str, Any] = {}


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
# Conversation buffer helpers
# -----------------------------------------------------------------------------
def push_buffer(role: str, text: str, constraints: Optional[Dict[str, Any]] = None, intent: Optional[str] = None):
    """Add a turn to the rolling conversation buffer."""
    conversation_buffer.append({
        "role": role,
        "text": text,
        "constraints": constraints,
        "intent": intent,
        "ts": dt.datetime.utcnow().isoformat(),
    })


def _merge_list_safe(existing: List[Any], incoming: List[Any]) -> List[Any]:
    """Dedupe while preserving order; safe for dict items."""
    out: List[Any] = []
    seen = set()

    def keyify(item):
        if isinstance(item, (str, int, float, bool, type(None))):
            return ("scalar", item)
        try:
            return ("json", json.dumps(item, sort_keys=True, default=str))
        except Exception:
            return ("repr", repr(item))

    for item in existing + incoming:
        k = keyify(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def merge_constraints(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge 'new' constraints into 'base' without blowing up on unhashables (dicts).
    - Scalars in new override base scalars.
    - Lists merge (dedupe) while preserving order. Works even if list items are dicts.
    - Missing keys preserved.
    """
    if not base:
        return dict(new) if new else {}
    merged = dict(base)

    for k, v in (new or {}).items():
        if isinstance(v, list):
            existing = merged.get(k, [])
            if not isinstance(existing, list):
                existing = [existing] if existing else []
            merged[k] = _merge_list_safe(existing, v)
        elif isinstance(v, str):
            if isinstance(merged.get(k), list):
                if v not in merged[k]:
                    merged[k].append(v)
            else:
                merged[k] = v
        else:
            merged[k] = v
    return merged


_INTENT_PROMPT_TMPL = """You are GovSight's conversation state classifier.

You will be given the recent conversation turns (most recent last), the current active context,
and the new user input. Determine how the user intends to interact.

Interaction types:
- "chat": commentary / acknowledgment / informal talk, no new info requested.
- "followup": user referring back to the recent topic (pronouns, "what about...", "and the funding?")
- "fact_lookup": user asking a new discrete question that likely requires retrieval.
- "recall": user asking what we discussed previously ("remember...", "what did we say...", "last time we talked...")
- "command": user directing the system to act ("track that", "add to watchlist", "store this", "summarize that")

Return ONLY valid JSON with this schema:
{{
  "interaction_type": "chat" | "followup" | "fact_lookup" | "recall" | "command",
  "needs_retrieval": true | false,
  "inherits_context": true | false,
  "explicit_entities": [strings],
  "implicit_topics": [strings],
  "time_reference": null | "<normalized date or relative spec>"
}}

Recent turns (oldest first):
{history_json}

Active context:
{active_json}

New user input: "{user_input}"

JSON:
"""

def classify_interaction(user_input: str) -> Dict[str, Any]:
    """Call GPT to classify how to handle this turn."""
    history = list(conversation_buffer)
    history_json = json.dumps(history, indent=2)
    active_json = json.dumps(active_context, indent=2)

    prompt = _INTENT_PROMPT_TMPL.format(
        history_json=history_json,
        active_json=active_json,
        user_input=user_input,
    )

    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
        js = _safe_extract_json(raw, {})
        if not isinstance(js, dict):
            js = {}
        return js
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")
        return {}


def seed_buffer_from_last_session():
    """
    Prime the conversation buffer & active_context with the latest summarized session,
    so continuity carries across days. Uses memory_manager.get_latest_session_summary().
    """
    latest = mem.get_latest_session_summary()
    if not latest:
        return
    summary_text = latest["summary"] or "(no prior summary)"
    push_buffer("assistant", f"[Prev Session Summary] {summary_text}",
                constraints={"entities": latest.get("entities"), "topics": latest.get("topics")},
                intent="session_summary")

    # Use entities/topics as starting active_context
    global active_context
    active_context = {
        "entities": latest.get("entities") or [],
        "topics": latest.get("topics") or [],
    }


# -----------------------------------------------------------------------------
# Pinecone Retrieval (with contamination guard)
# -----------------------------------------------------------------------------
def _constraint_tokens(constraints: Dict[str, Any]) -> List[str]:
    """Flatten constraint dict to a list of lowercase tokens."""
    toks: List[str] = []
    for v in constraints.values():
        if isinstance(v, str):
            toks.append(v.lower())
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    toks.append(item.lower())
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("title")
                    if isinstance(name, str):
                        toks.append(name.lower())
        elif isinstance(v, dict):
            name = v.get("name") or v.get("title")
            if isinstance(name, str):
                toks.append(name.lower())
    return [t for t in toks if t]


def _text_matches_tokens(text: str, tokens: List[str]) -> bool:
    if not tokens:
        return True  # nothing to test against
    tl = text.lower()
    return any(tok in tl for tok in tokens)


def get_pinecone_answer(user_input: str, constraints: Optional[Dict[str, Any]] = None):
    """
    Query Pinecone and return curated memory if sufficiently relevant.
    We prefer structured items (facts, session_summary) over raw turns to prevent contamination.
    If nothing relevant to constraints is found, return score=0 (force web fallback).
    """
    try:
        embedded = openai.embeddings.create(
            model=DEFAULT_EMBED_MODEL,
            input=[user_input]
        ).data[0].embedding
    except Exception as e:
        logger.error(f"🔥 Embedding failed: {e}")
        return None, 0.0, []

    try:
        results = index.query(vector=embedded, top_k=20, include_metadata=True)
    except Exception as e:
        logger.error(f"🔥 Pinecone query failed: {e}")
        return None, 0.0, []

    matches = results.get("matches", []) or []
    if not matches:
        return None, 0.0, []

    # Partition by metadata type
    curated_types = {"fact", "session_summary"}
    curated = []
    fallback = []
    for m in matches:
        mtype = m["metadata"].get("type")
        if mtype in curated_types:
            curated.append(m)
        else:
            fallback.append(m)

    # If no curated, fall back to non-curated but drop low-info assistant turns
    if not curated:
        for m in fallback:
            txt = (m["metadata"].get("text") or "").lower()
            if "i currently do not have" in txt or "i do not have information" in txt:
                continue
            curated.append(m)

    if not curated:
        return None, 0.0, []

    # Build context text; compute avg score
    tokens = _constraint_tokens(constraints or {})
    relevant = []
    for m in curated:
        txt = m["metadata"].get("text") or m["metadata"].get("summary", "") or ""
        if _text_matches_tokens(txt, tokens):
            relevant.append(m)

    # If nothing hits constraint tokens, treat memory as insufficient.
    if not relevant:
        return None, 0.0, []

    used = relevant
    combined_score = sum(m['score'] for m in used) / len(used)
    context_items = [m['metadata'].get('text') or m['metadata'].get('summary', '') for m in used]
    sources = [m['metadata'].get('title', m['id']) for m in used]

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
# Build contextualized search query for followups
# -----------------------------------------------------------------------------
def build_contextual_query(user_input: str, constraints: Dict[str, Any]) -> str:
    """
    Build an augmented search query that includes user_input plus salient constraint tokens.
    Generic (domain-agnostic): we just append stringified constraint values.
    We include up to ~5 short tokens to avoid query bloat.
    """
    tokens = []

    for k, v in constraints.items():
        if isinstance(v, str):
            tokens.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    if 1 <= len(item) <= 60:
                        tokens.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("title")
                    if isinstance(name, str) and name:
                        tokens.append(name)
        elif isinstance(v, dict):
            name = v.get("name") or v.get("title")
            if isinstance(name, str) and name:
                tokens.append(name)

    # Keep order; dedupe case-insensitive
    seen = set()
    deduped = []
    for t in tokens:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        deduped.append(t)

    if len(deduped) > 5:
        deduped = deduped[:5]

    if deduped:
        return user_input + " " + " ".join(deduped)
    return user_input


# -----------------------------------------------------------------------------
# CLI Main Loop with Persistent Memory + Conversation Buffer Instrumentation
# -----------------------------------------------------------------------------
def main():
    global active_context  # we update this inside the loop

    print("🧠 GovSight RAG CLI initialized. Ask anything — I’ll remember everything.")

    # open a memory session
    session_id = mem.open_session()
    print(f"(Memory session #{session_id} started)")

    # seed short-term buffer from most recent prior session (if any)
    seed_buffer_from_last_session()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting GovSight. Summarizing session...")
            break

        # log user turn (persistent memory)
        mem.log_turn(session_id, "user", user_input)

        # push into rolling conversation buffer (no constraints yet)
        push_buffer("user", user_input)

        # classify how to handle this turn (chat vs followup vs lookup vs recall vs command)
        intent_info = classify_interaction(user_input)
        intent_type = intent_info.get("interaction_type", "fact_lookup")
        inherits = intent_info.get("inherits_context", False)

        # DEBUG: show classification
        print(f"[DEBUG intent] {intent_type} | inherits={inherits} | needs_retrieval={intent_info.get('needs_retrieval')}")

        # Extract constraints from current user message
        constraints = extract_constraints_with_gpt(user_input)

        # Merge with active_context if followup/fact_lookup and inherits_context True
        if intent_type in ("followup", "fact_lookup") and inherits:
            constraints = merge_constraints(active_context, constraints)

        # Pinecone memory path (contamination-guarded)
        pinecone_context, pinecone_score, pinecone_sources = get_pinecone_answer(user_input, constraints)

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
                search_query = user_input
                if inherits:  # followup (or any inherits=True)
                    search_query = build_contextual_query(user_input, constraints)
                    print(f"[DEBUG search] contextualized query: {search_query}")

                web_result = answer_from_web(
                    query=search_query,
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

        # -------- Conversation buffer enrichment --------
        push_buffer("assistant", answer_text, constraints=constraints, intent=intent_type)

        # Update active_context if this was a fact-bearing turn (lookup or followup produced answer)
        if intent_type in ("fact_lookup", "followup"):
            active_context = merge_constraints(active_context, constraints)

        # -------- Persistent memory enrichment --------
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
