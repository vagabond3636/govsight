# web_reasoner.py – Constraint-aware web fallback reasoning for GovSight
#
# Public entry: answer_from_web(query, constraints, top_n=10, min_high_conf=3)
#
# Depends on:
#   openai, serp_client.serp_search_and_fetch
#   config.DEFAULT_OPENAI_MODEL (optional)
#
# Logs JSONL traces to logs/web_fallback.log

import os
import json
import logging
from typing import Dict, Any, List, Optional

import openai

from config import DEFAULT_OPENAI_MODEL
from serp_client import serp_search_and_fetch

logger = logging.getLogger(__name__)
LOG_PATH = os.path.join("logs", "web_fallback.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# ---------------------------
# Utility: basic constraint score (string heuristics)
# ---------------------------
def heuristic_score(text: str, constraints: Dict[str, Any]) -> float:
    if not text:
        return 0.0
    text_l = text.lower()
    score = 0
    hits = 0
    for k, v in constraints.items():
        if isinstance(v, str):
            if v and v.lower() in text_l:
                score += 1
                hits += 1
        elif isinstance(v, list):
            for vv in v:
                if vv and vv.lower() in text_l:
                    score += 1
                    hits += 1
    # Normalize (rough)
    return float(score)


# ---------------------------
# GPT: Evaluate document relevance + extract candidate facts
# ---------------------------
DOC_EVAL_PROMPT = """You are GovSight, a research-grade AI analyst.

User question:
{query}

Known constraints (may include domain, location, time, roles, entities, etc.):
{constraints_json}

Below is content from a single web source (truncated). Assess whether this source helps answer the user question.

Return ONLY valid JSON like:
{{
  "relevance_score": 0-1 float,
  "useful": true/false,
  "key_facts": [short bullet facts relevant to the question],
  "notes": "short diagnostic"
}}
"""

def evaluate_doc_with_gpt(query: str, constraints: Dict[str, Any], doc_text: str, url: str, model: str = None) -> Dict[str, Any]:
    if not model:
        model = DEFAULT_OPENAI_MODEL

    prompt = DOC_EVAL_PROMPT.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
    ) + f"\nSource URL: {url}\n\nContent:\n{doc_text[:4000]}\n\nJSON:"

    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = resp.choices[0].message.content.strip()
        js_start = raw.find("{")
        js_end = raw.rfind("}") + 1
        return json.loads(raw[js_start:js_end])
    except Exception as e:
        logger.warning(f"Doc eval failed for {url}: {e}")
        return {"relevance_score": 0.0, "useful": False, "key_facts": [], "notes": f"eval_error:{e}"}


# ---------------------------
# GPT: Synthesize final answer
# ---------------------------
SYNTH_PROMPT_TMPL = """You are GovSight, a high-accuracy research AI.

User question:
{query}

Constraints:
{constraints_json}

You reviewed multiple web sources. Here are the structured findings:
{findings_json}

Using ONLY supported facts, produce the best possible answer.
- Cite uncertainty where sources conflict.
- Prefer sources with higher relevance_score.
- If the answer is unknown or not confirmed, say so.
Answer:
"""

def synthesize_answer_with_gpt(query: str, constraints: Dict[str, Any], findings: List[Dict[str, Any]], model: str = None) -> str:
    if not model:
        model = DEFAULT_OPENAI_MODEL

    prompt = SYNTH_PROMPT_TMPL.format(
        query=query,
        constraints_json=json.dumps(constraints, indent=2),
        findings_json=json.dumps(findings, indent=2),
    )
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return "I could not confidently synthesize an answer from available sources."


# ---------------------------
# Public entrypoint
# ---------------------------
def answer_from_web(
    query: str,
    constraints: Dict[str, Any],
    top_n: int = 10,
    min_high_conf: int = 3,
    relevance_cutoff: float = 0.7,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. SerpAPI search (top_n results)
      2. Fetch page text
      3. GPT document eval (relevance_score)
      4. Stop once we collect min_high_conf docs >= cutoff (or run out)
      5. Synthesize final answer
      6. Log trace (JSONL)
    Returns dict {answer, sources, confidence, constraints, trace}
    """

    # --- Search + fetch
    raw_results = serp_search_and_fetch(query, engine="google", num=top_n, fetch=True)

    # --- Evaluate & collect
    findings: List[Dict[str, Any]] = []
    high_conf = 0
    for r_ in raw_results:
        doc_eval = evaluate_doc_with_gpt(query, constraints, r_.get("text", ""), r_["url"])
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
                break  # stop early per your spec

    # --- Sort findings by score desc
    findings.sort(key=lambda d: d["relevance_score"], reverse=True)

    # --- Synthesize final answer
    answer_text = synthesize_answer_with_gpt(query, constraints, findings)

    # --- Determine confidence label
    if high_conf >= min_high_conf:
        confidence = "High (web corroborated)"
    elif any(f["relevance_score"] >= relevance_cutoff for f in findings):
        confidence = "Medium (limited corroboration)"
    else:
        confidence = "Low (weak matches)"

    # --- Prepare sources (top 5 shown)
    sources = [
        {
            "title": f["title"],
            "url": f["url"],
            "score": f["relevance_score"],
        }
        for f in findings[:5]
    ]

    # --- Log trace
    _log_web_trace(
        query=query,
        constraints=constraints,
        findings=findings,
        answer=answer_text,
        confidence=confidence,
    )

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "constraints": constraints,
        "trace": findings,  # full detail
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
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        logger.warning(f"Could not log web fallback trace: {e}")
