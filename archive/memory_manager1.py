"""
memory_manager.py – Persistent conversational + fact memory for GovSight.

Features:
- Auto-create SQLite DB (data/memory.db) if missing.
- Log every conversation turn (user + assistant) with timestamps & session_id.
- Auto-summarize a session into key entities/topics/actions (GPT extracted).
- Extract candidate facts from assistant answers (GPT), store low-confidence facts w/ source & timestamp.
- Maintain watchlist for follow-up/updates when user expresses track/monitor intent.
- Embed conversation turns, session summaries, and facts to Pinecone for semantic recall.

Configuration: requires OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME in config.py.
"""

from __future__ import annotations

import os
import sqlite3
import json
import time
import datetime as dt
from typing import Optional, Dict, Any, List, Tuple

import openai
from pinecone import Pinecone

import config  # import module so we can safely getattr defaults

# ------------------------------------------------------------------
# Config / API keys / model names (with safe fallbacks)
# ------------------------------------------------------------------
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

DEFAULT_OPENAI_MODEL = getattr(config, "DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL  = getattr(config, "DEFAULT_EMBED_MODEL", "text-embedding-3-small")


# ------------------------------------------------------------------
# Paths & init
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "memory.db")

# init openai + pinecone
openai.api_key = OPENAI_API_KEY
_pc = Pinecone(api_key=PINECONE_API_KEY)
_pinecone_index = _pc.Index(PINECONE_INDEX_NAME)


# ------------------------------------------------------------------
# SQLite schema
# ------------------------------------------------------------------
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts TEXT NOT NULL,
    end_ts TEXT,
    summary_text TEXT,
    entities_json TEXT,
    topics_json TEXT,
    actions_json TEXT
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts TEXT NOT NULL,
    role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    entity_type TEXT,
    state TEXT,
    canonical_name TEXT,
    last_touched TEXT
);

CREATE TABLE IF NOT EXISTS session_entities (
    session_id INTEGER NOT NULL,
    entity_id INTEGER NOT NULL,
    PRIMARY KEY (session_id, entity_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER,
    attribute TEXT NOT NULL,
    value TEXT,
    effective_date TEXT,
    source_url TEXT,
    confidence REAL,
    added_ts TEXT NOT NULL,
    source_session_id INTEGER,
    FOREIGN KEY (entity_id) REFERENCES entities(id),
    FOREIGN KEY (source_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    entity_id INTEGER,
    created_ts TEXT NOT NULL,
    last_checked_ts TEXT,
    frequency TEXT DEFAULT 'weekly', -- daily|weekly|monthly
    active INTEGER DEFAULT 1,
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _connect()
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


# Initialize DB on import
_init_db()


# ------------------------------------------------------------------
# Session lifecycle
# ------------------------------------------------------------------
def open_session() -> int:
    """Create a new session row and return session_id."""
    conn = _connect()
    try:
        ts = dt.datetime.utcnow().isoformat()
        cur = conn.execute(
            "INSERT INTO sessions (start_ts) VALUES (?)",
            (ts,),
        )
        session_id = cur.lastrowid
        conn.commit()
        return session_id
    finally:
        conn.close()


def log_turn(session_id: int, role: str, text: str):
    """Log a conversation turn and embed it to Pinecone."""
    conn = _connect()
    try:
        ts = dt.datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO conversation_turns (session_id, ts, role, text) VALUES (?, ?, ?, ?)",
            (session_id, ts, role, text),
        )
        conn.commit()
    finally:
        conn.close()

    # Embed & upsert to Pinecone
    _embed_and_upsert_async_like(
        text=text,
        metadata={
            "type": "turn",
            "role": role,
            "session_id": session_id,
            "ts": ts,
        },
    )


def close_session(session_id: int):
    """
    Mark session end_ts; summarize session; update Pinecone with summary;
    create entities & watchlist items from summary/actions.
    """
    conn = _connect()
    try:
        end_ts = dt.datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE sessions SET end_ts=? WHERE id=?",
            (end_ts, session_id),
        )
        conn.commit()
    finally:
        conn.close()

    summarize_session(session_id)


# ------------------------------------------------------------------
# Session summarization (GPT)
# ------------------------------------------------------------------
_SESSION_SUMMARY_PROMPT = """You are GovSight, an AI memory analyst.

You will be given a chronological series of conversation turns (user & assistant).
Extract a structured memory summary that will help future recall.

Return ONLY valid JSON in this schema:
{{
  "summary": "short narrative of what the session covered",
  "entities": [
     {{"name": "...", "entity_type": "city|person|program|topic|other", "state": "TX|CA|", "confidence": 0-1}}
  ],
  "topics": ["chromium-6", "funding", "Grandview TX"],
  "actions": ["follow up with Coachella", "monitor chromium regs"]
}}

Conversation:
---
{conversation_text}
---
JSON:
"""


def summarize_session(session_id: int):
    """
    Generate a structured summary for the session using GPT,
    store in DB, embed to Pinecone, update entities & watchlist.
    """
    turns = get_session_turns(session_id)
    if not turns:
        return
    convo_txt = "\n".join([f"{t['role']}: {t['text']}" for t in turns])

    prompt = _SESSION_SUMMARY_PROMPT.format(conversation_text=convo_txt[:12000])  # cap tokens

    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        js = _safe_extract_json(raw, {})
    except Exception as e:
        js = {}
        print(f"[memory] Session summary failed: {e}")

    summary_text = js.get("summary", "")
    entities = js.get("entities", [])
    topics = js.get("topics", [])
    actions = js.get("actions", [])

    # write to DB
    conn = _connect()
    try:
        conn.execute(
            "UPDATE sessions SET summary_text=?, entities_json=?, topics_json=?, actions_json=? WHERE id=?",
            (
                summary_text,
                json.dumps(entities),
                json.dumps(topics),
                json.dumps(actions),
                session_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # upsert summary to Pinecone
    _embed_and_upsert_async_like(
        text=summary_text,
        metadata={
            "type": "session_summary",
            "session_id": session_id,
            "entities": json.dumps(entities),
            "topics": json.dumps(topics),
            "actions": json.dumps(actions),
        },
    )

    # create/update entities & watchlist items based on summary
    _apply_session_entities(session_id, entities)
    _apply_session_actions(session_id, actions, entities)


# ------------------------------------------------------------------
# Candidate Fact Extraction from Assistant Turns
# ------------------------------------------------------------------
_FACT_EXTRACT_PROMPT = """You are GovSight, an information extraction AI.

Given the USER question and the ASSISTANT answer, extract factual claims that could be stored in memory.
Return ONLY valid JSON list. Schema:
[
  {{
    "entity_name": "City of Coachella",
    "entity_type": "city|person|program|other",
    "attribute": "chromium-6 project funding status",
    "value": "pending EC-SDC application",
    "state": "CA",
    "effective_date": "YYYY-MM-DD or null",
    "confidence": 0-1,
    "source_url": null
  }}
]

USER:
{user_text}

ASSISTANT:
{assistant_text}

JSON:
"""


def extract_facts_from_turn(user_text: str, assistant_text: str) -> List[Dict[str, Any]]:
    """
    Use GPT to extract structured fact candidates from a user->assistant exchange.
    """
    prompt = _FACT_EXTRACT_PROMPT.format(
        user_text=user_text,
        assistant_text=assistant_text,
    )
    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end <= start:
            return []
        js = json.loads(raw[start:end])
        if not isinstance(js, list):
            return []
        return js
    except Exception as e:
        print(f"[memory] fact extract failed: {e}")
        return []


def store_facts(session_id: int, facts: List[Dict[str, Any]]):
    """Store low-confidence facts in DB + Pinecone."""
    if not facts:
        return
    conn = _connect()
    try:
        for f in facts:
            name = f.get("entity_name") or ""
            entity_id = None
            if name:
                entity_id = _get_or_create_entity(conn, name, f.get("entity_type"), f.get("state"))

            conn.execute(
                "INSERT INTO facts (entity_id, attribute, value, effective_date, source_url, confidence, added_ts, source_session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity_id,
                    f.get("attribute"),
                    f.get("value"),
                    f.get("effective_date"),
                    f.get("source_url"),
                    f.get("confidence", 0.3),
                    dt.datetime.utcnow().isoformat(),
                    session_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # Upsert each fact text to Pinecone
    for f in facts:
        fact_text = f"{f.get('entity_name')}: {f.get('attribute')} = {f.get('value')} (conf {f.get('confidence',0)})"
        _embed_and_upsert_async_like(
            text=fact_text,
            metadata={
                "type": "fact",
                "entity_name": f.get("entity_name"),
                "attribute": f.get("attribute"),
                "value": f.get("value"),
                "state": f.get("state"),
                "confidence": f.get("confidence"),
                "session_id": session_id,
            },
        )


# ------------------------------------------------------------------
# Watchlist handling
# ------------------------------------------------------------------
_WATCHLIST_PROMPT = """You are GovSight. Determine whether the USER wants to TRACK or MONITOR something over time,
based on their query in the context of ASSISTANT reply.

Return ONLY JSON:
{{
  "create_watch": true,
  "topic": "<short topic name or null>",
  "entity_name": "<entity or null>",
  "frequency": "daily|weekly|monthly"
}}

USER:
{user_text}

ASSISTANT:
{assistant_text}

JSON:
"""


def detect_watchlist_from_turn(user_text: str, assistant_text: str) -> Optional[Dict[str, Any]]:
    """
    Use GPT to detect whether user intent implies a watch/monitor track item.
    """
    prompt = _WATCHLIST_PROMPT.format(user_text=user_text, assistant_text=assistant_text)
    try:
        resp = openai.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        js = _safe_extract_json(raw, {})
        return js if isinstance(js, dict) else None
    except Exception:
        return None


def create_watchlist(topic: str, entity_name: Optional[str], frequency: str = "weekly"):
    """Insert a watchlist row."""
    conn = _connect()
    try:
        entity_id = None
        if entity_name:
            entity_id = _get_or_create_entity(conn, entity_name, None, None)
        conn.execute(
            "INSERT INTO watchlist (topic, entity_id, created_ts, frequency, active) VALUES (?, ?, ?, ?, 1)",
            (
                topic,
                entity_id,
                dt.datetime.utcnow().isoformat(),
                frequency,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------
# Entity helpers
# ------------------------------------------------------------------
def _get_or_create_entity(conn, name: str, entity_type: Optional[str], state: Optional[str]) -> int:
    cur = conn.execute("SELECT id FROM entities WHERE name=? COLLATE NOCASE", (name,))
    row = cur.fetchone()
    ts = dt.datetime.utcnow().isoformat()
    if row:
        conn.execute("UPDATE entities SET last_touched=? WHERE id=?", (ts, row["id"]))
        return row["id"]
    cur = conn.execute(
        "INSERT INTO entities (name, entity_type, state, canonical_name, last_touched) VALUES (?, ?, ?, ?, ?)",
        (name, entity_type, state, name, ts),
    )
    return cur.lastrowid


def _apply_session_entities(session_id: int, entities: List[Dict[str, Any]]):
    if not entities:
        return
    conn = _connect()
    try:
        for e in entities:
            name = e.get("name")
            if not name:
                continue
            entity_id = _get_or_create_entity(conn, name, e.get("entity_type"), e.get("state"))
            conn.execute(
                "INSERT OR IGNORE INTO session_entities (session_id, entity_id) VALUES (?, ?)",
                (session_id, entity_id),
            )
        conn.commit()
    finally:
        conn.close()


def _apply_session_actions(session_id: int, actions: List[str], entities: List[Dict[str, Any]]):
    """Create watchlist entries if actions imply tracking."""
    if not actions:
        return
    for act in actions:
        if any(word in act.lower() for word in ("monitor", "track", "watch", "follow up", "updates")):
            ent_name = entities[0]["name"] if entities else None
            create_watchlist(topic=act, entity_name=ent_name, frequency="weekly")


# ------------------------------------------------------------------
# Retrieval helpers
# ------------------------------------------------------------------
def get_session_turns(session_id: int) -> List[sqlite3.Row]:
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT ts, role, text FROM conversation_turns WHERE session_id=? ORDER BY id ASC",
            (session_id,),
        )
        return cur.fetchall()
    finally:
        conn.close()


# ------------------------------------------------------------------
# Embedding & Pinecone Upsert
# ------------------------------------------------------------------
def _embed_and_upsert_async_like(text: str, metadata: Dict[str, Any]):
    """
    Simple embedding + upsert. Synchronous for now (called 'async-like' to mark easy future swap).
    Upsert ID uses time hash.
    """
    if not text or not text.strip():
        return
    try:
        emb = openai.embeddings.create(
            model=DEFAULT_EMBED_MODEL,
            input=[text]
        ).data[0].embedding
    except Exception as e:
        print(f"[memory] embed failed: {e}")
        return

    uid = f"mem-{int(time.time()*1000)}-{abs(hash(text)) % 10_000_000}"
    try:
        _pinecone_index.upsert(
            vectors=[{
                "id": uid,
                "values": emb,
                "metadata": metadata | {"text": text},
            }]
        )
    except Exception as e:
        print(f"[memory] pinecone upsert failed: {e}")


# ------------------------------------------------------------------
# JSON fragment extractor (shared utility)
# ------------------------------------------------------------------
def _safe_extract_json(text: str, default: Any):
    if not text:
        return default
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return default
    try:
        return json.loads(text[start:end+1])
    except Exception:
        return default
