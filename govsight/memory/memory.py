from __future__ import annotations

"""
GovSight Memory Layer – High-Level API
======================================

This module provides a *single* abstraction for persistent memory used across
GovSight. It wraps a SQLite database (path provided via Settings) and exposes
methods for:

    • Session tracking (`start_session`)
    • Transcript logging (`log_message`, `get_recent_messages`)
    • Versioned structured facts (`remember_fact`, `get_fact`, `list_facts`)
    • File registry (`register_file`) for uploaded/ingested documents
    • Subject slug helpers (normalize city/state names, etc.)

Design Goals
------------
- Minimal deps (stdlib only: sqlite3, threading, json, re).
- Non-destructive: bootstrap schema if missing; extend existing DBs safely.
- Versioned facts: new insert marks prior row(s) superseded; history retained.
- Thread-aware: single connection guarded by an RLock; `check_same_thread=False`.
- Transitional: coexists with legacy `memory_manager.py` until all call sites
  migrate.

Usage (quick):
    from govsight.config import load_settings
    from govsight.memory import Memory

    s = load_settings()
    mem = Memory(s)
    sid = mem.start_session(profile=s.profile)
    mem.log_message(sid, "user", "Hello")
    mem.log_message(sid, "assistant", "Hi!")
    slug = mem.subject_slug_city("Grandview", "TX")
    mem.remember_fact(subject_type="city", subject_slug=slug,
                      attr="mayor", value="Bill Houston", source="user")
    fact = mem.get_fact(slug, "mayor")
    print(fact)

See also:
    govsight/memory/schema.py   -> DDL + bootstrap()
    govsight/memory/records.py  -> lightweight record dataclasses
"""

import json
import sqlite3
import threading
import re
from typing import List, Optional

from govsight.config import Settings
from . import records
from .schema import bootstrap, get_schema_version, SCHEMA_VERSION
from govsight.parser.fact_parser import parse_fact_from_text  # rule-based parser


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class MemoryError(RuntimeError):
    """Raised for unrecoverable memory layer problems."""


# ---------------------------------------------------------------------------
# Subject slug helpers
# ---------------------------------------------------------------------------
def _slugify_city_state(name: str, state: Optional[str] = None) -> str:
    """
    Normalize a name (and optional state) to a lowercase underscore slug.

    Examples
    --------
    >>> _slugify_city_state("Grandview", "TX")
    'grandview_tx'
    >>> _slugify_city_state("Coachella", "CA")
    'coachella_ca'
    >>> _slugify_city_state("Generic Name")
    'generic_name'
    """
    parts = [name.strip().lower().replace(" ", "_")]
    if state:
        parts.append(state.strip().lower())
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Memory Facade
# ---------------------------------------------------------------------------
class Memory:
    """
    Persistent memory facade over a SQLite database.

    Parameters
    ----------
    settings:
        GovSight Settings object (provides db_path, etc.).
    readonly:
        If True, database opened in SQLite read-only URI mode.

    Notes
    -----
    * Connection created once per instance; guarded by an RLock.
    * bootstrap() called on init to ensure schema exists.
    * Future migrations will trigger when SCHEMA_VERSION bumps.
    """

    def __init__(self, settings: Settings, *, readonly: bool = False) -> None:
        self.settings = settings
        self.db_path = settings.db_path
        self.readonly = readonly

        self._lock = threading.RLock()
        self._conn = self._connect()

        # Ensure tables exist (idempotent)
        bootstrap(self._conn)

        # Placeholder for future schema migrations
        v = get_schema_version(self._conn)
        if v != SCHEMA_VERSION:
            # TODO: apply migrations when we rev schema
            pass

    # ------------------------------------------------------------------
    # Connection mgmt
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Open and return a sqlite3 connection."""
        flags = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        if self.readonly:
            path = f"file:{self.db_path}?mode=ro"
            return sqlite3.connect(path, detect_types=flags, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, detect_types=flags, uri=False, check_same_thread=False)

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------
    def start_session(self, profile: str = "dev", notes: Optional[str] = None) -> int:
        """Insert a new session row and return its id."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO sessions(profile, notes) VALUES(?, ?)", (profile, notes))
            self._conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Message logging
    # ------------------------------------------------------------------
    def log_message(
        self,
        session_id: int,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        turn_index: Optional[int] = None,
    ) -> int:
        """Append a message turn to the transcript for `session_id`."""
        with self._lock:
            if turn_index is None:
                cur = self._conn.cursor()
                cur.execute(
                    "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE session_id=?",
                    (session_id,),
                )
                (turn_index,) = cur.fetchone()

            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages(session_id, turn_index, role, content, tokens) "
                "VALUES(?,?,?,?,?)",
                (session_id, turn_index, role, content, tokens),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_recent_messages(self, session_id: int, limit: int = 50) -> List[records.MessageRecord]:
        """Return most recent N messages for a session in chronological order."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, session_id, turn_index, role, content, tokens, created_at "
            "FROM messages WHERE session_id=? "
            "ORDER BY turn_index DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
        return [records.MessageRecord(*row) for row in rows][::-1]

    # ------------------------------------------------------------------
    # Fact storage (versioned)
    # ------------------------------------------------------------------
    def remember_fact(
        self,
        *,
        subject_type: str,
        subject_slug: str,
        attr: str,
        value: str,
        source: str = "user",
        confidence: float = 0.9,
        status: str = "pending-verify",
        provenance: Optional[dict] = None,
        latest: int = 1,
    ) -> int:
        """
        Insert a new fact row and mark older rows for same (subject_slug, attr).
        """
        prov_json = json.dumps(provenance) if provenance else None
        with self._lock:
            cur = self._conn.cursor()

            # mark old latest rows inactive
            cur.execute(
                "UPDATE facts "
                "SET latest=0, status='superseded', updated_at=CURRENT_TIMESTAMP "
                "WHERE subject_slug=? AND attr=? AND latest=1",
                (subject_slug, attr),
            )

            # insert new row
            cur.execute(
                "INSERT INTO facts(subject_type, subject_slug, attr, value, source, confidence, status, provenance, latest) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (subject_type, subject_slug, attr, value, source, confidence, status, prov_json, latest),
            )

            self._conn.commit()
            return cur.lastrowid

    def get_fact(self, subject_slug: str, attr: str, *, active_only: bool = True) -> Optional[records.FactRecord]:
        """Return the latest fact for (subject_slug, attr)."""
        cur = self._conn.cursor()
        if active_only:
            cur.execute(
                "SELECT id, subject_type, subject_slug, attr, value, source, confidence, "
                "status, provenance, latest, created_at, updated_at "
                "FROM facts WHERE subject_slug=? AND attr=? AND latest=1 "
                "ORDER BY id DESC LIMIT 1",
                (subject_slug, attr),
            )
        else:
            cur.execute(
                "SELECT id, subject_type, subject_slug, attr, value, source, confidence, "
                "status, provenance, latest, created_at, updated_at "
                "FROM facts WHERE subject_slug=? AND attr=? "
                "ORDER BY id DESC LIMIT 1",
                (subject_slug, attr),
            )
        row = cur.fetchone()
        if not row:
            return None
        return records.FactRecord(*row)

    def list_facts(
        self,
        subject_slug: Optional[str] = None,
        attr: Optional[str] = None,
        active_only: bool = True,
    ) -> List[records.FactRecord]:
        """List facts optionally filtered by subject and/or attr."""
        cur = self._conn.cursor()
        sql = (
            "SELECT id, subject_type, subject_slug, attr, value, source, confidence, "
            "status, provenance, latest, created_at, updated_at FROM facts"
        )
        where, params = [], []
        if subject_slug:
            where.append("subject_slug=?")
            params.append(subject_slug)
        if attr:
            where.append("attr=?")
            params.append(attr)
        if active_only:
            where.append("latest=1")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY subject_slug, attr, id DESC"
        cur.execute(sql, params)
        return [records.FactRecord(*row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # File registry
    # ------------------------------------------------------------------
    def register_file(
        self,
        path: str,
        *,
        sha256: str,
        mime: str,
        embedded: bool = False,
        meta: Optional[dict] = None,
    ) -> int:
        """Register a file and return its id."""
        meta_json = json.dumps(meta) if meta else None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO files(path, sha256, mime, embedded, meta) VALUES(?,?,?,?,?)",
                (path, sha256, mime, int(embedded), meta_json),
            )
            self._conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------
    # Public slug helpers
    # ------------------------------------------------------------------
    @staticmethod
    def subject_slug_city(city: str, state: str) -> str:
        return _slugify_city_state(city, state)

    @staticmethod
    def subject_slug_generic(name: str) -> str:
        return _slugify_city_state(name)

    # ------------------------------------------------------------------
    # Fact recall from text
    # ------------------------------------------------------------------
    def recall_fact_from_text(self, text: str) -> Optional[records.FactRecord]:
        """
        Attempt to extract (subject, attribute, value) from a question and recall from memory.
        Supports both corrections and 'Who is...' queries.
        """
        # 1) Rule-based correction style: "The mayor of X, ST is Y."
        parsed = parse_fact_from_text(text)
        if parsed and isinstance(parsed, dict):
            stype = parsed.get("subject_type")
            name = parsed.get("subject_name")
            st = parsed.get("state")
            attr = parsed.get("attr")
            if stype == "city" and name and st and attr:
                slug = self.subject_slug_city(name, st)
                return self.get_fact(slug, attr)

        # 2) Fallback question regex: "Who is the X of Y, ST?"
        q = re.search(
            r"who\s+is\s+the\s+(\w+)\s+of\s+([\w\s]+),?\s+([A-Za-z]{2})\??",
            text,
            re.IGNORECASE,
        )
        if q:
            attr = q.group(1).strip()
            name = q.group(2).strip()
            st = q.group(3).strip().upper()
            slug = self.subject_slug_city(name, st)
            return self.get_fact(slug, attr)

        return None
