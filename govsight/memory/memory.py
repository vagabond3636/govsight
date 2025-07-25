from __future__ import annotations

"""
GovSight Memory Layer – Final Version (R2)
============================================

Features:
- Subject–attribute–value triplet support
- Versioned memory with persistence
- Dynamic slugification of arbitrary subjects
- Conversation and session tracking
- Fact recall with NLP parser
"""

import json
import sqlite3
import threading
import re
from typing import List, Optional

from govsight.config import Settings
from . import records
from .schema import bootstrap, get_schema_version, SCHEMA_VERSION
from govsight.parser.fact_parser import parse_fact_from_text
from govsight.utils import slugify  # universal slug generator


class MemoryError(RuntimeError):
    """Raised for unrecoverable memory layer problems."""
    pass


class Memory:
    def __init__(self, settings: Settings, *, readonly: bool = False) -> None:
        self.settings = settings
        self.db_path = settings.db_path
        self.readonly = readonly

        self._lock = threading.RLock()
        self._conn = self._connect()

        bootstrap(self._conn)

        v = get_schema_version(self._conn)
        if v != SCHEMA_VERSION:
            pass  # Schema migration logic can go here

    def _connect(self) -> sqlite3.Connection:
        flags = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        if self.readonly:
            path = f"file:{self.db_path}?mode=ro"
            return sqlite3.connect(path, detect_types=flags, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, detect_types=flags, uri=False, check_same_thread=False)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------ Session Management ------------------

    def start_session(self, profile: str = "dev", notes: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO sessions(profile, notes) VALUES(?, ?)", (profile, notes))
            self._conn.commit()
            return cur.lastrowid

    def log_message(self, session_id: int, role: str, content: str, tokens: Optional[int] = None, turn_index: Optional[int] = None) -> int:
        with self._lock:
            if turn_index is None:
                cur = self._conn.cursor()
                cur.execute("SELECT COALESCE(MAX(turn_index), -1) + 1 FROM messages WHERE session_id=?", (session_id,))
                (turn_index,) = cur.fetchone()

            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages(session_id, turn_index, role, content, tokens) VALUES(?,?,?,?,?)",
                (session_id, turn_index, role, content, tokens),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_messages(self, session_id: int) -> List[dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT turn_index, role, content FROM messages WHERE session_id=? ORDER BY turn_index ASC", (session_id,))
            rows = cur.fetchall()
            return [{"turn_index": r[0], "role": r[1], "content": r[2]} for r in rows]

    # ------------------ Fact Storage ------------------

    def store_fact(self, source_text: str, *, session_id: Optional[int] = None) -> Optional[int]:
        """Parses a source text into a fact and stores it in memory."""
        try:
            parsed = parse_fact_from_text(source_text)
            if parsed is None:
                return None

            subject, attribute, value = parsed
            subject_slug = slugify(subject)
            attr_slug = slugify(attribute)

            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    INSERT INTO facts (subject, subject_slug, attribute, attr_slug, value, source_text, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (subject, subject_slug, attribute, attr_slug, value, source_text, session_id),
                )
                self._conn.commit()
                return cur.lastrowid
        except Exception as e:
            print(f"[Fact Storage Error] {e}")
            return None

    def search(self, query: str, *, session_id: Optional[int] = None) -> Optional[str]:
        """Attempts to match a known subject + attribute to return its value."""
        try:
            parsed = parse_fact_from_text(query)
            if parsed is None:
                return None

            subject, attribute, _ = parsed
            subject_slug = slugify(subject)
            attr_slug = slugify(attribute)

            with self._lock:
                cur = self._conn.cursor()
                cur.execute(
                    """
                    SELECT value FROM facts
                    WHERE subject_slug = ? AND attr_slug = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (subject_slug, attr_slug),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            print(f"[Fact Search Error] {e}")
            return None
