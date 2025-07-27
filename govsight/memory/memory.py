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
- Fact insertion via batch triples w/ provenance (NEW)
"""

import json
import sqlite3
import threading
import re
import time
from typing import List, Optional

from govsight.config import Settings
from . import records
from .schema import bootstrap, get_schema_version, SCHEMA_VERSION
from govsight.parser.fact_parser import parse_fact_from_text
from govsight.utils.slugify import slugify



class MemoryError(RuntimeError):
    """Raised for unrecoverable memory layer problems."""
    pass


class Memory:
    def __init__(self, settings: Settings, *, readonly: bool = False) -> None:
        self.settings = settings
        self.db_path = settings.db_path
        self.readonly = readonly

        self._lock = threading.Lock()
        with self._lock:
            self.conn = sqlite3.connect(self.db_path)
            bootstrap(self.conn)

    def _connect(self):
        if not hasattr(self, 'conn') or self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start_session(self, session_id: str) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT OR IGNORE INTO sessions (id, started_at)
                VALUES (?, ?)
            """, (session_id, int(time.time())))

    def log_message(self, session_id: str, role: str, message: str) -> None:
        with self.conn:
            self.conn.execute("""
                INSERT INTO messages (session_id, role, message, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, message, int(time.time())))

    def get_messages(self, session_id: str) -> list:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, message FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        return cursor.fetchall()

    def store_fact(self, subject: str, attribute: str, value: str,
                   source: Optional[str] = None, confidence: Optional[float] = None) -> None:
        slug = slugify(subject)
        with self.conn:
            self.conn.execute("""
                INSERT INTO facts (subject, slug, attribute, value, source, confidence, inserted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (subject, slug, attribute, value, source, confidence, int(time.time())))

    def insert_fact_triples(self, triples: List[dict]) -> None:
        for triple in triples:
            self.store_fact(
                subject=triple.get("subject"),
                attribute=triple.get("attribute"),
                value=triple.get("value"),
                source=triple.get("source"),
                confidence=triple.get("confidence")
            )

    def search(self, query: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT subject, attribute, value FROM facts
            WHERE subject LIKE ? OR attribute LIKE ? OR value LIKE ?
            ORDER BY inserted_at DESC LIMIT 1
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        result = cursor.fetchone()
        if result:
            subject, attribute, value = result
            return f"{subject} → {attribute}: {value}"
        return None
