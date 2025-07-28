from __future__ import annotations

"""Schema bootstrap + migrations for the GovSight memory database."""

import sqlite3
from typing import Optional

# Increment whenever schema changes (use migrations when >1)
SCHEMA_VERSION = 1

# -- DDL statements ---------------------------------------------------------
# Executed in order during bootstrap(). Safe to call repeatedly.
SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        profile TEXT,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
        turn_index INTEGER,
        role TEXT,
        content TEXT,
        tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_type TEXT,
        subject_slug TEXT,
        attr TEXT,
        value TEXT,
        source TEXT,
        confidence REAL,
        status TEXT DEFAULT 'active',
        provenance JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        latest INTEGER DEFAULT 1
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_slug);",
    "CREATE INDEX IF NOT EXISTS idx_facts_attr ON facts(attr);",
    "CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(subject_slug, attr, latest);",
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT,
        sha256 TEXT,
        mime TEXT,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        embedded INTEGER DEFAULT 0,
        meta JSON
    );
    """,
]


def bootstrap(conn: sqlite3.Connection) -> None:
    """Create tables if missing and stamp schema version.

    Safe to call more than once. Uses execscripts in sequence.
    """
    cur = conn.cursor()
    for stmt in SCHEMA_STATEMENTS:
        cur.executescript(stmt)
    # record schema version
    cur.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> Optional[int]:
    """Return current schema version (int) or None if unreadable."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT value FROM schema_meta WHERE key='version'")
        row = cur.fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except sqlite3.Error:
        return None
    return None
