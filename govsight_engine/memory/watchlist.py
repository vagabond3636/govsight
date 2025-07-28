from __future__ import annotations

"""
GovSight Watchlist Tracker
Tracks subjects that should be monitored (e.g., bills, cities, officials)
"""

import sqlite3
import time
from govsight.config import settings

WATCHLIST_DB = settings.db_path


def track_subject(subject: str, reason: str = "") -> None:
    conn = sqlite3.connect(WATCHLIST_DB)
    with conn:
        conn.execute("""
            INSERT INTO watchlist (subject, reason, added_at)
            VALUES (?, ?, ?)
        """, (subject, reason, int(time.time())))
    conn.close()


def get_watchlist() -> list:
    conn = sqlite3.connect(WATCHLIST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT subject, reason, added_at FROM watchlist")
    items = cursor.fetchall()
    conn.close()
    return items
