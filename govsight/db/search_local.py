import sqlite3
from typing import Optional

def search_local_facts(db_path: str, subject: str, attribute: Optional[str] = None) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if attribute:
        cursor.execute("""
            SELECT value FROM facts
            WHERE subject = ? AND attribute = ?
            ORDER BY inserted_at DESC
            LIMIT 1
        """, (subject.lower(), attribute.lower()))
    else:
        cursor.execute("""
            SELECT value FROM facts
            WHERE subject = ?
            ORDER BY inserted_at DESC
            LIMIT 1
        """, (subject.lower(),))

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None
