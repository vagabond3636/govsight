import sqlite3
import os
from govsight.config.settings import DB_PATH

def ensure_db_and_table():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            attribute TEXT NOT NULL,
            value TEXT NOT NULL,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def search_local_facts(query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT subject, attribute, value FROM facts")
        all_facts = cursor.fetchall()

        query_lower = query.lower()
        for subject, attribute, value in all_facts:
            combined = f"{subject} {attribute} {value}".lower()
            if query_lower in combined:
                return f"{subject} {attribute}: {value}"

        return None
    except Exception as e:
        print(f"[DB Search Error] {e}")
        return None
    finally:
        if conn:
            conn.close()

def upsert_fact(subject: str, attribute: str, value: str, source: str = None):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO facts (subject, attribute, value, source)
            VALUES (?, ?, ?, ?)
        """, (subject, attribute, value, source))
        conn.commit()
    except Exception as e:
        print(f"[DB Insert Error] {e}")
    finally:
        if conn:
            conn.close()

def upsert_embedding(text: str, vector: list[float], source: str, doc_type: str = "text"):
    print(f"[Stub] Embedding upsert not yet implemented for: {source}")
