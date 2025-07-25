import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "memory.db")

def search_local_facts(query):
    """
    Search the local SQLite database (memory.db) for any fact that matches the query.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # You can adjust this based on your actual table and column names
        cursor.execute("SELECT subject, attribute, value FROM facts")
        all_facts = cursor.fetchall()

        # Perform a simple fuzzy search through all stored facts
        query_lower = query.lower()
        for subject, attribute, value in all_facts:
            combined = f"{subject} {attribute} {value}".lower()
            if query_lower in combined:
                return f"{subject} {attribute}: {value}"

        return None  # Nothing matched
    except Exception as e:
        print(f"[DB Search Error] {e}")
        return None
    finally:
        if conn:
            conn.close()

def insert_fact(subject, attribute, value):
    """
    Insert a new fact into the local database for persistent storage.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO facts (subject, attribute, value) VALUES (?, ?, ?)",
            (subject, attribute, value)
        )
        conn.commit()
    except Exception as e:
        print(f"[DB Insert Error] {e}")
    finally:
        if conn:
            conn.close()
