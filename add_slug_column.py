import sqlite3
import os

DB_PATH = os.path.join("govsight", "data", "memory.db")  # Adjust if your DB is in a different path

def add_slug_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE facts ADD COLUMN slug TEXT")
        print("✅ 'slug' column added.")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("✅ 'slug' column already exists.")
        else:
            print("❌ Error adding 'slug' column:", e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    add_slug_column()
