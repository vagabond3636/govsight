import sqlite3
from govsight.config.settings import DB_PATH

def add_missing_columns():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(facts)")
    columns = [col[1] for col in cursor.fetchall()]

    def add_column_if_missing(name: str, coltype: str):
        if name not in columns:
            print(f"🔧 Adding '{name}' column to 'facts' table...")
            cursor.execute(f"ALTER TABLE facts ADD COLUMN {name} {coltype}")
            conn.commit()
        else:
            print(f"✅ '{name}' column already exists.")

    add_column_if_missing("subject", "TEXT")
    add_column_if_missing("attribute", "TEXT")
    add_column_if_missing("value", "TEXT")
    add_column_if_missing("inserted_at", "DATETIME")

    conn.close()
    print("✅ All required columns checked and added if needed.")

if __name__ == "__main__":
    add_missing_columns()
