import sqlite3

conn = sqlite3.connect("govsight/data/memory.db")  # Adjust path if needed
cursor = conn.cursor()

# This line adds the missing column
cursor.execute("ALTER TABLE facts ADD COLUMN attr_slug TEXT;")

conn.commit()
conn.close()

print("✅ Column 'attr_slug' added to facts table.")
