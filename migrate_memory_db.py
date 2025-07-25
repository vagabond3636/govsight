# migrate_memory_db.py

"""
One-time migration: upgrade legacy memory.db to R1 schema.

Steps:
1. Detect existing `facts` table structure.
2. If it does NOT match R1 schema, rename to `facts_legacy`.
3. Instantiate Memory() to bootstrap new schema.
4. Optional: Port rows from legacy format.

*** BACK UP data/memory.db before running! ***
"""

import os
import shutil
import sqlite3

from govsight.config import load_settings
from govsight.memory import Memory

BACKUP = 'data/memory.db.bak'
LEGACY_TABLE = 'facts_legacy'

s = load_settings()
print('Using DB:', s.db_path)

# 1. Backup
if not os.path.exists(BACKUP):
    shutil.copyfile(s.db_path, BACKUP)
    print('Backup created ->', BACKUP)
else:
    print('Backup already exists; not overwriting.')

conn = sqlite3.connect(s.db_path)
cur = conn.cursor()

# 2. Inspect existing facts schema
cur.execute("PRAGMA table_info(facts)")
cols = [r[1] for r in cur.fetchall()]
print('Existing facts columns:', cols)

REQUIRED = {"subject_type","subject_slug","attr","value","source","confidence","status","provenance","latest"}
if not REQUIRED.issubset(set(cols)):
    print('Legacy facts schema detected; renaming table...')
    cur.execute("ALTER TABLE facts RENAME TO %s" % LEGACY_TABLE)
    conn.commit()
else:
    print('Facts table already R1-compatible; no rename needed.')

# 3. Close legacy conn before bootstrap
conn.close()

# 4. Bootstrap new schema (will create fresh tables)
mem = Memory(s)
print('R1 schema ensured.')

# 5. Optional: Port rows (disabled by default)
# TODO: If desired, read from facts_legacy and insert with mem.remember_fact(...)

print('Migration complete.')
