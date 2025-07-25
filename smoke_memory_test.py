"""
GovSight Memory Layer Smoke Test
--------------------------------
Quick sanity check that the new govsight.memory package boots, creates tables,
logs a session + 2 messages, inserts a user fact, and reads it back.
"""

from govsight.config import load_settings
from govsight.memory import Memory

# Load runtime settings (reads env + legacy config.py)
s = load_settings()
print("Settings loaded:", s.asdict())

# Init memory layer (creates tables if needed)
mem = Memory(s)
print("DB path in use:", s.db_path)

# Start a new session
sid = mem.start_session(profile=s.profile)
print("New session id:", sid)

# Log 2 messages into the transcript
mem.log_message(sid, role="user", content="Hello from memory smoke test")
mem.log_message(sid, role="assistant", content="Hello back (memory test)")

# Insert a user correction-style fact
slug = mem.subject_slug_city("Grandview", "TX")
mem.remember_fact(
    subject_type="city",
    subject_slug=slug,
    attr="mayor",
    value="Bill Houston",
    source="user",
)

# Retrieve the fact
fact = mem.get_fact(slug, "mayor")
print("Retrieved fact:", fact)

# Show the last few messages just to prove logging worked
msgs = mem.get_recent_messages(sid, limit=10)
print("Recent messages:")
for m in msgs:
    print("  ", m)
