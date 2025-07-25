**GovSight Dev Session Restart Notes (as of July 25, 2025)**

---

### Overall Context

You (Mike) are building a persistent RAG system called **GovSight** to replicate ChatGPT-like capabilities for municipal lobbying, using:

- **Local facts** (SQLite via `memory.db`)
- **Pinecone vector search**
- **Web fallback search**

The system has a CLI (`gs`) and uses cascading logic: Local DB → Pinecone → Web. You're also transitioning everything to a **modular architecture** with subfolders like `govsight/llm`, `govsight/vector`, etc. Your priorities are:

- Persistent memory recall (factual Q&A)
- Dynamic fact parsing (subject, attribute, value triples)
- Resilient OpenAI API compatibility
- Web fallback when facts aren't known

You want to eliminate redundant code and streamline imports using the new modular structure.

---

### Files Currently Migrated / In Progress

#### 1. `govsight/llm/llm.py` ✅

- Fully updated to OpenAI SDK v1.
- Exposes `chat_completion`, `get_embedding`, `summarize_web_content`

#### 2. `govsight/web_reasoner.py` ✅

- Uses `summarize_web_content(query)`
- Receives `query` cleanly from fallback in `talk.py`
- Previous version had 71 lines but included some duplication and bad param calls. Final version is now fully lean.

#### 3. `talk.py` ✅

- Fully rewritten and simplified.
- No more legacy echo errors or repeat prompting.
- Uses cascading logic: Memory → Pinecone → Web.

#### 4. `govsight/memory/memory.py` ⚠️

- Current error: `.search()` works, but had a column mismatch (`attr_slug`) and OpenAI SDK incompatibility.
- **Memory DB schema** has now been patched manually.

#### 5. `govsight/vector/search.py` ⚠️

- Still using old `pinecone.init()` ❌
- Must migrate to new client:

```python
from pinecone import Pinecone
pc = Pinecone(api_key=...)  # and then use `pc.Index(...)`
```

---

### Outstanding Issues

#### ❌ OpenAI SDK breaking changes (Chat + Embedding)

- **FIXED in **`` but you need to make sure **all references throughout repo** use the new SDK method:

```python
from openai import OpenAI
client = OpenAI()
client.chat.completions.create(...)
client.embeddings.create(...)
```

#### ❌ `govsight.config.settings` missing

- You tried to delete this file, but it was still imported.
- We replaced it with a simplified `config.py`, but it must expose a `settings` object that contains:
  - `db_path`
  - `openai_key`
  - `pinecone_api_key`

#### ⚠️ `readline` import crash on Windows

- Remove or wrap in platform check:

```python
try:
    import readline
except ImportError:
    pass  # Windows
```

#### ⚠️ `.env` keys

- Keys are present and valid
- Final system will rotate these before production

---

### Next Steps After Break

1. **Upload next file to migrate**:
   - `parser.py`, `fact_parser.py`, or `search.py`
2. **Verify **``** and settings import** work across all files
3. **Fix Pinecone integration** with latest SDK
4. **Test: who is the mayor of grandview, tx?** end-to-end again

---

### Final Note

All final patched files must be pasted **in full**. You've requested this repeatedly due to cutoffs. Continue enforcing that every change yields a complete, copy-pasteable file. I will comply moving forward without exception.

When you're back, upload the next file, or say "resume" and I’ll recall this context.

