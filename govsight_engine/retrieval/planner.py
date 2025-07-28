from typing import Any, Optional
from govsight.retrieval.constraints import extract_constraints, QueryConstraints
from govsight.retrieval.structured import get_structured_fact
from govsight.retrieval.semantic import semantic_search
from govsight.retrieval.web_search import web_fallback
from govsight.memory.records import FactRecord

def retrieve(text: str) -> Optional[FactRecord]:
    """
    Top‑level retrieval cascade:
      1) extract constraints
      2) structured lookup
      3) semantic search
      4) web fallback
      5) return first good hit
    """
    qc: QueryConstraints = extract_constraints(text)

    # 2) Structured DB
    fact = get_structured_fact(qc)
    if fact:
        print("[planner] Found structured fact:", fact)
        return fact

    # 3) Semantic Memory/Pinecone
    sem_hits = semantic_search(text, metadata_filters={
        "subject_slug": f"{qc.subject_name.lower().replace(' ', '_')}_{qc.state.lower()}"
    } if qc.state else None)
    if sem_hits:
        print("[planner] Semantic hits:", sem_hits)
        # pick best sem_hits[0] and wrap into FactRecord if desired
        return None  # placeholder

    # 4) Live Web/API
    web_hits = web_fallback(text)
    if web_hits:
        print("[planner] Web/API hits:", web_hits)
        # pick best web_hits[0] and wrap into FactRecord if desired
        return None  # placeholder

    return None
