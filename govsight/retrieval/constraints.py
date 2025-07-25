from dataclasses import dataclass
from typing import Optional, Any, Dict
import re
from govsight.parser.fact_parser import parse_fact_from_text

@dataclass
class QueryConstraints:
    subject_type: Optional[str] = None
    subject_name: Optional[str] = None
    state:        Optional[str] = None
    attribute:    Optional[str] = None
    other:        Dict[str, Any] = None

def extract_constraints(text: str) -> QueryConstraints:
    qc = QueryConstraints(other={})

    # 1) user‑asserted fact
    fact = parse_fact_from_text(text)
    if isinstance(fact, dict):
        qc.subject_type = fact.get("subject_type")
        qc.subject_name = fact.get("subject_name")
        qc.state        = fact.get("state")
        qc.attribute    = fact.get("attr")
        return qc

    # 2) question pattern
    m = re.search(
        r"who\s+is\s+the\s+(\w+)\s+of\s+([\w\s]+),?\s*([A-Za-z]{2})\??",
        text,
        re.IGNORECASE,
    )
    if m:
        qc.attribute    = m.group(1).strip()
        qc.subject_name = m.group(2).strip()
        qc.state        = m.group(3).strip().upper()
        qc.subject_type = "city"
        return qc

    return qc
