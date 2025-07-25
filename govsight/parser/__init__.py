from typing import Optional, Tuple, Dict
from .fact_parser import parse_fact_from_text as _regex_parse
from .parser import parse_fact_from_text as _gpt_parse

def parse_fact_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Unified fact parser:
      1) Try regex-based rules.
      2) Fallback to GPT-based extraction.
    Returns a dict with keys:
      subject_type, subject_name, state, attr, value
    or None if neither parser matches.
    """
    # 1) Try fast regex
    d = _regex_parse(text)
    if isinstance(d, dict):
        return d

    # 2) Try GPT parser
    tup = _gpt_parse(text)
    if isinstance(tup, tuple) and len(tup) == 3:
        subject, attr, value = tup
        # Heuristically split subject into name+state if it ends with ", ST"
        parts = subject.split(",")
        name = parts[0].strip().title()
        state = parts[1].strip().upper() if len(parts) > 1 else None
        return {
            "subject_type": "generic",
            "subject_name": name,
            "state": state,
            "attr": attr.strip().lower(),
            "value": value.strip(),
        }

    return None
