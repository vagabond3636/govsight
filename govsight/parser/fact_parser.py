# govsight/parser/fact_parser.py

import re
from typing import Optional, Dict

# Bring in your GPT-based parser as a fallback
from .parser import parse_fact_from_text as gpt_parse

def parse_fact_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Universal fact parser for:
      • Declarative assertions: "The <attr> of <subject> is <value>."
      • Interrogative questions: "What is the <attr> of <subject>?"
      • Other free‑form patterns via GPT fallback.

    Returns a dict:
    {
      'subject':   "<subject>",   # e.g. "Grandview, TX" or "sky"
      'attr':      "<attribute>", # e.g. "mayor", "color", "temperature"
      'value':     "<value>"       # e.g. "Bill Houston" or None for questions
    }
    or None if nothing parseable.
    """
    text_stripped = text.strip()

    # 1) Declarative: "The <attr> of <subject> is <value>."
    m = re.match(
        r'(?i)the\s+(?P<attr>[\w\s]+?)\s+of\s+(?P<subject>[\w\s,]+?)\s+is\s+(?P<value>.+)',
        text_stripped
    )
    if m:
        return {
            "subject": m.group("subject").strip(),
            "attr":    m.group("attr").strip().lower(),
            "value":   m.group("value").strip(),
        }

    # 2) Interrogative: "What is the <attr> of <subject>?"
    m = re.match(
        r'(?i)what\s+is\s+the\s+(?P<attr>[\w\s]+?)\s+of\s+(?P<subject>[\w\s,]+?)\?',
        text_stripped
    )
    if m:
        return {
            "subject": m.group("subject").strip(),
            "attr":    m.group("attr").strip().lower(),
            "value":   None,
        }

    # 3) Interrogative: "Who is the <attr> of <subject>?"
    m = re.match(
        r'(?i)who\s+is\s+the\s+(?P<attr>[\w\s]+?)\s+of\s+(?P<subject>[\w\s,]+?)\?',
        text_stripped
    )
    if m:
        return {
            "subject": m.group("subject").strip(),
            "attr":    m.group("attr").strip().lower(),
            "value":   None,
        }

    # 4) Fallback to GPT-based parser for anything else
    triple = gpt_parse(text_stripped)
    if isinstance(triple, tuple) and len(triple) == 3:
        subj, attr, val = triple
        return {
            "subject": subj.strip(),
            "attr":    attr.strip().lower(),
            "value":   val.strip(),
        }

    return None
