from __future__ import annotations

"""
GovSight Fact Parser – Dynamic Triple Extraction
"""

import re
from typing import List, Dict
from govsight.utils.slugify import slugify


def parse_fact_from_text(text: str) -> List[Dict]:
    """
    Extract subject–attribute–value facts from plain text.
    Returns a list of dicts: [{subject, attribute, value}]
    """
    triples = []

    # Basic pattern matcher (can be upgraded with NLP later)
    # Example: "The mayor of Grandview is John Smith"
    pattern = re.compile(r"The (.*?) of (.*?) is (.*?)\\.", re.IGNORECASE)
    matches = pattern.findall(text)

    for match in matches:
        attribute, subject, value = match
        triples.append({
            "subject": subject.strip(),
            "attribute": attribute.strip(),
            "value": value.strip(),
            "source": "text-parser",
            "confidence": 0.6
        })

    return triples
