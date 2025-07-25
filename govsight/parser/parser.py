# govsight/parser/parser.py

"""
Parser Module – Extracts structured facts from natural language messages.
This uses GPT-based reasoning to identify subject–attribute–value triples
dynamically, making the system scalable to thousands of fact types.
"""

import openai
from govsight.config.settings import Settings
from typing import Optional, Tuple

# Load settings (e.g. model, temperature)
settings = Settings()
openai.api_key = settings.openai_api_key


def parse_fact_from_text(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Uses the OpenAI chat completions API to extract a (subject, attribute, value)
    triple from a sentence. Returns None if nothing extractable.
    """
    system_prompt = (
        "You are a fact parser. Extract the main subject, attribute, and value "
        "from this message. Output ONLY a tuple like this: (subject, attribute, value)."
    )

    user_prompt = f"Message: {text}\nWhat is the (subject, attribute, value)?"

    try:
        response = openai.chat.completions.create(
            model=settings.model,
            temperature=0.1,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ]
        )

        # New API returns .choices, each with a .message.content
        result = response.choices[0].message.content.strip()

        # Expect a tuple string like "(Alice, title, CEO)"
        if result.startswith("(") and result.endswith(")"):
            triple = eval(result)  # constrained eval
            if isinstance(triple, tuple) and len(triple) == 3:
                return tuple(str(x).strip() for x in triple)

    except Exception as e:
        print(f"[parser] Error extracting fact: {e}")

    return None
