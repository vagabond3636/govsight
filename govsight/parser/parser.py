"""
parser.py

This module defines LLM-powered parsing functions to extract structured facts
and intent from natural language user inputs.
"""

from typing import Optional, Tuple, Dict
from govsight.config.settings import settings
from govsight.llm.openai_wrapper import chat_completion
import json

print(f"DEBUG: settings from parser.py – {dir(settings)}")


def parse_fact_from_text(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Use LLM to extract a subject, attribute, and value from a sentence.

    Returns:
        Tuple (subject, attribute, value) or None if not found.
    """
    prompt = f"""
You are a fact extractor. Given a user input, extract a fact in the form:
Subject, Attribute, Value.

Example:
Input: "The mayor of Grandview, TX is Tommy Brandt"
Output: Subject: Grandview, TX | Attribute: mayor | Value: Tommy Brandt

Input: "{text}"
Output:
"""

    messages = [
        {"role": "system", "content": "Extract a single fact from the input in the form: Subject, Attribute, Value."},
        {"role": "user", "content": prompt}
    ]

    print(f"🧪 Using model: {settings.openai_model}")
    response = chat_completion(messages, model=settings.openai_model)

    if not response:
        return None

    try:
        lines = response.strip().split("|")
        if len(lines) == 3:
            subject = lines[0].split(":", 1)[1].strip()
            attribute = lines[1].split(":", 1)[1].strip()
            value = lines[2].split(":", 1)[1].strip()
            return subject, attribute, value
    except Exception as e:
        print(f"[⚠️ parse_fact_from_text error] {e}")
        return None

    return None


def parse_intent_and_facts(text: str) -> Dict:
    """
    Use LLM to classify user input intent and extract any known facts.

    Returns:
        dict with fields:
            - intent: "ask_question", "provide_fact", "chat", etc.
            - subject, attribute, value (if found)
    """
    prompt = f"""
You are an AI assistant that extracts user intent and optional structured facts.
From the user input, identify:

- intent: one of "ask_question", "provide_fact", "chat"
- subject: entity being discussed (if any)
- attribute: the attribute or field of interest (if any)
- value: the value of the attribute (if provided)

Format your response as a JSON object.

User input: "{text}"
"""

    messages = [
        {"role": "system", "content": "Extract user intent and any structured fact (subject, attribute, value)."},
        {"role": "user", "content": prompt}
    ]

    print(f"🧠 Parsing input: {text}")
    response = chat_completion(messages, model=settings.openai_model)

    if not response:
        print("[⚠️ parse_intent_and_facts] No response from LLM")
        return {"intent": "chat", "subject": None, "attribute": None, "value": None}

    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return parsed
        else:
            print(f"[⚠️ parse_intent_and_facts] Non-dict response: {parsed}")
    except Exception as e:
        print(f"[⚠️ JSON decode error] {e}\nRaw response: {response}")

    return {"intent": "chat", "subject": None, "attribute": None, "value": None}
