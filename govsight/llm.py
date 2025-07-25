"""
llm.py

This module provides a wrapper for making chat completion requests to OpenAI.
"""

import openai
from govsight.config import settings


def chat_completion(system: str, user: str, model: str = "gpt-4", temperature: float = 0.2) -> str:
    """
    Perform a chat completion using OpenAI's Chat API.

    Args:
        system (str): The system prompt (sets behavior of the assistant).
        user (str): The user input or query.
        model (str): The model to use (default: "gpt-4").
        temperature (float): Sampling temperature (default: 0.2).

    Returns:
        str: The assistant's reply.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"
