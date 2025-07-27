import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    try:
        response = openai.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"🔴 Embedding error: {e}")
        return []


def chat_completion(messages: list, model: str = "gpt-4") -> str:
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"🔴 Chat completion error: {e}")
        return "Error processing request"


def summarize_web_content(content: str, query: str = "") -> str:
    prompt = f"""
You are a government intelligence assistant. Your job is to summarize content scraped from a web page and return relevant facts based on a user query.

User query: "{query}"

Page content:
{content[:4000]}

Return a brief, clear summary of relevant information from this page.
"""
    messages = [
        {"role": "system", "content": "You are a helpful web summarization assistant."},
        {"role": "user", "content": prompt}
    ]
    return chat_completion(messages)
