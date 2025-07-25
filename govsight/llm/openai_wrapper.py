import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Chat completion wrapper
def chat_completion(system_prompt, user_prompt, model="gpt-4o", temperature=0.3):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Chat Error] {e}")
        return "Sorry, I encountered an issue."

# Embedding wrapper
def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            model=model,
            input=[text],
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

# Optional: used by web_reasoner.py
def summarize_web_content(content, query):
    prompt = f"Summarize this content to help answer the query: '{query}'.\n\n{content}"
    return chat_completion("You are a helpful assistant.", prompt)
