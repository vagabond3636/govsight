import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import re
from datetime import datetime, timedelta

# Load keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

def embed(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# 🧠 Count all bills by metadata
def count_all_bills():
    stats = index.describe_index_stats()
    count = stats["total_vector_count"]
    return count

# 🧠 Count only congress.gov bills
def count_recent_bills(days=30):
    threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    query_result = index.query(
        vector=[0.0]*1536,
        top_k=1000,
        include_metadata=True,
        filter={"source": "congress.gov"}
    )
    count = 0
    for match in query_result["matches"]:
        meta = match["metadata"]
        updated = meta.get("lastUpdated")
        if updated and updated > threshold_date:
            count += 1
    return count

# 🧠 Detect count-type question
def is_count_question(question):
    keywords = ["how many", "number of", "total", "introduced lately", "bills introduced"]
    return any(k in question.lower() for k in keywords)

# 💬 Ask user
user_input = input("Ask your question: ")

if is_count_question(user_input):
    print("\n🧠 Interpreting as a count-type query...")
    days_match = re.search(r"last (\d+)\s*days", user_input)
    if "all" in user_input or "total" in user_input:
        count = count_all_bills()
        print(f"\n📊 You currently have {count} total vectors in Pinecone.")
    else:
        days = int(days_match.group(1)) if days_match else 30
        count = count_recent_bills(days=days)
        print(f"\n📈 {count} bills have been introduced or updated in the last {days} days.")
else:
    # Default: semantic vector + GPT
    query_vector = embed(user_input)
    query_results = index.query(
        vector=query_vector,
        top_k=10,
        include_metadata=True
    )

    context_chunks = []
    for match in query_results["matches"]:
        meta = match["metadata"]
        snippet = f"Title: {meta.get('title', '')}\nSummary: {meta.get('summary', '')}\nLatest Action: {meta.get('latestAction', '')}"
        context_chunks.append(snippet)

    context = "\n\n---\n\n".join(context_chunks)

    # Load your profile
    profile = index.fetch(ids=["user_profile_mike"]).vectors["user_profile_mike"].metadata

    prompt = f"""
You are helping Mike Lane, a federal lobbyist focused on cities, infrastructure, and appropriations.

Prioritize clarity, brevity, and relevance to mayors, city managers, or congressional staff.

Here are relevant documents:
{context}

Question:
{user_input}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600
    )

    print("\n🤖 AI Answer:\n")
    print(response.choices[0].message.content.strip())
