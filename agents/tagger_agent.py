import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

# Load API keys from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# Topic tagging logic
def tag_topics(title, summary):
    prompt = f"""
You are a legislative policy assistant. Given the title and summary of a U.S. Congressional bill, return 1 to 3 concise topic tags from the following list:

["infrastructure", "transportation", "roads and bridges", "public transit", "water systems", "stormwater", "wastewater", "drinking water", "broadband", "public safety", "emergency services", "disaster relief", "flood mitigation", "wildfire response", "community resilience", "housing", "affordable housing", "homelessness", "tax policy", "grants and funding", "appropriations", "economic development", "main street revitalization", "zoning and land use", "environment", "climate change", "energy", "renewable energy", "air quality", "education", "K-12 schools", "school infrastructure", "after-school programs", "childcare", "early childhood education", "higher education", "workforce development", "job training", "labor", "small business", "procurement", "rural development", "veterans", "senior services", "public health", "mental health", "healthcare", "Medicaid", "opioid crisis", "crime and policing", "justice reform", "fire departments", "cybersecurity", "technology", "digital equity", "finance", "banking", "insurance", "civil rights", "elections", "immigration", "agriculture", "parks and recreation", "arts and culture", "historic preservation", "tourism", "foreign affairs", "defense", "military bases", "BRAC (Base Realignment and Closure)"]

Respond ONLY with a JSON array of tag strings.

Title: "{title}"

Summary: "{summary[:1000]}"

Tags:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()
        if result.startswith("[") and result.endswith("]"):
            return eval(result)  # safe here due to trusted structure
    except Exception as e:
        print(f"❌ Tagging failed: {e}")
    return []

# Query for bills
print("🔍 Querying Pinecone for bills to tag...")
query_result = index.query(
    vector=[0.0] * 1536,
    top_k=1000,
    include_metadata=True,
    filter={"source": "congress.gov"}
)

matches = query_result["matches"]
print(f"📦 Found {len(matches)} vectors to evaluate.\n")

# Tag each bill
for match in tqdm(matches, desc="🏷️ Tagging topics"):
    try:
        vector_id = match["id"]
        meta = match["metadata"]

        if "topics" in meta:
            continue  # already tagged

        title = meta.get("title", "")
        summary = meta.get("summary", "")
        if not summary:
            continue

        tags = tag_topics(title, summary)
        if tags:
            index.update(id=vector_id, set_metadata={"topics": tags})
            print(f"✅ {vector_id} tagged: {tags}")
        else:
            print(f"➖ No tags returned for {vector_id}")

    except Exception as e:
        print(f"❌ Error tagging {match['id']}: {e}")
