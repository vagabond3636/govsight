import os
import time
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime

# === Load API Keys ===
load_dotenv()
congress_key = os.getenv("CONGRESS_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if not congress_key:
    print("❌ ERROR: CONGRESS_API_KEY not found.")
    exit()

# === Initialize Services ===
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("gov-index")

# === Constants ===
BASE_URL = "https://api.congress.gov/v3"
HEADERS = {"X-API-Key": congress_key}
LIMIT = 50
MAX_BILLS_PER_CONGRESS = 2000
CONGRESS_LIST = [116, 117, 118]

# === Helpers ===
def get_bill_refs(congress, offset=0):
    url = f"{BASE_URL}/bill/{congress}?limit={LIMIT}&offset={offset}"
    res = requests.get(url, headers=HEADERS, timeout=15)
    if res.status_code != 200:
        print(f"❌ Error fetching list at offset {offset} (Congress {congress}): {res.status_code}")
        return []
    return res.json().get("bills", [])

def fetch_detail(url):
    res = requests.get(url, headers=HEADERS, timeout=15)
    if res.status_code != 200:
        print(f"⚠️ Detail fetch failed: {url}")
        return {}
    return res.json().get("bill", {})

def fetch_summary(congress, bill_type, bill_id):
    url = f"{BASE_URL}/bill/{congress}/{bill_type}/{bill_id}/summary"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 200:
            summaries = res.json().get("summaries", [])
            if summaries:
                return summaries[0].get("text", "No summary available.")
        return "No summary available."
    except Exception as e:
        print(f"⚠️ Summary error: {e}")
        return "No summary available."

def infer_topic(text):
    prompt = f"What is the main topic of this congressional bill?\n\nText:\n\"{text[:1000]}\"\n\nRespond with one or two keywords only."
    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20
        )
        return result.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"❌ Topic inference failed: {e}")
        return "unknown"

# === Start Loop ===
for CONGRESS_NUM in CONGRESS_LIST:
    total_ingested = 0
    skipped = 0
    offset = 0

    print(f"\n🚀 Starting bill ingestion for Congress {CONGRESS_NUM}...\n")

    while total_ingested < MAX_BILLS_PER_CONGRESS:
        refs = get_bill_refs(CONGRESS_NUM, offset)
        if not refs:
            print("✅ No more bills to process.")
            break

        for ref in refs:
            if total_ingested >= MAX_BILLS_PER_CONGRESS:
                break

            url = ref.get("url")
            if not url:
                continue

            bill = fetch_detail(url)
            if not bill:
                continue

            bill_id = bill.get("number", "unknown")
            bill_type = bill.get("type", "unknown")
            title = bill.get("title", "No Title")
            sponsor = bill.get("sponsor", {}).get("fullName", "Unknown")

            # Safe cosponsor extraction
            raw_cosponsors = bill.get("cosponsors", [])
            if isinstance(raw_cosponsors, list):
                cosponsors = [p.get("fullName", "") for p in raw_cosponsors if isinstance(p, dict)]
            else:
                cosponsors = []

            latest_action = bill.get("latestAction", {}).get("text", "")
            vector_id = f"{CONGRESS_NUM}_{bill_type}_{bill_id}"

            # Deduplication check
            existing = index.fetch(ids=[vector_id])
            if len(existing.vectors) > 0:
                print(f"🔁 Skipping existing: {title}")
                skipped += 1
                continue

            summary = fetch_summary(CONGRESS_NUM, bill_type, bill_id)
            content = f"{title}\n\nSponsor: {sponsor}\n\nLatest Action: {latest_action}\n\nSummary:\n{summary}"

            print(f"🧾 [{total_ingested + 1}] {title}")
            print(f"📄 Preview: {content[:120].replace(chr(10), ' ')}...")

            try:
                embed = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=content
                )
                embedding = embed.data[0].embedding

                topic = infer_topic(content)

                index.upsert(vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": "congress.gov",
                        "type": "bill",
                        "entity": sponsor,
                        "cosponsors": cosponsors,
                        "title": title,
                        "summary": summary[:1000],
                        "bill_id": bill_id,
                        "bill_type": bill_type,
                        "congress": str(CONGRESS_NUM),
                        "date": datetime.utcnow().isoformat(),
                        "topic": topic
                    }
                }])

                print(f"✅ Stored: {vector_id}\n")
                total_ingested += 1
                time.sleep(1)

            except Exception as e:
                print(f"❌ Embedding error: {e}\n")

        offset += LIMIT
        time.sleep(1)

    print(f"🎯 Congress {CONGRESS_NUM} — Stored: {total_ingested}, Skipped: {skipped}")

print("\n🎉 All done!")
