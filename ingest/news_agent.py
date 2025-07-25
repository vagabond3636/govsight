import os
import json
import feedparser
import trafilatura
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from datetime import datetime

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gov-index")

log_path = "news_ingestion_log.json"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log_data = json.load(f)
else:
    log_data = []

rss_url = "https://news.google.com/rss/search?q=Derek+Tran&hl=en-US&gl=US&ceid=US:en"
feed = feedparser.parse(rss_url)

print(f"🔎 Found {len(feed.entries)} articles.")

for entry in feed.entries[:10]:  # adjust as needed
    try:
        article_id = entry.link
        title = entry.title

        print(f"📰 Fetching: {title}")

        # Skip if already logged as successful
        if any(log.get("id") == article_id and log.get("status") == "success" for log in log_data):
            print("🔁 Already processed, skipping.\n")
            continue

        # Extract article text
        downloaded = trafilatura.fetch_url(article_id)
        text = trafilatura.extract(downloaded) if downloaded else ""
        text = text.strip() if text else ""

        if not text or len(text) < 200:
            print("⚠️ Skipped (too short or empty).\n")
            log_data.append({
                "id": article_id,
                "title": title,
                "status": "skipped",
                "reason": "too short",
                "timestamp": datetime.utcnow().isoformat()
            })
            continue

        print(f"🧾 Extracted {len(text)} characters.")
        print("🧠 Preview:", text[:200].replace("\n", " "), "\n")

        # Embed and upsert
        embed_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = embed_response.data[0].embedding

        upsert_response = index.upsert(vectors=[{
            "id": article_id,
            "values": embedding,
            "metadata": {
                "source": "news",
                "title": title,
                "url": article_id,
                "entity": "Derek Tran",
                "type": "news",
                "text": text[:3000]
            }
        }])

        print(f"✅ Stored in Pinecone: {upsert_response}\n")

        # Log success
        log_data.append({
            "id": article_id,
            "title": title,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print(f"❌ Failed: {e}\n")
        log_data.append({
            "id": entry.link,
            "title": entry.title,
            "status": "error",
            "reason": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

# Save log file
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)
