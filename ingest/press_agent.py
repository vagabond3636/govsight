import os
import feedparser
import trafilatura
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load API keys
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gov-index")

# Example: Press release RSS feeds from members of Congress
rss_feeds = {
    "Derek Tran": "https://tran.house.gov/rss.xml",
    "Michelle Steel": "https://steel.house.gov/rss.xml",
    "Katie Porter": "https://porter.house.gov/rss.xml",
    "Ro Khanna": "https://khanna.house.gov/rss.xml",
    "Mike Levin": "https://mikelevin.house.gov/rss.xml",
    "Young Kim": "https://youngkim.house.gov/rss.xml",
    "Adam Schiff": "https://schiff.house.gov/rss.xml",
    "Maxine Waters": "https://waters.house.gov/rss.xml",
    "Judy Chu": "https://chu.house.gov/rss.xml",
    "Kevin McCarthy": "https://kevinmccarthy.house.gov/rss.xml"
}

for name, rss_url in rss_feeds.items():
    print(f"🔍 Scanning press releases for {name}...")
    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        try:
            url = entry.link
            title = entry.title
            published = entry.published if "published" in entry else datetime.utcnow().isoformat()

            # Extract full text from the press release
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded) if downloaded else ""
            text = text.strip()

            if not text or len(text) < 200:
                print(f"⚠️ Skipped: '{title}' (too short)\n")
                continue

            print(f"🧾 Ingesting: {title}")
            preview = text[:200].replace('\n', ' ')
            print(f"📄 Preview: {preview}...\n")

            # Embed the article using OpenAI
            embed = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = embed.data[0].embedding

            # Store the embedding and metadata in Pinecone
            index.upsert(vectors=[{
                "id": url,
                "values": embedding,
                "metadata": {
                    "source": "press_release",
                    "url": url,
                    "title": title,
                    "entity": name,
                    "type": "press",
                    "date": published,
                    "text": text[:3000]
                }
            }])

            print("✅ Stored in Pinecone.\n")

        except Exception as e:
            print(f"❌ Error while processing '{entry.title}': {e}\n")
