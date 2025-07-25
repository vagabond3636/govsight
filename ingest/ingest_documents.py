import os
import re
from dotenv import load_dotenv
from pathlib import Path
from pinecone import Pinecone
from unstructured.partition.auto import partition
import openai
from datetime import datetime

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gov-index")

# Input folder
doc_folder = Path("documents/deltona")
model = "text-embedding-3-small"

# --- Helper functions ---

def embed_text(text):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def extract_date_from_filename(name):
    match = re.search(r"(20\d{2}[-_\.]?\d{1,2}[-_\.]?\d{1,2})", name)
    if match:
        raw = match.group(1).replace('_', '-').replace('.', '-')
        try:
            return str(datetime.strptime(raw, "%Y-%m-%d").date())
        except:
            try:
                return str(datetime.strptime(raw, "%Y-%m").date())
            except:
                pass
    return None

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def infer_topic(text):
    prompt = f"What is the main topic of this local government text?\n\nText:\n\"{text}\"\n\nRespond with one or two keywords only."
    try:
        result = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return result.choices[0].message.content.strip().lower()
    except Exception as e:
        print("❌ Topic inference failed:", e)
        return "unknown"

def process_file(file_path):
    print(f"📄 Processing: {file_path.name}")
    elements = partition(file_path)
    chunks = [el.text.strip() for el in elements if el.text and len(el.text.strip()) > 50]
    print(f"➡️ {len(chunks)} chunks found.")
    return chunks

# --- Main ingestion logic ---

def ingest_documents():
    for file in doc_folder.iterdir():
        if file.suffix.lower() not in [".pdf", ".docx"]:
            continue

        try:
            chunks = process_file(file)
            date = extract_date_from_filename(file.name) or "unknown"
            for i, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                topic = infer_topic(chunk[:1000])  # Use only the first 1000 characters
                vector_id = f"{file.stem}-{i}"
                index.upsert(vectors=[{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "source": file.name,
                        "chunk": i,
                        "city": "Deltona",
                        "date": date,
                        "topic": topic
                    }
                }])
            print(f"✅ Uploaded: {file.name}")
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    ingest_documents()