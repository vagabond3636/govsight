import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("gov-index")

# Get index stats
stats = index.describe_index_stats()
total = stats["total_vector_count"]
print(f"\n📦 Total vectors in Pinecone: {total}")

# Optional: summarize by metadata if using namespaces
if "namespaces" in stats:
    print("\n🔍 Breakdown by namespace (if used):")
    for ns, info in stats["namespaces"].items():
        print(f"  - {ns or '[default]'}: {info['vector_count']} vectors")
