from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

print("OpenAI Key:", openai_api_key[:5] + "...")
print("Pinecone Key:", pinecone_api_key[:5] + "...")
print("Pinecone Env:", pinecone_env)