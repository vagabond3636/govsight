import os
import requests
from dotenv import load_dotenv

# Load the API key from your .env file
load_dotenv()
api_key = os.getenv("CONGRESS_API_KEY")

if not api_key:
    print("❌ ERROR: CONGRESS_API_KEY is missing or not loaded from .env")
    exit()

# Prepare the request
headers = {"X-API-Key": api_key}
url = "https://api.congress.gov/v3/bill/118?limit=1"

print("📡 Sending request to Congress.gov...")
response = requests.get(url, headers=headers)

# Handle the response
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    try:
        data = response.json()
        print("✅ Response OK. Keys in response:")
        print(list(data.keys()))
        print("\n🔹 First bill reference:")
        if data.get("bills"):
            print(data["bills"][0])
        else:
            print("No bills found.")
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
else:
    print(f"❌ Request failed. Response: {response.text}")
