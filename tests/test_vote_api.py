import os
import requests
from dotenv import load_dotenv

load_dotenv()
headers = {
    "X-API-Key": os.getenv("CONGRESS_API_KEY")
}

url = "https://api.congress.gov/v3/vote/118/house/1/35"
res = requests.get(url, headers=headers)

print("Status:", res.status_code)
print("Response:", res.text[:1000])
