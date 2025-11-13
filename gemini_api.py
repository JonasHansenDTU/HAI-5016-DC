import os
from dotenv import load_dotenv
from google import genai

# Load .env from the workspace root (if present)
load_dotenv()

# Read API key from environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY environment variable. Set it and rerun.")

# Pass the API key explicitly to the client
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words"
)
print(response.text)