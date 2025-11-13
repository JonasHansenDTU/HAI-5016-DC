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


# Loop: read user input, send to model, print reply; type 'exit' to quit.
try:
    while True:
        user_input = input("Ask the model (type 'exit' to quit): ").strip()
        if not user_input:
            # skip empty input
            continue
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input
            )
            # Print model's text reply
            print(response.text)
        except Exception as e:
            # Simple error message and continue the loop
            print(f"Error calling model: {e}")
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting.")

