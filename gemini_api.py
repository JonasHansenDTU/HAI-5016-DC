import os
from dotenv import load_dotenv
from datetime import date
from google import genai

# Load .env from the workspace root (if present)
load_dotenv()

# Read API key from environment
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Missing GEMINI_API_KEY environment variable. Set it and rerun.")

# Pass the API key explicitly to the client
client = genai.Client(api_key=api_key)


# Simple in-memory conversation history. Each item is a tuple (role, text).
# role is 'user' or 'assistant'. We keep a small buffer to avoid sending
# an unbounded amount of history.
conversation_history = []
MAX_HISTORY_ITEMS = 12  # keep last N messages (user+assistant entries)


def build_prompt(history, latest_user_input):
    """Build a plain-text prompt that includes a brief system instruction
    and the previous turns from `history` followed by the new user input.
    This works with a simple `contents` string for the existing client API.
    """
    system = "You are a helpful assistant. Keep answers concise and include relevant context from previous turns when appropriate."
    # include today's date so the model can reference the current day
    today = date.today().isoformat()
    parts = ["System: " + system, f"Date: {today}", "\nConversation:\n"]

    # only keep the last MAX_HISTORY_ITEMS entries
    recent = history[-MAX_HISTORY_ITEMS:]
    for role, text in recent:
        prefix = "User: " if role == "user" else "Assistant: "
        parts.append(prefix + text + "\n")

    parts.append("User: " + latest_user_input + "\nAssistant:")
    return "\n".join(parts)


# Loop: read user input, send to model, print reply; type 'exit' to quit.
try:
    while True:
        user_input = input("Ask the model (type 'exit' to quit, 'clear' to reset memory, 'history' to show): ").strip()
        if not user_input:
            # skip empty input
            continue
        cmd = user_input.lower()
        if cmd == "exit":
            print("Exiting.")
            break
        if cmd == "clear":
            conversation_history.clear()
            print("Conversation memory cleared.")
            continue
        if cmd == "history":
            if not conversation_history:
                print("(no history)")
            else:
                print("--- Conversation history (most recent last) ---")
                for r, t in conversation_history:
                    role = "User" if r == "user" else "Assistant"
                    print(f"{role}: {t}")
                print("--- end history ---")
            continue

        # Build a prompt including recent conversation
        prompt = build_prompt(conversation_history, user_input)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            # Print model's text reply
            assistant_reply = getattr(response, "text", None) or str(response)
            print(assistant_reply)

            # Append to history and keep buffer size bounded
            conversation_history.append(("user", user_input))
            conversation_history.append(("assistant", assistant_reply))
            # trim if too large
            if len(conversation_history) > MAX_HISTORY_ITEMS:
                # keep last MAX_HISTORY_ITEMS entries
                conversation_history = conversation_history[-MAX_HISTORY_ITEMS:]

        except Exception as e:
            # Simple error message and continue the loop
            print(f"Error calling model: {e}")
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting.")

