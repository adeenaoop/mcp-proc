import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Make sure the token is actually loaded
token = os.environ.get("GITHUB_TOKEN")
if not token:
    raise SystemExit("GITHUB_TOKEN not found. Check your .env file.")

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=token,
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",   # use a small model to save rate limit
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain GitHub Models in one sentence."},
    ],
)

# --- Debug: see what type you actually got ---
print("TYPE:", type(response))

# Handle BOTH cases so it never crashes:
if isinstance(response, str):
    print("STRING RESPONSE:", response)
else:
    print("CHAT RESPONSE:", response.choices[0].message.content)
