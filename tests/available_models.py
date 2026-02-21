import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    vertexai=True,
    project="luis-sandbox-488104",
    location="us-central1"
)

for m in client.models.list():
    if "embed" in m.name.lower():
        print(m.name)