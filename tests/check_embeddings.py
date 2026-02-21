from google import genai
import os
from dotenv import load_dotenv

#client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

client = genai.Client(
    vertexai=True,
    project="luis-sandbox-488104",  # ej: "mi-proyecto-123"
    location="us-central1"
)

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?"
)

print(result.embeddings)