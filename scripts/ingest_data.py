import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

INDEX_NAME = "telco-complaints-index"
SHEET_URL = "https://docs.google.com/spreadsheets/d/15Bz1q07ahdOalhPGUMBsaYTtwbZhj7bxwpF6E9KRPLs/export?format=csv"

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project="luis-sandbox-488104",
    location="us-central1"
)

df = pd.read_csv(SHEET_URL).dropna(subset=["reclamo", "categoria"])
documents = [
    Document(page_content=row["reclamo"], metadata={"categoria": row["categoria"]})
    for _, row in df.iterrows()
]

print(f"Documentos a subir: {len(documents)}")

vector_store = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print(f"Éxito: {len(documents)} reclamos subidos al índice '{INDEX_NAME}'.")