from langchain_pinecone import PineconeVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

class ComplaintRetriever:
    def __init__(self):
        self.index_name = "telco-complaints-index"

        # Embeddings (versión antigua)
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project="luis-sandbox-488104",
            location="us-central1"
        )

        # Inicializar Pinecone
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Crear índice si no existe
        if self.index_name not in [idx.name for idx in self.pinecone_client.list_indexes()]:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=768,  # coincide con text-embedding-004
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        # Inicializar vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

    def get_similar_examples(self, query: str, k: int = 3):
        """Busca los k ejemplos más similares y formatea para el LLM"""
        results = self.vector_store.similarity_search(query, k=k)
        formatted_examples = ""
        for i, doc in enumerate(results):
            text = doc.page_content
            category = doc.metadata.get("categoria", "N/A")
            formatted_examples += f"Ejemplo {i+1}:\nReclamo: {text}\nCategoría: {category}\n\n"
        return formatted_examples