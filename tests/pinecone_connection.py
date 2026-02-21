from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv() 

def test_pinecone_connection():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    try:
        indexes = pc.list_indexes().names()
        print("Conexión exitosa a Pinecone.")
        print(f"Índices disponibles: {indexes}")
    except Exception as e:
        print("Error conectando a Pinecone:", e)

if __name__ == "__main__":
    test_pinecone_connection()