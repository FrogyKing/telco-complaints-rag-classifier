import os
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler

class ComplaintClassifier:
    def __init__(self):
        self.embeddings = VertexAIEmbeddings(
            model_name="text-embedding-005",
            project="luis-sandbox-488104",
            location="us-central1"
        )
        self.llm = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project="luis-sandbox-488104",
            location="us-central1"
        )
        
        self.vector_store = PineconeVectorStore(
            index_name="telco-complaints-index",
            embedding=self.embeddings
        )

        self.langfuse_handler = CallbackHandler()

    def classify(self, text: str):
        docs = self.vector_store.similarity_search(text, k=3)
        examples = "\n".join([f"- Reclamo: {d.page_content} -> Cat: {d.metadata['categoria']}" for d in docs])

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un clasificador de reclamos. Usa estos ejemplos como guía:\n\n{examples}"),
            ("human", f"Clasifica este reclamo: {text}\nResponde solo con la categoría, y .")
        ])

        chain = prompt | self.llm
        return chain.invoke(
            {},
            config={"callbacks": [self.langfuse_handler]}  # ← aquí se activa el monitoreo
        ).content