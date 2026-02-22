import os
import json
import re
from typing import Iterator
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
        self.categorias_validas = self._get_categorias()

    def _get_categorias(self) -> list[str]:
        docs = self.vector_store.similarity_search("reclamo", k=50)
        return list(set(d.metadata["categoria"] for d in docs))

    def classify(self, text: str) -> dict:
        docs = self.vector_store.similarity_search(text, k=3)
        examples = "\n".join([
            f"- Reclamo: {d.page_content} -> Categoría: {d.metadata['categoria']}"
            for d in docs
        ])
        categorias_str = ", ".join(self.categorias_validas)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Eres un clasificador experto de reclamos de telecomunicaciones.\n\n"
                f"Ejemplos similares:\n{examples}\n\n"
                f"CATEGORÍAS VÁLIDAS (usa SOLO una de estas exactamente): {categorias_str}\n\n"
                "No inventes categorías nuevas. Si no estás seguro, elige la más cercana.\n\n"
                "Responde SOLO con este JSON, sin texto adicional:\n"
                "{{\n"
                '  "categoria": "<categoria exacta de la lista>",\n'
                '  "confianza": "<alta|media|baja>",\n'
                '  "razonamiento": "<explicación breve>"\n'
                "}}"
            )),
            ("human", "Clasifica este reclamo: {text}")
        ])

        chain = prompt | self.llm
        response = chain.invoke(
            {"text": text},
            config={"callbacks": [self.langfuse_handler]}
        )

        raw = response.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {"categoria": raw, "confianza": "desconocida", "razonamiento": ""}

        # Validar que la categoría sea válida, si no, forzar la más cercana
        if result.get("categoria") not in self.categorias_validas:
            result["categoria"] = self.categorias_validas[0]  # fallback
            result["confianza"] = "baja"

        result["ejemplos_similares"] = [
            {"reclamo": d.page_content, "categoria": d.metadata["categoria"]}
            for d in docs
        ]
        return result

    def classify_stream(self, text: str) -> Iterator[str]:
        docs = self.vector_store.similarity_search(text, k=3)
        examples = "\n".join([
            f"- Reclamo: {d.page_content} -> Categoría: {d.metadata['categoria']}"
            for d in docs
        ])
        categorias_str = ", ".join(self.categorias_validas)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Eres un clasificador experto de reclamos de telecomunicaciones.\n\n"
                f"Ejemplos similares:\n{examples}\n\n"
                f"CATEGORÍAS VÁLIDAS (usa SOLO una de estas): {categorias_str}\n\n"
                "Analiza el reclamo paso a paso y al final indica la categoría."
            )),
            ("human", "Clasifica este reclamo: {text}")
        ])

        chain = prompt | self.llm
        for chunk in chain.stream(
            {"text": text},
            config={"callbacks": [self.langfuse_handler]}
        ):
            yield chunk.content