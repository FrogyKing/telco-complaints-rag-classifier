import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
from rich.panel import Panel

load_dotenv()

console = Console()

INDEX_NAME = "telco-complaints-index"
SHEET_URL = "https://docs.google.com/spreadsheets/d/15Bz1q07ahdOalhPGUMBsaYTtwbZhj7bxwpF6E9KRPLs/export?format=csv"

console.print(Panel.fit(
    "[bold cyan]Telco Complaint Classifier[/bold cyan]\n[dim]Pipeline de ingesta de datos[/dim]",
    border_style="cyan"
))

# --- Paso 1: Embeddings ---
with console.status("[bold yellow]⚙ Inicializando modelo de embeddings...[/bold yellow]"):
    embeddings = VertexAIEmbeddings(
        model_name="text-embedding-005",
        project="luis-sandbox-488104",
        location="us-central1"
    )
console.print("[green]✅ Modelo[/green] [bold]text-embedding-005[/bold] listo")

# --- Paso 2: Cargar datos ---
with console.status("[bold yellow]📄 Cargando datos desde Google Sheets...[/bold yellow]"):
    df = pd.read_csv(SHEET_URL).dropna(subset=["reclamo", "categoria"])

console.print(f"[green]✅ Datos cargados:[/green] [bold]{len(df)}[/bold] reclamos válidos encontrados")

# Mostrar distribución por categoría
table = Table(title="Distribución por Categoría", border_style="dim")
table.add_column("Categoría", style="cyan")
table.add_column("Cantidad", justify="right", style="magenta")
for cat, count in df["categoria"].value_counts().items():
    table.add_row(cat, str(count))
console.print(table)

# --- Paso 3: Crear documentos ---
documents = [
    Document(page_content=row["reclamo"], metadata={"categoria": row["categoria"]})
    for _, row in df.iterrows()
]

# --- Paso 4: Subir a Pinecone con barra de progreso ---
console.print(f"\n[bold]🚀 Subiendo {len(documents)} documentos a Pinecone...[/bold]")

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Generando embeddings y subiendo...", total=len(documents))

    # Subimos en batches para poder actualizar la barra
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if i == 0:
            # Primera vez: crea el vector store
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=INDEX_NAME
            )
        else:
            vector_store.add_documents(batch)
        progress.advance(task, len(batch))

console.print(f"\n[bold green]🎉 Éxito:[/bold green] {len(documents)} reclamos subidos al índice '[bold]{INDEX_NAME}[/bold]'")