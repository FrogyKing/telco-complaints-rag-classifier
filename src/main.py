from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from src.services.ai_service import ComplaintClassifier
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Telco Complaint Classifier", version="1.0.0")
service = ComplaintClassifier()


class Item(BaseModel):
    text: str


@app.get("/health")
async def health():
    return JSONResponse(
        content={"status": "ok", "model": "gemini-2.0-flash", "index": "telco-complaints-index"},
        status_code=200
    )


@app.post("/classify")
async def classify(item: Item):
    result = service.classify(item.text)
    return JSONResponse(content=result, status_code=200)


@app.post("/classify/stream")
async def classify_stream(item: Item):
    return StreamingResponse(
        service.classify_stream(item.text),
        media_type="text/plain"
    )