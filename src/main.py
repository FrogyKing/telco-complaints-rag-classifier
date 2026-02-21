from fastapi import FastAPI
from pydantic import BaseModel
from src.services.ai_service import ComplaintClassifier
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
service = ComplaintClassifier()

class Item(BaseModel):
    text: str

@app.post("/classify")
async def classify(item: Item):
    category = service.classify(item.text)
    return {"categoria": category}