"""Minimal FastAPI server."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Style(BaseModel):
    genre_id: int = 0
    tempo_range: list[int] | None = None
    mood_tags: list[str] | None = None
    instrumentation_list: list[str] | None = None

class GenerateRequest(BaseModel):
    lyrics: str
    style: Style

@app.post('/generate')
async def generate(req: GenerateRequest):
    return {'download_url': 'https://example.com/song.mid', 'metadata': req.dict()}

@app.get('/health')
async def health():
    return {'status': 'ok'}
