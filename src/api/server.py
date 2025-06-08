"""Minimal FastAPI server."""
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from src.inference import Pipeline
import uuid

app = FastAPI()
pipeline: Pipeline | None = None

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
    global pipeline
    if pipeline is None:
        pipeline = Pipeline(model_dir="models", config_path="config/training.yaml")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    midi_path = out_dir / f"{uuid.uuid4().hex}.mid"
    pipeline.generate(req.lyrics, req.style.dict(), str(midi_path))
    return {'download_url': str(midi_path), 'metadata': req.dict()}

@app.get('/health')
async def health():
    return {'status': 'ok'}
