"""FastAPI server that exposes the generation pipeline."""
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from src.models.text_encoder import TextEncoder
from src.models.style_encoder import StyleEncoder
from src.models.cross_attention import CrossModalAttention
from src.models.music_decoder import MusicDecoder
from src.utils.post_process import PostProcessor
import torch

app = FastAPI()

text_enc = TextEncoder(model_name="sshleifer/tiny-distilroberta-base")
style_enc = StyleEncoder(num_genres=10, embed_dim=text_enc.model.config.hidden_size, num_moods=50, num_instr=50)
cross = CrossModalAttention(dim=text_enc.model.config.hidden_size, heads=1)
music_dec = MusicDecoder(vocab_size=128, embed_dim=text_enc.model.config.hidden_size, heads=1)
post = PostProcessor()

class Style(BaseModel):
    genre_id: int = 0
    tempo_range: list[int] | None = None
    mood_tags: list[str] | None = None
    instrumentation_list: list[str] | None = None

class GenerateRequest(BaseModel):
    lyrics: str
    style: Style

def generate_music(lyrics: str, style: dict) -> str:
    """Run the inference pipeline and return path to MIDI file."""
    text_emb, _ = text_enc(lyrics)
    style_emb = style_enc(style)
    latent = cross(text_emb, style_emb)
    tokens = music_dec.generate(latent)
    out_path = Path("generated.mid")
    post.tokens_to_midi(tokens, out_path)
    return str(out_path)

@app.post('/generate')
async def generate(req: GenerateRequest):
    path = generate_music(req.lyrics, req.style.dict())
    return {'download_url': path, 'metadata': req.dict()}

@app.get('/health')
async def health():
    return {'status': 'ok'}
