import os
import yaml
import torch
from pathlib import Path
from src.models.text_encoder import TextEncoder
from src.models.style_encoder import StyleEncoder
from src.models.cross_attention import CrossModalAttention
from src.models.music_decoder import MusicDecoder
from src.utils.post_process import PostProcessor


class Pipeline:
    """Simple inference pipeline loading the trained modules."""

    def __init__(self, model_dir: str, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.text_enc = TextEncoder(cfg['model']['text_encoder'])
        self.style_enc = StyleEncoder(num_genres=10,
                                      embed_dim=cfg['model']['style_embed_dim'])
        self.attn = CrossModalAttention(dim=self.text_enc.model.config.hidden_size)
        dec_dim = cfg['model'].get('music_embed_dim', 512)
        self.music_dec = MusicDecoder(vocab_size=cfg['model']['music_vocab_size'], embed_dim=dec_dim)

        ckpt = torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu')
        self.text_enc.load_state_dict(ckpt['text_encoder'])
        self.style_enc.load_state_dict(ckpt['style_encoder'])
        self.attn.load_state_dict(ckpt['cross_attention'])
        self.music_dec.load_state_dict(ckpt['music_decoder'])
        self.processor = PostProcessor()

    def generate(self, lyrics: str, style: dict, midi_path: str, wav_path: str | None = None):
        text_emb = self.text_enc(lyrics)
        style_emb = self.style_enc(style)
        fused = self.attn(text_emb, style_emb)
        tokens = self.music_dec(fused, fused).argmax(-1).view(-1).tolist()
        Path(os.path.dirname(midi_path)).mkdir(parents=True, exist_ok=True)
        self.processor.tokens_to_midi(tokens, midi_path)
        if wav_path:
            self.processor.midi_to_wav(midi_path, wav_path)
        return midi_path
