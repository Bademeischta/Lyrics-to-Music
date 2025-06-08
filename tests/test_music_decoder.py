import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.models.music_decoder import MusicDecoder

def test_music_decoder():
    dec = MusicDecoder(vocab_size=10, embed_dim=8)
    latent = torch.randn(3,2,8)
    tgt = torch.randint(0,10,(3,2))
    out = dec(latent, tgt)
    assert out.shape == (3,2,10)

