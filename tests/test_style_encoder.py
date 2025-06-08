import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.models.style_encoder import StyleEncoder

def test_style_encoder():
    enc = StyleEncoder(num_genres=10)
    out = enc({'genre_id': 2})
    assert isinstance(out, torch.Tensor)

