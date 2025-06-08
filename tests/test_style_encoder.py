import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.models.style_encoder import StyleEncoder

def test_style_encoder():
    enc = StyleEncoder(num_genres=10)
    style = {
        'genre_id': 2,
        'tempo_range': [100, 120],
        'mood_tags': ['happy'],
        'instrumentation_list': ['guitar']
    }
    out = enc(style)
    assert out.shape[1] == enc.mlp[-1].out_features
