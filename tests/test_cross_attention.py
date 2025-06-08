import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.models.cross_attention import CrossModalAttention

def test_cross_attention():
    attn = CrossModalAttention(dim=4, heads=1)
    text = torch.randn(3,1,4)
    style = torch.randn(1,4)
    out = attn(text, style)
    assert out.shape == text.shape

