"""Pseudo-code cross-modal attention."""
import torch.nn as nn
import torch

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
    def forward(self, text_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        style_exp = style_emb.unsqueeze(0).expand(text_emb.size(0), -1, -1)
        fused, _ = self.attn(text_emb, style_exp, style_exp)
        return fused
