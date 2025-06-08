"""Cross modal attention with positional encoding."""
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, max_len: int = 512):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.register_parameter("pos_embed", nn.Parameter(torch.zeros(max_len, dim)))
        nn.init.xavier_uniform_(self.pos_embed)

    def forward(self, text_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        # text_emb [seq,batch,dim] -> [batch,seq,dim]
        if text_emb.dim() == 3 and text_emb.shape[0] != style_emb.shape[0]:
            text_emb = text_emb.transpose(0, 1)
        batch, seq, dim = text_emb.shape
        pos = self.pos_embed[:seq].unsqueeze(0).expand(batch, -1, -1)
        text_emb = text_emb + pos
        style_exp = style_emb.unsqueeze(1)
        fused, _ = self.attn(text_emb, style_exp, style_exp)
        return fused.transpose(0, 1)
