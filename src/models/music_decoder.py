"""Pseudo-code music decoder."""
import torch.nn as nn
import torch

class MusicDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(embed_dim, 8)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        self.output = nn.Linear(embed_dim, vocab_size)
    def forward(self, latent: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        out = self.decoder(tgt_seq, latent)
        return self.output(out)
