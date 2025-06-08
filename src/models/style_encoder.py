"""Pseudo-code style encoder."""
import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    def __init__(self, num_genres: int, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_genres, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, style_json: dict) -> torch.Tensor:
        genre_id = style_json.get("genre_id", 0)
        x = self.embed(torch.tensor([genre_id]))
        return self.mlp(x)
