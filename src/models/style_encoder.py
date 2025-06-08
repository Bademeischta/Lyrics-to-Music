"""Style encoder that embeds genre, tempo, mood and instrumentation."""
import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    def __init__(self, num_genres: int, embed_dim: int = 128, num_moods: int = 50, num_instr: int = 50):
        super().__init__()
        self.genre_embed = nn.Embedding(num_genres, embed_dim)
        self.mood_embed = nn.Embedding(num_moods, embed_dim)
        self.instr_embed = nn.Embedding(num_instr, embed_dim)
        self.tempo_fc = nn.Linear(2, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 4, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, style_json: dict) -> torch.Tensor:
        genre_id = style_json.get("genre_id", 0)
        tempo = style_json.get("tempo_range", [120, 120]) or [120, 120]
        tempo_range = torch.tensor(tempo, dtype=torch.float32)
        moods = style_json.get("mood_tags") or []
        instruments = style_json.get("instrumentation_list") or []

        genre_vec = self.genre_embed(torch.tensor([genre_id]))
        tempo_vec = self.tempo_fc(tempo_range.unsqueeze(0))

        if moods:
            mood_ids = torch.tensor([hash(m) % self.mood_embed.num_embeddings for m in moods])
            mood_vec = self.mood_embed(mood_ids).mean(dim=0, keepdim=True)
        else:
            mood_vec = torch.zeros_like(genre_vec)

        if instruments:
            instr_ids = torch.tensor([hash(i) % self.instr_embed.num_embeddings for i in instruments])
            instr_vec = self.instr_embed(instr_ids).mean(dim=0, keepdim=True)
        else:
            instr_vec = torch.zeros_like(genre_vec)

        x = torch.cat([genre_vec, tempo_vec, mood_vec, instr_vec], dim=1)
        return self.mlp(x)
