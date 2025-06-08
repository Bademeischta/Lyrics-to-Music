"""Simple autoregressive music decoder."""
import torch
import torch.nn as nn

class MusicDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        dec_layer = nn.TransformerDecoderLayer(embed_dim, heads)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, latent: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        tgt = self.embed(tgt_seq)
        out = self.decoder(tgt, latent)
        return self.output(out)

    @torch.no_grad()
    def generate(self, latent: torch.Tensor, max_len: int = 64, start_token: int = 0) -> torch.Tensor:
        """Greedy generation from latent memory."""
        generated = [start_token]
        memory = latent
        for _ in range(max_len - 1):
            tgt_seq = torch.tensor(generated).unsqueeze(1)
            logits = self.forward(memory, tgt_seq)
            next_token = logits[-1, 0].argmax().item()
            generated.append(next_token)
            if next_token == start_token:
                break
        return torch.tensor(generated)
