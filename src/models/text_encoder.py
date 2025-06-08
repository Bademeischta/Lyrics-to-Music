"""Simple text encoder using a pretrained transformer."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pyphen

class TextEncoder(nn.Module):
    """Encodes lyrics into contextual embeddings and extracts prosody."""

    def __init__(self, model_name: str = "facebook/bart-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._dic = pyphen.Pyphen(lang="en")

    def forward(self, lyrics: str):
        tokens = self.tokenizer(lyrics, return_tensors="pt")
        out = self.model(**tokens).last_hidden_state.transpose(0, 1)
        prosody = extract_prosody(lyrics, out.size(0), self._dic)
        return out, prosody

# helper functions
def extract_prosody(text: str, seq_len: int, dic: pyphen.Pyphen):
    """Return syllable count per token as tensor [seq_len, 1]."""
    words = text.split()
    syllables = []
    for word in words:
        count = len(dic.inserted(word).split("-"))
        syllables.append(count)
    if len(syllables) < seq_len:
        syllables.extend([0] * (seq_len - len(syllables)))
    syllables = syllables[:seq_len]
    return torch.tensor(syllables, dtype=torch.float32).unsqueeze(1)
