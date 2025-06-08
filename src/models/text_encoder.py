"""Simplified text encoder with tokenization and basic prosody extraction."""
import re
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """Wraps a transformer model and exposes a simple forward method."""

    def __init__(self, model_name: str = "facebook/bart-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, lyrics: str):
        tokens = tokenize(lyrics, self.tokenizer)
        prosody = extract_prosody(lyrics)
        # The underlying model does not actually use prosody, but we return
        # whatever it produces to keep the interface flexible.
        return self.model(tokens, prosody)

# helper functions
def tokenize(text: str, tokenizer: AutoTokenizer):
    """Tokenize text using a HuggingFace tokenizer."""
    return tokenizer(text, return_tensors="pt")["input_ids"]


def extract_prosody(text: str):
    """Very naive syllable count per word as prosody proxy."""
    return [len(re.findall(r"[aeiouy]+", w.lower())) for w in text.split()]
