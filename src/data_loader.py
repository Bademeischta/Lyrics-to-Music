import json
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class LyricsMusicDataset(Dataset):
    """Dataset reading JSONL files with 'lyrics' and 'midi_tokens'."""

    def __init__(self, path: str):
        self.samples: List[Tuple[str, List[int]]] = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append((obj["lyrics"], obj["midi_tokens"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        lyrics, tokens = self.samples[idx]
        return lyrics, torch.tensor(tokens, dtype=torch.long)


def create_dataloader(path: str, batch_size: int = 1, shuffle: bool = True):
    dataset = LyricsMusicDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
