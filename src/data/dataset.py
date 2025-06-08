import csv
from typing import Any, Dict, List
import torch
from torch.utils.data import Dataset
import pretty_midi

class LyricsMusicDataset(Dataset):
    """Simple dataset loading lyrics and midi paths from a CSV."""

    def __init__(self, csv_path: str, transform=None):
        self.rows: List[Dict[str, Any]] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        lyrics = row['lyrics']
        midi_path = row['midi']
        midi = pretty_midi.PrettyMIDI(midi_path)
        item = {
            'lyrics': lyrics,
            'midi': midi,
            'style': {
                'genre_id': int(row.get('genre_id', 0)),
                'tempo_range': [int(row.get('tempo_low', 0)), int(row.get('tempo_high', 0))],
                'mood_tags': row.get('mood_tags', '').split('|') if row.get('mood_tags') else [],
                'instrumentation_list': row.get('instrumentation_list', '').split('|') if row.get('instrumentation_list') else [],
            },
        }
        if self.transform:
            item = self.transform(item)
        return item
