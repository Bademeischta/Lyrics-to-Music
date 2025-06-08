import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
from unittest.mock import patch, MagicMock
from src.train import train
from src.inference import Pipeline
import yaml


def test_end_to_end(tmp_path):
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        f.write(json.dumps({"lyrics": "la la", "midi_tokens": [60, 62, 64]}))
        f.write("\n")
    model_dir = tmp_path / "model"

    with patch('src.models.text_encoder.AutoModel') as am, \
         patch('src.models.text_encoder.AutoTokenizer') as at, \
         patch('yaml.safe_load') as yload:
        inst = MagicMock(return_value=torch.randn(1,1,8))
        inst.config = MagicMock(hidden_size=8)
        am.from_pretrained.return_value = inst
        tok = MagicMock()
        tok.return_value = {"input_ids": torch.tensor([[1,2]])}
        at.from_pretrained.return_value = tok
        yload.return_value = {
            'model': {
                'text_encoder': 'stub',
                'style_embed_dim': 8,
                'music_vocab_size': 16,
                'music_embed_dim': 8
            },
            'train': {
                'batch_size': 1,
                'epochs': 1,
                'lr': 0.001
            }
        }

        train("config/training.yaml", str(data_path), str(model_dir))
        pipe = Pipeline(str(model_dir), "config/training.yaml")
        midi_path = tmp_path / "song.mid"
        pipe.generate("la la", {"genre_id": 0}, str(midi_path))
        assert midi_path.exists()
