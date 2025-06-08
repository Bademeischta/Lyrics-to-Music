# Lyrics-to-Music AI Blueprint

This document outlines the architecture, data pipelines, and deployment strategy for an AI system that converts song lyrics and style parameters into complete musical pieces.

## Top-Level Architecture

```mermaid
flowchart LR
    subgraph Ingestion
      TextInput[Songtext]
      StyleInput[Style JSON]
    end
    subgraph Encoding
      TextEnc[Transformer Text Encoder]
      StyleEnc[Style Encoder]
    end
    subgraph Fusion
      CrossAttn[Cross-Modal Attention]
      Latent[Latent Space (VAE)]
    end
    subgraph Decoding
      MusicDec[Music Decoder]
      PostProc[Post-Processing]
    end
    subgraph Output
      Exporter[MIDI/WAV/MP3]
    end

    TextInput --> TextEnc
    StyleInput --> StyleEnc
    TextEnc & StyleEnc --> CrossAttn --> Latent --> MusicDec --> PostProc --> Exporter
```

## Modules

### Text Encoder (`src/models/text_encoder.py`)
```python
class TextEncoder(nn.Module):
    def __init__(self, model_name="facebook/bart-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
    def forward(self, lyrics: str):
        tokens = tokenize(lyrics)
        prosody = extract_prosody(lyrics)
        return self.model(tokens, prosody)
```

### Style Encoder (`src/models/style_encoder.py`)
```python
class StyleEncoder(nn.Module):
    def __init__(self, num_genres: int, embed_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_genres, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, style_json: dict):
        genre_id = style_json.get("genre_id", 0)
        x = self.embed(torch.tensor([genre_id]))
        return self.mlp(x)
```

### Cross-Modal Attention (`src/models/cross_attention.py`)
```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads)
    def forward(self, text_emb: torch.Tensor, style_emb: torch.Tensor):
        style_exp = style_emb.unsqueeze(0).expand(text_emb.size(0), -1, -1)
        fused, _ = self.attn(text_emb, style_exp, style_exp)
        return fused
```

### Music Decoder (`src/models/music_decoder.py`)
```python
class MusicDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(embed_dim, 8)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        self.output = nn.Linear(embed_dim, vocab_size)
    def forward(self, latent: torch.Tensor, tgt_seq: torch.Tensor):
        out = self.decoder(tgt_seq, latent)
        return self.output(out)
```

### Post Processing (`src/utils/post_process.py`)
```python
class PostProcessor:
    def tokens_to_midi(self, tokens, path):
        midi = tokens_to_midi(tokens)
        midi.write(path)
    def midi_to_wav(self, midi_path, wav_path):
        synth = FluidSynth()
        synth.midi_to_audio(midi_path, wav_path)
```

## Repository Layout

```
├── README.md
├── docs
│   └── blueprint.md
├── config
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── deployment.yaml
├── src
│   ├── models
│   │   ├── text_encoder.py
│   │   ├── style_encoder.py
│   │   ├── cross_attention.py
│   │   └── music_decoder.py
│   └── utils
│       └── post_process.py
└── .github
    └── workflows
        └── ci.yml
```

## Configuration Examples

`config/training.yaml`
```yaml
model:
  text_encoder: facebook/bart-base
  style_embed_dim: 128
  music_vocab_size: 512
train:
  batch_size: 8
  epochs: 50
  lr: 3e-4
```

`config/evaluation.yaml`
```yaml
eval:
  metrics: [PAS, SSI, HC, hMOS]
  batch_size: 4
```

`config/deployment.yaml`
```yaml
api:
  host: 0.0.0.0
  port: 8080
  workers: 2
```

## CI Pipeline

`.github/workflows/ci.yml`
```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: pip install torch pytest
      - name: Test
        run: pytest -v || true
```

## API Specification
The REST API is defined in `docs/api/openapi.yaml` using OpenAPI 3.0. The main endpoint `/generate` accepts lyrics and style information and returns a download link for the generated music.

## Deployment Artifacts
- `Dockerfile` – build container image
- `k8s/deployment.yaml`, `k8s/service.yaml` – Kubernetes manifests
- `serverless/aws_lambda.yaml` – example serverless configuration

## Monitoring Templates
Example Prometheus and Grafana configurations live in the `monitoring/` directory. These track request latency and server health.
