import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.data.dataset import LyricsMusicDataset
from src.models.text_encoder import TextEncoder
from src.models.style_encoder import StyleEncoder
from src.models.cross_attention import CrossModalAttention
from src.models.music_decoder import MusicDecoder


def main(cfg_path: str, csv_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    text_enc = TextEncoder(cfg['model']['text_encoder'])
    style_enc = StyleEncoder(num_genres=10, embed_dim=cfg['model']['style_embed_dim'])
    cross = CrossModalAttention(dim=text_enc.model.config.hidden_size)
    music_dec = MusicDecoder(cfg['model']['music_vocab_size'])

    dataset = LyricsMusicDataset(csv_path)
    loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

    opt = AdamW(list(text_enc.parameters()) + list(style_enc.parameters()) + list(cross.parameters()) + list(music_dec.parameters()), lr=cfg['train']['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg['train']['epochs']):
        for item in loader:
            text_emb, prosody = text_enc(item['lyrics'])
            style_emb = style_enc(item['style'])
            latent = cross(text_emb, style_emb)
            tgt = torch.randint(0, cfg['model']['music_vocab_size'], (latent.size(0), 1))
            out = music_dec(latent, tgt)
            loss = loss_fn(out.squeeze(1), tgt.squeeze(1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(f"epoch {epoch} loss {loss.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/training.yaml')
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    main(args.config, args.csv)
