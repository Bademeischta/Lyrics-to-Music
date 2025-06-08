import os
import yaml
import torch
from torch import nn
from torch.optim import Adam
from src.models.text_encoder import TextEncoder
from src.models.style_encoder import StyleEncoder
from src.models.cross_attention import CrossModalAttention
from src.models.music_decoder import MusicDecoder
from src.data_loader import create_dataloader


def train(config_path: str, data_path: str, out_dir: str,
          epochs: int | None = None, batch_size: int | None = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if epochs is not None:
        cfg['train']['epochs'] = epochs
    if batch_size is not None:
        cfg['train']['batch_size'] = batch_size

    device = 'cpu'
    text_enc = TextEncoder(cfg['model']['text_encoder']).to(device)
    style_enc = StyleEncoder(num_genres=10,
                             embed_dim=cfg['model']['style_embed_dim']).to(device)
    attn = CrossModalAttention(dim=text_enc.model.config.hidden_size).to(device)
    dec_dim = cfg['model'].get('music_embed_dim', 512)
    music_dec = MusicDecoder(vocab_size=cfg['model']['music_vocab_size'], embed_dim=dec_dim).to(device)

    params = list(text_enc.parameters()) + list(style_enc.parameters()) + \
             list(attn.parameters()) + list(music_dec.parameters())
    lr = float(cfg['train']['lr'])
    opt = Adam(params, lr=lr)

    loader = create_dataloader(data_path,
                               batch_size=cfg['train']['batch_size'],
                               shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(cfg['train']['epochs']):
        for lyrics_batch, tgt in loader:
            lyrics = lyrics_batch[0] if isinstance(lyrics_batch, (list, tuple)) else lyrics_batch
            text_emb = text_enc(lyrics)
            style_emb = style_enc({'genre_id': 0})
            fused = attn(text_emb, style_emb)
            out = music_dec(fused, fused)
            loss = nn.functional.mse_loss(out, torch.zeros_like(out))
            loss.backward()
            opt.step()
            opt.zero_grad()

    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'text_encoder': text_enc.state_dict(),
        'style_encoder': style_enc.state_dict(),
        'cross_attention': attn.state_dict(),
        'music_decoder': music_dec.state_dict(),
    }, os.path.join(out_dir, 'model.pt'))
