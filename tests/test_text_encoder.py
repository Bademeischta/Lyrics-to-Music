import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.text_encoder import TextEncoder


def test_text_encoder_forward():
    enc = TextEncoder(model_name="sshleifer/tiny-distilroberta-base")
    emb, pros = enc("hello world")
    assert emb.shape[1] == 1
    assert pros.shape[0] == emb.shape[0]
