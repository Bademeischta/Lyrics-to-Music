import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, MagicMock
from src.models.text_encoder import TextEncoder

@patch('src.models.text_encoder.tokenize', return_value='tok')
@patch('src.models.text_encoder.extract_prosody', return_value='pros')
@patch('src.models.text_encoder.AutoModel')
def test_text_encoder_forward(mock_model, mock_prosody, mock_tokenize):
    inst = MagicMock(return_value='out')
    mock_model.from_pretrained.return_value = inst
    enc = TextEncoder()
    assert enc.forward('hi') == 'out'

