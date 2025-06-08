import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from src.api.server import app


def test_generate_route(tmp_path):
    client = TestClient(app)
    resp = client.post('/generate', json={'lyrics':'la la', 'style': {'genre_id':0}})
    assert resp.status_code == 200
    assert resp.json()['download_url']
