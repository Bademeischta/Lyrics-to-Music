# Lyrics-to-Music

This project provides a blueprint for an AI system that turns song lyrics and style parameters into complete music tracks.

## Quickstart

```bash
pip install -r requirements.txt
pytest -q
uvicorn src.api.server:app --reload
```

Open `http://localhost:8000/docs` to explore the API.

## Repository Layout

- `docs/` – design documents and the [OpenAPI spec](docs/api/openapi.yaml)
- `src/` – model implementations, API server and training scripts
- `config/` – configuration samples for training, evaluation, deployment and logging
- `k8s/` – Kubernetes manifests
- `serverless/` – example AWS Lambda configuration
- `monitoring/` – Prometheus and Grafana templates
- `.github/workflows/` – CI definitions

See `docs/blueprint.md` for the full architecture description.

## Usage Examples

Send a POST request to `/generate` with a JSON body:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"lyrics": "hello world", "style": {"genre_id": 1}}'
```

The server responds with a download URL for the generated music.

### Training

Prepare a CSV file with columns `lyrics` and `midi`. Then run:

```bash
python src/train.py --csv data/train.csv
```

### Command Line Generation

```bash
python src/cli.py "hello world" --genre_id 1
```

## Deployment

A `Dockerfile` is included for container builds. To deploy on Kubernetes run:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

For serverless environments see `serverless/aws_lambda.yaml`.
