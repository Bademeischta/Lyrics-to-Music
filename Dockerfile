FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch transformers fastapi uvicorn
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
