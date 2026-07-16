# CPU-only reranker sidecar (BAAI/bge-reranker-v2-m3). Serves POST /rerank + GET /health.
FROM python:3.11-slim

WORKDIR /app

# Torch CPU wheels + build deps for sentence-transformers; keep the image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Baking the model in at build time avoids a cold download on first request;
# override RERANKER_MODEL at build/run if you want a different checkpoint.
ARG RERANKER_MODEL=BAAI/bge-reranker-v2-m3
ENV RERANKER_MODEL=${RERANKER_MODEL}
RUN python -c "from sentence_transformers import CrossEncoder; import os; CrossEncoder(os.environ['RERANKER_MODEL'], device='cpu')"

EXPOSE 8009

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8009"]
