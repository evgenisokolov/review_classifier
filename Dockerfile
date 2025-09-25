# ---------- Stage 1: Training ----------
FROM python:3.12-slim as builder

WORKDIR /app

# install build tools (if needed for xgboost)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# copy requirements for training
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# copy source code
COPY src ./src

# accept data path as build arg
ARG DATA_PATH=/app/data/Books_10k.jsonl
ENV DATA_PATH=${DATA_PATH}

# copy data into container
COPY data ./data

# run training pipeline (writes to /app/models)
RUN mkdir -p /app/models
RUN python -m src.model.train \
    --data-path ${DATA_PATH} \
    --model-path /app/models/xgb_sentiment.joblib \
    --metrics-path /app/models/metrics.json

# ---------- Stage 2: Serving ----------
FROM python:3.12-slim

WORKDIR /app

# install only serving deps
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# copy source code needed for serving
COPY src ./src

# copy trained model from builder stage
COPY --from=builder /app/models /app/models

# expose FastAPI port
EXPOSE 8000

# default command
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
