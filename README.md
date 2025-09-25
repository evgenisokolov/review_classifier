# Amazon Review Sentiment Classifier

This project implements an end-to-end pipeline to classify Amazon book review sentences into **positive**, **neutral**, and **negative** sentiments.  
It covers:

- **Data loading & preprocessing** from JSONL reviews.
- **Sentence splitting** and text cleaning.
- **Embedding generation** using a SentenceTransformer model.
- **Training** an XGBoost classifier on embeddings.
- **Evaluation** and metrics reporting.
- **FastAPI serving** endpoint with structured logging and Prometheus metrics.
- **Docker multi-stage build** (train model in stage 1, serve model in stage 2).

---

## File Map

```text
review_classifier/
├─ src/
│  ├─ data/
│  │  └─ dataset.py          # load & preprocess reviews
│  ├─ features/
│  │  └─ embeddings.py       # embedding model + functions
│  ├─ model/
│  │  └─ train.py            # training, evaluation, pipeline runner
│  ├─ serving/
│  │  ├─ app.py              # FastAPI app
│  │  ├─ middleware.py       # logging + metrics middleware
│  ├─ utils/
│  │  └─ logger.py           # init_logger() helper
├─ models/                   # trained model + metrics.json
├─ data/                     # input datasets (.jsonl)
├─ tests/                    # pytest unit tests
├─ requirements-train.txt    # all dependencies including training
├─ requirements-serve.txt    # minimal dependencies for serving
├─ Dockerfile                # multi-stage build (train then serve)
└─ README.md                 # this file
```
---

## Local Development

### 1. Clone & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -r requirements-train.txt
```

### 2. Download NLTK punkt tokenizer

```bash
python -m nltk.downloader punkt punkt_tab
```

### 3. Run training pipeline

```bash
python -m src.model.train \
  --data-path data/Books_10k.jsonl \
  --model-path models/xgb_sentiment.joblib \
  --metrics-path models/metrics.json
```

This will:

- load and preprocess reviews,

- generate embeddings,

- train the XGBoost classifier,

- save model + metrics to models/.

### 4. Start API locally

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# or (safe cross-platform)
python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This book is amazing!", "Awful printing"]}'
```

Metrics at http://127.0.0.1:8000/metrics.

Run tests:
```bash
pytest -q

```

## Docker Build & Run

We use a multi-stage build:

- Stage 1 installs training dependencies and runs the pipeline inside the image to produce xgb_sentiment.joblib.

- Stage 2 installs only serving dependencies and copies the trained model.

Build image
```bash
docker build \
  --build-arg DATA_PATH=/app/data/Books_10k.jsonl \
  -t sentiment-api .
```


DATA_PATH controls which dataset is used inside the image for training.

The Dockerfile copies your data/ folder into the image and runs training.

Run container
```bash
docker run -p 8000:8000 sentiment-api
```


The API is now running at http://localhost:8000/predict.