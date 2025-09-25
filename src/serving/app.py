from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

from src.features.embeddings import get_model
from src.utils.logger import init_logger
from src.serving.middleware import create_logging_middleware

# ------------- Load models once -------------
XGB_MODEL_PATH = "models/xgb_sentiment.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
xgb_model = joblib.load(XGB_MODEL_PATH)
embedding_model = get_model(EMBEDDING_MODEL_NAME)

# ------------- App and logger -------------
app = FastAPI(title="Amazon Sentiment Classifier")
logger = init_logger()
app.middleware("http")(create_logging_middleware(logger))

# ------------- Prometheus endpoint -------------
@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------- Schemas -------------
class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    classes: List[int]
    probabilities: List[List[float]]

# ------------- Endpoint -------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    embeddings = embedding_model.encode(request.texts, show_progress_bar=False, batch_size=32)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    probs = xgb_model.predict_proba(embeddings)
    classes = probs.argmax(axis=1).tolist()
    return PredictResponse(classes=classes, probabilities=probs.tolist())
