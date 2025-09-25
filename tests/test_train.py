import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import json

from src.model import train
from src.features.embeddings import get_embedding

def make_tiny_df():
    return pd.DataFrame({
        "text": [
            "I love this product, it is amazing!",
            "This is okay, not bad but not great.",
            "Terrible experience, do not recommend."
        ],
        "rating": [5, 3, 1]
    })

def test_map_rating_to_class():
    assert train._map_rating_to_class(1) == 0
    assert train._map_rating_to_class(2) == 0
    assert train._map_rating_to_class(3) == 1
    assert train._map_rating_to_class(4) == 2
    assert train._map_rating_to_class(5) == 2

def test_train_model_returns_xgbclassifier():
    df = make_tiny_df()
    model = train._train_model(df, text_col="text")
    from xgboost import XGBClassifier
    assert isinstance(model, XGBClassifier)

def test_model_predicts_probability_shape():
    df = make_tiny_df()
    model = train._train_model(df, text_col="text")
    # new input embedding
    emb = get_embedding("This is wonderful!")
    emb = emb.reshape(1, -1)
    probs = model.predict_proba(emb)
    assert probs.shape == (1, 3)
    # probabilities sum to ~1
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)

def test_save_and_load_model_roundtrip(tmp_path):
    df = make_tiny_df()
    model = train._train_model(df, text_col="text")

    model_path = tmp_path / "xgb_model.joblib"
    train._save_model(model, model_path)
    assert model_path.exists()

    loaded_model = train.load_model(model_path)
    # check loaded model works
    emb = get_embedding("Horrible!")
    emb = emb.reshape(1, -1)
    pred = loaded_model.predict(emb)
    assert pred.shape == (1,)


def test_evaluate_model_returns_metrics():
    df = make_tiny_df()
    model = train._train_model(df, text_col="text")

    metrics = train._evaluate_model(model, df, text_col="text")
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "classification_report" in metrics
    assert 0 <= metrics["accuracy"] <= 1

def test_run_training_pipeline_creates_files(tmp_path):
    # create a dummy JSONL file for pipeline
    jsonl = tmp_path / "reviews.jsonl"
    # each line is a JSON object
    lines = [
        '{"rating":5,"text":"This is great. Love it!"}',
        '{"rating":5,"text":"Fantastic product"}',
        '{"rating":3,"text":"It is fine."}',
        '{"rating":3,"text":"Okay"}',
        '{"rating":1,"text":"Bad product."}',
        '{"rating":1,"text":"Awful"}'
    ]
    jsonl.write_text("\n".join(lines), encoding="utf-8")

    model_path = tmp_path / "xgb_model.joblib"
    metrics_path = tmp_path / "metrics.json"

    # run pipeline
    train.run_training_pipeline(
        data_path=jsonl,
        model_path=model_path,
        metrics_path=metrics_path,
        test_size=0.33,  # small eval split
    )

    # assert model and metrics files created
    assert model_path.exists()
    assert metrics_path.exists()

    # load and inspect metrics
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics
    assert "classification_report" in metrics

    # load and check model works
    model = train.load_model(model_path)
    assert isinstance(model, XGBClassifier)
    emb = get_embedding("Wonderful product").reshape(1, -1)
    pred = model.predict(emb)
    assert pred.shape == (1,)