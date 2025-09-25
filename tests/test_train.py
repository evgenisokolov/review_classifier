import numpy as np
import pandas as pd

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
    model = train.train_model(df, text_col="text")
    from xgboost import XGBClassifier
    assert isinstance(model, XGBClassifier)

def test_model_predicts_probability_shape():
    df = make_tiny_df()
    model = train.train_model(df, text_col="text")
    # new input embedding
    emb = get_embedding("This is wonderful!")
    emb = emb.reshape(1, -1)
    probs = model.predict_proba(emb)
    assert probs.shape == (1, 3)
    # probabilities sum to ~1
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)

def test_save_and_load_model_roundtrip(tmp_path):
    df = make_tiny_df()
    model = train.train_model(df, text_col="text")

    model_path = tmp_path / "xgb_model.joblib"
    train.save_model(model, model_path)
    assert model_path.exists()

    loaded_model = train.load_model(model_path)
    # check loaded model works
    emb = get_embedding("Horrible!")
    emb = emb.reshape(1, -1)
    pred = loaded_model.predict(emb)
    assert pred.shape == (1,)
