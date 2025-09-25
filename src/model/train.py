import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from src.features.embeddings import generate_embeddings

def _map_rating_to_class(rating: float) -> int:
    if rating <= 2:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:
        return 2  # positive

def train_model(df: pd.DataFrame, text_col: str = "text") -> XGBClassifier:
    # Map ratings to 0/1/2
    y = df["rating"].apply(_map_rating_to_class)

    # Generate embeddings
    df_emb = generate_embeddings(df, text_col=text_col, show_progress_bar=True)
    x = np.vstack(df_emb[f"{text_col}_embedding"].values)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist"  # fast on CPU
    )
    model.fit(x, y)
    return model

def save_model(model: XGBClassifier, path: str):
    joblib.dump(model, path)

def load_model(path: str) -> XGBClassifier:
    return joblib.load(path)
