from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Singleton for the embedding model
_model_instance: SentenceTransformer = None


def _get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Returns a singleton SentenceTransformer model instance.
    Loads the model once and reuses it.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


def get_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embedding for a single sentence or text.
    """
    model = _get_model(model_name)
    embedding = model.encode([text], show_progress_bar=False)
    return embedding[0]


def generate_embeddings(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Generate embeddings for all rows in df[text_col] and attach them
    as a new column f'{text_col}_embedding'.
    Returns the modified DataFrame.
    """
    model = _get_model(model_name)
    texts: List[str] = df[text_col].tolist()
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress_bar
    )

    df = df.copy()
    df[f"{text_col}_embedding"] = list(embeddings)
    return df
