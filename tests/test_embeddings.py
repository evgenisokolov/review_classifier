import numpy as np
import pandas as pd
import pytest
from src.features.embeddings import get_embedding, generate_embeddings, _get_model

def test_get_model_singleton():
    m1 = _get_model()
    m2 = _get_model()
    assert m1 is m2  # singleton pattern works

def test_get_embedding_shape():
    emb = get_embedding("This is a test")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1  # 1D vector
    assert emb.size > 0

def test_generate_embeddings_adds_column():
    df = pd.DataFrame({"text": ["This is one.", "And another."]})
    out = generate_embeddings(df, text_col="text", show_progress_bar=False)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df)
    assert "text_embedding" in out.columns
    # check one embedding
    first_emb = out["text_embedding"].iloc[0]
    assert isinstance(first_emb, np.ndarray)
    assert first_emb.ndim == 1
    assert first_emb.size > 0

@pytest.mark.parametrize("text_col", ["text"])
def test_generate_embeddings_text_col_param(text_col):
    df = pd.DataFrame({text_col: ["Sentence"]})
    out = generate_embeddings(df, text_col=text_col, show_progress_bar=False)
    assert f"{text_col}_embedding" in out.columns
