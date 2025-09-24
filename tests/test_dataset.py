import pandas as pd
import pytest
from src.data.dataset import load_reviews_file, preprocess_reviews, _clean_text

def test_load_reviews_file(tmp_path):
    # create a small jsonl file
    jsonl = tmp_path / "reviews.jsonl"
    lines = [
        '{"rating":5,"text":"<p>Hello world.</p>"}',
        '{"rating":1,"text":"Another review."}'
    ]
    jsonl.write_text("\n".join(lines), encoding="utf-8")

    df = load_reviews_file(jsonl)
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert len(df) == 2

def test_clean_text_strips_html():
    raw = "<p>Hello <b>world</b></p>"
    cleaned = _clean_text(raw)
    assert "<" not in cleaned  # HTML removed
    assert "world" in cleaned

@pytest.mark.parametrize("sentence_split,expected_rows", [(True, 2), (False, 1)])
def test_preprocess_reviews_sentence_split(sentence_split, expected_rows):
    df = pd.DataFrame({
        "rating": [5],
        "text": ["This is first sentence. This is second."]
    })
    out = preprocess_reviews(df, text_column="text", sentence_split=sentence_split)
    assert isinstance(out, pd.DataFrame)
    # should always have "text" column
    assert "text" in out.columns
    assert len(out) == expected_rows
    # rating column should be duplicated
    assert (out["rating"] == 5).all()
