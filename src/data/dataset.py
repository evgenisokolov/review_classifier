
import pandas as pd
import nltk
import re
from typing import Union
from pathlib import Path
from bs4 import BeautifulSoup  # for HTML tag removal

# ensure punkt + punkt_tab are present
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)
sent_tokenize = nltk.sent_tokenize


def load_reviews_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a JSONL file into a DataFrame (one row per review).
    """
    return pd.read_json(file_path, lines=True)


def _clean_text(text: str) -> str:
    """
    Remove HTML tags and excessive whitespace from text.
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews(
    df: pd.DataFrame,
    text_column: str = "text",
    sentence_split: bool = True
) -> pd.DataFrame:
    """
    Clean and optionally split review text into sentences.

    Args:
        df: raw reviews DataFrame (one row per review).
        text_column: column name containing review text.
        sentence_split: if True, split into sentences.

    Returns:
        DataFrame with unified "text" column (one row per sentence or review).
    """
    rows = []
    for _, row in df.iterrows():
        raw_text = row.get(text_column, "")
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        if sentence_split:
            parts = sent_tokenize(cleaned)
        else:
            parts = [cleaned]

        for part in parts:
            new_row = row.to_dict()
            new_row["text"] = part  # overwrite with sentence or full review
            rows.append(new_row)

    return pd.DataFrame(rows)
