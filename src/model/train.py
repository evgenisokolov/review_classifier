import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.features.embeddings import generate_embeddings
from src.utils.logger import init_logger

logger = init_logger(name="review_classifier_train")

def _map_rating_to_class(rating: float) -> int:
    if rating <= 2:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:
        return 2  # positive

def _train_model(df: pd.DataFrame, text_col: str = "text") -> XGBClassifier:
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

def _evaluate_model(model: XGBClassifier, df: pd.DataFrame, text_col: str = "text") -> dict:
    """
    Evaluate model on a DataFrame with text+rating.
    Returns a dict with accuracy and classification report.
    """
    y_true = df["rating"].apply(_map_rating_to_class)
    df_emb = generate_embeddings(df, text_col=text_col, show_progress_bar=False)
    x = np.vstack(df_emb[f"{text_col}_embedding"].values)

    y_pred = model.predict(x)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"accuracy": acc, "classification_report": report}

def _save_model(model: XGBClassifier, path: str):
    joblib.dump(model, path)

def load_model(path: str) -> XGBClassifier:
    return joblib.load(path)

def run_training_pipeline(
    data_path: str,
    text_col: str = "text",
    test_size: float = 0.2,
    model_path: str = "models/xgb_sentiment.joblib",
    metrics_path: str = "models/metrics.json"
):
    from src.data.dataset import load_reviews_file, preprocess_reviews
    import json

    logger.info("Loading and preprocessing data", extra={"data_path": data_path})
    df_raw = load_reviews_file(data_path)
    df = preprocess_reviews(df_raw, text_column=text_col, sentence_split=True)

    logger.info("Splitting into train/eval sets", extra={"test_size": test_size})
    from sklearn.model_selection import train_test_split
    train_df, eval_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["rating"])

    logger.info("Training model", extra={"n_train_rows": len(train_df)})
    model = _train_model(train_df, text_col=text_col)

    logger.info("Evaluating model", extra={"n_eval_rows": len(eval_df)})
    metrics = _evaluate_model(model, eval_df, text_col=text_col)

    _save_model(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Pipeline complete",
                extra={"model_path": model_path, "metrics_path": metrics_path,
                       "accuracy": metrics["accuracy"]})

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--data-path", required=True, help="Path to input JSONL reviews file")
    parser.add_argument("--text-col", default="text", help="Column name containing review text")
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split size")
    parser.add_argument("--model-path", default="models/xgb_sentiment.joblib", help="Where to save trained model")
    parser.add_argument("--metrics-path", default="models/metrics.json", help="Where to save metrics JSON")

    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data_path,
        text_col=args.text_col,
        test_size=args.test_size,
        model_path=args.model_path,
        metrics_path=args.metrics_path
    )