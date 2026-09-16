"""Training script for Spam Mail Detection System.

Builds, trains, evaluates, and serializes the complete Scikit-Learn Pipeline
(text cleaning, TF-IDF feature extraction, and Multinomial Naive Bayes classifier).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import urllib.request
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    clean_text,
    load_and_preprocess_data,
    split_dataset,
)
from src.evaluate import (
    compute_metrics,
    plot_and_save_confusion_matrix,
    print_evaluation_summary,
    save_metrics_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "spam_ham_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DEFAULT_CM_PATH = PROJECT_ROOT / "models" / "confusion_matrix.png"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATASET_REMOTE_URL = "https://raw.githubusercontent.com/kairess/toy-datasets/master/spam_ham_dataset.csv"


def download_dataset_if_missing(data_path: Path) -> None:
    """Download default dataset if it does not already exist."""
    if not data_path.exists():
        logger.info(f"Dataset not found at {data_path}. Downloading from mirror...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_REMOTE_URL, str(data_path))
        logger.info(f"Dataset downloaded successfully to: {data_path}")


def build_pipeline(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 10000,
    sublinear_tf: bool = True,
    alpha: float = 0.1,
) -> Pipeline:
    """Construct an end-to-end Scikit-Learn Pipeline.

    Bundles text preprocessing, TF-IDF feature extraction, and
    Multinomial Naive Bayes classification.

    Args:
        ngram_range: Lower and upper boundary for n-grams.
        max_features: Maximum number of vocabulary features.
        sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf)).
        alpha: Additive (Laplace/Lidstone) smoothing parameter for Naive Bayes.

    Returns:
        Configured scikit-learn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )
    classifier = MultinomialNB(alpha=alpha)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    return pipeline


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 0.1,
    ngram_max: int = 2,
    max_features: int = 10000,
) -> tuple[Pipeline, dict]:
    """Train, evaluate, and save the spam detection model pipeline.

    Args:
        data_path: Path to dataset CSV file.
        model_path: Destination path for serialized joblib pipeline.
        test_size: Test split ratio.
        random_state: Seed for reproducible data splitting.
        alpha: Laplace smoothing parameter.
        ngram_max: Maximum n-gram range.
        max_features: Vocabulary size limit.

    Returns:
        Tuple of (fitted_pipeline, metrics_dict).
    """
    download_dataset_if_missing(data_path)

    # 1. Load and clean data
    logger.info(f"Loading data from: {data_path}")
    df = load_and_preprocess_data(data_path, drop_duplicates=True)

    # 2. Split dataset reproducibly
    # Note: We pass raw original_text into train_test_split because the Pipeline's
    # TfidfVectorizer has preprocessor=clean_text, ensuring identical cleaning during
    # both training and real-time inference.
    X = df["original_text"]
    y = df["label_num"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples "
        f"(Spam test samples: {(y_test == 1).sum()}, Ham test samples: {(y_test == 0).sum()})"
    )

    # 3. Build scikit-learn Pipeline
    pipeline = build_pipeline(
        ngram_range=(1, ngram_max),
        max_features=max_features,
        alpha=alpha,
    )

    # 4. Train pipeline strictly on training data
    logger.info(f"Training Multinomial Naive Bayes pipeline (alpha={alpha})...")
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline training complete.")

    # 5. Evaluate on held-out test set
    logger.info("Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, target_names=["Not Spam", "Spam"])
    print_evaluation_summary(metrics)

    # 6. Save model artifact
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(model_path))
    logger.info(f"Saved trained pipeline to: {model_path}")

    # 7. Save metrics and confusion matrix
    save_metrics_json(metrics, DEFAULT_METRICS_PATH)
    plot_and_save_confusion_matrix(metrics["confusion_matrix"]["matrix"], DEFAULT_CM_PATH)

    # 8. Save metadata
    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": {
            "source_file": str(data_path.name),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": test_size,
            "random_state": random_state,
        },
        "model": {
            "pipeline_steps": [name for name, _ in pipeline.steps],
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, ngram_max],
            "max_features": max_features,
            "classifier": "MultinomialNB",
            "alpha": alpha,
            "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "spam_precision": metrics["spam_precision"],
            "spam_recall": metrics["spam_recall"],
            "spam_f1": metrics["spam_f1"],
            "ham_precision": metrics["ham_precision"],
            "ham_recall": metrics["ham_recall"],
            "ham_f1": metrics["ham_f1"],
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
        },
    }
    with open(DEFAULT_METADATA_PATH, "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {DEFAULT_METADATA_PATH}")

    return pipeline, metrics


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(description="Train Spam Mail Detection Naive Bayes Model")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to dataset CSV")
    parser.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save joblib model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Laplace smoothing alpha (default: 0.1)")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size (default: 2)")
    parser.add_argument("--max-features", type=int, default=10000, help="Max TF-IDF features (default: 10000)")

    args = parser.parse_args()

    train_model(
        data_path=Path(args.data),
        model_path=Path(args.model_out),
        test_size=args.test_size,
        random_state=args.random_state,
        alpha=args.alpha,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
