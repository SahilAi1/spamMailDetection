"""Unit tests for Spam Mail Detection pipeline and preprocessing."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    clean_text,
    detect_columns,
    normalize_labels,
    split_dataset,
)

MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"


def test_clean_text_basic():
    """Verify basic text cleaning functionality."""
    raw = "Subject: WIN $1,000,000 NOW! Visit http://promo.example.com or email win@prize.com."
    cleaned = clean_text(raw)

    assert "subject:" not in cleaned
    assert "win" in cleaned
    assert "currencysymb" in cleaned
    assert "httpaddr" in cleaned
    assert "emailaddr" in cleaned
    assert "numtoken" in cleaned


def test_clean_text_edge_cases():
    """Verify handling of None, empty strings, and non-string inputs."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text("   \n\t  ") == ""


def test_detect_columns():
    """Verify automatic column detection with various naming schemes."""
    # Standard format
    df1 = pd.DataFrame({"text": ["Hello"], "label_num": [0]})
    text_col, label_col = detect_columns(df1)
    assert text_col == "text"
    assert label_col == "label_num"

    # Alternative Kaggle SMS format
    df2 = pd.DataFrame({"v2": ["Hello"], "v1": ["ham"]})
    text_col, label_col = detect_columns(df2)
    assert text_col == "v2"
    assert label_col == "v1"

    # Custom column names
    df3 = pd.DataFrame({"message": ["Win cash"], "category": ["spam"]})
    text_col, label_col = detect_columns(df3)
    assert text_col == "message"
    assert label_col == "category"


def test_normalize_labels():
    """Verify mapping of various string labels to 0 and 1."""
    df = pd.DataFrame({"label": ["ham", "spam", "HAM", "SPAM", "Not Spam", "legitimate"]})
    normalized = normalize_labels(df, "label")

    expected_nums = [0, 1, 0, 1, 0, 0]
    expected_names = ["Not Spam", "Spam", "Not Spam", "Spam", "Not Spam", "Not Spam"]

    assert list(normalized["label_num"]) == expected_nums
    assert list(normalized["label_name"]) == expected_names


def test_split_dataset():
    """Verify stratified split preserves proportional splits."""
    df = pd.DataFrame({
        "cleaned_text": [f"msg {i}" for i in range(100)],
        "label_num": [0] * 70 + [1] * 30,
    })
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert sum(y_train) == 24  # 30 * 0.8
    assert sum(y_test) == 6    # 30 * 0.2


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Model artifact not yet generated")
def test_saved_model_inference():
    """Verify that the saved joblib pipeline loads and predicts valid classes and probabilities."""
    pipeline = joblib.load(MODEL_PATH)
    assert isinstance(pipeline, Pipeline)

    spam_sample = "Urgent! You have won a guaranteed £1000 cash prize. Claim now by calling 09050000321."
    ham_sample = "Hi Sarah, can you please review the attached slide deck before our 3pm sync today?"

    pred_spam = pipeline.predict([spam_sample])[0]
    proba_spam = pipeline.predict_proba([spam_sample])[0]

    pred_ham = pipeline.predict([ham_sample])[0]
    proba_ham = pipeline.predict_proba([ham_sample])[0]

    assert pred_spam in [0, 1]
    assert pred_ham in [0, 1]
    assert len(proba_spam) == 2
    assert abs(sum(proba_spam) - 1.0) < 1e-4
    assert abs(sum(proba_ham) - 1.0) < 1e-4

    # Spam probability for spam sample should be higher than for ham sample
    assert proba_spam[1] > proba_ham[1]
