"""Data preprocessing module for Spam Mail Detection.

Handles data loading, dynamic column mapping, duplicate removal,
missing value handling, text cleaning, label normalization, and train-test splitting.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Candidate column names for automatic schema detection
TEXT_CANDIDATES = ["text", "message", "email", "sms", "v2", "body", "content", "email_text", "mail"]
LABEL_CANDIDATES = ["label", "label_num", "v1", "category", "target", "class", "spam", "type"]


def clean_text(text: str) -> str:
    """Clean and normalize email text.

    Steps:
    1. Convert text to lowercase.
    2. Remove leading email header artifacts (e.g. 'Subject:').
    3. Replace URLs with a normalized token ('httpaddr').
    4. Replace email addresses with a normalized token ('emailaddr').
    5. Replace currency symbols with a normalized token ('currencysymb').
    6. Replace numbers with a normalized token ('numtoken').
    7. Remove punctuation / non-alphanumeric symbols except whitespace.
    8. Collapse excessive whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove leading 'Subject:' or 'Re:' prefixes typical in email corpora
    text = re.sub(r"^(subject|re|fwd):\s*", "", text, flags=re.IGNORECASE)

    # Replace web URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " httpaddr ", text)

    # Replace email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " emailaddr ", text)

    # Replace currency symbols ($ € £ ¥ ₹)
    text = re.sub(r"[$€£¥₹]", " currencysymb ", text)

    # Replace standalone numbers / numeric quantities
    text = re.sub(r"\b\d+\b", " numtoken ", text)

    # Remove punctuation and special characters, retaining word characters and whitespace
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize excessive spaces, tabs, and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Automatically detect text and label column names from candidate lists.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (text_column_name, label_column_name).

    Raises:
        ValueError: If suitable text or label column cannot be identified.
    """
    cols_lower = {col.lower(): col for col in df.columns}

    # Find text column
    text_col = None
    for candidate in TEXT_CANDIDATES:
        if candidate in cols_lower:
            text_col = cols_lower[candidate]
            break

    # Find label column
    label_col = None
    # Prefer label_num if present, then label, etc.
    if "label_num" in cols_lower:
        label_col = cols_lower["label_num"]
    else:
        for candidate in LABEL_CANDIDATES:
            if candidate in cols_lower:
                label_col = cols_lower[candidate]
                break

    if not text_col:
        raise ValueError(
            f"Could not automatically detect text column. Available columns: {list(df.columns)}. "
            f"Expected one of: {TEXT_CANDIDATES}"
        )
    if not label_col:
        raise ValueError(
            f"Could not automatically detect label column. Available columns: {list(df.columns)}. "
            f"Expected one of: {LABEL_CANDIDATES}"
        )

    logger.info(f"Detected columns: text='{text_col}', label='{label_col}'")
    return text_col, label_col


def normalize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Normalize label column into standard 0 (Not Spam / Ham) and 1 (Spam).

    Args:
        df: Input DataFrame.
        label_col: Name of the label column.

    Returns:
        DataFrame with added 'label_num' (0 or 1) and 'label_name' ('Not Spam' or 'Spam').
    """
    df = df.copy()
    col = df[label_col]

    # If already integer/float 0 and 1
    if pd.api.types.is_numeric_dtype(col):
        df["label_num"] = col.astype(int)
    else:
        mapping = {
            "ham": 0,
            "not spam": 0,
            "legitimate": 0,
            "normal": 0,
            "0": 0,
            "spam": 1,
            "junk": 1,
            "phishing": 1,
            "1": 1,
        }
        lowered = col.astype(str).str.strip().str.lower()
        if not lowered.isin(mapping.keys()).all():
            unrecognized = set(lowered[~lowered.isin(mapping.keys())].unique())
            logger.warning(f"Unrecognized labels encountered: {unrecognized}. Attempting best-effort mapping.")
        df["label_num"] = lowered.map(mapping).fillna(0).astype(int)

    df["label_name"] = df["label_num"].map({0: "Not Spam", 1: "Spam"})
    return df


def load_and_preprocess_data(
    file_path: str | Path,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load dataset, handle missing values, drop duplicates, and clean text.

    Args:
        file_path: Path to dataset CSV file.
        drop_duplicates: Whether to drop duplicate text messages.

    Returns:
        Cleaned DataFrame with 'original_text', 'cleaned_text', 'label_num', 'label_name'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    # Try common encodings
    encodings = ["utf-8", "latin1", "windows-1252"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"Successfully loaded dataset with encoding='{enc}'. Shape: {df.shape}")
            break
        except (UnicodeDecodeError, Exception) as e:
            logger.debug(f"Failed loading with {enc}: {e}")

    if df is None:
        raise RuntimeError(f"Failed to read CSV at {path} with encodings: {encodings}")

    # Detect columns
    text_col, label_col = detect_columns(df)

    initial_count = len(df)

    # Handle missing values
    df = df.dropna(subset=[text_col, label_col]).copy()
    after_dropna_count = len(df)
    if initial_count - after_dropna_count > 0:
        logger.info(f"Dropped {initial_count - after_dropna_count} records with missing values.")

    # Drop duplicate text records
    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_col]).copy()
        after_dedup_count = len(df)
        dropped_dups = after_dropna_count - after_dedup_count
        if dropped_dups > 0:
            logger.info(f"Dropped {dropped_dups} duplicate records. Remaining: {after_dedup_count}")

    # Normalize labels
    df = normalize_labels(df, label_col)

    # Retain original text and create cleaned text
    df["original_text"] = df[text_col].astype(str)
    df["cleaned_text"] = df["original_text"].apply(clean_text)

    # Filter out records that became empty after cleaning
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)

    logger.info(
        f"Preprocessing complete. Total samples: {len(df)} | "
        f"Not Spam (Ham): {(df['label_num'] == 0).sum()} | "
        f"Spam: {(df['label_num'] == 1).sum()}"
    )

    return df[["original_text", "cleaned_text", "label_num", "label_name"]]


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset into training and testing partitions.

    Args:
        df: Preprocessed DataFrame containing 'cleaned_text' and 'label_num'.
        test_size: Proportion of dataset to include in the test split (default 0.2).
        random_state: Random seed for reproducibility (default 42).
        stratify: Whether to preserve class proportions across splits (default True).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df["cleaned_text"]
    y = df["label_num"]

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples (test_size={test_size})"
    )
    return X_train, X_test, y_train, y_test
