"""Evaluation module for Spam Mail Detection.

Computes classification metrics: Accuracy, Precision, Recall, F1-Score,
Confusion Matrix, and full Classification Report. Generates metric dictionaries
and confusion matrix visualization artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    target_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0 = Not Spam, 1 = Spam).
        y_pred: Predicted binary labels.
        target_names: Display names for classes (default: ['Not Spam', 'Spam']).

    Returns:
        Dictionary of computed metrics.
    """
    if target_names is None:
        target_names = ["Not Spam", "Spam"]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spam_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "spam_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "ham_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "ham_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "ham_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        ),
    }

    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """Print a clean, formatted evaluation summary to stdout.

    Args:
        metrics: Metrics dictionary computed by compute_metrics().
    """
    cm = metrics["confusion_matrix"]
    total = sum([cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]])

    print("\n" + "=" * 60)
    print("           MODEL EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 60)
    print(f"Total Test Samples:        {total}")
    print(f"Overall Accuracy:          {metrics['accuracy'] * 100:.2f}%")
    print("-" * 60)
    print("Spam Detection (Positive Class = 1):")
    print(f"  * Precision:             {metrics['spam_precision'] * 100:.2f}%  (When predicted Spam, how often is it Spam?)")
    print(f"  * Recall:                {metrics['spam_recall'] * 100:.2f}%  (Out of all actual Spam, how many caught?)")
    print(f"  * F1-Score:              {metrics['spam_f1'] * 100:.2f}%")
    print("-" * 60)
    print("Legitimate Mail (Negative Class = 0):")
    print(f"  * Precision:             {metrics['ham_precision'] * 100:.2f}%")
    print(f"  * Recall:                {metrics['ham_recall'] * 100:.2f}%  (Minimizing false positives protects real mail)")
    print(f"  * F1-Score:              {metrics['ham_f1'] * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  [TN: {cm['true_negatives']:4d}] (Legitimate correctly identified)")
    print(f"  [FP: {cm['false_positives']:4d}] (Legitimate misclassified as Spam - FALSE ALARM)")
    print(f"  [FN: {cm['false_negatives']:4d}] (Spam missed - SLIPPED INTO INBOX)")
    print(f"  [TP: {cm['true_positives']:4d}] (Spam correctly blocked)")
    print("=" * 60 + "\n")


def plot_and_save_confusion_matrix(
    cm_matrix: list[list[int]],
    output_path: str | Path,
    target_names: Optional[list[str]] = None,
) -> None:
    """Save an aesthetic confusion matrix heatmap image.

    Args:
        cm_matrix: 2x2 confusion matrix nested list.
        output_path: Output file path for PNG image.
        target_names: Labels for axes.
    """
    if target_names is None:
        target_names = ["Not Spam (Ham)", "Spam"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_title("Spam Mail Detection - Confusion Matrix", fontsize=13, pad=12, weight="bold")
    ax.set_xlabel("Predicted Category", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Category", fontsize=11, labelpad=8)
    plt.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to: {output_path}")


def save_metrics_json(metrics: Dict[str, Any], output_path: str | Path) -> None:
    """Save metrics dictionary to a JSON file.

    Args:
        metrics: Metrics dictionary.
        output_path: Path to target JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON to: {path}")
