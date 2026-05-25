"""Metrics and plots for respiratory sound classification."""

import json
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


ASTHMA_LABEL = "asthma"
NOT_ASTHMA_LABEL = "not asthma"


def normalize_label(value: Any, json_field: str | None = "diagnosis") -> str:
    """Normalize plain labels or JSON diagnosis strings."""
    if isinstance(value, dict) and json_field is not None:
        value = value.get(json_field, value)
    elif isinstance(value, str) and json_field is not None:
        stripped = value.strip()
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = stripped
            if isinstance(decoded, dict):
                value = decoded.get(json_field, stripped)
            else:
                value = decoded

    return str(value).strip().lower()


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Return zero for undefined rates."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def binary_classification_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    positive_label: str = ASTHMA_LABEL,
    negative_label: str = NOT_ASTHMA_LABEL,
    json_field: str | None = "diagnosis",
) -> dict[str, float | int]:
    """Compute binary metrics for a configured label pair."""
    positive = normalize_label(positive_label, json_field=json_field)
    negative = normalize_label(negative_label, json_field=json_field)
    true_labels = [normalize_label(label, json_field=json_field) for label in y_true]
    pred_labels = [normalize_label(label, json_field=json_field) for label in y_pred]

    if len(true_labels) != len(pred_labels):
        raise ValueError("y_true and y_pred must have the same length")

    allowed_labels = {positive, negative}
    unsupported = sorted((set(true_labels) | set(pred_labels)) - allowed_labels)
    if unsupported:
        raise ValueError(f"Unsupported binary labels: {unsupported}")

    tp = sum(
        true == positive and pred == positive
        for true, pred in zip(true_labels, pred_labels)
    )
    fp = sum(
        true == negative and pred == positive
        for true, pred in zip(true_labels, pred_labels)
    )
    tn = sum(
        true == negative and pred == negative
        for true, pred in zip(true_labels, pred_labels)
    )
    fn = sum(
        true == positive and pred == negative
        for true, pred in zip(true_labels, pred_labels)
    )

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    f1 = _ratio(2 * precision * sensitivity, precision + sensitivity)

    return {
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "youden": sensitivity + specificity - 1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def binary_metrics_from_columns(
    frame: pd.DataFrame,
    label_column: str,
    prediction_column: str,
    positive_label: str = ASTHMA_LABEL,
    negative_label: str = NOT_ASTHMA_LABEL,
    json_field: str | None = "diagnosis",
) -> dict[str, float | int]:
    """Compute binary metrics from dataframe columns."""
    return binary_classification_metrics(
        frame[label_column],
        frame[prediction_column],
        positive_label=positive_label,
        negative_label=negative_label,
        json_field=json_field,
    )


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str) -> None:
    """Plot and save a confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
