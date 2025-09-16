from typing import Dict, Tuple

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(y_true, y_proba, save_path: str) -> None:
    try:
        RocCurveDisplay.from_predictions(y_true, y_proba)
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception:
        # Handle edge cases where ROC can't be computed
        plt.figure()
        plt.text(0.5, 0.5, "ROC not available", ha="center", va="center")
        plt.savefig(save_path)
        plt.close()


def save_feature_importance(feature_names, importances, save_json_path: str) -> None:
    data = {name: float(val) for name, val in zip(feature_names, importances)}
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



