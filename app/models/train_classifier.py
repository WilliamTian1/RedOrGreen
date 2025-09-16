from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import os
import pickle
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.utils.metrics import compute_classification_metrics, plot_confusion_matrix, plot_roc, save_feature_importance
from app.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class TrainedModel:
    model: object
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_cfg: Dict,
    artifacts_dir: str,
    experiment_name: str,
    use_wandb: bool = False,
) -> Tuple[TrainedModel, Dict[str, float]]:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    model_type = model_cfg.get("type", "logistic_regression")
    if model_type == "logistic_regression":
        lr_cfg = model_cfg.get("logistic_regression", {})
        model = LogisticRegression(
            C=float(lr_cfg.get("C", 1.0)),
            penalty=str(lr_cfg.get("penalty", "l2")),
            solver=str(lr_cfg.get("solver", "liblinear")),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
        )
    else:
        rf_cfg = model_cfg.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=int(rf_cfg.get("max_depth", 6)),
            n_jobs=-1,
            random_state=42,
        )

    os.makedirs(artifacts_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": model_type,
            **{f"lr_{k}": v for k, v in model_cfg.get("logistic_regression", {}).items()},
            **{f"rf_{k}": v for k, v in model_cfg.get("random_forest", {}).items()},
        })

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Artifacts
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc(y_test, y_proba, roc_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        # Feature importance if available
        if hasattr(model, "feature_importances_"):
            fi_path = os.path.join(artifacts_dir, "feature_importance.json")
            save_feature_importance(feature_names, model.feature_importances_, fi_path)
            mlflow.log_artifact(fi_path)

        # Save model
        model_path = os.path.join(artifacts_dir, "sk_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)

    trained = TrainedModel(model=model, y_test=y_test, y_pred=y_pred, y_proba=y_proba)
    return trained, metrics


