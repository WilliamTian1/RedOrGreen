from __future__ import annotations

from dataclasses import dataclass
from typing import List

import json
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PersistenceBundle:
    model_path: str
    sector_encoder_path: str
    scaler_path: str
    feature_schema_path: str
    label_encoder_path: str


def save_persistence(
    artifacts_dir: str,
    model,
    sector_encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_names: List[str],
) -> PersistenceBundle:
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "sk_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    sector_encoder_path = os.path.join(artifacts_dir, "sector_encoder.pkl")
    with open(sector_encoder_path, "wb") as f:
        pickle.dump(sector_encoder, f)
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    feature_schema_path = os.path.join(artifacts_dir, "feature_schema.json")
    with open(feature_schema_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    # For future use: binary label encoder stub (UP/DOWN mapping)
    label_encoder_path = os.path.join(artifacts_dir, "label_encoder.json")
    with open(label_encoder_path, "w", encoding="utf-8") as f:
        json.dump({"DOWN": 0, "UP": 1}, f, indent=2)
    return PersistenceBundle(
        model_path=model_path,
        sector_encoder_path=sector_encoder_path,
        scaler_path=scaler_path,
        feature_schema_path=feature_schema_path,
        label_encoder_path=label_encoder_path,
    )


def load_persistence(artifacts_dir: str) -> PersistenceBundle:
    return PersistenceBundle(
        model_path=os.path.join(artifacts_dir, "sk_model.pkl"),
        sector_encoder_path=os.path.join(artifacts_dir, "sector_encoder.pkl"),
        scaler_path=os.path.join(artifacts_dir, "scaler.pkl"),
        feature_schema_path=os.path.join(artifacts_dir, "feature_schema.json"),
        label_encoder_path=os.path.join(artifacts_dir, "label_encoder.json"),
    )


