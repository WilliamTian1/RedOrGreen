import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import yaml


@dataclass
class Config:
    seed: int
    use_lora: bool
    use_wandb: bool
    experiment_name: str
    mlflow_tracking_uri: str
    artifacts_dir: str
    features: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    paths: Dict[str, str]
    yfinance_offline: bool


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Env overrides
    data["use_lora"] = _env_bool("USE_LORA", data.get("use_lora", False))
    data["use_wandb"] = _env_bool("USE_WANDB", data.get("use_wandb", False))
    data["mlflow_tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI", data.get("mlflow_tracking_uri", "./mlruns"))
    data["yfinance_offline"] = _env_bool("YFINANCE_OFFLINE", data.get("yfinance_offline", False))

    return Config(
        seed=int(data.get("seed", 42)),
        use_lora=bool(data.get("use_lora", False)),
        use_wandb=bool(data.get("use_wandb", False)),
        experiment_name=str(data.get("experiment_name", "earnings-analyzer")),
        mlflow_tracking_uri=str(data.get("mlflow_tracking_uri", "./mlruns")),
        artifacts_dir=str(data.get("artifacts_dir", "artifacts")),
        features=dict(data.get("features", {})),
        model=dict(data.get("model", {})),
        train=dict(data.get("train", {})),
        paths=dict(data.get("paths", {})),
        yfinance_offline=bool(data.get("yfinance_offline", False)),
    )


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism toggles (best-effort)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}




