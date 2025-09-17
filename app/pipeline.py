from __future__ import annotations

import argparse
import os
from typing import Optional

import joblib
import mlflow
import numpy as np
import pandas as pd

from app.utils.config import load_config, set_seeds
from app.utils.logging import get_logger
from app.ingest.load_text import load_jsonl
from app.ingest.news_feeds import aggregate_live_news, save_live_news_cache, load_live_news_cache
from app.features.build_features import (
    build_features,
    load_ticker_sector_map,
)
from app.models.train_classifier import train_and_evaluate
from app.models.persist import save_persistence
from app.models.lora_finetune import maybe_lora_finetune


logger = get_logger(__name__)


def ingest_samples(path: str, use_live_news: bool = False, newsapi_key: Optional[str] = None) -> pd.DataFrame:
    """Load samples from file or live news sources."""
    if use_live_news:
        logger.info("Fetching live news data...")
        df = aggregate_live_news(newsapi_key=newsapi_key)
        if not df.empty:
            save_live_news_cache(df)
            return df
        else:
            logger.warning("Live news fetch failed, trying cache...")
            df = load_live_news_cache()
            if not df.empty:
                return df
            logger.warning("No cached news, falling back to sample data")
    
    return load_jsonl(path)


def run_pipeline(use_lora: Optional[bool] = None, limit_samples: int = 0, use_live_news: bool = False) -> None:
    cfg = load_config()
    set_seeds(cfg.seed)

    if use_lora is None:
        use_lora = cfg.use_lora

    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    # Respect offline flags to avoid network/model downloads during demo
    if cfg.yfinance_offline:
        os.environ["YFINANCE_OFFLINE"] = "true"
        os.environ["FINBERT_OFFLINE"] = "true"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # LoRA (optional)
    if use_lora:
        maybe_lora_finetune(enabled=True)

    # Ingest
    newsapi_key = os.getenv("NEWSAPI_KEY")
    df = ingest_samples(
        cfg.paths["sample_earnings"], 
        use_live_news=use_live_news,
        newsapi_key=newsapi_key
    )
    if limit_samples and limit_samples > 0:
        df = df.head(limit_samples)
    logger.info(f"Loaded {len(df)} samples.")

    mapping_df = load_ticker_sector_map(cfg.paths["ticker_sector_map"])

    # Features
    X, y, feats, enriched = build_features(
        df,
        mapping_df,
        include_price_features=cfg.features.get("include_price_features", True),
        price_lookback_days=cfg.features.get("price_lookback_days", 5),
        yfinance_offline=cfg.yfinance_offline,
    )
    logger.info(f"Feature matrix shape: {X.shape}")

    # Train/Eval
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    trained, metrics = train_and_evaluate(
        X,
        y,
        feats.feature_names,
        cfg.model,
        cfg.artifacts_dir,
        cfg.experiment_name,
        use_wandb=cfg.use_wandb,
    )

    # Persist
    _bundle = save_persistence(
        cfg.artifacts_dir,
        trained.model,
        feats.sector_encoder,
        feats.scaler,
        feats.feature_names,
    )
    logger.info("Saved model and preprocessing artifacts.")


def main():
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--use_lora", type=str, default="false", help="true/false to enable LoRA")
    parser.add_argument("--limit_samples", type=int, default=0, help="Limit number of samples for quick runs")
    parser.add_argument("--live_news", action="store_true", help="Fetch live news instead of using sample data")
    args = parser.parse_args()

    use_lora = args.use_lora.lower() in {"1", "true", "yes", "on"}
    run_pipeline(use_lora=use_lora, limit_samples=args.limit_samples, use_live_news=args.live_news)


if __name__ == "__main__":
    main()



