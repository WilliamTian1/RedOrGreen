from __future__ import annotations

import argparse
import json
import os
from typing import List

import pickle
import numpy as np
import pandas as pd

from app.ingest.load_text import parse_snippet
from app.features.build_features import load_ticker_sector_map, map_sector
from app.models.finbert_infer import FinBERTSentiment
from app.ingest.market_data import download_prices_yf, compute_price_features
from app.utils.config import load_config
from app.utils.logging import get_logger


logger = get_logger(__name__)


def _load_artifacts(cfg):
    art_dir = cfg.artifacts_dir
    try:
        with open(os.path.join(art_dir, "sk_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(art_dir, "sector_encoder.pkl"), "rb") as f:
            sector_encoder = pickle.load(f)
        with open(os.path.join(art_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(art_dir, "feature_schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        return model, sector_encoder, scaler, schema
    except Exception:
        # Bootstrap: run a quick offline pipeline to produce artifacts
        logger.warning("Artifacts missing; running a quick offline training to bootstrap.")
        os.environ["YFINANCE_OFFLINE"] = "true"
        os.environ["FINBERT_OFFLINE"] = "true"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from app.pipeline import run_pipeline
            run_pipeline(use_lora=False, limit_samples=0)
        except Exception as e:
            logger.error(f"Bootstrap training failed: {e}")
            raise
        with open(os.path.join(art_dir, "sk_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(art_dir, "sector_encoder.pkl"), "rb") as f:
            sector_encoder = pickle.load(f)
        with open(os.path.join(art_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(art_dir, "feature_schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        return model, sector_encoder, scaler, schema


def analyze(text: str) -> str:
    cfg = load_config()
    mapping_df = load_ticker_sector_map(cfg.paths["ticker_sector_map"]) 
    model, sector_encoder, scaler, schema = _load_artifacts(cfg)

    parsed = parse_snippet(text)
    try:
        sector = map_sector(parsed.ticker, mapping_df)
    except Exception:
        sector = "Unknown"

    # Sentiment
    finbert = FinBERTSentiment()
    sent = finbert.predict(parsed.text)

    # Price features
    if cfg.features.get("include_price_features", True) and not cfg.yfinance_offline:
        px = download_prices_yf(parsed.ticker, lookback_days=max(cfg.features.get("price_lookback_days", 5), 10))
        ret_5d, vol_5d = compute_price_features(px, lookback_days=cfg.features.get("price_lookback_days", 5))
    else:
        ret_5d, vol_5d = 0.0, 0.0

    # Assemble features in same order as training
    num_features = np.array([[sent.logits[0], sent.logits[1], sent.logits[2], sent.score, ret_5d, vol_5d]])
    num_scaled = scaler.transform(num_features)
    sector_ohe = sector_encoder.transform(pd.DataFrame({"sector": [sector]}))
    X = np.hstack([num_scaled, sector_ohe])

    proba = float(model.predict_proba(X)[0, 1])
    pred = "UP" if proba >= 0.5 else "DOWN"

    out = (
        f"Sector: {sector}\n"
        f"Sentiment: {sent.label.capitalize()} ({sent.score:.2f})\n"
        f"Predicted Next-Day Move: {pred} ({proba:.2f})"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze an earnings snippet")
    parser.add_argument("text", type=str, help="Input like 'AAPL: guidance raised for Q4'")
    args = parser.parse_args()
    result = analyze(args.text)
    print(result)


if __name__ == "__main__":
    main()


