from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.models.finbert_infer import FinBERTSentiment
from app.ingest.market_data import download_prices_yf, compute_price_features
from app.utils.logging import get_logger


logger = get_logger(__name__)


def load_ticker_sector_map(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"ticker", "company", "sector"}
    if not expected.issubset(df.columns):
        raise ValueError(f"ticker_sector_map must have columns {expected}")
    df["ticker"] = df["ticker"].str.upper()
    return df


def map_sector(ticker: str, mapping_df: pd.DataFrame) -> str:
    row = mapping_df[mapping_df["ticker"] == ticker.upper()]
    if row.empty:
        raise KeyError(f"Unknown ticker '{ticker}'. Add it to ticker_sector_map.csv")
    return str(row.iloc[0]["sector"])


@dataclass
class FeatureArtifacts:
    sector_encoder: OneHotEncoder
    scaler: StandardScaler
    feature_names: List[str]


def build_features(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    include_price_features: bool = True,
    price_lookback_days: int = 5,
    yfinance_offline: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], FeatureArtifacts, pd.DataFrame]:
    """Build features from dataframe with columns: ticker, text, optional label.

    Returns: X, y, artifacts, enriched_df
    """
    sentiment_model = FinBERTSentiment()

    sectors: List[str] = []
    sentiment_scores: List[float] = []
    sentiment_labels: List[str] = []
    logits_list: List[List[float]] = []
    ret_5d_list: List[float] = []
    vol_5d_list: List[float] = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        text = str(row["text"])

        try:
            sector = map_sector(ticker, mapping_df)
        except KeyError as e:
            logger.warning(str(e))
            sector = "Unknown"
        sectors.append(sector)

        sent = sentiment_model.predict(text)
        sentiment_scores.append(sent.score)
        sentiment_labels.append(sent.label)
        logits_list.append(sent.logits)

        if include_price_features and not yfinance_offline:
            px = download_prices_yf(ticker, lookback_days=max(price_lookback_days, 10))
            ret_5d, vol_5d = compute_price_features(px, lookback_days=price_lookback_days)
        else:
            ret_5d, vol_5d = 0.0, 0.0
        ret_5d_list.append(ret_5d)
        vol_5d_list.append(vol_5d)

    enriched = df.copy()
    enriched["sector"] = sectors
    enriched["sentiment_label"] = sentiment_labels
    enriched["sentiment_score"] = sentiment_scores
    enriched[["logit_neg", "logit_neu", "logit_pos"]] = pd.DataFrame(logits_list, index=enriched.index)
    enriched["ret_5d"] = ret_5d_list
    enriched["vol_5d"] = vol_5d_list

    # Sector one-hot
    sector_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    sector_ohe = sector_encoder.fit_transform(enriched[["sector"]])
    sector_feature_names = [f"sector_{c}" for c in sector_encoder.categories_[0]]

    # Numerical features
    num_features = enriched[["logit_neg", "logit_neu", "logit_pos", "sentiment_score", "ret_5d", "vol_5d"]].values
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_features)
    num_feature_names = ["logit_neg", "logit_neu", "logit_pos", "sentiment_score", "ret_5d", "vol_5d"]

    X = np.hstack([num_scaled, sector_ohe])
    feature_names = num_feature_names + sector_feature_names

    y = None
    if "label" in enriched.columns and not enriched["label"].isna().all():
        # Encode labels: UP->1, DOWN->0
        label_map = {"UP": 1, "DOWN": 0, "up": 1, "down": 0}
        y = enriched["label"].map(lambda v: label_map.get(str(v), np.nan)).astype(float)
        y = y.fillna(method="ffill").fillna(method="bfill").astype(int).values

    artifacts = FeatureArtifacts(sector_encoder=sector_encoder, scaler=scaler, feature_names=feature_names)
    return X, y, artifacts, enriched



