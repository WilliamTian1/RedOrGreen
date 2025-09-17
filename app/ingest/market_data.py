from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def download_prices_yf(ticker: str, lookback_days: int = 10) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf

        end = datetime.utcnow()
        start = end - timedelta(days=max(lookback_days * 3, 15))
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def compute_price_features(df: Optional[pd.DataFrame], lookback_days: int = 5) -> Tuple[float, float]:
    """Compute 5-day log return and realized volatility (std of daily returns).

    Returns (ret_5d, vol_5d). If df is None or insufficient, returns (0.0, 0.0) and warns upstream.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0, 0.0
    px = df["Close"].dropna()
    if len(px) < lookback_days + 1:
        return 0.0, 0.0
    log_ret = np.log(px).diff().dropna()
    window = log_ret[-lookback_days:]
    ret_5d = float(window.sum().iloc[0] if hasattr(window.sum(), 'iloc') else window.sum())
    vol_5d = float(window.std(ddof=1).iloc[0] if hasattr(window.std(ddof=1), 'iloc') else window.std(ddof=1))
    if not np.isfinite(ret_5d):
        ret_5d = 0.0
    if not np.isfinite(vol_5d):
        vol_5d = 0.0
    return ret_5d, vol_5d




