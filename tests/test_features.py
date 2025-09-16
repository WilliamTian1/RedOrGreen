import pandas as pd

from app.features.build_features import build_features, load_ticker_sector_map


def test_build_features_smoke():
    df = pd.DataFrame([
        {"ticker": "AAPL", "text": "AAPL: strong iPhone demand", "label": "UP"}
    ])
    mapping_df = load_ticker_sector_map("app/data/ticker_sector_map.csv")
    X, y, feats, enriched = build_features(df, mapping_df, include_price_features=False)
    # Expect 6 numeric + number of sectors one-hot
    assert X.shape[0] == 1
    assert X.shape[1] >= 6
    assert y is not None and y.shape[0] == 1




