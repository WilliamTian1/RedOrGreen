import pandas as pd

from app.features.build_features import load_ticker_sector_map, map_sector


def test_map_sector_known():
    df = load_ticker_sector_map("app/data/ticker_sector_map.csv")
    assert map_sector("AAPL", df) == "Technology"


def test_map_sector_unknown():
    df = load_ticker_sector_map("app/data/ticker_sector_map.csv")
    try:
        map_sector("ZZZZ", df)
    except KeyError as e:
        assert "Unknown ticker" in str(e)
    else:
        assert False, "Expected KeyError for unknown ticker"




