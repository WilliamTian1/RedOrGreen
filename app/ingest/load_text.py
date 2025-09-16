from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import json
import re
import pandas as pd


TICKER_TEXT_RE = re.compile(r"^\s*([A-Z]{1,6})\s*:\s*(.+)$")


@dataclass
class ParsedSnippet:
    ticker: str
    text: str
    label: Optional[str] = None
    timestamp: Optional[str] = None


def parse_snippet(line: str) -> ParsedSnippet:
    """Parse a line like "AAPL: guidance raised" -> (ticker, text).

    Raises ValueError if ticker prefix is missing.
    """
    m = TICKER_TEXT_RE.match(line)
    if not m:
        raise ValueError("Input must start with 'TICKER: text' e.g., 'AAPL: guidance raised'.")
    return ParsedSnippet(ticker=m.group(1), text=m.group(2).strip())


def load_jsonl(path: str) -> pd.DataFrame:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(obj)
    df = pd.DataFrame(rows)
    expected_cols = {"ticker", "text"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"JSONL must contain columns {expected_cols}")
    return df




