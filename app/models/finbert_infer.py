from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import numpy as np


DEFAULT_MODEL_ID = "yiyanghkust/finbert-tone"


@dataclass
class SentimentOutput:
    label: str
    score: float
    logits: List[float]


class FinBERTSentiment:
    """Lightweight wrapper around HuggingFace pipeline with offline fallback.

    If transformers/torch are unavailable or model can't be loaded, uses a simple heuristic
    as an offline fallback.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self._pipe = None
        self._available = False
        # Respect offline flags to avoid network/model downloads
        offline_env = os.getenv("TRANSFORMERS_OFFLINE") or os.getenv("HF_HUB_OFFLINE") or os.getenv("FINBERT_OFFLINE")
        if offline_env and offline_env.lower() in {"1", "true", "yes", "on"}:
            self._available = False
            self._pipe = None
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
            self._available = True
        except Exception:
            self._pipe = None
            self._available = False

    def predict(self, text: str) -> SentimentOutput:
        if self._available and self._pipe is not None:
            scores = self._pipe(text)[0]
            # scores is a list of dicts like [{'label': 'positive', 'score': 0.7}, ...]
            label_scores: Dict[str, float] = {s["label"].lower(): float(s["score"]) for s in scores}
            best_label = max(label_scores, key=label_scores.get)
            best_score = label_scores[best_label]
            
            # Create logits array in consistent order [negative, neutral, positive]
            logits = [
                label_scores.get("negative", 0.0),
                label_scores.get("neutral", 0.0), 
                label_scores.get("positive", 0.0)
            ]
            
            return SentimentOutput(label=best_label, score=best_score, logits=logits)
        # offline heuristic fallback
        return self._heuristic(text)

    def _heuristic(self, text: str) -> SentimentOutput:
        text_low = text.lower()
        pos_kw = ["beat", "exceed", "strong", "raised", "record", "surprise", "profit"]
        neg_kw = ["miss", "weak", "cut", "lowered", "loss", "delay", "probe"]
        pos = any(k in text_low for k in pos_kw)
        neg = any(k in text_low for k in neg_kw)
        if pos and not neg:
            return SentimentOutput(label="positive", score=0.7, logits=[0.15, 0.15, 0.7])
        if neg and not pos:
            return SentimentOutput(label="negative", score=0.7, logits=[0.7, 0.15, 0.15])
        return SentimentOutput(label="neutral", score=0.5, logits=[0.2, 0.6, 0.2])



